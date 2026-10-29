# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM-Omni AR schedulers that return per-request MoE routes (R3).

vLLM's stock scheduler persists every step's routed experts into a slot-indexed
CPU buffer (``RoutedExpertsManager``) and attaches the per-request slice to
``EngineCoreOutput.routed_experts``; the output processor then concatenates the
chunks into ``CompletionOutput.routed_experts``. vLLM-Omni's AR scheduler
inherits the scheduling loop but not that bookkeeping -- it only calls
``omni_routed_experts_for_request`` on stop, and that helper keys rows by
``Request.block_table``, which does not exist upstream, so it always returns
``None``. These subclasses restore the upstream behaviour without patching
vLLM-Omni.
"""

import logging
from typing import Any

import numpy as np
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler

logger = logging.getLogger(__file__)


class _RoutedExpertsARSchedulerMixin:
    """Attach the rollout's MoE routes to every omni AR engine output."""

    # Set by the first update_from_output; avoids spamming the log per step.
    _routed_experts_missing_warned = False

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> Any:
        routes_by_request = self._take_step_routed_experts(scheduler_output, model_runner_output)
        engine_core_outputs = super().update_from_output(scheduler_output, model_runner_output)
        if routes_by_request:
            for client_outputs in engine_core_outputs.values():
                for engine_core_output in client_outputs.outputs:
                    routes = routes_by_request.get(engine_core_output.request_id)
                    if routes is not None:
                        engine_core_output.routed_experts = routes
        return engine_core_outputs

    def _take_step_routed_experts(self, scheduler_output: Any, model_runner_output: Any) -> dict[str, np.ndarray]:
        """Slice this step's rows per request, mirroring vLLM's Scheduler.

        Route rows arrive in ``model_runner_output.req_ids`` order with
        ``num_scheduled_tokens[req_id]`` rows each. Prompt rows are read back
        from the slot buffer (prefix-cached blocks keep their routing there);
        decode rows are taken from this step's slice because the slots of a
        not-yet-forwarded sampled token must not be touched.
        """
        if not getattr(self, "enable_return_routed_experts", False):
            return {}
        routed_experts = getattr(model_runner_output, "routed_experts", None)
        if routed_experts is None:
            self._warn_missing_routed_experts()
            return {}

        # Persist this step's rows first: a request that stops on a token
        # generated in this very step still needs its prompt rows.
        routed_experts_mgr = self.routed_experts_mgr
        routed_experts_mgr.store_batch(routed_experts.routing_data, routed_experts.slot_mapping)
        routing_data = routed_experts.routing_data.astype(routed_experts_mgr.routed_experts_by_slot.dtype, copy=False)

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        sampled_token_ids = model_runner_output.sampled_token_ids
        routes_by_request: dict[str, np.ndarray] = {}
        offset = 0
        for req_id in model_runner_output.req_ids:
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            req_offset = offset
            offset += num_tokens_scheduled

            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            new_token_ids = sampled_token_ids[model_runner_output.req_id_to_index[req_id]] if sampled_token_ids else []
            if not new_token_ids:
                continue

            if len(request._output_token_ids) == 0:
                # Prefill (or re-prefill after preemption) just finished.
                block_ids = self._re_block_ids.pop(req_id, [])
                if not block_ids:
                    logger.warning("Missing KV block snapshot for request %s; skipping its routed experts.", req_id)
                    continue
                routes_by_request[req_id] = routed_experts_mgr.get(block_ids, request.num_prompt_tokens)
            elif scheduler_output.scheduled_spec_decode_tokens.get(req_id):
                # Spec decode accepts tokens at the START of the scheduled range.
                routes_by_request[req_id] = routing_data[req_offset : req_offset + len(new_token_ids)]
            else:
                # Normal decode: the sampled token(s) sit at the END of the range.
                end = req_offset + num_tokens_scheduled
                routes_by_request[req_id] = routing_data[end - len(new_token_ids) : end]
        return routes_by_request

    def _warn_missing_routed_experts(self) -> None:
        if self._routed_experts_missing_warned:
            return
        self._routed_experts_missing_warned = True
        logger.warning(
            "enable_return_routed_experts is set, but this step's ModelRunnerOutput carried no routes; "
            "the worker side (platform MoE kernels) is not capturing them."
        )


class R3ARScheduler(_RoutedExpertsARSchedulerMixin, OmniARScheduler):
    """Synchronous omni AR scheduler carrying MoE routes."""


class R3ARAsyncScheduler(_RoutedExpertsARSchedulerMixin, OmniARAsyncScheduler):
    """Async-scheduling omni AR scheduler carrying MoE routes."""


def verify_routed_experts_engine(engine: Any) -> None:
    """Fail fast when the rollout engine is not wired to return MoE routes.

    Args:
        engine: The ``AsyncOmni`` client created by the rollout server.

    Raises:
        RuntimeError: When a stage did not receive ``enable_return_routed_experts``
            or still runs a scheduler that drops the routes.
    """
    stage_configs = getattr(getattr(engine, "engine", None), "stage_vllm_configs", None) or []
    for stage_id, vllm_config in enumerate(stage_configs):
        if vllm_config is None:
            continue
        if not getattr(vllm_config.model_config, "enable_return_routed_experts", False):
            raise RuntimeError(
                f"R3 router replay is on, but stage {stage_id} did not receive enable_return_routed_experts=True; "
                "the rollout engine would never return routed experts."
            )
        scheduler_cls = vllm_config.scheduler_config.get_scheduler_cls()
        if not issubclass(scheduler_cls, _RoutedExpertsARSchedulerMixin):
            raise RuntimeError(
                f"R3 router replay is on, but stage {stage_id} runs scheduler {scheduler_cls!r}; "
                "vLLM-Omni's built-in AR scheduler drops routed experts, so the rollout must use "
                f"{R3ARAsyncScheduler.__module__}.{R3ARAsyncScheduler.__qualname__}."
            )
