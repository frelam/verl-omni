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

"""CPU tests for the routed-experts scheduler used by R3 router replay."""

from types import SimpleNamespace

import numpy as np
import pytest

from verl_omni.workers.rollout.vllm_rollout.routed_experts_scheduler import (
    R3ARAsyncScheduler,
    _RoutedExpertsARSchedulerMixin,
    verify_routed_experts_engine,
)


def _routes(rows: int) -> np.ndarray:
    """Distinct route rows so slicing mistakes show up as wrong values."""
    data = np.arange(rows * 2 * 2, dtype=np.int32).reshape(rows, 2, 2)
    return (data % 16).astype(np.uint8)


class _FakeRoutedExpertsManager:
    """Records the slot-buffer calls the upstream manager would receive."""

    def __init__(self) -> None:
        self.routed_experts_by_slot = np.zeros((64, 2, 2), dtype=np.uint8)
        self.stored: list[tuple[np.ndarray, np.ndarray]] = []
        self.prompt_requests: list[tuple[list[int], int]] = []
        self.prompt_routes = _routes(3)

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        self.stored.append((data.copy(), slot_mapping.copy()))

    def get(self, block_ids: list[int], num_tokens: int, token_start: int = 0) -> np.ndarray:
        self.prompt_requests.append((list(block_ids), num_tokens))
        return self.prompt_routes[token_start:num_tokens]


class _FakeRequest:
    def __init__(self, num_prompt_tokens: int, output_tokens: tuple[int, ...] = ()) -> None:
        self.num_prompt_tokens = num_prompt_tokens
        self._output_token_ids = list(output_tokens)

    def is_finished(self) -> bool:
        return False


class _FakeOmniScheduler:
    """Minimal stand-in for ``OmniARScheduler``; emits one output per request."""

    def __init__(self, emitted_request_ids: tuple[str, ...] = ()) -> None:
        self.requests: dict[str, _FakeRequest] = {}
        self.enable_return_routed_experts = True
        self.routed_experts_mgr = _FakeRoutedExpertsManager()
        self._re_block_ids: dict[str, list[int]] = {}
        self.emitted_request_ids = emitted_request_ids

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = [
            SimpleNamespace(request_id=request_id, routed_experts=None) for request_id in self.emitted_request_ids
        ]
        return {0: SimpleNamespace(outputs=outputs)}


class _Scheduler(_RoutedExpertsARSchedulerMixin, _FakeOmniScheduler):
    pass


def _model_output(rows: int, *, req_ids: list[str], sampled: list[list[int]], slots: np.ndarray | None = None):
    return SimpleNamespace(
        routed_experts=SimpleNamespace(
            routing_data=_routes(rows),
            slot_mapping=np.arange(rows, dtype=np.int64) if slots is None else slots,
        ),
        sampled_token_ids=sampled,
        req_ids=req_ids,
        req_id_to_index={rid: index for index, rid in enumerate(req_ids)},
    )


def _scheduler_output(num_scheduled_tokens: dict[str, int], spec_tokens: dict[str, list[int]] | None = None):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=spec_tokens or {},
    )


def test_prefill_completion_reads_prompt_rows_from_the_slot_buffer():
    scheduler = _Scheduler(emitted_request_ids=("req",))
    scheduler.requests["req"] = _FakeRequest(num_prompt_tokens=3)
    scheduler._re_block_ids["req"] = [4, 5]
    model_output = _model_output(3, req_ids=["req"], sampled=[[9]])
    stored_before = len(scheduler.routed_experts_mgr.stored)

    outputs = scheduler.update_from_output(_scheduler_output({"req": 3}), model_output)

    manager = scheduler.routed_experts_mgr
    assert len(manager.stored) == stored_before + 1
    assert manager.prompt_requests == [([4, 5], 3)]
    assert "req" not in scheduler._re_block_ids
    np.testing.assert_array_equal(outputs[0].outputs[0].routed_experts, manager.prompt_routes)


def test_decode_step_takes_the_tail_of_the_scheduled_range():
    scheduler = _Scheduler(emitted_request_ids=("req",))
    scheduler.requests["req"] = _FakeRequest(num_prompt_tokens=3, output_tokens=(9, 10))
    model_output = _model_output(4, req_ids=["req"], sampled=[[11]])

    outputs = scheduler.update_from_output(_scheduler_output({"req": 4}), model_output)

    expected = model_output.routed_experts.routing_data[3:4]
    np.testing.assert_array_equal(outputs[0].outputs[0].routed_experts, expected)
    assert scheduler.routed_experts_mgr.prompt_requests == []


def test_spec_decode_step_takes_the_head_of_the_scheduled_range():
    scheduler = _Scheduler(emitted_request_ids=("req",))
    scheduler.requests["req"] = _FakeRequest(num_prompt_tokens=3, output_tokens=(9,))
    model_output = _model_output(5, req_ids=["req"], sampled=[[11, 12]])

    outputs = scheduler.update_from_output(
        _scheduler_output({"req": 5}, spec_tokens={"req": [1, 2, 3, 4]}), model_output
    )

    expected = model_output.routed_experts.routing_data[0:2]
    np.testing.assert_array_equal(outputs[0].outputs[0].routed_experts, expected)


def test_offsets_follow_the_model_runner_request_order():
    scheduler = _Scheduler(emitted_request_ids=("a", "b"))
    scheduler.requests["a"] = _FakeRequest(num_prompt_tokens=2, output_tokens=(7,))
    scheduler.requests["b"] = _FakeRequest(num_prompt_tokens=3, output_tokens=(8, 9))
    model_output = _model_output(6, req_ids=["b", "a"], sampled=[[20, 21], [22]])

    outputs = scheduler.update_from_output(_scheduler_output({"a": 2, "b": 4}), model_output)

    routing_data = model_output.routed_experts.routing_data
    routes = {output.request_id: output.routed_experts for output in outputs[0].outputs}
    np.testing.assert_array_equal(routes["b"], routing_data[2:4])
    np.testing.assert_array_equal(routes["a"], routing_data[5:6])


def test_prefill_without_a_block_snapshot_is_skipped():
    scheduler = _Scheduler(emitted_request_ids=("req",))
    scheduler.requests["req"] = _FakeRequest(num_prompt_tokens=3)

    outputs = scheduler.update_from_output(
        _scheduler_output({"req": 3}), _model_output(3, req_ids=["req"], sampled=[[9]])
    )

    assert outputs[0].outputs[0].routed_experts is None


def test_missing_routes_warns_once_and_keeps_outputs():
    scheduler = _Scheduler(emitted_request_ids=("req",))
    scheduler.requests["req"] = _FakeRequest(num_prompt_tokens=3, output_tokens=(9,))
    model_output = SimpleNamespace(
        routed_experts=None, sampled_token_ids=[[11]], req_ids=["req"], req_id_to_index={"req": 0}
    )

    for _ in range(2):
        outputs = scheduler.update_from_output(_scheduler_output({"req": 4}), model_output)
        assert outputs[0].outputs[0].routed_experts is None

    assert scheduler._routed_experts_missing_warned


def test_capture_disabled_is_a_pass_through():
    scheduler = _Scheduler(emitted_request_ids=("req",))
    scheduler.enable_return_routed_experts = False
    scheduler.requests["req"] = _FakeRequest(num_prompt_tokens=3, output_tokens=(9,))

    outputs = scheduler.update_from_output(
        _scheduler_output({"req": 4}), _model_output(4, req_ids=["req"], sampled=[[11]])
    )

    assert scheduler.routed_experts_mgr.stored == []
    assert outputs[0].outputs[0].routed_experts is None


def _engine(*, enabled: bool, scheduler_cls: type) -> SimpleNamespace:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=enabled),
        scheduler_config=SimpleNamespace(get_scheduler_cls=lambda: scheduler_cls),
    )
    return SimpleNamespace(engine=SimpleNamespace(stage_vllm_configs=[vllm_config]))


def test_verify_engine_accepts_the_r3_scheduler():
    verify_routed_experts_engine(_engine(enabled=True, scheduler_cls=R3ARAsyncScheduler))


def test_verify_engine_rejects_a_scheduler_without_route_support():
    with pytest.raises(RuntimeError, match="drops routed experts"):
        verify_routed_experts_engine(_engine(enabled=True, scheduler_cls=type("OtherScheduler", (), {})))


def test_verify_engine_rejects_an_engine_without_capture():
    with pytest.raises(RuntimeError, match="enable_return_routed_experts"):
        verify_routed_experts_engine(_engine(enabled=False, scheduler_cls=R3ARAsyncScheduler))
