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

"""CPU tests for the vLLM-Omni AR rollout side of R3 router replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging import version

from verl_omni.workers.rollout.vllm_rollout import vllm_omni_ar_strategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy


@pytest.fixture(autouse=True)
def _supported_vllm(monkeypatch):
    """Pin a supported vLLM: the CPU stub reports 0.0.0 while CI installs the real wheel."""
    monkeypatch.setattr(vllm_omni_ar_strategy, "_VLLM_VERSION", version.parse("0.28.0"))


def _strategy(*, capture: bool, output_modalities=None) -> ARStrategy:
    """Build a strategy with the server state ``init_config`` would have created."""
    strategy = ARStrategy(SimpleNamespace(config=SimpleNamespace(enable_rollout_routing_replay=capture)))
    strategy.server.global_steps = 0
    strategy._rollout_output_modalities = output_modalities
    return strategy


def test_preprocess_engine_kwargs_requests_routed_experts():
    engine_kwargs: dict = {}
    _strategy(capture=True).preprocess_engine_kwargs(engine_kwargs)
    assert engine_kwargs["enable_return_routed_experts"] is True


def test_preprocess_engine_kwargs_untouched_when_disabled():
    engine_kwargs: dict = {}
    _strategy(capture=False).preprocess_engine_kwargs(engine_kwargs)
    assert "enable_return_routed_experts" not in engine_kwargs


def test_preprocess_engine_kwargs_requires_supported_vllm(monkeypatch):
    monkeypatch.setattr(vllm_omni_ar_strategy, "_VLLM_VERSION", version.parse("0.21.0"))
    with pytest.raises(RuntimeError, match="vLLM >= 0.22.0"):
        _strategy(capture=True).preprocess_engine_kwargs({})


def test_preprocess_engine_kwargs_rejects_multi_output_pipeline():
    with pytest.raises(ValueError, match="single-output"):
        _strategy(capture=True, output_modalities=["text", "audio"]).preprocess_engine_kwargs({})


def _request_output(**completion_fields) -> SimpleNamespace:
    completion = SimpleNamespace(token_ids=[5, 6], logprobs=None, finish_reason="stop", **completion_fields)
    return SimpleNamespace(outputs=[completion])


def test_process_output_carries_routes_as_int16_tensor():
    routes = np.arange(2 * 3 * 2, dtype=np.int32).reshape(2, 3, 2) % 8
    output = _strategy(capture=True).process_output(
        _request_output(routed_experts=routes), SimpleNamespace(logprobs=None), {}
    )
    assert output.routed_experts is not None
    assert output.routed_experts.dtype == torch.int16
    assert torch.equal(output.routed_experts, torch.as_tensor(routes).to(torch.int16))


def test_process_output_omits_routes_when_disabled():
    output = _strategy(capture=False).process_output(_request_output(), SimpleNamespace(logprobs=None), {})
    assert output.routed_experts is None


def test_process_output_fails_closed_when_routes_missing():
    with pytest.raises(RuntimeError, match="enable_return_routed_experts"):
        _strategy(capture=True).process_output(_request_output(), SimpleNamespace(logprobs=None), {})
