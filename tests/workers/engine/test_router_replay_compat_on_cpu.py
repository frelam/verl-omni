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
"""CPU tests for the scoped R3 activation-recompute compatibility shim."""

import sys
from enum import Enum
from types import ModuleType

import pytest
import torch

from verl_omni.workers.engine.router_replay_compat import install_r3_router_replay_compat

_UPSTREAM_MODULE = "verl.utils.megatron.router_replay_patch"


class _Action(Enum):
    RECORD = "record"
    REPLAY_BACKWARD = "replay_backward"


def _native_topk(scores, topk, **_kwargs):
    return torch.topk(scores, k=topk, dim=1)


def _install_router(monkeypatch):
    class RouterReplay:
        def __init__(self):
            self.router_replay_action = None
            self.replay_backward_list = []
            self.replay_backward_mask_list = []

        def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None, default_compute_topk=None):
            if self.router_replay_action == _Action.REPLAY_BACKWARD and self.replay_backward_list:
                indices = self.replay_backward_list.pop(0)
                self.replay_backward_mask_list.pop(0)
            else:
                return default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            return scores.gather(1, indices), indices

    module = ModuleType(_UPSTREAM_MODULE)
    module.RouterReplay = RouterReplay
    module.RouterReplayAction = _Action
    monkeypatch.setitem(sys.modules, _UPSTREAM_MODULE, module)
    install_r3_router_replay_compat()
    return RouterReplay


def test_backward_replay_skips_stale_recompute_entries(monkeypatch):
    router = _install_router(monkeypatch)()
    router.router_replay_action = _Action.REPLAY_BACKWARD
    stale = torch.full((200, 2), 3, dtype=torch.int64)
    fresh = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=torch.int64)
    router.replay_backward_list = [stale, fresh]
    router.replay_backward_mask_list = [None, None]
    scores = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    with pytest.warns(UserWarning, match="dropped unconsumed route entries"):
        _, indices = router.get_replay_topk(scores, 2, default_compute_topk=_native_topk)

    assert indices is fresh
    assert router.replay_backward_list == []
    assert router.replay_backward_mask_list == []


def test_backward_replay_rejects_unmatched_recompute(monkeypatch):
    router = _install_router(monkeypatch)()
    router.router_replay_action = _Action.REPLAY_BACKWARD
    router.replay_backward_list = [torch.zeros((200, 2), dtype=torch.int64)]
    router.replay_backward_mask_list = [None]
    scores = torch.zeros((6, 4))

    with pytest.raises(RuntimeError, match=r"no queued route tensor matches; queued=\[\(200, 2\)\]"):
        router.get_replay_topk(scores, 2, default_compute_topk=_native_topk)


def test_compat_install_is_idempotent_and_preserves_non_backward_paths(monkeypatch):
    router_cls = _install_router(monkeypatch)
    wrapped = router_cls.get_replay_topk
    install_r3_router_replay_compat()
    assert router_cls.get_replay_topk is wrapped

    router = router_cls()
    router.router_replay_action = _Action.RECORD
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    probs, indices = router.get_replay_topk(scores, 2, default_compute_topk=_native_topk)
    expected_probs, expected_indices = _native_topk(scores, 2)
    torch.testing.assert_close(probs, expected_probs)
    torch.testing.assert_close(indices, expected_indices)
