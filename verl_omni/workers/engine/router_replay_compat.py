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
"""Scoped compatibility for verl's RouterReplay activation-recompute queue.

Entries are matched by token count; equal-sized micro-batches remain indistinguishable.
"""

import functools
import warnings
from typing import Any


def install_r3_router_replay_compat() -> None:
    """Match R3 backward routes to the recompute token count within this worker."""
    from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction

    if getattr(RouterReplay.get_replay_topk, "_verl_omni_r3_recompute_compat", False):
        return

    original_get_replay_topk = RouterReplay.get_replay_topk
    stale_replay_warned = False

    @functools.wraps(original_get_replay_topk)
    def get_replay_topk(self: Any, scores: Any, topk: int, *args: Any, **kwargs: Any) -> Any:
        nonlocal stale_replay_warned

        if self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD and self.replay_backward_list:
            num_rows = scores.shape[0]
            position = next(
                (index for index, routes in enumerate(self.replay_backward_list) if routes.shape[0] == num_rows),
                None,
            )
            if position is None:
                queued = [tuple(routes.shape) for routes in self.replay_backward_list]
                raise RuntimeError(
                    "Router replay RECORD/REPLAY mismatch: this recompute has "
                    f"{num_rows} token rows but no queued route tensor matches; queued={queued}. "
                    "The recompute consumed routes of a different micro-batch, so the actor would silently "
                    "train on the wrong experts."
                )
            if position:
                if not stale_replay_warned:
                    stale_replay_warned = True
                    warnings.warn(
                        "Router replay dropped unconsumed route entries "
                        f"{[tuple(routes.shape) for routes in self.replay_backward_list[:position]]}: "
                        "those layers were never re-entered during activation recompute.",
                        stacklevel=2,
                    )
                del self.replay_backward_list[:position]
                del self.replay_backward_mask_list[:position]

        return original_get_replay_topk(self, scores, topk, *args, **kwargs)

    get_replay_topk._verl_omni_r3_recompute_compat = True
    RouterReplay.get_replay_topk = get_replay_topk
