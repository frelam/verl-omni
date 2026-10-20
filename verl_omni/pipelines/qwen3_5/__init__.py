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
"""Qwen3.5 pipeline adapters (training + rollout pipeline topology)."""

from .rollout_adapter import (
    QWEN3_5_MOE_PIPELINE,
    QWEN3_5_PIPELINE,
    Qwen3_5MoeRolloutAdapter,
    Qwen3_5RolloutAdapter,
)
from .training_adapter import Qwen3_5ThinkerAdapter

__all__ = [
    "Qwen3_5ThinkerAdapter",
    "Qwen3_5RolloutAdapter",
    "Qwen3_5MoeRolloutAdapter",
    "QWEN3_5_PIPELINE",
    "QWEN3_5_MOE_PIPELINE",
]
