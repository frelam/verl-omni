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

from __future__ import annotations

import re


def _extract_option_letter(response: str) -> str | None:
    m = re.search(r"\b([A-D])\b", response)
    if m:
        return m.group(1)
    return None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None, **kwargs) -> float:
    predicted = _extract_option_letter(solution_str)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0
