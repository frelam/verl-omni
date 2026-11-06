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

"""Geo3K reward with a degenerate-repetition guard.

Reuses verl's built-in ``hiyouga/geometry3k`` scorer (90% answer accuracy +
10% strict ``<think>...</think>\\boxed{...}`` format) and zeroes the content
score when the response is stuck in a repetition loop, even if the boxed
answer happens to match. The format score is judged as usual.

Config:
    reward.custom_reward_function.path=verl_omni/utils/reward_score/geo3k_reward.py
    reward.custom_reward_function.name=compute_score
"""

import re
from collections import Counter

from verl.utils.reward_score.geo3k import acc_reward, format_reward

# A few words looping forever should score zero. Count 4-grams over word /
# CJK-char tokens; natural math prose reuses phrases a few times, so allow 15
# repeated occurrences before the trajectory is treated as a repetition loop.
REPETITION_NGRAM_SIZE = 4
REPETITION_FREE_REPEATS = 15

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def count_repeated_ngrams(solution_str: str, n: int = REPETITION_NGRAM_SIZE) -> int:
    """Return repeated n-gram occurrences beyond the first one.

    A 4-gram that appears four times contributes three. Chinese characters are
    tokenized individually so CJK loops are detected as well.
    """
    tokens = _TOKEN_RE.findall((solution_str or "").lower())
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return sum(count - 1 for count in counts.values() if count > 1)


def has_repetition(solution_str: str) -> bool:
    """True when the response loops on repeated n-grams beyond the free quota."""
    return count_repeated_ngrams(solution_str) > REPETITION_FREE_REPEATS


def compute_score(
    solution_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    format_score: float = 0.1,
    **kwargs,
) -> float:
    """Score one Geo3K response, zeroing only the content score for repetition loops.

    The format score is still awarded normally, so a looping response can score
    at most ``format_score``.

    Extra ``data_source`` / ``extra_info`` kwargs are accepted because verl's
    reward managers pass them to every custom reward function.
    """
    acc = 0.0 if has_repetition(solution_str) else acc_reward(solution_str, ground_truth, use_boxed)
    return (1.0 - format_score) * acc + format_score * format_reward(solution_str)
