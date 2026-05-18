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
"""
Preprocess the GSM8K dataset to parquet format (for Qwen3-Omni-Thinker GSPO training).

GSM8K (Grade School Math 8K) is a dataset of grade-school math word problems
with step-by-step solutions. Each problem has a question and an answer in the
format "step-by-step reasoning #### <final_number>".

Supported data sources:
  - ``openai/gsm8k`` (default): the official GSM8K dataset from OpenAI,
    with "main" config containing train/test splits (7,473 + 1,319).
  - ``openai/gsm8k`` with ``config=socratic``: the Socratic variant.

Usage:
    python examples/gspo_trainer/data_process/gsm8k.py \\
        --dataset_name openai/gsm8k \\
        --output_dir ~/data/gsm8k

This produces:
    - ``<output_dir>/train.parquet``
    - ``<output_dir>/test.parquet``

Each row contains:
    - data_source: identifier for reward function selection
    - prompt: chat messages with the math question
    - images: empty list (text-only)
    - audios: empty list (text-only)
    - ability: task type string ("math")
    - reward_model: dict with style and ground_truth (final numerical answer)
    - extra_info: dict with split, index, question, answer
"""

from __future__ import annotations

import argparse
import os
import random
import re

import datasets
import pandas as pd


def extract_answer(answer_str: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_str)
    if m:
        return m.group(1).replace(",", "")
    return answer_str.strip().split()[-1] if answer_str.strip() else ""


SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Please reason through the problem carefully and provide your final "
    "numerical answer after ####."
)


def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_row(sample: dict, split: str, idx: int) -> dict:
    question = sample["question"]
    full_answer = sample["answer"]
    ground_truth = extract_answer(full_answer)

    prompt = build_prompt(question)

    return {
        "data_source": "openai/gsm8k",
        "prompt": prompt,
        "images": [],
        "audios": [],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "answer": full_answer,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K dataset for verl-omni GSPO training")
    parser.add_argument(
        "--dataset_name",
        default="openai/gsm8k",
        help="HuggingFace dataset name (default: openai/gsm8k).",
    )
    parser.add_argument(
        "--config",
        default="main",
        help="Dataset config name (default: main). Use 'socratic' for the Socratic variant.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/gsm8k",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help=(
            "Ratio of training data (0.0-1.0). Only used when the dataset has "
            "no built-in train/valid split. Default: use built-in splits."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading dataset: {args.dataset_name} (config={args.config})")
    ds = datasets.load_dataset(args.dataset_name, args.config)

    train_ds = ds.get("train")
    test_ds = ds.get("test")

    if train_ds is None and test_ds is None:
        all_ds = ds[list(ds.keys())[0]]
        if args.train_ratio is None:
            args.train_ratio = 0.85
            print(f"No built-in split found, using train_ratio={args.train_ratio}")
        all_samples = list(all_ds)
        random.shuffle(all_samples)
        n_train = int(len(all_samples) * args.train_ratio)
        train_ds = all_samples[:n_train]
        test_ds = all_samples[n_train:]
    elif train_ds is not None and test_ds is None:
        all_samples = list(train_ds)
        random.shuffle(all_samples)
        n_train = int(len(all_samples) * 0.85)
        test_ds = all_samples[n_train:]
        train_ds = all_samples[:n_train]
        print(f"No test split found, auto-split: train={len(train_ds)}, test={len(test_ds)}")

    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        test_ds = test_ds.select(range(min(max(args.max_samples // 4, 1), len(test_ds))))

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_rows = [build_row(sample, "train", i) for i, sample in enumerate(train_ds)]
    test_rows = [build_row(sample, "test", i) for i, sample in enumerate(test_ds)]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(test_df)} test samples to {test_path}")

    print("\nTo use this dataset with GSPO training, set:")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={test_path}")
    print("  reward.reward_manager.name=dapo")

    print("\nQuick verification:")
    print("  import pandas as pd")
    print(f"  df = pd.read_parquet('{train_path}')")
    print("  print(f'Samples: {{len(df)}}')")
    print("  print(f'Columns: {{df.columns.tolist()}}')")
    print("  print(f'Sample prompt: {{df.iloc[0][\"prompt\"]}}')")
    print("  print(f'Ground truth: {{df.iloc[0][\"reward_model\"]}}')")


if __name__ == "__main__":
    main()
