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
Preprocess the AVQA-R1-6K dataset to parquet format (for Qwen3-Omni-Thinker GSPO training).

The AVQA-R1-6K dataset (HarryHsing/AVQA-R1-6K) is derived from OmniInstruct-v1
and contains synchronized audio-image pairs with multiple-choice questions for
audio-visual reasoning. It already has train/valid splits (4490 + 1911 ≈ 6K).

Supported data sources:
  - ``HarryHsing/AVQA-R1-6K`` (default): the official AVQA-R1-6K dataset from
    EchoInk-R1, already in 4-option MCQ format with train/valid splits.
  - ``m-a-p/OmniInstruct_v1``: the upstream dataset; the script filters for
    source=="AVQA" samples and formats them as MCQ.
  - ``m-a-p/OmniBench``: already in 4-option MCQ format, used directly.

Usage:
    python examples/gspo_trainer/data_process/avqa_r1_6k.py \\
        --dataset_name HarryHsing/AVQA-R1-6K \\
        --output_dir ~/data/avqa_r1_6k

This produces:
    - ``<output_dir>/train.parquet``
    - ``<output_dir>/test.parquet``

Each row contains:
    - data_source: identifier for reward function selection
    - prompt: chat messages with <image> / <audio> placeholders
    - images: list of image dicts (bytes)
    - audios: list of audio dicts (bytes + sampling_rate)
    - ability: task type string
    - reward_model: dict with style and ground_truth (correct option letter)
    - extra_info: dict with split, index, category, question_type
"""

import argparse
import io
import os
import random

import datasets
import numpy as np
import pandas as pd
from PIL import Image


def audio_to_bytes(audio_dict):
    if audio_dict is None:
        return None
    if "array" in audio_dict and audio_dict["array"] is not None:
        import soundfile as sf

        arr = np.array(audio_dict["array"], dtype=np.float32)
        sampling_rate = audio_dict.get("sampling_rate", 24000)
        buf = io.BytesIO()
        sf.write(buf, arr, sampling_rate, format="WAV")
        wav_bytes = buf.getvalue()
        return {"bytes": wav_bytes, "sampling_rate": sampling_rate}
    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        return {"bytes": audio_dict["bytes"], "sampling_rate": audio_dict.get("sampling_rate", 24000)}
    return None


def image_to_bytes(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return {"bytes": buf.getvalue()}
    if isinstance(image, dict) and "bytes" in image:
        return {"bytes": image["bytes"]}
    return None


OPTION_LABELS = ["A", "B", "C", "D"]


def process_avqa_r1_6k(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    ds = datasets.load_dataset(dataset_name)

    train_samples = []
    valid_samples = []

    for split_name in ds:
        split_ds = ds[split_name]
        for example in split_ds:
            has_audio = example.get("audio") is not None
            has_image = example.get("image") is not None
            if not (has_audio and has_image):
                continue

            options = example.get("options", [])
            answer = example.get("answer", "")
            if not options or not answer:
                continue

            labeled_options = [f"{OPTION_LABELS[i]}. {opt}" for i, opt in enumerate(options)]
            correct_idx = options.index(answer) if answer in options else 0
            correct_label = OPTION_LABELS[correct_idx]

            sample = {
                "id": example.get("id", example.get("problem_id")),
                "question": example.get("question", example.get("problem", "")),
                "answer": answer,
                "category": example.get("category", example.get("data_source", "unknown")),
                "audio": example.get("audio"),
                "image": example.get("image"),
                "options": labeled_options,
                "correct_label": correct_label,
                "correct_answer": answer,
            }

            if "train" in split_name:
                train_samples.append(sample)
            else:
                valid_samples.append(sample)

    print(f"Found {len(train_samples)} train, {len(valid_samples)} valid samples")
    return train_samples, valid_samples


def process_omniinstruct(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    ds = datasets.load_dataset(dataset_name)

    train_samples = []
    valid_samples = []

    for split_name in ds:
        split_ds = ds[split_name]
        for example in split_ds:
            if example.get("source", "") != "AVQA":
                continue

            has_audio = example.get("audio") is not None
            has_image = example.get("image") is not None
            if not (has_audio and has_image):
                continue

            options = example.get("options", [])
            answer = example.get("answer", "")
            if not options or not answer:
                continue

            if answer not in options:
                continue

            labeled_options = [f"{OPTION_LABELS[i]}. {opt}" for i, opt in enumerate(options)]
            correct_idx = options.index(answer)
            correct_label = OPTION_LABELS[correct_idx]

            sample = {
                "id": example.get("id"),
                "question": example.get("question", ""),
                "answer": answer,
                "category": example.get("category", "unknown"),
                "audio": example.get("audio"),
                "image": example.get("image"),
                "options": labeled_options,
                "correct_label": correct_label,
                "correct_answer": answer,
            }

            if "train" in split_name:
                train_samples.append(sample)
            else:
                valid_samples.append(sample)

    print(f"Found {len(train_samples)} train, {len(valid_samples)} valid AVQA samples")
    return train_samples, valid_samples


def process_omnibench(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    ds = datasets.load_dataset(dataset_name)

    train_samples = []
    valid_samples = []

    for split_name in ds:
        split_ds = ds[split_name]
        for example in split_ds:
            has_audio = example.get("audio") is not None
            has_image = example.get("image") is not None
            if not (has_audio and has_image):
                continue

            options = example.get("options", [])
            answer = example.get("answer", "")
            if not options or not answer:
                continue

            labeled_options = [f"{OPTION_LABELS[i]}. {opt}" for i, opt in enumerate(options)]
            correct_idx = options.index(answer) if answer in options else 0
            correct_label = OPTION_LABELS[correct_idx]

            sample = {
                "id": example.get("index"),
                "question": example.get("question", ""),
                "answer": answer,
                "category": example.get("task type", "unknown"),
                "audio": example.get("audio"),
                "image": example.get("image"),
                "options": labeled_options,
                "correct_label": correct_label,
                "correct_answer": answer,
            }

            if "train" in split_name:
                train_samples.append(sample)
            else:
                valid_samples.append(sample)

    print(f"Found {len(train_samples)} train, {len(valid_samples)} valid OmniBench samples")
    return train_samples, valid_samples


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on both audio and visual information. "
    "Carefully listen to the audio and look at the image, then select the correct answer."
)


def build_prompt(question, options_text):
    user_content = (
        f"<audio>\n<image>\n{question}\n\n{options_text}\n\nPlease select the correct answer (A, B, C, or D)."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_row(sample, split, idx):
    audio_data = audio_to_bytes(sample.get("audio"))
    image_data = image_to_bytes(sample.get("image"))

    options = sample.get("options", [])
    options_text = "\n".join(options)

    prompt = build_prompt(sample["question"], options_text)

    row = {
        "data_source": "echoink/avqa_r1",
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "audios": [audio_data] if audio_data else [],
        "ability": "avqa",
        "reward_model": {
            "style": "multiple_choice",
            "ground_truth": sample.get("correct_label", "A"),
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "category": sample.get("category", "unknown"),
            "question": sample["question"],
            "correct_answer": sample.get("correct_answer", ""),
        },
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Preprocess AVQA-R1-6K dataset for verl-omni GSPO training")
    parser.add_argument(
        "--dataset_name",
        default="HarryHsing/AVQA-R1-6K",
        help=("HuggingFace dataset name. Supported: HarryHsing/AVQA-R1-6K, m-a-p/OmniInstruct_v1, m-a-p/OmniBench"),
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/avqa_r1_6k",
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
    np.random.seed(args.seed)

    if "AVQA-R1-6K" in args.dataset_name:
        train_samples, valid_samples = process_avqa_r1_6k(args.dataset_name)
    elif "OmniBench" in args.dataset_name:
        train_samples, valid_samples = process_omnibench(args.dataset_name)
    elif "OmniInstruct" in args.dataset_name:
        train_samples, valid_samples = process_omniinstruct(args.dataset_name)
    else:
        train_samples, valid_samples = process_avqa_r1_6k(args.dataset_name)

    if args.max_samples > 0:
        train_samples = train_samples[: args.max_samples]
        valid_samples = valid_samples[: max(args.max_samples // 4, 1)]

    if args.train_ratio is not None and not train_samples and not valid_samples:
        all_samples = train_samples + valid_samples
        random.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(n_total * args.train_ratio)
        train_samples = all_samples[:n_train]
        valid_samples = all_samples[n_train:]

    if not valid_samples and train_samples:
        random.shuffle(train_samples)
        n_total = len(train_samples)
        n_train = int(n_total * 0.85)
        valid_samples = train_samples[n_train:]
        train_samples = train_samples[:n_train]
        print(f"No valid split found, auto-split: train={len(train_samples)}, test={len(valid_samples)}")

    print(f"Train: {len(train_samples)}, Test: {len(valid_samples)}")

    train_rows = [build_row(s, "train", i) for i, s in enumerate(train_samples)]
    test_rows = [build_row(s, "test", i) for i, s in enumerate(valid_samples)]

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
    print("  data.image_key=images")
    print("  data.audio_key=audios")
    print("  reward.reward_manager.name=multiple_choice")


if __name__ == "__main__":
    main()
