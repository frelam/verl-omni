# AVQA-R1-6K 数据预处理与 GSPO 训练指南

本指南介绍如何使用 AVQA-R1-6K 数据集进行 Qwen3-Omni-Thinker 的 GSPO 强化学习训练。

## 数据集简介

AVQA-R1-6K 来自 [EchoInk-R1](https://github.com/HarryHsing/EchoInk) 项目，基于 [OmniInstruct-v1](https://huggingface.co/datasets/m-a-p/OmniInstruct_v1) 构建，包含同步的音频-图像对和四选一多选题，用于音频-视觉联合推理任务。

- 训练集：4,490 样本
- 验证集：1,911 样本

每个样本包含一段音频、一张图像、一个问题和四个选项，模型需要同时理解音频和图像内容来选择正确答案。

## Step 1: 环境准备

按照 [安装指南](install.md) 安装 VeRL-Omni 及其依赖，然后安装音频处理依赖：

```bash
pip install soundfile
```

## Step 2: 准备数据集

### 使用 AVQA-R1-6K（推荐）

```bash
python3 examples/gspo_trainer/data_process/avqa_r1_6k.py \
    --dataset_name HarryHsing/AVQA-R1-6K \
    --output_dir ~/data/avqa_r1_6k
```

该命令会自动从 HuggingFace 下载数据集并生成：

- `~/data/avqa_r1_6k/train.parquet`
- `~/data/avqa_r1_6k/test.parquet`

### 使用上游 OmniInstruct-v1

如果 AVQA-R1-6K 不可用，可以从上游数据集提取 AVQA 子集：

```bash
python3 examples/gspo_trainer/data_process/avqa_r1_6k.py \
    --dataset_name m-a-p/OmniInstruct_v1 \
    --output_dir ~/data/avqa_r1_6k
```

脚本会自动过滤 `source == "AVQA"` 的样本，并使用数据集自带的 train/valid 划分。

### 使用 OmniBench

```bash
python3 examples/gspo_trainer/data_process/avqa_r1_6k.py \
    --dataset_name m-a-p/OmniBench \
    --output_dir ~/data/avqa_r1_6k
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_name` | `HarryHsing/AVQA-R1-6K` | HuggingFace 数据集名称 |
| `--output_dir` | `~/data/avqa_r1_6k` | 输出目录 |
| `--train_ratio` | None | 训练集比例（仅当数据集无内置划分时使用） |
| `--seed` | 42 | 随机种子 |
| `--max_samples` | -1 | 最大处理样本数（-1 为全部，调试用） |

### 输出格式

每行 parquet 数据包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `data_source` | str | `"echoink/avqa_r1"`，用于 reward 函数路由 |
| `prompt` | list[dict] | 聊天消息，包含 `<audio>` / `<image>` 占位符 |
| `images` | list[dict] | 图像数据列表（bytes 格式） |
| `audios` | list[dict] | 音频数据列表（bytes + sampling_rate） |
| `ability` | str | `"avqa"` |
| `reward_model` | dict | `{"style": "multiple_choice", "ground_truth": "A/B/C/D"}` |
| `extra_info` | dict | 包含 split、index、category、question 等 |

## Step 3: 启动训练

使用 GSPO 训练脚本启动训练：

```bash
bash examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh
```

训练脚本中需要确保以下配置与数据集匹配：

```yaml
data:
  train_files: ~/data/avqa_r1_6k/train.parquet
  val_files: ~/data/avqa_r1_6k/test.parquet
  image_key: images
  audio_key: audios

reward:
  custom_reward_function:
    path: examples/gspo_trainer/avqa_reward.py
    name: compute_score
```

### Reward 函数

AVQA 多选题使用准确率奖励（accuracy reward），实现在 [avqa_reward.py](avqa_reward.py) 中：

- 从模型输出中提取选项字母（A/B/C/D）
- 与 `ground_truth` 比较，正确得 1.0 分，错误得 0.0 分
- 无法提取选项字母时得 0.0 分

## 调试技巧

### 快速验证数据预处理

使用 `--max_samples` 限制样本数，快速验证流程是否正常：

```bash
python3 examples/gspo_trainer/data_process/avqa_r1_6k.py \
    --max_samples 100 \
    --output_dir ~/data/avqa_r1_6k_test
```

### 检查生成的 parquet 文件

```python
import pandas as pd

df = pd.read_parquet("~/data/avqa_r1_6k/train.parquet")
print(f"Samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample prompt: {df.iloc[0]['prompt']}")
print(f"Ground truth: {df.iloc[0]['reward_model']}")
```
