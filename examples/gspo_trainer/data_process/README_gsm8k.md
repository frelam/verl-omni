# GSM8K 数据预处理与 GSPO 训练指南

本指南介绍如何使用 GSM8K 数据集进行 Qwen3-Omni-Thinker 的 GSPO 强化学习训练。

## 数据集简介

GSM8K (Grade School Math 8K) 是 OpenAI 发布的小学数学应用题数据集，包含带逐步推理过程的数学问题，答案格式为 `"逐步推理 #### <最终数字>"`。

- 训练集：7,473 样本
- 测试集：1,319 样本

每个样本包含一个数学应用题和带推理过程的答案，模型需要理解问题并给出正确的数值答案。GSM8K 是纯文本任务，不涉及图像或音频。

## Step 1: 环境准备

按照 [安装指南](install.md) 安装 VeRL-Omni 及其依赖。GSM8K 为纯文本数据集，无需额外依赖。

## Step 2: 准备数据集

### 使用 openai/gsm8k（推荐）

```bash
python3 examples/gspo_trainer/data_process/gsm8k.py \
    --dataset_name openai/gsm8k \
    --output_dir ~/data/gsm8k
```

该命令会自动从 HuggingFace 下载数据集并生成：

- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

### 使用 Socratic 变体

```bash
python3 examples/gspo_trainer/data_process/gsm8k.py \
    --dataset_name openai/gsm8k \
    --config socratic \
    --output_dir ~/data/gsm8k_socratic
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_name` | `openai/gsm8k` | HuggingFace 数据集名称 |
| `--config` | `main` | 数据集配置（`main` 或 `socratic`） |
| `--output_dir` | `~/data/gsm8k` | 输出目录 |
| `--train_ratio` | None | 训练集比例（仅当数据集无内置划分时使用） |
| `--seed` | 42 | 随机种子 |
| `--max_samples` | -1 | 最大处理样本数（-1 为全部，调试用） |

### 输出格式

每行 parquet 数据包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `data_source` | str | `"openai/gsm8k"`，用于 reward 函数路由 |
| `prompt` | list[dict] | 聊天消息（system + user），包含数学问题 |
| `images` | list | 空列表（纯文本任务） |
| `audios` | list | 空列表（纯文本任务） |
| `ability` | str | `"math"` |
| `reward_model` | dict | `{"style": "rule", "ground_truth": "<数字答案>"}` |
| `extra_info` | dict | 包含 split、index、question、answer |

## Step 3: 启动训练

使用 GSPO 训练脚本启动训练：

```bash
bash examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh
```

训练脚本默认已配置为使用 GSM8K 数据集（`TRAIN_FILE` 和 `VAL_FILE` 指向 `~/data/gsm8k/`），并使用 `reward.reward_manager.name=dapo` 进行规则匹配奖励。

如需自定义 reward 函数，可在训练脚本中添加以下配置：

```yaml
data:
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet

reward:
  custom_reward_function:
    path: examples/gspo_trainer/gsm8k_reward.py
    name: compute_score
```

### Reward 函数

GSM8K 使用规则匹配奖励（rule-based reward），实现在 [gsm8k_reward.py](gsm8k_reward.py) 中：

- 从模型输出中提取最终数字答案，支持多种格式：
  - `#### <数字>`（标准格式）
  - `the answer is <数字>`
  - `final answer: <数字>`
  - 回复中最后一个数字（兜底策略）
- 自动去除逗号等格式差异（如 `1,234` 与 `1234` 视为相同）
- 与 `ground_truth` 比较，正确得 1.0 分，错误得 0.0 分
- 无法提取数字时得 0.0 分

## 调试技巧

### 快速验证数据预处理

使用 `--max_samples` 限制样本数，快速验证流程是否正常：

```bash
python3 examples/gspo_trainer/data_process/gsm8k.py \
    --max_samples 100 \
    --output_dir ~/data/gsm8k_test
```

### 检查生成的 parquet 文件

```python
import pandas as pd

df = pd.read_parquet("~/data/gsm8k/train.parquet")
print(f"Samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample prompt: {df.iloc[0]['prompt']}")
print(f"Ground truth: {df.iloc[0]['reward_model']}")
```

### 验证 Reward 函数

```python
from gsm8k_reward import compute_score

# 标准格式
assert compute_score("", "Let me think... #### 18", "18") == 1.0

# 自然语言格式
assert compute_score("", "The answer is 42", "42") == 1.0

# 带逗号的数字
assert compute_score("", "Result: 1,234 #### 1234", "1234") == 1.0

# 错误答案
assert compute_score("", "I got 99", "100") == 0.0
```
