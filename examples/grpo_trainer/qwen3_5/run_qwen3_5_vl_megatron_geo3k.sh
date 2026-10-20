#!/usr/bin/env bash
# Qwen3.5 / Qwen3.6 VL GRPO on verl's native Megatron engine and vLLM-Omni.
#
# Qwen3.6 reuses the Qwen3.5 checkpoint layout, so both train with the same
# adapters: set PIPELINE_NAME=qwen3_5_moe for the MoE checkpoints
# (Qwen3.5-*-A*B, Qwen3.6-35B-A3B) and PIPELINE_NAME=qwen3_5 for the dense ones.
#
# Full-parameter language-model RL; the vision tower stays frozen.
# Colocated (hybrid) recipe: the Megatron actor and the vLLM-Omni servers share
# the same four 8-GPU nodes (32 GPUs) and take turns. Actor grid is
# TP4 x CP2 x PP2 x DP2 with EP8/ETP1; rollout uses TP4, i.e. eight replicas.
# The point of this recipe is to exercise Megatron hybrid TP+CP+PP+EP parallelism
# while the actor and rollout are time-multiplexed on the same devices.
# Prepare TRAIN_FILE/VAL_FILE with examples/gspo_trainer/data_process/geo3k.py.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen3.5 or Qwen3.6 checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to prepared Geo3K train.parquet}"
: "${VAL_FILE:?Set VAL_FILE to prepared Geo3K validation.parquet}"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/qwen35-run.XXXXXX")}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
# Each invocation gets a separate log and resolved configuration.
RUN_DIR=$(mktemp -d "${OUTPUT_DIR}/run.XXXXXX")
echo "Qwen3.5/Qwen3.6 artifacts: ${RUN_DIR}"
# qwen3_5_moe for MoE checkpoints, qwen3_5 for dense ones.
PIPELINE_NAME=${PIPELINE_NAME:-qwen3_5_moe}
# EP lives on its own axis (expert list), not on the dense TP/CP/PP/DP grid:
# with expert_tensor_parallel_size=1 the only requirement is that
# world_size % (EP * PP) == 0 and 32 / (EP * PP) == DP. Dense checkpoints
# have no experts -> EP=1.
TP=${TP:-4}
CP=${CP:-2}
PP=${PP:-2}
EP=${EP:-8}
export VERL_USE_EXTERNAL_MODULES=verl_omni
export TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

args=(
  --config-name omni_megatron_trainer
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  data.train_batch_size=64
  data.max_prompt_length=1024
  data.max_response_length=3072
  data.filter_overlong_prompts=true
  # `language_model` dispatches to verl's MegatronEngineWithLMHead while the
  # `_target_` still loads OmniModelConfig and the Qwen3.5 adapter.
  actor_rollout_ref.model.model_type=language_model
  # THD packing is on by default through OmniModelConfig.use_remove_padding.
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=false
  # CP>1 enables Megatron's per-token loss; only globally composable aggregations
  # are allowed. token-mean divides the summed token losses by the global token
  # count, which is exact after the CP reduction.
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.use_precision_aware_optimizer=true
  # Offload FP32 Adam states and use precision-aware updates for the actor.
  '+actor_rollout_ref.actor.optim.override_optimizer_config={optimizer_cpu_offload:true,optimizer_offload_fraction:1.0,overlap_cpu_optimizer_d2h_h2d:false}'
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
  actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  # Colocated actor/rollout: keep params, grads and Adam states off the GPUs
  # while the vLLM-Omni servers own the devices.
  actor_rollout_ref.actor.megatron.param_offload=true
  actor_rollout_ref.actor.megatron.optimizer_offload=true
  actor_rollout_ref.actor.megatron.grad_offload=true
  # Match the checkpoint: train/eval log-probs must not differ due to dropout.
  +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=none
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_language_model=false
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_vision_model=true
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  'actor_rollout_ref.ref.megatron.override_transformer_config={gradient_accumulation_fusion:false,attention_dropout:0.0,hidden_dropout:0.0}'
  # Colocated actor + rollout: one worker group, vLLM-Omni servers sleep while
  # the actor trains and wake for generation.
  actor_rollout_ref.hybrid_engine=true
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.n=1
  # Colocated: the rollout replicas are built from the actor worker group, so
  # there is no separate node budget; 32 GPUs / TP4 = 8 replicas.
  actor_rollout_ref.rollout.nnodes=0
  actor_rollout_ref.rollout.n_gpus_per_node=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP}
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.max_model_len=4096
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=${PIPELINE_NAME}
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.image=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.video=0
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  algorithm.rollout_correction.bypass_mode=false
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni
  '+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${oc.env:TENSORBOARD_DIR}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=${oc.env:CUDA_DEVICE_MAX_CONNECTIONS,1}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0"'
  '+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED="1"'
  # omni_sync == synchronous colocated trainer: actor and rollout share devices.
  trainer.v1.trainer_mode=omni_sync
  trainer.nnodes=4
  trainer.n_gpus_per_node=8
  trainer.total_training_steps=150
  trainer.val_before_train=true
  trainer.save_freq=-1
  trainer.test_freq=10
  trainer.resume_mode=disable
  'trainer.logger=[console,tensorboard]'
  trainer.project_name=qwen3_5_geo3k
  trainer.experiment_name=megatron_colocated_hybrid_parallel
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
  "trainer.rollout_data_dir=${RUN_DIR}/rollouts"
  "trainer.validation_data_dir=${RUN_DIR}/validation"
  "$@"
)
printf '%q ' "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
git rev-parse HEAD > "${RUN_DIR}/commit.txt"
VLLM_LOGGING_STREAM=ext://sys.stderr "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" \
  --cfg job --resolve > "${RUN_DIR}/config.yaml" 2> >(tee "${RUN_DIR}/config.log" >&2)
# Let CPU regression tests exercise the launcher's exact Hydra composition.
if [[ ${CONFIG_ONLY:-0} == 1 ]]; then
  exit 0
fi
"${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
