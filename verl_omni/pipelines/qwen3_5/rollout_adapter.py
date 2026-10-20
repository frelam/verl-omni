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
"""Qwen3.5 rollout pipeline adapters (single AR text stage).

vLLM-Omni registers no Qwen3.5 pipeline, so this module owns the topology:
one ``LLM_AR`` stage that owns the tokenizer, consumes multimodal data, and
emits text.
"""

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

QWEN3_5_PIPELINE = PipelineConfig(
    model_type="qwen3_5",
    model_arch="Qwen3_5ForConditionalGeneration",
    hf_architectures=("Qwen3_5ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

QWEN3_5_MOE_PIPELINE = PipelineConfig(
    model_type="qwen3_5_moe",
    model_arch="Qwen3_5MoeForConditionalGeneration",
    hf_architectures=("Qwen3_5MoeForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


class _Qwen3_5RolloutAdapter(OmniRolloutPipelineBase):
    """Shared thinker-only topology for both Qwen3.5 model types."""

    pipeline: PipelineConfig

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        """Return the single text-output stage of this adapter's pipeline.

        Args:
            pipeline_mode (str): Pipeline mode selector; ``thinker_only`` only.

        Returns:
            list: The one stage this adapter supports.

        Raises:
            ValueError: When a different mode is requested.
        """
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"{cls.__name__} implements thinker_only only, got {pipeline_mode!r}. "
                "Qwen3.5 has no talker/codec stage."
            )
        stages = list(cls.pipeline.stages)
        # Guard against topology edits that silently add stages.
        assert len(stages) == 1, (
            f"Expected 1 stage in the thinker-only pipeline, got {len(stages)}. "
            "The Qwen3.5 pipeline definition is wrong."
        )
        return stages

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        """Return this adapter's registered pipeline model_type."""
        return cls.pipeline.model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "thinker_only") -> None:
        """Register the out-of-tree Qwen3.5 pipeline in vLLM-Omni."""
        register_pipeline(cls.pipeline)


@OmniRolloutPipelineBase.register("qwen3_5")
class Qwen3_5RolloutAdapter(_Qwen3_5RolloutAdapter):
    """Rollout topology adapter for dense Qwen3.5 VL checkpoints."""

    pipeline = QWEN3_5_PIPELINE


@OmniRolloutPipelineBase.register("qwen3_5_moe")
class Qwen3_5MoeRolloutAdapter(_Qwen3_5RolloutAdapter):
    """Rollout topology adapter for MoE Qwen3.5 VL checkpoints."""

    pipeline = QWEN3_5_MOE_PIPELINE
