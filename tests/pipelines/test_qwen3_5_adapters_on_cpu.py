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
"""CPU tests for the Qwen3.5 training and rollout adapters."""

import json
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.model_base import OmniModelBase, OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_5.rollout_adapter import (
    QWEN3_5_MOE_PIPELINE,
    QWEN3_5_PIPELINE,
    Qwen3_5MoeRolloutAdapter,
    Qwen3_5RolloutAdapter,
)
from verl_omni.pipelines.qwen3_5.training_adapter import Qwen3_5ThinkerAdapter

_IMAGE_TOKEN = "<|image_pad|>"
_VIDEO_TOKEN = "<|video_pad|>"
_IMAGE_TOKEN_ID = 8
_VIDEO_TOKEN_ID = 9


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"],
)
def test_training_adapter_registered_for_both_architectures(architecture):
    assert OmniModelBase.get_class_by_name(architecture, "thinker") is Qwen3_5ThinkerAdapter


def test_rollout_adapters_registered_per_model_type():
    assert OmniRolloutPipelineBase.get_class("qwen3_5") is Qwen3_5RolloutAdapter
    assert OmniRolloutPipelineBase.get_class("qwen3_5_moe") is Qwen3_5MoeRolloutAdapter
    _require_real_vllm_omni()
    assert Qwen3_5RolloutAdapter.get_pipeline_id() == "qwen3_5"
    assert Qwen3_5MoeRolloutAdapter.get_pipeline_id() == "qwen3_5_moe"


def test_get_strip_modules_is_empty():
    assert Qwen3_5ThinkerAdapter.get_strip_modules(SimpleNamespace()) == []


# ---------------------------------------------------------------------------
# Processor / tokenizer
# ---------------------------------------------------------------------------


class Qwen3VLProcessor:
    """Stand-in for the real processor; named to match verl's dispatch table."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _fake_processor():
    return Qwen3VLProcessor(
        image_token=_IMAGE_TOKEN,
        video_token=_VIDEO_TOKEN,
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        tokenizer=SimpleNamespace(
            unk_token_id=0,
            convert_tokens_to_ids=lambda token: {_IMAGE_TOKEN: _IMAGE_TOKEN_ID, _VIDEO_TOKEN: _VIDEO_TOKEN_ID}.get(
                token, 0
            ),
        ),
    )


def _configure_processor(monkeypatch, *, model_type="qwen3_5", processor=None):
    from transformers import AutoConfig, AutoProcessor

    processor = processor if processor is not None else _fake_processor()
    config = SimpleNamespace(
        model_type=model_type,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    configured = Qwen3_5ThinkerAdapter.configure_processor(
        "/fake/qwen3-5",
        SimpleNamespace(trust_remote_code=False),
    )
    return configured, config


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_configure_processor_binds_rope_and_dedup(monkeypatch, model_type):
    pytest.importorskip("transformers")

    configured, config = _configure_processor(monkeypatch, model_type=model_type)

    assert configured.config is config
    assert callable(configured.get_rope_index)
    assert callable(configured.get_vision_position_ids)
    assert configured.dedup_pad_tokens([7, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 7, _VIDEO_TOKEN_ID, _VIDEO_TOKEN_ID]) == [
        7,
        _IMAGE_TOKEN_ID,
        7,
        _VIDEO_TOKEN_ID,
    ]


def test_get_rope_index_rebuilds_mm_token_type_ids(monkeypatch):
    """verl's agent loop drops mm_token_type_ids; the bound method must rebuild it."""
    pytest.importorskip("transformers")

    configured, _ = _configure_processor(monkeypatch)
    # One image (1x4x4 grid, spatial merge 2 -> 4 LLM tokens) between two texts.
    input_ids = torch.tensor([[5, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 6]])
    attention_mask = torch.ones_like(input_ids)

    position_ids, deltas = configured.get_rope_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )

    assert position_ids.shape == (3, 1, 6)
    assert position_ids.dtype == torch.long
    # Text before the image keeps the running counter; the text after it starts
    # one past the image block's largest 3D index (max(4, 4) // merge = 2).
    assert position_ids[:, 0, 0].tolist() == [0, 0, 0]
    assert position_ids[:, 0, 5].tolist() == [3, 3, 3]
    # The image block gets 3D positions offset by current_pos=1.
    assert position_ids[0, 0, 1:5].tolist() == [1, 1, 1, 1]
    assert position_ids[1, 0, 1:5].tolist() == [1, 1, 2, 2]
    assert position_ids[2, 0, 1:5].tolist() == [1, 2, 1, 2]
    assert deltas.shape == (1, 1)


def test_get_rope_index_honours_explicit_mm_token_type_ids(monkeypatch):
    """An explicit all-text map must not treat image tokens as vision."""
    pytest.importorskip("transformers")

    configured, _ = _configure_processor(monkeypatch)
    input_ids = torch.tensor([[5, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 6]])
    position_ids, _ = configured.get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=torch.zeros_like(input_ids),
        attention_mask=torch.ones_like(input_ids),
        image_grid_thw=None,
    )
    assert position_ids[:, 0, :].tolist() == [[0, 1, 2, 3]] * 3


def test_configure_tokenizer_prefers_sidecar_template(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "sidecar"}))
    tokenizer = SimpleNamespace(chat_template="builtin")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    configured = Qwen3_5ThinkerAdapter.configure_tokenizer(
        str(tmp_path), SimpleNamespace(trust_remote_code=False)
    )

    assert configured.chat_template == "sidecar"


def test_configure_tokenizer_falls_back_to_builtin_template(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tokenizer = SimpleNamespace(chat_template="builtin")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    configured = Qwen3_5ThinkerAdapter.configure_tokenizer(
        str(tmp_path), SimpleNamespace(trust_remote_code=False)
    )

    assert configured.chat_template == "builtin"


def test_configure_tokenizer_requires_a_template(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tokenizer = SimpleNamespace(chat_template=None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    with pytest.raises(FileNotFoundError, match="chat template"):
        Qwen3_5ThinkerAdapter.configure_tokenizer(str(tmp_path), SimpleNamespace(trust_remote_code=False))


# ---------------------------------------------------------------------------
# Rollout topology (needs the real vLLM-Omni engine stack)
# ---------------------------------------------------------------------------


def _require_real_vllm_omni():
    from vllm_omni.config import stage_config

    if not isinstance(QWEN3_5_PIPELINE, stage_config.PipelineConfig):
        pytest.skip("vllm-omni stage config is stubbed; topology assertions need the real engine stack")


def test_rollout_pipeline_topology_declares_single_text_stage():
    _require_real_vllm_omni()
    from vllm_omni.config.stage_config import StageExecutionType

    for pipeline, architecture in (
        (QWEN3_5_PIPELINE, "Qwen3_5ForConditionalGeneration"),
        (QWEN3_5_MOE_PIPELINE, "Qwen3_5MoeForConditionalGeneration"),
    ):
        assert pipeline.model_arch == architecture
        assert pipeline.hf_architectures == (architecture,)
        assert len(pipeline.stages) == 1
        stage = pipeline.stages[0]
        assert stage.stage_id == 0
        assert stage.execution_type == StageExecutionType.LLM_AR
        assert stage.final_output is True
        assert stage.final_output_type == "text"
        assert stage.engine_output_type == "text"
        assert stage.owns_tokenizer is True
        assert stage.requires_multimodal_data is True
        assert stage.input_sources == ()


def test_build_stage_configs_returns_the_single_stage():
    _require_real_vllm_omni()

    stages = Qwen3_5RolloutAdapter.build_stage_configs("thinker_only")
    assert len(stages) == 1
    assert stages[0] is QWEN3_5_PIPELINE.stages[0]

    moe_stages = Qwen3_5MoeRolloutAdapter.build_stage_configs("thinker_only")
    assert len(moe_stages) == 1
    assert moe_stages[0] is QWEN3_5_MOE_PIPELINE.stages[0]

    with pytest.raises(ValueError, match="thinker_only only"):
        Qwen3_5RolloutAdapter.build_stage_configs("full")


def test_ensure_pipeline_registered_registers_the_owning_pipeline(monkeypatch):
    from verl_omni.pipelines.qwen3_5 import rollout_adapter

    calls = []
    monkeypatch.setattr(rollout_adapter, "register_pipeline", lambda pipeline: calls.append(pipeline))

    rollout_adapter.Qwen3_5RolloutAdapter.ensure_pipeline_registered()
    rollout_adapter.Qwen3_5MoeRolloutAdapter.ensure_pipeline_registered()

    assert calls == [rollout_adapter.QWEN3_5_PIPELINE, rollout_adapter.QWEN3_5_MOE_PIPELINE]
