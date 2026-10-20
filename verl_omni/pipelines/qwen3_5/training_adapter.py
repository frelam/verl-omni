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
"""Qwen3.5 (dense and MoE) training adapter.

Implements ``OmniModelBase`` for the Qwen3.5 vision-language checkpoints.
Unlike Qwen3-Omni, Qwen3.5 has no talker/codec and its top-level HF config
already exposes ``text_config`` / ``vision_config``, so nothing is stripped
and no Megatron hook is needed: the model trains with verl's native
``MegatronEngineWithLMHead`` (``model_type=language_model``).
"""

import json
import logging
import os
import types
from typing import Any

import numpy as np
import torch

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


def _model_cls_from_config(config):
    """Return the HF ``Qwen3_5*Model`` class that owns ``get_rope_index``."""
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        return Qwen3_5Model
    if model_type == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

        return Qwen3_5MoeModel
    raise ValueError(f"Qwen3.5 adapter does not support model_type={model_type!r}.")


@OmniModelBase.register("Qwen3_5ForConditionalGeneration", stage="thinker")
@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")
class Qwen3_5ThinkerAdapter(OmniModelBase):
    """Training adapter for Qwen3.5 VL (dense and MoE).

    Registered for both architectures under ``stage="thinker"`` because
    ``OmniModelConfig.model_stage`` defaults to ``thinker`` and Qwen3.5
    has no multi-stage layout. Training-side lookup is
    ``(architecture, stage)`` only, so sharing one class is unambiguous.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Qwen3.5 ships a single VL model; nothing is dead weight."""
        return []

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3.5 processor via verl's ``hf_processor`` and bind RoPE helpers.

        Qwen3.5's top-level config already carries ``vision_config`` and
        ``text_config``, so it is exposed to the processor as-is. Binds
        ``get_rope_index`` / ``get_vision_position_ids`` (model methods the
        omni agent loop calls on the processor) and ``dedup_pad_tokens``
        (collapses consecutive multimodal pad tokens before vLLM-Omni
        re-expands them).

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured processor with RoPE and dedup helpers bound.
        """
        from verl.utils import hf_processor

        processor = hf_processor(model_path, trust_remote_code=model_config.trust_remote_code, use_fast=True)
        if processor is None:
            raise ValueError(f"Failed to load a multimodal processor from {model_path}.")
        # ``hf_processor`` already attaches the top-level HF config and binds
        # ``get_rope_index``, but it dispatches on ``processor.__class__.__name__``:
        # Qwen3.5 checkpoints resolve to ``Qwen3VLProcessor``, so it binds
        # Qwen3-VL's implementation. Rebinding to the Qwen3.5 model class below
        # is a deliberate override (Qwen3.5 splits videos by timestamp).
        config = processor.config

        model_cls = _model_cls_from_config(config)

        def _get_rope_index(self, input_ids, mm_token_type_ids=None, **kwargs):
            # verl's agent loop only forwards mm_token_type_ids when the
            # processor output contained it; rebuild it from the special
            # token ids otherwise (text=0, image=1, video=2).
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
                image_token_id = getattr(self, "image_token_id", None)
                video_token_id = getattr(self, "video_token_id", None)
                if image_token_id is not None:
                    mm_token_type_ids[input_ids == image_token_id] = 1
                if video_token_id is not None:
                    mm_token_type_ids[input_ids == video_token_id] = 2
            return model_cls.get_rope_index(
                self,
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )

        processor.get_rope_index = types.MethodType(_get_rope_index, processor)
        processor.get_vision_position_ids = types.MethodType(model_cls.get_vision_position_ids, processor)

        # Collapse consecutive multimodal pad tokens before vLLM-Omni re-expands
        # them; mirrors verl's qwen2_5_vl_dedup_image_tokens.
        def _dedup_pad_tokens(self, prompt_ids: list[int]) -> list[int]:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                return prompt_ids
            pad_ids: set[int] = set()
            for tok_attr in ("image_token", "video_token"):
                tok = getattr(self, tok_attr, None)
                if tok is None:
                    continue
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                except Exception:
                    continue
                if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
                    continue
                pad_ids.add(int(tid))
            if not pad_ids:
                return prompt_ids
            arr = np.asarray(prompt_ids, dtype=np.int64)
            if arr.size == 0:
                return prompt_ids
            is_pad = np.isin(arr, list(pad_ids))
            keep = np.ones(arr.size, dtype=bool)
            same_as_prev = is_pad[1:] & is_pad[:-1] & (arr[1:] == arr[:-1])
            keep[1:] &= ~same_as_prev
            return arr[keep].tolist()

        processor.dedup_pad_tokens = types.MethodType(_dedup_pad_tokens, processor)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer, preferring a standalone ``chat_template.json``.

        Qwen3.5 ships the chat template inside ``tokenizer_config.json``, so
        the separate file is an optional override rather than a requirement.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured tokenizer.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if os.path.isfile(chat_template_path):
            with open(chat_template_path) as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        elif getattr(tokenizer, "chat_template", None) is None:
            raise FileNotFoundError(
                f"Qwen3.5 chat template not found at {chat_template_path} and the tokenizer "
                f"does not define one in tokenizer_config.json."
            )
        return tokenizer
