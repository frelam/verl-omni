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
"""Qwen3-Omni Thinker training adapter.

Implements ``OmniModelBase`` for thinker-stage training of
Qwen3-Omni: sub-module stripping, forward redirection,
processor/tokenizer configuration, and LoRA key normalization for
vLLM-Omni weight sync.
"""

import json
import logging
import os
from typing import Any

import numpy as np
import torch

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")
class Qwen3OmniThinkerAdapter(OmniModelBase):
    """Thinker-stage training adapter for Qwen3-Omni.

    Handles model setup that is required before verl's FSDP engine
    loads and wraps the model: sub-module stripping, forward redirection
    to the thinker component, and processor/tokenizer configuration.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["talker", "code2wav", "code_predictor"]

    @classmethod
    def configure_model(cls, module, model_config):
        """Strip non-training stages and redirect forward to thinker.

        Args:
            module: The loaded Qwen3-Omni model before FSDP wrapping.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured module with talker/codec stripped and
            forward/embedding accessors redirected to thinker.
        """
        module = super().configure_model(module, model_config)
        module.forward = module.thinker.forward
        module.get_input_embeddings = module.thinker.get_input_embeddings
        module.set_input_embeddings = module.thinker.set_input_embeddings
        module._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3-Omni multimodal processor with RoPE + dedup helpers.

        Swaps ``processor.config`` to ``thinker_config`` (Qwen3-Omni nests
        multimodal settings under sub-configs). Binds ``get_rope_index`` and
        ``get_llm_pos_ids_for_vision`` (model methods the omni agent loop
        calls on the processor), and ``dedup_pad_tokens`` (collapses
        consecutive multimodal pad tokens before vLLM-Omni re-expands them).
        ``get_rope_index`` additionally demotes stray response-emitted audio
        special tokens to plain-text positions, because HF crashes on audio
        segments that have no matching ``audio_seqlens`` entry.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured processor with RoPE and dedup helpers bound.
        """
        import types

        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        from verl_omni.pipelines.qwen3_omni.video_processor import Qwen3OmniVideoProcessor

        processor = Qwen3OmniVideoProcessor.from_pretrained(
            model_path, trust_remote_code=model_config.trust_remote_code
        )
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)

        processor.config = config.thinker_config
        processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        processor.config.vision_start_token_id = config.talker_config.vision_start_token_id

        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration

        def _sanitize_stray_audio_specials(self, input_ids, audio_seqlens):
            """Demote audio special tokens not backed by a real audio to plain text.

            RL rollouts can emit ``<|audio_bos|>``/``<|audio_pad|>`` in the
            response; HF ``get_rope_index`` indexes ``audio_seqlens`` once per
            ``<|audio_bos|>`` segment it finds (no None/length guard), so any
            segment beyond the real audios must be treated as plain text to
            keep position computation from crashing. Returns ``input_ids``
            unchanged when there is nothing to fix; never mutates in place.
            """
            if not torch.is_tensor(input_ids):
                return input_ids
            audio_bos = getattr(self.config, "audio_start_token_id", None)
            audio_pad = getattr(self.config, "audio_token_id", None)
            if audio_bos is None or audio_pad is None:
                return input_ids
            tokenizer = getattr(self, "tokenizer", None)
            audio_eos = None
            if tokenizer is not None:
                try:
                    tid = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
                    if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                        audio_eos = int(tid)
                except Exception:
                    pass
            special_ids = {int(audio_bos), int(audio_pad)}
            if audio_eos is not None:
                special_ids.add(audio_eos)
            if audio_seqlens is None:
                n_real = 0
            elif torch.is_tensor(audio_seqlens):
                n_real = audio_seqlens.numel()
            else:
                n_real = len(audio_seqlens)
            replacement_id = getattr(tokenizer, "pad_token_id", None) or 0

            rows = input_ids.tolist()
            n_replaced = 0
            for row in rows:
                bos_positions = [i for i, tid in enumerate(row) if tid == audio_bos]
                if len(bos_positions) <= n_real:
                    continue
                if n_real:
                    # Keep the real segments intact; sanitize from the end of the
                    # last real segment (its ``<|audio_eos|>``) onward.
                    last_real_bos, first_stray_bos = bos_positions[n_real - 1], bos_positions[n_real]
                    eos = next((i for i in range(last_real_bos + 1, first_stray_bos) if row[i] == audio_eos), None)
                    start = eos + 1 if eos is not None else first_stray_bos
                else:
                    start = 0
                for i in range(start, len(row)):
                    if row[i] in special_ids:
                        row[i] = replacement_id
                        n_replaced += 1
            if not n_replaced:
                return input_ids
            logger.warning(
                "Demoted %d stray audio special token(s) to pad id %d for RoPE index computation; "
                "the rollout likely emitted <|audio_bos|>/<|audio_pad|> in the response.",
                n_replaced,
                replacement_id,
            )
            return torch.tensor(rows, dtype=input_ids.dtype, device=input_ids.device)

        # Cast to int64: HF returns float32, FSDP would otherwise bf16-round positions.
        def _get_rope_index_long(self, *args, **kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            sanitized = _sanitize_stray_audio_specials(self, input_ids, kwargs.get("audio_seqlens"))
            if sanitized is not input_ids:
                if "input_ids" in kwargs or not args:
                    kwargs["input_ids"] = sanitized
                else:
                    args = (sanitized, *args[1:])
            vision_position_ids, deltas = model_cls.get_rope_index(self, *args, **kwargs)
            return vision_position_ids.long(), deltas

        processor.get_rope_index = types.MethodType(_get_rope_index_long, processor)
        processor.get_llm_pos_ids_for_vision = types.MethodType(model_cls.get_llm_pos_ids_for_vision, processor)

        # Provide audio lengths to verl's generic V1 agent loop via get_rope_index_kwargs.
        def _get_rope_index_kwargs(multi_modal_inputs: dict) -> dict:
            result = {}
            seconds = multi_modal_inputs.get("video_second_per_grid")
            if seconds is not None:
                result["second_per_grids"] = seconds
            feature_attention_mask = multi_modal_inputs.get("feature_attention_mask")
            if feature_attention_mask is not None:
                result["audio_seqlens"] = feature_attention_mask.sum(-1)
            return result

        processor.get_rope_index_kwargs = _get_rope_index_kwargs

        # Collapse consecutive multimodal pad tokens before vLLM-Omni re-expands
        # them (token-IDs path still unfixed: https://github.com/vllm-project/vllm/issues/33672);
        # mirrors verl's qwen2_5_vl_dedup_image_tokens.
        def _dedup_pad_tokens(self, prompt_ids: list[int]) -> list[int]:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                return prompt_ids
            pad_ids: set[int] = set()
            for tok_attr in ("image_token", "video_token", "audio_token"):
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
        """Load the tokenizer with chat template from ``chat_template.json``.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured tokenizer with ``chat_template`` loaded from
            ``chat_template.json``.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if not os.path.isfile(chat_template_path):
            raise FileNotFoundError(
                f"Qwen3-Omni chat template not found at {chat_template_path}. "
                f"Ensure the model checkpoint includes chat_template.json."
            )
        with open(chat_template_path) as f:
            tokenizer.chat_template = json.load(f)["chat_template"]
        return tokenizer
