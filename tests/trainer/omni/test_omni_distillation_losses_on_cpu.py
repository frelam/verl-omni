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
"""CPU tests for the omni hidden-state distillation loss dispatch (Task 3)."""

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from verl_omni.trainer.distillation.losses import (
    omni_distillation_ppo_loss,
)
from verl_omni.workers.config.omni.distillation import HIDDEN_STATE_LOSS_MODES


class _LossConfig:
    def __init__(self, loss_mode="nitrobrew", use_policy_gradient=False, use_hidden_states=True, topk=True):
        self.loss_mode = loss_mode
        self.use_policy_gradient = use_policy_gradient
        self.loss_settings = type("S", (), {"use_topk": topk, "use_hidden_states": use_hidden_states})()


class _DistillConfig:
    def __init__(self, loss_mode="nitrobrew"):
        self.distillation_loss = _LossConfig(loss_mode=loss_mode)


class TestLogitsProcessorDispatch:
    def test_nitrobrew_goes_to_kernel(self, monkeypatch):
        called = {}

        def fake_multi_kl(
            student_logits=None,
            teacher_hidden_states=None,
            teacher_key_ids=None,
            teacher_unembeds=None,
            config=None,
            data_format=None,
        ):
            called["called"] = True
            assert student_logits.shape == (1, 4, 32)
            assert teacher_key_ids.shape[0] == 4
            assert set(teacher_unembeds.keys()) == {0}
            return {"distillation_losses": torch.zeros(1, 4)}

        monkeypatch.setattr("verl_omni.trainer.distillation.losses.compute_nitrobrew_multi_kl", fake_multi_kl)

        hidden = torch.randn(1, 4, 8)
        student_logits = torch.randn(1, 4, 32)
        data = TensorDict(
            {
                "teacher_hidden_states": hidden,
                "teacher_unembeds": NonTensorData({"t0": torch.randn(32, 8)}),
                "teacher_key_to_id": NonTensorData({"t0": 0}),
            },
            batch_size=[1],
        )

        out = omni_distillation_ppo_loss(
            config=None,
            distillation_config=_DistillConfig("nitrobrew"),
            data=data,
            student_logits=student_logits,
            data_format="thd",
        )
        assert called["called"]
        assert "distillation_losses" in out

    def test_estimator_mode_delegates(self, monkeypatch):
        """non-hidden loss_mode must fall through to verl's distillation_ppo_loss."""

        # provide a stub to prove delegation
        def fake_verl_ppo_loss(config, distillation_config, model_output, data, dp_group, student_logits, data_format):
            return "DELEGATED"

        monkeypatch.setattr("verl_omni.trainer.distillation.losses.distillation_ppo_loss", fake_verl_ppo_loss)
        out = omni_distillation_ppo_loss(
            config=None,
            distillation_config=_DistillConfig("kl"),
            student_logits=torch.randn(1, 2, 4),
            data_format="thd",
        )
        assert out == "DELEGATED"


class TestMegatronDispatch:
    """config.strategy == 'megatron' must route to the vocab-parallel kernel."""

    def _data(self, with_key_ids=False):
        fields = {
            "teacher_hidden_states": torch.randn(1, 4, 8),
            "teacher_unembeds": NonTensorData({"t0": torch.randn(32, 8)}),
            "teacher_key_to_id": NonTensorData({"t0": 0}),
            "local_cp_size": NonTensorData(2),
        }
        if with_key_ids:
            fields["teacher_key_ids"] = torch.tensor([0])
        return TensorDict(fields, batch_size=[1])

    def test_megatron_strategy_routes_to_megatron_kernel(self, monkeypatch):
        called = {}

        def fake_megatron_kl(
            student_logits=None,
            teacher_hidden_states=None,
            teacher_key_ids=None,
            teacher_unembeds=None,
            config=None,
            data_format=None,
            local_cp_size=None,
        ):
            called.update(
                student_logits=student_logits,
                teacher_key_ids=teacher_key_ids,
                local_cp_size=local_cp_size,
                unembed_keys=set(teacher_unembeds.keys()),
            )
            return {"distillation_losses": torch.zeros(1, 4)}

        monkeypatch.setattr(
            "verl_omni.trainer.distillation.megatron_nitrobrew_loss.compute_nitrobrew_multi_kl_megatron",
            fake_megatron_kl,
        )
        student_logits = torch.randn(1, 4, 16)  # [bsz, seqlen/cp, vocab/tp] shard
        cfg = type("C", (), {"strategy": "megatron"})()
        out = omni_distillation_ppo_loss(
            config=cfg,
            distillation_config=_DistillConfig("nitrobrew"),
            data=self._data(with_key_ids=True),
            student_logits=student_logits,
            data_format="thd",
        )
        assert "distillation_losses" in out
        assert called["student_logits"] is student_logits
        assert called["teacher_key_ids"].tolist() == [0]
        assert called["local_cp_size"] == 2  # unwrapped from NonTensorData
        assert called["unembed_keys"] == {0}

    def test_megatron_reverse_mode_uses_reverse_kernel(self, monkeypatch):
        called = {}

        def fake_reverse(**kwargs):
            called["reverse"] = True
            return {"distillation_losses": torch.zeros(1, 4)}

        def fake_forward(**kwargs):
            raise AssertionError("forward kernel must not be called in reverse mode")

        monkeypatch.setattr(
            "verl_omni.trainer.distillation.megatron_nitrobrew_loss.compute_nitrobrew_multi_reverse_kl_megatron",
            fake_reverse,
        )
        monkeypatch.setattr(
            "verl_omni.trainer.distillation.megatron_nitrobrew_loss.compute_nitrobrew_multi_kl_megatron",
            fake_forward,
        )
        cfg = type("C", (), {"strategy": "megatron"})()
        out = omni_distillation_ppo_loss(
            config=cfg,
            distillation_config=_DistillConfig("nitrobrew_reverse_kl"),
            data=self._data(),
            student_logits=torch.randn(1, 4, 16),
            data_format="thd",
        )
        assert called["reverse"]
        assert "distillation_losses" in out


class TestSingleTeacherRouting:
    def test_zero_key_ids_with_single_teacher(self):
        from verl_omni.trainer.distillation.losses import _per_token_teacher_key_ids

        hidden = torch.randn(1, 4, 8)
        data = TensorDict({"teacher_hidden_states": hidden}, batch_size=[1])
        ids = _per_token_teacher_key_ids(data, hidden, {"t0": 0})
        assert ids.shape[0] == 4
        assert ids.abs().sum().item() == 0


class TestMultiTeacherRouting:
    def test_missing_key_ids_raises(self):
        """Multi-teacher without per-sequence ids must fail loudly, not route to 0."""
        from verl_omni.trainer.distillation.losses import _per_token_teacher_key_ids

        hidden = torch.randn(1, 4, 8)
        data = TensorDict({"teacher_hidden_states": hidden}, batch_size=[1])
        with pytest.raises(KeyError, match="teacher_key_ids"):
            _per_token_teacher_key_ids(data, hidden, {"t0": 0, "t1": 1})

    def test_per_sequence_ids_expand_over_nested_tokens(self):
        from verl_omni.trainer.distillation.losses import _per_token_teacher_key_ids

        hidden = torch.nested.nested_tensor([torch.randn(3, 8), torch.randn(2, 8)], layout=torch.jagged)
        data = TensorDict(
            {"teacher_hidden_states": hidden, "teacher_key_ids": torch.tensor([1, 0])},
            batch_size=[2],
        )
        ids = _per_token_teacher_key_ids(data, hidden, {"t0": 0, "t1": 1})
        assert ids.tolist() == [1, 1, 1, 0, 0]


class TestAggregateRegistered:
    def test_nitrobrew_in_verl_registry(self):
        """importing the module must register nitrobrew aggregate into verl's registry."""
        from verl.trainer.distillation.losses import get_distillation_loss_fn, get_distillation_loss_settings

        for mode in HIDDEN_STATE_LOSS_MODES:
            fn = get_distillation_loss_fn(mode)
            settings = get_distillation_loss_settings(mode)
            assert callable(fn)
            assert set(settings.names) == set(HIDDEN_STATE_LOSS_MODES)


class TestNitrobrewAggregateOverlapMetric:
    """The nitrobrew aggregate must emit distillation/overlap_ratio from the
    kernel's stride-subsampled overlap_counts (-1 marks non-sampled rows)."""

    def _data(self):
        # bsz=1, prompt_len=3, resp_len=4 -> total_nnz=7.
        return TensorDict(
            {
                "prompts": torch.zeros(1, 3, dtype=torch.long),
                "responses": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 7, dtype=torch.long),
                "response_mask": torch.ones(1, 4, dtype=torch.bool),
            },
            batch_size=[1],
        )

    def _distill_config(self, topk=4):
        return type("D", (), {"distillation_loss": type("L", (), {"topk": topk})()})()

    def test_overlap_ratio_emitted(self):
        from verl_omni.trainer.distillation.losses import compute_nitrobrew_loss_aggregate

        # response slice covers flat indices 2..5 -> counts 1, 2, 0, 4.
        model_output = {
            "distillation_losses": torch.arange(7, dtype=torch.float32),
            "overlap_counts": torch.tensor([-1.0, -1.0, 1.0, 2.0, 0.0, 4.0, -1.0]),
        }
        losses, metrics = compute_nitrobrew_loss_aggregate(None, self._distill_config(topk=4), model_output, self._data())
        assert metrics["distillation/overlap_ratio"] == pytest.approx((1 + 2 + 0 + 4) / 4 / 4)
        assert losses.shape == (1, 4)

    def test_no_overlap_counts_yields_empty_metrics(self):
        from verl_omni.trainer.distillation.losses import compute_nitrobrew_loss_aggregate

        model_output = {"distillation_losses": torch.arange(7, dtype=torch.float32)}
        losses, metrics = compute_nitrobrew_loss_aggregate(None, self._distill_config(), model_output, self._data())
        assert metrics == {}
