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
"""CPU correctness tests for the Megatron vocab-parallel nitrobrew kernels.

Covers everything that does not require a real ``torch.distributed`` group:

- single-shard (``tp_group=None``) forward/backward equivalence with the FSDP
  kernel (``nitrobrew_loss.py``) and a naive materialized-logits reference,
  including temperature, ``log_prob_min_clamp`` and an fp64 reference;
- padded-vocab masking (student shard wider than the teacher vocab: garbage
  columns must be excluded from every sum and get zero gradient);
- ``_shard_range`` boundary math, including a fully-padded rank;
- two-shard simulations of the TP merge math via the pure-list twins
  (``_merge_*_from_partials``), plus autograd.Function-level two-shard runs
  with a fake TP group whose collectives are served from precomputed partials;
- the sharded top-k overlap diagnostic (single-shard vs FSDP, and a two-shard
  record/replay of the all-gather merge);
- multi-teacher grouping (``_grouped_nitrobrew_kl_megatron``) vs the FSDP
  grouped path.

NOT covered here (needs megatron-core, absent on this box): the CP-split entry
points ``compute_nitrobrew_(multi_|multi_reverse_)kl_megatron``, which lazily
import ``verl.models.mcore.util`` (top-level megatron import). Run those on a
megatron-enabled environment.
"""

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pytest
import torch
import torch.nn.functional as F

import verl_omni.trainer.distillation.megatron_nitrobrew_loss as mnb
from verl_omni.trainer.distillation.megatron_nitrobrew_loss import (
    _VocabParallelNitrobrewKL,
    _VocabParallelNitrobrewReverseKL,
    _fwd_teacher_pass_local,
    _grouped_nitrobrew_kl_megatron,
    _merge_fwd_teacher_from_partials,
    _merge_lse_from_partials,
    _merge_rev_teacher_from_partials,
    _rev_teacher_pass_local,
    _shard_range,
    _student_lse_local,
    _topk_overlap_diag_vp,
)
from verl_omni.trainer.distillation.nitrobrew_loss import (
    _NitrobrewKL,
    _NitrobrewReverseKL,
    _chunked_kl_forward,
    _chunked_reverse_kl_forward,
    _grouped_nitrobrew_kl,
    _topk_overlap_diag,
)


# ---------------------------------------------------------------------------
# Naive references (materialized full logits)
# ---------------------------------------------------------------------------


def _naive_forward_kl(z, w, s, temperature=1.0, log_prob_min_clamp=None):
    zt = (z @ w.T).float() / temperature
    s = s.float() / temperature
    log_pt = F.log_softmax(zt, dim=-1)
    log_ps = F.log_softmax(s, dim=-1)
    if log_prob_min_clamp is not None:
        # kernel semantics: absolute floor on the scaled student log-prob,
        # max(log_ps, clamp) — NOT relative to the logsumexp.
        log_ps = torch.clamp(log_ps, min=log_prob_min_clamp)
    pt = log_pt.exp()
    return (pt * (log_pt - log_ps)).sum(dim=-1)


def _naive_reverse_kl(z, w, s, temperature=1.0):
    zt = (z @ w.T).float() / temperature
    s = s.float() / temperature
    log_pt = F.log_softmax(zt, dim=-1)
    log_ps = F.log_softmax(s, dim=-1)
    ps = log_ps.exp()
    return (ps * (log_ps - log_pt)).sum(dim=-1)


def _naive_topk_overlap(z, w, s, k, temperature=1.0):
    zt = (z @ w.T).float() / temperature
    t_ids = zt.topk(k, dim=1).indices
    s_ids = s.float().topk(k, dim=1).indices
    return (t_ids.unsqueeze(-1) == s_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)


def _make_inputs(seed=0, n=4, v=32, d=8):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    w = torch.randn(v, d, generator=g) * 0.1
    s = torch.randn(n, v, generator=g)
    return z, w, s


# ---------------------------------------------------------------------------
# Single-shard (tp_group=None): equivalence with FSDP kernel / naive
# ---------------------------------------------------------------------------


class TestSingleShardForwardKL:
    def test_forward_matches_fsdp_and_naive(self):
        z, w, s = _make_inputs()
        kl = _VocabParallelNitrobrewKL.apply(z, w, s, 4, 1.0, None)
        kl_fsdp, _, _ = _chunked_kl_forward(z, w, s, chunk_V=4)
        assert torch.allclose(kl, kl_fsdp, atol=1e-6, rtol=1e-5)
        assert torch.allclose(kl, _naive_forward_kl(z, w, s), atol=1e-5, rtol=1e-4)

    def test_forward_chunk_larger_than_vocab(self):
        z, w, s = _make_inputs(seed=20, v=48)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s, 4096, 1.0, None)
        assert torch.allclose(kl, _naive_forward_kl(z, w, s), atol=1e-5, rtol=1e-4)

    def test_forward_with_temperature_and_clamp(self):
        z, w, s = _make_inputs(seed=2, v=64)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s, 8, 0.7, -8.0)
        expected = _naive_forward_kl(z, w, s, temperature=0.7, log_prob_min_clamp=-8.0)
        assert torch.allclose(kl, expected, atol=1e-5, rtol=1e-4)

    def test_backward_matches_naive(self):
        z, w, s = _make_inputs(seed=3)
        s_shard = s.clone().requires_grad_(True)
        s_naive = s.clone().requires_grad_(True)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s_shard, 4, 1.0, None)
        kl_naive = _naive_forward_kl(z, w, s_naive)
        grad_out = torch.randn_like(kl)
        kl.backward(grad_out)
        kl_naive.backward(grad_out)
        assert torch.allclose(s_shard.grad, s_naive.grad, atol=1e-5, rtol=1e-4)

    def test_backward_with_temperature(self):
        z, w, s = _make_inputs(seed=21)
        s_shard = s.clone().requires_grad_(True)
        s_naive = s.clone().requires_grad_(True)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s_shard, 4, 0.7, None)
        kl_naive = _naive_forward_kl(z, w, s_naive, temperature=0.7)
        grad_out = torch.randn_like(kl)
        kl.backward(grad_out)
        kl_naive.backward(grad_out)
        assert torch.allclose(s_shard.grad, s_naive.grad, atol=1e-5, rtol=1e-4)

    def test_forward_matches_fp64_reference(self):
        z, w, s = _make_inputs(seed=22, n=8, v=96, d=16)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s, 16, 1.0, None)
        zt = (z.double() @ w.double().T)
        log_pt = F.log_softmax(zt, dim=-1)
        log_ps = F.log_softmax(s.double(), dim=-1)
        ref = (log_pt.exp() * (log_pt - log_ps)).sum(dim=-1).float()
        assert torch.allclose(kl, ref, atol=1e-4, rtol=1e-3)


class TestSingleShardReverseKL:
    def test_forward_matches_fsdp_and_naive(self):
        z, w, s = _make_inputs(seed=1)
        kl = _VocabParallelNitrobrewReverseKL.apply(z, w, s, 4, 1.0)
        kl_fsdp, _, _ = _chunked_reverse_kl_forward(z, w, s, chunk_V=4)
        assert torch.allclose(kl, kl_fsdp, atol=1e-6, rtol=1e-5)
        assert torch.allclose(kl, _naive_reverse_kl(z, w, s), atol=1e-5, rtol=1e-4)

    def test_forward_with_temperature(self):
        z, w, s = _make_inputs(seed=23, v=64)
        kl = _VocabParallelNitrobrewReverseKL.apply(z, w, s, 8, 0.7)
        assert torch.allclose(kl, _naive_reverse_kl(z, w, s, temperature=0.7), atol=1e-5, rtol=1e-4)

    def test_backward_matches_naive(self):
        z, w, s = _make_inputs(seed=4)
        s_shard = s.clone().requires_grad_(True)
        s_naive = s.clone().requires_grad_(True)
        kl = _VocabParallelNitrobrewReverseKL.apply(z, w, s_shard, 4, 1.0)
        kl_naive = _naive_reverse_kl(z, w, s_naive)
        grad_out = torch.randn_like(kl)
        kl.backward(grad_out)
        kl_naive.backward(grad_out)
        assert torch.allclose(s_shard.grad, s_naive.grad, atol=1e-5, rtol=1e-4)


class TestSingleShardDiag:
    def test_diag_matches_fsdp(self):
        z, w, s = _make_inputs(seed=12, n=6, v=48)
        z_f = z.float()
        rows, overlap = _topk_overlap_diag_vp(z_f, w, s, 0, w.shape[0], 4, 1024, None)
        rows_ref, overlap_ref = _topk_overlap_diag(z_f, w.float(), s, 4, 1024)
        assert torch.equal(rows, rows_ref)
        assert torch.equal(overlap, overlap_ref)
        assert torch.equal(overlap, _naive_topk_overlap(z, w, s, 4))

    def test_diag_stride_subsampling(self):
        z, w, s = _make_inputs(seed=14, n=8, v=32)
        rows, overlap = _topk_overlap_diag_vp(z.float(), w, s, 0, w.shape[0], 4, 4, None)
        assert torch.equal(rows, torch.tensor([0, 2, 4, 6]))
        assert torch.equal(overlap, _naive_topk_overlap(z, w, s, 4)[rows])

    def test_function_diag_roundtrip(self):
        z, w, s = _make_inputs(seed=15)
        s_req = s.clone().requires_grad_(True)
        kl, rows, overlap = _VocabParallelNitrobrewKL.apply(z, w, s_req, 4, 1.0, None, 4, 1024, None)
        assert torch.equal(overlap, _naive_topk_overlap(z, w, s, 4))
        assert not rows.requires_grad and not overlap.requires_grad
        kl_plain = _VocabParallelNitrobrewKL.apply(z, w, s, 4, 1.0, None)
        assert torch.equal(kl, kl_plain)  # diag must not perturb the KL
        kl.sum().backward()
        s_naive = s.clone().requires_grad_(True)
        _naive_forward_kl(z, w, s_naive).sum().backward()
        assert torch.allclose(s_req.grad, s_naive.grad, atol=1e-5, rtol=1e-4)

    def test_function_diag_reverse(self):
        z, w, s = _make_inputs(seed=17)
        s_req = s.clone().requires_grad_(True)
        res = _VocabParallelNitrobrewReverseKL.apply(z, w, s_req, 4, 1.0, 4, 1024, None)
        assert len(res) == 3
        assert torch.equal(res[2], _naive_topk_overlap(z, w, s, 4))
        res[0].sum().backward()
        s_naive = s.clone().requires_grad_(True)
        _naive_reverse_kl(z, w, s_naive).sum().backward()
        assert torch.allclose(s_req.grad, s_naive.grad, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Padded-vocab masking: shard wider than the teacher vocabulary
# ---------------------------------------------------------------------------


class TestPaddedVocabMasking:
    def test_garbage_padding_excluded(self):
        """Vp > V with garbage in the padded columns must not change the KL."""
        z, w, s = _make_inputs(seed=30, n=4, v=30, d=8)
        pad = torch.full((s.shape[0], 2), 1.0e6)  # Vp = 32 > V = 30
        s_padded = torch.cat([s, pad], dim=1)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s_padded, 4, 1.0, None)
        assert torch.allclose(kl, _naive_forward_kl(z, w, s), atol=1e-5, rtol=1e-4)

    def test_garbage_padding_excluded_reverse(self):
        z, w, s = _make_inputs(seed=31, n=4, v=30, d=8)
        pad = torch.full((s.shape[0], 2), 1.0e6)
        s_padded = torch.cat([s, pad], dim=1)
        kl = _VocabParallelNitrobrewReverseKL.apply(z, w, s_padded, 4, 1.0)
        assert torch.allclose(kl, _naive_reverse_kl(z, w, s), atol=1e-5, rtol=1e-4)

    def test_padded_columns_get_zero_grad(self):
        z, w, s = _make_inputs(seed=32, n=4, v=30, d=8)
        pad = torch.full((s.shape[0], 2), 1.0e6)
        s_padded = torch.cat([s, pad], dim=1).requires_grad_(True)
        kl = _VocabParallelNitrobrewKL.apply(z, w, s_padded, 4, 1.0, None)
        kl.sum().backward()
        assert torch.equal(s_padded.grad[:, 30:], torch.zeros(4, 2))
        s_ref = s.clone().requires_grad_(True)
        _naive_forward_kl(z, w, s_ref).sum().backward()
        assert torch.allclose(s_padded.grad[:, :30], s_ref.grad, atol=1e-5, rtol=1e-4)


class TestShardRange:
    @pytest.mark.parametrize(
        "rank,vp,v_true,expected",
        [
            (0, 259, 517, (0, 259)),  # full shard
            (1, 259, 517, (259, 258)),  # one padded column
            (2, 259, 517, (518, 0)),  # fully padded rank
            (0, 128, 512, (0, 128)),
            (3, 128, 512, (384, 128)),
            (3, 128, 511, (384, 127)),  # v_true not divisible by tp
        ],
    )
    def test_boundaries(self, monkeypatch, rank, vp, v_true, expected):
        monkeypatch.setattr(mnb, "_tp_rank", lambda group: rank)
        s = torch.zeros(2, vp)
        w = torch.zeros(v_true, 4)
        assert _shard_range(s, w, object()) == expected


# ---------------------------------------------------------------------------
# Two-shard simulation of the TP merge math (V=517, Vp=259, one pad column)
# ---------------------------------------------------------------------------

_V2, _VP2 = 517, 259


def _make_two_shards(seed=40, n=4, d=8, temperature=1.0):
    """(z, w, s_full, shards): shards have garbage in padded columns."""
    z, w, s_full = _make_inputs(seed=seed, n=n, v=_V2, d=d)
    shards = []
    for r in range(2):
        v_start = r * _VP2
        v_valid = max(0, min(_VP2, _V2 - v_start))
        cols = [s_full[:, v_start : v_start + v_valid]]
        if v_valid < _VP2:
            cols.append(torch.full((n, _VP2 - v_valid), 1.0e6))
        shards.append(torch.cat(cols, dim=1))
    return z, w, s_full, shards


def _two_shard_partials(z, w, shards, temperature, clamp, chunk_V):
    """Per-rank local accumulators, computed exactly like the kernel does."""
    inv_T = 1.0 / temperature if temperature != 0.0 else 1.0
    z_f = z.float() * inv_T if inv_T != 1.0 else z.float()
    lse_parts, fwd_parts = [], []
    for r in range(2):
        v_start, v_valid = r * _VP2, max(0, min(_VP2, _V2 - r * _VP2))
        lse_parts.append(_student_lse_local(shards[r], v_valid, inv_T, chunk_V))
    s_lse = _merge_lse_from_partials(lse_parts)
    if clamp is not None:
        s_min = (s_lse + clamp).unsqueeze(1)
    else:
        s_min = torch.full((1, 1), float("-inf"))
    for r in range(2):
        v_start, v_valid = r * _VP2, max(0, min(_VP2, _V2 - r * _VP2))
        fwd_parts.append(
            _fwd_teacher_pass_local(z_f, w[v_start : v_start + v_valid], shards[r], s_min, inv_T, chunk_V)
        )
    return s_lse, lse_parts, fwd_parts


class TestTwoShardMergeMath:
    def test_merge_lse_matches_unsharded(self):
        z, w, s_full, shards = _make_two_shards()
        s_lse, _, _ = _two_shard_partials(z, w, shards, 1.0, None, 100)
        ref = s_full.float().logsumexp(dim=-1)
        assert torch.allclose(s_lse, ref, atol=1e-6, rtol=1e-5)

    def test_forward_kl_matches_fsdp(self):
        z, w, s_full, shards = _make_two_shards(seed=41)
        s_lse, _, fwd_parts = _two_shard_partials(z, w, shards, 0.7, -8.0, 100)
        m, st, tt, ut = _merge_fwd_teacher_from_partials(fwd_parts)
        t_lse = m + st.log()
        kl = tt / st - t_lse - ut / st + s_lse
        kl_ref, _, _ = _chunked_kl_forward(z, w, s_full, chunk_V=100, temperature=0.7, log_prob_min_clamp=-8.0)
        assert torch.allclose(kl, kl_ref, atol=1e-5, rtol=1e-4)

    def test_reverse_kl_matches_fsdp(self):
        z, w, s_full, shards = _make_two_shards(seed=42)
        inv_T = 1.0 / 0.7
        z_f = z.float() * inv_T
        lse_parts = [
            _student_lse_local(shards[r], max(0, min(_VP2, _V2 - r * _VP2)), inv_T, 100) for r in range(2)
        ]
        s_lse = _merge_lse_from_partials(lse_parts)
        rev_parts = []
        for r in range(2):
            v_start, v_valid = r * _VP2, max(0, min(_VP2, _V2 - r * _VP2))
            rev_parts.append(
                _rev_teacher_pass_local(z_f, w[v_start : v_start + v_valid], shards[r], s_lse, inv_T, 100)
            )
        m, st, ut, et = _merge_rev_teacher_from_partials(rev_parts)
        t_lse = m + st.log()
        kl = et - ut - s_lse + t_lse
        kl_ref, _, _ = _chunked_reverse_kl_forward(z, w, s_full, chunk_V=100, temperature=0.7)
        assert torch.allclose(kl, kl_ref, atol=1e-5, rtol=1e-4)


class _FakeTPGroup:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank


class TestTwoShardFunctionLevel:
    """Full autograd.Function per rank; collectives faked from precomputed partials."""

    def _patch_group(self, monkeypatch, rank):
        monkeypatch.setattr(mnb, "_tp_world_size", lambda g: g.world_size)
        monkeypatch.setattr(mnb, "_tp_rank", lambda g: g.rank)
        return _FakeTPGroup(2, rank)

    def test_forward_kl_two_shards(self, monkeypatch):
        z, w, s_full, shards = _make_two_shards(seed=43)
        temperature, clamp = 0.7, -8.0
        s_lse, lse_parts, fwd_parts = _two_shard_partials(z, w, shards, temperature, clamp, 100)
        monkeypatch.setattr(mnb, "_merge_lse", lambda ms, ss, group: _merge_lse_from_partials(lse_parts))
        monkeypatch.setattr(
            mnb,
            "_merge_fwd_teacher",
            lambda mt, st, tt, ut, group: _merge_fwd_teacher_from_partials(fwd_parts),
        )

        s_ref = s_full.clone().requires_grad_(True)
        kl_ref = _NitrobrewKL.apply(z, w, s_ref, 100, temperature, clamp)
        grad_out = torch.randn_like(kl_ref)
        kl_ref.backward(grad_out)

        for r in range(2):
            group = self._patch_group(monkeypatch, r)
            s_r = shards[r].clone().requires_grad_(True)
            kl_r = _VocabParallelNitrobrewKL.apply(z, w, s_r, 100, temperature, clamp, None, 1024, group)
            assert torch.allclose(kl_r, kl_ref, atol=1e-5, rtol=1e-4)
            kl_r.backward(grad_out)
            v_start, v_valid = r * _VP2, max(0, min(_VP2, _V2 - r * _VP2))
            assert torch.allclose(s_r.grad[:, :v_valid], s_ref.grad[:, v_start : v_start + v_valid], atol=1e-5, rtol=1e-4)
            if v_valid < _VP2:  # padded columns: zero grad despite garbage input
                assert torch.equal(s_r.grad[:, v_valid:], torch.zeros_like(s_r.grad[:, v_valid:]))

    def test_reverse_kl_two_shards(self, monkeypatch):
        z, w, s_full, shards = _make_two_shards(seed=44)
        temperature = 0.7
        inv_T = 1.0 / temperature
        z_f = z.float() * inv_T
        lse_parts = [
            _student_lse_local(shards[r], max(0, min(_VP2, _V2 - r * _VP2)), inv_T, 100) for r in range(2)
        ]
        s_lse = _merge_lse_from_partials(lse_parts)
        rev_parts = []
        for r in range(2):
            v_start, v_valid = r * _VP2, max(0, min(_VP2, _V2 - r * _VP2))
            rev_parts.append(
                _rev_teacher_pass_local(z_f, w[v_start : v_start + v_valid], shards[r], s_lse, inv_T, 100)
            )
        monkeypatch.setattr(mnb, "_merge_lse", lambda ms, ss, group: _merge_lse_from_partials(lse_parts))
        monkeypatch.setattr(
            mnb,
            "_merge_rev_teacher",
            lambda mt, st, ut, et, group: _merge_rev_teacher_from_partials(rev_parts),
        )

        s_ref = s_full.clone().requires_grad_(True)
        kl_ref = _NitrobrewReverseKL.apply(z, w, s_ref, 100, temperature)
        grad_out = torch.randn_like(kl_ref)
        kl_ref.backward(grad_out)

        for r in range(2):
            group = self._patch_group(monkeypatch, r)
            s_r = shards[r].clone().requires_grad_(True)
            kl_r = _VocabParallelNitrobrewReverseKL.apply(z, w, s_r, 100, temperature, None, 1024, group)
            assert torch.allclose(kl_r, kl_ref, atol=1e-5, rtol=1e-4)
            kl_r.backward(grad_out)
            v_start, v_valid = r * _VP2, max(0, min(_VP2, _V2 - r * _VP2))
            assert torch.allclose(s_r.grad[:, :v_valid], s_ref.grad[:, v_start : v_start + v_valid], atol=1e-5, rtol=1e-4)


class _GatherMailbox:
    """Serves ``_all_gather_last_dim`` for a simulated 2-rank group.

    Pass 1 (rank 0, record): stores the tensor, returns it unchanged (result
    discarded). Pass 2 (rank 1, record): stores and returns [r0, r1]. Pass 3
    (rank 0, replay): returns [r0_new, r1_stored]. Gather arguments are
    computed before any collective inside the diagnostic, so record/replay is
    exact.
    """

    def __init__(self):
        self.recorded = {}
        self.call_idx = 0
        self.rank = 0
        self.replay = False

    def reset(self, rank, replay):
        self.rank, self.replay, self.call_idx = rank, replay, 0

    def all_gather_last_dim(self, t, group):
        idx = self.call_idx
        self.call_idx += 1
        if not self.replay:
            self.recorded.setdefault(idx, {})[self.rank] = t
            if self.rank == 1:
                return torch.cat([self.recorded[idx][0], t], dim=-1)
            return t
        return torch.cat([t, self.recorded[idx][1]], dim=-1)


class TestTwoShardDiag:
    def test_diag_matches_fsdp(self, monkeypatch):
        v_true, vp = 48, 25  # rank 1 has 2 padded columns
        z, w, s_full = _make_inputs(seed=50, n=6, v=v_true, d=8)
        shards = []
        for r in range(2):
            v_start, v_valid = r * vp, max(0, min(vp, v_true - r * vp))
            cols = [s_full[:, v_start : v_start + v_valid]]
            if v_valid < vp:
                cols.append(torch.full((6, vp - v_valid), 1.0e6))
            shards.append(torch.cat(cols, dim=1))

        mailbox = _GatherMailbox()
        monkeypatch.setattr(mnb, "_tp_world_size", lambda g: 2)
        monkeypatch.setattr(mnb, "_all_gather_last_dim", mailbox.all_gather_last_dim)

        z_f = z.float()
        results = {}
        for r, replay in [(0, False), (1, False), (0, True)]:
            mailbox.reset(r, replay)
            v_start, v_valid = r * vp, max(0, min(vp, v_true - r * vp))
            rows, overlap = _topk_overlap_diag_vp(
                z_f, w[v_start : v_start + v_valid], shards[r], v_start, v_true, 4, 1024, object()
            )
            if replay or r == 1:
                results[r] = (rows, overlap)

        rows_ref, overlap_ref = _topk_overlap_diag(z_f, w.float(), s_full, 4, 1024)
        for r in range(2):
            assert torch.equal(results[r][0], rows_ref)
            assert torch.equal(results[r][1], overlap_ref)


# ---------------------------------------------------------------------------
# Multi-teacher grouping vs the FSDP grouped path
# ---------------------------------------------------------------------------


class _LossCfg:
    def __init__(self, temperature=1.0, clamp=None, topk=None):
        self.kd_temperature = temperature
        self.log_prob_min_clamp = clamp
        self.topk = topk


class _Cfg:
    def __init__(self, **kw):
        self.distillation_loss = _LossCfg(**kw)


class TestGroupedMultiTeacher:
    def test_forward_matches_fsdp_grouped(self):
        z, w, s = _make_inputs(seed=7, n=4)
        w0, w1 = w, w * 0.9
        key_ids = torch.tensor([0, 0, 1, 1])
        unembeds = {0: w0, 1: w1}
        cfg = _Cfg()
        out = _grouped_nitrobrew_kl_megatron(
            s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds=unembeds,
            config=cfg, reverse=False, out_shape=(1, 4), tp_group=None,
        )
        ref = _grouped_nitrobrew_kl(s[None], z[None], key_ids, unembeds, cfg, reverse=False)
        assert torch.allclose(out["distillation_losses"], ref["distillation_losses"], atol=1e-6, rtol=1e-5)
        assert not torch.isnan(out["distillation_losses"]).any()

    def test_reverse_matches_fsdp_grouped(self):
        z, w, s = _make_inputs(seed=8, n=4)
        w0, w1 = w, w * 1.1
        key_ids = torch.tensor([0, 1, 0, 1])
        unembeds = {0: w0, 1: w1}
        cfg = _Cfg()
        out = _grouped_nitrobrew_kl_megatron(
            s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds=unembeds,
            config=cfg, reverse=True, out_shape=(1, 4), tp_group=None,
        )
        ref = _grouped_nitrobrew_kl(s[None], z[None], key_ids, unembeds, cfg, reverse=True)
        assert torch.allclose(out["distillation_losses"], ref["distillation_losses"], atol=1e-6, rtol=1e-5)

    def test_out_shape_and_single_teacher(self):
        z, w, s = _make_inputs(seed=9, n=4)
        key_ids = torch.zeros(4, dtype=torch.long)
        out = _grouped_nitrobrew_kl_megatron(
            s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds={0: w},
            config=_Cfg(), reverse=False, out_shape=(2, 2), tp_group=None,
        )
        assert out["distillation_losses"].shape == (2, 2)
        expected = _naive_forward_kl(z, w, s)
        assert torch.allclose(out["distillation_losses"].flatten(), expected, atol=1e-5, rtol=1e-4)

    def test_temperature_and_clamp_applied(self):
        z, w, s = _make_inputs(seed=11, n=4, v=64)
        key_ids = torch.zeros(4, dtype=torch.long)
        cfg = _Cfg(temperature=0.7, clamp=-8.0)
        out = _grouped_nitrobrew_kl_megatron(
            s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds={0: w},
            config=cfg, reverse=False, out_shape=(1, 4), tp_group=None,
        )
        expected = _naive_forward_kl(z, w, s, temperature=0.7, log_prob_min_clamp=-8.0)
        assert torch.allclose(out["distillation_losses"][0], expected, atol=1e-5, rtol=1e-4)

    def test_overlap_counts_match_fsdp(self):
        z, w, s = _make_inputs(seed=16, n=4, v=48)
        w0, w1 = w, w * 0.9
        key_ids = torch.tensor([0, 0, 1, 1])
        cfg = _Cfg(topk=4)
        out = _grouped_nitrobrew_kl_megatron(
            s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds={0: w0, 1: w1},
            config=cfg, reverse=False, out_shape=(1, 4), tp_group=None,
        )
        ref = _grouped_nitrobrew_kl(s[None], z[None], key_ids, {0: w0, 1: w1}, cfg, reverse=False)
        assert torch.equal(out["overlap_counts"], ref["overlap_counts"])
        expected = torch.cat(
            [_naive_topk_overlap(z[:2], w0, s[:2], 4), _naive_topk_overlap(z[2:], w1, s[2:], 4)]
        )
        assert torch.equal(out["overlap_counts"][0], expected.float())

    def test_no_diag_omits_overlap_counts(self):
        z, w, s = _make_inputs(seed=18, n=4)
        key_ids = torch.zeros(4, dtype=torch.long)
        out = _grouped_nitrobrew_kl_megatron(
            s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds={0: w},
            config=_Cfg(), reverse=False, out_shape=(1, 4), tp_group=None,
        )
        assert "overlap_counts" not in out

    def test_missing_teacher_raises(self):
        z, w, s = _make_inputs(seed=10, n=4)
        key_ids = torch.tensor([0, 0, 5, 5])
        with pytest.raises(RuntimeError, match="no registered teacher unembedding"):
            _grouped_nitrobrew_kl_megatron(
                s_flat=s, z_flat=z, key_ids_flat=key_ids, teacher_unembeds={0: w},
                config=_Cfg(), reverse=False, out_shape=(1, 4), tp_group=None,
            )
