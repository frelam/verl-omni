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

"""Megatron vocab-parallel nitrobrew: full-vocabulary KL from hidden states.

This is the Megatron counterpart of ``nitrobrew_loss.py`` (FSDP). Under
Megatron the student's ``lm_head`` is a ``ColumnParallelLinear``, so the logits
processor receives a *vocab shard* ``[N, V/tp]`` instead of the full ``[N, V]``
tensor. This module computes the same exact (untruncated) KL without ever
gathering the full logits:

- **Teacher unembedding is sharded on device, not on the wire.** Ranks still
  receive the full CPU ``[V, D]`` matrix via ``set_teacher_unembeds`` (same
  contract as FSDP), but the kernel infers the local vocab range from the
  student logits shard itself — ``v_start = tp_rank * Vp`` with
  ``Vp = student_logits.shape[-1]`` — and moves only the local slice
  ``[V/tp, D]`` to the device. The student's own lm_head layout is the single
  source of truth, so Megatron's ``make_vocab_size_divisible_by`` padding logic
  is never replicated here. GPU memory for the teacher matrix drops by 1/tp
  and the ``h @ W.T`` reconstruction FLOPs drop by 1/tp.

- **Padded vocab is masked.** When ``tp * Vp > V_teacher``, columns beyond the
  teacher's true vocabulary are excluded from both the student logsumexp and
  the teacher sums, so numerics match the FSDP full-vocab kernel exactly.

- **Communication: 4 small all-reduces per forward, zero in backward.** Both
  the student logsumexp and the teacher online-softmax accumulators are
  order-independent partitions of the vocab sum: each rank accumulates its
  shard locally (chunked, ``O(N * chunk)`` peak memory), then merges across
  the TP group with one ``MAX`` and one stacked ``SUM`` all-reduce of ``[N]``
  tensors. The backward is dense per vocab entry — ``dKL/dz_v = p_S - p_T`` —
  so every rank differentiates its own shard with no collectives at all.

- **CP is handled by re-splitting the teacher payloads.** Mirroring verl's
  top-k Megatron loss (``verl/trainer/distillation/megatron/losses.py``), the
  nested ``teacher_hidden_states`` (and per-sequence ``teacher_key_ids``) are
  sharded with ``preprocess_thd_engine`` / ``preprocess_bshd_engine`` so they
  align row-for-row with the CP-sharded student logits. The engine gathers the
  returned per-token losses back across CP afterwards, and the final
  aggregation (``compute_nitrobrew_loss_aggregate``) is engine-agnostic.

Multi-teacher note: tokens are grouped per teacher and each group runs the
kernel with that teacher's unembedding. TP ranks hold identical batch data
(Megatron TP is data-replicated), so group membership — and therefore the
collective sequence inside the kernel — is identical across the TP group. CP
ranks hold different tokens but the collectives are TP-group-local.
"""

import torch
from verl.workers.config import DistillationConfig

_CHUNK_V: int = 1024
_DIAG_MAX_ROWS: int = 1024


# ---------------------------------------------------------------------------
# TP communication helpers (lazy megatron import; world-size-1 short-circuits)
# ---------------------------------------------------------------------------


def get_default_tp_group():
    """The current tensor-model-parallel group, or None when unavailable.

    None means "single shard covering the whole vocabulary" — the math below
    degenerates to the unsharded kernel, which is also how CPU tests run it.
    """
    try:
        from megatron.core import parallel_state as mpu

        if mpu.is_initialized():
            return mpu.get_tensor_model_parallel_group()
    except ImportError:
        pass
    return None


def _tp_world_size(group) -> int:
    return 1 if group is None else torch.distributed.get_world_size(group)


def _tp_rank(group) -> int:
    return 0 if group is None else torch.distributed.get_rank(group)


def _all_reduce_max(t: torch.Tensor, group) -> torch.Tensor:
    if _tp_world_size(group) == 1:
        return t
    t = t.contiguous().clone()
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
    return t


def _all_reduce_sum(t: torch.Tensor, group) -> torch.Tensor:
    if _tp_world_size(group) == 1:
        return t
    t = t.contiguous().clone()
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=group)
    return t


def _all_gather_last_dim(t: torch.Tensor, group) -> torch.Tensor:
    if _tp_world_size(group) == 1:
        return t
    world = _tp_world_size(group)
    outs = [torch.empty_like(t) for _ in range(world)]
    torch.distributed.all_gather(outs, t.contiguous(), group=group)
    return torch.cat(outs, dim=-1)


def _shard_range(student_logits: torch.Tensor, W: torch.Tensor, group) -> tuple[int, int]:
    """(v_start, v_valid): this rank's global vocab offset and valid width.

    The shard layout is inferred from the student logits shard width ``Vp``;
    ``v_valid`` clamps to the teacher's true vocabulary so Megatron's padded
    vocab rows (``tp * Vp > V``) are excluded from every sum.
    """
    vp = student_logits.shape[-1]
    v_true = W.shape[0]
    v_start = _tp_rank(group) * vp
    v_valid = max(0, min(vp, v_true - v_start))
    return v_start, v_valid


# ---------------------------------------------------------------------------
# Local (per-shard) online-softmax passes — zero communication
# ---------------------------------------------------------------------------


def _student_lse_local(vp_s: torch.Tensor, v_valid: int, inv_T: float, chunk_V: int):
    """Online (max, sum-exp) over the local valid student columns. Zero comm."""
    N = vp_s.shape[0]
    device = vp_s.device
    ms = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    ss = torch.zeros(N, dtype=torch.float32, device=device)
    for c in range(0, v_valid, chunk_V):
        s = vp_s[:, c : min(c + chunk_V, v_valid)].float()
        if inv_T != 1.0:
            s = s * inv_T
        m_new = torch.maximum(ms, s.max(dim=1).values)
        ss = ss * (ms - m_new).exp() + (s - m_new.unsqueeze(1)).exp().sum(dim=1)
        ms = m_new
    return ms, ss


def _merge_lse(ms: torch.Tensor, ss: torch.Tensor, group) -> torch.Tensor:
    """Merge per-shard online (max, sum-exp) into the global logsumexp."""
    m = _all_reduce_max(ms, group)
    s = _all_reduce_sum(ss * (ms - m).exp(), group)
    return m + s.log()


def _merge_lse_from_partials(partials: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Pure-list twin of ``_merge_lse`` — same formula, used by sharded-math tests."""
    m = torch.stack([p[0] for p in partials]).max(dim=0).values
    s = torch.stack([p[1] * (p[0] - m).exp() for p in partials]).sum(dim=0)
    return m + s.log()


def _move_shard(W: torch.Tensor, v_start: int, v_valid: int, device) -> torch.Tensor:
    """Move this rank's teacher unembedding slice to the device (bf16)."""
    return W[v_start : v_start + v_valid].to(device=device)


def _fwd_teacher_pass_local(
    z_f: torch.Tensor,  # (N, D_t) float32, temperature-scaled
    W_shard: torch.Tensor,  # (v_valid, D_t) on device
    vp_s: torch.Tensor,  # (N, Vp) raw student logits shard
    s_min: torch.Tensor,  # (N, 1) floor for student logits (or -inf)
    inv_T: float,
    chunk_V: int,
):
    """Local online accumulators for forward KL. Mirrors `_fwd_chunk_update`.

    Returns (mt, st, tt, ut): running teacher max, sum exp(zt-mt),
    sum exp(zt-mt)*zt, sum exp(zt-mt)*zs_clamped — all local to this shard.
    """
    N = z_f.shape[0]
    device = z_f.device
    v_valid = W_shard.shape[0]
    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    tt = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)
    for c in range(0, v_valid, chunk_V):
        c1 = min(c + chunk_V, v_valid)
        zt = z_f @ W_shard[c:c1].float().T
        s_c = vp_s[:, c:c1].float()
        if inv_T != 1.0:
            s_c = s_c * inv_T
        m_new = torch.maximum(mt, zt.max(dim=1).values)
        alpha = (mt - m_new).exp()
        pt = (zt - m_new.unsqueeze(1)).exp()
        st = st * alpha + pt.sum(dim=1)
        tt = tt * alpha + (pt * zt).sum(dim=1)
        ut = ut * alpha + (pt * torch.maximum(s_c, s_min)).sum(dim=1)
        mt = m_new
    return mt, st, tt, ut


def _merge_fwd_teacher(mt, st, tt, ut, group) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TP-merge forward-KL accumulators. One MAX + one stacked SUM all-reduce."""
    m = _all_reduce_max(mt, group)
    packed = torch.stack([st, tt, ut]) * (mt - m).exp()
    st_g, tt_g, ut_g = _all_reduce_sum(packed, group).unbind(0)
    return m, st_g, tt_g, ut_g


def _merge_fwd_teacher_from_partials(
    partials: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-list twin of ``_merge_fwd_teacher`` for sharded-math tests."""
    m = torch.stack([p[0] for p in partials]).max(dim=0).values
    packed = torch.stack([torch.stack([p[1], p[2], p[3]]) * (p[0] - m).exp() for p in partials]).sum(dim=0)
    st_g, tt_g, ut_g = packed.unbind(0)
    return m, st_g, tt_g, ut_g


def _rev_teacher_pass_local(
    z_f: torch.Tensor,  # (N, D_t) float32, temperature-scaled
    W_shard: torch.Tensor,  # (v_valid, D_t) on device
    vp_s: torch.Tensor,  # (N, Vp) raw student logits shard
    s_lse: torch.Tensor,  # (N,) global student logsumexp (already merged)
    inv_T: float,
    chunk_V: int,
):
    """Local online accumulators for reverse KL. Mirrors `_rev_fwd_chunk`.

    Returns (mt, st, ut, et): teacher max, sum exp(zt-mt), sum p_S*zt,
    sum p_S*zs — all local to this shard. ut/et use the global s_lse, so only
    st needs the max rescale at merge time.
    """
    N = z_f.shape[0]
    device = z_f.device
    v_valid = W_shard.shape[0]
    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)
    et = torch.zeros(N, dtype=torch.float32, device=device)
    for c in range(0, v_valid, chunk_V):
        c1 = min(c + chunk_V, v_valid)
        zt = z_f @ W_shard[c:c1].float().T
        s_c = vp_s[:, c:c1].float()
        if inv_T != 1.0:
            s_c = s_c * inv_T
        m_new = torch.maximum(mt, zt.max(dim=1).values)
        alpha = (mt - m_new).exp()
        st = st * alpha + (zt - m_new.unsqueeze(1)).exp().sum(dim=1)
        ps = (s_c - s_lse.unsqueeze(1)).exp()
        ut = ut + (ps * zt).sum(dim=1)
        et = et + (ps * s_c).sum(dim=1)
        mt = m_new
    return mt, st, ut, et


def _merge_rev_teacher(mt, st, ut, et, group) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TP-merge reverse-KL accumulators. One MAX + one stacked SUM all-reduce."""
    m = _all_reduce_max(mt, group)
    packed = torch.stack([st * (mt - m).exp(), ut, et])
    st_g, ut_g, et_g = _all_reduce_sum(packed, group).unbind(0)
    return m, st_g, ut_g, et_g


def _merge_rev_teacher_from_partials(
    partials: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-list twin of ``_merge_rev_teacher`` for sharded-math tests."""
    m = torch.stack([p[0] for p in partials]).max(dim=0).values
    packed = torch.stack([torch.stack([p[1] * (p[0] - m).exp(), p[2], p[3]]) for p in partials]).sum(dim=0)
    st_g, ut_g, et_g = packed.unbind(0)
    return m, st_g, ut_g, et_g


# ---------------------------------------------------------------------------
# Sharded top-k overlap diagnostic (mirrors nitrobrew_loss._topk_overlap_diag)
# ---------------------------------------------------------------------------


def _topk_overlap_diag_vp(
    z_f: torch.Tensor,  # (N, D_t) float32, temperature-scaled
    W_shard: torch.Tensor,  # (v_valid, D_t) on device
    vp_s: torch.Tensor,  # (N, Vp) raw student logits shard
    v_start: int,
    v_true: int,
    diag_topk: int,
    diag_max_rows: int,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher/student top-k overlap on stride-subsampled rows, vocab-sharded.

    Each rank keeps a running top-k over its own columns (global ids), then the
    candidates are merged across the TP group with two all-gathers per side.
    Returns (rows, overlap_count) like the FSDP diagnostic.
    """
    N = z_f.shape[0]
    device = z_f.device
    k = min(diag_topk, v_true)
    stride = max(1, (N + diag_max_rows - 1) // diag_max_rows)
    rows = torch.arange(0, N, stride, device=device)
    R = rows.numel()

    z_r = z_f[rows]
    v_valid = W_shard.shape[0]

    # Student top-k over the local valid columns (ids -> global).
    k_loc = min(k, v_valid)
    if k_loc > 0:
        s_vals, s_ids = vp_s[rows, :v_valid].topk(k_loc, dim=1)
        s_vals = s_vals.float()
        s_ids = s_ids + v_start
    else:  # this shard owns only padded rows
        s_vals = torch.full((R, 0), float("-inf"), dtype=torch.float32, device=device)
        s_ids = torch.zeros((R, 0), dtype=torch.long, device=device)
    # Pad to a fixed width so all-gather shapes agree across ranks.
    if k_loc < k:
        pad_vals = torch.full((R, k - k_loc), float("-inf"), dtype=torch.float32, device=device)
        pad_ids = torch.zeros((R, k - k_loc), dtype=torch.long, device=device)
        s_vals = torch.cat([s_vals, pad_vals], dim=1)
        s_ids = torch.cat([s_ids, pad_ids], dim=1)

    # Teacher running top-k over local chunks (ids -> global).
    run_vals = torch.full((R, k), float("-inf"), dtype=torch.float32, device=device)
    run_ids = torch.zeros((R, k), dtype=torch.long, device=device)
    for c in range(0, v_valid, _CHUNK_V):
        c1 = min(c + _CHUNK_V, v_valid)
        zt = z_r @ W_shard[c:c1].float().T
        vals, ids = zt.topk(min(k, c1 - c), dim=1)
        run_vals, sel = torch.cat([run_vals, vals], dim=1).topk(k, dim=1)
        run_ids = torch.cat([run_ids, ids + v_start + c], dim=1).gather(1, sel)

    if _tp_world_size(group) > 1:
        # Merge candidates across the TP group: gather + re-topk.
        t_vals = _all_gather_last_dim(run_vals, group)
        t_ids = _all_gather_last_dim(run_ids, group)
        _, sel = t_vals.topk(k, dim=1)
        run_ids = t_ids.gather(1, sel)

        s_vals_g = _all_gather_last_dim(s_vals, group)
        s_ids_g = _all_gather_last_dim(s_ids, group)
        _, sel = s_vals_g.topk(k, dim=1)
        s_topk_ids = s_ids_g.gather(1, sel)
    else:
        s_topk_ids = s_ids

    overlap_count = (run_ids.unsqueeze(-1) == s_topk_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return rows, overlap_count


# ---------------------------------------------------------------------------
# autograd.Functions — forward & backward over the vocab shard
# ---------------------------------------------------------------------------


class _VocabParallelNitrobrewKL(torch.autograd.Function):
    """KL(p_T || p_S) on a vocab shard; TP-merged forward, comm-free backward."""

    @staticmethod
    def forward(
        ctx,
        z,  # (N, D_t) teacher hidden states (constant data)
        W,  # (V_true, D_t) full teacher unembedding (CPU or device)
        vp_s,  # (N, Vp) local student logits shard
        chunk_V,
        temperature=1.0,
        log_prob_min_clamp=None,
        diag_topk=None,
        diag_max_rows=_DIAG_MAX_ROWS,
        tp_group=None,
    ):
        device = vp_s.device
        v_start, v_valid = _shard_range(vp_s, W, tp_group)
        W_shard = _move_shard(W, v_start, v_valid, device)

        z_f = z.to(device=device, dtype=torch.float32)
        inv_T = 1.0 / temperature if temperature != 0.0 else 1.0
        if inv_T != 1.0:
            z_f = z_f * inv_T

        ms, ss = _student_lse_local(vp_s, v_valid, inv_T, chunk_V)
        s_lse = _merge_lse(ms, ss, tp_group)

        if log_prob_min_clamp is not None:
            s_min = (s_lse + log_prob_min_clamp).unsqueeze(1)
        else:
            s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)

        mt, st, tt, ut = _fwd_teacher_pass_local(z_f, W_shard, vp_s, s_min, inv_T, chunk_V)
        m, st_g, tt_g, ut_g = _merge_fwd_teacher(mt, st, tt, ut, tp_group)
        t_lse = m + st_g.log()
        kl = tt_g / st_g - t_lse - ut_g / st_g + s_lse

        ctx.save_for_backward(z_f, W_shard, vp_s, t_lse, s_lse)
        ctx.chunk_V = chunk_V
        ctx.inv_T = inv_T
        if diag_topk:
            diag_rows, diag_overlap = _topk_overlap_diag_vp(
                z_f, W_shard, vp_s, v_start, W.shape[0], diag_topk, diag_max_rows, tp_group
            )
            ctx.mark_non_differentiable(diag_rows, diag_overlap)
            return kl, diag_rows, diag_overlap
        return kl

    @staticmethod
    def backward(ctx, grad_output, *_diag_grads):
        z_f, W_shard, vp_s, t_lse, s_lse = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        inv_T = ctx.inv_T
        N, Vp = vp_s.shape
        v_valid = W_shard.shape[0]
        device = vp_s.device
        grad = grad_output.float()

        grad_s = torch.zeros(N, Vp, dtype=torch.float32, device=device)
        for c in range(0, v_valid, chunk_V):
            c1 = min(c + chunk_V, v_valid)
            zt = z_f @ W_shard[c:c1].float().T
            p_T = (zt - t_lse.unsqueeze(1)).exp()
            s_c = vp_s[:, c:c1].float()
            if inv_T != 1.0:
                s_c = s_c * inv_T
            p_S = (s_c - s_lse.unsqueeze(1)).exp()
            grad_s[:, c:c1] = (p_S - p_T) * grad.unsqueeze(1)

        if inv_T != 1.0:
            grad_s = grad_s * inv_T

        # Padded vocab columns (v_valid..Vp) keep zero grad; z and W are data.
        return None, None, grad_s.to(vp_s.dtype), None, None, None, None, None, None


class _VocabParallelNitrobrewReverseKL(torch.autograd.Function):
    """KL(p_S || p_T) on a vocab shard; TP-merged forward, comm-free backward."""

    @staticmethod
    def forward(
        ctx,
        z,
        W,
        vp_s,
        chunk_V,
        temperature=1.0,
        diag_topk=None,
        diag_max_rows=_DIAG_MAX_ROWS,
        tp_group=None,
    ):
        device = vp_s.device
        v_start, v_valid = _shard_range(vp_s, W, tp_group)
        W_shard = _move_shard(W, v_start, v_valid, device)

        z_f = z.to(device=device, dtype=torch.float32)
        inv_T = 1.0 / temperature if temperature != 0.0 else 1.0
        if inv_T != 1.0:
            z_f = z_f * inv_T

        ms, ss = _student_lse_local(vp_s, v_valid, inv_T, chunk_V)
        s_lse = _merge_lse(ms, ss, tp_group)

        mt, st, ut, et = _rev_teacher_pass_local(z_f, W_shard, vp_s, s_lse, inv_T, chunk_V)
        m, st_g, ut_g, et_g = _merge_rev_teacher(mt, st, ut, et, tp_group)
        t_lse = m + st_g.log()
        kl = et_g - ut_g - s_lse + t_lse

        ctx.save_for_backward(z_f, W_shard, vp_s, t_lse, s_lse, kl)
        ctx.chunk_V = chunk_V
        ctx.inv_T = inv_T
        if diag_topk:
            diag_rows, diag_overlap = _topk_overlap_diag_vp(
                z_f, W_shard, vp_s, v_start, W.shape[0], diag_topk, diag_max_rows, tp_group
            )
            ctx.mark_non_differentiable(diag_rows, diag_overlap)
            return kl, diag_rows, diag_overlap
        return kl

    @staticmethod
    def backward(ctx, grad_output, *_diag_grads):
        z_f, W_shard, vp_s, t_lse, s_lse, kl = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        inv_T = ctx.inv_T
        N, Vp = vp_s.shape
        v_valid = W_shard.shape[0]
        device = vp_s.device
        grad = grad_output.float()

        grad_s = torch.zeros(N, Vp, dtype=torch.float32, device=device)
        for c in range(0, v_valid, chunk_V):
            c1 = min(c + chunk_V, v_valid)
            zt = z_f @ W_shard[c:c1].float().T
            log_pt = zt - t_lse.unsqueeze(1)
            s_c = vp_s[:, c:c1].float()
            if inv_T != 1.0:
                s_c = s_c * inv_T
            log_ps = s_c - s_lse.unsqueeze(1)
            ps = log_ps.exp()
            grad_s[:, c:c1] = ps * (log_ps - log_pt - kl.unsqueeze(1)) * grad.unsqueeze(1)

        if inv_T != 1.0:
            grad_s = grad_s * inv_T

        return None, None, grad_s.to(vp_s.dtype), None, None, None, None, None


# ---------------------------------------------------------------------------
# Multi-teacher grouping + CP-aware entry points
# ---------------------------------------------------------------------------


def _grouped_nitrobrew_kl_megatron(
    s_flat: torch.Tensor,  # (T, Vp) local student logits shard, flattened
    z_flat: torch.Tensor,  # (T, D_t) teacher hidden states, same row order
    key_ids_flat: torch.Tensor,  # (T,) teacher id per token
    teacher_unembeds: dict[int, torch.Tensor],
    config: DistillationConfig,
    reverse: bool,
    out_shape: tuple[int, ...],
    tp_group=None,
) -> dict[str, torch.Tensor]:
    """Grouped vocab-parallel nitrobrew KL. Returns per-token losses in out_shape."""
    T = s_flat.shape[0]
    device = s_flat.device

    loss_config = config.distillation_loss
    out = torch.full((T,), float("nan"), dtype=torch.float32, device=device)
    diag_topk = getattr(loss_config, "topk", None) or None
    diag_counts = torch.full((T,), -1.0, dtype=torch.float32, device=device) if diag_topk else None

    fn = _VocabParallelNitrobrewReverseKL if reverse else _VocabParallelNitrobrewKL
    for kid, W in teacher_unembeds.items():
        mask = key_ids_flat == int(kid)
        if not mask.any():
            continue
        group_rows = mask.nonzero(as_tuple=True)[0]
        if reverse:
            res = fn.apply(
                z_flat[mask],
                W,
                s_flat[mask],
                _CHUNK_V,
                loss_config.kd_temperature,
                diag_topk,
                _DIAG_MAX_ROWS,
                tp_group,
            )
        else:
            res = fn.apply(
                z_flat[mask],
                W,
                s_flat[mask],
                _CHUNK_V,
                loss_config.kd_temperature,
                loss_config.log_prob_min_clamp,
                diag_topk,
                _DIAG_MAX_ROWS,
                tp_group,
            )
        if diag_topk is not None:
            kl, diag_local_rows, diag_overlap = res
            diag_counts[group_rows[diag_local_rows]] = diag_overlap.float()
        else:
            kl = res
        out[mask] = kl

    missing = torch.isnan(out).sum().item()
    if missing:
        raise RuntimeError(
            f"{missing}/{T} tokens had no registered teacher unembedding; "
            f"teacher_key_to_id mapping and teacher_unembeds must cover all routes."
        )
    result = {"distillation_losses": out.view(out_shape)}
    if diag_counts is not None:
        result["overlap_counts"] = diag_counts.view(out_shape)
    return result


def compute_nitrobrew_kl_megatron(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_key_ids: torch.Tensor | None,
    teacher_unembeds: dict[int, torch.Tensor],
    config: DistillationConfig,
    data_format: str,
    local_cp_size: int | None = None,
    reverse: bool = False,
    tp_group=None,
) -> dict[str, torch.Tensor]:
    """Vocab-parallel nitrobrew KL for the Megatron engine.

    Args:
        student_logits: (bsz, seqlen/cp_size, vocab/tp_size) local shard from
            the engine's logits processor.
        teacher_hidden_states: nested (bsz, seqlen, D_t) full-sequence teacher
            hidden states from the batch (NOT CP-split yet).
        teacher_key_ids: (bsz,) per-sequence teacher ids, or None for a single
            teacher (every token routes to the sole unembedding).
        teacher_unembeds: id -> full (V, D_t) teacher unembedding (CPU ok).
        config: DistillationConfig carrying the omni loss settings.
        data_format: "thd" or "bshd".
        local_cp_size: dynamic-CP group size; None for static CP.
        reverse: KL(p_S || p_T) instead of KL(p_T || p_S).
        tp_group: TP group override (defaults to megatron's); None in tests.

    Returns:
        {"distillation_losses": (bsz, seqlen/cp_size)} plus optional
        {"overlap_counts": ...} when the top-k overlap diagnostic is on.
    """
    assert teacher_hidden_states.is_nested, "teacher_hidden_states must be a nested (jagged) tensor"
    if tp_group is None:
        tp_group = get_default_tp_group()

    # 1. CP-split the teacher payloads exactly like verl's top-k Megatron loss.
    if data_format == "thd":
        from verl.models.mcore.util import preprocess_thd_engine

        h_cp, *_ = preprocess_thd_engine(teacher_hidden_states, pre_process=True, local_cp_size=local_cp_size)
    else:
        from verl.models.mcore.util import preprocess_bshd_engine

        h_cp, *_ = preprocess_bshd_engine(teacher_hidden_states, pre_process=True)
    assert h_cp.shape[:2] == student_logits.shape[:2], (
        f"CP-split teacher hidden states {tuple(h_cp.shape[:2])} != student logits {tuple(student_logits.shape[:2])}"
    )

    # 2. Per-token teacher ids, CP-split along the same layout.
    if len(teacher_unembeds) > 1:
        if teacher_key_ids is None:
            raise KeyError(
                "multi-teacher hidden OPD requires per-sequence 'teacher_key_ids' in the batch; "
                "ActorRolloutRefWorker.update_actor derives them from the 'teacher_key' routing column."
            )
        seqlens = teacher_hidden_states.offsets().diff()
        ids_flat = torch.repeat_interleave(teacher_key_ids.to(seqlens.device), seqlens)
        cu_seqlens = torch.zeros(teacher_hidden_states.shape[0] + 1, dtype=torch.int64, device=ids_flat.device)
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
        ids_nested = torch.nested.nested_tensor_from_jagged(ids_flat, cu_seqlens)
        if data_format == "thd":
            from verl.models.mcore.util import preprocess_thd_engine

            ids_cp, *_ = preprocess_thd_engine(ids_nested, pre_process=True, local_cp_size=local_cp_size)
        else:
            from verl.models.mcore.util import preprocess_bshd_engine

            ids_cp, *_ = preprocess_bshd_engine(ids_nested, pre_process=True)
        key_ids_flat = ids_cp.flatten(0, 1).to(device=student_logits.device, dtype=torch.long)
    else:
        key_ids_flat = torch.zeros(
            student_logits.shape[0] * student_logits.shape[1], dtype=torch.long, device=student_logits.device
        )

    # 3. Grouped kernel on the flattened CP-local rows.
    z_flat = h_cp.flatten(0, 1)
    s_flat = student_logits.flatten(0, 1)
    return _grouped_nitrobrew_kl_megatron(
        s_flat=s_flat,
        z_flat=z_flat,
        key_ids_flat=key_ids_flat,
        teacher_unembeds=teacher_unembeds,
        config=config,
        reverse=reverse,
        out_shape=student_logits.shape[:2],
        tp_group=tp_group,
    )


def compute_nitrobrew_multi_kl_megatron(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_key_ids: torch.Tensor | None,
    teacher_unembeds: dict[int, torch.Tensor],
    config: DistillationConfig,
    data_format: str,
    local_cp_size: int | None = None,
    tp_group=None,
) -> dict[str, torch.Tensor]:
    """Multi-teacher forward KL KL(p_T || p_S) for the Megatron path."""
    return compute_nitrobrew_kl_megatron(
        student_logits,
        teacher_hidden_states,
        teacher_key_ids,
        teacher_unembeds,
        config,
        data_format,
        local_cp_size,
        reverse=False,
        tp_group=tp_group,
    )


def compute_nitrobrew_multi_reverse_kl_megatron(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_key_ids: torch.Tensor | None,
    teacher_unembeds: dict[int, torch.Tensor],
    config: DistillationConfig,
    data_format: str,
    local_cp_size: int | None = None,
    tp_group=None,
) -> dict[str, torch.Tensor]:
    """Multi-teacher reverse KL KL(p_S || p_T) for the Megatron path."""
    return compute_nitrobrew_kl_megatron(
        student_logits,
        teacher_hidden_states,
        teacher_key_ids,
        teacher_unembeds,
        config,
        data_format,
        local_cp_size,
        reverse=True,
        tp_group=tp_group,
    )
