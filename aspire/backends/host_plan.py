"""CPU mirrors of the per-step attention-plan arrays.

FlashInfer's plan phase consumes three host integer arrays — ``qo_indptr``,
``kv_indptr``, and ``kv_len`` — and every one of them is a pure integer
function of per-row lengths the scheduler already tracks on the CPU
(committed sequence lengths, draft counts, the verify decision, and the
finished mask). Deriving them on the device would need a device readback
and stream synchronization every step.

``HostPlanState`` maintains the same quantities in NumPy, applying exactly
the update rules the device-side ``prepare_forward_inputs`` applies to its
tensors, and renders the host arrays the wrapper's ``plan_host`` entry point
passes straight to the C++ planner. The page indices themselves stay on the
device — the plan never reads them.

Every formula here mirrors a specific device computation; the pairing is
verified end-to-end by the ``ASPIRE_PLAN_PARITY`` debug mode and the
boundary-parity/golden gates. When the runtime cannot supply host state
(warm-up, tests), the backend falls back to the synchronizing device path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HostPlanConfig:
    """Backend facts the mirror needs to reproduce the device plan arrays."""

    page_size: int
    max_batch: int
    # 'replan'   — draft tables refreshed for the whole batch every step
    #              (the aspire intra-draft-refresh backend)
    # 'sink_recent'— sink+recent selection, also refreshed every step
    # 'gated'    — refreshed only for previous-step verify rows, with
    #              page-boundary spill inserts in between
    draft_kind: str
    # 'effective_full' — dump wrapper plans full-context attention over
    #                    effective cache lengths (aspire)
    # 'mirror'         — dump wrapper reuses the mixed wrapper's arrays
    # None             — no dump wrapper (autoregressive)
    full_kind: str | None
    subset_cache_size: int
    table_width: int
    min_draft_pages: int | None
    draft_percent: float | None
    valid_floor_l: int
    # Pages allocated per slot in the (contiguous, arange) block table. The
    # 'effective_full' dump wrapper addresses each row's pages at the fixed
    # offset row*pages_per_slot in a static index buffer, so its indptr is a
    # constant stride and only kv_len varies per step.
    pages_per_slot: int = 0


class HostPlanState:
    """Host-side twin of the backend's length/page bookkeeping.

    The scheduler feeds it the pre-step row state (identical on every rank);
    ``build_step`` advances the mirror exactly as the device prep advances its
    tensors and returns the plan arrays plus the spill/refresh row lists the
    device path would otherwise discover through synchronizing reductions.
    """

    def __init__(self, config: HostPlanConfig):
        self.config = config
        mb = config.max_batch
        self.draft_num_pages = np.zeros(mb, dtype=np.int64)
        self._prev_verify = np.zeros(mb, dtype=np.bool_)

        # Rendered plan arrays (host int32, consumed synchronously by the
        # C++ planner inside plan_host — no pinning needed).
        self.qo_indptr = torch.zeros(mb + 2, dtype=torch.int32)
        self.kv_indptr = torch.zeros(mb + 2, dtype=torch.int32)
        self.kv_len = torch.zeros(mb + 1, dtype=torch.int32)
        self.full_kv_indptr = torch.zeros(mb + 2, dtype=torch.int32)
        self.full_kv_len = torch.zeros(mb + 1, dtype=torch.int32)

    # -- device-formula replicas -------------------------------------------

    def _num_pages(self, lens: np.ndarray) -> np.ndarray:
        """`_get_num_pages`: ceil(len / page_size) in integer arithmetic."""
        return (lens + self.config.page_size - 1) // self.config.page_size

    def _last_page_len(self, lens: np.ndarray) -> np.ndarray:
        """`_get_last_page_len`: (len - 1) % page + 1, 0 for empty rows."""
        return np.where(lens > 0, (lens - 1) % self.config.page_size + 1, 0)

    def _valid_counts(self, num_pages: np.ndarray) -> np.ndarray:
        """`PageSelector.compute_valid_counts`, including its fp32 rounding."""
        cfg = self.config
        if cfg.draft_kind == "sink_recent":
            # `the AR backend sink+recent selector` caps at the full
            # subset size instead of calling compute_valid_counts.
            return np.minimum(num_pages, cfg.subset_cache_size)
        hi = np.minimum(num_pages, cfg.subset_cache_size)
        if cfg.draft_percent is None:
            if cfg.min_draft_pages is not None:
                return np.minimum(num_pages, cfg.min_draft_pages)
            return hi
        floor = (
            cfg.min_draft_pages if cfg.min_draft_pages is not None else cfg.valid_floor_l
        )
        lo = np.minimum(num_pages, floor)
        # The device computes num_pages.float() * pct / 100 in fp32; replicate
        # the dtype so ceil ties break identically.
        target = np.ceil(
            num_pages.astype(np.float32)
            * np.float32(cfg.draft_percent)
            / np.float32(100.0)
        ).astype(np.int64)
        return np.clip(target, lo, hi)

    def _init_counts(self, num_pages: np.ndarray) -> np.ndarray:
        """`_build_recent_draft_rows` seeding rule (compute_valid_counts on
        the base selector even for the sink+recent subclass)."""
        cfg = self.config
        hi = np.minimum(num_pages, cfg.subset_cache_size)
        if cfg.draft_percent is None:
            if cfg.min_draft_pages is not None:
                return np.minimum(num_pages, cfg.min_draft_pages)
            return hi
        floor = (
            cfg.min_draft_pages if cfg.min_draft_pages is not None else cfg.valid_floor_l
        )
        lo = np.minimum(num_pages, floor)
        target = np.ceil(
            num_pages.astype(np.float32)
            * np.float32(cfg.draft_percent)
            / np.float32(100.0)
        ).astype(np.int64)
        return np.clip(target, lo, hi)

    # -- lifecycle ----------------------------------------------------------

    def reset_all(self, seq_lens) -> None:
        """Mirror `_init_draft_indices` after a full prefill."""
        lens = np.asarray(seq_lens, dtype=np.int64)
        self.draft_num_pages[: lens.size] = self._init_counts(self._num_pages(lens))
        self._prev_verify[:] = False

    def reset_slots(self, slot_ids, seq_lens) -> None:
        """Mirror the slot re-seeding when fixed batch slots are refilled."""
        ids = np.asarray(slot_ids, dtype=np.int64)
        lens = np.asarray(seq_lens, dtype=np.int64)
        self.draft_num_pages[ids] = self._init_counts(self._num_pages(lens))
        self._prev_verify[ids] = False

    # -- per-step -----------------------------------------------------------

    def build_step(
        self,
        *,
        verify: np.ndarray,
        finished: np.ndarray,
        draft_counts: np.ndarray,
        seq_lens: np.ndarray,
        total_tokens: int,
        bucket_tokens: int,
    ) -> dict:
        """Advance the mirror by one step and render the plan arrays.

        Inputs are the scheduler's pre-step row state. Returns a dict with the
        host plan tensors plus ``spill_rows`` / ``refresh_rows`` (NumPy index
        arrays) for the device prep to consume without mask reductions.
        """
        cfg = self.config
        # Working width is the actual generate batch (may be smaller than the
        # engine's max_batch when fewer prompts are supplied); the device prep
        # sizes everything from the same count.
        mb = verify.shape[0]
        verify = verify.astype(np.bool_)
        finished = finished.astype(np.bool_)
        active = ~finished
        # Device: modes = where(verify, 1, 0); is_draft = modes == DRAFT_MODE,
        # which therefore includes finished rows.
        is_mode_draft = ~verify
        dec = np.where(verify, draft_counts + 1, np.where(active, 1, 0)).astype(
            np.int64
        )

        # Device cachelens at plan time: pre-step invariant seq+drafts, minus
        # the runtime's verify rewind, plus prep's `+= dec`.
        cachelens = seq_lens + draft_counts - np.where(verify, draft_counts, 0) + dec
        # Device draft_cachelens at plan time: invariant seq+drafts, plus
        # prep's `+= is_draft` (which the runtime later reverts for finished
        # rows — after the plan has been consumed).
        draft_cachelens = seq_lens + draft_counts + np.where(is_mode_draft, 1, 0)
        effective = np.where(is_mode_draft, draft_cachelens, cachelens)

        verify_num_pages = self._num_pages(cachelens)
        last_page_len = self._last_page_len(effective)

        # Draft-table page counts, following each backend family's refresh rule.
        refresh_rows = np.empty(0, dtype=np.int64)
        spill_rows = np.empty(0, dtype=np.int64)
        if cfg.draft_kind in ("replan", "sink_recent"):
            self.draft_num_pages[:mb] = self._valid_counts(verify_num_pages)
        else:  # 'gated'
            refresh_rows = np.flatnonzero(self._prev_verify)
            if refresh_rows.size:
                self.draft_num_pages[refresh_rows] = self._valid_counts(
                    verify_num_pages[refresh_rows]
                )
            # Page-boundary spill: a draft token opening a new page must be
            # appended to any table that was not refreshed this step. The
            # block table is contiguous, so a stale tail never equals the new
            # page, and a just-refreshed
            # table always ends at it.
            just_crossed = (
                ((draft_cachelens - 1) % cfg.page_size == 0)
                & is_mode_draft
                & (dec > 0)
            )
            stale = just_crossed.copy()
            stale[refresh_rows] = False
            can_insert = self.draft_num_pages < cfg.table_width
            spill_rows = np.flatnonzero(stale & can_insert)
            self.draft_num_pages[spill_rows] += 1

        mixed_pages = np.where(
            is_mode_draft, self.draft_num_pages[:mb], verify_num_pages
        )

        qo = self.qo_indptr.numpy()
        qo[0] = 0
        qo[1 : mb + 1] = np.cumsum(dec)
        qo[mb + 1 :] = bucket_tokens
        if int(qo[mb]) != int(total_tokens):
            raise RuntimeError(
                f"host plan mirror diverged: token total {int(qo[mb])} != "
                f"scheduler total {int(total_tokens)}"
            )

        kvp = self.kv_indptr.numpy()
        kvp[0] = 0
        kvp[1 : mb + 1] = np.cumsum(mixed_pages)
        kvp[mb + 1 :] = kvp[mb]
        kvl = self.kv_len.numpy()
        # `get_seq_lens`: clamp(pages - 1, 0) * page + last_page_len, with the
        # padding row's last_page_len forced to zero.
        kvl[:mb] = np.maximum(mixed_pages - 1, 0) * cfg.page_size + last_page_len
        kvl[mb:] = 0

        # Device-input metadata rendered host-side so the backend can upload
        # it instead of recomputing it with ~20 small kernels per step. Every
        # array mirrors the exact device formula (parity-gated).
        input_pos = np.where(is_mode_draft, draft_cachelens - 1, cachelens - dec)
        sample_indices = np.repeat(np.arange(mb, dtype=np.int64), dec)
        qo_prefix = np.concatenate(([0], np.cumsum(dec)[:-1]))
        position_indices = np.arange(int(dec.sum()), dtype=np.int64) - np.repeat(
            qo_prefix, dec
        )
        result = {
            "qo_indptr": self.qo_indptr,
            "kv_indptr": self.kv_indptr,
            "kv_len": self.kv_len,
            "full_qo_indptr": None,
            "full_kv_indptr": None,
            "full_kv_len": None,
            "spill_rows": spill_rows,
            "refresh_rows": refresh_rows,
            # Exposed for the ASPIRE_PLAN_PARITY debug compare.
            "draft_num_pages": self.draft_num_pages,
            "dec": dec,
            "verify_num_pages": verify_num_pages,
            "effective_num_pages": self._num_pages(effective),
            "last_page_len": last_page_len,
            "input_pos": input_pos,
            "sample_indices": sample_indices,
            "position_indices": position_indices,
        }

        if cfg.full_kind == "mirror":
            result["full_qo_indptr"] = self.qo_indptr
            result["full_kv_indptr"] = self.kv_indptr
            result["full_kv_len"] = self.kv_len
        elif cfg.full_kind == "effective_full":
            effective_pages = self._num_pages(effective)
            fkvp = self.full_kv_indptr.numpy()
            fkvl = self.full_kv_len.numpy()
            if cfg.pages_per_slot > 0:
                # Static index layout: row i's pages sit at the fixed offset
                # i*pages_per_slot in the setup-time arange buffer, so the
                # indptr is a constant stride and the kernel still reads the
                # identical pages in the identical order (bitwise-identical
                # attention). Only kv_len changes per step.
                fkvp[: mb + 1] = np.arange(mb + 1, dtype=np.int64) * cfg.pages_per_slot
                fkvp[mb + 1 :] = mb * cfg.pages_per_slot
            else:
                fkvp[0] = 0
                fkvp[1 : mb + 1] = np.cumsum(effective_pages)
                fkvp[mb + 1 :] = fkvp[mb]
            fkvl[:mb] = (
                np.maximum(effective_pages - 1, 0) * cfg.page_size + last_page_len
            )
            fkvl[mb:] = 0
            result["full_qo_indptr"] = self.qo_indptr
            result["full_kv_indptr"] = self.full_kv_indptr
            result["full_kv_len"] = self.full_kv_len

        self._prev_verify[:mb] = verify
        return result


__all__ = ["HostPlanConfig", "HostPlanState"]
