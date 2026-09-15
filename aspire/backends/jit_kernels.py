"""
FlashInfer JIT attention-weight-dump kernels.

Provides ``build_dump_struct()`` for generating JIT CUDA structs across
all three FlashInfer backends (FA2, FA3/Hopper, holistic BatchAttention).

Each struct = backend-specific constructor + shared logits-transform body.

JIT scalar params:
  - sm_scale (FA2/FA3 only), num_heads, jit_kv_stride
"""

import torch
from flashinfer.jit.utils import dtype_map


# ===================================================================
# One template per backend.
# Placeholders: CLASS, MEMBERS, CTOR_INIT, LOGITS_BODY
# ===================================================================

_FA2_STRUCT = r"""
struct {{class_name}} : AttentionVariantBase {
static constexpr bool use_softmax = true;

{{members}}

template <typename Params>
__device__ __host__ {{class_name}}(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
{{ctor_init}}
}

REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
{{logits_body}}
})
};
"""

_FA3_STRUCT = r"""
struct {{class_name}} : AttentionVariantBase {
static constexpr bool use_softmax = true;

{{members}}
float scale_pv;

template <typename MainloopParams, typename BlockCoord>
__device__ {{class_name}}(const MainloopParams& params, const BlockCoord& block_coord) {
    auto [q_tile_idx_, qo_head_idx_, kv_head_idx_, qo_indptr_, kv_indptr_,
          qo_len_, kv_len_, batch_idx_] = block_coord;
    qo_len = qo_len_;
    kv_len = kv_len_;
    window_left = kv_len;
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    scale_pv = 1.0f;
{{ctor_init}}
}

template <int NUM_ROWS_PER_THREAD>
__device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, true>(sm_scale_log2);
}

REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
{{logits_body}}
})
};
"""

_HOLISTIC_STRUCT = r"""
struct {{class_name}} : AttentionVariantBase {
static constexpr bool use_softmax = true;
static constexpr bool use_logits_soft_cap = true;

{{members}}

template <typename Params>
__device__ __host__ {{class_name}}(const Params& params, uint32_t work_idx,
                                 uint8_t* smem_ptr) {
    uint32_t batch_idx = params.batch_idx_arr[work_idx];
    qo_len = params.q_len[work_idx];
    kv_len = params.kv_len[work_idx];
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
{{ctor_init}}
}

REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
{{logits_body}}
})
};
"""

_HOLISTIC_NOOP_STRUCT = r"""
struct {{class_name}} : AttentionVariantBase {
static constexpr bool use_softmax = true;
static constexpr bool use_logits_soft_cap = true;

uint32_t window_left;
float sm_scale_log2;

template <typename Params>
__device__ __host__ {{class_name}}(const Params& params, uint32_t work_idx,
                                 uint8_t* smem_ptr) {
    window_left = params.kv_len[work_idx];
    sm_scale_log2 = params.sm_scale * math::log2e;
}

REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return logits;
})
};
"""

_BACKEND_TEMPLATES = {"fa2": _FA2_STRUCT, "fa3": _FA3_STRUCT, "holistic": _HOLISTIC_STRUCT}


# ===================================================================
# Logits-transform bodies per dump_kind.
#
# The ``dump_enabled`` early-return is prepended by
# ``_with_mode_guard()`` only when use_mode_flags is True.
#
# WARNING — ``kv_idx < kv_len`` guard is intentionally omitted.
# ---------------------------------------------------------------
# These callbacks run for EVERY kv_idx in the CTA tile, including
# positions beyond the actual kv_len (the last tile may overshoot by
# up to CTA_TILE_KV − 1 positions).  Removing the guard is a
# performance optimisation; keeping it correct requires ALL of the
# following invariants to hold simultaneously:
#
#   1. **Stride padding** – the dump buffer's kv_stride must be
#      rounded up to a CTA_TILE_KV multiple (see
#      ``compute_max_cta_tile_kv`` and ``_allocate_dump_state``) so
#      that overshoot writes land inside allocated memory.
#
#   2. **-inf initialisation / reset** – the buffer must be filled
#      with −inf before each capture (``reset_attention_weights_buffer``).
#      This is critical for two reasons:
#
#      a) **Untouched KV positions** – positions beyond the last CTA
#         tile ([ceil(kv_len, CTA) … kv_stride)) are never written by
#         the kernel, so they must already contain −inf to produce
#         zero after exp2().  (Note: positions *inside* the CTA
#         overshoot ([kv_len … ceil(kv_len, CTA))) WILL be overwritten
#         with garbage logits — −inf does NOT protect them.)
#
#      b) **Unwritten query slots (first_last kernel)** – the
#         ``first_last`` callback assigns ``qo_slot = (qo_idx == 0)
#         ? 0 : 1``, so when ``qo_len == 1`` only slot 0 is written;
#         slot 1 is never touched.  Without the −inf reset, stale
#         logits from a *previous* capture (where qo_len > 1) remain
#         at VALID kv positions.  ``num_pages`` cannot mask these
#         because the pages are legitimate.  The stale logits then
#         get paired with the wrong LSE in
#         ``_select_first_last_impl``, producing arbitrary nonzero
#         scores that silently corrupt page ranking.
#
#   3. **Narrowing on read** – ``PageSelector._view_buffer`` must
#      slice the last dimension to ``[:max_seq_length]``, stripping
#      the padding tail where most overshoot lands.
#
#   4. **Per-sample num_pages masking** – ``_finalize_scores`` and
#      ``top_k_page_table_transform`` must receive the correct
#      per-sample ``num_pages`` so that pages beyond kv_len are
#      excluded from selection.  Together with the last-L-pages
#      force-selection (``scores[last_mask] = inf``), this makes
#      the small score distortion on the last partial page harmless.
#
# If any of these invariants is violated (e.g. the buffer is not
# padded, or num_pages is stale) the garbage overshoot values will
# silently corrupt page-selection scores.
# ===================================================================

_BODY_NOOP = """\
    return logits;"""

_BODY_LAST_TOKEN = """\
    if (qo_idx + 1u == qo_len) {
        attn_weights[batch_offset + (uint64_t)qo_head_idx * head_stride + kv_idx] = __float2half(logits);
    }
    return logits;"""

_BODY_FIRST_LAST = """\
    if (qo_idx == 0u || qo_idx + 1u == qo_len) {
        uint32_t qo_slot = (qo_idx == 0u) ? 0u : 1u;
        attn_weights[batch_offset + (uint64_t)qo_head_idx * head_stride + qo_slot * kv_stride + kv_idx] = __float2half(logits);
    }
    return logits;"""

_BODIES = {
    "noop": _BODY_NOOP,
    "last_token": _BODY_LAST_TOKEN,
    "first_last_compact": _BODY_FIRST_LAST,
}


def _with_mode_guard(body: str) -> str:
    """Prepend a ``dump_enabled`` early-return to *body*."""
    return (
        "    if (!dump_enabled) {\n"
        "        return logits;\n"
        "    }\n"
        + body
    )


# ===================================================================
# Struct members & constructor init per dump_kind.
# ===================================================================

_MEMBERS_NOOP = """\
uint32_t window_left, qo_len, kv_len;
float sm_scale_log2;"""

_MEMBERS_LAST_TOKEN = """\
uint32_t window_left, qo_len, kv_len;
float sm_scale_log2;
{{attn_type}}* attn_weights;
uint64_t head_stride, batch_offset;"""

_MEMBERS_LAST_TOKEN_MODE = """\
uint32_t window_left, qo_len, kv_len, dump_enabled;
float sm_scale_log2;
{{attn_type}}* attn_weights;
uint64_t head_stride, batch_offset;"""

_MEMBERS_FIRST_LAST = """\
uint32_t window_left, qo_len, kv_len;
float sm_scale_log2;
{{attn_type}}* attn_weights;
uint64_t kv_stride, head_stride, batch_offset;"""

_MEMBERS_FIRST_LAST_MODE = """\
uint32_t window_left, qo_len, kv_len, dump_enabled;
float sm_scale_log2;
{{attn_type}}* attn_weights;
uint64_t kv_stride, head_stride, batch_offset;"""

# {P} = param prefix ("params." or "params.additional_params.")
# {B} = batch index variable ("batch_idx" or "batch_idx_")

_INIT_LAST_TOKEN = """\
    attn_weights = {P}attention_weights;
    head_stride = (uint64_t){P}jit_kv_stride;
    batch_offset = (uint64_t){B} * {P}num_heads * head_stride;"""

_INIT_FIRST_LAST = """\
    attn_weights = {P}attention_weights;
    kv_stride = (uint64_t){P}jit_kv_stride;
    head_stride = kv_stride * 2u;
    batch_offset = (uint64_t){B} * {P}num_heads * head_stride;"""


# ===================================================================
# build_dump_struct
# ===================================================================

def _replace(template: str, **kwargs: str) -> str:
    """Replace ``{{key}}`` placeholders in *template* with *kwargs* values."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def build_dump_struct(
    backend: str,
    dump_kind: str,
    class_name: str,
    use_mode_flags: bool = False,
    attn_type: str = "float",
) -> str:
    """Build a complete CUDA JIT struct for attention-weight dumping.

    Args:
        backend: ``"fa2"``, ``"fa3"``, or ``"holistic"``.
        dump_kind: ``"noop"``, ``"last_token"``, or ``"first_last_compact"``.
        class_name: CUDA struct name.
        use_mode_flags: If ``True``, add a ``dump_enabled`` member and an
            early-return guard in the logits-transform body.
    """
    if backend not in ("fa2", "fa3", "holistic"):
        raise ValueError(f"Unknown backend={backend!r}")
    if dump_kind not in _BODIES:
        raise ValueError(f"Unknown dump_kind={dump_kind!r}")

    # Holistic noop is a special minimal struct.
    if dump_kind == "noop" and backend == "holistic":
        return _replace(_HOLISTIC_NOOP_STRUCT, class_name=class_name)

    # --- Param prefix per backend ---
    P = "params.additional_params." if backend == "fa3" else "params."
    B = "batch_idx_" if backend == "fa3" else "batch_idx"

    # --- Members ---
    use_mode = use_mode_flags and dump_kind != "noop"
    if dump_kind == "noop":
        members = _MEMBERS_NOOP
    elif dump_kind == "last_token":
        members = _MEMBERS_LAST_TOKEN_MODE if use_mode else _MEMBERS_LAST_TOKEN
    else:
        members = _MEMBERS_FIRST_LAST_MODE if use_mode else _MEMBERS_FIRST_LAST

    # --- Constructor init ---
    if dump_kind == "noop":
        ctor_init = ""
    elif dump_kind == "last_token":
        ctor_init = _INIT_LAST_TOKEN.format(P=P, B=B)
    else:
        ctor_init = _INIT_FIRST_LAST.format(P=P, B=B)

    if use_mode:
        ctor_init += f"\n    dump_enabled = ({P}mode_flags[{B}] != 0);"

    # --- Logits body ---
    body = _BODIES[dump_kind]
    if use_mode:
        body = _with_mode_guard(body)

    return _replace(
        _BACKEND_TEMPLATES[backend],
        class_name=class_name,
        members=members,
        ctor_init=ctor_init,
        logits_body=body,
        attn_type=attn_type,
    )


# ===================================================================
# Scalar / tensor param specs
# ===================================================================

_BASE_TENSOR_NAMES = ["attention_weights"]
_FA2_SCALAR_NAMES = ["sm_scale", "num_heads", "jit_kv_stride"]
_FA2_SCALAR_TYPES = ["double", "int64_t", "int64_t"]
_HOLISTIC_SCALAR_NAMES = ["num_heads", "jit_kv_stride"]
_HOLISTIC_SCALAR_TYPES = ["int64_t", "int64_t"]


def _scalar_spec(backend: str):
    """Return (scalar_names, scalar_types) for the given backend."""
    if backend == "holistic":
        return list(_HOLISTIC_SCALAR_NAMES), list(_HOLISTIC_SCALAR_TYPES)
    return list(_FA2_SCALAR_NAMES), list(_FA2_SCALAR_TYPES)


# ===================================================================
# build_dump_jit_args — public API for FA2/FA3 backends
# ===================================================================

def build_dump_jit_args(
    backend: str,
    dtype: torch.dtype,
    head_dim: int,
    dump_kind: str = "last_token",
    class_name: str = "BatchDumpAttentionWeights",
    op_name: str = "batch_prefill_dump_attention_weights",
    use_mode_flags: bool = False,
    dump_buffer_dtype: torch.dtype = torch.float32,
) -> list:
    """Build FlashInfer JIT args for an attention-weight-dump kernel.

    The wrapper.run() call signature becomes::

        wrapper.run(q, kv_cache, attention_weights_buf, sm_scale,
                    num_heads, jit_kv_stride, return_lse=..., out=...)
    """
    backend_key = "fa3" if backend == "fa3" else "fa2"
    scalar_names, scalar_types = _scalar_spec(backend_key)

    attn_type = dtype_map[dump_buffer_dtype]
    code = build_dump_struct(backend_key, dump_kind, class_name, use_mode_flags, attn_type=attn_type)

    tensor_names = list(_BASE_TENSOR_NAMES)
    tensor_types = [attn_type]
    if use_mode_flags and dump_kind != "noop":
        tensor_names.append("mode_flags")
        tensor_types.append("int32_t")

    return [
        op_name, dtype, dtype, dtype, torch.int32,
        head_dim, head_dim,
        tensor_names, tensor_types,
        scalar_names, scalar_types,
        class_name, code,
    ]
