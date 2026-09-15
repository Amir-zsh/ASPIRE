from __future__ import annotations

import torch


def local_topk_candidates(
    logits: torch.Tensor, top_k: int, vocab_start: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-width local candidates, padding undersized shards."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if logits.shape[-1] < 1:
        raise ValueError("logit shard must contain at least one vocabulary entry")
    local_k = min(top_k, logits.shape[-1])
    values, indices = logits.topk(local_k, dim=-1)
    indices = indices + vocab_start
    if local_k == top_k:
        return values, indices
    pad_shape = (*values.shape[:-1], top_k - local_k)
    values = torch.cat(
        (values, torch.full(pad_shape, -torch.inf, dtype=values.dtype, device=values.device)),
        dim=-1,
    )
    indices = torch.cat(
        (indices, torch.zeros(pad_shape, dtype=indices.dtype, device=indices.device)),
        dim=-1,
    )
    return values, indices


def merge_topk_candidates(
    values: torch.Tensor, indices: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge concatenated rank-local candidates into global top-k."""

    if values.shape != indices.shape:
        raise ValueError("candidate values and indices must have identical shapes")
    if top_k < 1 or top_k > values.shape[-1]:
        raise ValueError("top_k must fit within the candidate width")
    selected_values, selected = values.topk(top_k, dim=-1)
    selected_indices = torch.gather(indices, -1, selected)
    return selected_values, selected_indices


def verify_topk_speculation(
    draft_probs: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_counts: torch.Tensor,
    target_probs: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify top-k draft distributions in a K+1 local vocabulary."""
    from flashinfer.sampling import chain_speculative_sampling

    requests, max_draft, top_k = draft_probs.shape
    device = draft_probs.device
    local_vocab = top_k + 1
    draft_pos = torch.arange(max_draft, device=device).unsqueeze(0)
    valid_draft = draft_pos < draft_counts.unsqueeze(1)
    target_pos = torch.arange(max_draft + 1, device=device).unsqueeze(0)
    valid_target = target_pos <= draft_counts.unsqueeze(1)

    overlap = target_ids[:, :max_draft].unsqueeze(-1) == draft_ids.unsqueeze(-2)
    mapped_draft = (draft_probs.unsqueeze(-2) * overlap.float()).sum(dim=-1)
    mapped_draft[~valid_draft] = 0
    local_draft = torch.zeros(requests, max_draft, local_vocab, dtype=torch.float32, device=device)
    local_draft[:, :, :top_k] = mapped_draft

    masked_target = target_probs.clone()
    masked_target[~valid_target] = 0
    local_target = torch.zeros(requests, max_draft + 1, local_vocab, dtype=torch.float32, device=device)
    local_target[:, :, :top_k] = masked_target

    token_match = target_ids[:, :max_draft] == draft_tokens.unsqueeze(-1).long()
    in_target = token_match.any(dim=-1) & valid_draft
    local_tokens = torch.where(in_target, token_match.float().argmax(dim=-1), top_k).to(torch.int32)
    draft_self = draft_ids == draft_tokens.unsqueeze(-1).long()
    draft_probability = (draft_probs * draft_self.float()).sum(dim=-1)
    local_draft[:, :, top_k] = draft_probability * (~in_target & valid_draft).float()
    local_draft[~valid_draft, top_k] = 1.0

    output_local, _, emitted = chain_speculative_sampling(
        local_draft, local_tokens, local_target, deterministic=True
    )
    emitted = emitted.clamp(max=draft_counts)
    valid = (output_local >= 0) & (output_local < top_k)
    request_idx = torch.arange(requests, device=device).unsqueeze(1)
    decode_idx = torch.arange(max_draft + 1, device=device).unsqueeze(0)
    gathered = target_ids[
        request_idx,
        decode_idx,
        output_local.clamp(0, top_k - 1).long(),
    ]
    output = torch.where(valid, gathered, torch.full_like(gathered, -1))
    output[decode_idx > emitted.unsqueeze(1)] = -1
    return output, emitted


def first_eos(
    output: torch.Tensor,
    committed: torch.Tensor,
    eos_token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not eos_token_ids:
        return committed, torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
    positions = torch.arange(output.shape[1], device=output.device).unsqueeze(0)
    valid = positions < committed.unsqueeze(1)
    eos = torch.zeros_like(valid)
    for token_id in eos_token_ids:
        eos |= output == token_id
    eos &= valid
    found = eos.any(dim=1)
    first = eos.float().argmax(dim=1).to(torch.int32)
    return torch.where(found, first + 1, committed), found
