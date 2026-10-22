"""Method-neutral construction of token-sequence intervention controls."""

from __future__ import annotations

import torch

from .types import GraphPromptTokens


def global_token_mean(tokens: GraphPromptTokens) -> torch.Tensor:
    return tokens.values[tokens.mask].float().mean(dim=0).to(dtype=tokens.values.dtype)


def mean_token_control(tokens: GraphPromptTokens, mean_token: torch.Tensor) -> GraphPromptTokens:
    """Replace every valid position by one training-set mean token.

    This definition remains valid for non-canonical or variable-length token
    interfaces: it preserves each example's original token count while
    removing all graph- and slot-specific variation.
    """
    if mean_token.shape != (tokens.embedding_dim,):
        raise ValueError("mean_token has the wrong embedding dimension")
    values = tokens.values.clone()
    values[tokens.mask] = mean_token.to(dtype=values.dtype, device=values.device)
    return GraphPromptTokens(values, tokens.mask.clone(), tokens.example_ids, tokens.slot_semantics)


def sequence_permutation(tokens: GraphPromptTokens, permutation: torch.Tensor) -> GraphPromptTokens:
    """Swap whole token sequences; never silently align variable-length slots."""
    if permutation.ndim != 1 or permutation.shape[0] != tokens.num_examples:
        raise ValueError("permutation must contain one source index per example")
    if sorted(permutation.tolist()) != list(range(tokens.num_examples)):
        raise ValueError("permutation must be a reordering of all examples")
    return GraphPromptTokens(
        values=tokens.values[permutation].clone(),
        mask=tokens.mask[permutation].clone(),
        example_ids=tokens.example_ids,
        slot_semantics=tokens.slot_semantics,
    )
