"""Exact paper geometry excluding optional, expensive effective-rank SVDs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import GeometryStats, _mean_off_diagonal_cosine, _slotwise_inter_sample_cosine, _within_example_cosine
from .types import GraphPromptTokens


def compute_fast_geometry(tokens: GraphPromptTokens, vocabulary_embeddings: torch.Tensor | None = None) -> GeometryStats:
    """Compute cosine, residual ratio, norm ratio, and intra-sample redundancy.

    Effective rank is deliberately omitted (``None``): its full SVD is not
    needed for the paper protocol and can dominate runtime for multi-token
    variable-length interfaces.  All retained statistics are exact.
    """
    pooled = tokens.pooled()
    mean = pooled.mean(dim=0)
    mean_norm = mean.norm()
    residual = (pooled - mean).norm(dim=-1).mean()
    all_tokens = tokens.values[tokens.mask]
    norm_ratio = None
    residual_over_word_embedding = None
    if vocabulary_embeddings is not None:
        reference = vocabulary_embeddings.float().norm(dim=-1).median()
        norm_ratio = float("inf") if reference == 0 else (all_tokens.float().norm(dim=-1).mean() / reference).item()
        residual_over_word_embedding = float("inf") if reference == 0 else (residual / reference).item()
    return GeometryStats(
        num_examples=tokens.num_examples,
        mean_tokens_per_example=tokens.mask.sum(dim=1).float().mean().item(),
        pooled_inter_sample_cosine=_mean_off_diagonal_cosine(pooled),
        pooled_residual_ratio=float("inf") if mean_norm == 0 else (residual / mean_norm).item(),
        pooled_residual_over_word_embedding=residual_over_word_embedding,
        token_norm_ratio=norm_ratio,
        within_example_cosine=_within_example_cosine(tokens),
        slotwise_inter_sample_cosine=_slotwise_inter_sample_cosine(tokens),
        token_effective_rank=None,
        pooled_effective_rank=None,
    )
