"""Single- and multi-token geometry diagnostics.

The single-token measures match the paper's inter-sample cosine, residual
ratio, and norm ratio.  Multi-token methods additionally receive pooled
between-example, within-example, and (when slots are canonical) slotwise
between-example measures.  No sequence is flattened: that would conflate a
method's number of tokens with its graph-dependent variation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .types import GraphPromptTokens


@dataclass(frozen=True)
class GeometryStats:
    num_examples: int
    mean_tokens_per_example: float
    pooled_inter_sample_cosine: float
    pooled_residual_ratio: float
    pooled_residual_over_word_embedding: float | None
    token_norm_ratio: float | None
    within_example_cosine: float | None
    slotwise_inter_sample_cosine: float | None
    token_effective_rank: float
    pooled_effective_rank: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_off_diagonal_cosine(vectors: torch.Tensor) -> float:
    """Exact mean pairwise cosine for a non-empty matrix ``[N, D]``."""
    if vectors.shape[0] < 2:
        return float("nan")
    normalized = F.normalize(vectors.float(), dim=-1)
    similarities = normalized @ normalized.T
    return ((similarities.sum() - similarities.diagonal().sum()) / (vectors.shape[0] * (vectors.shape[0] - 1))).item()


def _effective_rank(vectors: torch.Tensor) -> float:
    """Participation-ratio effective rank after mean centering."""
    if vectors.shape[0] < 2:
        return float("nan")
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    squared = singular_values.square()
    denominator = squared.square().sum()
    if denominator == 0:
        return 0.0
    return (squared.sum().square() / denominator).item()


def _within_example_cosine(tokens: GraphPromptTokens) -> float | None:
    values = F.normalize(tokens.values.float(), dim=-1)
    total = torch.zeros((), dtype=values.dtype, device=values.device)
    pairs = 0
    for example, valid in zip(values, tokens.mask):
        rows = example[valid]
        if rows.shape[0] < 2:
            continue
        cosine = rows @ rows.T
        total += cosine.sum() - cosine.diagonal().sum()
        pairs += rows.shape[0] * (rows.shape[0] - 1)
    return None if pairs == 0 else (total / pairs).item()


def _slotwise_inter_sample_cosine(tokens: GraphPromptTokens) -> float | None:
    if not tokens.slot_semantics:
        return None
    weighted_sum = 0.0
    weighted_pairs = 0
    for slot in range(tokens.values.shape[1]):
        rows = tokens.values[tokens.mask[:, slot], slot]
        n_rows = rows.shape[0]
        if n_rows < 2:
            continue
        weighted_sum += _mean_off_diagonal_cosine(rows) * n_rows * (n_rows - 1)
        weighted_pairs += n_rows * (n_rows - 1)
    return None if weighted_pairs == 0 else weighted_sum / weighted_pairs


def compute_geometry(tokens: GraphPromptTokens, vocabulary_embeddings: torch.Tensor | None = None) -> GeometryStats:
    """Compute diagnostics without using padded token positions.

    ``vocabulary_embeddings`` should be the LLM's input-embedding matrix if a
    norm ratio is desired.  This keeps the reference scale architecture-local.
    """
    pooled = tokens.pooled()
    pooled_mean = pooled.mean(dim=0)
    mean_norm = pooled_mean.norm()
    residual = (pooled - pooled_mean).norm(dim=-1).mean()
    residual_ratio = float("inf") if mean_norm == 0 else (residual / mean_norm).item()
    all_tokens = tokens.values[tokens.mask]

    norm_ratio = None
    residual_over_word_embedding = None
    if vocabulary_embeddings is not None:
        if vocabulary_embeddings.ndim != 2 or vocabulary_embeddings.shape[1] != tokens.embedding_dim:
            raise ValueError("vocabulary_embeddings must have shape [vocabulary, embedding_dim]")
        reference_norm = vocabulary_embeddings.float().norm(dim=-1).median()
        norm_ratio = float("inf") if reference_norm == 0 else (all_tokens.float().norm(dim=-1).mean() / reference_norm).item()
        residual_over_word_embedding = float("inf") if reference_norm == 0 else (residual / reference_norm).item()

    return GeometryStats(
        num_examples=tokens.num_examples,
        mean_tokens_per_example=tokens.mask.sum(dim=1).float().mean().item(),
        pooled_inter_sample_cosine=_mean_off_diagonal_cosine(pooled),
        pooled_residual_ratio=residual_ratio,
        pooled_residual_over_word_embedding=residual_over_word_embedding,
        token_norm_ratio=norm_ratio,
        within_example_cosine=_within_example_cosine(tokens),
        slotwise_inter_sample_cosine=_slotwise_inter_sample_cosine(tokens),
        token_effective_rank=_effective_rank(all_tokens),
        pooled_effective_rank=_effective_rank(pooled),
    )
