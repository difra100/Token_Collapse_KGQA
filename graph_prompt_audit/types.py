"""Shared, method-neutral representation of a graph prompt token sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GraphPromptTokens:
    """Projected graph tokens supplied to an LLM.

    ``values`` is padded to ``[examples, token_slots, embedding_dim]`` and
    ``mask`` marks slots that are genuine graph tokens.  Padding is never
    included in any diagnostic.  ``slot_semantics`` is true only where a slot
    is comparable across examples (for example, DiffPool cluster slots).  It
    must be false for node-selection methods whose order is not canonical.
    """

    values: torch.Tensor
    mask: torch.Tensor
    example_ids: Sequence[str] | None = None
    slot_semantics: bool = False

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("values must have shape [examples, slots, dimension]")
        if self.mask.shape != self.values.shape[:2]:
            raise ValueError("mask must have shape values.shape[:2]")
        if self.mask.dtype != torch.bool:
            raise TypeError("mask must be boolean")
        if self.example_ids is not None and len(self.example_ids) != self.values.shape[0]:
            raise ValueError("example_ids must contain one id per example")
        if not torch.all(self.mask.any(dim=1)):
            raise ValueError("every example must contain at least one graph token")

    @property
    def num_examples(self) -> int:
        return self.values.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.values.shape[2]

    def pooled(self) -> torch.Tensor:
        """Mask-aware mean token, used only for sequence-level diagnostics."""
        weights = self.mask.unsqueeze(-1).to(dtype=self.values.dtype)
        return (self.values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
