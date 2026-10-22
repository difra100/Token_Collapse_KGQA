"""Read-only access to the projected prompt interface in GRetPool.

The adapter hooks the released model's existing ``projector`` module.  It does
not replace graph encoding, pooling, prompt construction, or decoding.  For
fixed-token pooling, projection happens in a single call per batch; for
variable-length pooling it happens once per example.  The latter distinction is
preserved instead of padded into a fictitious canonical token order.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch

from ..types import GraphPromptTokens


FIXED_POOLING = frozenset({"mean", "virtual", "diffpool", "mincutpool", "randk"})
VARIABLE_POOLING = frozenset({"topk", "sag", "sagM", "none"})


@dataclass
class ProjectorCapture:
    """Capture, and optionally replace, outputs of a projected-token module."""

    module: torch.nn.Module
    outputs: list[torch.Tensor] = field(default_factory=list)
    _replacement_calls: list[torch.Tensor] | None = None
    _handle: torch.utils.hooks.RemovableHandle | None = None

    def __enter__(self) -> "ProjectorCapture":
        self.outputs.clear()
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc_info) -> None:
        assert self._handle is not None
        self._handle.remove()
        self._handle = None

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        self.outputs.append(output.detach().cpu())
        if self._replacement_calls is None:
            return output
        if not self._replacement_calls:
            raise RuntimeError("more projector calls than replacement sequences")
        replacement = self._replacement_calls.pop(0).to(device=output.device, dtype=output.dtype)
        if replacement.shape != output.shape:
            raise ValueError(f"replacement shape {tuple(replacement.shape)} does not match projector output {tuple(output.shape)}")
        return replacement

    @contextmanager
    def replace(self, projected_calls: Sequence[torch.Tensor]) -> Iterator[None]:
        if self._replacement_calls is not None:
            raise RuntimeError("nested token replacements are not supported")
        self._replacement_calls = [value.detach().cpu() for value in projected_calls]
        try:
            yield
            if self._replacement_calls:
                raise RuntimeError("fewer projector calls than replacement sequences")
        finally:
            self._replacement_calls = None


class GRetPoolAdapter:
    """Adapter around an instantiated, released GRetPool ``GraphLLM``."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.pooling = model.pooling_type
        if self.pooling not in FIXED_POOLING | VARIABLE_POOLING:
            raise ValueError(f"unsupported GRetPool strategy: {self.pooling}")

    @property
    def fixed_length(self) -> bool:
        return self.pooling in FIXED_POOLING

    @property
    def slot_semantics(self) -> bool:
        # Randomly selected nodes have a fixed count but not a matched semantic
        # position.  Other fixed-count rows have model-defined positions.
        return self.pooling in {"mean", "virtual", "diffpool", "mincutpool"}

    def _project(self, batch: dict) -> list[torch.Tensor]:
        """Return projected tokens in the exact forms used by ``inference``."""
        with torch.no_grad():
            unprojected, _, _ = self.model.encode_graphs(batch)
            if self.fixed_length:
                batch_size, token_count, _ = unprojected.shape
                projected = self.model.projector(unprojected.reshape(batch_size * token_count, -1))
                return [projected.reshape(batch_size, token_count, -1).detach().cpu()]
            return [self.model.projector(tokens).detach().cpu() for tokens in unprojected]

    def capture_batch(self, batch: dict) -> GraphPromptTokens:
        """Capture real projected tokens without invoking LLM generation."""
        projected_calls = self._project(batch)
        if self.fixed_length:
            values = projected_calls[0]
            mask = torch.ones(values.shape[:2], dtype=torch.bool)
        else:
            rows = projected_calls
            width = max(row.shape[0] for row in rows)
            values = torch.zeros((len(rows), width, rows[0].shape[1]), dtype=rows[0].dtype)
            mask = torch.zeros((len(rows), width), dtype=torch.bool)
            for index, row in enumerate(rows):
                values[index, : row.shape[0]] = row
                mask[index, : row.shape[0]] = True
        return GraphPromptTokens(
            values=values,
            mask=mask,
            example_ids=[str(item) for item in batch["id"]],
            slot_semantics=self.slot_semantics,
        )

    def inference(self, batch: dict, replacement: GraphPromptTokens | None = None) -> dict:
        """Run the method's own inference path, optionally at its token boundary."""
        capture = ProjectorCapture(self.model.projector)
        with capture:
            if replacement is None:
                return self.model.inference(batch)
            if replacement.num_examples != len(batch["id"]):
                raise ValueError("replacement batch does not match batch ids")
            calls = self._replacement_calls(replacement)
            with capture.replace(calls):
                return self.model.inference(batch)

    def _replacement_calls(self, replacement: GraphPromptTokens) -> list[torch.Tensor]:
        if self.fixed_length:
            if not bool(replacement.mask.all()):
                raise ValueError("fixed-length pooling requires a full replacement mask")
            return [replacement.values.reshape(-1, replacement.embedding_dim)]
        return [replacement.values[index, replacement.mask[index]] for index in range(replacement.num_examples)]
