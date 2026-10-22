"""Read-only projected-token interface for the released GraSP encoder.

GraSP's ``SubgraphGNN`` produces one embedding per retrieved entity and its
``Graph2Prefix`` maps every such embedding into an LLM embedding.  This
adapter deliberately stops at that boundary: it neither changes GraSP graph
retrieval/GNN/Graph2Prefix nor invokes the repository's OpenAI answer stage.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..types import GraphPromptTokens


class GraSPPrefixAdapter:
    """Expose GraSP's post-``Graph2Prefix`` entity-token sequence."""

    def __init__(self, gnn: torch.nn.Module, graph2prefix: torch.nn.Module):
        self.gnn = gnn
        self.graph2prefix = graph2prefix

    @torch.no_grad()
    def capture(self, graph_batch, query_embeddings: torch.Tensor, example_ids: Sequence[str]) -> GraphPromptTokens:
        node_embeddings, _ = self.gnn(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_type=getattr(graph_batch, "edge_type", None),
            batch=graph_batch.batch,
            query=query_embeddings,
        )
        values, mask = self.graph2prefix(node_embeddings, graph_batch.batch)
        return GraphPromptTokens(
            values=values.detach().cpu(), mask=mask.detach().cpu(),
            example_ids=[str(identifier) for identifier in example_ids],
            # Entity order is not a canonical cross-graph slot alignment.
            slot_semantics=False,
        )

    def project_for_answerer(self, graph_batch, query_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return released prefix values/mask while retaining training gradients."""
        node_embeddings, _ = self.gnn(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_type=getattr(graph_batch, "edge_type", None),
            batch=graph_batch.batch,
            query=query_embeddings,
        )
        return self.graph2prefix(node_embeddings, graph_batch.batch)
