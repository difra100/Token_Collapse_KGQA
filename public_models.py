"""Portable graph-to-LLM model registrations for public training.

The upstream GraphLLM hard-codes a 4096-dimensional projector output.  The
models here retain its encoder, prompt construction, loss, and training path,
but derive the projector output width from the loaded decoder's input
embeddings.  They therefore support Llama- and Qwen-style causal LMs whenever
their Hugging Face implementation exposes ``model.get_input_embeddings()``.
"""

import torch.nn as nn

from src.model import load_model
from src.model.graph_llm import GraphLLM


class FlexibleGraphLLM(GraphLLM):
    """Upstream GraphLLM with a decoder-width-aware sigmoid projector."""

    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        output_dim = self.word_embedding.embedding_dim
        self.projector = self.build_projector(args.gnn_hidden_dim, output_dim).to(self.model.device)

    @staticmethod
    def build_projector(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, output_dim),
        )


class FlexibleGELUGraphLLM(FlexibleGraphLLM):
    """The same interface with GELU rather than sigmoid in the projector."""

    @staticmethod
    def build_projector(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim),
        )


def register() -> None:
    """Expose portable public model names to the upstream trainer."""
    load_model.update(
        {
            "flex_graph_llm": FlexibleGraphLLM,
            "flex_gelu_graph_llm": FlexibleGELUGraphLLM,
        }
    )
