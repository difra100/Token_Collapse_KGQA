import unittest

import torch

from graph_prompt_audit import GraphPromptTokens, compute_geometry


class GeometryTests(unittest.TestCase):
    def test_identical_single_tokens_collapse(self):
        values = torch.tensor([[[3.0, 4.0]], [[3.0, 4.0]], [[3.0, 4.0]]])
        prompts = GraphPromptTokens(values=values, mask=torch.ones(3, 1, dtype=torch.bool), slot_semantics=True)
        stats = compute_geometry(prompts, vocabulary_embeddings=torch.tensor([[3.0, 4.0], [6.0, 8.0]]))
        self.assertAlmostEqual(stats.pooled_inter_sample_cosine, 1.0, places=6)
        self.assertAlmostEqual(stats.pooled_residual_ratio, 0.0, places=6)
        self.assertAlmostEqual(stats.token_norm_ratio, 1.0, places=6)
        self.assertAlmostEqual(stats.pooled_residual_over_word_embedding, 0.0, places=6)
        self.assertIsNone(stats.within_example_cosine)
        self.assertAlmostEqual(stats.slotwise_inter_sample_cosine, 1.0, places=6)

    def test_mask_excludes_padding_and_slotwise_requires_semantics(self):
        values = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]],
            [[0.0, 1.0], [1.0, 0.0], [99.0, 99.0]],
        ])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        prompts = GraphPromptTokens(values=values, mask=mask, slot_semantics=False)
        stats = compute_geometry(prompts)
        self.assertAlmostEqual(stats.mean_tokens_per_example, 2.0)
        self.assertAlmostEqual(stats.pooled_inter_sample_cosine, 1.0, places=6)
        self.assertAlmostEqual(stats.within_example_cosine, 0.0, places=6)
        self.assertIsNone(stats.slotwise_inter_sample_cosine)


if __name__ == "__main__":
    unittest.main()
