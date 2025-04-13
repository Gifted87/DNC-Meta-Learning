import unittest
import torch
import sys
import os

# Add project root to path to import models and utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import cosine_similarity, outer_product, sparse_softmax_topk
# from models.memory_heads import MemoryReadHead, MemoryWriteHead # More complex tests

class TestMemoryUtils(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.num_keys = 1
        self.key_size = 8
        self.mem_slots = 10
        self.hidden_size = 16

        self.keys = torch.randn(self.batch_size, self.num_keys, self.key_size, device=self.device)
        self.memory_batched = torch.randn(self.batch_size, self.mem_slots, self.key_size, device=self.device)
        self.memory_shared = torch.randn(self.mem_slots, self.key_size, device=self.device)

        self.vec1 = torch.randn(self.batch_size, self.mem_slots, device=self.device)
        self.vec2 = torch.randn(self.batch_size, self.key_size, device=self.device)

        self.logits = torch.randn(self.batch_size, self.mem_slots, device=self.device)

    def test_cosine_similarity_batched(self):
        sim = cosine_similarity(self.keys, self.memory_batched)
        self.assertEqual(sim.shape, (self.batch_size, self.num_keys, self.mem_slots))
        self.assertTrue(torch.all(sim >= -1.01) and torch.all(sim <= 1.01)) # Allow for epsilon

        # Test self similarity (should be near 1)
        mem = self.memory_batched[:, 0:1, :] # Take first memory vector as key
        sim_self = cosine_similarity(mem, self.memory_batched)
        self.assertAlmostEqual(sim_self[0, 0, 0].item(), 1.0, delta=1e-5)

    def test_cosine_similarity_shared(self):
        sim = cosine_similarity(self.keys, self.memory_shared)
        self.assertEqual(sim.shape, (self.batch_size, self.num_keys, self.mem_slots))
        self.assertTrue(torch.all(sim >= -1.01) and torch.all(sim <= 1.01))

    def test_outer_product(self):
        outer = outer_product(self.vec1, self.vec2)
        self.assertEqual(outer.shape, (self.batch_size, self.mem_slots, self.key_size))
        # Check one element
        expected = self.vec1[0, 0] * self.vec2[0, 0]
        self.assertAlmostEqual(outer[0, 0, 0].item(), expected.item(), delta=1e-6)

    def test_sparse_softmax_topk(self):
        k = 3
        sparse_probs = sparse_softmax_topk(self.logits, k)
        self.assertEqual(sparse_probs.shape, self.logits.shape)
        # Check probabilities sum to 1
        self.assertTrue(torch.allclose(sparse_probs.sum(dim=-1), torch.ones(self.batch_size, device=self.device)))
        # Check that only k elements are non-zero (or near zero)
        num_non_zero = torch.sum(sparse_probs > 1e-6, dim=-1)
        self.assertTrue(torch.all(num_non_zero <= k))

        # Check edge case k=N
        sparse_probs_all = sparse_softmax_topk(self.logits, self.mem_slots)
        probs_all = F.softmax(self.logits, dim=-1)
        self.assertTrue(torch.allclose(sparse_probs_all, probs_all))

        # Check edge case k=1
        sparse_probs_one = sparse_softmax_topk(self.logits, 1)
        max_indices = torch.argmax(self.logits, dim=-1)
        self.assertTrue(torch.allclose(sparse_probs_one.sum(dim=-1), torch.ones(self.batch_size, device=self.device)))
        for b in range(self.batch_size):
             self.assertAlmostEqual(sparse_probs_one[b, max_indices[b]].item(), 1.0, delta=1e-6)

# TODO: Add tests for MemoryReadHead and MemoryWriteHead forward passes
# These are more involved as they require setting up mock inputs and states.

if __name__ == '__main__':
    unittest.main()
