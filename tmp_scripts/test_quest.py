import unittest
import mlx.core as mx
import numpy as np
import shutil
import os
from alayajet.features.quest.kv_cache import QuestController
from alayajet.features.quest.ops import append_kv, decode_estimate, decode_topk, decode_sparse_attn

class TestQuest(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_kv_cache"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
        self.num_layers = 1
        self.num_heads = 4
        self.head_dim = 16
        self.page_size = 16
        self.page_budget = 4
        self.max_seq_len = 128
        
        self.controller = QuestController(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            page_budget=self.page_budget,
            max_seq_len=self.max_seq_len,
            cache_dir=self.test_dir
        )
        
    def tearDown(self):
        # Clean up
        self.controller.clean_states()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
    def test_append_kv_and_metadata(self):
        # Simulate prefill of 40 tokens (2 full pages + 8 tokens)
        seq_len = 40
        self.controller.prepare_metadata(seq_len)
        
        k = mx.random.normal((seq_len, self.num_heads, self.head_dim))
        v = mx.random.normal((seq_len, self.num_heads, self.head_dim))
        
        append_kv(k, v, self.controller, layer_idx=0)
        
        # Check KV Cache
        self.assertEqual(self.controller.kv_cache.seq_len, seq_len)
        self.assertEqual(len(self.controller.kv_cache.active_indices), 3) # 16, 16, 8 -> 3 pages
        
        # Check Metadata Cache (RAM)
        # Check the first page metadata
        # It should be updated
        k_page_0 = k[:16]
        k_min_expected = mx.min(k_page_0, axis=0)
        
        # Metadata pool: (layers, capacity, 2, H, D)
        # Logical page 0 corresponds to physical index self.controller.kv_cache.active_indices[0]
        # BUT update_metadata uses logical page index as index into metadata_pool?
        # Let's check kv_cache.py: 
        # `self.metadata_pool[layer_idx, page_idx, ...]`
        # Yes, it uses page_idx (logical).
        
        k_min_stored = self.controller.metadata_pool[0, 0, 0] # Layer 0, Page 0, Min
        
        self.assertTrue(np.allclose(k_min_stored, np.array(k_min_expected), atol=1e-3))
        
        # Check Disk Persistence
        # First page should be on disk because we flushed it when appending 2nd and 3rd page
        phys_idx_0 = self.controller.kv_cache.active_indices[0]
        # Read from disk pool
        page_0_disk = self.controller.kv_cache.disk_pool[0, phys_idx_0] # (2, P, H, D)
        k_disk = page_0_disk[0]
        
        self.assertTrue(np.allclose(k_disk, np.array(k_page_0), atol=1e-3))
        
    def test_decode_estimate(self):
        # Fill cache first
        seq_len = 40
        self.controller.prepare_metadata(seq_len)
        k = mx.random.normal((seq_len, self.num_heads, self.head_dim))
        v = mx.random.normal((seq_len, self.num_heads, self.head_dim))
        append_kv(k, v, self.controller, layer_idx=0)
        
        # Prepare for decode
        self.controller.begin_forward(1)
        
        q = mx.random.normal((1, self.num_heads, self.head_dim))
        
        # Estimate
        scores = decode_estimate(q, self.controller, layer_idx=0)
        
        # Check shape: (H, num_pages_without_last)
        # 3 pages total, last one excluded -> 2 pages
        self.assertEqual(scores.shape, (self.num_heads, 2))
        
    def test_decode_sparse_attn(self):
        # Setup full pipeline
        seq_len = 48 # 3 full pages
        self.controller.prepare_metadata(seq_len)
        k = mx.random.normal((seq_len, self.num_heads, self.head_dim))
        v = mx.random.normal((seq_len, self.num_heads, self.head_dim))
        append_kv(k, v, self.controller, layer_idx=0)
        
        self.controller.begin_forward(1) # sets up without_last which has 2 pages (0, 1). Page 2 is last.
        
        q = mx.array(mx.random.normal((1, self.num_heads, self.head_dim)))
        
        # Estimate
        scores = decode_estimate(q, self.controller, layer_idx=0)
        
        # Select top 1 page out of 2 candidates
        topk_indices = decode_topk(scores, page_budget=1)
        
        # Attn
        out = decode_sparse_attn(q, topk_indices, self.controller, layer_idx=0)
        
        self.assertEqual(out.shape, (1, self.num_heads, self.head_dim))

if __name__ == '__main__':
    unittest.main()