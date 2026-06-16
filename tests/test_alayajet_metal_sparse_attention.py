import os
import tempfile
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from alayajet.features.quest.kv_cache import (
    BufferPool,
    MetalKVBufferPoolMetadata,
    QuestController,
    ResidentKVSlotPool,
)
from alayajet.features.quest.metal_sparse_attention import (
    fused_sparse_decode_attention,
    pack_kv_pages_head_major,
    pack_last_page_head_major,
    selected_direct_frame_decode_attention,
    selected_direct_frame_decode_attention_with_last_frame,
    selected_frame_decode_attention_with_last_frame,
    selected_page_decode_attention,
    selected_frame_decode_attention,
    selected_indexed_page_decode_attention,
)
from alayajet.features.quest.ops import (
    _load_selected_frame_pages,
    _select_cpu_union_candidates,
    _use_cpu_union_page_selection,
    append_kv,
    decode_sparse_attn,
)


def _reference_sparse_decode(q, k_min, k_max, kv_pages, last_k, last_v, page_budget, scale):
    q = np.asarray(q, dtype=np.float32)
    k_min = np.asarray(k_min, dtype=np.float32)
    k_max = np.asarray(k_max, dtype=np.float32)
    kv_pages = np.asarray(kv_pages, dtype=np.float32)
    last_k = np.asarray(last_k, dtype=np.float32)
    last_v = np.asarray(last_v, dtype=np.float32)

    hq, head_dim = q.shape
    hkv, num_pages, _ = k_min.shape
    group_size = hq // hkv
    selected_count = min(page_budget, num_pages)
    out = np.zeros((hq, head_dim), dtype=np.float32)

    for h in range(hq):
        kv_h = h // group_size
        qh = q[h]
        estimate = np.sum(
            np.where(qh[None, :] >= 0.0, k_max[kv_h], k_min[kv_h]) * qh[None, :],
            axis=-1,
        )
        if selected_count:
            selected = np.argpartition(estimate, -selected_count)[-selected_count:]
            selected = np.sort(selected)
        else:
            selected = np.empty((0,), dtype=np.int64)

        keys = []
        values = []
        for page in selected:
            keys.append(kv_pages[page, 0, :, kv_h, :])
            values.append(kv_pages[page, 1, :, kv_h, :])
        if last_k.shape[0]:
            keys.append(last_k[:, kv_h, :])
            values.append(last_v[:, kv_h, :])
        if not keys:
            continue
        k_full = np.concatenate(keys, axis=0)
        v_full = np.concatenate(values, axis=0)
        logits = (k_full @ qh) * scale
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        out[h] = probs @ v_full
    return out


def _reference_selected_page_decode(q, selected_k, selected_v, last_k, last_v, scale):
    q = np.asarray(q, dtype=np.float32)
    selected_k = np.asarray(selected_k, dtype=np.float32)
    selected_v = np.asarray(selected_v, dtype=np.float32)
    last_k = np.asarray(last_k, dtype=np.float32)
    last_v = np.asarray(last_v, dtype=np.float32)
    num_heads, head_dim = q.shape
    out = np.zeros((num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        keys = []
        values = []
        if selected_k.shape[1]:
            keys.append(selected_k[h])
            values.append(selected_v[h])
        if last_k.shape[1]:
            keys.append(last_k[h])
            values.append(last_v[h])
        if not keys:
            continue
        k_full = np.concatenate(keys, axis=0)
        v_full = np.concatenate(values, axis=0)
        logits = (k_full @ q[h]) * scale
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        out[h] = probs @ v_full
    return out


def _reference_selected_indexed_page_decode(
    q,
    selected_pages,
    kv_pages,
    last_k,
    last_v,
    scale,
    *,
    kv_layout="page_major",
    last_layout="token_major",
):
    q = np.asarray(q, dtype=np.float32)
    selected_pages = np.asarray(selected_pages, dtype=np.int32)
    kv_pages = np.asarray(kv_pages, dtype=np.float32)
    last_k = np.asarray(last_k, dtype=np.float32)
    last_v = np.asarray(last_v, dtype=np.float32)
    num_heads, head_dim = q.shape
    if kv_layout == "page_major":
        num_kv_heads = kv_pages.shape[3]
    elif kv_layout == "head_major":
        num_kv_heads = kv_pages.shape[0]
    else:
        raise ValueError(kv_layout)
    group_size = num_heads // num_kv_heads
    out = np.zeros((num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        kv_h = h // group_size
        keys = []
        values = []
        for page in selected_pages[h]:
            if kv_layout == "page_major":
                keys.append(kv_pages[page, 0, :, kv_h, :])
                values.append(kv_pages[page, 1, :, kv_h, :])
            else:
                keys.append(kv_pages[kv_h, page, 0, :, :])
                values.append(kv_pages[kv_h, page, 1, :, :])
        last_len = last_k.shape[0] if last_layout == "token_major" else last_k.shape[1]
        if last_len:
            if last_layout == "token_major":
                keys.append(last_k[:, kv_h, :])
                values.append(last_v[:, kv_h, :])
            elif last_layout == "head_major":
                keys.append(last_k[kv_h, :, :])
                values.append(last_v[kv_h, :, :])
            else:
                raise ValueError(last_layout)
        if not keys:
            continue
        k_full = np.concatenate(keys, axis=0)
        v_full = np.concatenate(values, axis=0)
        logits = (k_full @ q[h]) * scale
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        out[h] = probs @ v_full
    return out


def _reference_selected_frame_decode(
    q,
    selected_frames,
    frames,
    last_k,
    last_v,
    scale,
):
    q = np.asarray(q, dtype=np.float32)
    selected_frames = np.asarray(selected_frames, dtype=np.int32)
    frames = np.asarray(frames, dtype=np.float32)
    last_k = np.asarray(last_k, dtype=np.float32)
    last_v = np.asarray(last_v, dtype=np.float32)
    num_heads, head_dim = q.shape
    num_kv_heads = last_k.shape[0]
    group_size = num_heads // num_kv_heads
    out = np.zeros((num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        kv_h = h // group_size
        keys = []
        values = []
        for frame in selected_frames[h]:
            keys.append(frames[frame, 0])
            values.append(frames[frame, 1])
        if last_k.shape[1]:
            keys.append(last_k[kv_h])
            values.append(last_v[kv_h])
        if not keys:
            continue
        k_full = np.concatenate(keys, axis=0)
        v_full = np.concatenate(values, axis=0)
        logits = (k_full @ q[h]) * scale
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        out[h] = probs @ v_full
    return out


class TestAlayaJetCpuPageSelection(unittest.TestCase):
    def test_cpu_union_candidates_cover_per_head_topk(self):
        rng = np.random.default_rng(17)
        num_heads = 6
        num_kv_heads = 2
        group_size = num_heads // num_kv_heads
        num_pages = 11
        head_dim = 16
        page_budget = 3

        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        metadata = rng.normal(
            size=(num_pages, 2, num_kv_heads, head_dim)
        ).astype(np.float32)
        lo = np.minimum(metadata[:, 0], metadata[:, 1])
        hi = np.maximum(metadata[:, 0], metadata[:, 1])
        metadata[:, 0] = lo
        metadata[:, 1] = hi

        union = _select_cpu_union_candidates(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            metadata,
            page_budget,
            group_size,
        )

        for head_idx, q_head in enumerate(q.astype(np.float32)):
            kv_head = head_idx // group_size
            q_pos = np.maximum(q_head, 0.0)
            q_neg = np.minimum(q_head, 0.0)
            scores = (
                q_pos * metadata[:, 1, kv_head, :]
                + q_neg * metadata[:, 0, kv_head, :]
            ).sum(axis=-1)
            top = np.argpartition(scores, -page_budget)[-page_budget:]
            self.assertTrue(set(top.tolist()).issubset(set(union.tolist())))
        np.testing.assert_array_equal(union, np.sort(union))

    def test_cpu_union_auto_is_conservative_for_current_sweep_sizes(self):
        self.assertFalse(
            _use_cpu_union_page_selection(
                "auto",
                num_pages=512,
                num_heads=32,
                page_budget=8,
            )
        )
        self.assertTrue(
            _use_cpu_union_page_selection(
                "auto",
                num_pages=5000,
                num_heads=32,
                page_budget=8,
            )
        )


@unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
class TestAlayaJetMlxHostCopyBoundary(unittest.TestCase):
    def test_mx_array_from_numpy_does_not_track_host_mutation(self):
        frame = np.zeros((2, 3), dtype=np.float16)
        frame_mx = mx.array(frame)
        frame[...] = 7
        mx.eval(frame_mx)
        np.testing.assert_array_equal(
            np.asarray(frame_mx),
            np.zeros_like(frame),
        )


class TestMetalKVBufferPoolMetadata(unittest.TestCase):
    def test_basic_allocation_fills_fixed_slots(self):
        metadata = MetalKVBufferPoolMetadata(num_slots=2, page_size=32)

        slot_a, evicted_a = metadata.allocate("A")
        slot_b, evicted_b = metadata.allocate("B")

        self.assertIsNone(evicted_a)
        self.assertIsNone(evicted_b)
        self.assertEqual(metadata.page_to_slot, {"A": slot_a, "B": slot_b})
        self.assertEqual(metadata.slot_to_page[slot_a], "A")
        self.assertEqual(metadata.slot_to_page[slot_b], "B")
        self.assertEqual(metadata.free_slots, [])

    def test_repeated_allocation_reuses_slot_and_updates_lru(self):
        metadata = MetalKVBufferPoolMetadata(num_slots=2, page_size=32)

        slot_a, _ = metadata.allocate("A")
        metadata.allocate("B")
        repeated_slot_a, evicted = metadata.allocate("A")

        self.assertEqual(repeated_slot_a, slot_a)
        self.assertIsNone(evicted)
        self.assertEqual(len(metadata.page_to_slot), 2)
        self.assertEqual(list(metadata.recent_pages.keys()), ["B", "A"])

    def test_lru_eviction_reuses_least_recent_slot(self):
        metadata = MetalKVBufferPoolMetadata(num_slots=2, page_size=32)

        metadata.allocate("A")
        slot_b, _ = metadata.allocate("B")
        metadata.touch("A")
        slot_c, evicted = metadata.allocate("C")

        self.assertIsNotNone(evicted)
        self.assertEqual(evicted.page_id, "B")
        self.assertEqual(evicted.slot_id, slot_b)
        self.assertEqual(slot_c, slot_b)
        self.assertTrue(metadata.has_page("A"))
        self.assertFalse(metadata.has_page("B"))
        self.assertTrue(metadata.has_page("C"))

    def test_valid_length_is_bounded_by_page_size(self):
        metadata = MetalKVBufferPoolMetadata(num_slots=2, page_size=32)

        metadata.allocate("A", valid_length=10)
        self.assertEqual(metadata.get_valid_length("A"), 10)
        metadata.set_valid_length("A", 20)
        self.assertEqual(metadata.get_valid_length("A"), 20)
        with self.assertRaises(ValueError):
            metadata.set_valid_length("A", 33)
        with self.assertRaises(ValueError):
            metadata.allocate("B", valid_length=40)

    def test_dirty_recent_pages_reported_on_eviction(self):
        metadata = MetalKVBufferPoolMetadata(num_slots=1, page_size=32)

        metadata.allocate("A", dirty=True)
        evicted_dirty = metadata.evict("A")
        self.assertIsNotNone(evicted_dirty)
        self.assertTrue(evicted_dirty.dirty)

        metadata.allocate("A", dirty=True)
        metadata.mark_clean("A")
        evicted_clean = metadata.evict("A")
        self.assertIsNotNone(evicted_clean)
        self.assertFalse(evicted_clean.dirty)

    def test_capacity_stays_fixed_under_many_allocations(self):
        metadata = MetalKVBufferPoolMetadata(num_slots=3, page_size=32)

        for page_idx in range(20):
            slot_id, _ = metadata.allocate(("layer", page_idx))
            self.assertGreaterEqual(slot_id, 0)
            self.assertLess(slot_id, metadata.num_slots)
            self.assertLessEqual(len(metadata.page_to_slot), metadata.num_slots)
            self.assertLessEqual(len(metadata.recent_pages), metadata.num_slots)

        self.assertTrue(
            all(0 <= slot_id < metadata.num_slots for slot_id in metadata.page_to_slot.values())
        )


class TestAlayaJetSelectedPageBufferPool(unittest.TestCase):
    def _seed_disk_page(self, controller, layer_idx, page_idx, value):
        frame = np.full(
            (2, controller.page_size, controller.head_dim),
            value,
            dtype=controller.kv_cache.dtype,
        )
        for kv_head in range(controller.num_kv_heads):
            controller.kv_cache.buffer_pool.write_page_direct(
                (layer_idx, page_idx, kv_head),
                frame,
            )

    def _prepare_resident_decode_case(
        self,
        tmpdir,
        *,
        page_indices,
        buffer_pool_pages=2,
        max_seq_len=64,
        num_heads=2,
        num_kv_heads=1,
        page_size=2,
        head_dim=4,
    ):
        controller = QuestController(
            num_layers=1,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            page_budget=len(page_indices),
            max_seq_len=max_seq_len,
            cache_dir=tmpdir,
            buffer_pool_pages=buffer_pool_pages,
        )
        for i, page_idx in enumerate(page_indices):
            self._seed_disk_page(controller, 0, page_idx, i + 1)
        last_page_idx = max(page_indices) + 1
        controller.kv_cache.active_indices = list(page_indices) + [last_page_idx]
        controller.kv_cache.seq_len = len(page_indices) * page_size + 1
        controller.kv_indices_without_last = list(range(len(page_indices)))
        controller.kv_indices_with_last = list(range(len(page_indices) + 1))
        controller.resident_frame_pool.ensure_decode_page_slot(0, last_page_idx)
        controller.resident_frame_pool.last_page_valid_length[0] = 1
        return controller

    def _run_resident_decode(self, controller, topk_indices, captured=None):
        q = mx.ones((1, controller.num_heads, 1, controller.head_dim), dtype=mx.float16)

        def fake_attention(q, selected_frames_mx, frames_mx, **kwargs):
            if captured is not None:
                mx.eval(selected_frames_mx)
                captured.append(np.asarray(selected_frames_mx).copy())
            return mx.zeros_like(q)

        with mock.patch.dict(
            os.environ,
            {
                "ALAYAJET_QUEST_METAL_SELECTED_PAGE": "1",
                "ALAYAJET_QUEST_METAL_SELECTED_FRAME": "resident",
            },
            clear=False,
        ), mock.patch(
            "alayajet.features.quest.ops.selected_frame_decode_attention_with_stable_arena",
            side_effect=fake_attention,
        ):
            return decode_sparse_attn(
                q,
                mx.array(np.asarray(topk_indices, dtype=np.int64)),
                controller,
                layer_idx=0,
            )

    def test_selected_page_stats_and_hot_eviction_bias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=2,
                head_dim=4,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=2,
            )
            try:
                frame = np.zeros((2, 2, 4), dtype=np.float16)
                page0 = (0, 0, 0)
                page1 = (0, 1, 0)
                page2 = (0, 2, 0)
                page3 = (0, 3, 0)
                pool.insert_page_cache(page0, frame)
                pool.insert_page_cache(page1, frame + 1)
                pool.mark_selected_pages([page0])
                hits, misses = pool.pop_selected_page_stats()
                self.assertEqual((hits, misses), (1, 0))

                pool.insert_page_cache(page2, frame + 2)
                self.assertTrue(pool.has_page(page0))
                self.assertFalse(pool.has_page(page1))
                self.assertTrue(pool.has_page(page2))

                pool.mark_selected_pages([page3])
                hits, misses = pool.pop_selected_page_stats()
                self.assertEqual((hits, misses), (0, 1))
            finally:
                pool.close()

    def test_layer_frames_cache_updates_on_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=2,
                head_dim=4,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=2,
            )
            try:
                page0 = (0, 0, 0)
                frame_a = np.ones((2, 2, 4), dtype=np.float16)
                frame_b = np.full((2, 2, 4), 3, dtype=np.float16)
                pool.insert_page_cache(page0, frame_a)
                layer = pool.get_layer_frames_mx(0)
                mx.eval(layer)
                frame_idx = pool.get_frame_index(page0)
                local_idx = pool.layer_local_frame_index(0, frame_idx)
                np.testing.assert_array_equal(np.asarray(layer[local_idx]), frame_a)

                pool.write_page(page0, frame_b)
                layer = pool.get_layer_frames_mx(0)
                mx.eval(layer)
                np.testing.assert_array_equal(np.asarray(layer[local_idx]), frame_b)
            finally:
                pool.close()

    def test_layer_frames_cache_syncs_only_required_slots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=2,
                head_dim=4,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=3,
            )
            try:
                page0 = (0, 0, 0)
                page1 = (0, 1, 0)
                frame_a = np.ones((2, 2, 4), dtype=np.float16)
                frame_b = np.full((2, 2, 4), 5, dtype=np.float16)
                pool.insert_page_cache(page0, frame_a)
                pool.insert_page_cache(page1, frame_b)
                frame0 = pool.get_frame_index(page0)
                frame1 = pool.get_frame_index(page1)
                local0 = pool.layer_local_frame_index(0, frame0)
                local1 = pool.layer_local_frame_index(0, frame1)

                layer = pool.get_layer_frames_mx(
                    0,
                    required_frame_indices=[frame0],
                )
                mx.eval(layer)
                np.testing.assert_array_equal(np.asarray(layer[local0]), frame_a)
                np.testing.assert_array_equal(
                    np.asarray(layer[local1]),
                    np.zeros_like(frame_b),
                )

                layer = pool.get_layer_frames_mx(
                    0,
                    required_frame_indices=[frame1],
                )
                mx.eval(layer)
                np.testing.assert_array_equal(np.asarray(layer[local1]), frame_b)
            finally:
                pool.close()

    def test_hot_selected_frame_arena_reuses_slots_and_invalidates_on_evict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=2,
                head_dim=4,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=2,
            )
            try:
                page0 = (0, 0, 0)
                page1 = (0, 1, 0)
                page2 = (0, 2, 0)
                frame0 = np.ones((2, 2, 4), dtype=np.float16)
                frame1 = np.full((2, 2, 4), 2, dtype=np.float16)
                frame2 = np.full((2, 2, 4), 3, dtype=np.float16)
                pool.insert_page_cache(page0, frame0)
                pool.insert_page_cache(page1, frame1)
                idx0 = pool.get_frame_index(page0)
                idx1 = pool.get_frame_index(page1)

                slots_a, arena_a = pool.get_hot_selected_frames_mx(
                    0,
                    [page0, page1],
                    [idx0, idx1],
                )
                slots_b, arena_b = pool.get_hot_selected_frames_mx(
                    0,
                    [page1, page0],
                    [idx1, idx0],
                )
                self.assertIs(arena_a, arena_b)
                self.assertEqual(int(slots_a[0]), int(slots_b[1]))
                self.assertEqual(int(slots_a[1]), int(slots_b[0]))

                pool.insert_page_cache(page2, frame2)
                self.assertFalse(pool.has_page(page0))
                idx2 = pool.get_frame_index(page2)
                slots_c, _ = pool.get_hot_selected_frames_mx(
                    0,
                    [page2],
                    [idx2],
                )
                mx.eval(arena_a)
                np.testing.assert_array_equal(
                    np.asarray(arena_a[int(slots_c[0])]),
                    frame2,
                )
            finally:
                pool.close()

    def test_resident_frame_arena_accepts_mlx_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=4,
                head_dim=8,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=2,
            )
            try:
                page0 = (0, 0, 0)
                k = mx.ones((2, 8), dtype=mx.float16)
                v = mx.full((2, 8), 3, dtype=mx.float16)
                frame_idx = pool.write_kv_slice_mx_resident(
                    page_id=page0,
                    page_offset=1,
                    k_slice=k,
                    v_slice=v,
                    assume_zero=True,
                )
                arena = pool.get_resident_frames_mx([frame_idx])
                mx.eval(arena)
                np.testing.assert_array_equal(
                    np.asarray(arena[0, 0, 1:3]),
                    np.ones((2, 8), dtype=np.float16),
                )
                np.testing.assert_array_equal(
                    np.asarray(arena[0, 1, 1:3]),
                    np.full((2, 8), 3, dtype=np.float16),
                )
            finally:
                pool.close()

    def test_resident_frame_arena_rebinds_after_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=4,
                head_dim=8,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=2,
            )
            try:
                page0 = (0, 0, 0)
                k0 = mx.full((2, 8), 2, dtype=mx.float16)
                v0 = mx.full((2, 8), 5, dtype=mx.float16)
                frame_idx = pool.write_kv_slice_mx_resident(
                    page_id=page0,
                    page_offset=0,
                    k_slice=k0,
                    v_slice=v0,
                    assume_zero=True,
                )
                arena = pool.get_resident_frames_mx([frame_idx])
                mx.eval(arena)
                np.testing.assert_array_equal(
                    np.asarray(arena[0, 0, :2]),
                    np.full((2, 8), 2, dtype=np.float16),
                )
                np.testing.assert_array_equal(
                    np.asarray(arena[0, 1, :2]),
                    np.full((2, 8), 5, dtype=np.float16),
                )
            finally:
                pool.close()

    def test_resident_frame_selection_returns_direct_buffers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
                head_dim=4,
                page_size=2,
                page_budget=2,
                max_seq_len=8,
                cache_dir=tmpdir,
                buffer_pool_pages=2,
            )
            try:
                controller.kv_cache.write_kv_slice_mx_resident(
                    layer_idx=0,
                    page_idx=0,
                    page_offset=0,
                    k_mx=mx.ones((2, 1, 4), dtype=mx.float16),
                    v_mx=mx.full((2, 1, 4), 5, dtype=mx.float16),
                    assume_zero=True,
                )
                selected_frames, frames = controller.kv_cache.load_selected_resident_frame_indices_mx(
                    layer_idx=0,
                    physical_indices=np.array([[0, 0], [0, 0]], dtype=np.int64),
                    kv_head_indices=np.array([0, 0], dtype=np.int64),
                    dtype=mx.float16,
                )
                self.assertEqual(selected_frames.shape, (2, 2))
                self.assertIsInstance(frames, list)
                selected_slot = int(np.asarray(selected_frames)[0, 0])
                self.assertGreaterEqual(len(frames), selected_slot + 1)
                mx.eval(frames[selected_slot])
                np.testing.assert_array_equal(
                    np.asarray(frames[selected_slot][0, :2]),
                    np.ones((2, 4), dtype=np.float16),
                )
            finally:
                controller.clean_states()

    def test_resident_selected_miss_reads_disk_without_buffer_pool_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self._prepare_resident_decode_case(tmpdir, page_indices=[0])
            try:
                with mock.patch.object(
                    BufferPool,
                    "get_page",
                    side_effect=AssertionError("BufferPool.get_page called"),
                ) as get_page, mock.patch.object(
                    BufferPool,
                    "get_page_mx",
                    side_effect=AssertionError("BufferPool.get_page_mx called"),
                ) as get_page_mx, mock.patch.object(
                    BufferPool,
                    "get_frame_index",
                    side_effect=AssertionError("BufferPool.get_frame_index called"),
                ) as get_frame_index, mock.patch.object(
                    controller.kv_cache,
                    "load_page_mx",
                    side_effect=AssertionError("load_page_mx called"),
                ) as load_page_mx, mock.patch.object(
                    controller.kv_cache,
                    "read_page_mx_from_disk",
                    wraps=controller.kv_cache.read_page_mx_from_disk,
                ) as read_page_mx_from_disk:
                    self._run_resident_decode(
                        controller,
                        np.zeros((controller.num_heads, 1), dtype=np.int64),
                    )

                get_page.assert_not_called()
                get_page_mx.assert_not_called()
                get_frame_index.assert_not_called()
                load_page_mx.assert_not_called()
                read_page_mx_from_disk.assert_called_once_with(
                    0,
                    0,
                    dtype=mx.float16,
                )
                for kv_head in range(controller.num_kv_heads):
                    frame_id = controller.resident_frame_pool.frame_id_if_present(
                        0,
                        0,
                        kv_head,
                    )
                    self.assertIsNotNone(frame_id)
                    meta = controller.resident_frame_pool.meta[int(frame_id)]
                    self.assertFalse(meta.dirty)
                    self.assertEqual(meta.version, 0)
            finally:
                controller.clean_states()

    def test_resident_selected_hit_reuses_slot_without_disk_or_buffer_pool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self._prepare_resident_decode_case(tmpdir, page_indices=[0])
            try:
                captured = []
                with mock.patch.object(
                    controller.kv_cache,
                    "read_page_mx_from_disk",
                    wraps=controller.kv_cache.read_page_mx_from_disk,
                ) as read_page_mx_from_disk:
                    self._run_resident_decode(
                        controller,
                        np.zeros((controller.num_heads, 1), dtype=np.int64),
                        captured,
                    )
                self.assertEqual(read_page_mx_from_disk.call_count, 1)

                with mock.patch.object(
                    BufferPool,
                    "get_page",
                    side_effect=AssertionError("BufferPool.get_page called"),
                ), mock.patch.object(
                    BufferPool,
                    "get_page_mx",
                    side_effect=AssertionError("BufferPool.get_page_mx called"),
                ), mock.patch.object(
                    BufferPool,
                    "get_frame_index",
                    side_effect=AssertionError("BufferPool.get_frame_index called"),
                ), mock.patch.object(
                    controller.kv_cache,
                    "load_page_mx",
                    side_effect=AssertionError("load_page_mx called"),
                ), mock.patch.object(
                    controller.kv_cache,
                    "read_page_mx_from_disk",
                    side_effect=AssertionError("read_page_mx_from_disk called"),
                ):
                    self._run_resident_decode(
                        controller,
                        np.zeros((controller.num_heads, 1), dtype=np.int64),
                        captured,
                    )

                self.assertEqual(captured[0][:, 0].tolist(), captured[1][:, 0].tolist())
            finally:
                controller.clean_states()

    def test_resident_selected_slot_lru_evicts_least_recent_history_page(self):
        pool = ResidentKVSlotPool(
            num_layers=1,
            num_heads=1,
            max_blocks=8,
            slot_blocks=2,
            page_size=2,
            head_dim=4,
            mx_dtype=mx.float16,
        )

        pool.ensure_page_slot(0, 0)
        pool.ensure_page_slot(0, 1)
        pool.ensure_page_slot(0, 0)
        pool.ensure_page_slot(0, 2)

        self.assertTrue(pool.has_page(0, 0))
        self.assertFalse(pool.has_page(0, 1))
        self.assertTrue(pool.has_page(0, 2))

    def test_resident_selected_kernel_receives_physical_slot_ids(self):
        logical_page = 1024
        page_size = 2
        num_kv_heads = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self._prepare_resident_decode_case(
                tmpdir,
                page_indices=[logical_page],
                buffer_pool_pages=2,
                max_seq_len=(logical_page + 2) * page_size,
                num_heads=num_kv_heads,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                head_dim=4,
            )
            try:
                captured = []
                self._run_resident_decode(
                    controller,
                    np.zeros((controller.num_heads, 1), dtype=np.int64),
                    captured,
                )
                max_kernel_id = int(captured[0].max())
                self.assertLess(
                    max_kernel_id,
                    controller.resident_frame_pool.slot_blocks * num_kv_heads,
                )
            finally:
                controller.clean_states()

    def test_resident_storage_tracks_frame_slot_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=1,
                page_size=4,
                head_dim=8,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=2,
            )
            try:
                page0 = (0, 0, 0)
                frame_idx = pool.write_kv_slice_mx_resident(
                    page_id=page0,
                    page_offset=1,
                    k_slice=mx.ones((2, 8), dtype=mx.float16),
                    v_slice=mx.full((2, 8), 3, dtype=mx.float16),
                    assume_zero=True,
                )
                slot = pool.metal_kv_storage.slot_metadata(frame_idx)
                self.assertEqual(slot.page_id, page0)
                self.assertEqual(slot.valid_length, 3)
                self.assertEqual(slot.version, pool.frame_version[frame_idx])
                self.assertIs(slot.buffer, pool.resident_frames_mx[frame_idx])
            finally:
                pool.close()

    def test_resident_batched_write_updates_all_kv_heads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                page_size=4,
                page_budget=2,
                max_seq_len=8,
                cache_dir=tmpdir,
                buffer_pool_pages=4,
            )
            try:
                k = mx.array(
                    np.stack(
                        [
                            np.full((2, 8), 2, dtype=np.float16),
                            np.full((2, 8), 5, dtype=np.float16),
                        ],
                        axis=1,
                    )
                )
                v = mx.array(
                    np.stack(
                        [
                            np.full((2, 8), 7, dtype=np.float16),
                            np.full((2, 8), 11, dtype=np.float16),
                        ],
                        axis=1,
                    )
                )
                controller.kv_cache.write_kv_slice_mx_resident(
                    layer_idx=0,
                    page_idx=0,
                    page_offset=1,
                    k_mx=k,
                    v_mx=v,
                    assume_zero=True,
                )
                for kv_head, expected_k, expected_v in [(0, 2, 7), (1, 5, 11)]:
                    frame_idx = controller.kv_cache.buffer_pool.get_frame_index(
                        (0, 0, kv_head)
                    )
                    frame = controller.kv_cache.buffer_pool.get_resident_frame_buffers_mx(
                        [frame_idx],
                        dtype=mx.float16,
                    )[0]
                    mx.eval(frame)
                    np.testing.assert_array_equal(
                        np.asarray(frame[0, 1:3]),
                        np.full((2, 8), expected_k, dtype=np.float16),
                    )
                    np.testing.assert_array_equal(
                        np.asarray(frame[1, 1:3]),
                        np.full((2, 8), expected_v, dtype=np.float16),
                    )
            finally:
                controller.clean_states()

    def test_append_kv_writes_active_page_to_resident_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                head_dim=8,
                page_size=4,
                page_budget=2,
                max_seq_len=8,
                cache_dir=tmpdir,
                buffer_pool_pages=4,
            )
            try:
                controller.kv_cache.append_seq(1)
                k = mx.ones((1, 2, 8), dtype=mx.float16)
                v = mx.full((1, 2, 8), 3, dtype=mx.float16)
                with mock.patch.dict(
                    "os.environ",
                    {"ALAYAJET_QUEST_RESIDENT_FRAME_WRITE": "1"},
                    clear=False,
                ):
                    append_kv(k, v, controller, layer_idx=0)
                page_idx = controller.kv_cache.active_indices[-1]
                for kv_head in range(2):
                    frame_id = controller.resident_frame_pool.decode_frame_id_if_present(
                        0,
                        page_idx,
                        kv_head,
                    )
                    self.assertIsNotNone(frame_id)
                    meta = controller.resident_frame_pool.meta[int(frame_id)]
                    self.assertEqual(meta.valid_length, 1)
                    self.assertEqual(meta.layer_idx, 0)
                    self.assertEqual(meta.kv_head, kv_head)
                    self.assertTrue(meta.dirty)
            finally:
                controller.clean_states()

    def test_decode_rolling_slots_flush_in_two_page_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=1,
                num_kv_heads=1,
                head_dim=4,
                page_size=64,
                page_budget=2,
                max_seq_len=512,
                cache_dir=tmpdir,
                buffer_pool_pages=3,
            )
            try:
                slot_by_decode_page = []
                valid_lengths = {}
                with mock.patch.dict(
                    "os.environ",
                    {"ALAYAJET_QUEST_RESIDENT_FRAME_WRITE": "1"},
                    clear=False,
                ), mock.patch.object(
                    controller.kv_cache,
                    "_write_decode_rolling_buffer_page",
                    wraps=controller.kv_cache._write_decode_rolling_buffer_page,
                ) as write_page:
                    for token_idx in range(512):
                        controller.kv_cache.append_seq(1)
                        k = mx.full((1, 1, 4), token_idx + 1, dtype=mx.float16)
                        v = mx.full((1, 1, 4), token_idx + 3, dtype=mx.float16)
                        append_kv(k, v, controller, layer_idx=0)
                        page_idx = controller.kv_cache.active_indices[-1]
                        page_offset = token_idx % 64
                        slot_by_decode_page.append(
                            controller.resident_frame_pool.decode_page_to_slot[0][page_idx]
                        )
                        valid_lengths[token_idx + 1] = (
                            controller.resident_frame_pool.last_page_valid_length[0]
                        )
                        self.assertLessEqual(
                            len(controller.resident_frame_pool.decode_page_to_slot[0]),
                            2,
                        )
                        self.assertLessEqual(
                            len(controller.kv_cache.decode_rolling_buffers_mx[0]),
                            2,
                        )
                        self.assertEqual(
                            controller.resident_frame_pool.last_page_valid_length[0],
                            page_offset + 1,
                        )

                self.assertEqual(valid_lengths[1], 1)
                self.assertEqual(valid_lengths[64], 64)
                self.assertEqual(valid_lengths[65], 1)
                self.assertEqual(set(slot_by_decode_page), {0, 1})
                self.assertEqual(slot_by_decode_page[0], slot_by_decode_page[128])
                self.assertEqual(slot_by_decode_page[64], slot_by_decode_page[192])
                self.assertEqual(controller.kv_cache.decode_rolling_flush_batches, 4)
                self.assertEqual(write_page.call_count, 8)
                page0 = controller.kv_cache.active_indices[0]
                page0_mx = controller.kv_cache.read_page_mx_from_disk(0, page0)
                mx.eval(page0_mx)
                np.testing.assert_array_equal(
                    np.asarray(page0_mx[0, 0, 0]),
                    np.full((4,), 1, dtype=np.float16),
                )
                self.assertLessEqual(
                    controller.resident_frame_pool.layer_arena_blocks[0],
                    controller.resident_frame_pool.slot_blocks,
                )
            finally:
                controller.clean_states()

    def test_decode_rolling_flush_count_is_not_per_token_or_per_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=1,
                num_kv_heads=1,
                head_dim=4,
                page_size=64,
                page_budget=2,
                max_seq_len=256,
                cache_dir=tmpdir,
                buffer_pool_pages=3,
            )
            try:
                with mock.patch.dict(
                    "os.environ",
                    {"ALAYAJET_QUEST_RESIDENT_FRAME_WRITE": "1"},
                    clear=False,
                ):
                    for token_idx in range(128):
                        controller.kv_cache.append_seq(1)
                        k = mx.full((1, 1, 4), token_idx + 1, dtype=mx.float16)
                        v = mx.full((1, 1, 4), token_idx + 3, dtype=mx.float16)
                        append_kv(k, v, controller, layer_idx=0)
                    self.assertEqual(controller.kv_cache.decode_rolling_flush_batches, 1)

                    for token_idx in range(128, 256):
                        controller.kv_cache.append_seq(1)
                        k = mx.full((1, 1, 4), token_idx + 1, dtype=mx.float16)
                        v = mx.full((1, 1, 4), token_idx + 3, dtype=mx.float16)
                        append_kv(k, v, controller, layer_idx=0)
                    self.assertEqual(controller.kv_cache.decode_rolling_flush_batches, 2)
            finally:
                controller.clean_states()

    def test_decode_rolling_at_64k_keeps_resident_arena_fixed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            page_size = 64
            prompt_len = 64 * 1024
            controller = QuestController(
                num_layers=1,
                num_heads=1,
                num_kv_heads=1,
                head_dim=1,
                page_size=page_size,
                page_budget=2,
                max_seq_len=prompt_len + 512,
                cache_dir=tmpdir,
                buffer_pool_pages=3,
            )
            try:
                prompt_pages = prompt_len // page_size
                controller.kv_cache.active_indices = list(range(prompt_pages))
                controller.kv_cache.free_slots = set(
                    range(prompt_pages, controller.kv_cache.capacity)
                )
                controller.kv_cache.seq_len = prompt_len
                with mock.patch.dict(
                    "os.environ",
                    {"ALAYAJET_QUEST_RESIDENT_FRAME_WRITE": "1"},
                    clear=False,
                ):
                    for token_idx in range(512):
                        controller.kv_cache.append_seq(1)
                        k = mx.full((1, 1, 1), token_idx + 1, dtype=mx.float16)
                        v = mx.full((1, 1, 1), token_idx + 3, dtype=mx.float16)
                        append_kv(k, v, controller, layer_idx=0)
                        self.assertLessEqual(
                            len(controller.resident_frame_pool.decode_page_to_slot[0]),
                            2,
                        )
                self.assertEqual(controller.kv_cache.decode_rolling_flush_batches, 4)
                self.assertEqual(
                    controller.resident_frame_pool.layer_arena_blocks[0],
                    controller.resident_frame_pool.slot_blocks,
                )
                self.assertEqual(controller.resident_frame_pool.slot_blocks, 3)
            finally:
                controller.clean_states()

    def test_resident_attention_reads_rolling_page_without_disk_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=1,
                num_kv_heads=1,
                head_dim=4,
                page_size=64,
                page_budget=1,
                max_seq_len=128,
                cache_dir=tmpdir,
                buffer_pool_pages=3,
            )
            try:
                with mock.patch.dict(
                    "os.environ",
                    {"ALAYAJET_QUEST_RESIDENT_FRAME_WRITE": "1"},
                    clear=False,
                ):
                    for token_idx in range(128):
                        controller.kv_cache.append_seq(1)
                        k = mx.full((1, 1, 4), token_idx + 1, dtype=mx.float16)
                        v = mx.full((1, 1, 4), token_idx + 3, dtype=mx.float16)
                        append_kv(k, v, controller, layer_idx=0)
                controller.begin_forward(1)
                q = mx.ones((1, 1, 1, 4), dtype=mx.float16)

                def fake_attention(q, selected_frames_mx, frames_mx, **kwargs):
                    return mx.zeros_like(q)

                with mock.patch.dict(
                    "os.environ",
                    {
                        "ALAYAJET_QUEST_METAL_SELECTED_PAGE": "1",
                        "ALAYAJET_QUEST_METAL_SELECTED_FRAME": "resident",
                    },
                    clear=False,
                ), mock.patch.object(
                    controller.kv_cache,
                    "read_page_mx_from_disk",
                    side_effect=AssertionError("read_page_mx_from_disk called"),
                ), mock.patch.object(
                    BufferPool,
                    "get_page_mx",
                    side_effect=AssertionError("BufferPool.get_page_mx called"),
                ), mock.patch(
                    "alayajet.features.quest.ops.selected_frame_decode_attention_with_stable_arena",
                    side_effect=fake_attention,
                ):
                    decode_sparse_attn(
                        q,
                        mx.array(np.zeros((1, 1), dtype=np.int64)),
                        controller,
                        layer_idx=0,
                    )
            finally:
                controller.clean_states()

    def test_selected_frame_index_cache_reuses_and_invalidates_by_frame_slot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = QuestController(
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
                head_dim=4,
                page_size=2,
                page_budget=2,
                max_seq_len=8,
                cache_dir=tmpdir,
                buffer_pool_pages=2,
            )
            try:
                page0 = np.zeros((2, 2, 4), dtype=np.float16)
                page1 = np.ones((2, 2, 4), dtype=np.float16)
                physical = np.array([[0, 1], [1, 0]], dtype=np.int64)
                kv_heads = np.array([0, 0], dtype=np.int64)
                controller.kv_cache.buffer_pool.insert_page_cache((0, 0, 0), page0)
                controller.kv_cache.buffer_pool.insert_page_cache((0, 1, 0), page1)

                selected_a, frames_a = _load_selected_frame_pages(
                    controller,
                    0,
                    physical,
                    kv_heads,
                    mx.float16,
                )
                selected_b, frames_b = _load_selected_frame_pages(
                    controller,
                    0,
                    physical,
                    kv_heads,
                    mx.float16,
                )
                self.assertIs(selected_a, selected_b)
                self.assertIs(frames_a, frames_b)

                controller.kv_cache.buffer_pool.insert_page_cache((0, 2, 0), page1 + 2)
                controller.kv_cache.buffer_pool.insert_page_cache((0, 0, 0), page0)
                selected_c, _ = _load_selected_frame_pages(
                    controller,
                    0,
                    physical,
                    kv_heads,
                    mx.float16,
                )
                self.assertIsNot(selected_a, selected_c)
            finally:
                controller.clean_states()


@unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
class TestAlayaJetMetalSparseAttention(unittest.TestCase):
    def _run_selected_page_decode_case(self, *, head_dim):
        rng = np.random.default_rng(23)
        num_heads = 4
        selected_tokens = 24
        last_len = 7
        scale = head_dim**-0.5
        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        selected_k = rng.normal(size=(num_heads, selected_tokens, head_dim)).astype(np.float16)
        selected_v = rng.normal(size=(num_heads, selected_tokens, head_dim)).astype(np.float16)
        last_k = rng.normal(size=(num_heads, last_len, head_dim)).astype(np.float16)
        last_v = rng.normal(size=(num_heads, last_len, head_dim)).astype(np.float16)

        out = selected_page_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_k),
            mx.array(selected_v),
            mx.array(last_k),
            mx.array(last_v),
            scale=scale,
            max_last_tokens=8,
        )
        mx.eval(out)
        ref = _reference_selected_page_decode(
            q,
            selected_k,
            selected_v,
            last_k,
            last_v,
            scale,
        )
        np.testing.assert_allclose(
            np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2
        )

    def _run_selected_indexed_page_decode_case(self, *, head_dim, num_kv_heads=2):
        rng = np.random.default_rng(29)
        num_heads = 4
        num_pages = 5
        selected_count = 3
        page_size = 8
        last_len = 5
        scale = head_dim**-0.5
        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        kv_pages = rng.normal(
            size=(num_pages, 2, page_size, num_kv_heads, head_dim)
        ).astype(np.float16)
        selected_pages = np.array(
            [[0, 2, 4], [1, 2, 3], [0, 1, 4], [2, 3, 4]],
            dtype=np.int32,
        )[:, :selected_count]
        last_k = rng.normal(size=(last_len, num_kv_heads, head_dim)).astype(np.float16)
        last_v = rng.normal(size=(last_len, num_kv_heads, head_dim)).astype(np.float16)

        out = selected_indexed_page_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_pages),
            mx.array(kv_pages),
            mx.array(last_k),
            mx.array(last_v),
            scale=scale,
        )
        mx.eval(out)
        ref = _reference_selected_indexed_page_decode(
            q,
            selected_pages,
            kv_pages,
            last_k,
            last_v,
            scale,
        )
        np.testing.assert_allclose(
            np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2
        )

        kv_pages_head_major = kv_pages.transpose(3, 0, 1, 2, 4)
        last_k_head_major = last_k.transpose(1, 0, 2)
        last_v_head_major = last_v.transpose(1, 0, 2)
        out_head_major = selected_indexed_page_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_pages),
            mx.array(kv_pages_head_major),
            mx.array(last_k_head_major),
            mx.array(last_v_head_major),
            scale=scale,
            kv_layout="head_major",
            last_layout="head_major",
        )
        mx.eval(out_head_major)
        ref_head_major = _reference_selected_indexed_page_decode(
            q,
            selected_pages,
            kv_pages_head_major,
            last_k_head_major,
            last_v_head_major,
            scale,
            kv_layout="head_major",
            last_layout="head_major",
        )
        np.testing.assert_allclose(
            np.asarray(out_head_major[0, :, 0, :]),
            ref_head_major,
            rtol=2e-2,
            atol=2e-2,
        )

    def _run_selected_frame_decode_case(self, *, head_dim, num_kv_heads=2):
        rng = np.random.default_rng(31)
        num_heads = 4
        num_frames = 7
        selected_count = 3
        page_size = 8
        last_len = 5
        scale = head_dim**-0.5
        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        frames = rng.normal(
            size=(num_frames, 2, page_size, head_dim)
        ).astype(np.float16)
        selected_frames = np.array(
            [[0, 2, 4], [1, 2, 3], [0, 5, 6], [2, 3, 4]],
            dtype=np.int32,
        )[:, :selected_count]
        last_k = rng.normal(size=(num_kv_heads, last_len, head_dim)).astype(np.float16)
        last_v = rng.normal(size=(num_kv_heads, last_len, head_dim)).astype(np.float16)

        out = selected_frame_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_frames),
            mx.array(frames),
            mx.array(last_k),
            mx.array(last_v),
            scale=scale,
            last_layout="head_major",
        )
        mx.eval(out)
        ref = _reference_selected_frame_decode(
            q,
            selected_frames,
            frames,
            last_k,
            last_v,
            scale,
        )
        np.testing.assert_allclose(
            np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2
        )

    def _run_selected_direct_frame_decode_case(self, *, head_dim, num_kv_heads=2):
        rng = np.random.default_rng(43)
        num_heads = 4
        num_frames = 5
        selected_count = 3
        page_size = 8
        last_len = 5
        scale = head_dim**-0.5
        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        frames = rng.normal(
            size=(num_frames, 2, page_size, head_dim)
        ).astype(np.float16)
        selected_frames = np.array(
            [[0, 2, 4], [1, 2, 3], [0, 3, 4], [2, 3, 4]],
            dtype=np.int32,
        )[:, :selected_count]
        last_k = rng.normal(size=(num_kv_heads, last_len, head_dim)).astype(np.float16)
        last_v = rng.normal(size=(num_kv_heads, last_len, head_dim)).astype(np.float16)

        out = selected_direct_frame_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_frames),
            [mx.array(frame) for frame in frames],
            mx.array(last_k),
            mx.array(last_v),
            scale=scale,
            last_layout="head_major",
        )
        mx.eval(out)
        ref = _reference_selected_frame_decode(
            q,
            selected_frames,
            frames,
            last_k,
            last_v,
            scale,
        )
        np.testing.assert_allclose(
            np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2
        )

    def _run_selected_direct_frame_with_last_case(self, *, head_dim):
        rng = np.random.default_rng(47)
        num_heads = 4
        num_kv_heads = 2
        page_size = 8
        last_len = 3
        scale = head_dim**-0.5
        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        history = rng.normal(size=(4, 2, page_size, head_dim)).astype(np.float16)
        last = rng.normal(size=(2, 2, page_size, head_dim)).astype(np.float16)
        frames = np.concatenate([history, last], axis=0)
        selected_history = np.array(
            [[0, 2], [1, 3], [0, 1], [2, 3]],
            dtype=np.int32,
        )
        selected_with_last = np.concatenate(
            [
                selected_history,
                np.array([[4], [4], [5], [5]], dtype=np.int32),
            ],
            axis=1,
        )
        last_k = np.stack([last[0, 0, :last_len], last[1, 0, :last_len]], axis=0)
        last_v = np.stack([last[0, 1, :last_len], last[1, 1, :last_len]], axis=0)
        out = selected_direct_frame_decode_attention_with_last_frame(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_with_last),
            [mx.array(frame) for frame in frames],
            num_kv_heads=num_kv_heads,
            last_len=last_len,
            scale=scale,
        )
        ref = selected_direct_frame_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_history),
            [mx.array(frame) for frame in history],
            mx.array(last_k),
            mx.array(last_v),
            scale=scale,
            last_layout="head_major",
        )
        mx.eval(out, ref)
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(ref), rtol=2e-2, atol=2e-2
        )

    def _run_selected_frame_with_last_case(self, *, head_dim):
        rng = np.random.default_rng(49)
        num_heads = 4
        num_kv_heads = 2
        page_size = 8
        last_len = 3
        scale = head_dim**-0.5
        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        history = rng.normal(size=(4, 2, page_size, head_dim)).astype(np.float16)
        last = rng.normal(size=(2, 2, page_size, head_dim)).astype(np.float16)
        frames = np.concatenate([history, last], axis=0)
        selected_history = np.array(
            [[0, 2], [1, 3], [0, 1], [2, 3]],
            dtype=np.int32,
        )
        selected_with_last = np.concatenate(
            [selected_history, np.array([[4], [4], [5], [5]], dtype=np.int32)],
            axis=1,
        )
        last_k = np.stack([last[0, 0, :last_len], last[1, 0, :last_len]], axis=0)
        last_v = np.stack([last[0, 1, :last_len], last[1, 1, :last_len]], axis=0)
        out = selected_frame_decode_attention_with_last_frame(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_with_last),
            mx.array(frames),
            num_kv_heads=num_kv_heads,
            last_len=last_len,
            scale=scale,
        )
        ref = selected_frame_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(selected_history),
            mx.array(history),
            mx.array(last_k),
            mx.array(last_v),
            scale=scale,
            last_layout="head_major",
        )
        mx.eval(out, ref)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-2, atol=2e-2)

    def _run_selected_resident_frame_decode_case(self, *, head_dim):
        rng = np.random.default_rng(41)
        num_heads = 2
        num_kv_heads = 1
        page_size = 4
        last_len = 1
        scale = head_dim**-0.5
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = BufferPool(
                backing_file=f"{tmpdir}/kv.bin",
                num_layers=1,
                capacity=4,
                num_heads=num_kv_heads,
                page_size=page_size,
                head_dim=head_dim,
                dtype=np.float16,
                mx_dtype=mx.float16,
                num_frames=4,
            )
            try:
                page0 = (0, 0, 0)
                page1 = (0, 1, 0)
                k0 = rng.normal(size=(page_size, head_dim)).astype(np.float16)
                v0 = rng.normal(size=(page_size, head_dim)).astype(np.float16)
                k1 = rng.normal(size=(page_size, head_dim)).astype(np.float16)
                v1 = rng.normal(size=(page_size, head_dim)).astype(np.float16)
                frame0 = pool.write_kv_slice_mx_resident(
                    page0,
                    0,
                    mx.array(k0),
                    mx.array(v0),
                    assume_zero=True,
                )
                frame1 = pool.write_kv_slice_mx_resident(
                    page1,
                    0,
                    mx.array(k1),
                    mx.array(v1),
                    assume_zero=True,
                )
                frames = pool.get_resident_layer_slot_buffers_mx(0, [frame0, frame1])
                local0 = pool.layer_local_frame_index(0, frame0)
                local1 = pool.layer_local_frame_index(0, frame1)
                selected_frames = np.array(
                    [[local0, local1], [local1, local0]],
                    dtype=np.int32,
                )
                q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
                last_k = rng.normal(size=(num_kv_heads, last_len, head_dim)).astype(np.float16)
                last_v = rng.normal(size=(num_kv_heads, last_len, head_dim)).astype(np.float16)
                out = selected_direct_frame_decode_attention(
                    mx.array(q).reshape(1, num_heads, 1, head_dim),
                    mx.array(selected_frames),
                    frames,
                    mx.array(last_k),
                    mx.array(last_v),
                    scale=scale,
                    last_layout="head_major",
                )
                mx.eval(out)
                ref = _reference_selected_frame_decode(
                    q,
                    selected_frames,
                    np.asarray(mx.stack(frames, axis=0)),
                    last_k,
                    last_v,
                    scale,
                )
                np.testing.assert_allclose(
                    np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2
                )
            finally:
                pool.close()

    def _run_fused_sparse_decode_case(self, *, head_dim, num_kv_heads=2):
        rng = np.random.default_rng(7)
        num_heads = 4
        num_pages = 6
        page_size = 8
        last_len = 5
        page_budget = 3
        scale = head_dim**-0.5

        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        kv_pages = rng.normal(
            size=(num_pages, 2, page_size, num_kv_heads, head_dim)
        ).astype(np.float16)
        last_k = rng.normal(size=(last_len, num_kv_heads, head_dim)).astype(np.float16)
        last_v = rng.normal(size=(last_len, num_kv_heads, head_dim)).astype(np.float16)
        k_values = kv_pages[:, 0]
        k_min = k_values.min(axis=1).transpose(1, 0, 2).astype(np.float32)
        k_max = k_values.max(axis=1).transpose(1, 0, 2).astype(np.float32)

        out = fused_sparse_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(k_min),
            mx.array(k_max),
            mx.array(kv_pages),
            mx.array(last_k),
            mx.array(last_v),
            page_budget=page_budget,
            scale=scale,
        )
        mx.eval(out)
        ref = _reference_sparse_decode(
            q,
            k_min,
            k_max,
            kv_pages,
            last_k,
            last_v,
            page_budget,
            scale,
        )
        np.testing.assert_allclose(np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2)

        out_head_major = fused_sparse_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(k_min),
            mx.array(k_max),
            pack_kv_pages_head_major(mx.array(kv_pages)),
            pack_last_page_head_major(mx.array(last_k)),
            pack_last_page_head_major(mx.array(last_v)),
            page_budget=page_budget,
            scale=scale,
            kv_layout="head_major",
            last_layout="head_major",
        )
        mx.eval(out_head_major)
        np.testing.assert_allclose(
            np.asarray(out_head_major[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2
        )

    def test_fused_sparse_decode_matches_reference_gqa_d64(self):
        self._run_fused_sparse_decode_case(head_dim=64)

    def test_fused_sparse_decode_matches_reference_gqa_d128(self):
        self._run_fused_sparse_decode_case(head_dim=128)

    def test_fused_sparse_decode_matches_reference_mqa_d64(self):
        self._run_fused_sparse_decode_case(head_dim=64, num_kv_heads=1)

    def test_selected_page_decode_matches_reference_d64(self):
        self._run_selected_page_decode_case(head_dim=64)

    def test_selected_page_decode_matches_reference_d128(self):
        self._run_selected_page_decode_case(head_dim=128)

    def test_selected_indexed_page_decode_matches_reference_gqa_d64(self):
        self._run_selected_indexed_page_decode_case(head_dim=64)

    def test_selected_indexed_page_decode_matches_reference_gqa_d128(self):
        self._run_selected_indexed_page_decode_case(head_dim=128)

    def test_selected_indexed_page_decode_matches_reference_mqa_d64(self):
        self._run_selected_indexed_page_decode_case(head_dim=64, num_kv_heads=1)

    def test_selected_indexed_page_decode_matches_reference_mqa_d128(self):
        self._run_selected_indexed_page_decode_case(head_dim=128, num_kv_heads=1)

    def test_selected_frame_decode_matches_reference_gqa_d64(self):
        self._run_selected_frame_decode_case(head_dim=64)

    def test_selected_frame_decode_matches_reference_gqa_d128(self):
        self._run_selected_frame_decode_case(head_dim=128)

    def test_selected_frame_decode_matches_reference_mqa_d64(self):
        self._run_selected_frame_decode_case(head_dim=64, num_kv_heads=1)

    def test_selected_frame_decode_matches_reference_mqa_d128(self):
        self._run_selected_frame_decode_case(head_dim=128, num_kv_heads=1)

    def test_selected_direct_frame_decode_matches_reference_gqa_d64(self):
        self._run_selected_direct_frame_decode_case(head_dim=64)

    def test_selected_direct_frame_decode_matches_reference_gqa_d128(self):
        self._run_selected_direct_frame_decode_case(head_dim=128)

    def test_selected_direct_frame_decode_matches_reference_mqa_d64(self):
        self._run_selected_direct_frame_decode_case(head_dim=64, num_kv_heads=1)

    def test_selected_direct_frame_decode_matches_reference_mqa_d128(self):
        self._run_selected_direct_frame_decode_case(head_dim=128, num_kv_heads=1)

    def test_selected_direct_frame_with_last_matches_legacy_last_path_d64(self):
        self._run_selected_direct_frame_with_last_case(head_dim=64)

    def test_selected_frame_with_last_matches_legacy_last_path_d64(self):
        self._run_selected_frame_with_last_case(head_dim=64)

    def test_selected_resident_frame_decode_matches_reference_d64(self):
        self._run_selected_resident_frame_decode_case(head_dim=64)


if __name__ == "__main__":
    unittest.main()
