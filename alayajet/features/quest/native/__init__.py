from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx


_MODULE = None


def _try_load_native_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    try:
        from .build import build

        module_path = build()
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "alayajet.features.quest.native._quest_native",
            str(module_path),
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
        return module
    except Exception:
        return None


def resident_write(frame, k_slice, v_slice, page_offset: int):
    module = _try_load_native_module()
    if module is None:
        return None
    return module.resident_write(frame, k_slice, v_slice, page_offset)


def resident_write_batched(frames, k_slice, v_slice, page_offset: int):
    module = _try_load_native_module()
    if module is None or not hasattr(module, "resident_write_batched"):
        return None
    return module.resident_write_batched(frames, k_slice, v_slice, page_offset)


def resident_arena_write_batched(arena, frame_ids, k_slice, v_slice, page_offset: int):
    module = _try_load_native_module()
    if module is None or not hasattr(module, "resident_arena_write_batched"):
        return None
    return module.resident_arena_write_batched(arena, frame_ids, k_slice, v_slice, page_offset)
