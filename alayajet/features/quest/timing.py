import json
import os
import time
from collections import defaultdict
from typing import Optional

import mlx.core as mx


class QuestTiming:
    def __init__(self, enabled: bool, sync: bool, trace_output: Optional[str] = None):
        self.enabled = enabled
        self.sync = sync
        self.trace_output = trace_output
        self._stats = defaultdict(lambda: [0.0, 0])
        self._trace_events = []
        self._trace_t0 = None
        self._trace_tid_map = {}
        self._trace_next_tid = 0

    def record(self, key: str, start_time: float, *sync_arrays):
        if not self.enabled:
            return
        if self.sync and sync_arrays:
            to_sync = [arr for arr in sync_arrays if arr is not None]
            if to_sync:
                mx.eval(to_sync)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        stat = self._stats[key]
        stat[0] += elapsed
        stat[1] += 1
        if self.trace_output:
            self._record_trace_event(key, start_time, end_time)

    def report(self, reset: bool = False, prefix: str = "[Quest][Timing]"):
        if not self._stats:
            return
        print(prefix)

        def fmt_line(key: str, denom: float = None):
            if key not in self._stats:
                return
            total, count = self._stats[key]
            avg = total / count if count else 0.0
            if denom:
                pct = (total / denom) * 100.0 if denom > 0 else 0.0
                print(f"  {key}: {total:.3f}s ({pct:.1f}%), avg {avg*1000:.3f}ms, n={count}")
            else:
                print(f"  {key}: {total:.3f}s, avg {avg*1000:.3f}ms, n={count}")

        if "prefill_total" in self._stats:
            fmt_line("prefill_total")

        total_time = sum(
            total
            for key, (total, _) in self._stats.items()
            if key != "prefill_total"
        )
        print(f"  total: {total_time:.3f}s")

        print("  prefill:")
        for key in (
            "prefill_proj",
            "prefill_rope",
            "prefill_append_kv",
            "prefill_attn",
            "prefill_o_proj",
        ):
            fmt_line(key, denom=total_time)

        print("  decode:")
        for key in (
            "decode_proj",
            "decode_rope",
            "decode_append_kv",
            "decode_estimate_topk",
            "decode_sparse_attn",
            "decode_o_proj",
        ):
            fmt_line(key, denom=total_time)

        if "decode_sparse_attn" in self._stats:
            decode_total = self._stats["decode_sparse_attn"][0]
            print("  decode_sparse_attn breakdown:")
            for key in (
                "decode_indices",
                "decode_disk_read",
                "decode_disk_to_mx",
                "decode_last_page",
                "decode_concat",
                "decode_sdpa",
            ):
                fmt_line(key, denom=decode_total)

        if reset:
            self._stats.clear()

    def write_trace(self):
        if not self.trace_output:
            return
        self._write_trace_output(self.trace_output)

    def _record_trace_event(self, key: str, start_time: float, end_time: float):
        if self._trace_t0 is None:
            self._trace_t0 = start_time
        if key not in self._trace_tid_map:
            tid = self._trace_next_tid
            self._trace_tid_map[key] = tid
            self._trace_next_tid += 1
            self._trace_events.append(
                {
                    "ph": "M",
                    "pid": 1,
                    "tid": tid,
                    "name": "thread_name",
                    "args": {"name": key},
                }
            )
        else:
            tid = self._trace_tid_map[key]
        start_us = (start_time - self._trace_t0) * 1_000_000.0
        dur_us = (end_time - start_time) * 1_000_000.0
        self._trace_events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": tid,
                "name": key,
                "ts": start_us,
                "dur": dur_us,
                "cat": "quest",
            }
        )

    def _write_trace_output(self, path: str):
        if not self._trace_events:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".svg"):
            self._write_svg_timeline(path)
        else:
            trace_path = path if path.endswith(".json") else f"{path}.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump({"traceEvents": self._trace_events}, f)
            print(f"[Quest] Trace saved to {trace_path}")

    def _write_svg_timeline(self, path: str):
        events = [e for e in self._trace_events if e.get("ph") == "X"]
        if not events:
            return
        max_ts = max(e["ts"] + e["dur"] for e in events)
        if max_ts <= 0:
            return

        tid_to_name = {tid: name for name, tid in self._trace_tid_map.items()}
        row_height = 18
        row_gap = 6
        top_margin = 20
        left_margin = 140
        width = 1400
        height = top_margin + (row_height + row_gap) * len(tid_to_name) + 20
        scale = (width - left_margin - 20) / max_ts

        def color_for(name: str) -> str:
            h = abs(hash(name)) % 360
            return f"hsl({h},60%,65%)"

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<style>text{font-family:Arial, sans-serif;font-size:12px;}</style>',
            f'<text x="{left_margin}" y="14">Quest timing timeline (us)</text>',
        ]

        for tid, name in sorted(tid_to_name.items(), key=lambda item: item[0]):
            y = top_margin + tid * (row_height + row_gap)
            lines.append(f'<text x="8" y="{y + row_height - 4}">{name}</text>')

        for e in events:
            tid = e["tid"]
            name = tid_to_name.get(tid, "unknown")
            x = left_margin + e["ts"] * scale
            w = max(e["dur"] * scale, 1.0)
            y = top_margin + tid * (row_height + row_gap)
            lines.append(
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row_height}" '
                f'fill="{color_for(name)}" opacity="0.8" />'
            )

        lines.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Quest] Timeline saved to {path}")
