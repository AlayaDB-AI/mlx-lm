import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from typing import Optional

import mlx.core as mx


class QuestTiming:
    def __init__(
        self,
        enabled: bool,
        sync: bool,
        trace_output: Optional[str] = None,
        trace_phase: Optional[str] = None,
        trace_decode_steps: Optional[int] = 3,
    ):
        self.enabled = enabled
        self.sync = sync
        self.trace_output = trace_output
        self.trace_phase = trace_phase if trace_phase not in ("", None, "all") else None
        if trace_decode_steps is None or trace_decode_steps <= 0:
            self.trace_decode_steps = None
        else:
            self.trace_decode_steps = trace_decode_steps
        self._stats = defaultdict(lambda: [0.0, 0])
        self._trace_events = []
        self._trace_t0 = None
        self._trace_tid_map = {}
        self._trace_next_tid = 0
        self._trace_allow = True
        self._trace_step_ts = []
        self._tracked_decode_keys = {
            "decode_append_kv",
            "decode_estimate_topk",
            "decode_sparse_attn",
            "decode_indices",
            "decode_disk_read",
            "decode_last_page",
            "decode_stream_attn",
            "decode_fused_metal_sparse_attn",
        }
        self._trace_hidden_keys = {
            "decode_sparse_attn",
            "decode_fused_metal_sparse_attn",
        }

    def record(self, key: str, start_time: float, *sync_arrays):
        if not self.enabled:
            return
        if not self._should_track_key(key):
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
        if self.trace_output and self._trace_should_record(key):
            self._record_trace_event(key, start_time, end_time)

    def begin_step(self, phase: str, step_index: Optional[int] = None):
        if not self.trace_output:
            return
        if phase != "decode" or self.trace_decode_steps is None:
            self._trace_allow = True
            return
        if step_index is None:
            self._trace_allow = False
            return
        self._trace_allow = step_index < self.trace_decode_steps
        if self._trace_allow:
            now = time.perf_counter()
            if self._trace_t0 is None:
                self._trace_t0 = now
            self._trace_step_ts.append(now)

    def _trace_should_record(self, key: str) -> bool:
        if not self._trace_allow:
            return False
        if key in self._trace_hidden_keys:
            return False
        if self.trace_phase:
            prefix = f"{self.trace_phase}_"
            return key.startswith(prefix)
        return True

    def _should_track_key(self, key: str) -> bool:
        if key.startswith("decode_"):
            return key in self._tracked_decode_keys
        if key.startswith("prefill_"):
            return False
        return False

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

        total_time = sum(total for total, _ in self._stats.values())
        print(f"  total: {total_time:.3f}s")

        print("  decode:")
        for key in (
            "decode_append_kv",
            "decode_estimate_topk",
            "decode_sparse_attn",
            "decode_fused_metal_sparse_attn",
        ):
            fmt_line(key, denom=total_time)

        if "decode_sparse_attn" in self._stats:
            decode_total = self._stats["decode_sparse_attn"][0]
            print("  decode_sparse_attn breakdown:")
            for key in (
                "decode_indices",
                "decode_disk_read",
                "decode_last_page",
                "decode_stream_attn",
            ):
                fmt_line(key, denom=decode_total)

        if reset:
            self._stats.clear()

    def write_trace(self):
        if not self.trace_output:
            return
        self._write_trace_output(self.trace_output)

    def _filter_trace_events(self):
        if not self.trace_phase:
            return list(self._trace_events)
        prefix = f"{self.trace_phase}_"
        x_events = [
            e for e in self._trace_events
            if e.get("ph") == "X" and e.get("name", "").startswith(prefix)
        ]
        allowed_names = {e["name"] for e in x_events}
        filtered = []
        for e in self._trace_events:
            ph = e.get("ph")
            if ph == "X":
                if e.get("name") in allowed_names:
                    filtered.append(e)
            elif ph == "M":
                if e.get("args", {}).get("name") in allowed_names:
                    filtered.append(e)
        return filtered

    def _trace_row_order(self, names, first_ts=None):
        if first_ts:
            return sorted(names, key=lambda name: (first_ts.get(name, 0.0), name))
        if self.trace_phase == "decode":
            preferred = [
                "decode_append_kv",
                "decode_estimate_topk",
                "decode_sparse_attn",
                "decode_indices",
                "decode_disk_read",
                "decode_last_page",
                "decode_stream_attn",
            ]
        elif self.trace_phase == "prefill":
            preferred = []
        else:
            preferred = []
        ordered = [name for name in preferred if name in names]
        ordered.extend(sorted(name for name in names if name not in set(ordered)))
        return ordered

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
        events = self._filter_trace_events()
        if not events:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".png"):
            svg_path = path[:-4] + ".svg"
            self._write_svg_timeline(svg_path, events)
            if not self._convert_svg_to_png(svg_path, path):
                print(f"[Quest] PNG export failed; keeping SVG at {svg_path}")
            else:
                print(f"[Quest] Timeline saved to {path}")
        elif path.endswith(".svg"):
            self._write_svg_timeline(path, events)
        else:
            trace_path = path if path.endswith(".json") else f"{path}.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump({"traceEvents": events}, f)
            print(f"[Quest] Trace saved to {trace_path}")

    def _write_svg_timeline(self, path: str, events):
        events = [e for e in events if e.get("ph") == "X"]
        if not events:
            return
        max_ts = max(e["ts"] + e["dur"] for e in events)
        if max_ts <= 0:
            return

        names = sorted({e["name"] for e in events})
        first_ts = {}
        for e in events:
            name = e["name"]
            ts = e["ts"]
            if name not in first_ts or ts < first_ts[name]:
                first_ts[name] = ts
        row_order = self._trace_row_order(names, first_ts=first_ts)
        name_to_row = {name: idx for idx, name in enumerate(row_order)}
        row_height = 18
        row_gap = 6
        top_margin = 20
        left_margin = 140
        base_width = 1400
        char_width = 7
        max_label_len = max((len(name) for name in row_order), default=0)
        plot_width = base_width - left_margin - 20
        if plot_width < 600:
            plot_width = 600
            width = left_margin + plot_width + 20
        else:
            width = base_width
        height = top_margin + (row_height + row_gap) * len(row_order) + 20
        scale = plot_width / max_ts

        def color_for(name: str) -> str:
            h = abs(hash(name)) % 360
            return f"hsl({h},60%,65%)"

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<style>text{font-family:Arial, sans-serif;font-size:12px;}</style>',
            f'<text x="{left_margin}" y="14">{self._timeline_title()}</text>',
        ]

        for name in row_order:
            y = top_margin + name_to_row[name] * (row_height + row_gap)
            lines.append(f'<text x="8" y="{y + row_height - 4}">{name}</text>')

        for e in events:
            name = e["name"]
            row = name_to_row.get(name)
            if row is None:
                continue
            x = left_margin + e["ts"] * scale
            w = max(e["dur"] * scale, 1.0)
            y = top_margin + row * (row_height + row_gap)
            lines.append(
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row_height}" '
                f'fill="{color_for(name)}" opacity="0.8" />'
            )

        if self._trace_step_ts and self._trace_t0 is not None:
            y1 = top_margin - 6
            y2 = height - 10
            for step_t in self._trace_step_ts:
                step_us = (step_t - self._trace_t0) * 1_000_000.0
                if step_us <= 0:
                    continue
                x = left_margin + step_us * scale
                lines.append(
                    f'<line x1="{x:.2f}" y1="{y1}" x2="{x:.2f}" y2="{y2}" '
                    f'stroke=\"#999999\" stroke-dasharray=\"4,4\" stroke-width=\"1\" />'
                )

        lines.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Quest] Timeline saved to {path}")

    def _convert_svg_to_png(self, svg_path: str, png_path: str) -> bool:
        if shutil.which("rsvg-convert"):
            try:
                subprocess.run(
                    ["rsvg-convert", "-o", png_path, svg_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                return False
        if sys.platform == "darwin" and shutil.which("qlmanage"):
            out_dir = os.path.dirname(png_path) or "."
            try:
                subprocess.run(
                    ["qlmanage", "-t", "-s", "1000", "-o", out_dir, svg_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return False
            generated = os.path.join(out_dir, os.path.basename(svg_path) + ".png")
            if not os.path.exists(generated):
                return False
            try:
                if os.path.abspath(generated) != os.path.abspath(png_path):
                    os.replace(generated, png_path)
                return True
            except Exception:
                return False
        return False

    def _timeline_title(self) -> str:
        if self.trace_phase == "decode":
            title = "Quest decode timing timeline (us)"
            if self.trace_decode_steps is not None:
                title += f", sampled first {self.trace_decode_steps} steps"
            return title
        if self.trace_phase == "prefill":
            return "Quest prefill timing timeline (us)"
        return "Quest timing timeline (us)"
