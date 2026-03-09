import argparse
import json
import logging
import os
import shutil
import socket
import time
import warnings
from http.server import ThreadingHTTPServer

import mlx.core as mx
import mlx_lm.server as base_server

from ..engine import AlayaEngine
from .quest_kv_cache import QuestDiskCache
from ..features.timing import TimingFeature
from ..features.quest.integration import QuestFeature

DEFAULT_PREFILL_STEP_SIZE = 8192
DEFAULT_PAGE_BUDGET = 64


class AlayaModelProvider(base_server.ModelProvider):
    def __init__(self, cli_args: argparse.Namespace):
        self._engine = None
        self._cache_cleared = False
        super().__init__(cli_args)

    def _maybe_clear_cache(self):
        if not self.cli_args.quest_reset_cache or self._cache_cleared:
            return
        if os.path.exists(self.cli_args.cache_dir):
            shutil.rmtree(self.cli_args.cache_dir)
        self._cache_cleared = True

    def _attach_engine(self, model):
        if self._engine is not None:
            self._engine.detach()
            self._engine = None

        if self.cli_args.quest:
            self._maybe_clear_cache()
            self._engine = AlayaEngine.with_quest(
                page_budget=(
                    self.cli_args.page_budget
                    if self.cli_args.page_budget is not None
                    else DEFAULT_PAGE_BUDGET
                ),
                cache_dir=self.cli_args.cache_dir,
                async_disk_write=self.cli_args.quest_async_disk_write,
                timing=self.cli_args.quest_timing,
                timing_sync=self.cli_args.quest_timing_sync,
                trace_output=self.cli_args.quest_trace_output or None,
                trace_decode_steps=self.cli_args.quest_trace_decode_steps,
                log_prefill_layer_timing=self.cli_args.quest_prefill_layer_log,
                log_prefill_io_overlap=self.cli_args.quest_prefill_io_log,
                log_decode_lru_hit_rate=self.cli_args.quest_lru_log,
                log_memory=self.cli_args.quest_mem_log,
                log_memory_sync=self.cli_args.quest_timing_sync,
                log_prefill_progress=True,
                disable_os_cache=self.cli_args.quest_disable_os_cache,
            )
        elif self.cli_args.timing:
            self._engine = AlayaEngine()
            self._engine.add_feature(TimingFeature(timing_sync=self.cli_args.timing_sync))

        if self._engine is not None:
            self._engine.attach(model)

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        model_path = self.default_model_map.get(model_path, model_path)
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        if self._engine is not None:
            self._engine.detach()
            self._engine = None

        model, tokenizer = super().load(model_path, adapter_path, draft_model_path)
        self._attach_engine(model)
        return model, tokenizer


class AlayaResponseGenerator(base_server.ResponseGenerator):
    def __init__(
        self,
        model_provider: base_server.ModelProvider,
        prompt_cache: base_server.LRUPromptCache,
        *,
        prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
        allow_batch: bool = False,
    ):
        self.prefill_step_size = prefill_step_size
        self.allow_batch = allow_batch
        self.quest_disk_cache = QuestDiskCache()
        super().__init__(model_provider, prompt_cache)

    def _is_batchable(self, args):
        if not self.allow_batch:
            return False
        if getattr(self.model_provider.cli_args, "quest", False):
            return False
        return super()._is_batchable(args)

    def _get_quest_feature(self):
        engine = getattr(self.model_provider, "_engine", None)
        if engine is None:
            return None
        for feature in engine.features:
            if isinstance(feature, QuestFeature):
                return feature
        return None

    def _serve_single(self, request):
        rqueue, request, args = request

        def progress(tokens_processed, tokens_total):
            rqueue.put((tokens_processed, tokens_total))

        try:
            model, tokenizer = self.model_provider.load(
                args.model.model, args.model.adapter, args.model.draft
            )
            draft_model = self.model_provider.draft_model

            prompt = self._tokenize(tokenizer, request)

            ctx = base_server.GenerationContext(
                has_tool_calling=tokenizer.has_tool_calling,
                tool_call_start=tokenizer.tool_call_start,
                tool_call_end=tokenizer.tool_call_end,
                tool_parser=tokenizer.tool_parser,
                has_thinking=tokenizer.has_thinking,
                think_start_id=tokenizer.think_start_id,
                think_end=tokenizer.think_end,
                think_end_id=tokenizer.think_end_id,
                eos_token_ids=tokenizer.eos_token_ids,
                stop_token_sequences=[
                    tokenizer.encode(stop_word, add_special_tokens=False)
                    for stop_word in args.stop_words
                ],
                prompt=prompt,
            )
            rqueue.put(ctx)

            if args.seed is not None:
                mx.random.seed(args.seed)

            sampler = base_server._make_sampler(args, tokenizer)
            logits_processors = base_server._make_logits_processors(args)

            cache = None
            rest = prompt
            cache_key = prompt[:]
            t_request_start = time.perf_counter()
            first_prompt_tps = None
            first_prompt_tokens = None
            t_first_token = None

            load_path = getattr(args, "load_kv_cache_path", None)
            save_path = getattr(args, "save_kv_cache_path", None)
            model_id = args.model.model if args and args.model else None

            quest_feature = self._get_quest_feature()
            quest_controller = None
            source_cache_dir = (
                self.quest_disk_cache.resolve_cache_dir(load_path)
                if load_path
                else None
            )
            if save_path:
                working_cache_dir = self.quest_disk_cache.resolve_cache_dir(save_path)
            elif load_path:
                working_cache_dir = source_cache_dir
            else:
                working_cache_dir = self.model_provider.cli_args.cache_dir
            quest_hit = False
            if quest_feature is not None:
                quest_controller = quest_feature.ensure_controller(
                    cache_dir=working_cache_dir
                )
                # Treat regular API requests as request-scoped Quest sessions.
                # Without this reset, a prior large request leaves KV/metadata
                # state behind and can make later short prompts pay for stale
                # pages from earlier requests.
                if not load_path and not save_path:
                    quest_controller.clean_states()

            prefix_len_for_restore = None
            prefix_active_indices = None
            backup_pages = None
            backup_phys_page = None

            try:
                if quest_controller is not None and load_path:
                    cached_tokens, rest, prefix_len = self.quest_disk_cache.load(
                        quest_controller,
                        load_path,
                        prompt,
                        model_id=model_id,
                    )
                    quest_hit = cached_tokens is not None
                    if quest_hit:
                        logging.info(
                            "[QuestCache] hit path=%s prefix_len=%d prompt_len=%d rest=%d",
                            load_path,
                            prefix_len,
                            len(prompt),
                            len(rest),
                        )
                    else:
                        logging.info("[QuestCache] miss path=%s", load_path)
                    if quest_hit and save_path is None:
                        kv_cache = quest_controller.kv_cache
                        prefix_len_for_restore = prefix_len
                        prefix_active_indices = list(kv_cache.active_indices)
                        if prefix_len_for_restore > 0 and prefix_active_indices:
                            if prefix_len_for_restore % kv_cache.page_size != 0:
                                last_logical = (prefix_len_for_restore - 1) // kv_cache.page_size
                                if last_logical < len(prefix_active_indices):
                                    backup_phys_page = prefix_active_indices[last_logical]
                                    backup_pages = []
                                    pool = kv_cache.buffer_pool
                                    for layer_idx in range(kv_cache.num_layers):
                                        for kv_head in range(kv_cache.num_heads):
                                            page_id = (layer_idx, backup_phys_page, kv_head)
                                            offset = pool._page_offset(page_id)
                                            data = os.pread(pool.fd, pool.page_bytes, offset)
                                            if len(data) != pool.page_bytes:
                                                raise RuntimeError(
                                                    f"Short read for backup page {page_id}: "
                                                    f"{len(data)} != {pool.page_bytes}"
                                                )
                                            backup_pages.append((offset, data))
                if quest_controller is not None and (load_path or save_path) and not quest_hit:
                    quest_controller.clean_states()

                if cache is None:
                    cache = base_server.make_prompt_cache(self.model_provider.model)
                    if self.model_provider.draft_model is not None:
                        cache += base_server.make_prompt_cache(self.model_provider.draft_model)

                for gen in base_server.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rest,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=cache,
                    draft_model=draft_model,
                    num_draft_tokens=args.num_draft_tokens,
                    prompt_progress_callback=progress,
                    prefill_step_size=self.prefill_step_size,
                ):
                    if first_prompt_tps is None:
                        first_prompt_tps = gen.prompt_tps
                        first_prompt_tokens = gen.prompt_tokens
                        t_first_token = time.perf_counter()
                    top_tokens = None
                    if args.logprobs > 0:
                        sorted_indices = mx.argpartition(
                            -gen.logprobs, kth=args.logprobs - 1
                        )
                        top_indices = sorted_indices[: args.logprobs]
                        top_logprobs = gen.logprobs[top_indices]
                        top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                        top_tokens = tuple(top_token_info)

                    rqueue.put(
                        base_server.Response(
                            gen.text,
                            gen.token,
                            gen.logprobs[gen.token].item(),
                            gen.finish_reason,
                            top_tokens,
                        )
                    )
                    cache_key.append(gen.token)

                    if ctx._should_stop:
                        break

                rqueue.put(None)
                t_request_end = time.perf_counter()

                # Persist Quest KV cache only when save path is provided.
                save_target = save_path
                if quest_controller is not None and save_target:
                    self.quest_disk_cache.save(
                        quest_controller,
                        save_target,
                        prompt_tokens=prompt,
                        model_id=model_id,
                    )

                # Timing log for KV cache workflows (save/load) so we can compare
                # end-to-end request latency between the first (save/miss) and
                # subsequent (load/hit) requests.
                if load_path or save_path:
                    gen_tokens = max(0, len(cache_key) - len(prompt))
                    total_time = t_request_end - t_request_start
                    prompt_time = None
                    if first_prompt_tps and first_prompt_tokens:
                        prompt_time = first_prompt_tokens / first_prompt_tps
                    if load_path:
                        op = "load_hit" if quest_hit else "load_miss"
                    else:
                        op = "save"

                    if prompt_time is not None:
                        logging.info(
                            "[QuestCache][Timing] op=%s prompt_tokens=%d prompt_time=%.3fs "
                            "gen_tokens=%d total_time=%.3fs",
                            op,
                            first_prompt_tokens,
                            prompt_time,
                            gen_tokens,
                            total_time,
                        )
                    else:
                        logging.info(
                            "[QuestCache][Timing] op=%s gen_tokens=%d total_time=%.3fs",
                            op,
                            gen_tokens,
                            total_time,
                        )
            finally:
                if (
                    save_path is None
                    and quest_controller is not None
                    and prefix_active_indices is not None
                    and prefix_len_for_restore is not None
                ):
                    kv_cache = quest_controller.kv_cache
                    pool = kv_cache.buffer_pool
                    current_active = list(kv_cache.active_indices)
                    prefix_pages = len(prefix_active_indices)
                    new_pages = current_active[prefix_pages:]

                    def invalidate_phys(phys_page: int):
                        for layer_idx in range(kv_cache.num_layers):
                            for kv_head in range(kv_cache.num_heads):
                                pool._invalidate_page((layer_idx, phys_page, kv_head))

                    if backup_phys_page is not None:
                        invalidate_phys(backup_phys_page)
                    for phys_page in new_pages:
                        invalidate_phys(phys_page)

                    if backup_pages:
                        for offset, data in backup_pages:
                            written = os.pwrite(pool.fd, data, offset)
                            if written != len(data):
                                raise RuntimeError(
                                    f"Short write when restoring kv_cache_pool.bin "
                                    f"at offset {offset}: {written} != {len(data)}"
                                )

                    if new_pages:
                        zero = b"\x00" * pool.page_bytes
                        for phys_page in new_pages:
                            for layer_idx in range(kv_cache.num_layers):
                                for kv_head in range(kv_cache.num_heads):
                                    offset = pool._page_offset(
                                        (layer_idx, phys_page, kv_head)
                                    )
                                    written = os.pwrite(pool.fd, zero, offset)
                                    if written != pool.page_bytes:
                                        raise RuntimeError(
                                            f"Short write when zeroing page "
                                            f"({layer_idx}, {phys_page}, {kv_head}): "
                                            f"{written} != {pool.page_bytes}"
                                        )

                    kv_cache.seq_len = prefix_len_for_restore
                    kv_cache.active_indices = list(prefix_active_indices)
                    kv_cache.free_slots = set(range(kv_cache.capacity)) - set(
                        prefix_active_indices
                    )
                    kv_cache.active_buffers_mx.clear()
                    kv_cache._last_page_written_layers.clear()
                    kv_cache.page_cached_all.fill(False)
                    if new_pages:
                        quest_controller.metadata_pool[:, new_pages] = 0
                elif quest_controller is not None and not load_path and not save_path:
                    quest_controller.clean_states()

        except Exception as e:
            rqueue.put(e)


class NullPromptCache:
    def fetch_nearest_cache(self, model, tokens):
        return None, tokens

    def insert_cache(self, model, tokens, prompt_cache):
        return None


def run(
    host: str,
    port: int,
    model_provider: AlayaModelProvider,
    *,
    prefill_step_size: int,
    allow_batch: bool,
    prompt_cache,
    server_class=ThreadingHTTPServer,
    handler_class=base_server.APIHandler,
):
    server_address = (host, port)
    response_generator = AlayaResponseGenerator(
        model_provider,
        prompt_cache,
        prefill_step_size=prefill_step_size,
        allow_batch=allow_batch,
    )
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            response_generator,
            system_fingerprint=base_server.get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "alayajet.api_server is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        response_generator.stop_and_join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlayaJet OpenAI-compatible HTTP server."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-7B-Instruct-1M-4bit",
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.set_defaults(use_default_chat_template=True)
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="JSON formatted args for apply_chat_template, e.g. '{\"enable_thinking\":false}'",
        default="{}",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help=f"Prefill step size for prompt processing (default: {DEFAULT_PREFILL_STEP_SIZE})",
    )
    parser.add_argument(
        "--allow-batch",
        action="store_true",
        help="Allow request batching (disabled by default to honor prefill step size).",
    )
    parser.add_argument("--quest", action="store_true", help="Enable Quest Feature")
    parser.set_defaults(quest=True)
    parser.add_argument(
        "--page-budget",
        type=int,
        default=DEFAULT_PAGE_BUDGET,
        help="Quest page budget",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./kv_cache_eval",
        help="Quest cache directory",
    )
    parser.add_argument(
        "--quest-reset-cache",
        action="store_true",
        help="Clear the Quest cache directory on startup",
    )
    parser.add_argument(
        "--quest-no-reset-cache",
        dest="quest_reset_cache",
        action="store_false",
        help="Do not clear the Quest cache directory on startup (default)",
    )
    parser.set_defaults(quest_reset_cache=False)
    parser.set_defaults(quest_disable_os_cache=False)
    parser.add_argument(
        "--quest-async-disk-write",
        dest="quest_async_disk_write",
        action="store_true",
        default=True,
        help="Enable async disk write for Quest (default)",
    )
    parser.add_argument(
        "--quest-sync-disk-write",
        dest="quest_async_disk_write",
        action="store_false",
        help="Disable async disk write for Quest",
    )
    parser.add_argument(
        "--quest-disable-os-cache",
        dest="quest_disable_os_cache",
        action="store_true",
        help="Disable OS page cache for Quest KV backing file",
    )
    parser.add_argument(
        "--quest-enable-os-cache",
        dest="quest_disable_os_cache",
        action="store_false",
        help="Allow OS page cache for Quest KV backing file (default)",
    )
    parser.add_argument("--timing", action="store_true", help="Enable baseline timing")
    parser.add_argument(
        "--timing-sync",
        dest="timing_sync",
        action="store_true",
        default=True,
        help="Sync MLX ops for accurate timing (default)",
    )
    parser.add_argument(
        "--timing-no-sync",
        dest="timing_sync",
        action="store_false",
        help="Do not sync MLX ops (lower overhead, less accurate)",
    )
    parser.add_argument(
        "--quest-timing", action="store_true", help="Enable Quest timing breakdown"
    )
    parser.add_argument(
        "--quest-timing-sync",
        dest="quest_timing_sync",
        action="store_true",
        default=True,
        help="Sync MLX ops for accurate timing (default)",
    )
    parser.add_argument(
        "--quest-timing-no-sync",
        dest="quest_timing_sync",
        action="store_false",
        help="Do not sync MLX ops (lower overhead, less accurate)",
    )
    parser.add_argument(
        "--quest-trace-output",
        type=str,
        default="",
        help="Write Quest timing trace (e.g. timeline.svg or trace.json)",
    )
    parser.add_argument(
        "--quest-trace-decode-steps",
        type=int,
        default=3,
        help="Number of decode steps to include in Quest trace (<=0 disables decode tracing)",
    )
    parser.add_argument(
        "--quest-prefill-layer-log",
        action="store_true",
        help="Log per-layer prefill time for Quest",
    )
    parser.add_argument(
        "--quest-prefill-io-log",
        action="store_true",
        help="Log per-chunk prefill read/attn overlap for Quest",
    )
    parser.add_argument(
        "--quest-lru-log",
        action="store_true",
        help="Log per-step LRU hit rate during Quest decode",
    )
    parser.add_argument(
        "--quest-mem-log",
        action="store_true",
        help="Log Quest memory checkpoints during prefill",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if mx.metal.is_available():
        wired_limit = mx.metal.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addFilter(
        lambda record: "Prompt processing progress:" not in record.getMessage()
    )

    if args.quest and args.timing:
        logging.warning(
            "--timing is ignored when --quest is enabled. "
            "Use --quest-timing for Quest-specific breakdowns."
        )

    run(
        args.host,
        args.port,
        AlayaModelProvider(args),
        prefill_step_size=args.prefill_step_size,
        allow_batch=args.allow_batch,
        prompt_cache=NullPromptCache(),
    )


if __name__ == "__main__":
    main()
