from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(rel: str, old: str, new: str, label: str) -> None:
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already applied: {label}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one current-upstream anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"applied: {label}")


# docs/cli.md — preserve #117's new --gpu docs and add only the telemetry flag row.
replace_once(
    "docs/cli.md",
    """| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |\n| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |\n""",
    """| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |\n| `--moe-collect-stats` | off | Expose throttled decode cache-hit and logical H2D payload counters in `/v1/stats` / `ft ctl stats` |\n| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |\n""",
    "docs telemetry flag",
)

# server/args.py — EngineConfig already owns moe_collect_stats; expose it in the server CLI.
replace_once(
    "python/freetoken/server/args.py",
    """    parser.add_argument(\n        \"--moe-cache-policy\",\n        default=ServerArgs.moe_cache_policy,\n        choices=[\"lru\"],\n        help=\"The unified MoE cache eviction policy.\",\n    )\n\n    parser.add_argument(\n        \"--moe-cpu-threads\",\n""",
    """    parser.add_argument(\n        \"--moe-cache-policy\",\n        default=ServerArgs.moe_cache_policy,\n        choices=[\"lru\"],\n        help=\"The unified MoE cache eviction policy.\",\n    )\n\n    parser.add_argument(\n        \"--moe-collect-stats\",\n        action=\"store_true\",\n        default=ServerArgs.moe_collect_stats,\n        help=(\n            \"Collect decode expert-cache counters on-device and expose throttled \"\n            \"snapshots through /v1/stats and `ft ctl stats`.\"\n        ),\n    )\n\n    parser.add_argument(\n        \"--moe-cpu-threads\",\n""",
    "server telemetry CLI",
)

# scheduler/scheduler.py — snapshot device counters only when the existing decode status cadence emits.
replace_once(
    "python/freetoken/scheduler/scheduler.py",
    """        self.status_reporter.report_batch(\n            batch,\n            running_reqs=len(self.decode_manager.running_reqs),\n            queue_reqs=len(self.prefill_manager.pending_list),\n            kv_used_pages=used,\n            kv_total_pages=total,\n            page_size=self.config.page_size,\n            mamba_slots=mamba_slots,\n            swa_tokens=swa_tokens,\n        )\n        self.send_result(reply)\n""",
    """        decode_status_emitted = self.status_reporter.report_batch(\n            batch,\n            running_reqs=len(self.decode_manager.running_reqs),\n            queue_reqs=len(self.prefill_manager.pending_list),\n            kv_used_pages=used,\n            kv_total_pages=total,\n            page_size=self.config.page_size,\n            mamba_slots=mamba_slots,\n            swa_tokens=swa_tokens,\n        )\n        moe_cache = self.engine.moe_offload_cache\n        if (\n            reply\n            and decode_status_emitted\n            and moe_cache is not None\n            and moe_cache.collect_stats\n        ):\n            # One host read per decode status interval, never per layer/token. The\n            # counters are accumulated on-device inside the captured decode graph.\n            moe_stats = moe_cache.decode_miss_stats()\n            for m in reply:\n                m.moe_layer_calls = moe_stats[\"layer_calls\"]\n                m.moe_active_experts = moe_stats[\"active_experts\"]\n                m.moe_missing_experts = moe_stats[\"missing_experts\"]\n                m.moe_fetched_experts = moe_stats[\"fetched_experts\"]\n        self.send_result(reply)\n""",
    "scheduler throttled telemetry snapshot",
)

# server/api_server.py — bind frontend snapshot invalidation to the MoE rebuild request identity.
replace_once(
    "python/freetoken/server/api_server.py",
    """    rebuild_futures: Dict[str, asyncio.Future] = field(default_factory=dict)\n    # Lifecycle gate. Starts \"loading\" (uvicorn binds before weights finish; the three\n""",
    """    rebuild_futures: Dict[str, asyncio.Future] = field(default_factory=dict)\n    # MoE-resize requests whose successful terminal reply starts a new telemetry epoch.\n    # Keep request intent separate from final geometry: a same-size cold rebuild still resets\n    # device counters, and a timed-out HTTP waiter may receive its terminal reply later.\n    moe_stats_reset_rebuilds: set[str] = field(default_factory=set)\n    # Lifecycle gate. Starts \"loading\" (uvicorn binds before weights finish; the three\n""",
    "frontend rebuild telemetry intent field",
)
replace_once(
    "python/freetoken/server/api_server.py",
    """            \"error\": msg.error,\n        }\n        fut = self.rebuild_futures.pop(msg.request_id, None)\n""",
    """            \"error\": msg.error,\n        }\n        reset_moe_snapshot = msg.request_id in self.moe_stats_reset_rebuilds\n        self.moe_stats_reset_rebuilds.discard(msg.request_id)\n        if msg.status == \"ok\" and reset_moe_snapshot:\n            self.stats.reset_moe_snapshot()\n        fut = self.rebuild_futures.pop(msg.request_id, None)\n""",
    "successful rebuild frontend epoch reset",
)
replace_once(
    "python/freetoken/server/api_server.py",
    """        def _resolve_all() -> None:\n            for request_id in list(self.rebuild_futures):\n""",
    """        def _resolve_all() -> None:\n            # No terminal rebuild reply can arrive after the backend dies. Drop every\n            # telemetry-epoch intent, including one whose HTTP waiter already timed out.\n            self.moe_stats_reset_rebuilds.clear()\n            for request_id in list(self.rebuild_futures):\n""",
    "backend-death epoch-intent cleanup",
)
replace_once(
    "python/freetoken/server/api_server.py",
    """    state.rebuild_futures[request_id] = fut\n    state.maintenance_state = \"rebuilding\"\n""",
    """    state.rebuild_futures[request_id] = fut\n    if moe_cache_size is not None:\n        state.moe_stats_reset_rebuilds.add(request_id)\n    state.maintenance_state = \"rebuilding\"\n""",
    "MoE rebuild intent admission",
)
replace_once(
    "python/freetoken/server/api_server.py",
    """        state.rebuild_futures.pop(request_id, None)\n        state.maintenance_state = \"serving\"\n        return {\"status\": \"failed\", \"error\": f\"failed to dispatch rebuild: {e!r}\"}\n""",
    """        state.rebuild_futures.pop(request_id, None)\n        state.moe_stats_reset_rebuilds.discard(request_id)\n        state.maintenance_state = \"serving\"\n        return {\"status\": \"failed\", \"error\": f\"failed to dispatch rebuild: {e!r}\"}\n""",
    "dispatch-failure epoch-intent cleanup",
)

# server/stats.py — preserve #117's `gpus` block while layering in MoE snapshot state.
replace_once(
    "python/freetoken/server/stats.py",
    """        self.vram_bytes = 0\n\n    @property\n""",
    """        self.vram_bytes = 0\n        self.moe_layer_calls = 0\n        self.moe_active_experts = 0\n        self.moe_missing_experts = 0\n        self.moe_fetched_experts = 0\n\n    def reset_moe_snapshot(self) -> None:\n        \"\"\"Invalidate the last throttled MoE snapshot at a device-counter epoch boundary.\"\"\"\n        self.moe_layer_calls = 0\n        self.moe_active_experts = 0\n        self.moe_missing_experts = 0\n        self.moe_fetched_experts = 0\n\n    @property\n""",
    "stats tracker MoE snapshot state",
)
replace_once(
    "python/freetoken/server/stats.py",
    """        if getattr(reply, \"gpu_mem_bytes\", 0) > 0:\n            self.vram_bytes = reply.gpu_mem_bytes\n        if getattr(reply, \"finished\", False):\n""",
    """        if getattr(reply, \"gpu_mem_bytes\", 0) > 0:\n            self.vram_bytes = reply.gpu_mem_bytes\n        if getattr(reply, \"moe_layer_calls\", 0) > 0:\n            self.moe_layer_calls = reply.moe_layer_calls\n            self.moe_active_experts = reply.moe_active_experts\n            self.moe_missing_experts = reply.moe_missing_experts\n            self.moe_fetched_experts = reply.moe_fetched_experts\n        if getattr(reply, \"finished\", False):\n""",
    "stats tracker MoE observation",
)
replace_once(
    "python/freetoken/server/stats.py",
    """    swa = (\n        {\"used_pages\": tr.swa_used_tokens // sps, \"total_pages\": tr.swa_total_tokens // sps,\n         \"page_size\": sps}\n        if tr.swa_total_tokens > 0 else None\n    )\n    return {\n""",
    """    swa = (\n        {\"used_pages\": tr.swa_used_tokens // sps, \"total_pages\": tr.swa_total_tokens // sps,\n         \"page_size\": sps}\n        if tr.swa_total_tokens > 0 else None\n    )\n    unit_bytes = getattr(state, \"unit_bytes\", None) or {}\n    bytes_per_expert = int(unit_bytes.get(\"moe_bytes_per_expert\", 0) or 0)\n    moe = None\n    if tr.moe_layer_calls > 0:\n        hits = max(0, tr.moe_active_experts - tr.moe_missing_experts)\n        host_computed = max(0, tr.moe_missing_experts - tr.moe_fetched_experts)\n        moe = {\n            \"scope\": \"decode_since_start_or_cache_rebuild\",\n            \"layer_calls\": tr.moe_layer_calls,\n            \"active_experts\": tr.moe_active_experts,\n            \"cache_hits\": hits,\n            \"cache_misses\": tr.moe_missing_experts,\n            \"cache_hit_rate\": round(hits / tr.moe_active_experts, 4)\n            if tr.moe_active_experts else 0.0,\n            \"fetched_experts\": tr.moe_fetched_experts,\n            \"host_computed_experts\": host_computed,\n            \"bytes_per_expert\": bytes_per_expert,\n            # Logical row payload, not sampled PCIe bus utilization. Transport overhead\n            # and overlap require timing support that these counters do not provide.\n            \"h2d_payload_bytes\": tr.moe_fetched_experts * bytes_per_expert,\n        }\n    return {\n""",
    "stats document MoE derivation",
)
replace_once(
    "python/freetoken/server/stats.py",
    """        \"swa\": swa,\n        \"vram_bytes\": tr.vram_bytes,\n        \"gpus\": list(getattr(state, \"gpus\", None) or []),\n""",
    """        \"swa\": swa,\n        \"moe\": moe,\n        \"vram_bytes\": tr.vram_bytes,\n        \"gpus\": list(getattr(state, \"gpus\", None) or []),\n""",
    "stats document MoE field preserving #117 gpus",
)
