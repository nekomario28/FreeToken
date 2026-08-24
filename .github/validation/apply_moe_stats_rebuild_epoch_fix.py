from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        print(f"already applied: {path}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one source match in {path}, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path}")


replace_once(
    "python/freetoken/server/stats.py",
    """        self.moe_missing_experts = 0\n        self.moe_fetched_experts = 0\n\n    @property\n""",
    """        self.moe_missing_experts = 0\n        self.moe_fetched_experts = 0\n\n    def reset_moe_snapshot(self) -> None:\n        \"\"\"Invalidate the last throttled MoE snapshot at a device-counter epoch boundary.\"\"\"\n        self.moe_layer_calls = 0\n        self.moe_active_experts = 0\n        self.moe_missing_experts = 0\n        self.moe_fetched_experts = 0\n\n    @property\n""",
)

replace_once(
    "python/freetoken/server/api_server.py",
    """    rebuild_futures: Dict[str, asyncio.Future] = field(default_factory=dict)\n    # Lifecycle gate. Starts \"loading\" (uvicorn binds before weights finish; the three\n""",
    """    rebuild_futures: Dict[str, asyncio.Future] = field(default_factory=dict)\n    # MoE-resize requests whose successful terminal reply starts a new telemetry epoch.\n    # Keep request intent separate from final geometry: a same-size cold rebuild still resets\n    # device counters, and a timed-out HTTP waiter may receive its terminal reply later.\n    moe_stats_reset_rebuilds: set[str] = field(default_factory=set)\n    # Lifecycle gate. Starts \"loading\" (uvicorn binds before weights finish; the three\n""",
)

replace_once(
    "python/freetoken/server/api_server.py",
    """    state.rebuild_futures[request_id] = fut\n    state.maintenance_state = \"rebuilding\"\n    try:\n""",
    """    state.rebuild_futures[request_id] = fut\n    if moe_cache_size is not None:\n        state.moe_stats_reset_rebuilds.add(request_id)\n    state.maintenance_state = \"rebuilding\"\n    try:\n""",
)

replace_once(
    "python/freetoken/server/api_server.py",
    """        state.rebuild_futures.pop(request_id, None)\n        state.maintenance_state = \"serving\"\n        return {\"status\": \"failed\", \"error\": f\"failed to dispatch rebuild: {e!r}\"}\n""",
    """        state.rebuild_futures.pop(request_id, None)\n        state.moe_stats_reset_rebuilds.discard(request_id)\n        state.maintenance_state = \"serving\"\n        return {\"status\": \"failed\", \"error\": f\"failed to dispatch rebuild: {e!r}\"}\n""",
)

replace_once(
    "python/freetoken/server/api_server.py",
    """        self.last_rebuild = {\n            \"request_id\": msg.request_id,\n            \"status\": msg.status,\n            \"moe_cache_size\": msg.moe_cache_size,\n            \"num_pages\": msg.num_pages,\n            \"mamba_slots\": msg.mamba_slots,\n            \"num_swa_pages\": msg.num_swa_pages,\n            \"error\": msg.error,\n        }\n        fut = self.rebuild_futures.pop(msg.request_id, None)\n""",
    """        self.last_rebuild = {\n            \"request_id\": msg.request_id,\n            \"status\": msg.status,\n            \"moe_cache_size\": msg.moe_cache_size,\n            \"num_pages\": msg.num_pages,\n            \"mamba_slots\": msg.mamba_slots,\n            \"num_swa_pages\": msg.num_swa_pages,\n            \"error\": msg.error,\n        }\n        reset_moe_snapshot = msg.request_id in self.moe_stats_reset_rebuilds\n        self.moe_stats_reset_rebuilds.discard(msg.request_id)\n        if msg.status == \"ok\" and reset_moe_snapshot:\n            self.stats.reset_moe_snapshot()\n        fut = self.rebuild_futures.pop(msg.request_id, None)\n""",
)

replace_once(
    "python/freetoken/server/api_server.py",
    """        def _resolve_all() -> None:\n            for request_id in list(self.rebuild_futures):\n                fut = self.rebuild_futures.pop(request_id, None)\n""",
    """        def _resolve_all() -> None:\n            # No terminal rebuild reply can arrive after the backend dies. Drop every\n            # telemetry-epoch intent, including one whose HTTP waiter already timed out.\n            self.moe_stats_reset_rebuilds.clear()\n            for request_id in list(self.rebuild_futures):\n                fut = self.rebuild_futures.pop(request_id, None)\n""",
)
