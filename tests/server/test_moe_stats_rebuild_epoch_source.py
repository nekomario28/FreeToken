from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
API_SERVER = ROOT / "python/freetoken/server/api_server.py"


def _resolve_rebuild_function():
    """Load the exact FrontendManager._resolve_rebuild body without importing server deps."""
    tree = ast.parse(API_SERVER.read_text(encoding="utf-8"), filename=str(API_SERVER))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FrontendManager"
    )
    fn = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_rebuild"
    )
    fn.decorator_list = []
    fn.returns = None
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    if fn.args.vararg is not None:
        fn.args.vararg.annotation = None
    if fn.args.kwarg is not None:
        fn.args.kwarg.annotation = None
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(API_SERVER), "exec"), namespace)
    return namespace["_resolve_rebuild"]


class _Stats:
    def __init__(self) -> None:
        self.moe_layer_calls = 80
        self.moe_active_experts = 640
        self.moe_missing_experts = 160
        self.moe_fetched_experts = 120
        self.reset_calls = 0

    def reset_moe_snapshot(self) -> None:
        self.reset_calls += 1
        self.moe_layer_calls = 0
        self.moe_active_experts = 0
        self.moe_missing_experts = 0
        self.moe_fetched_experts = 0


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        last_rebuild={"status": "ok", "moe_cache_size": 128},
        rebuild_futures={},
        fatal_error=None,
        maintenance_state="rebuilding",
        stats=_Stats(),
    )


def _reply(*, status: str, moe_cache_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="rebuild-1",
        status=status,
        moe_cache_size=moe_cache_size,
        num_pages=8192,
        mamba_slots=0,
        num_swa_pages=0,
        error=None,
    )


class MoeStatsRebuildEpochTests(unittest.TestCase):
    def test_successful_moe_cache_rebuild_invalidates_frontend_snapshot(self) -> None:
        resolve = _resolve_rebuild_function()
        state = _state()

        # A MoE cache resize cold-starts OffloadMoeCache's device counters. The frontend
        # must invalidate its last throttled cumulative snapshot at the same terminal
        # rebuild boundary; otherwise /v1/stats temporarily labels pre-rebuild counts as
        # decode_since_start_or_cache_rebuild.
        resolve(state, _reply(status="ok", moe_cache_size=256))

        self.assertEqual(state.maintenance_state, "serving")
        self.assertEqual(state.last_rebuild["status"], "ok")
        self.assertEqual(
            state.stats.reset_calls,
            1,
            "successful MoE cache rebuild left the pre-rebuild frontend snapshot live",
        )
        self.assertEqual(state.stats.moe_layer_calls, 0)

    def test_failed_rebuild_does_not_discard_last_valid_snapshot(self) -> None:
        resolve = _resolve_rebuild_function()
        state = _state()

        resolve(state, _reply(status="failed", moe_cache_size=256))

        self.assertEqual(state.maintenance_state, "failed")
        self.assertEqual(state.stats.reset_calls, 0)
        self.assertEqual(state.stats.moe_layer_calls, 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
