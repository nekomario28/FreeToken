from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[2]
API_SERVER = ROOT / "python/freetoken/server/api_server.py"
STATS = ROOT / "python/freetoken/server/stats.py"


def _strip_annotations(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    fn.decorator_list = []
    fn.returns = None
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    if fn.args.vararg is not None:
        fn.args.vararg.annotation = None
    if fn.args.kwarg is not None:
        fn.args.kwarg.annotation = None


def _extract_method(path: Path, class_name: str, method_name: str, namespace: dict[str, object]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fn = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    _strip_annotations(fn)
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _extract_top_level(path: Path, function_name: str, namespace: dict[str, object]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    _strip_annotations(fn)
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


class _CacheRebuildMsg:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _DispatchState:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.rebuild_futures: dict[str, object] = {}
        self.moe_stats_reset_rebuilds: set[str] = set()
        self.maintenance_state = "serving"
        self.fail_send = fail_send
        self.sent: list[object] = []

    async def send_one(self, msg: object) -> None:
        self.sent.append(msg)
        if self.fail_send:
            raise RuntimeError("synthetic enqueue failure")


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


def _resolve_state(*, marked: bool = True) -> SimpleNamespace:
    marker = {"rebuild-1"} if marked else set()
    return SimpleNamespace(
        last_rebuild={"status": "ok", "moe_cache_size": 128},
        rebuild_futures={},
        moe_stats_reset_rebuilds=marker,
        fatal_error=None,
        maintenance_state="rebuilding",
        stats=_Stats(),
    )


def _reply(*, status: str, moe_cache_size: int = 256) -> SimpleNamespace:
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
    def test_stats_reset_primitive_zeroes_only_moe_snapshot_fields(self) -> None:
        reset = _extract_method(STATS, "StatsTracker", "reset_moe_snapshot", {})
        state = SimpleNamespace(
            moe_layer_calls=80,
            moe_active_experts=640,
            moe_missing_experts=160,
            moe_fetched_experts=120,
            completed=9,
        )

        reset(state)

        self.assertEqual(
            (
                state.moe_layer_calls,
                state.moe_active_experts,
                state.moe_missing_experts,
                state.moe_fetched_experts,
            ),
            (0, 0, 0, 0),
        )
        self.assertEqual(state.completed, 9)

    def test_moe_rebuild_timeout_keeps_intent_for_eventual_terminal_reply(self) -> None:
        dispatch = _extract_top_level(
            API_SERVER,
            "dispatch_rebuild",
            {"asyncio": asyncio, "uuid": uuid, "CacheRebuildMsg": _CacheRebuildMsg},
        )
        state = _DispatchState()

        result = asyncio.run(
            dispatch(
                state,
                moe_cache_size=256,
                num_pages=None,
                num_mamba_slots=None,
                num_swa_pages=None,
                mode="if_idle",
                timeout=0.001,
            )
        )

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(state.maintenance_state, "rebuilding")
        self.assertEqual(state.rebuild_futures, {})
        self.assertEqual(state.moe_stats_reset_rebuilds, {result["request_id"]})

    def test_dispatch_failure_removes_moe_rebuild_intent(self) -> None:
        dispatch = _extract_top_level(
            API_SERVER,
            "dispatch_rebuild",
            {"asyncio": asyncio, "uuid": uuid, "CacheRebuildMsg": _CacheRebuildMsg},
        )
        state = _DispatchState(fail_send=True)

        result = asyncio.run(
            dispatch(
                state,
                moe_cache_size=256,
                num_pages=None,
                num_mamba_slots=None,
                num_swa_pages=None,
                mode="if_idle",
                timeout=0.001,
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(state.maintenance_state, "serving")
        self.assertEqual(state.rebuild_futures, {})
        self.assertEqual(state.moe_stats_reset_rebuilds, set())

    def test_successful_moe_cache_rebuild_invalidates_frontend_snapshot(self) -> None:
        resolve = _extract_method(API_SERVER, "FrontendManager", "_resolve_rebuild", {})
        state = _resolve_state(marked=True)

        resolve(state, _reply(status="ok"))

        self.assertEqual(state.maintenance_state, "serving")
        self.assertEqual(state.last_rebuild["status"], "ok")
        self.assertEqual(
            state.stats.reset_calls,
            1,
            "successful MoE cache rebuild left the pre-rebuild frontend snapshot live",
        )
        self.assertEqual(state.stats.moe_layer_calls, 0)
        self.assertNotIn("rebuild-1", state.moe_stats_reset_rebuilds)

    def test_failed_moe_rebuild_consumes_intent_without_discarding_snapshot(self) -> None:
        resolve = _extract_method(API_SERVER, "FrontendManager", "_resolve_rebuild", {})
        state = _resolve_state(marked=True)

        resolve(state, _reply(status="failed"))

        self.assertEqual(state.maintenance_state, "failed")
        self.assertEqual(state.stats.reset_calls, 0)
        self.assertEqual(state.stats.moe_layer_calls, 80)
        self.assertNotIn("rebuild-1", state.moe_stats_reset_rebuilds)

    def test_successful_non_moe_rebuild_keeps_last_moe_snapshot(self) -> None:
        resolve = _extract_method(API_SERVER, "FrontendManager", "_resolve_rebuild", {})
        state = _resolve_state(marked=False)

        resolve(state, _reply(status="ok"))

        self.assertEqual(state.maintenance_state, "serving")
        self.assertEqual(state.stats.reset_calls, 0)
        self.assertEqual(state.stats.moe_layer_calls, 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
