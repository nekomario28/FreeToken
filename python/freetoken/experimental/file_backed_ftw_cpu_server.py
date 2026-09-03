"""Opt-in server launcher for file-backed FTW expert sources on the CPU MoE backend.

Normal ``ft serve`` is untouched.  This module patches only the spawned scheduler worker,
forces the existing split-residency planning path, replaces the engine's expert-bank loader
with the explicit file-backed FTW provider, and exposes one read-only owner-native runtime
identity route for this experimental server only.  It is intentionally fail-closed: an
unsupported checkpoint/backend never falls back to the anonymous HostBank loader.

This launcher does not weaken resource admission.  A real model must still pass a separate,
architecture-aware load gate before this entry point is executed against real weights.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any


def _option_value(argv: Sequence[str], name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1] if index + 1 < len(argv) else None
        prefix = name + "="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _required_model_path(argv: Sequence[str], *, prog: str) -> str:
    """Return the one exact model path declared through either supported alias.

    Repeated aliases are accepted only when they resolve to the same literal value.  This keeps
    the identity producer from hashing one directory while the canonical parser later serves
    another one.
    """
    values: list[str] = []
    names = ("--model", "--model-path")
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name:
                if index + 1 >= len(argv) or not argv[index + 1]:
                    raise SystemExit(f"{prog}: error: {name} requires a value")
                values.append(argv[index + 1])
            else:
                prefix = name + "="
                if arg.startswith(prefix):
                    value = arg[len(prefix):]
                    if not value:
                        raise SystemExit(f"{prog}: error: {name} requires a value")
                    values.append(value)
    if not values:
        raise SystemExit(f"{prog}: error: explicit --model/--model-path is required")
    unique = set(values)
    if len(unique) != 1:
        raise SystemExit(
            f"{prog}: error: model path aliases disagree; runtime identity would be ambiguous"
        )
    return values[0]


def _require_explicit_cpu_backend(argv: Sequence[str], *, prog: str) -> None:
    value = _option_value(argv, "--moe-backend")
    if value != "cpu":
        raise SystemExit(
            f"{prog}: error: explicit --moe-backend cpu is required for the "
            "file-backed FTW experiment"
        )


def _make_file_backed_loader(provider: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt the canonical ``load_expert_banks`` call shape to the explicit provider."""

    def load_expert_banks(
        model_path,
        model_config,
        *,
        device,
        dtype,
        dummy: bool = False,
        parallel=None,
        workers: int = 8,
        chunk: int = 8 << 20,
        decode_target: str = "gpu",
        layer_sink=None,
        layer_residency=None,
    ):
        # These arguments exist in the canonical provider contract but have no valid meaning
        # for a zero-copy file-backed FTW source. Reject semantic mismatches rather than
        # silently switching back to materialized host banks.
        del device, dtype, parallel, workers, chunk
        if dummy:
            raise ValueError("file-backed FTW experiment does not support dummy expert banks")
        if layer_sink is not None:
            raise ValueError("file-backed FTW experiment is a serving source, not a converter sink")
        return provider(
            model_path,
            model_config,
            decode_target=decode_target,
            layer_residency=layer_residency,
        )

    return load_expert_banks


def _apply_worker_patch() -> None:
    """Install the experiment only inside the spawned scheduler worker."""
    import freetoken.engine.engine as engine_mod
    from freetoken.checkpoint.file_backed_ftw_cpu import load_file_backed_ftw_cpu_banks

    # Plain Linux normally has an uncapped pin budget. A zero experimental budget activates
    # the engine's already-existing CPU split-residency plan, which disables prefill overlap
    # before the expert source is attached. No production default is changed.
    engine_mod._pin_budget_bytes = lambda: 0  # type: ignore[attr-defined]
    engine_mod.load_expert_banks = _make_file_backed_loader(  # type: ignore[assignment]
        load_file_backed_ftw_cpu_banks
    )


def _run_scheduler_file_backed(args, ack_queue) -> None:
    _apply_worker_patch()
    from freetoken.server.launch import _run_scheduler

    _run_scheduler(args, ack_queue)


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    display_prog = prog or "python -m freetoken.experimental.file_backed_ftw_cpu_server"
    _require_explicit_cpu_backend(effective_argv, prog=display_prog)
    model_path = _required_model_path(effective_argv, prog=display_prog)

    # Seal the exact local artifact before any server/backend is started. The identity is derived
    # from checkpoint bytes, never accepted as a CLI claim, and cached for the process lifetime.
    from freetoken.experimental.file_backed_ftw_runtime_identity import (
        compute_ftw_artifact_identity,
        register_runtime_identity_route,
        unregister_runtime_identity_route,
    )

    artifact_identity = compute_ftw_artifact_identity(model_path)

    from freetoken.server import api_server as server_api
    from freetoken.server import launch as server_launch

    route = register_runtime_identity_route(
        server_api.app,
        server_api.get_global_state,
        artifact_identity,
    )
    original = server_launch._run_scheduler
    server_launch._run_scheduler = _run_scheduler_file_backed
    try:
        server_launch.launch_server(
            run_shell=run_shell,
            argv=effective_argv,
            prog=display_prog,
        )
    finally:
        # Keep embedding/tests deterministic if argument parsing, startup, or shutdown returns.
        server_launch._run_scheduler = original
        unregister_runtime_identity_route(server_api.app, route)


def main() -> None:
    launch_server()


if __name__ == "__main__":
    main()


__all__ = [
    "_make_file_backed_loader",
    "_option_value",
    "_required_model_path",
    "_require_explicit_cpu_backend",
    "launch_server",
]
