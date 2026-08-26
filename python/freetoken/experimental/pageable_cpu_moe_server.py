"""Experimental FreeToken server launcher for explicitly pageable CPU-MoE layers.

This is intentionally not wired into ``ft serve``.  It is a narrow experiment
surface for validating whether the already-supported PAGEABLE host-bank consumers
can reduce pinned-RAM pressure on plain Linux without changing production defaults.

Use only with an explicit ``--moe-cpu-layers`` selection.  The spawned scheduler
worker forces split-residency accounting and leaves the selected CPU layers
PAGEABLE instead of mlocking them; all other residency requests retain FreeToken's
original settle behavior.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any


def _cpu_layer_spec(argv: Sequence[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--moe-cpu-layers":
            return argv[i + 1] if i + 1 < len(argv) else None
        if arg.startswith("--moe-cpu-layers="):
            return arg.split("=", 1)[1]
    return None


def _make_pageable_settle(
    original_settle: Callable[[Any, str], None],
    *,
    locked_value: str,
) -> Callable[[Any, str], None]:
    """Return a settle function that intentionally leaves LOCKED requests pageable.

    The engine uses LOCKED only for CPU-selected expert-bank layers in its split
    residency plan.  PINNED and any future residency labels continue through the
    original implementation unchanged.
    """

    def settle(bank: Any, residency: str) -> None:
        if residency == locked_value:
            return
        original_settle(bank, residency)

    return settle


def _apply_worker_patch() -> None:
    # Import inside the spawned scheduler worker, after CUDA/process isolation.
    import freetoken.engine.engine as engine_mod
    import freetoken.moe.host_banks as host_banks

    # Plain Linux normally reports no pin quota, so an explicit --moe-cpu-layers
    # selection remains pinned.  A zero synthetic budget activates the existing
    # split-residency plan without changing which layers the user selected.
    engine_mod._pin_budget_bytes = lambda: 0  # type: ignore[attr-defined]

    # The existing loader already accepts PAGEABLE consumers.  The engine currently
    # requests LOCKED for CPU layers; for this experiment only, leave those banks as
    # their native anonymous-mmap PAGEABLE residency instead of relying on mlock
    # failure to obtain the same state.
    host_banks._settle = _make_pageable_settle(  # type: ignore[attr-defined]
        host_banks._settle,
        locked_value=host_banks.HostResidency.LOCKED.value,
    )


def _run_scheduler_pageable(args, ack_queue) -> None:
    _apply_worker_patch()
    # Under multiprocessing spawn this module is imported fresh in the child, so
    # freetoken.server.launch still contains its original scheduler implementation.
    from freetoken.server.launch import _run_scheduler

    _run_scheduler(args, ack_queue)


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    spec = _cpu_layer_spec(effective_argv)
    if spec is None or not spec.strip() or spec.strip() in {"0", "0.0"}:
        raise SystemExit(
            f"{prog or 'pageable-cpu-moe-server'}: error: an explicit non-zero "
            "--moe-cpu-layers selection is required"
        )

    from freetoken.server import launch as server_launch

    original = server_launch._run_scheduler
    server_launch._run_scheduler = _run_scheduler_pageable
    try:
        server_launch.launch_server(run_shell=run_shell, argv=effective_argv, prog=prog)
    finally:
        # Keep embedding/tests deterministic if launch exits before process teardown.
        server_launch._run_scheduler = original


def main() -> None:
    launch_server(prog="python -m freetoken.experimental.pageable_cpu_moe_server")


if __name__ == "__main__":
    main()
