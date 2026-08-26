"""Experimental server launcher for an all-CPU, file-backed FTW load path.

Not wired into ``ft serve``. The launcher fails closed unless ``--moe-cpu-layers
1.0`` is explicit and the checkpoint is already FTW. Inside the spawned worker it
replaces two host-memory-heavy FTW seams only:

* dense weights use a direct file-backed iterator with no anonymous prefetch;
* native NVFP4 expert banks use PAGEABLE MAP_PRIVATE per-layer tensors.

Every normal FreeToken launch/default remains unchanged.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from freetoken.experimental.pageable_cpu_moe_server import _cpu_layer_spec


def _requires_all_cpu(spec: str | None) -> bool:
    if spec is None:
        return False
    try:
        return float(spec.strip()) == 1.0
    except (TypeError, ValueError):
        return False


def _make_filebacked_expert_loader(original: Callable[..., Any]) -> Callable[..., Any]:
    def load(
        model_path,
        model_config,
        *,
        device,
        dtype,
        dummy=False,
        parallel=None,
        workers=8,
        chunk=8 << 20,
        decode_target="gpu",
        layer_sink=None,
        layer_residency=None,
    ):
        # This experiment has no quiet fallback to the ordinary 18+ GiB anonymous
        # bank path. Any mismatch is a hard stop before expert payload IO.
        if dummy:
            raise RuntimeError("file-backed CPU-MoE experiment does not support dummy weights")
        if layer_sink is not None:
            raise RuntimeError("file-backed CPU-MoE experiment is serving-only, not conversion")
        if decode_target != "cpu":
            raise RuntimeError(
                f"file-backed CPU-MoE experiment requires decode_target='cpu', got {decode_target!r}"
            )

        from freetoken.checkpoint.ftw import is_ftw_checkpoint
        from freetoken.experimental.ftw_filebacked_cpu_moe import (
            load_ftw_banks_filebacked_cpu,
        )

        if not model_path or not is_ftw_checkpoint(model_path):
            raise RuntimeError(
                "file-backed CPU-MoE experiment requires an already-converted FTW checkpoint"
            )
        if layer_residency is None:
            raise RuntimeError("file-backed CPU-MoE experiment requires explicit CPU layers")

        return load_ftw_banks_filebacked_cpu(
            model_path,
            num_layers=model_config.num_moe_layers,
            layer_residency=layer_residency,
        )

    return load


def _install_filebacked_dense_iterator(ftw_mod) -> None:
    """Install the low-RAM dense FTW reader on the worker-local FTW module."""
    from freetoken.experimental.ftw_filebacked_dense import iter_ftw_weights_filebacked

    ftw_mod.iter_ftw_weights = iter_ftw_weights_filebacked


def _apply_worker_patch() -> None:
    import freetoken.checkpoint.ftw as ftw_mod
    import freetoken.engine.engine as engine_mod

    # Dense model load happens before the expert cache is initialized. load_weight()
    # imports iter_ftw_weights from this module when called, so patch the module before
    # Engine construction rather than trying to replace the already-imported load_weight.
    _install_filebacked_dense_iterator(ftw_mod)

    # Plain Linux otherwise sees an uncapped pin budget and can leave explicit CPU
    # layers pinned. Zero activates the existing split-residency accounting; the
    # experimental expert loader then maps all requested CPU layers file-backed PAGEABLE.
    engine_mod._pin_budget_bytes = lambda: 0  # type: ignore[attr-defined]
    engine_mod.load_expert_banks = _make_filebacked_expert_loader(  # type: ignore[assignment]
        engine_mod.load_expert_banks
    )


def _run_scheduler_filebacked(args, ack_queue) -> None:
    _apply_worker_patch()
    from freetoken.server.launch import _run_scheduler

    _run_scheduler(args, ack_queue)


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    spec = _cpu_layer_spec(effective_argv)
    if not _requires_all_cpu(spec):
        raise SystemExit(
            f"{prog or 'ftw-filebacked-cpu-moe-server'}: error: this first prototype "
            "requires explicit --moe-cpu-layers 1.0"
        )

    from freetoken.server import launch as server_launch

    original = server_launch._run_scheduler
    server_launch._run_scheduler = _run_scheduler_filebacked
    try:
        server_launch.launch_server(run_shell=run_shell, argv=effective_argv, prog=prog)
    finally:
        server_launch._run_scheduler = original


def main() -> None:
    launch_server(prog="python -m freetoken.experimental.ftw_filebacked_cpu_moe_server")


if __name__ == "__main__":
    main()
