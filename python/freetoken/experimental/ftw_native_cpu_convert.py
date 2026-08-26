"""Experimental FTW conversion wrapper that preserves native NVFP4 CPU rows.

The production converter chooses the NVFP4 serving backend for the conversion GPU.
That may physically repack expert banks into marlin/b12x, which a CPU executor cannot
read. This wrapper leaves the production converter untouched and, for the duration
of one explicit conversion call, forces its expert-bank load to use
``decode_target="cpu"``. Native NVFP4 is streamable per layer, so the resulting FTW
is the layout consumed by ``ftw_filebacked_cpu_moe``.

The experimental call also swaps only the converter module's writer symbol for a
NumPy-free FTWWriter subclass.  This removes ``tensor.numpy()`` from the low-RAM path
without altering the normal FTW writer or ``ft convert`` surface.

No conversion is started merely by importing this module.
"""
from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from typing import Any

import torch

_PATCH_LOCK = threading.Lock()


def _make_cpu_decode_expert_loader(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``load_expert_banks`` while preserving every argument except decode target."""

    def load(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["decode_target"] = "cpu"
        return original(*args, **kwargs)

    return load


def convert_checkpoint_native_cpu(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    shard_limit: int | None = None,
    device: str | None = None,
) -> dict:
    """Run the standard FTW converter with CPU-readable experts and NumPy-free writes.

    Both monkeypatches are process-local, serialized, and restored in ``finally``.  The
    writer replacement is scoped to ``freetoken.checkpoint.convert.FTWWriter`` rather
    than mutating the production FTWWriter class globally.  The resulting index must
    report native ``nvfp4`` expert banks or the experiment fails closed.
    """

    import freetoken.checkpoint.convert as convert_mod
    import freetoken.moe.expert_banks as expert_banks
    from freetoken.checkpoint.ftw import DEFAULT_SHARD_LIMIT
    from freetoken.experimental.ftw_numpyless_writer import NumpylessFTWWriter

    kwargs = {
        "dtype": dtype,
        "moe_backend": "offload",
        "shard_limit": DEFAULT_SHARD_LIMIT if shard_limit is None else shard_limit,
        "device": device,
    }

    with _PATCH_LOCK:
        original_loader = expert_banks.load_expert_banks
        original_writer = convert_mod.FTWWriter
        expert_banks.load_expert_banks = _make_cpu_decode_expert_loader(original_loader)
        convert_mod.FTWWriter = NumpylessFTWWriter
        try:
            index = convert_mod.convert_checkpoint(model_path, out_dir, **kwargs)
        finally:
            convert_mod.FTWWriter = original_writer
            expert_banks.load_expert_banks = original_loader

    if index.get("quant_format") != "nvfp4":
        raise RuntimeError(
            "native CPU FTW conversion expected quant_format='nvfp4', got "
            f"{index.get('quant_format')!r}; output is not admitted for the "
            "file-backed NVFP4 CPU runtime"
        )
    if not index.get("expert_bank_num_layers"):
        raise RuntimeError("native CPU FTW conversion produced no expert-bank layer metadata")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental native-NVFP4 NumPy-free FTW converter for file-backed CPU-MoE"
    )
    parser.add_argument("model_path")
    parser.add_argument("out_dir")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    convert_checkpoint_native_cpu(args.model_path, args.out_dir, device=args.device)


if __name__ == "__main__":
    main()
