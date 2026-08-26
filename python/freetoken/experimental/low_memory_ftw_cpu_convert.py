"""Opt-in low-memory native-NVFP4 -> FTW conversion for the CPU file-backed experiment.

This is deliberately separate from ``ft checkpoint``.  The canonical converter may choose a
GPU-specific NVFP4 layout (marlin/b12x); silently replacing that with native CPU-readable
NVFP4 would change normal serving semantics.  This module instead provides an explicit
Qwen3.5-MoE/native-NVFP4 path whose safety properties are different and intentionally narrow:

* default CLI action is **preflight only**; real conversion requires ``--execute``;
* preflight inspects lightweight HF config + safetensors headers before importing the Qwen
  runtime/model/weight stack;
* the resource gate runs before any output is created;
* routed experts use the strict one-layer streamer, so only one native expert-bank layer is
  resident at a time;
* the checkpoint is built in a hidden sibling staging directory and published with one rename
  only after ``FTWWriter.finalize`` succeeds;
* a failed conversion closes the writer and removes the staging directory, leaving the final
  output path unpublished.

The first production target is the HF ``Qwen3_5MoeForConditionalGeneration`` family with
routed experts detected as native NVFP4.  Other architectures/formats fail closed until they
have an explicit source-spec contract and equivalent tests.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from .ftw_resource_gate import preflight_low_memory_nvfp4_conversion

_DEFAULT_SHARD_LIMIT = 8 << 30
_SUPPORTED_ARCH = "Qwen3_5MoeForConditionalGeneration"
_CONVERSION_TARGET = "cpu_file_backed_native_nvfp4"


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _architectures_from_hf(hf_config: Any) -> list[str]:
    architectures = list(_get(hf_config, "architectures", None) or [])
    text = _get(hf_config, "text_config", None)
    if not architectures and text is not None:
        architectures = list(_get(text, "architectures", None) or [])
    return architectures


def _routed_expert_quant_from_hf(hf_config: Any) -> str:
    """Tiny read-only copy of the Qwen3.5 routed-expert quant detection contract."""
    quant = _get(hf_config, "quantization_config", None)
    if quant is None:
        return "none"
    algo = str(_get(quant, "quant_algo", None) or _get(quant, "quant_method", None) or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" in algo:
        layers = _get(quant, "quantized_layers", None) or {}
        if isinstance(layers, dict):
            for name, layer_spec in layers.items():
                if not (name.endswith(".mlp.experts") or ".mlp.experts." in name):
                    continue
                layer_algo = str(_get(layer_spec or {}, "quant_algo", "")).lower()
                if "fp4" in layer_algo:
                    return "nvfp4"
                if "fp8" in layer_algo:
                    return "fp8"
    return "none"


def _preflight_config_from_hf(hf_config: Any):
    """Extract only fields needed by the resource gate, without model/runtime imports."""
    architectures = _architectures_from_hf(hf_config)
    if _SUPPORTED_ARCH not in architectures:
        raise ValueError(
            f"experimental low-memory converter supports {_SUPPORTED_ARCH!r} only, "
            f"checkpoint reports architectures={architectures!r}"
        )
    if _routed_expert_quant_from_hf(hf_config) != "nvfp4":
        raise ValueError("Qwen3.5 low-memory conversion requires routed experts in native NVFP4")

    text = _get(hf_config, "text_config", None) or hf_config
    num_experts = int(_get(text, "num_experts", 0) or 0)
    hidden_size = int(_get(text, "hidden_size", 0) or 0)
    moe_intermediate_size = int(_get(text, "moe_intermediate_size", 0) or 0)
    num_layers = int(
        _get(text, "num_hidden_layers", None)
        or _get(text, "num_layers", 0)
        or 0
    )
    if min(num_experts, hidden_size, moe_intermediate_size, num_layers) <= 0:
        raise ValueError(
            "Qwen3.5 preflight config is missing positive num_experts/hidden_size/"
            "moe_intermediate_size/num_hidden_layers"
        )
    return SimpleNamespace(
        architectures=architectures,
        is_moe=True,
        expert_quant="nvfp4",
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_moe_layers=num_layers,
        num_layers=num_layers,
    )


def _resolve_qwen35_preflight_config(model_path: str):
    from freetoken.utils import cached_load_hf_config

    return _preflight_config_from_hf(cached_load_hf_config(model_path))


def _resolve_qwen35_runtime_config(model_path: str, preflight_config: Any):
    """Import the real Qwen runtime parser only after resource admission succeeded."""
    from freetoken.utils import cached_load_hf_config
    from freetoken.models.qwen3_5_moe.config import parse_config

    model_config = parse_config(cached_load_hf_config(model_path))
    if not getattr(model_config, "is_moe", False):
        raise ValueError("Qwen3.5 low-memory conversion requires an MoE checkpoint")
    if getattr(model_config, "expert_quant", None) != "nvfp4":
        raise ValueError(
            "Qwen3.5 runtime parser disagrees with preflight: expected expert_quant='nvfp4', got "
            f"{getattr(model_config, 'expert_quant', None)!r}"
        )
    for name in ("num_experts", "hidden_size", "moe_intermediate_size", "num_moe_layers"):
        before = int(getattr(preflight_config, name))
        after = int(getattr(model_config, name))
        if before != after:
            raise RuntimeError(
                f"Qwen3.5 config changed between preflight and runtime parse: {name} "
                f"{before} -> {after}"
            )
    return model_config


def _abort_writer(writer: Any) -> None:
    """Best-effort close for an FTWWriter that failed before ``finalize``."""
    handle = getattr(writer, "_f", None)
    if handle is not None:
        try:
            handle.close()
        finally:
            try:
                writer._f = None
            except Exception:
                pass


def _staging_dir(output: Path) -> Path:
    return output.parent / f".{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"


def _write_native_nvfp4_ftw(
    model_path: str,
    out_dir: str,
    model_config: Any,
    spec: Any,
    *,
    dense_weights: Callable[[], Iterable[tuple[str, Any]]],
    expert_streamer: Callable[..., dict[str, int]],
    writer_factory: Callable[..., Any],
    metadata_copier: Callable[[str, str], list[str]],
    drop_page_cache: Callable[[str], None],
    fingerprint: str | None,
    shard_limit: int = _DEFAULT_SHARD_LIMIT,
    alloc_layer: Callable[[int, int, int], dict] | None = None,
) -> dict:
    """Build then atomically publish one CPU-target FTW checkpoint.

    Resource admission is intentionally *outside* this low-level function so synthetic tests
    can exercise the writer/publish boundary with tiny fixtures.  The public execution path
    below always calls the preflight gate first.
    """
    output = Path(out_dir)
    if output.exists():
        raise ValueError(
            f"execution requires a non-existing output path for atomic publish: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_dir(output)
    if staging.exists():
        raise RuntimeError(f"unexpected pre-existing staging directory: {staging}")

    writer = None
    published = False
    try:
        writer = writer_factory(str(staging), shard_limit=shard_limit)
        dense_count = 0
        dense_bytes = 0
        for name, tensor in dense_weights():
            writer.add_tensor(name, tensor, kind="weight")
            dense_count += 1
            dense_bytes += int(tensor.numel()) * int(tensor.element_size())

        expert_stats = expert_streamer(
            model_path,
            model_config,
            spec,
            writer=writer,
            drop_page_cache=drop_page_cache,
            alloc_layer=alloc_layer,
        )
        num_layers = int(expert_stats["layers_streamed"])
        expected_layers = int(model_config.num_moe_layers)
        if num_layers != expected_layers:
            raise RuntimeError(
                f"strict NVFP4 streamer wrote {num_layers} layers, expected {expected_layers}"
            )
        expert_entries = int(expert_stats["ftw_entries_written"])
        expert_bytes = int(expert_stats["ftw_expert_bytes_written"])
        if expert_entries <= 0 or expert_bytes <= 0:
            raise RuntimeError("strict NVFP4 streamer produced no expert-bank payload")

        copied = metadata_copier(model_path, str(staging))
        index = writer.finalize({
            "source_model_path": os.path.abspath(model_path),
            "fingerprint": fingerprint,
            "quant_format": "nvfp4",
            "expert_bank_num_layers": num_layers,
            "conversion_target": _CONVERSION_TARGET,
            "counts": {"weight": dense_count, "experts_bank": expert_entries},
            "bytes": {"weight": dense_bytes, "experts_bank": expert_bytes},
            "copied_metadata": copied,
        })

        # The final path remains absent until every shard, index, and metadata file is closed.
        os.rename(staging, output)
        published = True
        return index
    except BaseException:
        if writer is not None:
            _abort_writer(writer)
        raise
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def preflight_qwen35_native_nvfp4(model_path: str, out_dir: str) -> dict[str, Any]:
    """Read-only admission report.  Never creates the output directory."""
    model_config = _resolve_qwen35_preflight_config(model_path)
    report = preflight_low_memory_nvfp4_conversion(model_path, out_dir, model_config)
    return {
        "mode": "preflight",
        "architecture": _SUPPORTED_ARCH,
        "conversion_target": _CONVERSION_TARGET,
        "model_path": os.path.abspath(model_path),
        "out_dir": os.path.abspath(out_dir),
        "resources": report.as_dict(),
    }


def execute_qwen35_native_nvfp4(
    model_path: str,
    out_dir: str,
    *,
    shard_limit: int = _DEFAULT_SHARD_LIMIT,
) -> dict[str, Any]:
    """Run the admitted experimental conversion.  This is never the CLI default."""
    preflight_config = _resolve_qwen35_preflight_config(model_path)
    report = preflight_low_memory_nvfp4_conversion(model_path, out_dir, preflight_config)
    if Path(out_dir).exists():
        # The generic gate permits an empty directory for inspection workflows. Atomic publish
        # does not: refusing it avoids deleting/replacing a user-owned directory.
        raise ValueError("--execute requires --out to not exist; choose a fresh output path")

    # Runtime/model/kernel imports begin only after the read-only resource gate is green.
    import torch

    model_config = _resolve_qwen35_runtime_config(model_path, preflight_config)

    # Match canonical FTW conversion's single-process invariant before loading dense weights.
    from freetoken.distributed import set_tp_info, try_get_tp_info

    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
        tp = try_get_tp_info()
    if tp is None or tp.size != 1:
        raise RuntimeError(
            "experimental FTW conversion requires TP size 1 before weight loading"
        )

    from freetoken.checkpoint.convert import _copy_metadata, _source_fingerprint
    from freetoken.checkpoint.ftw import FTWWriter
    from freetoken.checkpoint.low_memory_nvfp4 import stream_nvfp4_layers_to_ftw
    from freetoken.models.loader import drop_page_cache
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_SOURCE_SPEC
    from freetoken.models.weight import load_weight

    try:
        fingerprint = _source_fingerprint(
            model_path, model_config, device=torch.device("cuda:0")
        )
    except Exception:
        fingerprint = None

    index = _write_native_nvfp4_ftw(
        model_path,
        out_dir,
        model_config,
        _NVFP4_SOURCE_SPEC,
        dense_weights=lambda: load_weight(
            model_path, torch.device("cpu"), include_moe_experts=False
        ),
        expert_streamer=stream_nvfp4_layers_to_ftw,
        writer_factory=FTWWriter,
        metadata_copier=_copy_metadata,
        drop_page_cache=drop_page_cache,
        fingerprint=fingerprint,
        shard_limit=shard_limit,
    )
    return {
        "mode": "executed",
        "architecture": _SUPPORTED_ARCH,
        "conversion_target": _CONVERSION_TARGET,
        "model_path": os.path.abspath(model_path),
        "out_dir": os.path.abspath(out_dir),
        "resources": report.as_dict(),
        "counts": index.get("counts"),
        "bytes": index.get("bytes"),
        "fingerprint": index.get("fingerprint"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m freetoken.experimental.low_memory_ftw_cpu_convert",
        description=(
            "Preflight (default) or explicitly execute the Qwen3.5 native-NVFP4 "
            "low-memory CPU/file-backed FTW conversion experiment."
        ),
    )
    parser.add_argument("--model", required=True, help="local Qwen3.5-MoE checkpoint directory")
    parser.add_argument("--out", required=True, help="new FTW output path")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually convert; without this flag the command is read-only preflight",
    )
    parser.add_argument(
        "--shard-limit-gib",
        type=float,
        default=8.0,
        help="FTW shard ceiling in GiB during --execute (default: 8)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.shard_limit_gib <= 0:
        raise SystemExit("--shard-limit-gib must be > 0")
    shard_limit = int(args.shard_limit_gib * (1 << 30))
    # FTW requires block-aligned shard boundaries.
    shard_limit = (shard_limit // 4096) * 4096
    if shard_limit <= 0:
        raise SystemExit("--shard-limit-gib is too small after 4096-byte alignment")

    if args.execute:
        result = execute_qwen35_native_nvfp4(
            args.model, args.out, shard_limit=shard_limit
        )
    else:
        result = preflight_qwen35_native_nvfp4(args.model, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "execute_qwen35_native_nvfp4",
    "main",
    "preflight_qwen35_native_nvfp4",
]
