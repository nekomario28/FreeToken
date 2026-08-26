"""Metadata-only admission preflight for the file-backed FTW CPU experiment.

This module deliberately does *not* open tensor payloads.  It reads only local checkpoint
metadata (``config.json`` and the safetensors index), stats the referenced shard files, and
reads host/filesystem capacity.  A successful result is not permission to run conversion; it
means only that the checkpoint shape/layout is eligible and no obvious metadata/resource
blocker was found.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

_QWEN35_ARCH = "Qwen3_5MoeForConditionalGeneration"
_RESULT_OK = "METADATA_OK_RESOURCE_UNPROVEN"
_RESULT_BLOCK = "BLOCK"


def _expert_quant_from_hf_config(config: dict[str, Any]) -> str:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return "none"
    algo = str(quant.get("quant_algo") or quant.get("quant_method") or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" not in algo:
        return "none"
    layers = quant.get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if not (str(name).endswith(".mlp.experts") or ".mlp.experts." in str(name)):
            continue
        expert_algo = str((spec or {}).get("quant_algo", "")).lower() if isinstance(spec, dict) else ""
        if "fp4" in expert_algo:
            return "nvfp4"
    return "none"


def native_nvfp4_layer_bytes(*, num_experts: int, hidden_size: int, intermediate_size: int) -> int:
    """Exact bytes of the six native NVFP4 HostBanks allocated by one-layer streaming."""
    E, H, I = int(num_experts), int(hidden_size), int(intermediate_size)
    if E <= 0 or H <= 0 or I <= 0:
        raise ValueError("NVFP4 dimensions must be positive")
    if H % 16 or I % 16:
        raise ValueError("native NVFP4 streaming requires hidden/intermediate sizes divisible by 16")
    return (
        E * 2 * I * (H // 2)
        + E * 2 * I * (H // 16)
        + E * 2 * I * 2
        + E * H * (I // 2)
        + E * H * (I // 16)
        + E * H * 2
    )


def _mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise ValueError(f"no existing ancestor for output path {path}")
        current = parent
    return current


def _safe_local_shards(root: Path, weight_map: dict[str, Any]) -> list[Path]:
    names = sorted({str(value) for value in weight_map.values()})
    if not names:
        raise ValueError("safetensors weight_map references no shards")
    shards: list[Path] = []
    for name in names:
        raw = Path(name)
        if raw.is_absolute():
            raise ValueError(f"absolute shard path is not allowed: {name}")
        resolved = (root / raw).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"shard path escapes checkpoint root: {name}") from exc
        if not resolved.is_file():
            raise ValueError(f"referenced safetensors shard is missing: {name}")
        shards.append(resolved)
    return shards


def preflight_file_backed_ftw_cpu_conversion(model_path: str, out_dir: str) -> dict[str, Any]:
    """Inspect a *local* Qwen3.5 ModelOpt NVFP4 checkpoint without reading tensor payloads."""
    root = Path(model_path).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    blockers: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        return {
            "schema_version": 1,
            "admission": _RESULT_BLOCK,
            "blockers": ["model_path must be an existing local directory"],
            "warnings": [],
        }
    if out == root or root in out.parents:
        blockers.append("output directory must be outside the source checkpoint tree")
    if out.exists() and out.is_dir() and any(out.iterdir()):
        blockers.append("output directory already exists and is non-empty")

    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    metadata_missing = False
    if not config_path.is_file():
        blockers.append("config.json is missing")
        metadata_missing = True
    if not index_path.is_file():
        blockers.append("model.safetensors.index.json is missing")
        metadata_missing = True
    if metadata_missing:
        return {
            "schema_version": 1,
            "admission": _RESULT_BLOCK,
            "checkpoint_root": str(root),
            "output_dir": str(out),
            "blockers": blockers,
            "warnings": warnings,
        }

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1,
            "admission": _RESULT_BLOCK,
            "checkpoint_root": str(root),
            "output_dir": str(out),
            "blockers": [f"checkpoint metadata is unreadable: {exc}"],
            "warnings": warnings,
        }

    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if isinstance(architectures, list) and architectures else None
    expert_quant = _expert_quant_from_hf_config(config)
    if architecture != _QWEN35_ARCH:
        blockers.append(f"architecture must be {_QWEN35_ARCH}, got {architecture!r}")
    if expert_quant != "nvfp4":
        blockers.append(f"routed experts must be native ModelOpt NVFP4, got {expert_quant!r}")

    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    try:
        num_layers = int(text["num_hidden_layers"])
        num_experts = int(text["num_experts"])
        hidden_size = int(text["hidden_size"])
        intermediate_size = int(text["moe_intermediate_size"])
        layer_bytes = native_nvfp4_layer_bytes(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(f"invalid or missing MoE dimensions: {exc}")
        num_layers = num_experts = hidden_size = intermediate_size = 0
        layer_bytes = 0

    weight_map = index.get("weight_map")
    shards: list[Path] = []
    if not isinstance(weight_map, dict):
        blockers.append("safetensors index has no weight_map object")
    else:
        try:
            shards = _safe_local_shards(root, weight_map)
        except ValueError as exc:
            blockers.append(str(exc))

    source_bytes = sum(path.stat().st_size for path in shards)
    largest_shard = max((path.stat().st_size for path in shards), default=0)
    mem_available = _mem_available_bytes()
    fs_root = _existing_ancestor(out.parent)
    disk_free = shutil.disk_usage(fs_root).free

    if layer_bytes and mem_available is not None:
        ratio = mem_available / layer_bytes
        if ratio < 1.5:
            warnings.append(
                f"MemAvailable is only {ratio:.2f}x one native expert layer; real conversion is high risk"
            )
    else:
        ratio = None
    if source_bytes:
        disk_ratio = disk_free / source_bytes
        if disk_ratio < 1.5:
            warnings.append(
                f"output filesystem free space is only {disk_ratio:.2f}x source shard bytes; "
                "dense dequantization may expand the FTW"
            )
    else:
        disk_ratio = None

    return {
        "schema_version": 1,
        "admission": _RESULT_BLOCK if blockers else _RESULT_OK,
        "checkpoint_root": str(root),
        "output_dir": str(out),
        "architecture": architecture,
        "expert_quant": expert_quant,
        "num_moe_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "moe_intermediate_size": intermediate_size,
        "native_expert_layer_bytes": layer_bytes,
        "native_expert_total_bytes": layer_bytes * num_layers,
        "source_shard_count": len(shards),
        "source_shard_bytes": source_bytes,
        "largest_source_shard_bytes": largest_shard,
        "mem_available_bytes": mem_available,
        "mem_available_to_expert_layer_ratio": ratio,
        "output_filesystem_free_bytes": disk_free,
        "output_free_to_source_ratio": disk_ratio,
        "blockers": blockers,
        "warnings": warnings,
        "payload_tensors_opened": False,
    }


__all__ = ["native_nvfp4_layer_bytes", "preflight_file_backed_ftw_cpu_conversion"]
