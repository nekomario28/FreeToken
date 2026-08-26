"""Read-only resource admission for the low-memory native-NVFP4 -> FTW experiment.

The real conversion can be tens of GiB and may expand some packed tensors while fusing or
converting them. This module performs a deliberately conservative preflight before any FTW
shard is created. It reads only the safetensors index/header metadata plus OS free-space /
MemAvailable counters; no weight payload is materialized.

The estimates are guards, not performance predictions:

* disk guard = 4x all source tensor payload bytes + one FTW alignment page per source tensor
  + 2 GiB fixed headroom. 4x covers the worst common packed-NVFP4 uint8 -> bf16 expansion.
* RAM guard = exact six-bank native-NVFP4 bytes for ONE MoE layer + 8x the largest raw source
  tensor + 2 GiB fixed headroom. The 8x term covers multi-part dense fusion/dequant
  transients without pretending to be an exact allocator model.

Failing a guard means "do not start the real conversion". Passing means only that these
coarse resource floors are satisfied; runtime correctness still belongs to the synthetic /
ROCm integration gates and the converter itself.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ALIGN = 4096
DISK_EXPANSION_FACTOR = 4
DENSE_TRANSIENT_FACTOR = 8
FIXED_HEADROOM_BYTES = 2 << 30
INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class ConversionPreflight:
    source_shards: int
    source_file_bytes: int
    source_tensor_bytes: int
    source_tensor_count: int
    largest_source_tensor_bytes: int
    expert_layer_bytes: int
    output_guard_bytes: int
    disk_free_bytes: int
    ram_guard_bytes: int
    mem_available_bytes: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def native_nvfp4_expert_layer_bytes(model_config: Any) -> int:
    """Exact bytes of the six native NVFP4 source banks for one MoE layer."""
    E = int(model_config.num_experts)
    H = int(model_config.hidden_size)
    I = int(model_config.moe_intermediate_size)
    if E <= 0 or H <= 0 or I <= 0:
        raise ValueError("NVFP4 dimensions must be positive")
    if H % 16 or I % 16:
        raise ValueError(
            f"native NVFP4 banks require hidden/intermediate divisible by 16, got H={H}, I={I}"
        )
    return (
        E * (2 * I) * (H // 2)
        + E * (2 * I) * (H // 16)
        + E * (2 * I) * 2
        + E * H * (I // 2)
        + E * H * (I // 16)
        + E * H * 2
    )


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError as exc:
        raise RuntimeError("cannot read /proc/meminfo for conversion admission") from exc
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _existing_probe_dir(path: Path) -> Path:
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise ValueError(f"no existing parent for output path {path}")
        probe = parent
    if not probe.is_dir():
        probe = probe.parent
    return probe


def _validate_source_output_separation(source: Path, output: Path) -> None:
    """Never write the conversion product into the source checkpoint tree."""
    source_root = source.expanduser().resolve()
    output_root = output.expanduser().resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output directory must be outside the source checkpoint tree")


def _validate_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise ValueError(f"output directory must be empty before conversion: {out_dir}")


def _source_shards(model_path: Path) -> list[Path]:
    if not model_path.is_dir():
        raise ValueError("low-memory NVFP4 conversion currently requires a local checkpoint directory")
    index_path = model_path / INDEX_NAME
    if not index_path.is_file():
        raise ValueError(f"missing {INDEX_NAME}: {model_path}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no non-empty weight_map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name for name in shard_names):
        raise ValueError("safetensors weight_map contains an invalid shard name")
    shards = []
    root = model_path.resolve()
    for name in shard_names:
        candidate = Path(name)
        if candidate.is_absolute():
            raise ValueError("safetensors shard path must be checkpoint-relative")
        shard = (root / candidate).resolve()
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"safetensors shard escapes checkpoint directory: {name}") from exc
        if not shard.is_file():
            raise ValueError(f"safetensors shard is missing: {name}")
        shards.append(shard)
    return shards


def _safetensors_header_stats(shards: list[Path]) -> tuple[int, int, int]:
    total = 0
    count = 0
    largest = 0
    for shard in shards:
        file_size = shard.stat().st_size
        with shard.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header length: {shard.name}")
            header_len = struct.unpack("<Q", raw)[0]
            if header_len <= 0 or 8 + header_len > file_size:
                raise ValueError(f"invalid safetensors header length in {shard.name}")
            try:
                header = json.loads(handle.read(header_len))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid safetensors JSON header in {shard.name}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header is not an object: {shard.name}")
        data_region = file_size - (8 + header_len)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"invalid tensor metadata for {name!r} in {shard.name}")
            offsets = meta.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in offsets)
            ):
                raise ValueError(f"invalid data_offsets for {name!r} in {shard.name}")
            begin, end = offsets
            if begin < 0 or end < begin or end > data_region:
                raise ValueError(f"out-of-range data_offsets for {name!r} in {shard.name}")
            nbytes = end - begin
            total += nbytes
            count += 1
            largest = max(largest, nbytes)
    if count == 0:
        raise ValueError("checkpoint contains no safetensors tensors")
    return total, count, largest


def preflight_low_memory_nvfp4_conversion(
    model_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    model_config: Any,
    *,
    disk_free_bytes: int | None = None,
    mem_available_bytes: int | None = None,
) -> ConversionPreflight:
    """Return the resource report or raise before any conversion output is written."""
    source = Path(model_path)
    output = Path(out_dir)
    _validate_source_output_separation(source, output)
    _validate_output_dir(output)
    shards = _source_shards(source)
    tensor_bytes, tensor_count, largest = _safetensors_header_stats(shards)
    source_file_bytes = sum(path.stat().st_size for path in shards)
    expert_layer = native_nvfp4_expert_layer_bytes(model_config)

    output_guard = (
        DISK_EXPANSION_FACTOR * tensor_bytes
        + ALIGN * tensor_count
        + FIXED_HEADROOM_BYTES
    )
    ram_guard = (
        expert_layer
        + DENSE_TRANSIENT_FACTOR * largest
        + FIXED_HEADROOM_BYTES
    )
    if disk_free_bytes is None:
        disk_free_bytes = shutil.disk_usage(_existing_probe_dir(output)).free
    if mem_available_bytes is None:
        mem_available_bytes = _mem_available_bytes()
    disk_free_bytes = int(disk_free_bytes)
    mem_available_bytes = int(mem_available_bytes)
    if disk_free_bytes < output_guard:
        raise RuntimeError(
            "low-memory FTW conversion blocked: output filesystem has "
            f"{disk_free_bytes / 2**30:.2f} GiB free, conservative guard requires "
            f"{output_guard / 2**30:.2f} GiB"
        )
    if mem_available_bytes < ram_guard:
        raise RuntimeError(
            "low-memory FTW conversion blocked: MemAvailable is "
            f"{mem_available_bytes / 2**30:.2f} GiB, conservative guard requires "
            f"{ram_guard / 2**30:.2f} GiB"
        )
    return ConversionPreflight(
        source_shards=len(shards),
        source_file_bytes=source_file_bytes,
        source_tensor_bytes=tensor_bytes,
        source_tensor_count=tensor_count,
        largest_source_tensor_bytes=largest,
        expert_layer_bytes=expert_layer,
        output_guard_bytes=output_guard,
        disk_free_bytes=disk_free_bytes,
        ram_guard_bytes=ram_guard,
        mem_available_bytes=mem_available_bytes,
    )


__all__ = [
    "ALIGN",
    "ConversionPreflight",
    "DENSE_TRANSIENT_FACTOR",
    "DISK_EXPANSION_FACTOR",
    "FIXED_HEADROOM_BYTES",
    "native_nvfp4_expert_layer_bytes",
    "preflight_low_memory_nvfp4_conversion",
]
