"""Read-only resource admission for the low-memory native-NVFP4 -> FTW experiment.

This gate accounts resources by *runtime/storage representation*, not checkpoint size:

* safetensors CPU source tensors are file-backed/reclaimable and are reported separately
  from anonymous allocations;
* dense transform allocations come from the metadata-only Qwen3.5 envelope;
* the current experimental expert path fragment-streams native NVFP4 directly into FTW, so
  a full 6-bank MoE layer is diagnostic geometry, not anonymous residency;
* dense and expert conversion phases are sequential, so their anonymous peaks are combined
  with ``max`` rather than addition;
* execution re-checks MemAvailable after runtime/GPU initialization;
* disk admission estimates the actual native-FTW representation instead of multiplying the
  entire checkpoint by a generic 4x expansion.

The legacy conservative RAM path remains available for callers that do not provide exact
metadata-derived dense envelopes. Passing this gate authorizes only the separate
experimental conversion step; it does not prove conversion, loading, serving, or inference.
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
DENSE_TRANSIENT_FACTOR = 8  # legacy fallback only
DISK_POST_CONVERSION_RESERVE_BYTES = 4 << 30
RUNTIME_MARGIN_BYTES = 512 << 20
# Backward-compatible constants retained for older experimental callers/tests.
DISK_EXPANSION_FACTOR = 4
DISK_FIXED_HEADROOM_BYTES = 2 << 30
FIXED_HEADROOM_BYTES = DISK_FIXED_HEADROOM_BYTES
INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class ConversionPreflight:
    source_shards: int
    source_file_bytes: int
    source_tensor_bytes: int
    source_tensor_count: int
    largest_source_tensor_bytes: int
    routed_source_tensor_bytes: int
    non_routed_source_tensor_bytes: int
    expert_layer_bytes: int
    expert_bank_total_bytes: int
    expert_file_backed_fragment_peak_bytes: int
    expert_anonymous_peak_bytes: int
    dense_anonymous_peak_bytes: int
    phase_anonymous_peak_bytes: int
    runtime_margin_bytes: int
    disk_payload_envelope_bytes: int
    output_guard_bytes: int
    disk_free_bytes: int
    ram_guard_bytes: int
    mem_available_bytes: int
    ram_model: str
    disk_model: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _geometry(model_config: Any) -> tuple[int, int, int, int]:
    E = int(model_config.num_experts)
    H = int(model_config.hidden_size)
    I = int(model_config.moe_intermediate_size)
    L = int(
        getattr(model_config, "num_moe_layers", None)
        or getattr(model_config, "num_layers", 0)
        or 0
    )
    if min(E, H, I, L) <= 0:
        raise ValueError("NVFP4 dimensions/layer count must be positive")
    if H % 16 or I % 16:
        raise ValueError(
            f"native NVFP4 banks require hidden/intermediate divisible by 16, got H={H}, I={I}"
        )
    return E, H, I, L


def native_nvfp4_expert_layer_bytes(model_config: Any) -> int:
    """Physical bytes of all six native-NVFP4 banks for one complete MoE layer."""
    E, H, I, _L = _geometry(model_config)
    return (
        E * (2 * I) * (H // 2)
        + E * (2 * I) * (H // 16)
        + E * (2 * I) * 2
        + E * H * (I // 2)
        + E * H * (I // 16)
        + E * H * 2
    )


def native_nvfp4_fragment_memory(model_config: Any) -> dict[str, int]:
    """Memory contract of ``ftw_fragment_stream`` for one in-flight expert fragment.

    Packed/scale fragments returned by safetensors are file-backed CPU views. The only
    payload synthesized by the fragment stream is an FP16 global-scale row. Therefore the
    source-fragment window is reported for I/O/page-cache diagnostics but only the generated
    row is charged as anonymous expert conversion memory.
    """
    _E, H, I, _L = _geometry(model_config)
    source_peak = max(
        I * (H // 2),      # gate/up packed U8
        H * (I // 2),      # down packed U8
        I * (H // 16),     # gate/up FP8 block scale
        H * (I // 16),     # down FP8 block scale
        4,                  # scalar source global scale
    )
    generated_peak = max(I, H) * 2  # one generated FP16 global row
    return {
        "file_backed_source_fragment_peak_bytes": int(source_peak),
        "anonymous_generated_fragment_peak_bytes": int(generated_peak),
    }


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


def _read_safetensors_header(shard: Path) -> tuple[dict[str, Any], int]:
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
    return header, file_size - (8 + header_len)


def _validated_tensor_meta(name: str, meta: Any, shard: Path, data_region: int) -> int:
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
    return end - begin


def load_safetensors_headers(model_path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Return merged validated tensor headers without touching tensor payload bytes."""
    merged: dict[str, dict[str, Any]] = {}
    for shard in _source_shards(Path(model_path)):
        header, data_region = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            _validated_tensor_meta(name, meta, shard, data_region)
            if name in merged:
                raise ValueError(f"duplicate safetensors tensor name across shards: {name}")
            merged[name] = meta
    if not merged:
        raise ValueError("checkpoint contains no safetensors tensors")
    return merged


def _safetensors_header_stats(shards: list[Path]) -> tuple[int, int, int, int, int]:
    total = count = largest = routed = non_routed = 0
    names: set[str] = set()
    for shard in shards:
        header, data_region = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError(f"duplicate safetensors tensor name across shards: {name}")
            names.add(name)
            nbytes = _validated_tensor_meta(name, meta, shard, data_region)
            total += nbytes
            count += 1
            largest = max(largest, nbytes)
            if ".mlp.experts." in name:
                routed += nbytes
            else:
                non_routed += nbytes
    if count == 0:
        raise ValueError("checkpoint contains no safetensors tensors")
    return total, count, largest, routed, non_routed


def require_current_memory(
    report: ConversionPreflight,
    *,
    mem_available_bytes: int | None = None,
) -> int:
    """Re-check RAM against the same envelope at the current runtime state."""
    available = _mem_available_bytes() if mem_available_bytes is None else int(mem_available_bytes)
    if available < int(report.ram_guard_bytes):
        raise RuntimeError(
            "low-memory FTW conversion blocked after runtime initialization: MemAvailable is "
            f"{available / 2**30:.2f} GiB, phase envelope + margin requires "
            f"{report.ram_guard_bytes / 2**30:.2f} GiB"
        )
    return available


def preflight_low_memory_nvfp4_conversion(
    model_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    model_config: Any,
    *,
    dense_anonymous_peak_bytes: int | None = None,
    expert_anonymous_peak_bytes: int | None = None,
    expert_file_backed_fragment_peak_bytes: int | None = None,
    disk_free_bytes: int | None = None,
    mem_available_bytes: int | None = None,
) -> ConversionPreflight:
    """Return a representation-aware resource report or raise before output is written."""
    source = Path(model_path)
    output = Path(out_dir)
    _validate_source_output_separation(source, output)
    _validate_output_dir(output)
    shards = _source_shards(source)
    tensor_bytes, tensor_count, largest, routed, non_routed = _safetensors_header_stats(shards)
    source_file_bytes = sum(path.stat().st_size for path in shards)
    expert_layer = native_nvfp4_expert_layer_bytes(model_config)
    _E, _H, _I, moe_layers = _geometry(model_config)
    expert_total = expert_layer * moe_layers

    # Phase-13/15 output-envelope logic, now local to the exact native-FTW representation.
    checkpoint_expand = (source_file_bytes * 135 + 99) // 100
    native_output = expert_total + (non_routed * 150 + 99) // 100
    disk_payload = max(checkpoint_expand, native_output)
    output_guard = disk_payload + DISK_POST_CONVERSION_RESERVE_BYTES
    disk_model = "native_ftw_representation_envelope"

    if dense_anonymous_peak_bytes is None:
        dense_peak = DENSE_TRANSIENT_FACTOR * largest
        expert_peak = expert_layer
        source_fragment_peak = 0
        # Historical fail-closed behavior for callers that have not proven exact phase
        # semantics. Do not silently weaken this fallback.
        phase_peak = expert_peak + dense_peak
        runtime_margin = DISK_FIXED_HEADROOM_BYTES
        ram_model = "legacy_conservative_fallback"
    else:
        dense_peak = int(dense_anonymous_peak_bytes)
        if dense_peak < 0:
            raise ValueError("dense_anonymous_peak_bytes must be non-negative")
        if expert_anonymous_peak_bytes is None:
            fragment = native_nvfp4_fragment_memory(model_config)
            expert_peak = int(fragment["anonymous_generated_fragment_peak_bytes"])
            source_fragment_peak = int(fragment["file_backed_source_fragment_peak_bytes"])
            ram_model = "phase_max_exact_dense_fragment_expert_auto"
        else:
            expert_peak = int(expert_anonymous_peak_bytes)
            source_fragment_peak = int(expert_file_backed_fragment_peak_bytes or 0)
            if expert_peak < 0 or source_fragment_peak < 0:
                raise ValueError("fragment expert memory values must be non-negative")
            ram_model = "phase_max_exact_dense_fragment_expert"
        phase_peak = max(dense_peak, expert_peak)
        runtime_margin = RUNTIME_MARGIN_BYTES

    ram_guard = phase_peak + runtime_margin
    if disk_free_bytes is None:
        disk_free_bytes = shutil.disk_usage(_existing_probe_dir(output)).free
    if mem_available_bytes is None:
        mem_available_bytes = _mem_available_bytes()
    disk_free_bytes = int(disk_free_bytes)
    mem_available_bytes = int(mem_available_bytes)
    if disk_free_bytes < output_guard:
        raise RuntimeError(
            "low-memory FTW conversion blocked: output filesystem has "
            f"{disk_free_bytes / 2**30:.2f} GiB free, representation-aware guard requires "
            f"{output_guard / 2**30:.2f} GiB"
        )
    if mem_available_bytes < ram_guard:
        raise RuntimeError(
            "low-memory FTW conversion blocked: MemAvailable is "
            f"{mem_available_bytes / 2**30:.2f} GiB, RAM model {ram_model} requires "
            f"{ram_guard / 2**30:.2f} GiB"
        )
    return ConversionPreflight(
        source_shards=len(shards),
        source_file_bytes=source_file_bytes,
        source_tensor_bytes=tensor_bytes,
        source_tensor_count=tensor_count,
        largest_source_tensor_bytes=largest,
        routed_source_tensor_bytes=routed,
        non_routed_source_tensor_bytes=non_routed,
        expert_layer_bytes=expert_layer,
        expert_bank_total_bytes=expert_total,
        expert_file_backed_fragment_peak_bytes=source_fragment_peak,
        expert_anonymous_peak_bytes=expert_peak,
        dense_anonymous_peak_bytes=dense_peak,
        phase_anonymous_peak_bytes=phase_peak,
        runtime_margin_bytes=runtime_margin,
        disk_payload_envelope_bytes=disk_payload,
        output_guard_bytes=output_guard,
        disk_free_bytes=disk_free_bytes,
        ram_guard_bytes=ram_guard,
        mem_available_bytes=mem_available_bytes,
        ram_model=ram_model,
        disk_model=disk_model,
    )


__all__ = [
    "ALIGN",
    "ConversionPreflight",
    "DENSE_TRANSIENT_FACTOR",
    "DISK_EXPANSION_FACTOR",
    "DISK_FIXED_HEADROOM_BYTES",
    "DISK_POST_CONVERSION_RESERVE_BYTES",
    "FIXED_HEADROOM_BYTES",
    "RUNTIME_MARGIN_BYTES",
    "load_safetensors_headers",
    "native_nvfp4_expert_layer_bytes",
    "native_nvfp4_fragment_memory",
    "preflight_low_memory_nvfp4_conversion",
    "require_current_memory",
]
