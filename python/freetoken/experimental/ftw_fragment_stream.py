"""Fragment-streamed native-NVFP4 expert writer for the experimental FTW converter.

The one-layer path already removes whole-model expert-bank residency, but a Qwen3.5 layer
still materializes six [E, ...] banks at once.  For native NVFP4 that assembly is unnecessary:
the final FTW bank is mostly a deterministic concatenation of checkpoint fragments that are
already in the target packed representation.

This module keeps the canonical converter unchanged.  Its ``_ConvertSink`` owns an
``FTWWriter``; during the expert call only, we teach that writer to accept a private
``FragmentTensor`` descriptor.  The normal sink still owns naming, accounting, layer counts,
progress and bank release.  Source tensors remain safetensors CPU/file-backed views, while
only tiny generated FP16 global-scale rows are anonymous allocations.
"""
from __future__ import annotations

import math
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import safetensors
import torch

_ALIGN = 4096
_NATIVE_BANK_ORDER = (
    "gate_up_packed",
    "gate_up_scale",
    "gate_up_global",
    "down_packed",
    "down_scale",
    "down_global",
)


@dataclass(frozen=True)
class FragmentTensor:
    dtype: torch.dtype
    shape: tuple[int, ...]
    fragments: Callable[[], Iterator[torch.Tensor]]

    @property
    def nbytes(self) -> int:
        count = math.prod(self.shape)
        return int(count * torch.empty((), dtype=self.dtype).element_size())


class FragmentBank:
    """Minimal HostBank-shaped object consumed by the canonical conversion sink."""

    def __init__(self, tensor: FragmentTensor) -> None:
        self.tensor = tensor
        self.nbytes = tensor.nbytes
        self.released = False

    def release(self) -> None:
        if self.released:
            raise RuntimeError("fragment bank released twice")
        self.released = True


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _write_fragment_tensor(writer: Any, name: str, descriptor: FragmentTensor, kind: str) -> None:
    """Stream a logical FTW tensor from contiguous fragments without assembling it in RAM."""
    expected = descriptor.nbytes
    if expected <= 0:
        raise ValueError("fragment tensor must contain at least one byte")

    # Mirror FTWWriter.add_tensor placement semantics exactly so readers see the same layout.
    if writer._f is None or (
        expected <= writer.shard_limit and writer._cur + expected > writer.shard_limit
    ):
        writer._roll()
    global_off = writer._global
    if global_off % _ALIGN:
        raise AssertionError("fragment tensor start must be FTW aligned")

    written = 0
    for fragment in descriptor.fragments():
        if not isinstance(fragment, torch.Tensor):
            raise TypeError("fragment producer must yield torch.Tensor objects")
        if fragment.dtype != descriptor.dtype:
            raise ValueError(
                f"fragment dtype mismatch for {name}: {fragment.dtype} != {descriptor.dtype}"
            )
        # Safetensors CPU tensors are already contiguous file-backed views; contiguous() is a
        # no-op there. Generated global rows are tiny and intentionally materialized.
        part = fragment.detach().cpu().contiguous()
        raw = part.reshape(-1).view(torch.uint8)
        nbytes = int(raw.numel())
        if written + nbytes > expected:
            raise ValueError(
                f"fragment stream for {name} exceeded declared size: {written + nbytes}/{expected}"
            )
        if nbytes:
            writer._write_raw(memoryview(raw.numpy()))
        written += nbytes

    if written != expected:
        raise ValueError(f"fragment stream for {name} incomplete: {written}/{expected} bytes")
    writer._tensors.append(
        {
            "name": name,
            "kind": kind,
            "dtype": _dtype_str(descriptor.dtype),
            "shape": list(descriptor.shape),
            "global_off": global_off,
            "nbytes": expected,
        }
    )
    pad = ((writer._global + _ALIGN - 1) // _ALIGN * _ALIGN) - writer._global
    if pad:
        writer._write_raw(memoryview(bytes(pad)))


@contextmanager
def _fragment_writer_scope(writer: Any):
    """Intercept only FragmentTensor writes; ordinary FTW writes keep canonical behavior."""
    required = ("_roll", "_write_raw", "_tensors", "_global", "_cur", "shard_limit")
    missing = [name for name in required if not hasattr(writer, name)]
    if missing:
        raise ValueError(f"canonical FTW writer lacks fragment contract: {missing}")
    original = writer.add_tensor

    def add_tensor(name: str, tensor: Any, kind: str = "weight"):
        if isinstance(tensor, FragmentTensor):
            return _write_fragment_tensor(writer, name, tensor, kind)
        return original(name, tensor, kind=kind)

    writer.add_tensor = add_tensor
    try:
        yield
    finally:
        writer.add_tensor = original


def _records_for_layer(layer_items: dict[str, list], spec: Any) -> dict[tuple[int, str, str], tuple[str, str]]:
    records: dict[tuple[int, str, str], tuple[str, str]] = {}
    for shard, items in layer_items.items():
        for name, match in items:
            expert = int(match.group("expert"))
            proj = match.group("proj")
            role = spec.proj_to_role.get(proj)
            kind = match.group("kind")
            if role not in {"gate", "up", "down"}:
                raise ValueError(f"{spec.desc}: unsupported projection role {role!r}")
            key = (expert, role, kind)
            if key in records:
                raise ValueError(f"{spec.desc}: duplicate expert source {key}")
            records[key] = (shard, name)
    return records


def _expect_shape(tensor: torch.Tensor, shape: tuple[int, ...], desc: str) -> torch.Tensor:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{desc}: source shape {tuple(tensor.shape)} != expected {shape}")
    return tensor


def stream_nvfp4_fragments_serial(
    model_path: str,
    config: Any,
    spec: Any,
    layer_sink: Any,
) -> dict[str, int]:
    """Stream Qwen3.5 native-NVFP4 experts directly into the canonical FTW sink.

    Peak source payload view is one expert projection fragment. It is file-backed. The only
    generated payload is a per-expert FP16 global-scale row (at most max(2*I, H) elements).
    """
    from freetoken.checkpoint.low_memory_nvfp4 import _index_by_bank_layer
    from freetoken.models.loader import drop_page_cache

    writer = getattr(layer_sink, "_writer", None)
    if writer is None:
        raise ValueError("fragmented NVFP4 conversion requires the canonical FTW conversion sink")

    folder, by_layer = _index_by_bank_layer(model_path, config, spec)
    E = int(config.num_experts)
    H = int(config.hidden_size)
    I = int(config.moe_intermediate_size)
    L = int(config.num_moe_layers)
    if set(by_layer) != set(range(L)):
        raise ValueError(
            f"{spec.desc}: fragmented source layers {sorted(by_layer)} != expected {list(range(L))}"
        )

    source_fragment_peak = max(I * (H // 2), H * (I // 2), I * (H // 16), H * (I // 16))
    generated_fragment_peak = max(2 * I, H) * 2
    entries_written = 0
    source_fragments = 0

    with _fragment_writer_scope(writer):
        for layer in range(L):
            records = _records_for_layer(by_layer[layer], spec)
            expected_keys = {
                (expert, role, kind)
                for expert in range(E)
                for role in ("gate", "up", "down")
                for kind in ("weight", "weight_scale", "weight_scale_2")
            }
            if set(records) != expected_keys:
                missing = sorted(expected_keys - set(records))[:8]
                extra = sorted(set(records) - expected_keys)[:8]
                raise ValueError(
                    f"{spec.desc}: incomplete fragmented layer {layer}; missing={missing}, extra={extra}"
                )

            with ExitStack() as stack:
                shard_names = sorted({shard for shard, _name in records.values()})
                handles = {
                    shard: stack.enter_context(
                        safetensors.safe_open(
                            os.path.join(folder, shard), framework="pt", device="cpu"
                        )
                    )
                    for shard in shard_names
                }

                def source(expert: int, role: str, kind: str) -> torch.Tensor:
                    nonlocal source_fragments
                    shard, name = records[(expert, role, kind)]
                    source_fragments += 1
                    return handles[shard].get_tensor(name)

                def packed_gate_up() -> Iterator[torch.Tensor]:
                    for expert in range(E):
                        yield _expect_shape(source(expert, "gate", "weight"), (I, H // 2), "gate packed")
                        yield _expect_shape(source(expert, "up", "weight"), (I, H // 2), "up packed")

                def scale_gate_up() -> Iterator[torch.Tensor]:
                    for expert in range(E):
                        yield _expect_shape(source(expert, "gate", "weight_scale"), (I, H // 16), "gate scale")
                        yield _expect_shape(source(expert, "up", "weight_scale"), (I, H // 16), "up scale")

                def global_gate_up() -> Iterator[torch.Tensor]:
                    for expert in range(E):
                        for role in ("gate", "up"):
                            scalar = source(expert, role, "weight_scale_2")
                            if scalar.numel() != 1:
                                raise ValueError(f"{spec.desc}: {role} global scale is not scalar")
                            yield scalar.reshape(1).to(torch.float16).expand(I).contiguous()

                def packed_down() -> Iterator[torch.Tensor]:
                    for expert in range(E):
                        yield _expect_shape(source(expert, "down", "weight"), (H, I // 2), "down packed")

                def scale_down() -> Iterator[torch.Tensor]:
                    for expert in range(E):
                        yield _expect_shape(source(expert, "down", "weight_scale"), (H, I // 16), "down scale")

                def global_down() -> Iterator[torch.Tensor]:
                    for expert in range(E):
                        scalar = source(expert, "down", "weight_scale_2")
                        if scalar.numel() != 1:
                            raise ValueError(f"{spec.desc}: down global scale is not scalar")
                        yield scalar.reshape(1).to(torch.float16).expand(H).contiguous()

                descriptors = {
                    "gate_up_packed": FragmentTensor(torch.uint8, (E, 2 * I, H // 2), packed_gate_up),
                    "gate_up_scale": FragmentTensor(torch.float8_e4m3fn, (E, 2 * I, H // 16), scale_gate_up),
                    "gate_up_global": FragmentTensor(torch.float16, (E, 2 * I), global_gate_up),
                    "down_packed": FragmentTensor(torch.uint8, (E, H, I // 2), packed_down),
                    "down_scale": FragmentTensor(torch.float8_e4m3fn, (E, H, I // 16), scale_down),
                    "down_global": FragmentTensor(torch.float16, (E, H), global_down),
                }
                banks = {name: FragmentBank(descriptors[name]) for name in _NATIVE_BANK_ORDER}
                layer_sink(layer, banks)
                if not all(bank.released for bank in banks.values()):
                    raise RuntimeError("canonical conversion sink did not release every fragment bank")
                entries_written += len(banks)

            for shard in sorted({shard for shard, _name in records.values()}):
                drop_page_cache(os.path.join(folder, shard))

    return {
        "layers_streamed": L,
        "ftw_entries_written": entries_written,
        "source_fragments_read": source_fragments,
        "source_fragment_peak_bytes": source_fragment_peak,
        "generated_fragment_peak_bytes": generated_fragment_peak,
        "expert_fragment_peak_bytes": max(source_fragment_peak, generated_fragment_peak),
    }


__all__ = [
    "FragmentBank",
    "FragmentTensor",
    "stream_nvfp4_fragments_serial",
]
