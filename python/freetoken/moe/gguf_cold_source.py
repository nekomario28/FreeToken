"""Qwen3.8/Qwen4Exp GGUF expert-range compiler for cold staging.

The compiler reads only GGUF metadata and turns each expert of the merged
``ffn_{gate,up,down}_exps`` tensors into an immutable byte range. No expert
payload is copied while compiling the index. The source then materializes only
the router-selected experts into the bounded staging seam from ``cold_source``.

PeasantSmith's target layout is deliberately explicit in v1:

* gate/up: IQ2_XXS, kept packed and concatenated along output rows;
* down: Q4_0, kept packed;
* expert is the outermost GGUF tensor dimension, so each expert is one contiguous
  byte slice of the merged tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping, Sequence

import torch

from freetoken.moe.cold_source import ColdBankLayout, ColdTensorRange


@dataclass(frozen=True)
class Qwen4ExpGgufIndex:
    model_path: str
    num_layers: int
    num_experts: int
    ranges: Mapping[tuple[int, int, str], ColdTensorRange]
    gate_layout: ColdBankLayout
    up_layout: ColdBankLayout
    down_layout: ColdBankLayout
    gate_up_layout: ColdBankLayout
    gate_ggml_type: int
    up_ggml_type: int
    down_ggml_type: int


@dataclass(frozen=True)
class _TensorGeometry:
    name: str
    data_offset: int
    n_bytes: int
    ggml_type: int
    num_experts: int
    rows_per_expert: int
    row_bytes: int

    @property
    def expert_bytes(self) -> int:
        return self.rows_per_expert * self.row_bytes

    @property
    def layout(self) -> ColdBankLayout:
        return ColdBankLayout((self.rows_per_expert, self.row_bytes), torch.uint8)

    def range_for(self, model_path: str, expert: int) -> ColdTensorRange:
        if not 0 <= expert < self.num_experts:
            raise ValueError(f"expert {expert} outside [0, {self.num_experts})")
        return ColdTensorRange(
            path=model_path,
            offset=self.data_offset + expert * self.expert_bytes,
            layout=self.layout,
        )


def _tensor_geometry(tensor, *, num_experts: int, quant_sizes) -> _TensorGeometry:
    ne = tuple(int(v) for v in tensor.shape)  # GGML order: fastest first.
    if len(ne) != 3:
        raise ValueError(f"{tensor.name}: expected rank-3 merged expert tensor, got {ne!r}")
    if ne[-1] != num_experts:
        raise ValueError(
            f"{tensor.name}: outer expert dimension {ne[-1]} != {num_experts}"
        )
    block, type_size = quant_sizes[tensor.tensor_type]
    if ne[0] % block:
        raise ValueError(
            f"{tensor.name}: fastest dimension {ne[0]} is not divisible by quant block {block}"
        )
    row_bytes = ne[0] // block * type_size
    rows = math.prod(ne[1:])
    if rows % num_experts:
        raise ValueError(f"{tensor.name}: packed row count is not expert-divisible")
    rows_per_expert = rows // num_experts
    expected = rows * row_bytes
    if int(tensor.n_bytes) != expected:
        raise ValueError(
            f"{tensor.name}: n_bytes={int(tensor.n_bytes)} != packed geometry {expected}"
        )
    if int(tensor.data_offset) < 0:
        raise ValueError(f"{tensor.name}: negative data_offset")
    return _TensorGeometry(
        name=tensor.name,
        data_offset=int(tensor.data_offset),
        n_bytes=int(tensor.n_bytes),
        ggml_type=int(tensor.tensor_type),
        num_experts=num_experts,
        rows_per_expert=rows_per_expert,
        row_bytes=row_bytes,
    )


def compile_qwen4exp_gguf_expert_index(
    model_path: str,
    *,
    num_layers: int = 48,
    num_experts: int = 512,
    gate_up_quant: str = "IQ2_XXS",
    down_quant: str = "Q4_0",
) -> Qwen4ExpGgufIndex:
    """Compile exact packed expert byte ranges without reading expert payloads.

    The target quant names are frozen inputs rather than inferred from the file so
    accidentally passing another GGUF fails closed instead of silently changing
    the kernel/backend contract.
    """
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")

    import gguf

    try:
        gate_up_type = gguf.GGMLQuantizationType[gate_up_quant]
        down_type = gguf.GGMLQuantizationType[down_quant]
    except KeyError as exc:
        raise ValueError(f"unknown GGUF quantization type: {exc.args[0]}") from exc

    reader = gguf.GGUFReader(model_path)
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    ranges: dict[tuple[int, int, str], ColdTensorRange] = {}
    first: tuple[_TensorGeometry, _TensorGeometry, _TensorGeometry] | None = None
    file_size = os.path.getsize(model_path)

    for layer in range(num_layers):
        names = {
            "gate": f"blk.{layer}.ffn_gate_exps.weight",
            "up": f"blk.{layer}.ffn_up_exps.weight",
            "down": f"blk.{layer}.ffn_down_exps.weight",
        }
        try:
            tensors = {role: by_name[name] for role, name in names.items()}
        except KeyError as exc:
            raise ValueError(f"missing Qwen4Exp expert tensor {exc.args[0]!r}") from exc

        if tensors["gate"].tensor_type != gate_up_type or tensors["up"].tensor_type != gate_up_type:
            raise ValueError(
                f"layer {layer}: gate/up must both be {gate_up_quant}; got "
                f"{tensors['gate'].tensor_type.name}/{tensors['up'].tensor_type.name}"
            )
        if tensors["down"].tensor_type != down_type:
            raise ValueError(
                f"layer {layer}: down must be {down_quant}; got {tensors['down'].tensor_type.name}"
            )

        gate = _tensor_geometry(
            tensors["gate"], num_experts=num_experts, quant_sizes=gguf.GGML_QUANT_SIZES
        )
        up = _tensor_geometry(
            tensors["up"], num_experts=num_experts, quant_sizes=gguf.GGML_QUANT_SIZES
        )
        down = _tensor_geometry(
            tensors["down"], num_experts=num_experts, quant_sizes=gguf.GGML_QUANT_SIZES
        )
        if gate.layout != up.layout:
            raise ValueError(f"layer {layer}: gate/up packed layouts differ")
        for geom in (gate, up, down):
            if geom.data_offset + geom.n_bytes > file_size:
                raise ValueError(f"{geom.name}: tensor byte range exceeds file size")

        if first is None:
            first = (gate, up, down)
        else:
            g0, u0, d0 = first
            if (gate.layout, up.layout, down.layout) != (g0.layout, u0.layout, d0.layout):
                raise ValueError(f"layer {layer}: packed expert layout drift")
            if (gate.ggml_type, up.ggml_type, down.ggml_type) != (
                g0.ggml_type,
                u0.ggml_type,
                d0.ggml_type,
            ):
                raise ValueError(f"layer {layer}: expert quantization type drift")

        for expert in range(num_experts):
            ranges[(layer, expert, "gate")] = gate.range_for(model_path, expert)
            ranges[(layer, expert, "up")] = up.range_for(model_path, expert)
            ranges[(layer, expert, "down")] = down.range_for(model_path, expert)

    assert first is not None
    gate0, up0, down0 = first
    gate_up_layout = ColdBankLayout(
        (gate0.rows_per_expert + up0.rows_per_expert, gate0.row_bytes), torch.uint8
    )
    return Qwen4ExpGgufIndex(
        model_path=model_path,
        num_layers=num_layers,
        num_experts=num_experts,
        ranges=ranges,
        gate_layout=gate0.layout,
        up_layout=up0.layout,
        down_layout=down0.layout,
        gate_up_layout=gate_up_layout,
        gate_ggml_type=gate0.ggml_type,
        up_ggml_type=up0.ggml_type,
        down_ggml_type=down0.ggml_type,
    )


def _read_exact(entry: ColdTensorRange) -> torch.Tensor:
    fd = os.open(entry.path, os.O_RDONLY)
    try:
        payload = os.pread(fd, entry.layout.nbytes, entry.offset)
    finally:
        os.close(fd)
    if len(payload) != entry.layout.nbytes:
        raise OSError(
            f"short GGUF expert read at offset={entry.offset}: "
            f"expected={entry.layout.nbytes} observed={len(payload)}"
        )
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(entry.layout.shape)


class Qwen4ExpGgufColdExpertSource:
    """Serve PeasantSmith-style packed Qwen4Exp experts to ``ColdSourceBinding``."""

    bank_schema = ("gate_up", "down")

    def __init__(self, index: Qwen4ExpGgufIndex) -> None:
        if not isinstance(index, Qwen4ExpGgufIndex):
            raise TypeError("index must be Qwen4ExpGgufIndex")
        self.index = index

    @property
    def layouts(self) -> dict[str, ColdBankLayout]:
        return {"gate_up": self.index.gate_up_layout, "down": self.index.down_layout}

    def load(self, layer_id: int, expert_ids: Sequence[int]) -> Mapping[str, torch.Tensor]:
        if not isinstance(layer_id, int) or isinstance(layer_id, bool) or not 0 <= layer_id < self.index.num_layers:
            raise ValueError("layer_id outside compiled Qwen4Exp index")
        experts = tuple(expert_ids)
        if not experts:
            raise ValueError("expert_ids must be non-empty")
        if len(set(experts)) != len(experts):
            raise ValueError("expert_ids must be unique")
        if any(
            not isinstance(expert, int)
            or isinstance(expert, bool)
            or not 0 <= expert < self.index.num_experts
            for expert in experts
        ):
            raise ValueError("expert id outside compiled Qwen4Exp index")

        gate_up_rows = []
        down_rows = []
        for expert in experts:
            gate = _read_exact(self.index.ranges[(layer_id, expert, "gate")])
            up = _read_exact(self.index.ranges[(layer_id, expert, "up")])
            down = _read_exact(self.index.ranges[(layer_id, expert, "down")])
            gate_up_rows.append(torch.cat((gate, up), dim=0))
            down_rows.append(down)
        return {
            "gate_up": torch.stack(gate_up_rows, dim=0).contiguous(),
            "down": torch.stack(down_rows, dim=0).contiguous(),
        }


__all__ = [
    "Qwen4ExpGgufIndex",
    "compile_qwen4exp_gguf_expert_index",
    "Qwen4ExpGgufColdExpertSource",
]
