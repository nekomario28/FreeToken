"""Disk-backed IQ4_NL PLE table for Qwen3.8 GGUF checkpoints.

PeasantSmith stores ``per_layer_token_embd.weight`` as IQ4_NL. The table is far
larger than host RAM should be asked to pin, while one decode token addresses
only the model-native n-gram rows. This backend reads those packed rows directly
from the GGUF, dequantizes only the selected rows on the existing GGUF device
kernel, and returns the normal PLE bf16 shape.

v1 deliberately leaves RAM eviction to the OS page cache and uses synchronous
``pread``. Async/aligned/io_uring reads are a later performance gate, not part of
the correctness seam.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch

GGML_IQ4_NL = 20
IQ4_NL_BLOCK = 32
IQ4_NL_BLOCK_BYTES = 18
_PLE_TENSOR = "per_layer_token_embd.weight"


@dataclass(frozen=True)
class GgufIq4NlPleIndex:
    model_path: str
    data_offset: int
    num_rows: int
    head_dim: int
    row_bytes: int
    ggml_type: int = GGML_IQ4_NL


def compile_iq4_nl_ple_index(
    model_path: str,
    *,
    tensor_name: str = _PLE_TENSOR,
    expected_head_dim: int = 160,
) -> GgufIq4NlPleIndex:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    import gguf

    reader = gguf.GGUFReader(model_path)
    hit = next((tensor for tensor in reader.tensors if tensor.name == tensor_name), None)
    if hit is None:
        raise ValueError(f"missing GGUF PLE tensor {tensor_name!r}")
    if int(hit.tensor_type) != GGML_IQ4_NL:
        raise ValueError(
            f"{tensor_name}: expected IQ4_NL type {GGML_IQ4_NL}, got {int(hit.tensor_type)}"
        )
    ne = tuple(int(v) for v in hit.shape)  # GGML order [head_dim, rows]
    if len(ne) != 2:
        raise ValueError(f"{tensor_name}: expected rank 2, got {ne!r}")
    head_dim, num_rows = ne
    if head_dim != expected_head_dim:
        raise ValueError(
            f"{tensor_name}: head_dim {head_dim} != expected {expected_head_dim}"
        )
    if head_dim % IQ4_NL_BLOCK:
        raise ValueError("PLE head_dim is not IQ4_NL block aligned")
    row_bytes = head_dim // IQ4_NL_BLOCK * IQ4_NL_BLOCK_BYTES
    expected_bytes = num_rows * row_bytes
    if int(hit.n_bytes) != expected_bytes:
        raise ValueError(
            f"{tensor_name}: n_bytes={int(hit.n_bytes)} != {expected_bytes}"
        )
    data_offset = int(hit.data_offset)
    if data_offset < 0 or data_offset + expected_bytes > os.path.getsize(model_path):
        raise ValueError(f"{tensor_name}: byte range outside GGUF file")
    return GgufIq4NlPleIndex(
        model_path=model_path,
        data_offset=data_offset,
        num_rows=num_rows,
        head_dim=head_dim,
        row_bytes=row_bytes,
    )


class GgufIq4NlPleTable:
    """PLETableBackend-compatible sparse row store over a GGUF IQ4_NL tensor."""

    dtype = torch.bfloat16

    def __init__(self, index: GgufIq4NlPleIndex, *, device: torch.device | None = None) -> None:
        if not isinstance(index, GgufIq4NlPleIndex):
            raise TypeError("index must be GgufIq4NlPleIndex")
        self.index = index
        self.num_rows = index.num_rows
        self.head_dim = index.head_dim
        self._device = device or torch.device(
            "cuda", torch.cuda.current_device()
        )

    def _read_unique(self, ids: torch.Tensor) -> torch.Tensor:
        rows = torch.empty((ids.numel(), self.index.row_bytes), dtype=torch.uint8)
        fd = os.open(self.index.model_path, os.O_RDONLY)
        try:
            for i, row_id in enumerate(ids.tolist()):
                if not 0 <= row_id < self.num_rows:
                    raise IndexError(f"PLE row id {row_id} outside [0, {self.num_rows})")
                off = self.index.data_offset + row_id * self.index.row_bytes
                payload = os.pread(fd, self.index.row_bytes, off)
                if len(payload) != self.index.row_bytes:
                    raise OSError(
                        f"short PLE read row={row_id}: expected={self.index.row_bytes} "
                        f"observed={len(payload)}"
                    )
                rows[i].copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        finally:
            os.close(fd)
        return rows

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        # PLE hashes are integer ids; bring only the tiny id vector to host. torch.unique
        # also gives an inverse so repeated n-grams trigger one file read in this lookup.
        flat_cpu = row_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
        unique, inverse = torch.unique(flat_cpu, sorted=False, return_inverse=True)
        packed = self._read_unique(unique)
        if self._device.type == "cuda":
            packed = packed.pin_memory().to(self._device, non_blocking=True)
        else:
            packed = packed.to(self._device)

        from freetoken.kernel.gguf import ggml_dequantize

        dense_unique = ggml_dequantize(
            packed,
            GGML_IQ4_NL,
            unique.numel(),
            self.head_dim,
            self.dtype,
        )
        inverse = inverse.to(dense_unique.device)
        rows = dense_unique.index_select(0, inverse).reshape(*row_ids.shape, self.head_dim)
        rows = rows.flatten(-2)
        if out is None:
            return rows
        out.copy_(rows)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        # Correctness-only v1: no speculative I/O and no hidden resident host cache.
        return None


__all__ = [
    "GGML_IQ4_NL",
    "GgufIq4NlPleIndex",
    "compile_iq4_nl_ple_index",
    "GgufIq4NlPleTable",
]
