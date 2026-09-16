"""Experimental file-backed expert source for bounded MoE staging.

This module is deliberately narrow. It lets an ``OffloadMoeCache`` consume
router-selected expert rows without keeping the complete host expert banks
resident. FreeToken still owns routing, victim selection, slot bookkeeping and
the GPU cache. The source owns only immutable byte ranges and bounded staging.

v1 is graph-disabled, GPU-offload-only for real execution, and intentionally
uses the OS page cache through ``pread``. Predictive prefetch, custom RAM
eviction, hybrid CPU overflow compute, O_DIRECT/io_uring and checkpoint parsing
are separate gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True)
class ColdBankLayout:
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if not self.shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.shape
        ):
            raise ValueError("shape must contain positive integers")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")

    @property
    def nbytes(self) -> int:
        elems = 1
        for dim in self.shape:
            elems *= dim
        return elems * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class ColdTensorRange:
    path: str
    offset: int
    layout: ColdBankLayout

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be a non-empty string")
        if not isinstance(self.offset, int) or isinstance(self.offset, bool) or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")


class ColdExpertSource(Protocol):
    bank_schema: tuple[str, ...]

    def load(
        self, layer_id: int, expert_ids: Sequence[int]
    ) -> Mapping[str, torch.Tensor]: ...


class IndexedFileColdExpertSource:
    """Read exact packed expert rows from immutable file byte ranges.

    ``index[(layer, expert, bank)]`` identifies one complete packed bank row.
    The source never materializes unrelated experts. ``pread`` intentionally
    leaves RAM admission/eviction to the kernel page cache in this first gate.
    """

    def __init__(
        self,
        *,
        bank_schema: Sequence[str],
        index: Mapping[tuple[int, int, str], ColdTensorRange],
    ) -> None:
        schema = tuple(bank_schema)
        if not schema or any(not isinstance(name, str) or not name for name in schema):
            raise ValueError("bank_schema must contain non-empty strings")
        if len(set(schema)) != len(schema):
            raise ValueError("bank_schema entries must be unique")
        self.bank_schema = schema
        self._index = dict(index)

    @staticmethod
    def _read_exact(entry: ColdTensorRange) -> torch.Tensor:
        fd = os.open(entry.path, os.O_RDONLY)
        try:
            payload = os.pread(fd, entry.layout.nbytes, entry.offset)
        finally:
            os.close(fd)
        if len(payload) != entry.layout.nbytes:
            raise OSError(
                f"short cold-source read: {entry.path!r} offset={entry.offset} "
                f"expected={entry.layout.nbytes} observed={len(payload)}"
            )
        # bytearray gives torch a writable, owned Python buffer. The returned
        # tensor keeps that buffer alive for the bounded staging lifetime.
        return torch.frombuffer(bytearray(payload), dtype=entry.layout.dtype).reshape(
            entry.layout.shape
        )

    def load(
        self, layer_id: int, expert_ids: Sequence[int]
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
            raise ValueError("layer_id must be a non-negative integer")
        experts = tuple(expert_ids)
        if any(
            not isinstance(expert, int) or isinstance(expert, bool) or expert < 0
            for expert in experts
        ):
            raise ValueError("expert_ids must contain non-negative integers")
        if len(set(experts)) != len(experts):
            raise ValueError("expert_ids must be unique")

        out: dict[str, torch.Tensor] = {}
        for bank in self.bank_schema:
            rows = []
            expected: ColdBankLayout | None = None
            for expert in experts:
                key = (layer_id, expert, bank)
                try:
                    entry = self._index[key]
                except KeyError as exc:
                    raise KeyError(f"cold-source index has no row for {key!r}") from exc
                if expected is None:
                    expected = entry.layout
                elif entry.layout != expected:
                    raise ValueError(
                        f"cold-source layout drift for layer={layer_id} bank={bank!r}"
                    )
                rows.append(self._read_exact(entry))
            if not rows:
                raise ValueError("cold-source load requires at least one expert")
            out[bank] = torch.stack(rows, dim=0).contiguous()
        return out


@dataclass(frozen=True)
class ColdCopyStats:
    experts: int
    microbatches: int
    bank_rows: int
    source_bytes: int


class ColdSourceBinding:
    """Bind one bounded cold source to an existing FreeToken cache instance.

    The binding replaces only this cache instance's ``copy_missing`` method.
    ``ensure_experts`` / ``materialize_layer`` remain authoritative for the
    pending logical experts and destination slots.
    """

    def __init__(
        self,
        *,
        cache,
        source: ColdExpertSource,
        layouts: Mapping[str, ColdBankLayout],
        microbatch_size: int,
    ) -> None:
        if not isinstance(microbatch_size, int) or isinstance(microbatch_size, bool) or microbatch_size <= 0:
            raise ValueError("microbatch_size must be a positive integer")
        if cache.decode_target != "gpu":
            raise ValueError("cold-source v1 supports GPU offload only")
        if cache.prefill_overlap:
            raise ValueError("cold-source v1 does not support prefill overlap")
        if cache.bank_sources or cache.banks:
            raise ValueError("cold-source binding requires a cache with no persistent host banks")
        schema = tuple(cache.bank_schema)
        if tuple(source.bank_schema) != schema:
            raise ValueError(
                f"source bank schema {tuple(source.bank_schema)!r} != cache schema {schema!r}"
            )
        if set(layouts) != set(schema):
            raise ValueError("layouts must exactly cover the cache bank schema")

        self.cache = cache
        self.source = source
        self.layouts = {name: layouts[name] for name in schema}
        self.microbatch_size = microbatch_size

        for name in schema:
            layout = self.layouts[name]
            cache.bank_caches[name] = torch.empty(
                (cache.cache_size, *layout.shape),
                dtype=layout.dtype,
                device=cache.device,
            )
        # ``bank_views`` consumes this list but the cold path never dereferences
        # the empty per-layer source lists.
        cache.banks = [([], cache.bank_caches[name]) for name in schema]
        cache._copy_fused_ok = False
        cache._copy_dst_ptrs = None
        cache._copy_src_ptrs = None
        cache._copy_feat_bytes = None
        cache._cold_source_binding = self
        # Research seam: instance-local replacement avoids changing the stable
        # cache class until the bounded probe is qualified.
        cache.copy_missing = self.copy_pending

    def _validate_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        count: int,
    ) -> dict[str, torch.Tensor]:
        if set(batch) != set(self.cache.bank_schema):
            raise ValueError("cold-source batch bank schema drift")
        checked: dict[str, torch.Tensor] = {}
        for name in self.cache.bank_schema:
            tensor = batch[name]
            layout = self.layouts[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"cold-source bank {name!r} is not a tensor")
            if tensor.device.type != "cpu":
                raise ValueError(f"cold-source bank {name!r} must be on CPU")
            if not tensor.is_contiguous():
                raise ValueError(f"cold-source bank {name!r} must be contiguous")
            if tuple(tensor.shape) != (count, *layout.shape):
                raise ValueError(
                    f"cold-source bank {name!r} shape {tuple(tensor.shape)!r} "
                    f"!= {(count, *layout.shape)!r}"
                )
            if tensor.dtype != layout.dtype:
                raise ValueError(
                    f"cold-source bank {name!r} dtype {tensor.dtype} != {layout.dtype}"
                )
            checked[name] = tensor
        return checked

    def copy_pending(self) -> ColdCopyStats:
        cache = self.cache
        layer_id = cache._pending_src_layer
        if layer_id is None:
            raise RuntimeError("no pending FreeToken miss plan")
        if cache.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cold-source v1 is graph-disabled")

        count = int(cache.num_indices.item())
        if count == 0:
            return ColdCopyStats(experts=0, microbatches=0, bank_rows=0, source_bytes=0)

        expert_ids = tuple(int(v) for v in cache.src_indices[:count].tolist())
        source_bytes = 0
        bank_rows = 0
        microbatches = 0

        for start in range(0, count, self.microbatch_size):
            stop = min(start + self.microbatch_size, count)
            experts = expert_ids[start:stop]
            n = len(experts)
            batch = self._validate_batch(self.source.load(layer_id, experts), n)
            dst = cache.evict_slots[start:stop]

            if cache.device.type == "cuda":
                from freetoken.kernel import fast_index_copy_jit

                compact_src = torch.arange(n, dtype=dst.dtype, device=cache.device)
                for name in cache.bank_schema:
                    staged = batch[name]
                    if not staged.is_pinned():
                        staged = staged.pin_memory()
                    fast_index_copy_jit(
                        cache.bank_caches[name],
                        dst,
                        staged,
                        compact_src,
                        None,
                    )
                    source_bytes += staged.numel() * staged.element_size()
                    bank_rows += n
                # Leases/buffers may die after this microbatch, so keep v1
                # explicitly synchronous. Async lifetime is a later gate.
                torch.cuda.current_stream(cache.device).synchronize()
            else:
                for name in cache.bank_schema:
                    staged = batch[name]
                    cache.bank_caches[name].index_copy_(0, dst, staged)
                    source_bytes += staged.numel() * staged.element_size()
                    bank_rows += n
            microbatches += 1

        return ColdCopyStats(
            experts=count,
            microbatches=microbatches,
            bank_rows=bank_rows,
            source_bytes=source_bytes,
        )


def bind_cold_source(
    cache,
    *,
    source: ColdExpertSource,
    layouts: Mapping[str, ColdBankLayout],
    microbatch_size: int = 4,
) -> ColdSourceBinding:
    """Attach the experimental bounded cold-source seam to ``cache``."""
    return ColdSourceBinding(
        cache=cache,
        source=source,
        layouts=layouts,
        microbatch_size=microbatch_size,
    )


__all__ = [
    "ColdBankLayout",
    "ColdTensorRange",
    "ColdExpertSource",
    "IndexedFileColdExpertSource",
    "ColdCopyStats",
    "ColdSourceBinding",
    "bind_cold_source",
]
