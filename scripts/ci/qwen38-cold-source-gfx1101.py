#!/usr/bin/env python3
"""Physical G4: Qwen3.8 mixed packed expert cold-source identity on one GPU.

Requires the real PeasantSmith GGUF locally but never loads the complete artifact
into RAM. It reads exactly one layer's selected expert ranges, computes them once
as a compact resident batch, then routes the same experts through FreeToken's
normal slot-cache ownership plus the experimental bounded cold source.

Acceptance:
- exact packed cache bytes equal the direct selected expert bytes;
- resident and staged mixed IQ2_XXS/Q4_0 outputs are bit-identical;
- a second identical route is a cache hit and performs no additional cold load;
- no production/full-model/throughput claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from freetoken.moe.cold_source import bind_cold_source
from freetoken.moe.fused_iq2_xxs_q4_0 import fused_experts_gguf_iq2_xxs_q4_0
from freetoken.moe.gguf_cold_source import (
    Qwen4ExpGgufColdExpertSource,
    compile_qwen4exp_gguf_expert_index,
)
from freetoken.moe.offload_cache import OffloadMoeCache


@dataclass
class CountingSource:
    inner: Qwen4ExpGgufColdExpertSource
    calls: int = 0
    experts_loaded: int = 0

    @property
    def bank_schema(self):
        return self.inner.bank_schema

    def load(self, layer_id: int, expert_ids: Sequence[int]) -> Mapping[str, torch.Tensor]:
        self.calls += 1
        self.experts_loaded += len(expert_ids)
        return self.inner.load(layer_id, expert_ids)


@dataclass(frozen=True)
class Receipt:
    schema: str
    model_path: str
    model_size_bytes: int
    layer_id: int
    expert_ids: tuple[int, ...]
    cache_size: int
    microbatch_size: int
    device: str
    device_name: str
    torch_version: str
    hip_version: str | None
    packed_bytes_selected: int
    direct_output_sha256: str
    staged_output_sha256: str
    packed_cache_sha256: str
    direct_packed_sha256: str
    first_cold_calls: int
    first_experts_loaded: int
    second_cold_calls_delta: int
    second_pending_count: int
    peak_torch_vram_bytes: int
    packed_bytes_identical: bool
    output_bit_exact: bool
    second_route_cache_hit: bool
    decision: str


def _digest_tensor(t: torch.Tensor) -> str:
    raw = t.detach().contiguous().to("cpu").view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _parse_ids(raw: str) -> tuple[int, ...]:
    ids = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if not ids:
        raise ValueError("--experts must contain at least one id")
    if len(set(ids)) != len(ids):
        raise ValueError("--experts must be unique")
    if any(v < 0 or v >= 512 for v in ids):
        raise ValueError("expert ids must be in [0, 512)")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local PeasantSmith Qwen3.8 GGUF")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument(
        "--experts",
        default="5,17,53,89,127,191,255,319,383,447",
        help="comma-separated unique routed expert ids",
    )
    ap.add_argument("--cache-size", type=int, default=64)
    ap.add_argument("--microbatch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=38017)
    ap.add_argument("--receipt", default="qwen38-cold-source-g4.json")
    args = ap.parse_args()

    path = str(Path(args.model).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("G4 requires the physical GPU path")
    if not 0 <= args.layer < 48:
        raise ValueError("--layer must be in [0, 48)")
    if args.cache_size <= 0:
        raise ValueError("--cache-size must be positive")
    experts = _parse_ids(args.experts)
    if args.cache_size < len(experts):
        raise ValueError("cache must have at least as many slots as selected experts")

    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(dev)

    index = compile_qwen4exp_gguf_expert_index(path)
    direct_source = Qwen4ExpGgufColdExpertSource(index)
    counted = CountingSource(direct_source)

    # Read only the selected packed experts for the resident oracle.
    direct = direct_source.load(args.layer, experts)
    direct_gate_up = direct["gate_up"].pin_memory().to(dev, non_blocking=True)
    direct_down = direct["down"].pin_memory().to(dev, non_blocking=True)
    torch.cuda.synchronize(dev)

    n = len(experts)
    hidden = torch.randn((1, 2560), dtype=torch.bfloat16, device=dev) * 0.25
    weights = torch.rand((1, n), dtype=torch.float32, device=dev)
    weights /= weights.sum(dim=-1, keepdim=True)
    direct_ids = torch.arange(n, dtype=torch.int32, device=dev).view(1, n)
    direct_out = fused_experts_gguf_iq2_xxs_q4_0(
        hidden.clone(),
        direct_gate_up,
        direct_down,
        weights,
        direct_ids,
        "silu",
    )
    torch.cuda.synchronize(dev)

    # q4_0 supplies the same two-bank schema. The experimental binding replaces
    # only source materialization/layout; FreeToken still authors cache slots.
    cache = OffloadMoeCache(
        num_layers=48,
        num_experts=512,
        cache_size=args.cache_size,
        device=dev,
        quant_format="q4_0",
        decode_target="gpu",
        prefill_overlap=False,
    )
    bind_cold_source(
        cache,
        source=counted,
        layouts=direct_source.layouts,
        microbatch_size=args.microbatch_size,
    )

    staged_ids = torch.tensor([experts], dtype=torch.int32, device=dev)
    cache.ensure_experts(args.layer, staged_ids)
    cache.copy_missing()
    torch.cuda.synchronize(dev)

    first_calls = counted.calls
    first_loaded = counted.experts_loaded
    slot_ids = staged_ids.clone()
    views = cache.bank_views()
    staged_out = fused_experts_gguf_iq2_xxs_q4_0(
        hidden.clone(),
        views[0],
        views[1],
        weights,
        slot_ids,
        "silu",
    )
    torch.cuda.synchronize(dev)

    # Compare the exact selected slot rows, in route order, against direct packed bytes.
    slots = slot_ids.reshape(-1).long()
    staged_gate_up = views[0].index_select(0, slots).cpu()
    staged_down = views[1].index_select(0, slots).cpu()
    direct_packed = torch.cat(
        (direct["gate_up"].reshape(-1), direct["down"].reshape(-1)), dim=0
    )
    staged_packed = torch.cat(
        (staged_gate_up.reshape(-1), staged_down.reshape(-1)), dim=0
    )
    packed_equal = torch.equal(direct_packed, staged_packed)
    output_equal = torch.equal(direct_out, staged_out)

    # Same route again: ensure_experts must author hits and schedule zero cold rows.
    second_ids = torch.tensor([experts], dtype=torch.int32, device=dev)
    calls_before = counted.calls
    cache.ensure_experts(args.layer, second_ids)
    cache.copy_missing()
    torch.cuda.synchronize(dev)
    second_delta = counted.calls - calls_before
    second_pending = int(cache.num_indices.item())
    second_hit = second_delta == 0 and second_pending == 0

    decision = (
        "PASS_GFX1101_MIXED_PACKED_COLD_SOURCE_IDENTITY"
        if packed_equal and output_equal and second_hit
        else "FAIL_GFX1101_MIXED_PACKED_COLD_SOURCE_IDENTITY"
    )
    receipt = Receipt(
        schema="freetoken.qwen38-cold-source-g4.v1",
        model_path=path,
        model_size_bytes=os.path.getsize(path),
        layer_id=args.layer,
        expert_ids=experts,
        cache_size=args.cache_size,
        microbatch_size=args.microbatch_size,
        device=str(dev),
        device_name=torch.cuda.get_device_name(dev),
        torch_version=torch.__version__,
        hip_version=getattr(torch.version, "hip", None),
        packed_bytes_selected=direct_packed.numel(),
        direct_output_sha256=_digest_tensor(direct_out),
        staged_output_sha256=_digest_tensor(staged_out),
        packed_cache_sha256=_digest_tensor(staged_packed),
        direct_packed_sha256=_digest_tensor(direct_packed),
        first_cold_calls=first_calls,
        first_experts_loaded=first_loaded,
        second_cold_calls_delta=second_delta,
        second_pending_count=second_pending,
        peak_torch_vram_bytes=int(torch.cuda.max_memory_allocated(dev)),
        packed_bytes_identical=packed_equal,
        output_bit_exact=output_equal,
        second_route_cache_hit=second_hit,
        decision=decision,
    )
    out_path = Path(args.receipt)
    out_path.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n")
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))
    print(f"QWEN38_COLD_SOURCE_G4={decision}")
    return 0 if decision.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
