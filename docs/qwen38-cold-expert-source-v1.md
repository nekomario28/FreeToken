# Qwen3.8 cold expert source v1

Date: 2026-09-17

Status: `BOUNDED_PROBE / NO_PRODUCTION / NO_PREDICTIVE_PREFETCH`

## Goal

Prove the smallest missing mechanism for running a model whose packed expert pool
is larger than host RAM:

```text
FreeToken-authored miss plan
-> exact file byte ranges
-> bounded CPU staging
-> existing FreeToken GPU slots
```

The full host expert banks are never constructed.

## Base

- FreeToken upstream release head: `cac247a860e316e06580d05aeb05f2e647bde214`
  (`0.1.3`).
- This branch starts exactly from that commit, not from the older fork main.
- Prior Dormant Giant evidence already qualified the older SafeTensors staged
  source seam through real `gfx1101` output identity. This branch does not claim
  that old physical result automatically qualifies the new 0.1.3 code.

## v1 mechanism

`python/freetoken/moe/cold_source.py` adds a deliberately isolated research seam:

- `IndexedFileColdExpertSource`: explicit `(layer, expert, bank) -> file range`
  reads with `pread`; no unrelated expert is materialized.
- `ColdSourceBinding`: consumes only `src_indices`, `evict_slots`,
  `num_indices`, and `_pending_src_layer` already authored by FreeToken.
- bounded microbatches;
- existing `fast_index_copy_jit` on CUDA;
- synchronous completion before bounded CPU staging is released;
- OS page cache remains the RAM-tier owner.

The branch does not yet parse GGUF. The next adapter will compile PeasantSmith
GGUF tensor metadata into `ColdTensorRange` entries.

## Deliberate exclusions

Not in v1:

- hybrid CPU overflow compute;
- predictive expert prefetch;
- custom RAM eviction;
- O_DIRECT/io_uring;
- PLE changes;
- IQ2_XXS kernels;
- DSpark/MTP;
- async staging lifetime;
- CUDA graph capture;
- production enablement.

## Gates

1. deterministic CPU unit probe for exact selected-row placement;
2. `gfx1101` packed Q4_0 resident-vs-file-backed output identity;
3. GGUF range-index compiler;
4. IQ2_XXS gate/up + Q4_0 down mixed-bank conformance;
5. exact Qwen3.8 one-layer / one-token probe;
6. only then full-model residency sweep.

The first physical gate must keep FreeToken routing and destination-slot
authority unchanged.
