# Qwen3.8 cold expert source v1

Date: 2026-09-17

Status: `BOUNDED_IMPLEMENTATION / STATIC-PROBE STAGE / NO_PRODUCTION / NO_PREDICTIVE_PREFETCH`

## Goal

Run the PeasantSmith Qwen3.8-Flash-Next 75.2 GB GGUF without requiring the full artifact or the full routed-expert pool to be resident in RAM/VRAM.

The target memory ownership is:

```text
core/runtime state             -> VRAM as capacity permits
routed expert hot set          -> FreeToken unified VRAM slot cache
routed expert warm/cold bytes  -> immutable GGUF + Linux page cache
PLE selected n-gram rows       -> immutable GGUF + bounded staging
RAM eviction                   -> kernel page cache, not a second userspace LRU
```

## Base

- exact FreeToken upstream release head: `cac247a860e316e06580d05aeb05f2e647bde214` (`0.1.3`);
- research branch starts from that exact upstream commit, not the older fork main;
- prior Dormant Giant work already proved the older SafeTensors staged-source seam through real `gfx1101` output identity, but that physical result is not silently transferred to this new 0.1.3 code.

## What v1 now implements

### 1. Bounded file-backed expert source

`python/freetoken/moe/cold_source.py`

- exact `(layer, expert, bank) -> file range` reads with `pread`;
- no complete host expert bank;
- FreeToken remains owner of routing, victim choice, slot mapping and GPU cache;
- only the selected misses are staged;
- bounded microbatches;
- existing `fast_index_copy_jit` on the device path;
- explicit synchronization before a temporary staged buffer may die;
- OS page cache remains the RAM-tier owner.

### 2. Qwen4Exp GGUF expert-range compiler

`python/freetoken/moe/gguf_cold_source.py`

Reads GGUF metadata only and compiles the merged expert tensors:

```text
blk.L.ffn_gate_exps.weight
blk.L.ffn_up_exps.weight
blk.L.ffn_down_exps.weight
```

into exact per-expert byte ranges. The payload is not read while compiling the index.

PeasantSmith contract is fail-closed:

```text
gate/up = IQ2_XXS
down    = Q4_0
```

Gate and up stay packed and are concatenated only for the selected bounded staging batch.

For Qwen3.8 geometry `H=2560`, `I=640`:

```text
IQ2_XXS row bytes = 2560 / 256 * 66 = 660
gate              = 640 * 660       = 422,400 B
up                 = 640 * 660       = 422,400 B
Q4_0 down row      = 640 / 32 * 18   = 360 B
down               = 2560 * 360      = 921,600 B
one packed expert  = 1,766,400 B
all 48*512 experts = 40.4296875 GiB
one all-cold token = 48*10 experts   = 808.59375 MiB logical expert payload
```

This matches the published ~19.3 GiB gate/up + ~21.1 GiB down split.

### 3. Mixed IQ2_XXS gate/up + Q4_0 down compute wrapper

`python/freetoken/moe/fused_iq2_xxs_q4_0.py`

No new device kernel is introduced. FreeToken 0.1.3 already vendors both routed GGUF kernels. The wrapper calls the existing `ggml_moe_a8_vec` twice:

```text
gate/up -> GGML type 16 (IQ2_XXS)
activation
down    -> GGML type 2  (Q4_0)
```

then applies router weights and reduces top-k experts.

### 4. Disk-backed IQ4_NL GGUF PLE probe

`python/freetoken/models/qwen4_exp/ple_gguf.py`

PeasantSmith's 51.2B-parameter PLE table is not pinned into RAM. The backend:

- locates `per_layer_token_embd.weight` from GGUF metadata;
- requires IQ4_NL exactly;
- computes `160 / 32 * 18 = 90` packed bytes per n-gram row;
- reads only requested rows;
- deduplicates repeated row IDs inside a lookup;
- stages only those packed rows;
- uses existing `ggml_dequantize(..., IQ4_NL)` to produce the normal BF16 PLE rows.

v1 is synchronous/page-cache-backed on purpose. Aligned batched reads and io_uring are performance gates after correctness.

## VRAM cache sweep narrowed by exact expert size

Each unified expert slot costs exactly `1,766,400 B` for the target mixed format. The first useful sweep is therefore:

```text
2048 slots ~= 3.37 GiB  ~= 42.7 slots/layer average
3072 slots ~= 5.05 GiB  = 64.0 slots/layer average
4096 slots ~= 6.74 GiB  ~= 85.3 slots/layer average
```

Do not start with a broad heuristic sweep. First measure which of these fit after the dense model body, recurrent/KV state and runtime workspace are resident.

## What is deliberately still excluded

- predictive expert prefetch;
- custom RAM eviction / a second userspace page cache;
- DSpark/MTP;
- precision retuning beyond the published PeasantSmith quantization;
- async staged-expert lifetime;
- hybrid CPU-overflow compute from storage-backed experts;
- production enablement;
- claims about final tokens/s before a physical full-path measurement.

## Current remaining integration delta

The probes intentionally avoid modifying stable FreeToken dispatch until the contracts are clear. Remaining owner-level integration is now small and explicit:

1. add the mixed packed expert format to the existing expert bank/cache format registry;
2. route `OffloadMoELayer._expert_gemm` to the mixed wrapper;
3. add a Qwen4Exp GGUF registry/config/weight adapter, following the existing Gemma4 GGUF owner pattern;
4. bind `GgufIq4NlPleTable` instead of the FP8 SafeTensors PLE backend for the GGUF variant;
5. add exact GGUF tensor-name/shape validation against the real PeasantSmith header;
6. run one physical `gfx1101` resident-vs-file-backed packed expert output gate;
7. only then run one-layer / one-token Qwen4Exp and full-model residency sweeps.

## Evidence gates

Current order:

```text
G0  pure exact-range/source placement
G1  GGUF metadata -> exact expert ranges
G2  mixed IQ2_XXS/Q4_0 wrapper contract
G3  IQ4_NL sparse PLE row-store contract
G4  current-0.1.3 gfx1101 packed resident-vs-staged numerical identity
G5  exact PeasantSmith header/config adapter
G6  one-layer / one-token Qwen4Exp
G7  2048/3072/4096-slot physical residency sweep
G8  larger native router trace + memory traffic
G9  DSpark only if expert-union bytes / accepted token improves
```

Predictive prefetch stays HOLD until G8 provides substantially stronger locality evidence.
