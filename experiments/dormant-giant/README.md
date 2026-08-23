# Dormant Giant A0 staged-source experiment

This directory is intentionally isolated from the upstream ROCm follow-up PR.

Base runtime tree: `4e11ed0a1f23e17af49ba70c31cee5336fb33ab0` (`fix(rocm): complete RDNA3 runtime path`). That tree is the real-hardware-tested FreeToken RDNA3 implementation. Do not add Dormant Giant source/cache policy to the upstream ROCm PR itself.

The only Dormant Giant execution change staged here is `a0-staged-source.patch`, copied from the canonical Dormant Giant patch. It touches only `python/freetoken/moe/offload_cache.py` and preserves FreeToken routing, victim selection, slot maps, cache ownership, copy kernel dispatch, GEMM, and activation authority.

## Static check

```bash
bash experiments/dormant-giant/run-a0-staged-source.sh --check
```

This creates a temporary detached worktree, runs `git apply --check`, applies the patch only there, byte-compiles `offload_cache.py`, checks the diff, and removes the worktree. It does not install packages or alter this branch's tracked FreeToken source.

## Physical ROCm probe

Use only an already-valid ROCm Python environment and a Dormant Giant checkout:

```bash
export DG_ROOT=/path/to/dormant-giant
bash experiments/dormant-giant/run-a0-staged-source.sh --probe
```

`--probe` requires the active Python to import ROCm PyTorch with a visible GPU. It then runs the Dormant Giant canonical resource guard, the tiny native fast-index-copy probe, the existing A0 movement smoke, and the guard again. It performs no dependency bootstrap.

The accepted physical marker is:

```text
DG_A0_EXPERIMENT=PASS_NATIVE_ROCM_STAGED_SOURCE
```

Anything else remains diagnostic/blocking evidence. In particular, FreeToken's already-recorded `gfx1101` native-runtime PASS is upstream physical reference evidence; it is not by itself a PASS for the Dormant Giant staged-source composition.

## Provenance

Canonical Dormant Giant composition:

- audited FreeToken base: `f0abe587a11cca53bb3c37a9596fad24973ace62`
- ROCm PR #23 head: `27c0977b6f2ffd476b85de116e2db839b614d76a`
- native RDNA3 follow-up: `4e11ed0a1f23e17af49ba70c31cee5336fb33ab0`
- DG seam: `patches/freetoken-f0abe587-dg-a0-staged-source.patch` in `nekomario28/dormant-giant`

Keep these pins explicit when recording a result.
