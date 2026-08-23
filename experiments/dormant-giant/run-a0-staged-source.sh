#!/usr/bin/env bash
set -euo pipefail

# Non-destructive Dormant Giant A0 composition harness for the real-hardware-tested
# FreeToken RDNA3 tree. It never installs dependencies and never edits this checkout:
# the DG patch is applied only inside a temporary detached worktree.

NATIVE_HEAD="4e11ed0a1f23e17af49ba70c31cee5336fb33ab0"
MODE="${1:---check}"
ROOT="$(git rev-parse --show-toplevel)"
PATCH_REL="experiments/dormant-giant/a0-staged-source.patch"
PATCH="$ROOT/$PATCH_REL"

case "$MODE" in
  --check|--probe) ;;
  *) echo "usage: $0 [--check|--probe]" >&2; exit 2 ;;
esac

test -f "$PATCH"
git -C "$ROOT" merge-base --is-ancestor "$NATIVE_HEAD" HEAD

TMP="$(mktemp -d "${TMPDIR:-/tmp}/freetoken-dg-a0.XXXXXX")"
WORKTREE="$TMP/tree"
cleanup() {
  git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

git -C "$ROOT" worktree add --detach "$WORKTREE" HEAD >/dev/null

git -C "$WORKTREE" apply --check "$PATCH_REL"
git -C "$WORKTREE" apply "$PATCH_REL"
python -m py_compile "$WORKTREE/python/freetoken/moe/offload_cache.py"
git -C "$WORKTREE" diff --check

for marker in \
  'def init_staged_bank_caches(' \
  'def set_staged_copy_callback(' \
  'def copy_missing_bank_from_compact(' \
  'staged A0 forbids persistent full host banks' \
  'staged_copy(self, layer_id)'; do
  grep -Fq "$marker" "$WORKTREE/python/freetoken/moe/offload_cache.py"
done

if [[ "$MODE" == "--check" ]]; then
  echo "DG_A0_EXPERIMENT=PASS_PATCH_APPLY_STATIC native=$NATIVE_HEAD"
  exit 0
fi

: "${DG_ROOT:?set DG_ROOT to a Dormant Giant checkout containing scripts/ci}"
DG_ROOT="$(cd "$DG_ROOT" && pwd)"
test -f "$DG_ROOT/scripts/ci/resource-guard.fish"
test -f "$DG_ROOT/scripts/ci/freetoken-a0-rocm-pr23-contract.py"
test -f "$DG_ROOT/scripts/ci/freetoken-a0-cuda-copy-smoke.py"

# Never bootstrap/install here. A complete already-validated ROCm PyTorch environment
# must be active before this physical probe is allowed to run.
python - <<'PY'
import torch
hip = getattr(torch.version, "hip", None)
assert hip is not None, torch.__version__
assert torch.cuda.is_available()
print(f"DG_A0_ROCM_ENV torch={torch.__version__} hip={hip} device={torch.cuda.get_device_name(0)}")
PY

fish "$DG_ROOT/scripts/ci/resource-guard.fish"

export PYTHONPATH="$WORKTREE/python:$DG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$DG_ROOT/scripts/ci/freetoken-a0-rocm-pr23-contract.py" "$WORKTREE" --probe-copy
python "$DG_ROOT/scripts/ci/freetoken-a0-cuda-copy-smoke.py"

fish "$DG_ROOT/scripts/ci/resource-guard.fish"
echo "DG_A0_EXPERIMENT=PASS_NATIVE_ROCM_STAGED_SOURCE native=$NATIVE_HEAD"
