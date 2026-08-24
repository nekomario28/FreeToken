from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_rocm_dlpack_devices_are_explicit_not_macro_aliased():
    utils = _read("python/freetoken/kernel/csrc/include/freetoken/utils.cuh")
    fast = _read("python/freetoken/kernel/csrc/jit/fast_index_copy.cuh")
    index = _read("python/freetoken/kernel/csrc/jit/index.cu")
    store = _read("python/freetoken/kernel/csrc/jit/store.cu")
    pynccl = _read("python/freetoken/kernel/csrc/src/pynccl.cu")

    # Do not globally rewrite DLPack CUDA tokens under HIP. Besides being hard to
    # reason about, that would also silently change the still-CUDA/NCCL-only
    # multi-GPU wrapper, which is outside this RDNA3 single-GPU PR's scope.
    assert "#define kDLCUDA kDLROCM" not in utils
    assert "#define kDLCUDAHost kDLROCMHost" not in utils

    # Single-GPU JIT paths that execute on ROCm explicitly accept ROCm DLPack
    # devices while retaining CUDA acceptance for the existing NVIDIA path.
    assert (
        ".with_device<kDLCUDA, kDLCUDAHost, kDLROCM, kDLROCMHost, kDLCPU>()"
        in fast
    )
    assert "dev.device_type == kDLCUDA || dev.device_type == kDLROCM" in fast
    assert fast.count(".with_device<kDLCUDA, kDLROCM>(device)") >= 6
    assert index.count(".with_device<kDLCUDA, kDLROCM>(device_)") == 3
    assert store.count(".with_device<kDLCUDA, kDLROCM>(device_)") == 3

    # Preserve the PR's stated boundary: RCCL/multi-GPU migration is not being
    # claimed by a side effect of a preprocessor alias.
    assert "kDLROCM" not in pynccl
    assert "device_type == kDLCUDA" in pynccl
