from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HIP_COMPAT = ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include" / "freetoken" / "hip_compat.h"


def test_rocm_host_pointer_capability_is_not_aliased_to_uva():
    text = HIP_COMPAT.read_text()

    assert (
        "#define cudaDevAttrCanUseHostPointerForRegisteredMem "
        "hipDeviceAttributeCanUseHostPointerForRegisteredMem"
    ) in text
    assert (
        "#define cudaDevAttrCanUseHostPointerForRegisteredMem "
        "hipDeviceAttributeUnifiedAddressing"
    ) not in text


if __name__ == "__main__":
    test_rocm_host_pointer_capability_is_not_aliased_to_uva()
    print("ROCM_HOST_POINTER_SOURCE_CONTRACT=PASS")
