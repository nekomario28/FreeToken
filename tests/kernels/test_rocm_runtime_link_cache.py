from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
from unittest import mock


UTILS = Path(__file__).parents[2] / "python" / "freetoken" / "kernel" / "utils.py"


def _load_utils_module():
    spec = importlib.util.spec_from_file_location("_freetoken_kernel_utils_test", UTILS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_versioned_rocm_runtime_link_cache_tracks_runtime_origin() -> None:
    module = _load_utils_module()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        home = root / "home"
        first_root = root / "rocm-first"
        second_root = root / "rocm-second"
        first_lib = first_root / "lib" / "libamdhip64.so.7.14"
        second_lib = second_root / "lib" / "libamdhip64.so.7.15"
        first_lib.parent.mkdir(parents=True)
        second_lib.parent.mkdir(parents=True)
        home.mkdir()
        first_lib.write_bytes(b"first")
        second_lib.write_bytes(b"second")

        with mock.patch.dict(os.environ, {"HOME": str(home), "ROCM_HOME": str(first_root)}, clear=False):
            module._rocm_link_flags.cache_clear()
            first_flags = module._rocm_link_flags()

        first_link_dir = Path(next(flag[2:] for flag in first_flags if flag.startswith("-L")))
        first_link = first_link_dir / "libamdhip64.so"
        assert first_link.is_symlink()
        assert first_link.resolve() == first_lib.resolve()

        # Model a long-lived cache surviving a ROCm SDK/image change. A stale
        # shared compat symlink must not pin JIT linking to the vanished runtime.
        first_lib.unlink()

        with mock.patch.dict(os.environ, {"HOME": str(home), "ROCM_HOME": str(second_root)}, clear=False):
            module._rocm_link_flags.cache_clear()
            second_flags = module._rocm_link_flags()

        second_link_dir = Path(next(flag[2:] for flag in second_flags if flag.startswith("-L")))
        second_link = second_link_dir / "libamdhip64.so"
        assert second_link.is_symlink()
        assert second_link.exists()
        assert second_link.resolve() == second_lib.resolve()
        assert second_link_dir != first_link_dir


if __name__ == "__main__":
    test_versioned_rocm_runtime_link_cache_tracks_runtime_origin()
    print("ROCM_RUNTIME_LINK_CACHE_REFRESH=PASS")
