from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).parents[2]
PYTHON_ROOT = ROOT / "python"
UTILS = PYTHON_ROOT / "freetoken" / "kernel" / "utils.py"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _load_utils_module():
    spec = importlib.util.spec_from_file_location("_freetoken_kernel_utils_test", UTILS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_versioned_rocm_runtime_link_cache_tracks_runtime_origin_and_numeric_version() -> None:
    module = _load_utils_module()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        home = root / "home"
        first_root = root / "rocm-first"
        second_root = root / "rocm-second"
        first_old = first_root / "lib" / "libamdhip64.so.7.9"
        first_new = first_root / "lib" / "libamdhip64.so.7.14"
        first_debug = first_root / "lib" / "libamdhip64.so.debug"
        second_lib = second_root / "lib" / "libamdhip64.so.7.15"
        first_old.parent.mkdir(parents=True)
        second_lib.parent.mkdir(parents=True)
        home.mkdir()
        first_old.write_bytes(b"first-old")
        first_new.write_bytes(b"first-new")
        first_debug.write_bytes(b"debug")
        second_lib.write_bytes(b"second")

        with mock.patch.dict(os.environ, {"HOME": str(home), "ROCM_HOME": str(first_root)}, clear=False):
            module._rocm_link_flags.cache_clear()
            first_flags = module._rocm_link_flags()

        first_link_dir = Path(next(flag[2:] for flag in first_flags if flag.startswith("-L")))
        first_link = first_link_dir / "libamdhip64.so"
        assert first_link.is_symlink()
        assert first_link.resolve() == first_new.resolve()

        # Model a long-lived cache surviving a ROCm SDK/image change. A stale
        # compat symlink must not pin JIT linking to the vanished runtime, and
        # the first selection must be numeric (7.14 > 7.9), not lexical.
        first_new.unlink()

        with mock.patch.dict(os.environ, {"HOME": str(home), "ROCM_HOME": str(second_root)}, clear=False):
            module._rocm_link_flags.cache_clear()
            second_flags = module._rocm_link_flags()

        second_link_dir = Path(next(flag[2:] for flag in second_flags if flag.startswith("-L")))
        second_link = second_link_dir / "libamdhip64.so"
        assert second_link.is_symlink()
        assert second_link.exists()
        assert second_link.resolve() == second_lib.resolve()
        assert second_link_dir != first_link_dir

        utils_text = UTILS.read_text()
        assert "select_versioned_rocm_runtime" in utils_text
        assert 'sorted(library_dir.glob("libamdhip64.so.*"))' not in utils_text


if __name__ == "__main__":
    test_versioned_rocm_runtime_link_cache_tracks_runtime_origin_and_numeric_version()
    print("ROCM_JIT_RUNTIME_RESOLUTION=PASS_NUMERIC_SELECTION_AND_CACHE_LIFETIME")
