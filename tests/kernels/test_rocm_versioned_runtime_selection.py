from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
SETUP = ROOT / "setup.py"


def _load_toolchain():
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain_test", TOOLCHAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rocm_versioned_runtime_selection_is_numeric_not_lexical() -> None:
    module = _load_toolchain()
    candidates = [
        Path("/sdk/lib/libamdhip64.so.7"),
        Path("/sdk/lib/libamdhip64.so.7.9"),
        Path("/sdk/lib/libamdhip64.so.7.14"),
        Path("/sdk/lib/libamdhip64.so.debug"),
    ]
    selected = module.select_versioned_rocm_runtime(candidates)
    assert selected is not None
    assert selected.name == "libamdhip64.so.7.14"

    setup_text = SETUP.read_text()
    assert "select_versioned_rocm_runtime" in setup_text
    assert 'sorted(library_dir.glob("libamdhip64.so.*"))' not in setup_text


if __name__ == "__main__":
    test_rocm_versioned_runtime_selection_is_numeric_not_lexical()
    print("ROCM_VERSIONED_RUNTIME_SELECTION=PASS_NUMERIC_7_14_OVER_7_9")
