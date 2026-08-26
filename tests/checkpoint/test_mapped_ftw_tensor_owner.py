from __future__ import annotations

import gc
import importlib.util
import json
import sys
import types
import weakref
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "python" / "freetoken" / "checkpoint"
PACKAGE = "ftwb_owner_probe"

pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(CHECKPOINT_ROOT)]
sys.modules[PACKAGE] = pkg


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load(f"{PACKAGE}.mapped_ftw_core", CHECKPOINT_ROOT / "mapped_ftw_core.py")
MAPPED = _load(f"{PACKAGE}.mapped_ftw", CHECKPOINT_ROOT / "mapped_ftw.py")


def test_bank_source_tensor_itself_retains_mapping_owner(tmp_path):
    shard = tmp_path / "freetoken-00000.ftw"
    shard.write_bytes(bytes(range(8)))
    (tmp_path / CORE.INDEX_NAME).write_text(
        json.dumps({
            "format": "freetoken_weight",
            "version": 1,
            "quant_format": "nvfp4",
            "expert_bank_num_layers": 1,
            "tensors": [{
                "name": "gate_up_packed#L00000",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [2, 4],
                "global_off": 0,
                "nbytes": 8,
            }],
            "shards": [{"file": shard.name, "global_off": 0, "nbytes": 8}],
        }),
        encoding="utf-8",
    )

    bundle = MAPPED.map_ftw_expert_sources(
        tmp_path,
        1,
        expected_banks={"gate_up_packed"},
        expected_quant_format="nvfp4",
        num_experts=2,
    )
    tensor = bundle.sources["gate_up_packed"][0]
    storage = getattr(tensor, "_freetoken_mapped_ftw_storage")
    storage_ref = weakref.ref(storage)
    assert storage_ref() is bundle.owners[0].storage
    assert storage_ref().mapping.closed is False
    assert tensor.flatten().tolist() == list(range(8))

    # Simulate the canonical Engine path: it keeps only bank source tensors after the
    # temporary ExpertBanks/bundle return object leaves scope.
    del storage
    del bundle
    gc.collect()

    retained = storage_ref()
    assert retained is not None
    assert retained.mapping.closed is False
    assert getattr(tensor, "_freetoken_mapped_ftw_storage") is retained
    assert tensor.flatten().tolist() == list(range(8))

    del retained
    del tensor
    gc.collect()
    assert storage_ref() is None
