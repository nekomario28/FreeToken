from __future__ import annotations

import importlib.util
import json
import mmap
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "checkpoint" / "mapped_ftw_core.py"
)
SPEC = importlib.util.spec_from_file_location("mapped_ftw_core_convergence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def _write_index(root: Path, *, tensors, shards, **meta):
    (root / CORE.INDEX_NAME).write_text(
        json.dumps({
            "format": "freetoken_weight", "version": 1,
            "tensors": tensors, "shards": shards, **meta,
        }),
        encoding="utf-8",
    )


def _entry(name="gate_up_packed#L00000", *, off=0, nbytes=8, kind="experts_bank"):
    return {
        "name": name, "kind": kind, "dtype": "uint8", "shape": [nbytes],
        "global_off": off, "nbytes": nbytes,
    }


class MappedFTWCoreTests(unittest.TestCase):
    def test_private_mapping_is_cow_and_owner_stays_live(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = bytes(range(8))
            (root / "freetoken-00000.ftw").write_bytes(payload)
            _write_index(root, tensors=[_entry()], shards=[{
                "file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 8,
            }])
            owner = CORE.map_ftw_range(root, "gate_up_packed#L00000")
            self.assertEqual(owner.mapping[owner.data_offset:owner.data_offset + 8], payload)
            owner.mapping[owner.data_offset] = 255
            self.assertEqual(owner.mapping[owner.data_offset], 255)
            self.assertEqual(owner.shard_path.read_bytes(), payload)
            self.assertFalse(owner.mapping.closed)

    def test_unaligned_file_offset_maps_aligned_enclosing_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prefix = b"p" * 123
            payload = b"expert"
            (root / "freetoken-00000.ftw").write_bytes(prefix + payload)
            _write_index(root, tensors=[_entry(off=123, nbytes=len(payload))], shards=[{
                "file": "freetoken-00000.ftw", "global_off": 0,
                "nbytes": len(prefix) + len(payload),
            }])
            owner = CORE.map_ftw_range(root, "gate_up_packed#L00000")
            self.assertEqual(owner.mapping_offset % mmap.ALLOCATIONGRANULARITY, 0)
            self.assertEqual(owner.data_offset, 123 - owner.mapping_offset)
            self.assertEqual(
                owner.mapping[owner.data_offset:owner.data_offset + len(payload)], payload
            )

    def test_cross_shard_entry_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.ftw").write_bytes(b"abcd")
            (root / "b.ftw").write_bytes(b"efgh")
            _write_index(root, tensors=[_entry(off=0, nbytes=8)], shards=[
                {"file": "a.ftw", "global_off": 0, "nbytes": 4},
                {"file": "b.ftw", "global_off": 4, "nbytes": 4},
            ])
            with self.assertRaisesRegex(ValueError, "exactly one shard"):
                CORE.map_ftw_range(root, "gate_up_packed#L00000")

    def test_missing_duplicate_and_out_of_file_ranges_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.ftw").write_bytes(b"abcd")
            row = _entry(nbytes=4)
            _write_index(root, tensors=[row, dict(row)], shards=[
                {"file": "a.ftw", "global_off": 0, "nbytes": 4}
            ])
            with self.assertRaisesRegex(ValueError, "exactly one FTW tensor"):
                CORE.map_ftw_range(root, row["name"])
            with self.assertRaisesRegex(ValueError, "exactly one FTW tensor"):
                CORE.map_ftw_range(root, "missing#L00000")

            _write_index(root, tensors=[_entry(nbytes=8)], shards=[
                {"file": "a.ftw", "global_off": 0, "nbytes": 8}
            ])
            with self.assertRaisesRegex(ValueError, "exceeds its shard file"):
                CORE.map_ftw_range(root, "gate_up_packed#L00000")

    def test_shard_path_cannot_escape_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "checkpoint"
            root.mkdir()
            (base / "outside.ftw").write_bytes(b"abcdefgh")
            _write_index(root, tensors=[_entry()], shards=[{
                "file": "../outside.ftw", "global_off": 0, "nbytes": 8,
            }])
            with self.assertRaisesRegex(ValueError, "escapes checkpoint directory"):
                CORE.map_ftw_range(root, "gate_up_packed#L00000")

    def test_per_layer_grouping_matches_existing_source_shape(self):
        tensors = []
        for bank in ("gate_up_packed", "down_packed"):
            for layer in range(2):
                tensors.append(_entry(f"{bank}#L{layer:05d}", off=len(tensors) * 8))
        tensors.append({
            "name": "gate_up_alpha", "kind": "experts_bank", "dtype": "float32",
            "shape": [2], "global_off": 32, "nbytes": 8,
        })
        index = {
            "format": "freetoken_weight", "version": 1,
            "expert_bank_num_layers": 2, "tensors": tensors, "shards": [],
        }
        grouped = CORE.group_per_layer_expert_entries(
            index, 2, expected_banks={"gate_up_packed", "down_packed"}
        )
        self.assertEqual(grouped["gate_up_packed"], [
            "gate_up_packed#L00000", "gate_up_packed#L00001"
        ])
        self.assertEqual(grouped["down_packed"], [
            "down_packed#L00000", "down_packed#L00001"
        ])

    def test_grouping_rejects_flat_missing_and_mismatched_layout(self):
        flat = {
            "format": "freetoken_weight", "tensors": [_entry("gate_up_packed")],
            "shards": [], "expert_bank_num_layers": 2,
        }
        with self.assertRaisesRegex(ValueError, "require per-layer entries"):
            CORE.group_per_layer_expert_entries(flat, 2)

        missing = {
            "format": "freetoken_weight", "tensors": [_entry("gate_up_packed#L00000")],
            "shards": [], "expert_bank_num_layers": 2,
        }
        with self.assertRaisesRegex(ValueError, "expected \[0, 1\]"):
            CORE.group_per_layer_expert_entries(missing, 2)
        with self.assertRaisesRegex(ValueError, "does not match requested"):
            CORE.group_per_layer_expert_entries(missing, 3)


if __name__ == "__main__":
    unittest.main()
