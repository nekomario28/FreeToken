from __future__ import annotations

import importlib.util
import json
import mmap
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "freetoken"
    / "checkpoint"
    / "mapped_ftw_core.py"
)
SPEC = importlib.util.spec_from_file_location("mapped_ftw_core_p0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def _write_fixture(
    root: Path,
    *,
    payload: bytes = bytes(range(32)),
    global_off: int = 0,
    shards: list[dict] | None = None,
    tensors: list[dict] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if shards is None:
        shards = [{"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": len(payload)}]
    if tensors is None:
        tensors = [
            {
                "name": "gate_up_packed#L00000",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [len(payload)],
                "global_off": global_off,
                "nbytes": len(payload),
            }
        ]
    (root / "freetoken-00000.ftw").write_bytes(payload)
    index = {
        "format": "freetoken_weight",
        "version": 1,
        "align": 4096,
        "tensors": tensors,
        "shards": shards,
    }
    (root / CORE.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")


class MappedFTWCoreTests(unittest.TestCase):
    def test_private_mapping_reads_payload_without_checkpoint_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = bytes(range(32))
            _write_fixture(root, payload=payload)
            owner = CORE.map_ftw_range(root, "gate_up_packed#L00000")

            self.assertEqual(owner.mapping[owner.data_offset : owner.data_offset + 32], payload)
            owner.mapping[owner.data_offset] = 255
            self.assertEqual(owner.mapping[owner.data_offset], 255)
            self.assertEqual(owner.shard_path.read_bytes(), payload)
            self.assertFalse(owner.mapping.closed)

    def test_unaligned_shard_relative_offset_uses_aligned_enclosing_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prefix = b"p" * 123
            payload = b"expert-bank"
            full = prefix + payload
            _write_fixture(
                root,
                payload=full,
                global_off=len(prefix),
                shards=[{"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": len(full)}],
                tensors=[
                    {
                        "name": "gate_up_packed#L00000",
                        "kind": "experts_bank",
                        "dtype": "uint8",
                        "shape": [len(payload)],
                        "global_off": len(prefix),
                        "nbytes": len(payload),
                    }
                ],
            )
            owner = CORE.map_ftw_range(root, "gate_up_packed#L00000")
            self.assertEqual(owner.mapping_offset % mmap.ALLOCATIONGRANULARITY, 0)
            self.assertEqual(owner.data_offset, len(prefix) - owner.mapping_offset)
            self.assertEqual(
                owner.mapping[owner.data_offset : owner.data_offset + len(payload)], payload
            )

    def test_entry_spanning_multiple_shards_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = b"abcdefgh"
            _write_fixture(
                root,
                payload=payload[:4],
                shards=[
                    {"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 4},
                    {"file": "freetoken-00001.ftw", "global_off": 4, "nbytes": 4},
                ],
                tensors=[
                    {
                        "name": "gate_up_packed#L00000",
                        "kind": "experts_bank",
                        "dtype": "uint8",
                        "shape": [8],
                        "global_off": 0,
                        "nbytes": 8,
                    }
                ],
            )
            (root / "freetoken-00001.ftw").write_bytes(payload[4:])
            with self.assertRaisesRegex(ValueError, "exactly one shard"):
                CORE.map_ftw_range(root, "gate_up_packed#L00000")

    def test_duplicate_or_missing_tensor_name_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry = {
                "name": "gate_up_packed#L00000",
                "kind": "experts_bank",
                "dtype": "uint8",
                "shape": [4],
                "global_off": 0,
                "nbytes": 4,
            }
            _write_fixture(root, payload=b"abcd", tensors=[entry, dict(entry)])
            with self.assertRaisesRegex(ValueError, "exactly one FTW tensor"):
                CORE.map_ftw_range(root, "gate_up_packed#L00000")
            with self.assertRaisesRegex(ValueError, "exactly one FTW tensor"):
                CORE.map_ftw_range(root, "missing#L00000")

    def test_malformed_index_and_out_of_file_range_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            root.mkdir(exist_ok=True)
            (root / CORE.INDEX_NAME).write_text(
                json.dumps({"format": "wrong", "tensors": [], "shards": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a FreeToken Weight checkpoint"):
                CORE.map_ftw_range(root, "x")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fixture(
                root,
                payload=b"abcd",
                shards=[{"file": "freetoken-00000.ftw", "global_off": 0, "nbytes": 8}],
                tensors=[
                    {
                        "name": "gate_up_packed#L00000",
                        "kind": "experts_bank",
                        "dtype": "uint8",
                        "shape": [8],
                        "global_off": 0,
                        "nbytes": 8,
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "exceeds its shard file"):
                CORE.map_ftw_range(root, "gate_up_packed#L00000")


if __name__ == "__main__":
    unittest.main()
