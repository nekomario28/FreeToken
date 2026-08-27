from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.experimental import ftw_readonly_dense as readonly


def _write_fixture(root: Path, tensor: torch.Tensor, *, shard_limit: int = 1 << 20) -> Path:
    writer = FTWWriter(str(root), shard_limit=shard_limit)
    writer.add_tensor("dense.weight", tensor, kind="weight")
    writer.finalize({})
    return root


def _shard_hash(root: Path) -> str:
    shard = next(root.glob("*.bin"))
    return hashlib.sha256(shard.read_bytes()).hexdigest()


class ReadonlyDenseTests(unittest.TestCase):
    def test_cpu_fixture_preserves_bytes_and_file_identity(self):
        with tempfile.TemporaryDirectory() as td:
            source = torch.arange(8192, dtype=torch.int32)
            root = _write_fixture(Path(td), source)
            before = _shard_hash(root)
            events = []
            with mock.patch.object(
                readonly,
                "host_register_transfer",
                side_effect=lambda addr, nbytes: events.append(("register", addr, nbytes)),
            ), mock.patch.object(
                readonly, "host_unregister", side_effect=lambda addr: events.append(("unregister", addr))
            ):
                target, receipt = readonly.copy_ftw_dense_readonly_windows(
                    root, "dense.weight", device="cpu", window_bytes=4096
                )
            self.assertTrue(torch.equal(target, source))
            self.assertEqual(before, _shard_hash(root))
            self.assertEqual(receipt.source_storage, "file_backed_readonly_mmap")
            self.assertEqual(receipt.gpu_copy_path, "torch_cpu_copy")
            registers = [row for row in events if row[0] == "register"]
            unregisters = [row for row in events if row[0] == "unregister"]
            self.assertEqual(len(registers), receipt.windows)
            self.assertEqual([row[1] for row in registers], [row[1] for row in unregisters])

    def test_rejects_expert_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writer = FTWWriter(str(root), shard_limit=1 << 20)
            writer.add_tensor("bank", torch.arange(4096, dtype=torch.uint8), kind="experts_bank")
            writer.finalize({})
            with self.assertRaisesRegex(ValueError, "kind=weight"):
                readonly.copy_ftw_dense_readonly_windows(
                    root, "bank", device="cpu", window_bytes=4096
                )

    def test_fails_closed_when_entry_spans_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = _write_fixture(Path(td), torch.arange(8192, dtype=torch.uint8), shard_limit=4096)
            with mock.patch.object(readonly, "host_register_transfer"), mock.patch.object(
                readonly, "host_unregister"
            ):
                with self.assertRaisesRegex(ValueError, "exactly one shard"):
                    readonly.copy_ftw_dense_readonly_windows(
                        root, "dense.weight", device="cpu", window_bytes=4096
                    )

    def test_rejects_unaligned_window_before_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = _write_fixture(Path(td), torch.arange(4096, dtype=torch.uint8))
            with mock.patch.object(
                readonly,
                "host_register_transfer",
                side_effect=AssertionError("must not register"),
            ):
                with self.assertRaisesRegex(ValueError, "page aligned"):
                    readonly.copy_ftw_dense_readonly_windows(
                        root, "dense.weight", device="cpu", window_bytes=4095
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
