from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.experimental import ftw_registered_dense as registered


def _write_fixture(root: Path, tensor: torch.Tensor, *, shard_limit: int = 1 << 20) -> Path:
    writer = FTWWriter(str(root), shard_limit=shard_limit)
    writer.add_tensor("dense.weight", tensor, kind="weight")
    writer.finalize({})
    return root


class RegisteredDenseTests(unittest.TestCase):
    def test_cpu_fixture_preserves_bytes_and_balances_windows(self):
        with tempfile.TemporaryDirectory() as td:
            source = torch.arange(8192, dtype=torch.int32)
            root = _write_fixture(Path(td), source)
            events = []
            with mock.patch.object(
                registered,
                "host_register_transfer",
                side_effect=lambda addr, nbytes: events.append(("register_transfer", addr, nbytes)),
            ), mock.patch.object(
                registered, "host_unregister", side_effect=lambda addr: events.append(("unregister", addr))
            ):
                target, receipt = registered.copy_ftw_dense_registered_windows(
                    root, "dense.weight", device="cpu", window_bytes=4096
                )
            self.assertTrue(torch.equal(target, source))
            self.assertEqual(receipt.nbytes, source.numel() * source.element_size())
            self.assertEqual(receipt.window_bytes, 4096)
            self.assertEqual(receipt.windows, receipt.nbytes // 4096)
            registers = [row for row in events if row[0] == "register_transfer"]
            unregisters = [row for row in events if row[0] == "unregister"]
            self.assertEqual(len(registers), receipt.windows)
            self.assertEqual(len(unregisters), receipt.windows)
            self.assertEqual([row[1] for row in registers], [row[1] for row in unregisters])
            self.assertTrue(all(row[2] == 4096 for row in registers))
            self.assertEqual(
                receipt.registration_lifetime,
                "one_window_default_register_direct_copy_unregister",
            )
            self.assertEqual(receipt.gpu_copy_path, "torch_cpu_copy")

    def test_scalar_entry_is_one_element_and_one_window(self):
        with tempfile.TemporaryDirectory() as td:
            source = torch.tensor(1.25, dtype=torch.float32)
            root = _write_fixture(Path(td), source)
            events = []
            with mock.patch.object(
                registered,
                "host_register_transfer",
                side_effect=lambda addr, nbytes: events.append(("register", addr, nbytes)),
            ), mock.patch.object(
                registered,
                "host_unregister",
                side_effect=lambda addr: events.append(("unregister", addr)),
            ):
                target, receipt = registered.copy_ftw_dense_registered_windows(
                    root, "dense.weight", device="cpu", window_bytes=4096
                )
            self.assertEqual(tuple(target.shape), ())
            self.assertTrue(torch.equal(target, source))
            self.assertEqual(receipt.shape, ())
            self.assertEqual(receipt.nbytes, 4)
            self.assertEqual(receipt.windows, 1)
            self.assertEqual([row[2] for row in events if row[0] == "register"], [4])

    def test_rejects_expert_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writer = FTWWriter(str(root), shard_limit=1 << 20)
            writer.add_tensor("bank", torch.arange(4096, dtype=torch.uint8), kind="experts_bank")
            writer.finalize({})
            with self.assertRaisesRegex(ValueError, "kind=weight"):
                registered.copy_ftw_dense_registered_windows(
                    root, "bank", device="cpu", window_bytes=4096
                )

    def test_fails_closed_when_entry_spans_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = _write_fixture(Path(td), torch.arange(8192, dtype=torch.uint8), shard_limit=4096)
            with mock.patch.object(registered, "host_register_transfer"), mock.patch.object(
                registered, "host_unregister"
            ):
                with self.assertRaisesRegex(ValueError, "exactly one shard"):
                    registered.copy_ftw_dense_registered_windows(
                        root, "dense.weight", device="cpu", window_bytes=4096
                    )

    def test_rejects_unaligned_window_before_registration(self):
        with tempfile.TemporaryDirectory() as td:
            root = _write_fixture(Path(td), torch.arange(4096, dtype=torch.uint8))
            with mock.patch.object(
                registered,
                "host_register_transfer",
                side_effect=AssertionError("must not register"),
            ):
                with self.assertRaisesRegex(ValueError, "page aligned"):
                    registered.copy_ftw_dense_registered_windows(
                        root, "dense.weight", device="cpu", window_bytes=4095
                    )

    def test_module_is_not_wired_into_default_ftw_loader(self):
        from freetoken.checkpoint import ftw

        source = Path(ftw.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ftw_registered_dense", source)
        self.assertNotIn("copy_ftw_dense_registered_windows", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
