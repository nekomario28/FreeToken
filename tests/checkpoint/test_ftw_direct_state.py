from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.experimental import ftw_direct_state as direct_state
from freetoken.experimental.ftw_registered_dense import RegisteredDenseTransferReceipt


class DirectDenseStateTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, torch.Tensor]:
        tensors = {
            "same": torch.arange(4096, dtype=torch.uint8),
            "cast": torch.arange(2048, dtype=torch.bfloat16),
            "extra": torch.tensor([1.5], dtype=torch.float32),
        }
        writer = FTWWriter(str(root), shard_limit=1 << 20)
        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor, kind="weight")
        writer.finalize({})
        return tensors

    def test_complete_contract_preserves_keys_and_gpu_cast_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tensors = self._fixture(root)
            expected = {
                "same": direct_state.DenseExpectedSpec((4096,), torch.uint8),
                "cast": direct_state.DenseExpectedSpec((2048,), torch.float32),
                "extra": direct_state.DenseExpectedSpec((1,), torch.float32),
            }

            def fake_copy(path, name, *, device, window_bytes):
                source = tensors[name].clone()
                return source, RegisteredDenseTransferReceipt(
                    name=name,
                    dtype=str(source.dtype).removeprefix("torch."),
                    shape=tuple(source.shape),
                    nbytes=source.numel() * source.element_size(),
                    window_bytes=window_bytes,
                    windows=1,
                    source_storage="file_backed_readonly_mmap",
                )

            with mock.patch.object(direct_state, "copy_ftw_dense_registered_windows", side_effect=fake_copy):
                state, receipt = direct_state.materialize_ftw_dense_state_direct(
                    root, expected, device="cpu", window_bytes=4096
                )

            self.assertEqual(set(state), set(expected))
            self.assertEqual(state["same"].dtype, torch.uint8)
            self.assertEqual(state["cast"].dtype, torch.float32)
            self.assertEqual(state["extra"].dtype, torch.float32)
            self.assertTrue(torch.equal(state["same"], tensors["same"]))
            self.assertTrue(torch.equal(state["cast"], tensors["cast"].to(torch.float32)))
            self.assertEqual(receipt.tensor_count, 3)
            self.assertEqual(receipt.dtype_cast_count, 1)
            self.assertEqual(
                receipt.transfer_path,
                "file_backed_readonly_registered_window_direct_runtime_h2d",
            )
            self.assertGreater(receipt.max_cast_source_bytes, 0)
            self.assertGreater(receipt.max_cast_final_bytes, receipt.max_cast_source_bytes)

    def test_key_contract_fails_closed_before_payload_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._fixture(root)
            expected = {"same": direct_state.DenseExpectedSpec((4096,), torch.uint8)}
            with mock.patch.object(
                direct_state,
                "copy_ftw_dense_registered_windows",
                side_effect=AssertionError("payload copy must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "key contract mismatch"):
                    direct_state.materialize_ftw_dense_state_direct(
                        root, expected, device="cpu", window_bytes=4096
                    )

    def test_shape_contract_fails_closed_before_payload_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._fixture(root)
            expected = {
                "same": direct_state.DenseExpectedSpec((4095,), torch.uint8),
                "cast": direct_state.DenseExpectedSpec((2048,), torch.bfloat16),
                "extra": direct_state.DenseExpectedSpec((1,), torch.float32),
            }
            with mock.patch.object(
                direct_state,
                "copy_ftw_dense_registered_windows",
                side_effect=AssertionError("payload copy must not start"),
            ):
                with self.assertRaisesRegex(ValueError, "shape"):
                    direct_state.materialize_ftw_dense_state_direct(
                        root, expected, device="cpu", window_bytes=4096
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
