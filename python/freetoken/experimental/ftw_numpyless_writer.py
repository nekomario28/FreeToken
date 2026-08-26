"""Experimental NumPy-free FTW writer for low-RAM conversion.

Production :class:`FTWWriter` currently bridges CPU tensor storage through
``tensor.numpy()`` before writing.  This experiment keeps the exact FTW layout and
metadata contract but exposes the already-contiguous CPU tensor storage through a
``ctypes`` buffer view instead, avoiding both a NumPy runtime dependency and a second
whole-tensor byte copy.

The view is valid only while the local tensor is alive; ``add_tensor`` writes it
synchronously before returning.  This class is only installed into the experimental
native-CPU conversion wrapper and does not modify the normal ``ft convert`` path.
"""
from __future__ import annotations

import ctypes

import torch

from freetoken.checkpoint.ftw import ALIGN, FTWWriter, _align_up, _dtype_str


class NumpylessFTWWriter(FTWWriter):
    """FTWWriter variant whose tensor-byte bridge uses the Python buffer protocol."""

    def add_tensor(self, name: str, tensor: torch.Tensor, kind: str = "weight") -> None:
        t = tensor.detach().cpu().contiguous()
        nbytes = int(t.numel()) * int(t.element_size())

        # Keep production's shard-placement behavior exactly: a small tensor never spans
        # a shard when rolling first can keep it whole.
        if self._f is None or (
            nbytes <= self.shard_limit and self._cur + nbytes > self.shard_limit
        ):
            self._roll()
        global_off = self._global
        assert global_off % ALIGN == 0, "tensor start must be aligned (invariant)"

        if nbytes:
            owner = (ctypes.c_ubyte * nbytes).from_address(t.data_ptr())
            view = memoryview(owner).cast("B")
            try:
                self._write_raw(view)
            finally:
                view.release()
                # ``owner`` references t's storage and must not escape this synchronous write.
                del owner

        self._tensors.append(
            {
                "name": name,
                "kind": kind,
                "dtype": _dtype_str(t.dtype),
                "shape": list(t.shape),
                "global_off": global_off,
                "nbytes": nbytes,
            }
        )
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))


__all__ = ["NumpylessFTWWriter"]
