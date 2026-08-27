from __future__ import annotations

import json

import pytest
import torch

from freetoken.checkpoint.ftw import FTWWriter
from freetoken.experimental import ftw_registered_dense as registered


def _write_fixture(tmp_path, tensor: torch.Tensor, *, shard_limit: int = 1 << 20):
    writer = FTWWriter(str(tmp_path), shard_limit=shard_limit)
    writer.add_tensor("dense.weight", tensor, kind="weight")
    writer.finalize({})
    return tmp_path


def test_registered_dense_cpu_fixture_preserves_bytes_and_balances_windows(tmp_path, monkeypatch):
    source = torch.arange(8192, dtype=torch.int32)
    root = _write_fixture(tmp_path, source)
    events = []
    monkeypatch.setattr(registered, "host_register", lambda addr, nbytes: events.append(("register", addr, nbytes)))
    monkeypatch.setattr(registered, "host_unregister", lambda addr: events.append(("unregister", addr)))

    target, receipt = registered.copy_ftw_dense_registered_windows(
        root, "dense.weight", device="cpu", window_bytes=4096
    )

    assert torch.equal(target, source)
    assert receipt.nbytes == source.numel() * source.element_size()
    assert receipt.window_bytes == 4096
    assert receipt.windows == receipt.nbytes // 4096
    registers = [row for row in events if row[0] == "register"]
    unregisters = [row for row in events if row[0] == "unregister"]
    assert len(registers) == len(unregisters) == receipt.windows
    assert [row[1] for row in registers] == [row[1] for row in unregisters]
    assert all(row[2] == 4096 for row in registers)


def test_registered_dense_rejects_expert_entry(tmp_path):
    writer = FTWWriter(str(tmp_path), shard_limit=1 << 20)
    writer.add_tensor("bank", torch.arange(4096, dtype=torch.uint8), kind="experts_bank")
    writer.finalize({})
    with pytest.raises(ValueError, match="kind=weight"):
        registered.copy_ftw_dense_registered_windows(
            tmp_path, "bank", device="cpu", window_bytes=4096
        )


def test_registered_dense_fails_closed_when_entry_spans_shards(tmp_path, monkeypatch):
    source = torch.arange(8192, dtype=torch.uint8)
    root = _write_fixture(tmp_path, source, shard_limit=4096)
    monkeypatch.setattr(registered, "host_register", lambda *_: None)
    monkeypatch.setattr(registered, "host_unregister", lambda *_: None)
    with pytest.raises(ValueError, match="single shard"):
        registered.copy_ftw_dense_registered_windows(
            root, "dense.weight", device="cpu", window_bytes=4096
        )


def test_registered_dense_rejects_unaligned_window_before_registration(tmp_path, monkeypatch):
    source = torch.arange(4096, dtype=torch.uint8)
    root = _write_fixture(tmp_path, source)
    monkeypatch.setattr(registered, "host_register", lambda *_: pytest.fail("must not register"))
    with pytest.raises(ValueError, match="page aligned"):
        registered.copy_ftw_dense_registered_windows(
            root, "dense.weight", device="cpu", window_bytes=4095
        )


def test_registered_dense_module_is_not_wired_into_default_ftw_loader():
    from freetoken.checkpoint import ftw

    source = open(ftw.__file__, encoding="utf-8").read()
    assert "ftw_registered_dense" not in source
    assert "copy_ftw_dense_registered_windows" not in source
