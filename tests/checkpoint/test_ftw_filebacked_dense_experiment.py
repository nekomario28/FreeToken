from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import freetoken.experimental.ftw_filebacked_dense as dense_mod
from freetoken.checkpoint.ftw import FORMAT_TAG, FORMAT_VERSION, INDEX_NAME
from freetoken.experimental.ftw_filebacked_cpu_moe_server import _install_filebacked_dense_iterator

ALIGN = 4096


def _write_tiny_ftw(path: Path) -> None:
    entries = [
        {
            "name": "dense_a",
            "kind": "weight",
            "dtype": "uint8",
            "shape": [2, 4],
            "global_off": 0,
            "nbytes": 8,
        },
        {
            "name": "dense_b",
            "kind": "weight",
            "dtype": "uint8",
            "shape": [4],
            "global_off": ALIGN,
            "nbytes": 4,
        },
        {
            "name": "ignored_expert#L00000",
            "kind": "experts_bank",
            "dtype": "uint8",
            "shape": [4],
            "global_off": 2 * ALIGN,
            "nbytes": 4,
        },
    ]
    total = 3 * ALIGN
    shard = path / "freetoken-00000.ftw"
    with shard.open("wb") as fh:
        fh.truncate(total)
        fh.seek(0)
        fh.write(bytes([3]) * 8)
        fh.seek(ALIGN)
        fh.write(bytes([7]) * 4)
        fh.seek(2 * ALIGN)
        fh.write(bytes([11]) * 4)
    index = {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "align": ALIGN,
        "shard_limit": total,
        "total_bytes": total,
        "tensors": entries,
        "shards": [{"file": shard.name, "global_off": 0, "nbytes": total}],
    }
    (path / INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")


def test_dense_iterator_maps_weights_only_and_keeps_cow(tmp_path, monkeypatch):
    _write_tiny_ftw(tmp_path)
    drops = []
    monkeypatch.setattr(dense_mod, "_drop_mapping_pages", lambda mm: drops.append(mm))

    items = list(dense_mod.iter_ftw_weights_filebacked(str(tmp_path)))
    assert [name for name, _ in items] == ["dense_a", "dense_b"]
    assert torch.equal(items[0][1], torch.full((2, 4), 3, dtype=torch.uint8))
    assert torch.equal(items[1][1], torch.full((4,), 7, dtype=torch.uint8))
    assert len(drops) == 2

    shard = tmp_path / "freetoken-00000.ftw"
    before = shard.read_bytes()[0]
    items[0][1][0, 0] = 99
    assert int(items[0][1][0, 0]) == 99
    assert shard.read_bytes()[0] == before


def test_dense_iterator_accepts_production_compat_kwargs(tmp_path):
    _write_tiny_ftw(tmp_path)
    names = [
        name
        for name, _ in dense_mod.iter_ftw_weights_filebacked(
            str(tmp_path), workers=1, chunk=4096, prefetch=0
        )
    ]
    assert names == ["dense_a", "dense_b"]


def test_worker_dense_install_replaces_only_iterator():
    marker = object()
    fake = SimpleNamespace(iter_ftw_weights=marker, other="keep")
    _install_filebacked_dense_iterator(fake)
    assert fake.iter_ftw_weights is dense_mod.iter_ftw_weights_filebacked
    assert fake.other == "keep"
