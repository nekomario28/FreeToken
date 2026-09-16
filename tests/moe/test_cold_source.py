import os

import pytest
import torch

from freetoken.moe.cold_source import (
    ColdBankLayout,
    ColdTensorRange,
    IndexedFileColdExpertSource,
    bind_cold_source,
)


def _write_fixture(tmp_path, *, layers=1, experts=4):
    path = tmp_path / "packed-experts.bin"
    gate_layout = ColdBankLayout((2, 3), torch.uint8)
    down_layout = ColdBankLayout((2, 2), torch.uint8)
    layouts = {"gate_up": gate_layout, "down": down_layout}
    schema = ("gate_up", "down")
    index = {}
    expected = {}
    offset = 0

    with path.open("wb") as fh:
        for layer in range(layers):
            for expert in range(experts):
                for bank in schema:
                    layout = layouts[bank]
                    value = 10 * layer + 2 * expert + (1 if bank == "down" else 0)
                    payload = bytes([value]) * layout.nbytes
                    fh.write(payload)
                    index[(layer, expert, bank)] = ColdTensorRange(
                        str(path), offset, layout
                    )
                    expected[(layer, expert, bank)] = torch.full(
                        layout.shape, value, dtype=layout.dtype
                    )
                    offset += len(payload)
    return path, schema, layouts, index, expected


def test_indexed_file_source_reads_requested_packed_rows(tmp_path):
    _, schema, layouts, index, expected = _write_fixture(tmp_path)
    source = IndexedFileColdExpertSource(bank_schema=schema, index=index)

    batch = source.load(0, (3, 1))

    assert set(batch) == set(schema)
    assert batch["gate_up"].shape == (2, *layouts["gate_up"].shape)
    assert batch["down"].shape == (2, *layouts["down"].shape)
    assert torch.equal(batch["gate_up"][0], expected[(0, 3, "gate_up")])
    assert torch.equal(batch["gate_up"][1], expected[(0, 1, "gate_up")])
    assert torch.equal(batch["down"][0], expected[(0, 3, "down")])
    assert torch.equal(batch["down"][1], expected[(0, 1, "down")])


def test_cold_binding_copies_only_freetoken_authored_rows(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _, schema, layouts, index, expected = _write_fixture(tmp_path)
    source = IndexedFileColdExpertSource(bank_schema=schema, index=index)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        quant_format="q4_0",
    )
    binding = bind_cold_source(
        cache,
        source=source,
        layouts=layouts,
        microbatch_size=1,
    )
    for bank in schema:
        cache.bank_caches[bank].zero_()

    # Freeze a synthetic FreeToken-authored pending miss plan: expert 3 -> slot 5,
    # expert 1 -> slot 2. The cold source is not allowed to choose destinations.
    cache._pending_src_layer = 0
    cache.src_indices[:2] = torch.tensor([3, 1], dtype=cache.src_indices.dtype)
    cache.evict_slots[:2] = torch.tensor([5, 2], dtype=cache.evict_slots.dtype)
    cache.num_indices[0] = 2

    stats = cache.copy_missing()

    assert stats.experts == 2
    assert stats.microbatches == 2
    assert stats.bank_rows == 4
    assert cache.bank_sources == {}
    assert cache._cold_source_binding is binding
    assert torch.equal(cache.bank_caches["gate_up"][5], expected[(0, 3, "gate_up")])
    assert torch.equal(cache.bank_caches["gate_up"][2], expected[(0, 1, "gate_up")])
    assert torch.equal(cache.bank_caches["down"][5], expected[(0, 3, "down")])
    assert torch.equal(cache.bank_caches["down"][2], expected[(0, 1, "down")])
    assert torch.count_nonzero(cache.bank_caches["gate_up"][[0, 1, 3, 4]]) == 0
    assert torch.count_nonzero(cache.bank_caches["down"][[0, 1, 3, 4]]) == 0


def test_cold_binding_rejects_full_host_banks(tmp_path):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _, schema, layouts, index, _ = _write_fixture(tmp_path)
    source = IndexedFileColdExpertSource(bank_schema=schema, index=index)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
        quant_format="q4_0",
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.zeros((4, *layouts["gate_up"].shape), dtype=torch.uint8)],
            "down": [torch.zeros((4, *layouts["down"].shape), dtype=torch.uint8)],
        }
    )

    with pytest.raises(ValueError, match="no persistent host banks"):
        bind_cold_source(cache, source=source, layouts=layouts)


def test_indexed_source_fails_closed_on_short_read(tmp_path):
    path, schema, layouts, index, _ = _write_fixture(tmp_path)
    os.truncate(path, os.path.getsize(path) - 1)
    source = IndexedFileColdExpertSource(bank_schema=schema, index=index)

    with pytest.raises(OSError, match="short cold-source read"):
        source.load(0, (3,))
