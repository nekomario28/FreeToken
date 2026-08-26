"""Low-memory NVFP4 expert streaming for FTW conversion experiments.

This module is intentionally converter-only. Serving still uses the canonical
``load_expert_banks`` path. The normal NVFP4 source loader allocates HostBanks for every
MoE layer up front; a layer sink releases completed layers, but checkpoint shard ordering can
commit pages in several layers before any one layer completes. For memory-constrained FTW
conversion that is avoidable.

``stream_nvfp4_layers_serial`` groups the checkpoint index by MoE layer and processes one
layer at a time:

    allocate one layer -> read only that layer -> sink(layer) -> release -> next layer

``stream_nvfp4_layers_to_ftw`` is the converter bridge: it writes every bank of that one
layer as a ``bank#Lxxxxx`` FTW ``experts_bank`` entry and releases the layer before the next
allocation. It therefore preserves the hard one-layer allocation bound all the way through
the FTW writer boundary.

The function never pins the banks and never constructs a whole-model ExpertBanks object.
It trades repeated shard opens for a hard one-layer expert-bank allocation bound.
"""
from __future__ import annotations

import collections
import json
import os
from typing import Any, Callable

import safetensors
import torch

DropPageCache = Callable[[str], None]
LayerSink = Callable[[int, dict], None]


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Any, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


def _alloc_nvfp4_layer_banks(E: int, H: int, I: int) -> dict:
    """Allocate the six native NVFP4 HostBanks for exactly one MoE layer."""
    from freetoken.moe.host_banks import alloc_banks

    fp8 = torch.float8_e4m3fn
    return alloc_banks({
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), fp8),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), fp8),
        "down_global": ((E, H), torch.float16),
    })


def _local_or_resolved_folder(model_path: str) -> str:
    """Keep local conversion dependency-light; only import hub resolution for hub ids."""
    if os.path.isdir(model_path):
        return model_path
    from freetoken.utils import download_hf_weight

    return download_hf_weight(model_path)


def _index_by_bank_layer(model_path: str, config, spec: Any):
    """Return ``bank_layer -> shard -> [(name, match)]`` from the HF weight map."""
    folder = _local_or_resolved_folder(model_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    by_layer: dict[int, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        bank_layer = _bank_layer(spec, layer, config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = match.group("kind")
        if kind not in {"weight", "weight_scale", "weight_scale_2"}:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")
        by_layer[bank_layer][shard].append((name, match))
    return folder, by_layer


def stream_nvfp4_layers_serial(
    model_path: str,
    config,
    spec: Any,
    *,
    drop_page_cache: DropPageCache,
    layer_sink: LayerSink,
    alloc_layer: Callable[[int, int, int], dict] | None = None,
) -> dict[str, int]:
    """Stream native NVFP4 expert banks to ``layer_sink`` one layer at a time.

    This is deliberately serial and conversion-oriented. It does not pin, does not create
    GPU aliases, and does not return tensors after the sink consumes them. A valid layer has
    three projections (gate/up/down), each with packed weight, block scale, and global scale.
    The global scale is expanded into the native per-row FP16 bank when its block scale is
    placed, matching ``load_nvfp4_expert_source_banks`` byte-for-byte.

    Returns counters suitable for conversion receipts/tests. The allocation bound is
    structural: the next layer is not allocated until the sink returns for the current one.
    """
    folder, by_layer = _index_by_bank_layer(model_path, config, spec)
    E = int(config.num_experts)
    H = int(config.hidden_size)
    I = int(config.moe_intermediate_size)
    num_layers = _num_moe_layers(config)
    allocator = alloc_layer or _alloc_nvfp4_layer_banks

    expected_layers = set(range(num_layers))
    actual_layers = set(by_layer)
    if actual_layers != expected_layers:
        raise ValueError(
            f"{spec.desc}: checkpoint maps expert data to bank layers {sorted(actual_layers)}, "
            f"expected {sorted(expected_layers)}"
        )

    tensors_read = 0
    bytes_streamed = 0
    for bank_layer in range(num_layers):
        banks = allocator(E, H, I)
        gate_up_packed = banks["gate_up_packed"].tensor
        gate_up_scale = banks["gate_up_scale"].tensor
        gate_up_global = banks["gate_up_global"].tensor
        down_packed = banks["down_packed"].tensor
        down_scale = banks["down_scale"].tensor
        down_global = banks["down_global"].tensor

        # Tiny scalar globals are needed when the block scale is placed. Keep only this
        # layer's E*3 scalars; never a whole-model globals map.
        globals_map: dict[tuple[int, str], torch.Tensor] = {}
        shards = by_layer[bank_layer]
        for shard in sorted(shards):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match in shards[shard]:
                    if match.group("kind") != "weight_scale_2":
                        continue
                    key = (int(match.group("expert")), match.group("proj"))
                    globals_map[key] = f.get_tensor(name).reshape(1).to(torch.float16)
                    tensors_read += 1
            drop_page_cache(path)

        bulk_seen: set[tuple[int, str, str]] = set()
        for shard in sorted(shards):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match in shards[shard]:
                    kind = match.group("kind")
                    if kind == "weight_scale_2":
                        continue
                    expert = int(match.group("expert"))
                    proj = match.group("proj")
                    role = spec.proj_to_role[proj]
                    tensor = f.get_tensor(name)
                    if kind == "weight":
                        if role == "gate":
                            gate_up_packed[expert, :I] = tensor
                        elif role == "up":
                            gate_up_packed[expert, I:] = tensor
                        elif role == "down":
                            down_packed[expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        try:
                            global_scale = globals_map[(expert, proj)]
                        except KeyError as exc:
                            raise ValueError(
                                f"{spec.desc}: missing weight_scale_2 for bank layer "
                                f"{bank_layer}, expert {expert}, projection {proj}"
                            ) from exc
                        if role == "gate":
                            gate_up_scale[expert, :I] = tensor
                            gate_up_global[expert, :I] = global_scale
                        elif role == "up":
                            gate_up_scale[expert, I:] = tensor
                            gate_up_global[expert, I:] = global_scale
                        elif role == "down":
                            down_scale[expert] = tensor
                            down_global[expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    bulk_seen.add((expert, proj, kind))
                    tensors_read += 1
            drop_page_cache(path)

        expected_bulk = E * 3 * 2
        expected_globals = E * 3
        if len(bulk_seen) != expected_bulk or len(globals_map) != expected_globals:
            raise ValueError(
                f"{spec.desc}: incomplete bank layer {bank_layer}: "
                f"bulk={len(bulk_seen)}/{expected_bulk}, globals={len(globals_map)}/{expected_globals}"
            )

        layer_bytes = sum(bank.nbytes for bank in banks.values())
        layer_sink(bank_layer, banks)
        bytes_streamed += layer_bytes
        # The sink owns/releases ``banks`` before it returns. Crucially, no reference to
        # their tensors is retained here when the next layer is allocated.

    return {
        "layers_streamed": num_layers,
        "tensors_read": tensors_read,
        "expert_bank_bytes_streamed": bytes_streamed,
    }


def stream_nvfp4_layers_to_ftw(
    model_path: str,
    config,
    spec: Any,
    *,
    writer: Any,
    drop_page_cache: DropPageCache,
    alloc_layer: Callable[[int, int, int], dict] | None = None,
) -> dict[str, int]:
    """Write one-layer-at-a-time native NVFP4 banks into an FTW writer.

    ``writer`` only needs the canonical ``add_tensor(name, tensor, kind=...)`` method. Each
    bank is written as ``<bank>#L<layer:05d>`` with ``kind='experts_bank'``. All six banks
    of the layer are released in a ``finally`` block, including when the writer fails, so a
    conversion error cannot strand the current layer's resident pages or permit a second
    live layer to be allocated accidentally.
    """
    entries_written = 0
    bytes_written = 0

    def _sink(layer_id: int, banks: dict) -> None:
        nonlocal entries_written, bytes_written
        try:
            for bank_name, bank in banks.items():
                writer.add_tensor(
                    f"{bank_name}#L{layer_id:05d}",
                    bank.tensor,
                    kind="experts_bank",
                )
                entries_written += 1
                bytes_written += int(bank.nbytes)
        finally:
            for bank in banks.values():
                bank.release()

    stats = stream_nvfp4_layers_serial(
        model_path,
        config,
        spec,
        drop_page_cache=drop_page_cache,
        layer_sink=_sink,
        alloc_layer=alloc_layer,
    )
    if bytes_written != stats["expert_bank_bytes_streamed"]:
        raise AssertionError(
            f"FTW writer accounted {bytes_written} expert bytes, "
            f"streamer accounted {stats['expert_bank_bytes_streamed']}"
        )
    return {
        **stats,
        "ftw_entries_written": entries_written,
        "ftw_expert_bytes_written": bytes_written,
    }


__all__ = ["stream_nvfp4_layers_serial", "stream_nvfp4_layers_to_ftw"]
