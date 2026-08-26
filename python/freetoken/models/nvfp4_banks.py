from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
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


def _alloc_nvfp4_host_banks(num_layers: int, E: int, H: int, I: int):
    """6 NVFP4 source banks, one ``[E, ...]`` tensor per layer (independent allocations),
    unpinned (pin-after-fill): register only after fill to skip cudaHostAlloc's slow
    commit. Caller fills each layer's ``.tensor`` then pins it (per-layer, via
    ``PinPipeline``, as its writes complete)."""
    from freetoken.moe.host_banks import alloc_layer_banks

    fp8 = torch.float8_e4m3fn
    return alloc_layer_banks({
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), fp8),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), fp8),
        "down_global": ((E, H), torch.float16),
    }, num_layers)


def _bounded_serial_nvfp4_sources(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    layer_sink,
) -> dict[str, list[torch.Tensor]]:
    """Converter-only hard-bound path: allocate/fill/release exactly one MoE layer at a time.

    The public loader ABI still returns ``dict[bank, list[Tensor]]``. Those tensor objects are
    retained only as released source-shape placeholders after ``layer_sink`` has consumed the
    layer; ``HostBank.release()`` drops their resident anonymous pages with ``MADV_DONTNEED``.
    The converter checks ``ExpertBanks.streamed`` and never reads these released tensors.
    """
    from freetoken.checkpoint.low_memory_nvfp4 import stream_nvfp4_layers_serial

    num_layers = _num_moe_layers(config)
    sources: dict[str, list[torch.Tensor | None]] = {}

    def _sink(layer_id: int, banks: dict) -> None:
        names = set(banks)
        if not sources:
            sources.update({name: [None] * num_layers for name in sorted(names)})
        elif names != set(sources):
            raise AssertionError(
                f"{spec.desc}: streamed layer {layer_id} banks {sorted(names)} "
                f"differ from first layer {sorted(sources)}"
            )

        # Retain only the tensor view needed to preserve the historical loader ABI. The sink
        # owns HostBank release; on a sink failure, drop every current-layer bank defensively.
        refs = {name: bank.tensor for name, bank in banks.items()}
        try:
            layer_sink(layer_id, banks)
        except BaseException:
            for bank in banks.values():
                bank.release()
            raise
        for name, tensor in refs.items():
            sources[name][layer_id] = tensor

    stats = stream_nvfp4_layers_serial(
        model_path,
        config,
        spec,
        drop_page_cache=drop_page_cache,
        layer_sink=_sink,
    )
    if int(stats["layers_streamed"]) != num_layers:
        raise AssertionError(
            f"{spec.desc}: bounded streamer reported {stats['layers_streamed']} layers, "
            f"expected {num_layers}"
        )
    if not sources or any(tensor is None for per_layer in sources.values() for tensor in per_layer):
        raise AssertionError(f"{spec.desc}: bounded streamer returned an incomplete source geometry")
    return {
        name: [tensor for tensor in per_layer if tensor is not None]
        for name, per_layer in sources.items()
    }


def load_nvfp4_expert_source_banks(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Build the 6 native NVFP4 source banks by streaming checkpoint shards (serial per-shard read).

    ModelOpt row layout: gate/up fused on the output-row axis, down separate; the per-tensor
    global scale (weight_scale_2) is kept as a separate per-output-row FP16 bank (``*_global``),
    so dequant is ``fp4 * block_scale * global``. Each bank is one ``[E, ...]`` tensor per
    layer, indexed by ``[bank_layer][expert]``. (The marlin/b12x backends repack these and
    fold the global into per-expert alphas; see moe/nvfp4_backends.py.)

    ``layer_sink=None`` (serving): pin each bank layer as its writes complete, via an
    internally-owned :class:`PinPipeline`.

    ``layer_sink`` given (converter; for marlin/b12x the provider wraps it in a per-layer
    repacking sink first): use the hard-bounded serial converter path, which allocates and
    fills exactly one MoE layer before handing it to the sink and releasing its resident
    pages. The returned tensor lists preserve the loader ABI only; their released contents
    must not be read after streaming.
    """
    if layer_sink is not None:
        return _bounded_serial_nvfp4_sources(
            model_path,
            config,
            spec,
            drop_page_cache=drop_page_cache,
            layer_sink=layer_sink,
        )

    folder = download_hf_weight(model_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    for shard in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard))

    weight_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    global_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
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
        if kind == "weight_scale_2":
            global_shards[shard].append((name, match, bank_layer))
        elif kind in {"weight", "weight_scale"}:
            weight_shards[shard].append((name, match, bank_layer))
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_shards):
        path = os.path.join(folder, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name, match, _bank_layer_id in global_shards[shard]:
                key = (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("proj"),
                )
                globals_map[key] = f.get_tensor(name).to(torch.float16)
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for shard in tqdm(sorted(weight_shards), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match, bank_layer_id in weight_shards[shard]:
                    layer = int(match.group("layer"))
                    expert = int(match.group("expert"))
                    proj = match.group("proj")
                    role = spec.proj_to_role[proj]
                    kind = match.group("kind")
                    tensor = f.get_tensor(name)
                    if kind == "weight":
                        if role == "gate":
                            gate_up_packed[bank_layer_id][expert, :I] = tensor
                        elif role == "up":
                            gate_up_packed[bank_layer_id][expert, I:] = tensor
                        elif role == "down":
                            down_packed[bank_layer_id][expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        global_scale = globals_map[(layer, expert, proj)]
                        if role == "gate":
                            gate_up_scale[bank_layer_id][expert, :I] = tensor
                            gate_up_global[bank_layer_id][expert, :I] = global_scale
                        elif role == "up":
                            gate_up_scale[bank_layer_id][expert, I:] = tensor
                            gate_up_global[bank_layer_id][expert, I:] = global_scale
                        elif role == "down":
                            down_scale[bank_layer_id][expert] = tensor
                            down_global[bank_layer_id][expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    tracker.note(bank_layer_id)
                    placed += 1
            drop_page_cache(path)
        return placed

    with PinPipeline() as pins:
        placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


def load_nvfp4_expert_source_banks_parallel(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel counterpart of :func:`load_nvfp4_expert_source_banks`.

    Serving (``layer_sink=None``) keeps the chunked multi-threaded O_DIRECT reader. Converter
    calls deliberately fall back to the hard-bounded serial path: the parallel reader retains
    whole-shard anonymous prefetch buffers and therefore cannot provide the one-layer memory
    bound that low-RAM FTW conversion requires.
    """
    if layer_sink is not None:
        return load_nvfp4_expert_source_banks(
            model_path,
            config,
            spec,
            drop_page_cache=drop_page_cache,
            primary=primary,
            layer_sink=layer_sink,
        )

    from freetoken.models.weight import iter_expert_tensors_parallel

    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    weight_info: dict[str, tuple[re.Match[str], int]] = {}  # name -> (match, bank_layer)
    global_names_by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        kind = match.group("kind")
        if kind == "weight_scale_2":
            global_names_by_shard[shard].append(name)
        elif kind in {"weight", "weight_scale"}:
            weight_info[name] = (match, bank_layer)
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    # Pass 1: tiny per-tensor global scales (serial; data is scalar-per-expert).
    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_names_by_shard):
        path = os.path.join(folder, shard)
        drop_page_cache(path)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in global_names_by_shard[shard]:
                m = spec.key_pattern.match(name)
                globals_map[(int(m.group("layer")), int(m.group("expert")), m.group("proj"))] = (
                    f.get_tensor(name).to(torch.float16)
                )
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    # Pass 2: bulk weight/weight_scale via the common parallel reader; place by name.
    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for name, tensor in iter_expert_tensors_parallel(
            folder, lambda n: n in weight_info, workers=workers, chunk=chunk
        ):
            match, bank_layer_id = weight_info[name]
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            proj = match.group("proj")
            role = spec.proj_to_role[proj]
            kind = match.group("kind")
            if kind == "weight":
                if role == "gate":
                    gate_up_packed[bank_layer_id][expert, :I] = tensor
                elif role == "up":
                    gate_up_packed[bank_layer_id][expert, I:] = tensor
                else:
                    down_packed[bank_layer_id][expert] = tensor
            else:
                g = globals_map[(layer, expert, proj)]
                if role == "gate":
                    gate_up_scale[bank_layer_id][expert, :I] = tensor
                    gate_up_global[bank_layer_id][expert, :I] = g
                elif role == "up":
                    gate_up_scale[bank_layer_id][expert, I:] = tensor
                    gate_up_global[bank_layer_id][expert, I:] = g
                else:
                    down_scale[bank_layer_id][expert] = tensor
                    down_global[bank_layer_id][expert] = g
            tracker.note(bank_layer_id)
            placed += 1
        return placed

    with PinPipeline() as pins:
        placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


__all__ = [
    "Nvfp4ExpertSourceSpec",
    "load_nvfp4_expert_source_banks",
    "load_nvfp4_expert_source_banks_parallel",
]
