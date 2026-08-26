"""Opt-in low-memory FTW conversion for the file-backed CPU MoE experiment.

The canonical ``ft checkpoint`` path is intentionally unchanged.  This wrapper runs the
canonical :func:`freetoken.checkpoint.convert.convert_checkpoint` while temporarily replacing
only its expert-bank provider with the already-tested one-MoE-layer-at-a-time NVFP4 streamer.
Dense conversion, metadata copying, FTW writing/finalization, and source fingerprinting remain
owned by the canonical converter.

The experiment is deliberately narrow and fail-closed:

* Qwen3.5 ModelOpt NVFP4 MoE only;
* offload conversion with a layer sink only;
* no dummy banks, no caller-supplied residency plan, no serving fallback;
* native ``nvfp4`` FTW banks, intended for the separate file-backed CPU-only server experiment.

This module does not authorize a real checkpoint conversion.  External architecture/resource
admission must pass before real weight payloads are read.
"""
from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_NATIVE_NVFP4_BANKS = (
    "gate_up_packed",
    "gate_up_scale",
    "gate_up_global",
    "down_packed",
    "down_scale",
    "down_global",
)
_SUPPORTED_ARCH = "Qwen3_5MoeForConditionalGeneration"


def _source_spec_for_model(model_config):
    architectures = tuple(getattr(model_config, "architectures", ()) or ())
    if not architectures or architectures[0] != _SUPPORTED_ARCH:
        raise ValueError(
            "low-memory FTW conversion currently supports only "
            f"{_SUPPORTED_ARCH}; got {architectures!r}"
        )
    if getattr(model_config, "expert_quant", None) != "nvfp4":
        raise ValueError(
            "low-memory FTW conversion requires native ModelOpt NVFP4 routed experts"
        )
    module = importlib.import_module("freetoken.models.qwen3_5_moe.weight")
    return module._NVFP4_SOURCE_SPEC


def _stream_one_layer_at_a_time(model_path: str, model_config, source_spec, layer_sink) -> dict:
    from freetoken.checkpoint.low_memory_nvfp4 import stream_nvfp4_layers_serial
    from freetoken.models.loader import drop_page_cache

    return stream_nvfp4_layers_serial(
        model_path,
        model_config,
        source_spec,
        drop_page_cache=drop_page_cache,
        layer_sink=layer_sink,
    )


def _low_memory_load_expert_banks(
    model_path,
    model_config,
    *,
    device,
    dtype,
    dummy: bool = False,
    parallel=None,
    workers: int = 8,
    chunk: int = 8 << 20,
    decode_target: str = "gpu",
    layer_sink=None,
    layer_residency=None,
):
    """Canonical ``load_expert_banks``-shape adapter for converter-only streaming.

    The canonical converter supplies ``layer_sink``.  We stream directly into that sink and
    return an empty source shell marked ``streamed=True`` because the converter intentionally
    consumes no expert tensors after the sink has written and released each layer.
    """
    del device, dtype, workers, chunk, decode_target
    if dummy:
        raise ValueError("low-memory FTW conversion does not support dummy expert banks")
    if layer_sink is None:
        raise ValueError("low-memory FTW conversion requires the canonical converter layer sink")
    if layer_residency is not None:
        raise ValueError("low-memory FTW conversion does not accept a serving residency plan")
    if parallel not in (None, False):
        raise ValueError("low-memory FTW conversion owns serial one-layer source scheduling")

    source_spec = _source_spec_for_model(model_config)
    _stream_one_layer_at_a_time(model_path, model_config, source_spec, layer_sink)

    from freetoken.moe.expert_banks import ExpertBanks

    return ExpertBanks(
        "nvfp4",
        {name: [] for name in _NATIVE_NVFP4_BANKS},
        streamed=True,
    )


@contextmanager
def _patched_expert_loader() -> Iterator[None]:
    """Install the converter-only provider for one bounded call and always restore it."""
    import freetoken.moe.expert_banks as expert_banks_mod

    original = expert_banks_mod.load_expert_banks
    expert_banks_mod.load_expert_banks = _low_memory_load_expert_banks
    try:
        yield
    finally:
        expert_banks_mod.load_expert_banks = original


def convert_checkpoint_low_memory_nvfp4(
    model_path: str,
    out_dir: str,
    *,
    dtype=None,
    moe_backend: str = "offload",
    shard_limit: int | None = None,
    device: str | None = None,
    _convert_fn: Callable[..., Any] | None = None,
):
    """Run canonical FTW conversion with the bounded one-layer NVFP4 expert provider.

    ``_convert_fn`` is an internal test seam.  Production calls leave it ``None`` and execute
    the repository's canonical converter.  ``moe_backend`` must stay ``offload`` because the
    output expert-bank layout is specifically the native CPU-MoE ``nvfp4`` ABI.
    """
    if moe_backend != "offload":
        raise ValueError("low-memory FTW conversion requires moe_backend='offload'")

    if _convert_fn is None:
        from freetoken.checkpoint.convert import convert_checkpoint as _convert_fn
        from freetoken.checkpoint.ftw import DEFAULT_SHARD_LIMIT
        import torch

        if dtype is None:
            dtype = torch.bfloat16
        if shard_limit is None:
            shard_limit = DEFAULT_SHARD_LIMIT
    elif shard_limit is None:
        # The injected test converter owns its own default; avoid importing the full FTW stack.
        shard_limit = 8 << 30

    kwargs = {
        "dtype": dtype,
        "moe_backend": moe_backend,
        "shard_limit": shard_limit,
        "device": device,
    }
    with _patched_expert_loader():
        return _convert_fn(model_path, out_dir, **kwargs)


__all__ = [
    "convert_checkpoint_low_memory_nvfp4",
    "_low_memory_load_expert_banks",
    "_patched_expert_loader",
]
