"""Converter-only native-NVFP4 provider for the low-memory FTW experiment.

The canonical ``ft checkpoint`` path is not modified. This module supplies a bounded adapter
with the same call shape as ``freetoken.moe.expert_banks.load_expert_banks`` and installs it
only for one canonical conversion call. The dispatcher is owner-thread scoped: unrelated
threads continue to use the original loader while the experiment runs.

The path is deliberately narrow and fail-closed:

* Qwen3.5 ModelOpt NVFP4 MoE only;
* canonical converter ``layer_sink`` required;
* no dummy banks or serving residency plan;
* strict one-layer serial expert scheduling;
* native ``nvfp4`` bundle only;
* complete streamed layer count required before a streamed bundle is returned.
"""
from __future__ import annotations

import importlib
import threading
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
_EXPERT_LOADER_PATCH_LOCK = threading.RLock()


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


def _make_low_memory_loader(
    expert_banks_cls,
    *,
    source_spec_for_model: Callable[[Any], Any] = _source_spec_for_model,
    stream_one_layer: Callable[[str, Any, Any, Any], dict] = _stream_one_layer_at_a_time,
):
    """Build the canonical ``load_expert_banks``-shape converter adapter."""

    def load_expert_banks(
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
        del device, dtype, workers, chunk, decode_target
        if dummy:
            raise ValueError("low-memory FTW conversion does not support dummy expert banks")
        if layer_sink is None:
            raise ValueError("low-memory FTW conversion requires the canonical converter layer sink")
        if layer_residency is not None:
            raise ValueError("low-memory FTW conversion does not accept a serving residency plan")
        if parallel not in (None, False):
            raise ValueError("low-memory FTW conversion owns serial one-layer source scheduling")

        source_spec = source_spec_for_model(model_config)
        stats = stream_one_layer(model_path, model_config, source_spec, layer_sink)
        expected_layers = int(model_config.num_moe_layers)
        actual_layers = int((stats or {}).get("layers_streamed", -1))
        if actual_layers != expected_layers:
            raise RuntimeError(
                "low-memory NVFP4 streamer produced an incomplete layer set: "
                f"{actual_layers}/{expected_layers}"
            )
        return expert_banks_cls(
            "nvfp4",
            {name: [] for name in _NATIVE_NVFP4_BANKS},
            streamed=True,
        )

    return load_expert_banks


@contextmanager
def _patched_expert_loader(expert_banks_mod=None) -> Iterator[Callable[..., Any]]:
    """Install an owner-thread-scoped converter provider and always restore the original.

    ``convert_checkpoint`` imports ``load_expert_banks`` at call time. A plain global
    monkeypatch could therefore hijack unrelated serving/conversion work in another thread.
    This dispatcher routes only the thread that entered the context to the experimental
    loader; all other threads keep using the original. Contexts are serialized so dispatchers
    cannot stack over each other.
    """
    if expert_banks_mod is None:
        import freetoken.moe.expert_banks as expert_banks_mod

    with _EXPERT_LOADER_PATCH_LOCK:
        original = expert_banks_mod.load_expert_banks
        replacement = _make_low_memory_loader(expert_banks_mod.ExpertBanks)
        owner_thread = threading.get_ident()

        def scoped_loader(*args, **kwargs):
            target = replacement if threading.get_ident() == owner_thread else original
            return target(*args, **kwargs)

        expert_banks_mod.load_expert_banks = scoped_loader
        try:
            yield replacement
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
    _expert_banks_mod=None,
):
    """Run canonical FTW conversion with the bounded native-NVFP4 expert provider."""
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
        shard_limit = 8 << 30

    kwargs = {
        "dtype": dtype,
        "moe_backend": moe_backend,
        "shard_limit": shard_limit,
        "device": device,
    }
    with _patched_expert_loader(_expert_banks_mod):
        return _convert_fn(model_path, out_dir, **kwargs)


__all__ = [
    "convert_checkpoint_low_memory_nvfp4",
    "_make_low_memory_loader",
    "_patched_expert_loader",
]
