"""Opt-in low-memory FTW conversion for the file-backed CPU MoE experiment.

Canonical ``ft checkpoint`` remains untouched.  This module reuses the canonical
``convert_checkpoint`` implementation for dense weights, FTW writing, metadata copying and
fingerprinting, but temporarily substitutes the expert-bank provider during that one call.
The substitute accepts only Qwen3.5 native NVFP4 and only the converter's ``layer_sink``
contract.  It streams exactly one native expert layer at a time and never constructs the
whole-model anonymous HostBank set.

The output expert layout is deliberately ``quant_format='nvfp4'`` (six native banks per
layer), which is the only layout accepted by the file-backed CPU serving experiment.
GPU-oriented marlin/b12x FTW layouts are never silently reinterpreted as CPU banks.

This is an experimental conversion route, not model-load authority.  A real conversion still
requires a separate resource/admission decision before checkpoint payload is read.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

_QWEN35_ARCH = "Qwen3_5MoeForConditionalGeneration"


def _architecture(model_config) -> str | None:
    architectures = getattr(model_config, "architectures", None)
    if not architectures:
        return None
    return str(architectures[0])


def _make_low_memory_native_nvfp4_loader(
    *,
    streamer: Callable[..., dict[str, int]],
    spec: Any,
    drop_page_cache: Callable[[str], None],
    bundle_factory: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Adapt the one-layer streamer to the canonical ``load_expert_banks`` call shape.

    The converter passes a ``layer_sink`` that writes/release each completed layer.  The
    returned bundle intentionally carries no materialized ``sources``: once ``streamed`` is
    true, canonical ``convert_checkpoint`` consumes only the sink output, quant format and
    layer count observed by its sink.  Keeping ``sources`` empty makes an accidental serving
    use fail loudly instead of retaining invalid/released tensor views.
    """

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
        del device, dtype, parallel, workers, chunk, decode_target
        if dummy:
            raise ValueError("low-memory file-backed FTW conversion does not support dummy banks")
        if layer_sink is None:
            raise ValueError(
                "low-memory file-backed FTW expert loader is converter-only and requires layer_sink"
            )
        if layer_residency is not None:
            raise ValueError("converter-only expert streaming does not accept serving residency")
        if _architecture(model_config) != _QWEN35_ARCH:
            raise ValueError(
                "low-memory file-backed FTW conversion currently supports only "
                f"{_QWEN35_ARCH}"
            )
        if getattr(model_config, "expert_quant", None) != "nvfp4":
            raise ValueError(
                "low-memory file-backed FTW conversion requires native ModelOpt NVFP4 experts"
            )

        stats = streamer(
            model_path,
            model_config,
            spec,
            drop_page_cache=drop_page_cache,
            layer_sink=layer_sink,
        )
        expected_layers = int(model_config.num_moe_layers)
        actual_layers = int(stats.get("layers_streamed", -1))
        if actual_layers != expected_layers:
            raise RuntimeError(
                "low-memory NVFP4 streamer produced an incomplete layer set: "
                f"{actual_layers}/{expected_layers}"
            )
        if bundle_factory is None:
            from freetoken.moe.expert_banks import ExpertBanks

            return ExpertBanks("nvfp4", {}, streamed=True)
        return bundle_factory(quant_format="nvfp4", sources={}, streamed=True)

    return load_expert_banks


@contextmanager
def _temporary_expert_loader(module: Any, loader: Callable[..., Any]):
    """Patch one module attribute for exactly one conversion call and always restore it."""
    original = module.load_expert_banks
    module.load_expert_banks = loader
    try:
        yield
    finally:
        module.load_expert_banks = original


def _require_metadata_preflight(
    model_path: str,
    out_dir: str,
    *,
    preflight_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail before tensor payload access when the metadata-only preflight blocks.

    A successful preflight is deliberately not a resource authorization; it only proves
    that the local checkpoint metadata/layout is eligible for this experimental path.
    """
    if preflight_fn is None:
        from freetoken.experimental.file_backed_ftw_cpu_preflight import (
            preflight_file_backed_ftw_cpu_conversion,
        )

        preflight_fn = preflight_file_backed_ftw_cpu_conversion
    result = preflight_fn(model_path, out_dir)
    if result.get("admission") == "BLOCK":
        blockers = result.get("blockers") or ["unspecified metadata preflight blocker"]
        raise ValueError(
            "file-backed FTW CPU conversion preflight blocked: " + "; ".join(map(str, blockers))
        )
    if result.get("admission") != "METADATA_OK_RESOURCE_UNPROVEN":
        raise ValueError(
            "file-backed FTW CPU conversion preflight returned unexpected admission: "
            f"{result.get('admission')!r}"
        )
    return result


def convert_file_backed_ftw_cpu(
    model_path: str,
    out_dir: str,
    *,
    dtype=None,
    shard_limit: int | None = None,
    device: str | None = None,
):
    """Convert Qwen3.5 ModelOpt NVFP4 to native, per-layer FTW for CPU file-backed serving.

    This function intentionally has no model-download/load side effects beyond those already
    performed by canonical ``convert_checkpoint``.  Callers are responsible for external
    resource admission before invoking it on a real checkpoint.
    """
    _require_metadata_preflight(model_path, out_dir)

    import torch
    import freetoken.moe.expert_banks as expert_module
    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.checkpoint.ftw import DEFAULT_SHARD_LIMIT
    from freetoken.checkpoint.low_memory_nvfp4 import stream_nvfp4_layers_serial
    from freetoken.models.loader import drop_page_cache
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_SOURCE_SPEC

    loader = _make_low_memory_native_nvfp4_loader(
        streamer=stream_nvfp4_layers_serial,
        spec=_NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
    )
    with _temporary_expert_loader(expert_module, loader):
        return convert_checkpoint(
            model_path,
            out_dir,
            dtype=torch.bfloat16 if dtype is None else dtype,
            moe_backend="offload",
            shard_limit=DEFAULT_SHARD_LIMIT if shard_limit is None else shard_limit,
            device=device,
        )


__all__ = [
    "convert_file_backed_ftw_cpu",
    "_make_low_memory_native_nvfp4_loader",
    "_require_metadata_preflight",
    "_temporary_expert_loader",
]
