"""Opt-in low-memory FTW conversion for the file-backed CPU MoE experiment.

Canonical ``ft checkpoint`` remains untouched. This module reuses the canonical
``convert_checkpoint`` implementation for dense weights, FTW writing, metadata copying and
fingerprinting, but temporarily substitutes narrowly-scoped low-memory pieces during one call:

* native NVFP4 experts stream exactly one layer at a time into the converter sink;
* CPU conversion skips only the owner's invalid ``cuda.set_device(cpu)`` initialization;
* the Qwen3.5 token embedding is hidden from ``safe_open().keys()`` for the owner thread and
  copied byte-for-byte from its safetensors range into FTW with a bounded buffer at finalize.

The embedding optimization is deliberately exact and model-specific. It is accepted only when
metadata proves one unquantized BF16 ``embed_tokens.weight`` source; fusion, quantized tensors,
norm transforms and every other dense weight continue through the canonical loader.

The output expert layout remains native ``quant_format='nvfp4'`` (six banks per layer), the
only layout accepted by the file-backed CPU serving experiment. All temporary process-global
patch points dispatch non-owner threads to their original behavior and are restored on success
or failure.

This is an experimental conversion route, not model-load authority. A real conversion still
requires a separate resource/admission decision before checkpoint payload is read.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable

_QWEN35_ARCH = "Qwen3_5MoeForConditionalGeneration"
_EMBED_RAW_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
)
_EMBED_FTW_NAME = "model.embed_tokens.weight"
_DENSE_PASSTHROUGH_CHUNK = 8 << 20
_EXPERT_LOADER_PATCH_LOCK = threading.RLock()
_CPU_DEVICE_INIT_PATCH_LOCK = threading.RLock()
_DENSE_SAFE_OPEN_PATCH_LOCK = threading.RLock()
_FTW_WRITER_PATCH_LOCK = threading.RLock()


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
    """Adapt the one-layer streamer to the canonical ``load_expert_banks`` call shape."""

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
    """Scope the experimental expert loader to the calling thread and always restore it."""
    with _EXPERT_LOADER_PATCH_LOCK:
        original = module.load_expert_banks
        owner_thread = threading.get_ident()

        def scoped_loader(*args, **kwargs):
            target = loader if threading.get_ident() == owner_thread else original
            return target(*args, **kwargs)

        module.load_expert_banks = scoped_loader
        try:
            yield
        finally:
            module.load_expert_banks = original


@contextmanager
def _temporary_cpu_conversion_device(torch_module: Any):
    """Skip only the canonical owner's invalid ``cuda.set_device(cpu)`` call."""
    with _CPU_DEVICE_INIT_PATCH_LOCK:
        original = torch_module.cuda.set_device
        owner_thread = threading.get_ident()

        def scoped_set_device(device):
            dev = torch_module.device(device)
            if threading.get_ident() == owner_thread and getattr(dev, "type", None) == "cpu":
                return None
            return original(device)

        torch_module.cuda.set_device = scoped_set_device
        try:
            yield
        finally:
            torch_module.cuda.set_device = original


def _resolve_embedding_passthrough(
    model_path: str,
    *,
    tensor_entry_fn: Callable[[str | os.PathLike[str], str], tuple[int, int, str, list[int]]] | None = None,
) -> dict[str, Any]:
    """Locate and validate the one Qwen3.5 BF16 embedding eligible for raw passthrough.

    Only safetensors metadata/header bytes are read. Quantized embeddings, duplicate aliases,
    missing entries and unexpected shapes fail closed before conversion payload access.
    """
    if tensor_entry_fn is None:
        from freetoken.experimental.safetensors_ftw_passthrough import _tensor_entry

        tensor_entry_fn = _tensor_entry

    root = Path(model_path)
    if not root.is_dir():
        raise ValueError("dense passthrough requires a local safetensors checkpoint directory")
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("dense passthrough found no safetensors shards")

    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    hits: list[tuple[str, Path]] = []
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(raw_map, dict):
            raise ValueError("invalid safetensors weight_map")
        weight_map = {str(k): str(v) for k, v in raw_map.items()}
        for raw_name in _EMBED_RAW_CANDIDATES:
            shard_name = weight_map.get(raw_name)
            if shard_name is not None:
                source = root / shard_name
                if not source.is_file():
                    raise ValueError(f"embedding shard is missing: {shard_name}")
                hits.append((raw_name, source))
    else:
        for source in shards:
            for raw_name in _EMBED_RAW_CANDIDATES:
                try:
                    tensor_entry_fn(source, raw_name)
                except KeyError:
                    continue
                hits.append((raw_name, source))

    if len(hits) != 1:
        raise ValueError(f"expected exactly one Qwen3.5 embedding source, got {len(hits)}")
    raw_name, source = hits[0]
    base = raw_name[: -len(".weight")]
    scale_names = (base + ".weight_scale", base + ".weight_scale_2", base + ".input_scale")
    if weight_map is not None:
        if any(name in weight_map for name in scale_names):
            raise ValueError("embedding passthrough refuses quantized/scaled source")
    else:
        for shard in shards:
            for scale_name in scale_names:
                try:
                    tensor_entry_fn(shard, scale_name)
                except KeyError:
                    continue
                raise ValueError("embedding passthrough refuses quantized/scaled source")

    _file_off, nbytes, dtype, shape = tensor_entry_fn(source, raw_name)
    if dtype != "bfloat16":
        raise ValueError(f"embedding passthrough requires BF16 source, got {dtype}")
    if len(shape) != 2 or nbytes <= 0:
        raise ValueError("embedding passthrough requires a non-empty rank-2 tensor")
    return {
        "name": _EMBED_FTW_NAME,
        "raw_name": raw_name,
        "source_path": os.path.realpath(source),
        "payload_bytes": int(nbytes),
        "dtype": dtype,
        "shape": list(shape),
    }


class _HiddenTensorHandle:
    def __init__(self, handle: Any, hidden_name: str) -> None:
        self._handle = handle
        self._hidden_name = hidden_name

    def keys(self):
        return [name for name in self._handle.keys() if name != self._hidden_name]

    def get_tensor(self, name: str):
        if name == self._hidden_name:
            raise RuntimeError("hidden passthrough tensor must not be materialized")
        return self._handle.get_tensor(name)

    def __getattr__(self, name: str):
        return getattr(self._handle, name)


class _FilteredSafeOpen:
    def __init__(self, context: Any, hidden_name: str) -> None:
        self._context = context
        self._hidden_name = hidden_name
        self._handle = None

    def __enter__(self):
        self._handle = self._context.__enter__()
        return _HiddenTensorHandle(self._handle, self._hidden_name)

    def __exit__(self, exc_type, exc, tb):
        return self._context.__exit__(exc_type, exc, tb)


@contextmanager
def _temporary_safetensors_tensor_skip(
    safetensors_module: Any,
    *,
    source_path: str,
    raw_name: str,
):
    """Hide one tensor from owner-thread ``safe_open().keys()``; delegate everything else."""
    with _DENSE_SAFE_OPEN_PATCH_LOCK:
        original = safetensors_module.safe_open
        owner_thread = threading.get_ident()
        target_path = os.path.realpath(source_path)

        def scoped_safe_open(path, *args, **kwargs):
            context = original(path, *args, **kwargs)
            if (
                threading.get_ident() == owner_thread
                and os.path.realpath(os.fspath(path)) == target_path
            ):
                return _FilteredSafeOpen(context, raw_name)
            return context

        safetensors_module.safe_open = scoped_safe_open
        try:
            yield
        finally:
            safetensors_module.safe_open = original


def _make_dense_passthrough_writer(
    base_writer_cls: type,
    target: dict[str, Any],
    *,
    chunk_bytes: int = _DENSE_PASSTHROUGH_CHUNK,
) -> type:
    """Append the hidden embedding once, then correct canonical counts before finalize."""

    class DensePassthroughWriter(base_writer_cls):
        def finalize(self, meta: dict) -> dict:
            if any(item.get("name") == target["name"] for item in self._tensors):
                raise RuntimeError("dense passthrough output already exists in FTW writer")
            receipt = self.add_safetensors_passthrough(
                name=target["name"],
                safetensors_path=target["source_path"],
                safetensors_name=target["raw_name"],
                kind="weight",
                chunk_bytes=chunk_bytes,
            )
            patched = dict(meta)
            counts = dict(patched.get("counts") or {})
            counts["weight"] = int(counts.get("weight", 0)) + 1
            patched["counts"] = counts
            patched["dense_passthrough"] = {
                "name": target["name"],
                "raw_name": target["raw_name"],
                "payload_bytes": int(receipt["payload_bytes"]),
                "max_read_buffer_bytes": int(receipt["max_read_buffer_bytes"]),
            }
            return super().finalize(patched)

    return DensePassthroughWriter


@contextmanager
def _temporary_ftw_writer(convert_module: Any, writer_cls: type):
    """Use ``writer_cls`` only for the owner conversion; other threads keep canonical writer."""
    with _FTW_WRITER_PATCH_LOCK:
        original = convert_module.FTWWriter
        owner_thread = threading.get_ident()

        def scoped_writer(*args, **kwargs):
            target = writer_cls if threading.get_ident() == owner_thread else original
            return target(*args, **kwargs)

        convert_module.FTWWriter = scoped_writer
        try:
            yield
        finally:
            convert_module.FTWWriter = original


def _require_metadata_preflight(
    model_path: str,
    out_dir: str,
    *,
    preflight_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail before tensor payload access when the metadata-only preflight blocks."""
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
    """Convert Qwen3.5 ModelOpt NVFP4 to native, file-backed CPU-readable FTW.

    The default conversion device is CPU. The metadata preflight is not resource authority;
    callers must satisfy the external conversion admission gate before invoking real payload.
    """
    _require_metadata_preflight(model_path, out_dir)

    import importlib
    import torch
    import freetoken.moe.expert_banks as expert_module
    from freetoken.checkpoint.ftw import DEFAULT_SHARD_LIMIT
    from freetoken.checkpoint.low_memory_nvfp4 import stream_nvfp4_layers_serial
    from freetoken.experimental.safetensors_ftw_passthrough import BoundedPassthroughFTWWriter
    from freetoken.models.loader import drop_page_cache
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_SOURCE_SPEC

    convert_module = importlib.import_module("freetoken.checkpoint.convert")
    qwen_weight_module = importlib.import_module("freetoken.models.qwen3_5_moe.weight")
    dense_target = _resolve_embedding_passthrough(model_path)
    writer_cls = _make_dense_passthrough_writer(BoundedPassthroughFTWWriter, dense_target)
    loader = _make_low_memory_native_nvfp4_loader(
        streamer=stream_nvfp4_layers_serial,
        spec=_NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
    )
    conversion_device = "cpu" if device is None else device

    with _temporary_expert_loader(expert_module, loader):
        with _temporary_cpu_conversion_device(torch):
            with _temporary_safetensors_tensor_skip(
                qwen_weight_module.safetensors,
                source_path=dense_target["source_path"],
                raw_name=dense_target["raw_name"],
            ):
                with _temporary_ftw_writer(convert_module, writer_cls):
                    return convert_module.convert_checkpoint(
                        model_path,
                        out_dir,
                        dtype=torch.bfloat16 if dtype is None else dtype,
                        moe_backend="offload",
                        shard_limit=DEFAULT_SHARD_LIMIT if shard_limit is None else shard_limit,
                        device=conversion_device,
                    )


__all__ = [
    "convert_file_backed_ftw_cpu",
    "_make_dense_passthrough_writer",
    "_make_low_memory_native_nvfp4_loader",
    "_require_metadata_preflight",
    "_resolve_embedding_passthrough",
    "_temporary_cpu_conversion_device",
    "_temporary_expert_loader",
    "_temporary_ftw_writer",
    "_temporary_safetensors_tensor_skip",
]
