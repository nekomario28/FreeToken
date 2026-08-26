"""Metadata-only anonymous-RAM envelope for Qwen3.5 mixed-precision FTW conversion.

This module mirrors the allocation-relevant parts of
``models/qwen3_5_moe/weight.py`` and ``checkpoint/ftw.py`` without opening tensor payloads
or importing torch. It is intentionally narrow: ModelOpt MIXED_PRECISION Qwen3.5 MoE
with per-tensor FP8 attention/GDN, native NVFP4 routed/shared experts and optional native
NVFP4 lm_head.

The estimate is an anonymous-allocation upper bound for the dense conversion iterator,
not a whole-process RSS prediction and not resource admission. Safetensors source tensors
are file-backed views; their reclaimable page-cache pressure belongs in an external margin.
Known conversion outputs are contiguous, so ``FTWWriter.add_tensor`` does not require a
second payload-sized tensor copy on these paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

_QWEN35_ARCH = "Qwen3_5MoeForConditionalGeneration"
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
_FP8_FUSIONS = {
    ".self_attn.qkv_proj": (
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
    ),
    ".linear_attn.in_proj_qkvz": (
        ".linear_attn.in_proj_qkv",
        ".linear_attn.in_proj_z",
    ),
}
_BF16_FUSIONS = {
    ".linear_attn.in_proj_ba": (
        ".linear_attn.in_proj_b",
        ".linear_attn.in_proj_a",
    ),
}
_NVFP4_MLP_LAYOUTS = (
    (
        ".mlp.shared_expert.gate_proj",
        ".mlp.shared_expert.up_proj",
        ".mlp.shared_expert.down_proj",
        ".mlp.shared_expert.",
    ),
    (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".mlp."),
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    @property
    def rows(self) -> int:
        return self.shape[0] if self.shape else 1


def _product(shape: Iterable[int]) -> int:
    total = 1
    for value in shape:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError("tensor shape dimensions must be non-negative")
        total *= ivalue
    return total


def _meta(name: str, spec: dict[str, Any]) -> TensorMeta:
    dtype = str(spec.get("dtype") or "")
    shape = spec.get("shape")
    offsets = spec.get("data_offsets")
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype for dense envelope: {dtype!r}")
    if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
        raise ValueError("invalid safetensors shape")
    if not (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(isinstance(v, int) and v >= 0 for v in offsets)
        and offsets[1] >= offsets[0]
    ):
        raise ValueError("invalid safetensors data_offsets")
    nbytes = int(offsets[1]) - int(offsets[0])
    expected = _product(shape) * _DTYPE_BYTES[dtype]
    if nbytes != expected:
        raise ValueError("safetensors dtype/shape byte count disagrees with data_offsets")
    return TensorMeta(name=name, dtype=dtype, shape=tuple(shape), nbytes=nbytes)


def _rename(raw_name: str) -> str | None:
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _quant_get(config: dict[str, Any], key: str, default=None):
    quant = config.get("quantization_config")
    return quant.get(key, default) if isinstance(quant, dict) else default


def _is_supported_mixed(config: dict[str, Any]) -> bool:
    archs = config.get("architectures") or []
    arch = archs[0] if isinstance(archs, list) and archs else None
    algo = str(_quant_get(config, "quant_algo", "") or "").lower()
    method = str(_quant_get(config, "quant_method", "") or "").lower()
    return arch == _QWEN35_ARCH and method == "modelopt" and "mixed" in algo


def _expert_quant_nvfp4(config: dict[str, Any]) -> bool:
    layers = _quant_get(config, "quantized_layers", {})
    if not isinstance(layers, dict):
        return False
    for name, spec in layers.items():
        if not (str(name).endswith(".mlp.experts") or ".mlp.experts." in str(name)):
            continue
        if isinstance(spec, dict) and "fp4" in str(spec.get("quant_algo", "")).lower():
            return True
    return False


def _lm_head_nvfp4(config: dict[str, Any]) -> bool:
    layers = _quant_get(config, "quantized_layers", {})
    if not isinstance(layers, dict):
        return False
    for name, spec in layers.items():
        if str(name) == "lm_head" or str(name).endswith(".lm_head"):
            if isinstance(spec, dict) and "fp4" in str(spec.get("quant_algo", "")).lower():
                return True
    return False


def _fusion_key(base: str, groups: dict[str, tuple[str, ...]]) -> str | None:
    for fused_suffix, parts in groups.items():
        for part in parts:
            if base.endswith(part):
                return base[: -len(part)] + fused_suffix
    return None


def _is_gemma_norm(name: str) -> bool:
    return name == "model.norm.weight" or name.endswith(_GEMMA_NORM_SUFFIXES)


def estimate_qwen35_mixed_dense_anonymous_peak(
    config: dict[str, Any],
    headers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return a fail-closed anonymous allocation envelope from safetensors metadata only.

    ``headers`` is the merged tensor-name -> safetensors metadata mapping. Tensor names are
    used only for classification and are not returned.
    """
    if not _is_supported_mixed(config):
        raise ValueError("dense envelope supports only Qwen3.5 ModelOpt MIXED_PRECISION")
    if not _expert_quant_nvfp4(config):
        raise ValueError("dense envelope requires native NVFP4 routed experts")

    metas = {name: _meta(name, spec) for name, spec in headers.items() if name != "__metadata__"}
    lmhead_native = _lm_head_nvfp4(config)
    dense_native = True  # parse_config: expert_quant==nvfp4 -> dense_quant==nvfp4.

    counts: dict[str, int] = {}
    max_by_class: dict[str, int] = {}
    unsupported = 0
    local_peak = 0
    largest_source_tensor = 0
    fp8_groups: dict[str, list[TensorMeta]] = {}
    bf16_groups: dict[str, list[TensorMeta]] = {}
    nvfp4_groups: dict[str, list[tuple[TensorMeta, TensorMeta, TensorMeta]]] = {}

    def record(kind: str, anonymous_bytes: int) -> None:
        nonlocal local_peak
        counts[kind] = counts.get(kind, 0) + 1
        max_by_class[kind] = max(max_by_class.get(kind, 0), int(anonymous_bytes))
        local_peak = max(local_peak, int(anonymous_bytes))

    for raw_name, meta in metas.items():
        if ".mlp.experts." in raw_name:
            continue
        if raw_name.endswith(_SCALE_SUFFIXES):
            continue
        name = _rename(raw_name)
        if name is None:
            continue
        largest_source_tensor = max(largest_source_tensor, meta.nbytes)

        if not name.endswith(".weight"):
            record("passthrough", 0)
            continue

        base = name[: -len(".weight")]
        raw_base = raw_name[: -len(".weight")]
        scale = metas.get(raw_base + ".weight_scale")
        scale2 = metas.get(raw_base + ".weight_scale_2")

        if scale is not None and scale2 is None:
            fusion_key = _fusion_key(base, _FP8_FUSIONS)
            if fusion_key is not None:
                fp8_groups.setdefault(fusion_key, []).append(meta)
                continue
            record("fp8_standalone", meta.rows * 4)
            continue

        if scale2 is not None:
            is_lmhead = base == "lm_head" or base.endswith(".lm_head")
            native_target = (lmhead_native and is_lmhead) or (
                dense_native and any(
                    base.endswith((gate, up, down))
                    for gate, up, down, _infix in _NVFP4_MLP_LAYOUTS
                )
            )
            if not native_target:
                unsupported += 1
                continue
            global_bytes = meta.rows * 2
            if is_lmhead:
                record("nvfp4_standalone", global_bytes)
                continue
            grouped = False
            for gate, up, down, infix in _NVFP4_MLP_LAYOUTS:
                if base.endswith(down):
                    record("nvfp4_standalone", global_bytes)
                    grouped = True
                    break
                if base.endswith(gate) or base.endswith(up):
                    key = base.rsplit(infix, 1)[0] + infix + "gate_up_proj"
                    nvfp4_groups.setdefault(key, []).append((meta, scale, scale2))
                    grouped = True
                    break
            if not grouped:
                unsupported += 1
            continue

        fusion_key = _fusion_key(base, _BF16_FUSIONS)
        if fusion_key is not None:
            bf16_groups.setdefault(fusion_key, []).append(meta)
            continue
        if _is_gemma_norm(name):
            record("gemma_norm_add", meta.nbytes)
            continue
        record("passthrough", 0)

    for members in fp8_groups.values():
        if len(members) not in {2, 3}:
            unsupported += 1
            continue
        weight_out = sum(m.nbytes for m in members)
        scale_out = sum(m.rows * 4 for m in members)
        record("fp8_fusion", weight_out + scale_out + 16)

    for members in bf16_groups.values():
        if len(members) != 2:
            unsupported += 1
            continue
        record("bf16_fusion", sum(m.nbytes for m in members))

    buffered_native_global_upper = 0
    for members in nvfp4_groups.values():
        if len(members) != 2:
            unsupported += 1
            continue
        weight_out = sum(m[0].nbytes for m in members)
        scale_out = sum(m[1].nbytes for m in members)
        globals_in = sum(m[0].rows * 2 for m in members)
        globals_out = globals_in
        buffered_native_global_upper += min(m[0].rows * 2 for m in members)
        record("nvfp4_gate_up_fusion", weight_out + scale_out + globals_in + globals_out)

    if unsupported:
        raise ValueError(f"dense envelope encountered {unsupported} unsupported/incomplete groups")

    # In adversarial shard/header order, the first half of every native gate/up pair may
    # retain its per-row global vector before the partner arrives. Add that small upper bound
    # to the largest local allocation event rather than assuming favorable iteration order.
    anonymous_peak = local_peak + buffered_native_global_upper

    return {
        "schema_version": 1,
        "model_family": "qwen3_5_moe_modelopt_mixed",
        "anonymous_peak_bytes": anonymous_peak,
        "largest_file_backed_source_tensor_bytes": largest_source_tensor,
        "writer_payload_copy_bytes": 0,
        "buffered_native_global_upper_bytes": buffered_native_global_upper,
        "operation_class_counts": dict(sorted(counts.items())),
        "max_anonymous_bytes_by_operation_class": dict(sorted(max_by_class.items())),
        "claim_boundary": {
            "proves": [
                "allocation-aware anonymous dense conversion envelope for supported mixed path",
                "largest file-backed source tensor from metadata",
                "known FTW writer path adds no payload-sized tensor copy for contiguous outputs",
            ],
            "does_not_prove": [
                "whole-process RSS peak",
                "file-page residency peak",
                "CUDA context or allocator overhead",
                "real checkpoint conversion success",
                "resource admission",
                "model load or inference",
            ],
        },
    }


__all__ = ["estimate_qwen35_mixed_dense_anonymous_peak"]
