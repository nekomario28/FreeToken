"""Safety wrapper for the opt-in Qwen3.5 native-NVFP4 low-memory FTW experiment.

The heavy conversion path intentionally reuses the canonical FTW converter. Only its
expert-bank provider is replaced, for one bounded call, by
``experimental.low_memory_ftw_converter``. This module owns the outer admission/publication
contract:

* CLI defaults to read-only preflight; ``--execute`` is explicit;
* preflight reads local ``config.json`` plus safetensors headers with stdlib only -- no
  FreeToken model/runtime import is needed before admission;
* execution re-parses the runtime config after admission and cross-checks dimensions;
* canonical conversion writes into a hidden sibling staging directory;
* the resulting canonical FTW index is validated as native NVFP4 with all expected layers;
* a machine-readable receipt is fsynced before publication;
* publication uses Linux ``renameat2(RENAME_NOREPLACE)`` so a concurrently-created output is
  never replaced;
* any failure removes staging and leaves the final output unpublished.

The normal ``ft checkpoint`` and normal serving paths remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .ftw_resource_gate import preflight_low_memory_nvfp4_conversion

_DEFAULT_SHARD_LIMIT = 8 << 30
_SUPPORTED_ARCH = "Qwen3_5MoeForConditionalGeneration"
_CONVERSION_TARGET = "cpu_file_backed_native_nvfp4"
_RECEIPT_NAME = "freetoken_low_memory_conversion_receipt.json"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _architectures_from_hf(hf_config: Any) -> list[str]:
    architectures = list(_get(hf_config, "architectures", None) or [])
    text = _get(hf_config, "text_config", None)
    if not architectures and text is not None:
        architectures = list(_get(text, "architectures", None) or [])
    return architectures


def _routed_expert_quant_from_hf(hf_config: Any) -> str:
    """Tiny read-only copy of the Qwen3.5 routed-expert quant detection contract."""
    quant = _get(hf_config, "quantization_config", None)
    if quant is None:
        return "none"
    algo = str(_get(quant, "quant_algo", None) or _get(quant, "quant_method", None) or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" in algo:
        layers = _get(quant, "quantized_layers", None) or {}
        if isinstance(layers, dict):
            for name, layer_spec in layers.items():
                if not (str(name).endswith(".mlp.experts") or ".mlp.experts." in str(name)):
                    continue
                layer_algo = str(_get(layer_spec or {}, "quant_algo", "")).lower()
                if "fp4" in layer_algo:
                    return "nvfp4"
                if "fp8" in layer_algo:
                    return "fp8"
    return "none"


def _preflight_config_from_hf(hf_config: Any):
    """Extract only fields required by the resource gate; do not import model/runtime code."""
    architectures = _architectures_from_hf(hf_config)
    if _SUPPORTED_ARCH not in architectures:
        raise ValueError(
            f"experimental low-memory converter supports {_SUPPORTED_ARCH!r} only, "
            f"checkpoint reports architectures={architectures!r}"
        )
    if _routed_expert_quant_from_hf(hf_config) != "nvfp4":
        raise ValueError("Qwen3.5 low-memory conversion requires routed experts in native NVFP4")

    text = _get(hf_config, "text_config", None) or hf_config
    num_experts = int(_get(text, "num_experts", 0) or 0)
    hidden_size = int(_get(text, "hidden_size", 0) or 0)
    moe_intermediate_size = int(_get(text, "moe_intermediate_size", 0) or 0)
    num_layers = int(_get(text, "num_hidden_layers", None) or _get(text, "num_layers", 0) or 0)
    if min(num_experts, hidden_size, moe_intermediate_size, num_layers) <= 0:
        raise ValueError(
            "Qwen3.5 preflight config is missing positive num_experts/hidden_size/"
            "moe_intermediate_size/num_hidden_layers"
        )
    return SimpleNamespace(
        architectures=architectures,
        is_moe=True,
        expert_quant="nvfp4",
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_moe_layers=num_layers,
        num_layers=num_layers,
    )


def _resolve_qwen35_preflight_config(model_path: str):
    """Read local config.json directly; resource preflight is intentionally runtime-free."""
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("preflight requires an existing local checkpoint directory")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"preflight requires config.json: {root}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint config.json is unreadable: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("checkpoint config.json must contain a JSON object")
    return _preflight_config_from_hf(config)


def _resolve_qwen35_runtime_config(model_path: str, preflight_config: Any):
    """Import the real Qwen parser only after the read-only resource gate passed."""
    from freetoken.utils import cached_load_hf_config
    from freetoken.models.qwen3_5_moe.config import parse_config

    model_config = parse_config(cached_load_hf_config(model_path))
    architectures = tuple(getattr(model_config, "architectures", ()) or ())
    if not architectures or architectures[0] != _SUPPORTED_ARCH:
        raise ValueError(
            "Qwen3.5 runtime parser disagrees with preflight architecture: "
            f"{architectures!r}"
        )
    if not getattr(model_config, "is_moe", False):
        raise ValueError("Qwen3.5 low-memory conversion requires an MoE checkpoint")
    if getattr(model_config, "expert_quant", None) != "nvfp4":
        raise ValueError(
            "Qwen3.5 runtime parser disagrees with preflight: expected expert_quant='nvfp4', got "
            f"{getattr(model_config, 'expert_quant', None)!r}"
        )
    for name in ("num_experts", "hidden_size", "moe_intermediate_size", "num_moe_layers"):
        before = int(getattr(preflight_config, name))
        after = int(getattr(model_config, name))
        if before != after:
            raise RuntimeError(
                f"Qwen3.5 config changed between preflight and runtime parse: {name} "
                f"{before} -> {after}"
            )
    return model_config


def _staging_dir(output: Path) -> Path:
    return output.parent / f".{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"


def _validate_canonical_index(index: Any, expected_layers: int) -> dict:
    if not isinstance(index, dict):
        raise RuntimeError("canonical FTW converter returned a non-object index")
    if index.get("quant_format") != "nvfp4":
        raise RuntimeError(
            "canonical low-memory conversion produced unexpected quant_format="
            f"{index.get('quant_format')!r}"
        )
    got_layers = int(index.get("expert_bank_num_layers") or 0)
    if got_layers != int(expected_layers):
        raise RuntimeError(
            f"canonical low-memory conversion wrote {got_layers} expert layers, "
            f"expected {expected_layers}"
        )
    counts = index.get("counts") or {}
    if int(counts.get("experts_bank", 0)) <= 0:
        raise RuntimeError("canonical low-memory conversion produced no expert-bank entries")
    return index


def _write_receipt(staging: Path, receipt: dict[str, Any]) -> None:
    path = staging / _RECEIPT_NAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "atomic no-replace publication requires Linux renameat2; refusing unsafe fallback"
        )
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    err = ctypes.get_errno()
    if err == errno.EEXIST:
        raise FileExistsError(err, "output appeared during conversion; refusing to replace it", str(destination))
    if err in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError(
            "atomic no-replace publication is unavailable on this host/filesystem; refusing unsafe fallback"
        ) from OSError(err, os.strerror(err), str(destination))
    raise OSError(err, os.strerror(err), str(destination))


def _atomic_staged_canonical_conversion(
    model_path: str,
    out_dir: str,
    *,
    model_config: Any,
    preflight_report: Any,
    convert_fn: Callable[..., dict],
    shard_limit: int,
    device: str,
) -> tuple[dict, dict[str, Any]]:
    """Run an admitted canonical conversion in a sibling staging dir, then publish it."""
    output = Path(out_dir)
    if output.exists():
        raise ValueError(
            f"execution requires a non-existing output path for atomic publish: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_dir(output)
    if staging.exists():
        raise RuntimeError(f"unexpected pre-existing staging directory: {staging}")

    published = False
    try:
        index = convert_fn(
            model_path,
            str(staging),
            shard_limit=shard_limit,
            device=device,
        )
        _validate_canonical_index(index, int(model_config.num_moe_layers))
        receipt = {
            "architecture": _SUPPORTED_ARCH,
            "conversion_target": _CONVERSION_TARGET,
            "model_path": os.path.abspath(model_path),
            "out_dir": os.path.abspath(out_dir),
            "device": device,
            "resources": preflight_report.as_dict(),
            "counts": index.get("counts"),
            "fingerprint": index.get("fingerprint"),
            "quant_format": index.get("quant_format"),
            "expert_bank_num_layers": index.get("expert_bank_num_layers"),
        }
        _write_receipt(staging, receipt)
        _rename_noreplace(staging, output)
        published = True
        return index, receipt
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def preflight_qwen35_native_nvfp4(model_path: str, out_dir: str) -> dict[str, Any]:
    """Read-only admission report. Never creates the output directory."""
    model_config = _resolve_qwen35_preflight_config(model_path)
    report = preflight_low_memory_nvfp4_conversion(model_path, out_dir, model_config)
    return {
        "mode": "preflight",
        "architecture": _SUPPORTED_ARCH,
        "conversion_target": _CONVERSION_TARGET,
        "model_path": os.path.abspath(model_path),
        "out_dir": os.path.abspath(out_dir),
        "resources": report.as_dict(),
    }


def execute_qwen35_native_nvfp4(
    model_path: str,
    out_dir: str,
    *,
    shard_limit: int = _DEFAULT_SHARD_LIMIT,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the admitted experiment. This is never the CLI default."""
    preflight_config = _resolve_qwen35_preflight_config(model_path)
    report = preflight_low_memory_nvfp4_conversion(model_path, out_dir, preflight_config)
    if Path(out_dir).exists():
        raise ValueError("--execute requires --out to not exist; choose a fresh output path")

    # Heavy/runtime imports start only after admission.
    import torch

    runtime_config = _resolve_qwen35_runtime_config(model_path, preflight_config)
    from .low_memory_ftw_converter import convert_checkpoint_low_memory_nvfp4

    def _convert(model: str, staging: str, *, shard_limit: int, device: str):
        return convert_checkpoint_low_memory_nvfp4(
            model,
            staging,
            dtype=torch.bfloat16,
            moe_backend="offload",
            shard_limit=shard_limit,
            device=device,
        )

    index, receipt = _atomic_staged_canonical_conversion(
        model_path,
        out_dir,
        model_config=runtime_config,
        preflight_report=report,
        convert_fn=_convert,
        shard_limit=shard_limit,
        device=device,
    )
    return {
        "mode": "executed",
        "architecture": _SUPPORTED_ARCH,
        "conversion_target": _CONVERSION_TARGET,
        "model_path": os.path.abspath(model_path),
        "out_dir": os.path.abspath(out_dir),
        "resources": report.as_dict(),
        "counts": index.get("counts"),
        "fingerprint": index.get("fingerprint"),
        "receipt": os.path.join(os.path.abspath(out_dir), _RECEIPT_NAME),
        "device": receipt["device"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m freetoken.experimental.low_memory_ftw_cpu_convert",
        description=(
            "Preflight (default) or explicitly execute the Qwen3.5 native-NVFP4 "
            "low-memory CPU/file-backed FTW conversion experiment."
        ),
    )
    parser.add_argument("--model", required=True, help="local Qwen3.5-MoE checkpoint directory")
    parser.add_argument("--out", required=True, help="new FTW output path")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually convert; without this flag the command is read-only preflight",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="torch device used by the canonical converter during --execute (default: cuda:0)",
    )
    parser.add_argument(
        "--shard-limit-gib",
        type=float,
        default=8.0,
        help="FTW shard ceiling in GiB during --execute (default: 8)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.shard_limit_gib <= 0:
        raise SystemExit("--shard-limit-gib must be > 0")
    shard_limit = int(args.shard_limit_gib * (1 << 30))
    shard_limit = (shard_limit // 4096) * 4096
    if shard_limit <= 0:
        raise SystemExit("--shard-limit-gib is too small after 4096-byte alignment")

    if args.execute:
        result = execute_qwen35_native_nvfp4(
            args.model,
            args.out,
            shard_limit=shard_limit,
            device=args.device,
        )
    else:
        result = preflight_qwen35_native_nvfp4(args.model, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "execute_qwen35_native_nvfp4",
    "main",
    "preflight_qwen35_native_nvfp4",
]
