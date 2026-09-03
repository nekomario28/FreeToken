"""Owner-native immutable identity for the opt-in file-backed FTW server.

This module is deliberately standard-library-only.  It hashes the exact local FTW directory
selected by the experimental server, derives one immutable model identity, and exposes a
read-only runtime document bound to the current frontend process generation.  It never loads
weights, runs inference, starts a service, performs network I/O, or grants execution authority.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from freetoken.checkpoint.mapped_ftw_core import (
    INDEX_NAME,
    _as_int,
    _safe_shard_path,
    load_ftw_index,
)

SCHEMA_VERSION = "freetoken-file-backed-ftw-runtime-identity/0.1"
IDENTITY_ROUTE = "/v1/runtime/identity"
PRODUCER_MODE = "freetoken.experimental.file_backed_ftw_cpu_server"
HASH_CHUNK = 8 * 1024 * 1024


def _sha256_file(path: Path, digest: Any) -> int:
    total = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(HASH_CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return total


def compute_ftw_artifact_identity(model_path: str | Path) -> dict[str, Any]:
    """Hash one validated local FTW checkpoint without loading model tensors.

    The logical payload digest is the concatenation of declared shard bytes ordered by
    ``global_off``.  This intentionally matches the project-side FTW load-admission identity
    algorithm so the two independent owners can be compared exactly later.
    """
    root = Path(model_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("file-backed FTW runtime identity requires a local checkpoint directory")

    index_path = root / INDEX_NAME
    index_raw = index_path.read_bytes()
    index = load_ftw_index(root)
    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("FTW checkpoint has no shards")

    rows: list[tuple[int, dict[str, Any]]] = []
    offsets: set[int] = set()
    paths: set[Path] = set()
    for pos, row in enumerate(shards):
        if not isinstance(row, dict):
            raise ValueError(f"FTW shard row {pos} is invalid")
        global_off = _as_int(row.get("global_off"), "FTW shard global_off")
        if global_off in offsets:
            raise ValueError("FTW shard global_off values must be unique")
        offsets.add(global_off)
        rows.append((global_off, row))

    payload = hashlib.sha256()
    total = 0
    for _, row in sorted(rows, key=lambda item: item[0]):
        shard_path = _safe_shard_path(root, row)
        if shard_path in paths:
            raise ValueError("FTW shard file is declared more than once")
        paths.add(shard_path)
        expected = _as_int(row.get("nbytes"), "FTW shard nbytes", minimum=1)
        observed = _sha256_file(shard_path, payload)
        if observed != expected:
            raise ValueError("FTW shard size changed or differs from the index")
        total += observed

    payload_sha256 = payload.hexdigest()
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    return {
        "payload_sha256": payload_sha256,
        "index_sha256": index_sha256,
        "payload_bytes_hashed": total,
        "shards_hashed": len(rows),
        "model_identity": (
            f"ftw-sha256:{payload_sha256}:index-sha256:{index_sha256}"
        ),
    }


def _validated_artifact_identity(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "payload_sha256",
        "index_sha256",
        "payload_bytes_hashed",
        "shards_hashed",
        "model_identity",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("runtime artifact identity has an unexpected shape")
    payload = value["payload_sha256"]
    index = value["index_sha256"]
    if not isinstance(payload, str) or len(payload) != 64:
        raise ValueError("payload_sha256 must be a 64-character digest")
    if not isinstance(index, str) or len(index) != 64:
        raise ValueError("index_sha256 must be a 64-character digest")
    try:
        int(payload, 16)
        int(index, 16)
    except ValueError as exc:
        raise ValueError("artifact digests must be lowercase hexadecimal") from exc
    if payload.lower() != payload or index.lower() != index:
        raise ValueError("artifact digests must be lowercase hexadecimal")
    for field in ("payload_bytes_hashed", "shards_hashed"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{field} must be a positive integer")
    expected_identity = f"ftw-sha256:{payload}:index-sha256:{index}"
    if value["model_identity"] != expected_identity:
        raise ValueError("model_identity does not match the artifact digests")
    return dict(value)


def build_runtime_identity_document(
    state: Any,
    artifact_identity: dict[str, Any],
    *,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Bind the sealed artifact identity to the current frontend server generation."""
    artifact = _validated_artifact_identity(artifact_identity)
    maintenance = str(getattr(state, "maintenance_state", "unknown"))
    config = getattr(state, "config", None)
    model_id = getattr(config, "served_model_name", None)
    instance_id = getattr(state, "instance_id", None)
    serving = (
        maintenance == "serving"
        and isinstance(model_id, str)
        and bool(model_id)
        and isinstance(instance_id, str)
        and bool(instance_id)
    )
    stamp = observed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OBSERVED" if serving else "NOT_SERVING",
        "serving": serving,
        "maintenance_state": maintenance,
        "observed_at": stamp,
        "instance_id": instance_id if isinstance(instance_id, str) and instance_id else None,
        "model_id": model_id if isinstance(model_id, str) and model_id else None,
        "model_identity": artifact["model_identity"],
        "artifact": {
            "payload_sha256": artifact["payload_sha256"],
            "index_sha256": artifact["index_sha256"],
            "payload_bytes_hashed": artifact["payload_bytes_hashed"],
            "shards_hashed": artifact["shards_hashed"],
        },
        "producer": {
            "mode": PRODUCER_MODE,
            "scope": "current_frontend_process_generation",
        },
        "authority": {
            "inference_invoked": False,
            "service_mutation_invoked": False,
            "model_download_invoked": False,
            "model_load_invoked": False,
            "process_kill_invoked": False,
            "authority_minted": False,
            "execution_authority_granted": False,
        },
        "claim_boundary": [
            "identity observation is not inference success",
            "identity observation is not candidate-evaluation authority",
            "identity observation is not semantic or task-quality truth",
            "identity observation is not promotion authority",
        ],
    }


def register_runtime_identity_route(
    app: Any,
    get_state: Callable[[], Any],
    artifact_identity: dict[str, Any],
) -> Any:
    """Register the experimental route and return its exact route object for later removal."""
    artifact = _validated_artifact_identity(artifact_identity)
    existing = [route for route in getattr(app, "routes", ()) if getattr(route, "path", None) == IDENTITY_ROUTE]
    if existing:
        raise RuntimeError(f"runtime identity route already registered: {IDENTITY_ROUTE}")
    before = {id(route) for route in getattr(app, "routes", ())}

    async def runtime_identity():
        return build_runtime_identity_document(get_state(), artifact)

    app.add_api_route(
        IDENTITY_ROUTE,
        runtime_identity,
        methods=["GET"],
        name="file_backed_ftw_runtime_identity",
    )
    created = [
        route
        for route in getattr(app, "routes", ())
        if id(route) not in before and getattr(route, "path", None) == IDENTITY_ROUTE
    ]
    if len(created) != 1:
        raise RuntimeError("failed to register exactly one runtime identity route")
    return created[0]


def unregister_runtime_identity_route(app: Any, route: Any) -> None:
    """Remove only the exact route object installed by this experiment."""
    routes = getattr(app, "routes", None)
    if routes is None:
        return
    for index, candidate in enumerate(list(routes)):
        if candidate is route:
            del routes[index]
            return


__all__ = [
    "IDENTITY_ROUTE",
    "PRODUCER_MODE",
    "SCHEMA_VERSION",
    "build_runtime_identity_document",
    "compute_ftw_artifact_identity",
    "register_runtime_identity_route",
    "unregister_runtime_identity_route",
]
