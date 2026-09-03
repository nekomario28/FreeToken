"""Owner-native immutable identity for the opt-in file-backed FTW server.

This module is deliberately standard-library-only. It hashes the exact local FTW directory
selected by the experimental server, seals a bounded identity of the source bytes that produce
this experimental identity surface, and exposes a read-only runtime document bound to the
current frontend process generation. It never loads weights, runs inference, starts a service,
performs network I/O, or grants execution authority.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "freetoken-file-backed-ftw-runtime-identity/0.2"
IDENTITY_ROUTE = "/v1/runtime/identity"
PRODUCER_MODE = "freetoken.experimental.file_backed_ftw_cpu_server"
HASH_CHUNK = 8 * 1024 * 1024
SOURCE_IDENTITY_SCHEMA = "freetoken-bounded-owner-source-set/0.1"
SOURCE_IDENTITY_SCOPE = "runtime_identity_producer_launcher_and_storage_core"
SOURCE_IDENTITY_PATHS = (
    "python/freetoken/experimental/file_backed_ftw_runtime_identity.py",
    "python/freetoken/experimental/file_backed_ftw_cpu_server.py",
    "python/freetoken/checkpoint/mapped_ftw_core.py",
)


def _load_mapped_ftw_core():
    """Load the torch-free storage core without executing checkpoint/__init__.py.

    Importing ``freetoken.checkpoint.mapped_ftw_core`` normally executes the checkpoint package
    initializer first, which imports torch. Runtime identity must remain usable before any model
    runtime dependency is imported, so load the already-existing torch-free source file directly.
    """
    path = Path(__file__).resolve().parents[1] / "checkpoint" / "mapped_ftw_core.py"
    name = "_freetoken_file_backed_identity_mapped_ftw_core"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load mapped FTW storage core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_STORAGE_CORE = _load_mapped_ftw_core()
INDEX_NAME = _STORAGE_CORE.INDEX_NAME
_as_int = _STORAGE_CORE._as_int
_safe_shard_path = _STORAGE_CORE._safe_shard_path
load_ftw_index = _STORAGE_CORE.load_ftw_index


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


def _lower_hex(value: Any, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be lowercase hexadecimal") from exc
    if value.lower() != value:
        raise ValueError(f"{field} must be lowercase hexadecimal")
    return value


def _source_set_digest(rows: list[dict[str, Any]]) -> str:
    """Digest one exact ordered source-set identity without serialisation ambiguity."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes_hashed"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(row["git_blob_sha1"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def compute_bounded_software_identity(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the exact bounded owner source bytes used by this evidence producer.

    This is a content-derived source-set revision, not a claim about the complete repository,
    dependency environment, wheel/container image, or serving engine. The Git-blob SHA-1 is an
    interoperability locator; SHA-256 remains the primary content identity.
    """
    root = (
        Path(repo_root).resolve(strict=True)
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if not root.is_dir():
        raise ValueError("bounded source identity requires a repository directory")

    rows: list[dict[str, Any]] = []
    for relative in SOURCE_IDENTITY_PATHS:
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("bounded source path escapes repository root") from exc
        if not path.is_file():
            raise ValueError(f"bounded source path is not a file: {relative}")
        raw = path.read_bytes()
        if not raw:
            raise ValueError(f"bounded source file is empty: {relative}")
        git_header = b"blob " + str(len(raw)).encode("ascii") + b"\0"
        rows.append(
            {
                "path": relative,
                "bytes_hashed": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob_sha1": hashlib.sha1(
                    git_header + raw,
                    usedforsecurity=False,
                ).hexdigest(),
            }
        )

    return {
        "schema_version": SOURCE_IDENTITY_SCHEMA,
        "scope": SOURCE_IDENTITY_SCOPE,
        "files": rows,
        "source_set_sha256": _source_set_digest(rows),
    }


def _validated_software_identity(value: dict[str, Any]) -> dict[str, Any]:
    expected = {"schema_version", "scope", "files", "source_set_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("bounded software identity has an unexpected shape")
    if value["schema_version"] != SOURCE_IDENTITY_SCHEMA:
        raise ValueError("bounded software identity schema mismatch")
    if value["scope"] != SOURCE_IDENTITY_SCOPE:
        raise ValueError("bounded software identity scope mismatch")
    files = value["files"]
    if not isinstance(files, list) or len(files) != len(SOURCE_IDENTITY_PATHS):
        raise ValueError("bounded software identity file set mismatch")

    rows: list[dict[str, Any]] = []
    for expected_path, row in zip(SOURCE_IDENTITY_PATHS, files, strict=True):
        fields = {"path", "bytes_hashed", "sha256", "git_blob_sha1"}
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("bounded software identity row has an unexpected shape")
        if row["path"] != expected_path:
            raise ValueError("bounded software identity path/order mismatch")
        count = row["bytes_hashed"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("bounded software identity bytes_hashed must be positive")
        sha256 = _lower_hex(row["sha256"], length=64, field="source sha256")
        git_blob_sha1 = _lower_hex(
            row["git_blob_sha1"],
            length=40,
            field="source git_blob_sha1",
        )
        rows.append(
            {
                "path": expected_path,
                "bytes_hashed": count,
                "sha256": sha256,
                "git_blob_sha1": git_blob_sha1,
            }
        )

    source_set_sha256 = _lower_hex(
        value["source_set_sha256"],
        length=64,
        field="source_set_sha256",
    )
    if source_set_sha256 != _source_set_digest(rows):
        raise ValueError("bounded software identity aggregate digest mismatch")
    return {
        "schema_version": SOURCE_IDENTITY_SCHEMA,
        "scope": SOURCE_IDENTITY_SCOPE,
        "files": rows,
        "source_set_sha256": source_set_sha256,
    }


def compute_ftw_artifact_identity(model_path: str | Path) -> dict[str, Any]:
    """Hash one validated local FTW checkpoint without loading model tensors.

    The logical payload digest is the concatenation of declared shard bytes ordered by
    ``global_off``. This intentionally matches the project-side FTW load-admission identity
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
    payload = _lower_hex(value["payload_sha256"], length=64, field="payload_sha256")
    index = _lower_hex(value["index_sha256"], length=64, field="index_sha256")
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
    software_identity: dict[str, Any],
    *,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Bind sealed artifact + bounded source identities to one frontend generation."""
    artifact = _validated_artifact_identity(artifact_identity)
    software = _validated_software_identity(software_identity)
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
        "software": software,
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
            "bounded source identity is not complete repository or runtime provenance",
        ],
    }


def register_runtime_identity_route(
    app: Any,
    get_state: Callable[[], Any],
    artifact_identity: dict[str, Any],
) -> Any:
    """Seal owner identities, register one GET route, and return the exact route object."""
    artifact = _validated_artifact_identity(artifact_identity)
    # Seal source bytes once before the server/backend starts. The endpoint never accepts a
    # caller-provided software revision and never reinterprets a CLI/environment revision claim.
    software = compute_bounded_software_identity()
    existing = [
        route
        for route in getattr(app, "routes", ())
        if getattr(route, "path", None) == IDENTITY_ROUTE
    ]
    if existing:
        raise RuntimeError(f"runtime identity route already registered: {IDENTITY_ROUTE}")
    before = {id(route) for route in getattr(app, "routes", ())}

    async def runtime_identity():
        return build_runtime_identity_document(get_state(), artifact, software)

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
    "SOURCE_IDENTITY_PATHS",
    "SOURCE_IDENTITY_SCHEMA",
    "SOURCE_IDENTITY_SCOPE",
    "build_runtime_identity_document",
    "compute_bounded_software_identity",
    "compute_ftw_artifact_identity",
    "register_runtime_identity_route",
    "unregister_runtime_identity_route",
]
