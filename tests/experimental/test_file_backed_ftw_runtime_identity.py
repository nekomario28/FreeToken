from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from freetoken.experimental.file_backed_ftw_cpu_server import _required_model_path
from freetoken.experimental.file_backed_ftw_runtime_identity import (
    IDENTITY_ROUTE,
    PRODUCER_MODE,
    SCHEMA_VERSION,
    build_runtime_identity_document,
    compute_ftw_artifact_identity,
    register_runtime_identity_route,
    unregister_runtime_identity_route,
)


class FakeRoute:
    def __init__(self, path, endpoint, methods, name):
        self.path = path
        self.endpoint = endpoint
        self.methods = methods
        self.name = name


class FakeApp:
    def __init__(self):
        self.routes = []

    def add_api_route(self, path, endpoint, *, methods, name):
        self.routes.append(FakeRoute(path, endpoint, tuple(methods), name))


def write_checkpoint(root: Path, *, escape: bool = False, wrong_size: bool = False) -> bytes:
    first = b"AAA"
    second = b"BB"
    (root / "first.ftw").write_bytes(first)
    (root / "second.ftw").write_bytes(second)
    shard_name = "../outside.ftw" if escape else "first.ftw"
    if escape:
        (root.parent / "outside.ftw").write_bytes(first)
    index = {
        "format": "freetoken_weight",
        "version": 1,
        "tensors": [],
        # Intentionally reverse logical order to prove global_off controls the digest.
        "shards": [
            {"file": "second.ftw", "global_off": 3, "nbytes": len(second)},
            {
                "file": shard_name,
                "global_off": 0,
                "nbytes": len(first) + (1 if wrong_size else 0),
            },
        ],
    }
    raw = (json.dumps(index, separators=(",", ":"), sort_keys=False) + "\n").encode()
    (root / "freetoken_weight.json").write_bytes(raw)
    return raw


class FileBackedFTWRuntimeIdentityTest(unittest.TestCase):
    def test_hashes_payload_in_global_offset_order_and_raw_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            raw = write_checkpoint(root)
            identity = compute_ftw_artifact_identity(root)

        payload = hashlib.sha256(b"AAA" + b"BB").hexdigest()
        index = hashlib.sha256(raw).hexdigest()
        self.assertEqual(identity["payload_sha256"], payload)
        self.assertEqual(identity["index_sha256"], index)
        self.assertEqual(identity["payload_bytes_hashed"], 5)
        self.assertEqual(identity["shards_hashed"], 2)
        self.assertEqual(
            identity["model_identity"],
            f"ftw-sha256:{payload}:index-sha256:{index}",
        )

    def test_declared_shard_size_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            write_checkpoint(root, wrong_size=True)
            with self.assertRaisesRegex(ValueError, "size"):
                compute_ftw_artifact_identity(root)

    def test_shard_path_escape_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            write_checkpoint(root, escape=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                compute_ftw_artifact_identity(root)

    def test_invalid_ftw_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            (root / "freetoken_weight.json").write_text(
                json.dumps({"format": "not-ftw", "tensors": [], "shards": []}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                compute_ftw_artifact_identity(root)

    def test_duplicate_global_offset_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            (root / "a.ftw").write_bytes(b"a")
            (root / "b.ftw").write_bytes(b"b")
            index = {
                "format": "freetoken_weight",
                "tensors": [],
                "shards": [
                    {"file": "a.ftw", "global_off": 0, "nbytes": 1},
                    {"file": "b.ftw", "global_off": 0, "nbytes": 1},
                ],
            }
            (root / "freetoken_weight.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "global_off"):
                compute_ftw_artifact_identity(root)

    def artifact(self):
        payload = "a" * 64
        index = "b" * 64
        return {
            "payload_sha256": payload,
            "index_sha256": index,
            "payload_bytes_hashed": 123,
            "shards_hashed": 2,
            "model_identity": f"ftw-sha256:{payload}:index-sha256:{index}",
        }

    def state(self, maintenance="serving"):
        return SimpleNamespace(
            maintenance_state=maintenance,
            instance_id="instance-123",
            config=SimpleNamespace(served_model_name="unit-model"),
        )

    def test_serving_document_binds_current_instance_model_and_artifact(self):
        doc = build_runtime_identity_document(
            self.state(),
            self.artifact(),
            observed_at="2026-09-04T00:00:00Z",
        )
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        self.assertEqual(doc["status"], "OBSERVED")
        self.assertTrue(doc["serving"])
        self.assertEqual(doc["instance_id"], "instance-123")
        self.assertEqual(doc["model_id"], "unit-model")
        self.assertEqual(doc["model_identity"], self.artifact()["model_identity"])
        self.assertEqual(doc["producer"]["mode"], PRODUCER_MODE)
        self.assertFalse(any(doc["authority"].values()))
        serialized = json.dumps(doc, sort_keys=True)
        self.assertNotIn("/tmp/", serialized)
        self.assertNotIn("first.ftw", serialized)
        self.assertNotIn("tensor", serialized.lower())

    def test_nonserving_lifecycle_never_claims_observed_binding(self):
        for maintenance in ("loading", "rebuilding", "failed", "stopping"):
            with self.subTest(maintenance=maintenance):
                doc = build_runtime_identity_document(
                    self.state(maintenance),
                    self.artifact(),
                    observed_at="2026-09-04T00:00:00Z",
                )
                self.assertEqual(doc["status"], "NOT_SERVING")
                self.assertFalse(doc["serving"])

    def test_missing_runtime_instance_or_model_does_not_claim_serving(self):
        for state in (
            SimpleNamespace(
                maintenance_state="serving",
                instance_id=None,
                config=SimpleNamespace(served_model_name="unit-model"),
            ),
            SimpleNamespace(
                maintenance_state="serving",
                instance_id="instance-1",
                config=SimpleNamespace(served_model_name=None),
            ),
        ):
            doc = build_runtime_identity_document(
                state,
                self.artifact(),
                observed_at="2026-09-04T00:00:00Z",
            )
            self.assertEqual(doc["status"], "NOT_SERVING")
            self.assertFalse(doc["serving"])

    def test_route_registration_is_explicit_and_removable(self):
        app = FakeApp()
        state = self.state()
        route = register_runtime_identity_route(app, lambda: state, self.artifact())
        self.assertEqual(route.path, IDENTITY_ROUTE)
        self.assertEqual(route.methods, ("GET",))
        doc = asyncio.run(route.endpoint())
        self.assertEqual(doc["instance_id"], "instance-123")
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            register_runtime_identity_route(app, lambda: state, self.artifact())
        unregister_runtime_identity_route(app, route)
        self.assertFalse(any(item.path == IDENTITY_ROUTE for item in app.routes))

    def test_model_path_aliases_must_bind_one_literal_value(self):
        prog = "ftw-server"
        self.assertEqual(
            _required_model_path(["--model", "/model"], prog=prog),
            "/model",
        )
        self.assertEqual(
            _required_model_path(
                ["--model=/model", "--model-path", "/model"],
                prog=prog,
            ),
            "/model",
        )
        with self.assertRaises(SystemExit):
            _required_model_path([], prog=prog)
        with self.assertRaises(SystemExit):
            _required_model_path(
                ["--model", "/model-a", "--model-path=/model-b"],
                prog=prog,
            )

    def test_identity_helper_and_launcher_do_not_import_effect_adapters(self):
        root = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "experimental"
        prohibited = {"subprocess", "socket", "requests", "httpx", "urllib", "http"}
        for name in (
            "file_backed_ftw_runtime_identity.py",
            "file_backed_ftw_cpu_server.py",
        ):
            tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(imported & prohibited, f"{name}: prohibited imports {imported & prohibited}")


if __name__ == "__main__":
    unittest.main()
