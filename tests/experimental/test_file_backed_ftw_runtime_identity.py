from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import freetoken.experimental.file_backed_ftw_runtime_identity as identity_mod
from freetoken.experimental.file_backed_ftw_cpu_server import _required_model_path
from freetoken.experimental.file_backed_ftw_runtime_identity import (
    IDENTITY_ROUTE,
    PRODUCER_MODE,
    SCHEMA_VERSION,
    SOURCE_IDENTITY_PATHS,
    SOURCE_IDENTITY_SCHEMA,
    SOURCE_IDENTITY_SCOPE,
    build_runtime_identity_document,
    compute_bounded_software_identity,
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


def git_blob_sha1(raw: bytes) -> str:
    header = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


class FileBackedFTWRuntimeIdentityTest(unittest.TestCase):
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

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

    def software(self):
        return compute_bounded_software_identity(self.repo_root())

    def state(self, maintenance="serving"):
        return SimpleNamespace(
            maintenance_state=maintenance,
            instance_id="instance-123",
            config=SimpleNamespace(served_model_name="unit-model"),
        )

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

    def test_bounded_source_identity_hashes_exact_declared_files(self):
        identity = self.software()
        self.assertEqual(identity["schema_version"], SOURCE_IDENTITY_SCHEMA)
        self.assertEqual(identity["scope"], SOURCE_IDENTITY_SCOPE)
        self.assertEqual(
            [row["path"] for row in identity["files"]],
            list(SOURCE_IDENTITY_PATHS),
        )
        self.assertEqual(len(identity["source_set_sha256"]), 64)
        for row in identity["files"]:
            raw = (self.repo_root() / row["path"]).read_bytes()
            self.assertEqual(row["bytes_hashed"], len(raw))
            self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(row["git_blob_sha1"], git_blob_sha1(raw))

    def test_bounded_source_identity_changes_when_any_declared_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, relative in enumerate(SOURCE_IDENTITY_PATHS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"source-{index}".encode("ascii"))
            before = compute_bounded_software_identity(root)
            target = root / SOURCE_IDENTITY_PATHS[1]
            target.write_bytes(target.read_bytes() + b"-changed")
            after = compute_bounded_software_identity(root)
        self.assertNotEqual(before["source_set_sha256"], after["source_set_sha256"])
        self.assertNotEqual(before["files"][1]["sha256"], after["files"][1]["sha256"])

    def test_bounded_source_identity_missing_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in SOURCE_IDENTITY_PATHS[:-1]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                compute_bounded_software_identity(root)

    def test_serving_document_binds_instance_model_artifact_and_bounded_source(self):
        software = self.software()
        doc = build_runtime_identity_document(
            self.state(),
            self.artifact(),
            software,
            observed_at="2026-09-04T00:00:00Z",
        )
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        self.assertEqual(doc["status"], "OBSERVED")
        self.assertTrue(doc["serving"])
        self.assertEqual(doc["instance_id"], "instance-123")
        self.assertEqual(doc["model_id"], "unit-model")
        self.assertEqual(doc["model_identity"], self.artifact()["model_identity"])
        self.assertEqual(doc["software"], software)
        self.assertEqual(doc["producer"]["mode"], PRODUCER_MODE)
        self.assertFalse(any(doc["authority"].values()))
        serialized = json.dumps(doc, sort_keys=True)
        self.assertNotIn(str(self.repo_root()), serialized)
        self.assertNotIn("first.ftw", serialized)
        self.assertNotIn("tensor", serialized.lower())

    def test_malformed_bounded_source_identity_fails_closed(self):
        for mutate in ("digest", "scope", "order", "extra"):
            with self.subTest(mutate=mutate):
                software = json.loads(json.dumps(self.software()))
                if mutate == "digest":
                    software["source_set_sha256"] = "0" * 64
                elif mutate == "scope":
                    software["scope"] = "whole-repository"
                elif mutate == "order":
                    software["files"][0], software["files"][1] = (
                        software["files"][1],
                        software["files"][0],
                    )
                else:
                    software["future"] = True
                with self.assertRaises(ValueError):
                    build_runtime_identity_document(
                        self.state(),
                        self.artifact(),
                        software,
                        observed_at="2026-09-04T00:00:00Z",
                    )

    def test_nonserving_lifecycle_never_claims_observed_binding(self):
        software = self.software()
        for maintenance in ("loading", "rebuilding", "failed", "stopping"):
            with self.subTest(maintenance=maintenance):
                doc = build_runtime_identity_document(
                    self.state(maintenance),
                    self.artifact(),
                    software,
                    observed_at="2026-09-04T00:00:00Z",
                )
                self.assertEqual(doc["status"], "NOT_SERVING")
                self.assertFalse(doc["serving"])

    def test_missing_runtime_instance_or_model_does_not_claim_serving(self):
        software = self.software()
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
                software,
                observed_at="2026-09-04T00:00:00Z",
            )
            self.assertEqual(doc["status"], "NOT_SERVING")
            self.assertFalse(doc["serving"])

    def test_route_registration_seals_bounded_source_once_and_is_removable(self):
        app = FakeApp()
        state = self.state()
        sealed = self.software()
        calls = []
        original = identity_mod.compute_bounded_software_identity

        def fake_compute():
            calls.append(True)
            return sealed

        identity_mod.compute_bounded_software_identity = fake_compute
        try:
            route = register_runtime_identity_route(app, lambda: state, self.artifact())
            self.assertEqual(calls, [True])
            self.assertEqual(route.path, IDENTITY_ROUTE)
            self.assertEqual(route.methods, ("GET",))
            first = asyncio.run(route.endpoint())
            second = asyncio.run(route.endpoint())
            self.assertEqual(calls, [True])
            self.assertEqual(first["software"], sealed)
            self.assertEqual(second["software"], sealed)
            self.assertEqual(first["instance_id"], "instance-123")
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                register_runtime_identity_route(app, lambda: state, self.artifact())
            unregister_runtime_identity_route(app, route)
            self.assertFalse(any(item.path == IDENTITY_ROUTE for item in app.routes))
        finally:
            identity_mod.compute_bounded_software_identity = original

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
        root = self.repo_root() / "python" / "freetoken" / "experimental"
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
