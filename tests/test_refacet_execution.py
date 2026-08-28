import hashlib
import json
import os

import pytest

from parastell.refacet_execution import (
    COLLECTION_RECEIPT_SCHEMA,
    OPERATIONAL_SEAL_SCHEMA,
    PUBLICATION_RECEIPT_SCHEMA,
    STAGE_RECEIPT_SCHEMA,
    StageReceiptChain,
    atomic_create_json,
    collect_terminal_evidence,
    publish_artifact_set,
    seal_operational_layout,
    sha256_file,
    validate_operational_layout,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def test_atomic_json_is_create_only_and_valid(tmp_path):
    output = tmp_path / "receipt.json"
    receipt = atomic_create_json(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    assert receipt["sha256"] == sha256_file(output)
    with pytest.raises(FileExistsError):
        atomic_create_json(output, {"value": 2})


def test_atomic_json_rejects_dangling_symlink(tmp_path):
    if os.name == "nt":
        pytest.skip("ordinary Windows test users may not create symlinks")
    output = tmp_path / "receipt.json"
    output.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        atomic_create_json(output, {"value": 1})


def test_stage_receipts_form_hash_chain(tmp_path):
    chain = StageReceiptChain(
        tmp_path / "stages",
        "attempt-001",
        "a" * 64,
        "IMPORT_FRAGMENT_DIAGNOSTIC",
        {"control_sha256": "b" * 64},
        pid=123,
        hostname="test-host",
    )
    first = chain.emit("SOURCE_STAGE", "STARTED")
    second = chain.emit("SOURCE_STAGE", "COMPLETED", details={"count": 8})
    assert first["schema"] == STAGE_RECEIPT_SCHEMA
    assert first["previous_receipt_sha256"] is None
    assert second["previous_receipt_sha256"] == first["receipt"]["sha256"]
    assert chain.head_sha256 == second["receipt"]["sha256"]
    with pytest.raises(FileExistsError):
        StageReceiptChain(
            tmp_path / "stages",
            "attempt-001",
            "a" * 64,
            "IMPORT_FRAGMENT_DIAGNOSTIC",
            {},
        )


def test_operational_layout_rejects_changes_and_unexpected_files(tmp_path):
    root = tmp_path / "operational"
    immutable = root / "immutable" / "control.json"
    _write(immutable, "{}")
    seal_path = root / "immutable" / "seal.json"
    seal = seal_operational_layout(
        root,
        immutable_paths=["immutable/control.json"],
        writable_paths=["evidence/run.log", "evidence/terminal.json"],
        receipt_path=seal_path,
        attempt_id="attempt-001",
        launch_nonce="a" * 64,
    )
    assert seal["schema"] == OPERATIONAL_SEAL_SCHEMA
    assert all(validate_operational_layout(root, seal).values())
    _write(root / "unexpected.txt", "bad")
    assert not validate_operational_layout(root, seal)["no_unexpected_paths"]
    (root / "unexpected.txt").unlink()
    immutable.write_text("changed", encoding="utf-8")
    assert not validate_operational_layout(root, seal)["immutable"]


def test_publication_is_ordered_create_only_and_hash_bound(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    hashes = {
        "dagmc.h5m": _write(scratch / "dagmc.h5m", "mesh"),
        "candidate_build_manifest.json": _write(
            scratch / "candidate_build_manifest.json", "{}"
        ),
        "refaceting_build_seal.json": _write(
            scratch / "refaceting_build_seal.json", "{}"
        ),
    }
    operational = tmp_path / "operational"
    operational.mkdir()
    receipt_path = operational / "publication.json"
    result = publish_artifact_set(
        scratch,
        tmp_path / "publish",
        ordinary_artifacts=["dagmc.h5m"],
        manifest_name="candidate_build_manifest.json",
        seal_name="refaceting_build_seal.json",
        expected_hashes=hashes,
        receipt_path=receipt_path,
        attempt_id="attempt-001",
        launch_nonce="a" * 64,
        stage_chain_sha256="b" * 64,
    )
    assert result["schema"] == PUBLICATION_RECEIPT_SCHEMA
    assert result["publication_order"][-2:] == [
        "candidate_build_manifest.json",
        "refaceting_build_seal.json",
    ]
    with pytest.raises(FileExistsError):
        publish_artifact_set(
            scratch,
            tmp_path / "publish",
            ordinary_artifacts=["dagmc.h5m"],
            manifest_name="candidate_build_manifest.json",
            seal_name="refaceting_build_seal.json",
            expected_hashes=hashes,
            receipt_path=operational / "other.json",
            attempt_id="attempt-001",
            launch_nonce="a" * 64,
            stage_chain_sha256="b" * 64,
        )


def test_collection_rehashes_terminal_evidence(tmp_path):
    root = tmp_path / "operational"
    root.mkdir()
    _write(root / "run.log", "complete\n")
    evidence = {
        "run.log": {
            "size_bytes": (root / "run.log").stat().st_size,
            "sha256": sha256_file(root / "run.log"),
        }
    }
    terminal = root / "terminal.json"
    terminal.write_text(
        json.dumps({"schema": "terminal/v1", "evidence": evidence}),
        encoding="utf-8",
    )
    result = collect_terminal_evidence(
        root,
        terminal_receipt_path=terminal,
        evidence_paths=["run.log"],
        output_path=root / "collection.json",
    )
    assert result["schema"] == COLLECTION_RECEIPT_SCHEMA
    assert result["pass"]


def test_collection_detects_changed_evidence(tmp_path):
    root = tmp_path / "operational"
    root.mkdir()
    _write(root / "run.log", "before\n")
    terminal = root / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "schema": "terminal/v1",
                "evidence": {
                    "run.log": {
                        "size_bytes": (root / "run.log").stat().st_size,
                        "sha256": sha256_file(root / "run.log"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "run.log").write_text("after\n", encoding="utf-8")
    result = collect_terminal_evidence(
        root,
        terminal_receipt_path=terminal,
        evidence_paths=["run.log"],
        output_path=root / "collection.json",
    )
    assert not result["pass"]
