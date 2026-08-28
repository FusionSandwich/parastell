"""Fail-closed execution evidence for source-CAD refaceting.

This module intentionally has no CAD, Gmsh, MOAB, or OpenMC imports.  Its
receipt and publication primitives can therefore be tested in a minimal
environment before a geometry runtime is allowed to use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
from typing import Any, Mapping, Sequence


STAGE_RECEIPT_SCHEMA = "parastell.refacet_stage_receipt/v1.0.0"
OPERATIONAL_SEAL_SCHEMA = "parastell.refacet_operational_seal/v1.0.0"
PUBLICATION_RECEIPT_SCHEMA = "parastell.refacet_publication/v1.0.0"
COLLECTION_RECEIPT_SCHEMA = "parastell.refacet_collection/v1.0.0"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular, non-symlink file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON value with deterministic compact serialization."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialized_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _safe_token(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    """Atomically publish bytes without overwriting or following symlinks."""
    path = Path(path)
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"invalid publication parent: {parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"create-only path already exists: {path}")
    temporary = parent / f".{path.name}.{os.getpid()}.tmp"
    if os.path.lexists(temporary):
        raise FileExistsError(
            f"create-only temporary path exists: {temporary}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"create-only path appeared during publication: {path}"
            ) from None
        _fsync_directory(parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    actual = path.read_bytes()
    if actual != payload:
        raise RuntimeError(f"published bytes failed readback: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": len(actual),
        "sha256": hashlib.sha256(actual).hexdigest(),
    }


def atomic_create_json(
    path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically publish one create-only JSON object."""
    if not isinstance(payload, Mapping):
        raise TypeError("JSON receipt payload must be a mapping")
    return atomic_create_bytes(Path(path), _serialized_json(dict(payload)))


def snapshot_files(
    root: Path, relative_paths: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Hash an exact ordered inventory below *root*."""
    root = Path(root).resolve()
    rows: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"path escapes inventory root: {relative}"
            ) from error
        if relative in rows:
            raise ValueError(f"duplicate inventory path: {relative}")
        rows[relative] = {
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    return rows


def seal_operational_layout(
    operational_root: Path,
    *,
    immutable_paths: Sequence[str],
    writable_paths: Sequence[str],
    receipt_path: Path,
    attempt_id: str,
    launch_nonce: str,
) -> dict[str, Any]:
    """Seal immutable inputs and the exact writable-output allowlist."""
    operational_root = Path(operational_root).resolve()
    if operational_root.is_symlink() or not operational_root.is_dir():
        raise ValueError("operational root must be a non-symlink directory")
    immutable = snapshot_files(operational_root, immutable_paths)
    immutable_set = set(immutable)
    writable_set = set(writable_paths)
    if len(writable_set) != len(writable_paths):
        raise ValueError("writable allowlist contains duplicates")
    if immutable_set & writable_set:
        raise ValueError("immutable and writable paths overlap")
    for relative in writable_set:
        _safe_token(Path(relative).name, label="writable path name")
        target = operational_root / relative
        if os.path.lexists(target):
            raise FileExistsError(f"writable output already exists: {target}")
    payload = {
        "schema": OPERATIONAL_SEAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": _safe_token(attempt_id, label="attempt ID"),
        "launch_nonce": _safe_token(launch_nonce, label="launch nonce"),
        "operational_root": str(operational_root),
        "seal_path": Path(receipt_path)
        .resolve()
        .relative_to(operational_root)
        .as_posix(),
        "immutable_inventory": immutable,
        "immutable_inventory_sha256": canonical_json_sha256(immutable),
        "writable_allowlist": sorted(writable_set),
    }
    atomic_create_json(receipt_path, payload)
    return payload


def validate_operational_layout(
    operational_root: Path,
    seal: Mapping[str, Any],
    *,
    allowed_existing_outputs: Sequence[str] = (),
) -> dict[str, bool]:
    """Reject immutable changes and unexpected operational-root additions."""
    operational_root = Path(operational_root).resolve()
    immutable = seal.get("immutable_inventory", {})
    if not isinstance(immutable, Mapping):
        raise ValueError("operational seal lacks immutable inventory")
    current = snapshot_files(operational_root, list(immutable))
    allowed = set(seal.get("writable_allowlist", []))
    existing_allowed = set(allowed_existing_outputs)
    if not existing_allowed <= allowed:
        raise ValueError("existing outputs are not within writable allowlist")
    seal_relative = seal.get("seal_path")
    if not isinstance(seal_relative, str):
        raise ValueError("operational seal lacks its relative seal path")
    seal_path = operational_root / seal_relative
    known = set(immutable) | existing_allowed | {seal_relative}
    actual = {
        path.relative_to(operational_root).as_posix()
        for path in operational_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    gates = {
        "schema": seal.get("schema") == OPERATIONAL_SEAL_SCHEMA,
        "root": seal.get("operational_root") == str(operational_root),
        "immutable": current == immutable,
        "inventory_hash": seal.get("immutable_inventory_sha256")
        == canonical_json_sha256(immutable),
        "seal_file": seal_path.is_file() and not seal_path.is_symlink(),
        "no_unexpected_paths": actual <= known,
        "allowed_outputs_exist": all(
            (operational_root / relative).is_file()
            and not (operational_root / relative).is_symlink()
            for relative in existing_allowed
        ),
    }
    return gates


@dataclass
class StageReceiptChain:
    """Write an ordered hash chain of exclusive-create stage receipts."""

    directory: Path
    attempt_id: str
    launch_nonce: str
    run_mode: str
    bindings: Mapping[str, Any]
    pid: int = field(default_factory=os.getpid)
    hostname: str = field(default_factory=socket.gethostname)
    _ordinal: int = field(default=0, init=False)
    _previous_sha256: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if os.path.lexists(self.directory):
            raise FileExistsError(
                f"stage receipt directory already exists: {self.directory}"
            )
        self.directory.mkdir(mode=0o700, parents=False)
        self.attempt_id = _safe_token(self.attempt_id, label="attempt ID")
        self.launch_nonce = _safe_token(
            self.launch_nonce, label="launch nonce"
        )
        self.run_mode = _safe_token(self.run_mode, label="run mode")
        if canonical_json_sha256(self.bindings) == "":
            raise AssertionError("unreachable empty binding hash")

    @property
    def head_sha256(self) -> str | None:
        return self._previous_sha256

    def emit(
        self,
        stage: str,
        state: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = _safe_token(stage, label="stage")
        state = _safe_token(state, label="stage state")
        if state not in {"STARTED", "COMPLETED", "FAILED"}:
            raise ValueError(f"unsupported stage state: {state}")
        self._ordinal += 1
        payload = {
            "schema": STAGE_RECEIPT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempt_id": self.attempt_id,
            "launch_nonce": self.launch_nonce,
            "run_mode": self.run_mode,
            "ordinal": self._ordinal,
            "stage": stage,
            "state": state,
            "pid": self.pid,
            "hostname": self.hostname,
            "bindings": dict(self.bindings),
            "bindings_sha256": canonical_json_sha256(self.bindings),
            "previous_receipt_sha256": self._previous_sha256,
            "details": dict(details or {}),
        }
        filename = f"{self._ordinal:04d}_{stage.lower()}_{state.lower()}.json"
        published = atomic_create_json(self.directory / filename, payload)
        payload["receipt"] = published
        self._previous_sha256 = published["sha256"]
        return payload


def copy_file_exclusive(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Copy one file without overwrite and verify source/destination hashes."""
    source = Path(source)
    destination = Path(destination)
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"source hash mismatch: {source}")
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"invalid destination parent: {parent}")
    if os.path.lexists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    temporary = parent / f".{destination.name}.{os.getpid()}.copy"
    if os.path.lexists(temporary):
        raise FileExistsError(f"copy temporary already exists: {temporary}")
    try:
        with source.open("rb") as input_stream:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as output_stream:
                shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise RuntimeError(f"copied temporary hash mismatch: {temporary}")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"destination appeared during copy: {destination}"
            ) from None
        _fsync_directory(parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RuntimeError(f"published copy hash mismatch: {destination}")
    return {
        "path": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": actual,
    }


def publish_artifact_set(
    scratch_root: Path,
    publish_root: Path,
    *,
    ordinary_artifacts: Sequence[str],
    manifest_name: str,
    seal_name: str,
    expected_hashes: Mapping[str, str],
    receipt_path: Path,
    attempt_id: str,
    launch_nonce: str,
    stage_chain_sha256: str,
) -> dict[str, Any]:
    """Publish artifacts create-only, with manifest then seal last."""
    scratch_root = Path(scratch_root).resolve()
    publish_root = Path(publish_root)
    if os.path.lexists(publish_root):
        raise FileExistsError(f"publish root already exists: {publish_root}")
    publish_root.mkdir(mode=0o700, parents=False)
    order = [*ordinary_artifacts, manifest_name, seal_name]
    if len(order) != len(set(order)):
        raise ValueError("publication inventory contains duplicates")
    if set(order) != set(expected_hashes):
        raise ValueError("publication hashes do not match exact inventory")
    published: dict[str, dict[str, Any]] = {}
    for relative in order:
        published[relative] = copy_file_exclusive(
            scratch_root / relative,
            publish_root / relative,
            expected_sha256=expected_hashes[relative],
        )
    payload = {
        "schema": PUBLICATION_RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": _safe_token(attempt_id, label="attempt ID"),
        "launch_nonce": _safe_token(launch_nonce, label="launch nonce"),
        "scratch_root": str(scratch_root),
        "publish_root": str(publish_root.resolve()),
        "publication_order": order,
        "artifacts": published,
        "stage_chain_sha256": stage_chain_sha256,
        "manifest_published_before_seal": order[-2:]
        == [
            manifest_name,
            seal_name,
        ],
    }
    atomic_create_json(receipt_path, payload)
    return payload


def collect_terminal_evidence(
    operational_root: Path,
    *,
    terminal_receipt_path: Path,
    evidence_paths: Sequence[str],
    output_path: Path,
) -> dict[str, Any]:
    """Rehash evidence after exit and compare it with terminal declarations."""
    operational_root = Path(operational_root).resolve()
    terminal_receipt_path = Path(terminal_receipt_path)
    terminal = json.loads(terminal_receipt_path.read_text(encoding="utf-8"))
    actual = snapshot_files(operational_root, evidence_paths)
    declared = terminal.get("evidence", {})
    gates = {
        "terminal_schema": isinstance(terminal.get("schema"), str),
        "terminal_receipt_non_symlink": not terminal_receipt_path.is_symlink(),
        "exact_evidence_inventory": set(actual) == set(declared),
        "evidence_hashes_and_sizes": actual == declared,
    }
    payload = {
        "schema": COLLECTION_RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operational_root": str(operational_root),
        "terminal_receipt": {
            "path": str(terminal_receipt_path.resolve()),
            "sha256": sha256_file(terminal_receipt_path),
        },
        "evidence": actual,
        "gates": gates,
        "pass": all(gates.values()),
    }
    atomic_create_json(output_path, payload)
    return payload
