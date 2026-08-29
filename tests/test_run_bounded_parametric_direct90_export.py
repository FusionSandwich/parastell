from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_bounded_parametric_direct90_export as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "PARAMETRIC_GEOMETRY_MANIFEST.json"
    manifest.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    producer = tmp_path / "producer.py"
    producer.write_text("# producer\n", encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    lease_root = tmp_path / "leases"
    lease_root.mkdir()
    monkeypatch.setattr(runner, "EXPECTED_LEASE_ROOT", lease_root.resolve())
    monkeypatch.setattr(
        runner,
        "_discover_lease",
        lambda _root: {
            "lease_id": "lease-v1",
            "cores": 32,
            "pid": 77,
            "start_time": 88,
            "session_id": 77,
        },
    )
    for name, value in runner.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    wrapper = Path(runner.__file__).resolve()
    exporter = wrapper.parent / "export_parametric_direct90_dagmc.py"
    return argparse.Namespace(
        attempt_id="parametric-export-v1",
        nonce="nonce-v1",
        lease_root=lease_root,
        exporter=exporter,
        expected_exporter_sha256=_sha(exporter),
        expected_wrapper_sha256=_sha(wrapper),
        runtime_receipt=runtime,
        expected_runtime_receipt_sha256=_sha(runtime),
        source_root=source,
        expected_manifest_sha256=_sha(manifest),
        audit_path=audit,
        expected_audit_sha256=_sha(audit),
        audit_seal_path=seal,
        expected_audit_seal_sha256=_sha(seal),
        audit_producer_path=producer,
        expected_audit_producer_sha256=_sha(producer),
        output_root=tmp_path / "output",
        export_log=evidence / "export.log",
        terminal_receipt=evidence / "terminal.json",
        requested_threads=32,
        policy_total_physical_cores=256,
        policy_limit_cores=64,
    )


def _write_outputs(root: Path) -> None:
    root.mkdir()
    for name in runner.OUTPUT_NAMES:
        (root / name).write_bytes(name.encode())


def test_command_propagates_all_physical_audit_bindings(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    command = runner._command(args)
    assert command[1:3] == ["-m", "scripts.export_parametric_direct90_dagmc"]
    expected = {
        "--manifest-sha256": args.expected_manifest_sha256,
        "--audit-sha256": args.expected_audit_sha256,
        "--audit-seal-sha256": args.expected_audit_seal_sha256,
        "--audit-producer-sha256": args.expected_audit_producer_sha256,
        "--threads": "32",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value


def test_success_hashes_all_outputs_and_preserves_log(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    launched = {}

    class Process:
        pid = 101

        def wait(self, timeout):
            _write_outputs(args.output_root)
            return 0

        def poll(self):
            return 0

    def popen(command, **kwargs):
        launched["command"] = command
        launched.update(kwargs)
        return Process()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    assert runner.run(args) == 0
    receipt = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert (
        receipt["terminal_status"] == "EXPORT_COMPLETED_NATIVE_GATES_PENDING"
    )
    assert receipt["automatic_successor_launched"] is False
    assert all(row["present"] for row in receipt["outputs"].values())
    assert launched["cwd"] == Path(runner.__file__).resolve().parents[1]
    assert launched["start_new_session"] is True


@pytest.mark.parametrize(
    "field,expected_field",
    [
        ("expected_audit_sha256", "audit"),
        ("expected_audit_seal_sha256", "seal"),
        ("expected_audit_producer_sha256", "producer"),
        ("expected_manifest_sha256", "manifest"),
    ],
)
def test_hash_mismatch_rejects_before_launch(
    tmp_path, monkeypatch, field, expected_field
):
    args = _case(tmp_path, monkeypatch)
    setattr(args, field, "0" * 64)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not launch"),
    )
    assert runner.run(args) == 125
    receipt = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert expected_field in receipt["error"].lower()
    assert receipt["child_pid"] is None


def test_thread_environment_mismatch_rejects_before_launch(
    tmp_path, monkeypatch
):
    args = _case(tmp_path, monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "31")
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not launch"),
    )
    assert runner.run(args) == 125
    receipt = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert "thread environment mismatch" in receipt["error"]


def test_nonzero_exit_is_terminal_and_has_no_successor(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)

    class Process:
        pid = 102

        def wait(self, timeout):
            return 7

        def poll(self):
            return 7

    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *_args, **_kw: Process()
    )
    assert runner.run(args) == 7
    receipt = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == "EXPORT_FAILED_NO_AUTOMATIC_SUCCESSOR"
    assert receipt["automatic_successor_launched"] is False


def test_timeout_reaps_child_and_records_terminal_status(
    tmp_path, monkeypatch
):
    args = _case(tmp_path, monkeypatch)

    class Process:
        pid = 103

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("export", timeout)

        def poll(self):
            return None

    process = Process()
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *_args, **_kw: process
    )
    monkeypatch.setattr(
        runner,
        "_terminate",
        lambda child: (
            setattr(child, "poll", lambda: 124)
            or {"actions": ["SIGTERM"], "errors": [], "child_reaped": True}
        ),
    )
    assert runner.run(args) == 124
    receipt = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == (
        "EXPORT_TIMEOUT_REAPED_NO_AUTOMATIC_SUCCESSOR"
    )
    assert receipt["child_reaped"] is True


def _write_proc_stat(proc_root: Path, pid: int, start_time: int) -> None:
    path = proc_root / str(pid)
    path.mkdir(parents=True)
    fields = ["S"] + ["0"] * 18 + [str(start_time)]
    (path / "stat").write_text(
        f"{pid} (guarded command) " + " ".join(fields) + "\n",
        encoding="utf-8",
    )


def test_lease_discovery_requires_exact_live_32_core_session(tmp_path):
    leases = tmp_path / "leases"
    leases.mkdir()
    proc = tmp_path / "proc"
    _write_proc_stat(proc, 101, 201)
    (leases / "wrong.lease").write_text("4 101 201 301\n", encoding="utf-8")
    (leases / "exact.lease").write_text("32 101 201 301\n", encoding="utf-8")
    assert (
        runner._discover_lease(
            leases,
            guard_pid=101,
            session_id=301,
            proc_root=proc,
            timeout_seconds=0,
        )["lease_id"]
        == "exact"
    )


def test_existing_create_only_target_rejects_before_receipt(
    tmp_path, monkeypatch
):
    args = _case(tmp_path, monkeypatch)
    args.output_root.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        runner.run(args)
