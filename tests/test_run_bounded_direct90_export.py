from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import pytest

from scripts.run_bounded_direct90_export import _discover_lease, run


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path):
    exporter = tmp_path / "exporter.py"
    exporter.write_text(
        "REFERENCE_MANIFEST_SHA256='x'\n"
        "COMPONENT_ORDER=()\nSTEP_SHA256={}\n"
        "REQUIRED_THREAD_ENVIRONMENT={}\n"
        "def _validated_step_artifacts(value): return {}\n",
        encoding="utf-8",
    )
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("wrapper", encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("manifest", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    return (
        argparse.Namespace(
            attempt_id="attempt-v3",
            nonce="nonce-v3",
            lease_id="AUTO_DISCOVER",
            exporter=exporter,
            expected_exporter_sha256=_sha(exporter),
            expected_wrapper_sha256=_sha(wrapper),
            runtime_receipt=runtime,
            expected_runtime_receipt_sha256=_sha(runtime),
            source_root=source,
            output_root=tmp_path / "result",
            terminal_receipt=logs / "terminal.json",
            requested_threads=32,
            policy_total_threads=256,
            policy_limit_threads=64,
        ),
        wrapper,
    )


def _patch_preflight(monkeypatch, wrapper):
    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.Path.resolve",
        lambda self: (
            wrapper if self.name == "run_bounded_direct90_export.py" else self
        ),
    )
    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export._load_exporter",
        lambda path: type(
            "Exporter",
            (),
            {
                "REFERENCE_MANIFEST_SHA256": _sha(
                    path.parent / "source" / "manifest.json"
                ),
                "COMPONENT_ORDER": (),
                "REQUIRED_THREAD_ENVIRONMENT": {},
                "_validated_step_artifacts": staticmethod(lambda value: {}),
            },
        ),
    )
    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export._discover_lease",
        lambda: {
            "lease_id": "lease-v3",
            "cores": 32,
            "pid": 99,
            "start_time": 123,
            "session_id": 99,
        },
    )


def test_runner_persists_terminal_failure_receipt(tmp_path, monkeypatch):
    args, wrapper = _case(tmp_path)
    _patch_preflight(monkeypatch, wrapper)

    class Process:
        pid = 77

        def wait(self, timeout):
            return 7

        def poll(self):
            return 7

    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.subprocess.Popen",
        lambda command, start_new_session: Process(),
    )
    assert run(args) == 7
    value = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert value["return_code"] == 7
    assert value["terminal_status"] == "EXPORT_FAILED_NO_AUTOMATIC_SUCCESSOR"
    assert value["child_reaped"] is True


def test_runner_timeout_reaps_and_persists_receipt(tmp_path, monkeypatch):
    args, wrapper = _case(tmp_path)
    _patch_preflight(monkeypatch, wrapper)

    class Process:
        pid = 78

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("export", timeout)

        def poll(self):
            return None

    process = Process()
    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.subprocess.Popen",
        lambda command, start_new_session: process,
    )
    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export._terminate",
        lambda received: (
            setattr(received, "poll", lambda: 124)
            or {"actions": ["SIGTERM"], "errors": [], "child_reaped": True}
        ),
    )
    assert run(args) == 124
    value = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert value["terminal_status"] == (
        "EXPORT_TIMEOUT_REAPED_NO_AUTOMATIC_SUCCESSOR"
    )
    assert value["child_reaped"] is True


def test_terminate_is_bounded_and_reports_incomplete_reaping(monkeypatch):
    from scripts.run_bounded_direct90_export import _terminate

    class Process:
        pid = 80

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("export", timeout)

        def poll(self):
            return None

    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.os.killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
        raising=False,
    )
    result = _terminate(Process())
    assert result["child_reaped"] is False
    assert len(result["errors"]) == 3


def test_terminate_captures_oserror_from_final_wait(monkeypatch):
    from scripts.run_bounded_direct90_export import _terminate

    class Process:
        pid = 81

        def __init__(self):
            self.wait_count = 0

        def wait(self, timeout):
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired("export", timeout)
            raise OSError("wait failed")

        def poll(self):
            raise OSError("poll failed")

    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.os.killpg",
        lambda *_args: None,
        raising=False,
    )
    result = _terminate(Process())
    assert result["child_reaped"] is False
    assert any("final wait OSError" in error for error in result["errors"])
    assert any("poll OSError" in error for error in result["errors"])


def test_runner_records_uppercase_success_receipt(tmp_path, monkeypatch):
    args, wrapper = _case(tmp_path)
    _patch_preflight(monkeypatch, wrapper)
    args.output_root.mkdir()
    (args.output_root / "dagmc.h5m").write_bytes(b"h5m")
    (args.output_root / "DAGMC_EXPORT_RECEIPT.json").write_text(
        "{}", encoding="utf-8"
    )

    class Process:
        pid = 79

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.subprocess.Popen",
        lambda command, start_new_session: Process(),
    )
    assert run(args) == 0
    value = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert value["outputs"]["DAGMC_EXPORT_RECEIPT.json"]["present"] is True


def test_runner_rejects_existing_receipt_before_launch(tmp_path, monkeypatch):
    args, _ = _case(tmp_path)
    args.terminal_receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_bounded_direct90_export.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("must not launch"),
    )
    with pytest.raises(FileExistsError, match="create-only"):
        run(args)


def _write_proc_stat(proc_root, pid, start_time):
    path = proc_root / str(pid)
    path.mkdir(parents=True)
    fields = ["S"] + ["0"] * 18 + [str(start_time)]
    (path / "stat").write_text(
        f"{pid} (guarded command) " + " ".join(fields) + "\n",
        encoding="utf-8",
    )


def test_lease_discovery_matches_exact_live_session(tmp_path):
    leases = tmp_path / "leases"
    leases.mkdir()
    proc = tmp_path / "proc"
    _write_proc_stat(proc, 101, 201)
    (leases / "other.lease").write_text("4 100 200 300\n", encoding="utf-8")
    (leases / "exact.lease").write_text("32 101 201 301\n", encoding="utf-8")
    assert _discover_lease(
        leases, session_id=301, proc_root=proc, timeout_seconds=0
    ) == {
        "lease_id": "exact",
        "cores": 32,
        "pid": 101,
        "start_time": 201,
        "session_id": 301,
    }


def test_lease_discovery_rejects_ambiguous_session(tmp_path):
    leases = tmp_path / "leases"
    leases.mkdir()
    proc = tmp_path / "proc"
    _write_proc_stat(proc, 101, 201)
    _write_proc_stat(proc, 102, 202)
    (leases / "one.lease").write_text("32 101 201 301\n", encoding="utf-8")
    (leases / "two.lease").write_text("32 102 202 301\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="multiple live"):
        _discover_lease(
            leases, session_id=301, proc_root=proc, timeout_seconds=0
        )


@pytest.mark.parametrize(
    "lease_text",
    ["malformed\n", "1 101 201 301\n", "32 101 999 301\n"],
)
def test_lease_discovery_rejects_malformed_wrong_core_and_stale_pid(
    tmp_path, lease_text
):
    leases = tmp_path / "leases"
    leases.mkdir()
    proc = tmp_path / "proc"
    _write_proc_stat(proc, 101, 201)
    (leases / "candidate.lease").write_text(lease_text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="timed out"):
        _discover_lease(
            leases, session_id=301, proc_root=proc, timeout_seconds=0
        )


def test_runner_rejects_explicit_lease_bypass(tmp_path, monkeypatch):
    args, wrapper = _case(tmp_path)
    _patch_preflight(monkeypatch, wrapper)
    args.lease_id = "unverified-explicit-token"
    assert run(args) == 125
    value = json.loads(args.terminal_receipt.read_text(encoding="utf-8"))
    assert "explicit ssh-poly lease IDs are forbidden" in value["error"]
