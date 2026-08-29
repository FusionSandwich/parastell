from pathlib import Path
from types import SimpleNamespace

from scripts import run_dagmc_native_qualification as runner


def test_short_option_detection_accepts_supported_native_variant(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Options:\n  -p <points>\n  -t <threads>\n",
        ),
    )

    assert runner._supports_short_option("overlap_check", "-t") is True


def test_short_option_detection_rejects_unsupported_native_variant(
    monkeypatch,
):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Options:\n  -p <points>\n",
        ),
    )

    assert runner._supports_short_option("overlap_check", "-t") is False


def test_overlap_command_omits_unsupported_thread_option():
    command = runner._overlap_command(
        "/native/overlap_check",
        precision=4,
        threads=4,
        dagmc_path=Path("candidate.h5m"),
        thread_option_supported=False,
    )

    assert command == [
        "/native/overlap_check",
        "-p",
        "4",
        "candidate.h5m",
    ]


def test_overlap_command_preserves_supported_thread_option():
    command = runner._overlap_command(
        "/native/overlap_check",
        precision=2,
        threads=4,
        dagmc_path=Path("candidate.h5m"),
        thread_option_supported=True,
    )

    assert command == [
        "/native/overlap_check",
        "-p",
        "2",
        "-t",
        "4",
        "candidate.h5m",
    ]
