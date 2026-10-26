from __future__ import annotations

import os
from pathlib import Path

import pytest

MARKERS = {"cpu": "cpu", "integration": "integration", "cuda": "cuda"}
# The only CUDA skip allowed in a release run: tests that need hardware beyond one GPU.
ALLOWED_RELEASE_SKIPS = ("needs two CUDA devices",)
_cuda_skips: list[tuple[str, str]] = []


def _require_cuda() -> bool:
    return os.getenv("HOTSHIFT_REQUIRE_CUDA") == "1"


def pytest_sessionstart(session):
    if not _require_cuda():
        return
    try:
        import torch
    except ImportError:
        pytest.exit("HOTSHIFT_REQUIRE_CUDA=1 but PyTorch is not installed", returncode=3)
    if not torch.cuda.is_available():
        pytest.exit("HOTSHIFT_REQUIRE_CUDA=1 but no CUDA device is available", returncode=3)


def pytest_collection_modifyitems(config, items):
    for item in items:
        folder = Path(str(item.fspath)).parent.name
        if folder in MARKERS:
            item.add_marker(getattr(pytest.mark, MARKERS[folder]))


def pytest_runtest_logreport(report):
    if report.skipped and "cuda" in report.keywords:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        _cuda_skips.append((report.nodeid, reason))


def _disallowed_skips() -> list[tuple[str, str]]:
    return [(node, reason) for node, reason in _cuda_skips if not any(allowed in reason for allowed in ALLOWED_RELEASE_SKIPS)]


def pytest_sessionfinish(session, exitstatus):
    # Release validation must not count a skipped GPU test as a GPU pass.
    if _require_cuda() and _disallowed_skips() and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _cuda_skips:
        return
    if _require_cuda():
        for node, reason in _disallowed_skips():
            terminalreporter.write_line(f"DISALLOWED CUDA SKIP: {node}: {reason}", red=True)
    terminalreporter.write_sep("!", f"GPU SKIP SUMMARY: {len(_cuda_skips)} CUDA test(s) skipped -- skipped tests are NOT a GPU pass", red=True, bold=True)
