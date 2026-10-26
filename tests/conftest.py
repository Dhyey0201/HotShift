from __future__ import annotations

import os
from pathlib import Path

import pytest

MARKERS = {"cpu": "cpu", "integration": "integration", "cuda": "cuda"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        folder = Path(str(item.fspath)).parent.name
        if folder in MARKERS:
            item.add_marker(getattr(pytest.mark, MARKERS[folder]))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # Release validation must not count a skipped GPU test as a GPU pass.
    if os.getenv("HOTSHIFT_REQUIRE_CUDA") == "1" and report.skipped and item.get_closest_marker("cuda"):
        report.outcome = "failed"
        report.longrepr = "HOTSHIFT_REQUIRE_CUDA=1 but this CUDA test was skipped"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = [report for report in terminalreporter.stats.get("skipped", []) if "cuda" in report.keywords]
    if skipped:
        terminalreporter.write_sep("!", f"GPU SKIP SUMMARY: {len(skipped)} CUDA test(s) skipped -- this is NOT a GPU pass", red=True, bold=True)
