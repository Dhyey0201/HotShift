"""Release gate. Exits 0 only when every mandatory check passes.

CPU-side checks always run. Without a CUDA host the script exits 2 (BLOCKED) after them;
it never reports skipped GPU work as passed. A JSON release summary is written either way.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hotshift.benchmark.common import environment, git_state, is_complete_run
from hotshift.preflight import preflight

REQUIRED_FILES = [
    "README.md", "LICENSE", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md", "docs/BENCHMARKS.md",
    "docs/DECISIONS.md", "docs/STATUS.md", "docs/DEMO.md", "docs/PERFORMANCE.md", "docs/RELEASE.md",
    "requirements/cpu-py311-linux.lock", "frontend/package-lock.json", "csrc/probe.cu", "csrc/pool.cu",
]
# Phrases that need measured backing; a line passes only when it negates or qualifies them.
CLAIM_PATTERNS = re.compile(r"(novel|state[- ]of[- ]the[- ]art|production[- ](ready|scale)|\d+(\.\d+)?x faster|universally|NVIDIA[- ]scale)", re.IGNORECASE)
QUALIFIERS = re.compile(r"\b(not|no|never|avoid|without|nor|does not|do not)\b", re.IGNORECASE)

results: list[dict[str, object]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append({"check": name, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
    return ok


def run(name: str, command: list[str], env: dict[str, str] | None = None) -> bool:
    import os

    completed = subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})}, check=False)
    return record(name, completed.returncode == 0, " ".join(command))


def claim_scan() -> bool:
    offenders = []
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CLAIM_PATTERNS.search(line) and not QUALIFIERS.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    return record("unsupported claim scan", not offenders, ", ".join(offenders))


def evidence_roots() -> list[Path]:
    """Local ``artifacts/`` first, then committed evidence folders under ``evidence/``."""
    return [ROOT / "artifacts", *sorted(path for path in (ROOT / "evidence").glob("*") if path.is_dir())]


def gpu_evidence(root: Path | None = None) -> bool:
    roots = [root] if root else evidence_roots()
    full = []
    for path in (run for base in roots for run in sorted((base / "runs").glob("*-cuda-*"))):
        if not is_complete_run(path):
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        scenarios = {row["scenario"] for row in summary}
        required = {"stable", "abrupt", "brief_burst", "uniform"}
        if not manifest.get("smoke") and required <= scenarios and all(row["output_check"]["passed"] for row in summary):
            full.append(path.relative_to(ROOT).as_posix())
    return record("full CUDA matrix with verified outputs", bool(full), ", ".join(full) or "no complete non-smoke CUDA run found")


def resolve_commit(commit: str, root: Path) -> str | None:
    """Return ``commit`` if it exists, else its ID after a history rewrite.

    A rewrite keeps code trees but changes commit IDs. The mapping comes from the evidence
    folder's ``provenance.json`` (``rewritten_commits``) or a local ``git filter-repo`` map.
    """
    exists = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=False, capture_output=True)
    if exists.returncode == 0:
        return commit
    provenance = root / "provenance.json"
    if provenance.is_file():
        rewritten = json.loads(provenance.read_text(encoding="utf-8")).get("rewritten_commits", {})
        if commit in rewritten:
            return resolve_commit(rewritten[commit], root) if rewritten[commit] != commit else None
    commit_map = ROOT / ".git" / "filter-repo" / "commit-map"
    if commit_map.is_file():
        for line in commit_map.read_text(encoding="utf-8").splitlines():
            old, _, new = line.partition(" ")
            if old == commit and new.strip("0"):
                return new
    return None


def imported_gpu_evidence() -> bool:
    """Accept GPU gates run elsewhere (for example Kaggle) when the tested code matches HEAD.

    Looks in ``artifacts/`` and each ``evidence/*`` folder for the log from
    ``scripts/gpu_evidence.sh`` reporting every step passed, four clean sanitizer logs, and a
    full verified CUDA matrix whose commit has the same ``src``, ``csrc``, and ``tests`` as HEAD.
    """
    root = next((base for base in evidence_roots() if (base / "logs" / "gpu_evidence.log").is_file()), None)
    if root is None:
        return False
    log = root / "logs" / "gpu_evidence.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    tests = re.findall(r"^(\d+) passed(?:, (\d+) skipped)?[^\n]*$", text, re.MULTILINE)
    ok = record("imported GPU evidence log", "SUMMARY: all steps passed" in text and bool(tests) and not re.search(r"^\d+ failed", text, re.MULTILINE),
                f"{log.relative_to(ROOT)}; test lines: {', '.join(t[0] + ' passed' for t in tests)}")
    dirty = []
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        path = root / "sanitizers" / f"{tool}.log"
        content = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if not re.search(r"(ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards)", content):
            dirty.append(tool)
    ok &= record("imported Compute Sanitizer logs", not dirty, ", ".join(dirty) or "memcheck, racecheck, initcheck, synccheck clean")
    if not gpu_evidence(root):
        return False
    commits = set()
    for path in (root / "runs").glob("*-cuda-*"):
        if is_complete_run(path):
            manifest = json.loads((path / "manifest.json").read_text())
            if not manifest.get("smoke") and not manifest.get("git_dirty"):
                commits.add(manifest.get("git_commit"))
    matching = [
        commit for commit in commits if commit and (resolved := resolve_commit(commit, root)) and subprocess.run(
            ["git", "diff", "--quiet", resolved, "HEAD", "--", "src", "csrc", "tests"], cwd=ROOT, check=False,
        ).returncode == 0
    ]
    ok &= record("imported evidence matches tested code", bool(matching),
                 f"src, csrc, tests unchanged since {', '.join(c[:12] for c in matching)}" if matching else f"code changed since {', '.join(sorted(c[:12] for c in commits if c))}")
    return ok


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    record("required files", not missing, ", ".join(missing))
    claim_scan()
    run("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"])
    run("CPU and integration tests", [sys.executable, "-m", "pytest", "tests/cpu", "tests/integration", "-q"])
    run("frontend unit tests", ["npm", "run", "test", "--prefix", "frontend"])
    run("frontend production build", ["npm", "run", "build", "--prefix", "frontend"])
    gpu = preflight("cuda")
    local_cuda = gpu["cuda_status"] == "available"
    blocked = not local_cuda
    if blocked and imported_gpu_evidence():
        blocked = False
    elif blocked:
        record("CUDA gates", False, f"BLOCKED: {gpu['cuda_block_reason']}")
    else:
        run("build extension", [sys.executable, "-m", "hotshift.cli", "build-extension"])
        run("CUDA tests (zero skips allowed)", [sys.executable, "-m", "pytest", "tests/cuda", "-q"], {"HOTSHIFT_REQUIRE_CUDA": "1"})
        run("Compute Sanitizer", ["bash", "scripts/run_sanitizers.sh"])
        run("CUDA smoke benchmark", [sys.executable, "-m", "hotshift.cli", "bench-cuda", "--matrix", "configs/smoke.yaml", "--smoke"])
        gpu_evidence()
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        **git_state(),
        "environment": environment(include_cuda=local_cuda),
        "gpu_evidence": "local" if local_cuda else ("imported" if not blocked else "missing"),
        "cuda_status": gpu["cuda_status"],
        "checks": results,
        "commands": ["python3 scripts/release_check.py", "bash scripts/gpu_evidence.sh"],
        "remaining_optional_work": [
            "FP16 storage", "GPU hash-table mapping", "overlapping batches with slot lifetimes",
            "external FBGEMM or NVIDIA embedding-cache baseline", "licensed real music dataset",
        ],
    }
    target = ROOT / "artifacts" / "release"
    target.mkdir(parents=True, exist_ok=True)
    (target / "release_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [item for item in results if not item["ok"]]
    if blocked and all(item["ok"] for item in results if item["check"] != "CUDA gates"):
        print("RELEASE BLOCKED: CPU-side checks pass; mandatory CUDA validation requires an NVIDIA CUDA host", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
