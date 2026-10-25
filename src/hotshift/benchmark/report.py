from __future__ import annotations

import json
from pathlib import Path


def write_report(runs_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(runs_dir)
    records: list[tuple[str, dict]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = summary_path.parent / "manifest.json"
        backend = "unknown"
        if manifest_path.is_file():
            backend = json.loads(manifest_path.read_text(encoding="utf-8")).get("backend", "unknown")
        records.extend((backend, row) for row in (payload if isinstance(payload, list) else [payload]))
    lines = [
        "# HotShift Benchmark Report",
        "",
        "CPU replay timing is not GPU timing. CUDA batch wall time includes host planning, transfer, synchronization, kernels, and policy maintenance. Compare only runs with matching hardware, data, traces, and configuration.",
        "",
        "| Backend | Scenario | Seed | Policy | Batch | Occurrence hit rate | Host miss bytes | Batch p50 ms | Requests/s |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend, result in records:
        hit_rate = result.get("occurrence_hit_rate")
        throughput = result.get("request_throughput")
        batch_p50 = result.get("batch_wall_ms_p50", result.get("batch_engine_ms_p50"))
        lines.append(
            f"| {backend} | {result.get('scenario', 'n/a')} | {result.get('seed', 'n/a')} | {result.get('policy', 'n/a')} | "
            f"{result.get('batch_size', 'n/a')} | "
            f"{'n/a' if hit_rate is None else f'{hit_rate:.4f}'} | "
            f"{result.get('host_miss_bytes', 'n/a')} | "
            f"{'n/a' if batch_p50 is None else f'{batch_p50:.4f}'} | "
            f"{'n/a' if throughput is None else f'{throughput:.2f}'} |"
        )
    destination = Path(output_path) if output_path else root / "REPORT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination

