from __future__ import annotations

import json
from pathlib import Path


def write_report(runs_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(runs_dir)
    summaries: list[dict] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.extend(payload if isinstance(payload, list) else [payload])
    lines = [
        "# HotShift Replay Report",
        "",
        "These measurements are from the CPU reference and cache simulator. They are not GPU results or evidence of serving latency improvements.",
        "",
        "| Scenario | Policy | Batch | Occurrence hit rate | Host miss bytes | CPU requests/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in summaries:
        hit_rate = result.get("occurrence_hit_rate")
        throughput = result.get("request_throughput")
        lines.append(
            f"| {result.get('scenario', 'n/a')} | {result.get('policy', 'n/a')} | "
            f"{result.get('batch_size', 'n/a')} | "
            f"{'n/a' if hit_rate is None else f'{hit_rate:.4f}'} | "
            f"{result.get('host_miss_bytes', 'n/a')} | "
            f"{'n/a' if throughput is None else f'{throughput:.2f}'} |"
        )
    destination = Path(output_path) if output_path else root / "REPORT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination

