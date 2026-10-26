"""Markdown report and dependency-free SVG charts built only from completed run artifacts."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

from hotshift.benchmark.common import is_complete_run

PALETTE = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load(directory: Path) -> tuple[dict[str, Any], Any, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    aggregate_path = directory / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else None
    return manifest, summary, aggregate


def line_chart(series: dict[str, list[tuple[float, float]]], title: str, x_label: str, y_label: str, markers: list[float] | None = None) -> str:
    width, height, left, right, top, bottom = 760, 360, 64, 160, 40, 48
    points = [point for values in series.values() for point in values]
    if not points:
        return ""
    x_min, x_max = min(p[0] for p in points), max(p[0] for p in points)
    y_min, y_max = 0.0, max(1e-9, max(p[1] for p in points))
    y_max = max(1.0, y_max)
    span_x = max(x_max - x_min, 1e-9)

    def sx(x: float) -> float:
        return left + (x - x_min) / span_x * (width - left - right)

    def sy(y: float) -> float:
        return top + (1 - (y - y_min) / (y_max - y_min)) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="14" font-weight="bold">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>',
        f'<text x="{(left + width - right) / 2}" y="{height - 12}" text-anchor="middle">{escape(x_label)}</text>',
        f'<text x="16" y="{(top + height - bottom) / 2}" transform="rotate(-90 16 {(top + height - bottom) / 2})" text-anchor="middle">{escape(y_label)}</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        parts.append(f'<text x="{left - 6}" y="{sy(value) + 4}" text-anchor="end" fill="#4b5563">{value:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{sy(value)}" x2="{width - right}" y2="{sy(value)}" stroke="#e5e7eb"/>')
    for marker in markers or []:
        parts.append(f'<line x1="{sx(marker)}" y1="{top}" x2="{sx(marker)}" y2="{height - bottom}" stroke="#111827" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{sx(marker) + 4}" y="{top + 12}" fill="#111827">phase change</text>')
    for index, (name, values) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
        parts.append(f'<rect x="{width - right + 16}" y="{top + index * 20}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 34}" y="{top + index * 20 + 10}">{escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _abrupt_chart(directory: Path, summary: list[dict[str, Any]]) -> tuple[str, str] | None:
    windows_path = directory / "windows.csv"
    abrupt = [row for row in summary if row.get("scenario") == "abrupt" and row.get("condition", "warm") == "warm"]
    if not windows_path.is_file() or not abrupt:
        return None
    seed = abrupt[0]["seed"]
    batch = abrupt[0]["batch_size"]
    fraction = abrupt[0].get("cache_fraction")
    series: dict[str, list[tuple[float, float]]] = {}
    with windows_path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["scenario"] != "abrupt" or int(row["seed"]) != seed or int(row["batch_size"]) != batch:
                continue
            if row.get("condition", "warm") != "warm" or row["variant"] != "optimized" or row["occurrence_hit_rate"] in ("", None):
                continue
            if fraction is not None and abs(float(row["cache_fraction"]) - float(fraction)) > 1e-12:
                continue
            if row["policy"].startswith("torch_"):
                continue
            series.setdefault(row["policy"], []).append((float(row["start_event"]), float(row["occurrence_hit_rate"])))
    boundary = abrupt[0].get("recovery", {}).get("boundary_event") if abrupt[0].get("recovery") else None
    svg = line_chart(series, f"Window hit rate, abrupt shift (seed {seed}, batch {batch})", "trace event index", "occurrence hit rate", [boundary] if boundary is not None else None)
    return ("abrupt_hit_rate.svg", svg) if svg else None


def _matrix_section(directory: Path, manifest: dict[str, Any], summary: list[dict[str, Any]], aggregate: list[dict[str, Any]] | None, charts_dir: Path) -> list[str]:
    environment = manifest.get("environment", {})
    lines = [
        f"## Run `{manifest.get('run_id')}`",
        "",
        f"- Backend: **{manifest.get('backend')}**{' on ' + str(environment.get('gpu')) if environment.get('gpu') else ''}; smoke: {manifest.get('smoke')}",
        f"- Commit: `{manifest.get('git_commit')}`; dirty tree: {manifest.get('git_dirty')}; release eligible: {manifest.get('release_eligible')}",
        f"- Table: {manifest.get('rows')} x {manifest.get('dimensions')}; window: {manifest.get('window_requests')} trace requests",
        f"- Raw data: `{directory.as_posix()}/summary.json`, `windows.csv`, `batch_samples.csv.gz`",
        "",
    ]
    if manifest.get("backend") == "cpu":
        lines += ["CPU simulator timings are not GPU timings. Hit rates describe policy semantics only.", ""]
    rows = aggregate or []
    if rows:
        lines += [
            "| Scenario | Policy | Variant | Batch | Cond. | Cache frac. | n | Hit rate (median) | Requests/s median [min, max] | Batch p99 ms (median) | H2D bytes (median) | Recovery | Outputs |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
        ]
        for row in sorted(rows, key=lambda item: (item["scenario"], item["batch_size"], item["cache_fraction"], item["condition"], item["policy"], item["variant"])):
            recovery = ", ".join(row.get("recovery_statuses") or []) or "n/a"
            if row.get("recovery_requests_median") is not None:
                recovery += f" (median {row['recovery_requests_median']:.0f} req)"
            lines.append(
                f"| {row['scenario']} | {row['policy']} | {row['variant']} | {row['batch_size']} | {row['condition']} | {row['cache_fraction']:.3f} | {row['repeats']} | "
                f"{_fmt(row.get('occurrence_hit_rate_median'))} | {_fmt(row.get('request_throughput_median'), 1)} [{_fmt(row.get('request_throughput_min'), 1)}, {_fmt(row.get('request_throughput_max'), 1)}] | "
                f"{_fmt(row.get('batch_wall_ms_p99_median'))} | {_fmt(row.get('h2d_bytes_median'), 0)} | {recovery} | {'pass' if row['output_check_passed'] else 'FAIL'} |"
            )
        lines += ["", *_losing_cases(rows), ""]
    chart = _abrupt_chart(directory, summary)
    if chart:
        name, svg = chart
        target = charts_dir / f"{manifest.get('run_id')}-{name}"
        target.write_text(svg, encoding="utf-8")
        lines += [f"![Window hit rate around the abrupt shift]({target.relative_to(charts_dir.parent).as_posix()})", ""]
    return lines


def _losing_cases(rows: list[dict[str, Any]]) -> list[str]:
    """Report every configuration where adaptive loses to a simpler policy or caching loses to no cache."""
    lines = ["### Unfavorable and low-benefit cases", ""]
    groups: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["variant"] != "optimized":
            continue
        key = (row["scenario"], row["batch_size"], row["condition"], row["cache_fraction"])
        groups.setdefault(key, {})[row["policy"]] = row
    found = False
    for (scenario, batch, condition, fraction), policies in sorted(groups.items()):
        throughput = {name: row.get("request_throughput_median") for name, row in policies.items() if row.get("request_throughput_median") is not None}
        if "adaptive" in throughput:
            winners = [name for name in ("lru", "static", "decayed_lfu", "no_cache") if throughput.get(name, 0) > throughput["adaptive"]]
            if winners:
                found = True
                lines.append(f"- {scenario}, batch {batch}, {condition}, cache {fraction:.3f}: adaptive throughput is below {', '.join(winners)}.")
        if "no_cache" in throughput:
            slower = [name for name in ("static", "lru", "decayed_lfu", "adaptive") if name in throughput and throughput[name] < throughput["no_cache"]]
            if slower:
                found = True
                lines.append(f"- {scenario}, batch {batch}, {condition}, cache {fraction:.3f}: {', '.join(slower)} slower than no cache.")
    if not found:
        lines.append("- No configuration in this run had adaptive below a simpler policy or a cache below no cache on median throughput.")
    return lines


def _kernel_section(manifest: dict[str, Any], summary: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## Kernel microbenchmark `{manifest.get('run_id')}` ({manifest.get('environment', {}).get('gpu')})",
        "",
        "Median CUDA event span per call; device operation only.",
        "",
        "| D | Bag length | Unique | Fused pool us | Gather + embedding_bag us | Resident embedding_bag us | Gather/fused | Max abs error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['dimensions']} | {row['bag_length']} | {row['unique_ids']} | {row['fused_pool_us']:.2f} | {row['gather_then_bag_us']:.2f} | "
            f"{row['resident_bag_us']:.2f} | {_fmt(row.get('fused_vs_gather_ratio'), 2)} | {max(row['fused_max_abs_error'], row['gather_max_abs_error']):.2e} |"
        )
    return [*lines, ""]


def _load_section(manifest: dict[str, Any], summary: list[dict[str, Any]]) -> list[str]:
    row = summary[0]
    return [
        f"## Open-loop service replay `{manifest.get('run_id')}`",
        "",
        f"- Backend {manifest.get('backend')}, policy {row.get('policy')}, rate scale {manifest.get('rate_scale')}, max in flight {manifest.get('max_in_flight')}",
        f"- Offered {row['offered_requests']}, successful {row['successful_requests']}, rejected {row['rejected_requests']}, timed out {row['expired_requests']}, dropped by client {row['dropped_dispatches']}",
        f"- Client latency p50/p95/p99: {_fmt(row['client_latency_ms_p50'], 2)} / {_fmt(row['client_latency_ms_p95'], 2)} / {_fmt(row['client_latency_ms_p99'], 2)} ms; dispatch lag p99 {_fmt(row['dispatch_lag_ms_p99'], 2)} ms",
        "",
    ]


def write_report(runs_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(runs_dir)
    destination = Path(output_path) if output_path else root / "REPORT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    charts_dir = destination.parent / "charts"
    charts_dir.mkdir(exist_ok=True)
    preamble = (
        "Generated from completed artifacts only; incomplete runs are skipped. CPU replay timing is not GPU timing. "
        "CUDA batch wall time includes host planning, transfers, synchronization, kernels, and policy maintenance. "
        "Medians are across repeated seeds; individual runs remain in each run's summary.json. "
        "Compare only rows with matching hardware, data, traces, and budget."
    )
    lines = ["# HotShift Benchmark Report", "", preamble, ""]
    legacy: list[str] = []
    for directory in sorted(path for path in root.iterdir() if is_complete_run(path)) if root.is_dir() else []:
        try:
            manifest, summary, aggregate = _load(directory)
        except (OSError, ValueError):
            continue
        if int(manifest.get("schema_version", 1)) < 2:
            legacy.append(directory.name)
            continue
        kind = manifest.get("kind", "replay_matrix")
        if kind == "kernel_microbenchmark":
            lines += _kernel_section(manifest, summary)
        elif kind == "open_loop_service_replay":
            lines += _load_section(manifest, summary)
        elif isinstance(summary, list) and summary and "scenario" in summary[0]:
            lines += _matrix_section(directory, manifest, summary, aggregate, charts_dir)
    if legacy:
        lines += [f"Skipped {len(legacy)} schema-v1 run(s) recorded before the current artifact format: {', '.join(legacy)}.", ""]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
