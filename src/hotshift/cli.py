from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from hotshift.config import load_config
from hotshift.data.catalog import generate_catalog
from hotshift.data.generate import generate_from_config
from hotshift.data.store import EmbeddingStore
from hotshift.engine.cpu import pool_mean_reference
from hotshift.traffic.generate import SCENARIOS


def _reference_smoke(config_path: str) -> None:
    config = load_config(config_path)
    store = EmbeddingStore(config.output_dir)
    ids = [1, 1, 2]
    offsets = [0, 2, 2, 3]
    result = pool_mean_reference(store, ids, offsets)
    if result.shape != (3, config.dimensions) or not np.isfinite(result).all():
        raise RuntimeError("reference pooling returned an invalid result")
    if not np.allclose(result[0], store.table[1], atol=1e-5, rtol=1e-4):
        raise RuntimeError("duplicate IDs did not preserve mean-pooling semantics")
    if not np.array_equal(result[1], np.zeros(config.dimensions, dtype=np.float32)):
        raise RuntimeError("empty bag did not return a zero vector")
    if not np.allclose(result[2], store.table[2], atol=1e-5, rtol=1e-4):
        raise RuntimeError("single-ID bag returned an unexpected vector")
    print(json.dumps({"backend": "cpu", "shape": list(result.shape), "status": "ok"}))


def _dataset_ready(config_path: str) -> bool:
    config = load_config(config_path)
    try:
        store = EmbeddingStore(config.output_dir)
    except (OSError, ValueError):
        return False
    return store.rows == config.rows and store.dimensions == config.dimensions and int(store.manifest.get("seed", -1)) == config.seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hotshift")
    parser.add_argument("--log-level", default=os.getenv("HOTSHIFT_LOG_LEVEL", "INFO"))
    commands = parser.add_subparsers(dest="command", required=True)
    data_parser = commands.add_parser("data", help="manage deterministic demo data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    generate_parser = data_commands.add_parser("generate")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--if-missing", action="store_true", help="skip when a valid matching dataset exists")
    catalog_parser = data_commands.add_parser("catalog")
    catalog_parser.add_argument("--config", required=True)
    reference_parser = commands.add_parser("test-reference", help="run the CPU reference smoke check")
    reference_parser.add_argument("--config", required=True)
    preflight_parser = commands.add_parser("preflight", help="inspect the selected execution backend")
    preflight_parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    preflight_parser.add_argument("--config")
    trace_parser = commands.add_parser("trace", help="generate deterministic request traces")
    trace_commands = trace_parser.add_subparsers(dest="trace_command", required=True)
    trace_generate = trace_commands.add_parser("generate")
    trace_generate.add_argument("--config", default="configs/smoke.yaml")
    trace_generate.add_argument("--scenario", choices=SCENARIOS, default="abrupt")
    trace_generate.add_argument("--seed", type=int, default=42)
    trace_generate.add_argument("--requests", type=int, default=10000)
    trace_generate.add_argument("--history-length", type=int, default=32)
    trace_generate.add_argument("--rate", type=float, default=1000.0, help="scheduled requests per second")
    trace_generate.add_argument("--output")
    bench_parser = commands.add_parser("bench", help="replay the benchmark matrix (CPU simulator by default)")
    bench_parser.add_argument("--matrix", required=True)
    bench_parser.add_argument("--smoke", action="store_true")
    bench_parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    cuda_bench_parser = commands.add_parser("bench-cuda", help="replay the benchmark matrix with the CUDA engine")
    cuda_bench_parser.add_argument("--matrix", required=True)
    cuda_bench_parser.add_argument("--smoke", action="store_true")
    kernels_parser = commands.add_parser("bench-kernels", help="CUDA operator microbenchmarks across D and bag length")
    kernels_parser.add_argument("--repeats", type=int, default=50)
    profile_parser = commands.add_parser("profile-torch", help="short CUDA replay under torch.profiler")
    profile_parser.add_argument("--config", default="configs/smoke.yaml")
    profile_parser.add_argument("--requests", type=int, default=2048)
    load_parser = commands.add_parser("loadtest", help="open-loop replay of a trace against a running service")
    load_parser.add_argument("--trace", required=True)
    load_parser.add_argument("--url", default="http://127.0.0.1:8000")
    load_parser.add_argument("--rate-scale", type=float, default=1.0)
    load_parser.add_argument("--max-in-flight", type=int, default=256)
    load_parser.add_argument("--timeout-ms", type=int, default=2000)
    load_parser.add_argument("--limit", type=int)
    report_parser = commands.add_parser("report", help="summarize completed local runs")
    report_parser.add_argument("--runs", default="artifacts/runs")
    report_parser.add_argument("--output")
    serve_parser = commands.add_parser("serve", help="start one local API worker")
    serve_parser.add_argument("--config", default="configs/demo.yaml")
    serve_parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    serve_parser.add_argument("--host", default=os.getenv("HOTSHIFT_HOST", "127.0.0.1"))
    serve_parser.add_argument("--port", type=int, default=int(os.getenv("HOTSHIFT_PORT", "8000")))
    serve_parser.add_argument("--shutdown-timeout", type=float, default=10.0)
    build_parser = commands.add_parser("build-extension", help="build owned CUDA operators")
    build_parser.add_argument("--verbose", action="store_true")
    build_parser.add_argument("--no-device-check", action="store_true", help="compile without a visible GPU (image builds)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")

    try:
        if args.command == "data" and args.data_command == "generate":
            if args.if_missing and _dataset_ready(args.config):
                print(json.dumps({"status": "exists", "config": args.config}))
            else:
                manifest = generate_from_config(load_config(args.config))
                print(json.dumps(manifest, indent=2, sort_keys=True))
        elif args.command == "data" and args.data_command == "catalog":
            config = load_config(args.config)
            catalog = generate_catalog(
                config.rows, config.dimensions, config.seed, config.output_dir
            )
            print(json.dumps({"catalog_items": len(catalog), "candidate_count": min(512, len(catalog)), "dataset_version": "synthetic-v1"}))
        elif args.command == "test-reference":
            _reference_smoke(args.config)
        elif args.command == "preflight":
            from hotshift.preflight import preflight

            result = preflight(args.backend, args.config)
            print(json.dumps(result, indent=2, sort_keys=True))
            if args.backend == "cuda" and result["cuda_status"] != "available":
                return 2
        elif args.command == "trace" and args.trace_command == "generate":
            from hotshift.traffic.generate import generate_trace

            config = load_config(args.config)
            target = Path(args.output or f"artifacts/traces/{args.scenario}-seed{args.seed}.jsonl")
            result = generate_trace(
                target,
                config.rows,
                requests=args.requests,
                history_length=args.history_length,
                seed=args.seed,
                scenario=args.scenario,
                request_rate=args.rate,
            )
            result["path"] = str(target)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "bench":
            from hotshift.benchmark.runner import run_matrix

            print(run_matrix(args.matrix, smoke=args.smoke, backend=args.backend))
        elif args.command == "bench-cuda":
            from hotshift.benchmark.cuda_runner import run_cuda_matrix

            print(run_cuda_matrix(args.matrix, smoke=args.smoke))
        elif args.command == "bench-kernels":
            from hotshift.benchmark.kernels import run_kernel_benchmark

            print(run_kernel_benchmark(repeats=args.repeats))
        elif args.command == "profile-torch":
            from hotshift.benchmark.kernels import run_torch_profile

            print(run_torch_profile(args.config, requests=args.requests))
        elif args.command == "loadtest":
            from hotshift.benchmark.loadtest import run_loadtest

            print(run_loadtest(args.trace, args.url, args.rate_scale, args.max_in_flight, args.timeout_ms, args.limit))
        elif args.command == "report":
            from hotshift.benchmark.report import write_report

            print(write_report(args.runs, args.output))
        elif args.command == "serve":
            try:
                import uvicorn
            except ImportError as error:
                raise RuntimeError("install service dependencies with `pip install -e '.[service]'`") from error
            os.environ["HOTSHIFT_CONFIG"] = args.config
            os.environ["HOTSHIFT_BACKEND"] = args.backend
            os.environ.setdefault("HOTSHIFT_SHUTDOWN_TIMEOUT", str(args.shutdown_timeout / 2))
            # One worker only: each worker would own a separate cache and GPU context.
            uvicorn.run(
                "hotshift.serving.app:app",
                host=args.host,
                port=args.port,
                workers=1,
                reload=False,
                timeout_graceful_shutdown=int(args.shutdown_timeout),
            )
        elif args.command == "build-extension":
            from hotshift.engine.cuda import build_cuda_extension

            build_cuda_extension(verbose=args.verbose, require_device=not args.no_device_check)
            print("HotShift CUDA extension built and loaded")
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        print(f"hotshift: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
