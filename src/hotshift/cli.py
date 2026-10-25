from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from hotshift.config import load_config
from hotshift.data.generate import generate_from_config
from hotshift.data.catalog import generate_catalog
from hotshift.data.store import EmbeddingStore
from hotshift.engine.cpu import pool_mean_reference


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hotshift")
    commands = parser.add_subparsers(dest="command", required=True)
    data_parser = commands.add_parser("data", help="manage deterministic demo data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    generate_parser = data_commands.add_parser("generate")
    generate_parser.add_argument("--config", required=True)
    catalog_parser = data_commands.add_parser("catalog")
    catalog_parser.add_argument("--config", required=True)
    reference_parser = commands.add_parser("test-reference", help="run the CPU reference smoke check")
    reference_parser.add_argument("--config", required=True)
    preflight_parser = commands.add_parser("preflight", help="inspect the selected execution backend")
    preflight_parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    try:
        if args.command == "data" and args.data_command == "generate":
            manifest = generate_from_config(load_config(args.config))
            print(json.dumps(manifest, indent=2, sort_keys=True))
        elif args.command == "data" and args.data_command == "catalog":
            config = load_config(args.config)
            catalog = generate_catalog(
                config.rows, config.dimensions, config.seed, config.output_dir
            )
            print(json.dumps({"catalog_items": len(catalog), "dataset_version": "synthetic-v1"}))
        elif args.command == "test-reference":
            _reference_smoke(args.config)
        elif args.command == "preflight":
            from hotshift.preflight import preflight

            result = preflight(args.backend)
            print(json.dumps(result, indent=2, sort_keys=True))
            if args.backend == "cuda" and result["cuda_status"] != "available":
                return 2
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"hotshift: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

