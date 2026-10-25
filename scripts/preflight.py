from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotshift.preflight import preflight


parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
arguments = parser.parse_args()
report = preflight(arguments.backend)
print(json.dumps(report, indent=2, sort_keys=True))
if arguments.backend == "cuda" and report["cuda_status"] != "available":
    raise SystemExit(2)

