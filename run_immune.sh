#!/usr/bin/env bash
# Run the haemocyte/clearance layer self-test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$ROOT/venv/bin/python" "$ROOT/immune.py"
