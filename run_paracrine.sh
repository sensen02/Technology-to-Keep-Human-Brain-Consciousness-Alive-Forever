#!/usr/bin/env bash
# Run the spatial paracrine-ligand field self-test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$ROOT/venv/bin/python" "$ROOT/paracrine.py"
