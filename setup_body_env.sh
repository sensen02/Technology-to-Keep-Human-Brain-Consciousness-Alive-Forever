#!/usr/bin/env bash
# Create a SEPARATE environment for the embodied body backend.
#
# WHY SEPARATE: the main venv carries NEURON 9.0.2 and the neural suites.  The
# body backend needs MuJoCo/FlyGym, and installing those into the same
# environment risks breaking the neural stack that 16 suites depend on.  The two
# environments are therefore kept apart and their versions recorded.
#
# FlyGym 2.1.0 requires python >=3.12,<3.15 and mujoco >=3.9,<3.10.
set -euo pipefail

ROOT=/run/media/sensen/Data2/cell_wound_prototype
ENV="$ROOT/venv_body"
LOG="$ROOT/outputs/embodied_body"

mkdir -p "$LOG"

# The main venv is 3.12.14; use the same interpreter major.minor for the body env
# so both environments agree about numpy ABI-era.
MAIN_PY="$ROOT/venv/bin/python"
BASE_PY="$("$MAIN_PY" -c 'import sys;print(sys.base_prefix)')/bin/python3.12"
if [ ! -x "$BASE_PY" ]; then BASE_PY="$(command -v python3.12 || true)"; fi
if [ -z "${BASE_PY:-}" ] || [ ! -x "$BASE_PY" ]; then
  echo "python3.12 not found; falling back to system python3" >&2
  BASE_PY="$(command -v python3)"
fi
echo "base interpreter: $BASE_PY"
"$BASE_PY" -V

if [ ! -x "$ENV/bin/python" ]; then
  "$BASE_PY" -m venv "$ENV"
fi

"$ENV/bin/python" -m pip install --quiet --upgrade pip wheel

# Pin the versions we verified.  flygym 2.1.0 is the version whose pyproject we
# read (Apache-2.0, mujoco>=3.9,<3.10).
"$ENV/bin/python" -m pip install --quiet \
  "flygym==2.1.0" \
  "mujoco>=3.9,<3.10" \
  "numpy>=2.0,<3.0"

"$ENV/bin/python" - <<'PY' | tee "$LOG/body_env_versions.json"
import importlib.metadata as md, json, platform, sys
out = {"python": sys.version, "executable": sys.executable,
       "platform": platform.platform(),
       "pin_note": "flygym==2.1.0 pinned; mujoco pinned >=3.9,<3.10 per its pyproject",
       "packages": {}}
for name in ("flygym", "mujoco", "numpy", "scipy", "numba", "mediapy", "imageio",
             "pyyaml", "tabulate", "pillow", "matplotlib", "loguru", "jaxtyping"):
    try:
        out["packages"][name] = md.version(name)
    except md.PackageNotFoundError:
        out["packages"][name] = None
print(json.dumps(out, indent=2))
PY

echo "--- import check ---"
"$ENV/bin/python" - <<'PY'
import json
import flygym, mujoco, numpy
print(json.dumps({"flygym": flygym.__version__ if hasattr(flygym, "__version__") else "imported",
                  "mujoco": mujoco.__version__,
                  "numpy": numpy.__version__,
                  "mujoco_gl": str(mujoco.GLContext is not None)}, indent=2))
PY

echo "OK: body environment ready at $ENV"
