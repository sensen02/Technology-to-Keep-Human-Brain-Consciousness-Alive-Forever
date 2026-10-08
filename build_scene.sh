#!/usr/bin/env bash
# Build (or rebuild) the single-scene payload for the workbench.
#
#   bash build_scene.sh            # build
#   bash build_scene.sh --check    # build, then run the scene regression test
#
# The builder needs numpy + pyarrow. pyarrow is vendored under vendor/pylibs
# because it exists nowhere else on this machine's project environments and the
# only other copy lived in /tmp (lost on reboot).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/venv_body/bin/python"
LIB="$ROOT/vendor/pylibs"

if [ ! -x "$PY" ]; then
  echo "missing interpreter: $PY" >&2
  exit 1
fi
if [ ! -d "$LIB/pyarrow" ]; then
  echo "missing vendored pyarrow at $LIB/pyarrow" >&2
  exit 1
fi

CHECK=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--check" ]; then CHECK=1; else ARGS+=("$a"); fi
done

echo "== building scene payload =="
PYTHONPATH="$LIB" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PY" "$ROOT/scene_build_cli.py" ${ARGS[@]+"${ARGS[@]}"}

echo
echo "== payload =="
"$PY" - <<'PY'
import json, pathlib
m = json.loads(pathlib.Path(__file__).parent.joinpath('viewer/scene/manifest.json').read_text())
print('bytes  :', m['budget']['bytes_total'], '(limit', m['budget']['limit_bytes'], ')')
print('layers :', len(m['layers']))
for l in m['layers']:
    b = sum(m['chunks'][c]['bytes'] for c in l.get('chunks', []))
    print('  %-22s %8.0f KB  %s' % (l['id'], b / 1024, l['label_zh']))
print('rigid  :', m['transform_audit']['rigid'], m['transform_audit']['max_pairwise_distance_error_mm'], 'mm')
PY

if [ "$CHECK" = "1" ]; then
  echo
  echo "== scene regression (needs the service on 8766) =="
  node "$ROOT/tools_test_scene.cjs"
fi
