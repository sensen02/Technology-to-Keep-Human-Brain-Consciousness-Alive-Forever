#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mj=$(bash env.sh -c 'import mujoco;from pathlib import Path;print(Path(mujoco.__file__).parent)')
cc -std=c11 -O3 -Wall -Wextra -shared -fPIC -I"$mj/include" foot_load.c "$mj/libmujoco.so.3.9.0" -Wl,-rpath,"$mj" -o foot_load.so
