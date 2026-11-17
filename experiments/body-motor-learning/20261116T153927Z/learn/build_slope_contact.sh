#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")"
mujoco_dir=$(./drive.sh -c 'from pathlib import Path; import mujoco; print(Path(mujoco.__file__).parent)')
readarray -t mujoco_libs < <(find "$mujoco_dir" -maxdepth 1 -name 'libmujoco.so.*' -print)
test "${#mujoco_libs[@]}" -eq 1
c++ -O3 -std=c++17 -shared -fPIC -I "$mujoco_dir/include" slope_contact.cpp "${mujoco_libs[0]}" \
    -Wl,-rpath,"$mujoco_dir" -o slope_contact.so
