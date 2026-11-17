#!/usr/bin/env bash
set -eu
cd "$(dirname -- "$0")"
root="$PWD"
prior="$root/../20261117T022700Z-sac-startup-replay"
old_run="$root/../20261117T020300Z-sac-single-box"
occlusion="$root/../20261117T021000Z-sac-occlusion"
left="$root/$1"
right="$root/$2"
[[ -f "$left/REFERENCE.json" && -f "$right/REFERENCE.json" ]]
run="$prior/drive.sh"
"$run" "$root/validate.py" "$left" "$right" --extra \
    "$old_run/SEED/flat" "$occlusion/SEED/offset_left" "$occlusion/SEED/offset_right" \
    "$old_run/SEED/contact" "$occlusion/LIVE/online_left" "$occlusion/LIVE/online_right"
"$run" "$prior/seed_imitation.py" "$old_run/WARM.pt" --output "$root/WARM_COMPLETE.pt" \
    "$old_run/SEED/flat" "$occlusion/SEED/offset_left" "$occlusion/SEED/offset_right" "$left" "$right"
"$run" "$root/paired_sac.py" "$root/WARM_COMPLETE.pt" "$root/PAIRED"
"$run" "$root/render_evidence.py" "$root/FIGURES"
