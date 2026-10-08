#!/usr/bin/env python3
"""Rebuild recording_index.json from the episode directories on disk.

WHY THIS EXISTS: a throw-away test run (different fps and seed) overwrote the index
with its own three episodes, which made the REPORT describe the wrong experiment.  The
per-episode artefacts were all intact, so the index is a derived file and can be
rebuilt from them -- and it now keeps a note saying it was rebuilt, so the difference
between "as recorded" and "reconstructed from the directories" is visible.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python tools_rebuild_recording_index.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "electrode_payload"


def main() -> int:
    old_path = OUT / "recording_index.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    eps = []
    for d in sorted(OUT.glob("episode_*_seed*")):
        if not d.is_dir():
            continue
        m = re.match(r"episode_(.+)_seed(\d+)$", d.name)
        if not m:
            continue
        label, seed = m.group(1), int(m.group(2))
        payload = json.loads((d / "payload.json").read_text()) if \
            (d / "payload.json").exists() else None
        sig = json.loads((d / "model_signature.json").read_text()) if \
            (d / "model_signature.json").exists() else {}
        cam = json.loads((d / "camera.json").read_text()) if \
            (d / "camera.json").exists() else {}
        camd = json.loads((d / "camera_detail.json").read_text()) if \
            (d / "camera_detail.json").exists() else {}
        n_world = len(list((d / "frames").glob("*.png")))
        n_detail = len(list((d / "frames_detailcam").glob("*.png")))
        eps.append({
            "seed": seed, "condition": label,
            "load_fraction_requested": (payload or {}).get("config", {}).get(
                "load_fraction"),
            "mass_scale": (payload or {}).get("config", {}).get("mass_scale"),
            "load_fraction_achieved": (payload or {}).get("load_fraction_achieved"),
            "payload_total_kg": (payload or {}).get("total_payload_kg"),
            "body_mass_sum_kg": (sig.get("signature") or {}).get("mass_sum_kg"),
            "mass_sha1": (sig.get("signature") or {}).get("mass_sha1"),
            "n_frames": n_world,
            "frames_per_camera": {"worldcam": n_world, "detailcam": n_detail},
            "fps": cam.get("fps_requested"),
            "fps_achieved": cam.get("fps_achieved"),
            "camera_res": cam.get("image_size_px"),
            "wall_seconds": None,
            "dir": str(d),
        })
    idx = {
        "seconds": old.get("seconds"),
        "fps_requested": old.get("fps"),
        "conditions": sorted({e["condition"] for e in eps}),
        "seeds": sorted({e["seed"] for e in eps}),
        "episodes": eps,
        "rebuilt_from_directories": True,
        "rebuild_note": ("this index was regenerated from the episode directories by "
                         "tools_rebuild_recording_index.py after a throw-away test run "
                         "overwrote the original; per-episode numbers are read from the "
                         "episodes themselves and frame counts are counted from disk"),
        "original_started_utc": old.get("started_utc"),
        "original_finished_utc": old.get("finished_utc"),
    }
    # seconds/fps are per-episode facts here; take them from the episode metadata
    if eps:
        idx["seconds"] = None
    dest = OUT / "recording_index.json"
    dest.write_text(json.dumps(idx, indent=2, sort_keys=True, default=str))
    print(f"wrote {dest}: {len(eps)} episodes, conditions "
          f"{idx['conditions']}, seeds {idx['seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
