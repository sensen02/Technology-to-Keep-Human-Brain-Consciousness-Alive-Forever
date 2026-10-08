#!/usr/bin/env python3
"""Per-marker VISIBILITY map for an episode: how many cameras see each marker, per frame.

WHY THIS IS THE METRIC THAT MATTERS.  A joint angle needs BOTH of its endpoints seen by enough
cameras to triangulate.  Identity gates, detector thresholds and calibration are all downstream
of that one number, so the pipeline reports it directly instead of only its consequences.

Run:
    OPENBLAS_NUM_THREADS=1 venv/bin/python tools_marker_coverage.py outputs/arena/arena_bare_seed2
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_arena_reconstruct import load_cameras_for_episode, load_projection_offsets  # noqa: E402

ARENA = HERE / "outputs" / "arena"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


def coverage(ep: Path, gate_px: float = 1.5):
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    load_projection_offsets(ep / "camera_offsets.json")
    cams, _ = load_cameras_for_episode(geom, ep, ARENA / "camera_poses.json")
    segs = list(geom["marker_body_segments"])
    z = np.load(ep / "episode.npz")
    FT = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)
    F = FT.shape[0]
    vis = np.zeros((len(segs), F), dtype=int)
    for f in range(F):
        for cam in cams:
            a = int(z[f"px/{cam.name}/offsets"][f]); b = int(z[f"px/{cam.name}/offsets"][f + 1])
            R = np.asarray(z[f"px/{cam.name}/rows"][a:b], dtype=float)
            C = np.asarray(z[f"px/{cam.name}/cols"][a:b], dtype=float)
            if R.size == 0:
                continue
            pr = cam.project(FT[f])
            d = np.hypot(R[None, :] - pr[:, 0:1], C[None, :] - pr[:, 1:2])
            vis[d.min(axis=1) <= gate_px, f] += 1
    return segs, vis


def main() -> int:
    eps = [Path(x) for x in sys.argv[1:]] or [ARENA / "arena_bare_seed2"]
    tables = {}
    for ep in eps:
        if not (ep / "episode.npz").exists():
            print("missing", ep)
            continue
        segs, vis = coverage(ep)
        tables[ep.name] = (segs, vis)
    names = list(tables)
    print(f"{'segment':<24}" + "".join(f"{n:>26}" for n in names))
    print(f"{'':<24}" + "".join(f"{'mean cams  >=3views':>26}" for _ in names))
    rows = []
    for i, s in enumerate(tables[names[0]][0]):
        cells = []
        for n in names:
            m = tables[n][1][i]
            cells.append(f"{m.mean():>13.2f}{(m >= 3).mean() * 100:>12.0f}%")
        rows.append((s, cells))
    for s, cells in sorted(rows, key=lambda r: -float(r[1][0].split()[0])):
        print(f"{s:<24}" + "".join(f"{c:>26}" for c in cells))
    print()
    for n in names:
        segs, vis = tables[n]
        idx = {x: i for i, x in enumerate(segs)}
        usable = 0
        for leg in LEGS:
            for bone, (a, b) in (("femur", (f"{leg}_trochanterfemur", f"{leg}_tibia")),
                                 ("tibia", (f"{leg}_tibia", f"{leg}_tarsus5"))):
                ok = ((vis[idx[a]] >= 3) & (vis[idx[b]] >= 3)).mean()
                if ok >= 0.5:
                    usable += 1
        print(f"{n}: {usable} of 12 bones have BOTH endpoints in >=3 cameras for >=50% of frames")
    out = ARENA / "marker_coverage.json"
    out.write_text(json.dumps({n: {s: float((tables[n][1][i] >= 3).mean())
                                   for i, s in enumerate(tables[n][0])}
                               for n in names}, indent=2, sort_keys=True))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
