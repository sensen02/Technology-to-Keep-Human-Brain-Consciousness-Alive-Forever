#!/usr/bin/env python3
"""Compare the analytic projection matrix with one recovered from the image.

THE DECISIVE TEST FOR A CALIBRATION BUG.  If the analytic P (built from the declared pose)
is right, then the P recovered from six known 3D points and their measured image positions
must be proportional to it.  A mismatch is reported per row, which localises the error:
  * rows 0 and 1 off in different ways -> extrinsic (R or t) wrong;
  * all rows off by the same scale in the third column -> focal length wrong;
  * a sign flip in row 0 or 1 -> one axis convention wrong.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_dlt_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tools_marker_discriminate import components, render  # noqa: E402
from run_arena_reconstruct import load_cameras  # noqa: E402


def main() -> int:
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    X = np.array([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    cams = load_cameras(geom)
    cam = cams[0]
    out = {}
    for radius, tag in ((0.30, "r0.30"), (0.60, "r0.60")):
        img, proj = render(radius, (1, 1, 1, 1))
        g = img.mean(axis=2)
        comps = [c for c in components(g >= 250.0) if c["area"] >= 8]
        # greedy nearest pairing, each marker used once
        pairs = []
        used = set()
        for c in sorted(comps, key=lambda c: -c["area"]):
            order = np.argsort([np.hypot(r - c["row"], co - c["col"]) for r, co in proj])
            for j in order:
                if int(j) not in used:
                    used.add(int(j))
                    pairs.append((int(j), c["row"], c["col"]))
                    break
        rows = []
        for k, (j, r, co) in enumerate(sorted(pairs)):
            if j < len(proj):
                rows.append((j, r, co, float(np.hypot(proj[j][0] - r, proj[j][1] - co))))
        rows.sort()
        n = len(rows)
        entry = {"n_pairs": n, "pairs": [{"marker": j, "det_row": r, "det_col": c,
                                          "dist_px": round(d, 2)}
                                         for j, r, c, d in rows]}
        if n >= 6 and len({j for j, _r, _c, _d in rows}) == 6:
            A = []
            for j, r, co, _d in rows:
                Xw = np.append(X[j], 1.0)
                A.append(np.concatenate([Xw, [0, 0, 0, 0], -co * Xw]))
                A.append(np.concatenate([[0, 0, 0, 0], Xw, -r * Xw]))
            A = np.asarray(A)
            _u, _s, vt = np.linalg.svd(A)
            P = vt[-1].reshape(3, 4)
            P = P / P[2, 3]
            entry["dlt_P"] = P.tolist()
            entry["analytic_P"] = cam.P.tolist()
            entry["row_ratios"] = [
                [None if abs(cam.P[i][k]) < 1e-9 else round(float(P[i][k] / cam.P[i][k]), 3)
                 for k in range(4)] for i in range(3)]
        out[tag] = entry
        print(f"--- {tag}: {n} components matched to projections ---")
        for j, r, c, d in rows:
            print(f"   mk{j} det({r:6.1f},{c:6.1f}) proj({proj[j][0]:6.1f},{proj[j][1]:6.1f})"
                  f"  d={d:5.1f} px")
        if "row_ratios" in entry:
            print("   P row ratios (dlt/analytic):")
            for i, row in enumerate(entry["row_ratios"]):
                print(f"     row{i}: {row}")
    dest = HERE / "outputs" / "arena" / "dlt_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
