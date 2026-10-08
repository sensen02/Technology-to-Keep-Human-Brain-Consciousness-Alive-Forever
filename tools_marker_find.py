#!/usr/bin/env python3
"""WHERE do the markers actually land in the image?  A visual + numeric answer.

The discriminator test found only ONE component near a projection out of six, which means
either the markers are hidden, or they are not where the camera model says.  Those are
different problems, so this tool renders a big frame, marks every pixel whose CHROMA is
strong, and prints their positions next to the projections -- then SAVES an image so the
answer can be seen instead of inferred.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_marker_find.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tools_marker_discriminate import components, render  # noqa: E402


def main() -> int:
    out = HERE / "outputs" / "arena" / "marker_find"
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw
    img, proj = render(0.30, (1, 1, 1, 1))
    g = img.mean(axis=2)
    print("projections:", [(round(float(r), 1), round(float(c), 1)) for r, c in proj])
    for name, level in (("ge250", 250.0), ("ge240", 240.0)):
        comps = components(g >= level)
        print(f"{name}: {len(comps)} components; the 10 nearest to any projection:")
        scored = sorted(comps, key=lambda c: min(
            np.hypot(proj[:, 0] - c["row"], proj[:, 1] - c["col"])))
        for c in scored[:10]:
            d = min(np.hypot(proj[:, 0] - c["row"], proj[:, 1] - c["col"]))
            print(f"   ({c['row']:6.1f},{c['col']:6.1f}) area {c['area']:3d} "
                  f"fill {c['fill']:.2f}  dist_to_nearest_proj {d:5.1f}")
    # draw every bright component and every projection
    im = Image.fromarray(img.astype("uint8"))
    dr = ImageDraw.Draw(im)
    for c in components(g >= 250.0):
        dr.ellipse([c["col"] - 8, c["row"] - 8, c["col"] + 8, c["row"] + 8],
                   outline=(255, 0, 255), width=1)
    for k, (r, col) in enumerate(proj):
        dr.ellipse([col - 5, r - 5, col + 5, r + 5], outline=(255, 255, 0), width=1)
        dr.text((col + 6, r - 4), str(k), fill=(0, 0, 0))
    dest = out / "marker_find.png"
    im.save(dest)
    print("wrote", dest)
    (out / "marker_find.json").write_text(json.dumps(
        {"projections": [[float(r), float(c)] for r, c in proj],
         "components_ge250": components(g >= 250.0)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
