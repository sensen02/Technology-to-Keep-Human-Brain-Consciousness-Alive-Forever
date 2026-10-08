"""Compare old vs new ground using the AUDIT'S OWN instruments, so the numbers
are directly comparable to the published targets.

My earlier attempt used my own detector and produced numbers that were NOT
comparable to the audit's threshold (the detector's normalisation depends on
image size, and the audit calibrated it on 1024x900 fields).  This script
therefore imports `periodicity`, `label_peaks`, `radial_psd`, `loglog_slope`,
`render_free` and the band constants FROM the audit script, and renders the
ground with the audit's exact camera geometry (straight down, distance 36 mm,
fovy 35 deg, 1024x900).

Run:  MUJOCO_GL=egl venv_body/bin/python compare_ground_audit_instruments.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = "/run/media/sensen/Data2/cell_wound_prototype"
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

import scene_realism_audit_run as A  # noqa: E402
from engine.embodied.body_backend import BodyBackend, BodyConfig  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "embodied_body")
CAM_FOVY = getattr(A, "CAM_FOVY", 35.0)


def measure(preset):
    b, _ = A.build(preset)
    b.attach_cpg_baseline()
    for _ in range(400):
        b.step()
    th = b.observe().thorax_position_mm
    r = mujoco.Renderer(b.model, height=A.CAM_H, width=A.CAM_W)
    # exactly the audit's "in" camera, looking straight down at the ground
    img = A.render_free(b, r, lookat=[float(th[0]), float(th[1]), 0.0],
                        distance=36.0, fovy=CAM_FOVY)
    r.close()
    g = img.mean(axis=2).astype(float)
    # blank out the fly itself (middle of the frame) so we measure GROUND only
    h, w = g.shape
    g2 = g.copy()
    g2[int(h * 0.30):int(h * 0.72), int(w * 0.28):int(w * 0.72)] = np.nan
    gm = np.nanmean(g2)
    g2 = np.where(np.isnan(g2), gm, g2)          # fill, keep the spectrum clean

    per = A.periodicity(g2)
    peaks = A.label_peaks(per["top_peaks"], A.MM_PER_PX_IN)
    f, psd = A.radial_psd(g2, nbins=A.RADIAL_BINS)
    sl = A.loglog_slope(f, psd, A.FIT_BAND_CMM[0], A.FIT_BAND_CMM[1])

    m = b.model
    tex = [(str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i)),
            int(m.tex_type[i]), int(m.tex_width[i]), int(m.tex_height[i]))
           for i in range(m.ntex)]
    gmats = []
    for gi in range(m.ngeom):
        if int(m.geom_type[gi]) in (0, 1):
            mid = int(m.geom_matid[gi])
            nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi))
            mat = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid)) if mid >= 0 else None
            texids = ([int(v) for v in np.asarray(m.mat_texid[mid])] if mid >= 0 else [])
            gtex = None
            for role, ti in enumerate(texids):
                if ti >= 0:
                    gtex = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, ti))
                    break
            gmats.append(dict(geom=nm, matid=mid, material=mat, texture=gtex))

    eye = b.raw_vision()
    clip = [float(np.mean(eye[i] == 255)) for i in range(eye.shape[0])]
    return dict(preset=preset,
                textures=tex, ground_bindings=gmats,
                nlight=int(m.nlight), nhfield=int(getattr(m, "nhfield", 0)),
                ground_mean=float(g.mean()), ground_std=float(g.std()),
                ground_unique_colors=int(len(np.unique(img[int(h*0.30):int(h*0.72)]))),
                peak_over_median=float(per["peak_over_median"]),
                top_peaks=[dict(radial_wavelength_mm=p["radial_wavelength_mm"],
                                axis_wavelength_mm=p["axis_wavelength_mm"],
                                orientation_deg=p["orientation_deg"])
                           for p in peaks[:3]],
                fitted_slope=sl), img


def main():
    print("Using the AUDIT'S OWN instruments and camera geometry "
          f"(fovy={CAM_FOVY}, distance 36 mm, {A.CAM_H}x{A.CAM_W})")
    print(f"audit fit band = {A.FIT_BAND_CMM} cycles/mm; mm_per_px = {A.MM_PER_PX_IN:.5f}\n")
    res, imgs = {}, {}
    for p in ("blank", "natural_fbm"):
        try:
            r, img = measure(p)
        except Exception as exc:
            print(f"{p}: FAILED {type(exc).__name__}: {exc}")
            continue
        res[p], imgs[p] = r, img
        print(f"=== {p}")
        print(f"    textures: {r['textures']}")
        print(f"    ground binding: {r['ground_bindings']}")
        print(f"    nlight={r['nlight']} nhfield={r['nhfield']}")
        print(f"    ground mean={r['ground_mean']:.1f} std={r['ground_std']:.1f} "
              f"unique_colors={r['ground_unique_colors']}")
        print(f"    peak/median = {r['peak_over_median']:.1f}")
        for k, pk in enumerate(r["top_peaks"]):
            print(f"      peak{k}: axis_wavelength={pk['axis_wavelength_mm']:.3f} mm "
                  f"radial={pk['radial_wavelength_mm']:.3f} mm "
                  f"orient={pk['orientation_deg']:.1f} deg")
        print(f"    fitted slope {json.dumps(r['fitted_slope'])[:160]}")
        print()

    tgt = None
    try:
        tgt = json.load(open(os.path.join(OUT, "scene_realism_audit.json")))["target_for_a_natural_scene"]
    except Exception:
        pass
    if tgt:
        print("audit target peak/median =", tgt["periodicity_peak_over_median"]["value"])
        print("audit target slope band  =", tgt["fitted_radial_power_law_slope"]["range"],
              "r2 >=", tgt["fitted_radial_power_law_slope"]["r2_required"])

    if len(res) == 2:
        b, n = res["blank"], res["natural_fbm"]
        for k in ("peak_over_median", "ground_std", "ground_unique_colors"):
            print(f"  {k:22s} blank={b[k]:>12.3f}   natural_fbm={n[k]:>12.3f}   "
                  f"ratio={n[k]/b[k] if b[k] else float('nan'):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for i, p in enumerate(("blank", "natural_fbm")):
        if p in imgs:
            ax[i].imshow(imgs[p])
            ax[i].set_title(f"{p}: ground straight down (audit camera)", fontsize=9)
            ax[i].axis("off")
    fig.suptitle("Ground measured with the AUDIT'S OWN camera geometry "
                 "(1024x900, fovy 35, distance 36 mm)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(OUT, "compare_ground_audit_view.png")
    fig.savefig(png, dpi=100)
    plt.close(fig)
    with open(os.path.join(OUT, "compare_ground_audit_instruments.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
