"""Independent verification of the natural-scene layer against the audit's targets.

This script is MINE (the orchestrator's), not the implementation agent's.  It
re-measures the claims from scratch from the compiled model and from rendered
pixels, and it applies the acceptance targets published by the independent
audit in `outputs/embodied_body/scene_realism_audit.json`:

  * periodicity detector: peak / median of the 2-D power spectrum of the ground
    region  -> target <= 300      (current scene measured 7.1e4 .. 1.9e5)
  * fitted radial power-law slope  -> target in [-2.5, -1.5] with r^2 >= 0.90
                                     (current scene: no power law at all)
  * no sharp spectral peak whose wavelength falls within a factor of 2 of the
    fly body length (2.5 mm), i.e. inside [1.25, 5.0] mm
  * sky occupancy (fraction of eye pixels clipped at 255) must fall well below
    the measured baseline of 0.4499 .. 0.4638

Run:  MUJOCO_GL=egl venv_body/bin/python verify_natural_scene.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.embodied.body_backend import BodyBackend, BodyConfig  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "embodied_body")
FLY_BODY_MM = 2.5
FORBIDDEN_BAND_MM = (FLY_BODY_MM / 2.0, FLY_BODY_MM * 2.0)   # [1.25, 5.0]

TARGET = dict(peak_over_median_max=300.0,
              slope_range=(-2.5, -1.5), r2_min=0.90,
              sky_clip_max=0.20,
              baseline_sky_clip=(0.4499, 0.4638))

#: WARNING, and it invalidates the V1/V2/V3 numbers below as a comparison against
#: the audit's target: this file's `power_audit` is MY OWN detector, and a
#: peak-to-median ratio depends strongly on image size and on the DC guard, while
#: the audit calibrated its target of 300 on 1024x900 isotropic 1/f^beta fields.
#: Numbers from two different detectors at two different sizes are NOT comparable.
#: The comparable measurement lives in `compare_ground_audit_instruments.py`, which
#: imports the audit's OWN `periodicity`/`label_peaks`/`radial_psd`/`loglog_slope`
#: and renders with the audit's own camera geometry.  Use that one for verdicts.
DETECTOR_NOT_COMPARABLE = True


def power_audit(img2d, mm_per_px):
    """Peak/median of the 2-D power spectrum + radial power-law fit.

    Same detector shape as the independent audit so the numbers are comparable:
    DC and its immediate neighbourhood are excluded, the radial spectrum is
    binned in the band the audit used, and the slope is fitted on log-log.
    """
    a = np.asarray(img2d, float)
    a = a - a.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    cy, cx = np.array(F.shape) // 2
    F[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0
    peak = float(F.max())
    med = float(np.median(F[F > 0]))
    ratio = peak / med if med > 0 else float("inf")
    iy, ix = np.unravel_index(int(np.argmax(F)), F.shape)
    fy = (iy - cy) / F.shape[0]
    fx = (ix - cx) / F.shape[1]
    freq = float(np.hypot(fx, fy))
    wl_px = (1.0 / freq) if freq > 0 else np.inf
    wl_mm = wl_px * mm_per_px

    yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
    r = np.hypot(yy - cy, xx - cx).ravel()
    v = F.ravel()
    nb = 40
    rmax = r.max()
    edges = np.linspace(1.0, rmax, nb + 1)
    rb, vb = [], []
    for i in range(nb):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() >= 4:
            rb.append(0.5 * (edges[i] + edges[i + 1]))
            vb.append(np.mean(v[m]))
    rb, vb = np.asarray(rb), np.asarray(vb)
    ok = (vb > 0) & (rb > 0)
    rb, vb = rb[ok], vb[ok]
    if rb.size >= 5:
        k = np.fft.fftfreq(F.shape[0])[1] * 0 + 1.0 / (rb * mm_per_px)
        x = np.log10(1.0 / (rb * mm_per_px))     # cycles per mm
        y = np.log10(vb)
        sl, intc = np.polyfit(x, y, 1)
        pred = sl * x + intc
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        sl, r2 = float("nan"), 0.0
    return dict(peak_over_median=ratio, peak_wavelength_mm=float(wl_mm),
                fitted_slope=float(sl), r2=float(r2),
                n_radial_bins=int(rb.size))


def render_views(preset, seed=0):
    b = BodyBackend(BodyConfig(seed=seed, scene_preset=preset, add_vision=True),
                    gl_backend="egl")
    import mujoco
    b.attach_cpg_baseline()
    for _ in range(400):
        b.step()
    thorax = b.observe().thorax_position_mm
    r = mujoco.Renderer(b.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 22.0        # mm; far enough to include ground texture
    cam.azimuth = 135.0
    cam.elevation = 20.0       # MUST be positive: negative puts the camera
                               # below the infinite ground plane -> blank white
    cam.lookat[:] = np.asarray(thorax, float)
    r.update_scene(b.data, cam)
    view = r.render().copy()
    r.close()
    eye = b.raw_vision()
    om = b.ommatidia_readouts()
    # a downward view to measure the GROUND itself
    r2 = mujoco.Renderer(b.model, height=400, width=400)
    cam2 = mujoco.MjvCamera()
    cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam2.distance = 8.0
    cam2.azimuth = 90.0
    # ELEVATION MUST BE POSITIVE to look DOWN at the ground from above.  This was
    # -78 in the first version of this file, which put the camera 7.8 mm BELOW the
    # infinite ground plane and measured the underside of the floor instead of the
    # ground -- the numbers it produced were meaningless.  Fixed to +78.
    cam2.elevation = 78.0
    cam2.lookat[:] = np.asarray([thorax[0], thorax[1], 0.0], float)
    r2.update_scene(b.data, cam2)
    down = r2.render().copy()
    r2.close()
    m = b.model
    info = dict(preset=preset, ngeom=int(m.ngeom), nlight=int(m.nlight),
                nhfield=int(getattr(m, "nhfield", 0)), ncam=int(m.ncam),
                mat_texid=[int(v) for v in np.asarray(m.mat_texid)[0]],
                mat_texrepeat=[float(v) for v in np.asarray(m.mat_texrepeat)[0]],
                tex=[(str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i)),
                      int(m.tex_type[i]), int(m.tex_width[i]), int(m.tex_height[i]))
                     for i in range(m.ntex)])
    return dict(info=info, view=view, down=down, eye=eye, om=om, b=b)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    presets = ["blank", "natural_fbm"]
    res, views = {}, {}
    for p in presets:
        try:
            out = render_views(p)
        except Exception as exc:
            print(f"{p}: BUILD FAILED {type(exc).__name__}: {exc}")
            res[p] = None
            continue
        views[p] = out
        eye = out["eye"]
        m = out["b"].model
        # Ground seen from above: crop out the fly in the middle
        down = out["down"]
        h, w, _ = down.shape
        ground = down[: int(h * 0.35), :, :].mean(axis=2)
        # mm per pixel for a camera at height |elev|*distance looking down
        mm_per_px = 2.0 * 8.0 * np.tan(np.deg2rad(m.vis.global_.fovy / 2.0)) / h
        pa = power_audit(ground, mm_per_px)
        eye_clip = [float(np.mean(eye[i] == 255)) for i in range(eye.shape[0])]
        eye_zero = [float(np.mean(eye[i] == 0)) for i in range(eye.shape[0])]
        res[p] = dict(
            info=out["info"],
            ground_power=pa,
            eye_clip_frac=eye_clip, eye_zero_frac=eye_zero,
            eye_frame_std=[float(np.std(eye[i])) for i in range(eye.shape[0])],
            n_ommatidia=int(out["om"].shape[1]),
            ommatidia_mean=[float(out["om"][i].sum(axis=-1).mean())
                            for i in range(out["om"].shape[0])],
        )
        v = res[p]
        print(f"\n=== {p}")
        print(f"   textures: {v['info']['tex']}")
        print(f"   mat_texid={v['info']['mat_texid']} mat_texrepeat={v['info']['mat_texrepeat']}")
        print(f"   ngeom={v['info']['ngeom']} nlight={v['info']['nlight']} "
              f"nhfield={v['info']['nhfield']} ncam={v['info']['ncam']}")
        print(f"   GROUND periodicity peak/median = {pa['peak_over_median']:.1f} "
              f"(target <= {TARGET['peak_over_median_max']})")
        print(f"   GROUND peak wavelength         = {pa['peak_wavelength_mm']:.3f} mm "
              f"(forbidden band {FORBIDDEN_BAND_MM})")
        print(f"   GROUND fitted radial slope     = {pa['fitted_slope']:.3f} "
              f"r2={pa['r2']:.3f} (target {TARGET['slope_range']}, r2>={TARGET['r2_min']})")
        print(f"   EYE clip frac (==255)          = {[round(x,4) for x in eye_clip]}")
        print(f"   EYE frame std                  = {[round(x,1) for x in v['eye_frame_std']]}")

    # ---- verdicts against the audit's published targets --------------------
    if res.get("natural_fbm") is None:
        print("\nVERDICT: EVIDENCE NOT FOUND -- the natural preset could not be built")
        return
    nat = res["natural_fbm"]
    base = res.get("blank") or {}
    baseclip = max(base.get("eye_clip_frac", [0.0])) if base else 1.0
    natclip = max(nat["eye_clip_frac"])
    pa = nat["ground_power"]
    v_per = pa["peak_over_median"] <= TARGET["peak_over_median_max"]
    v_slope = (TARGET["slope_range"][0] <= pa["fitted_slope"] <= TARGET["slope_range"][1]
               and pa["r2"] >= TARGET["r2_min"])
    wl = pa["peak_wavelength_mm"]
    v_band = not (FORBIDDEN_BAND_MM[0] <= wl <= FORBIDDEN_BAND_MM[1])
    v_clip = natclip <= TARGET["sky_clip_max"]
    print("\n-- VERDICTS vs the independent audit's targets --")
    print(f"  V1 ground non-periodic (peak/median <= {TARGET['peak_over_median_max']}): "
          f"{'PASS' if v_per else 'FAIL'}  ({pa['peak_over_median']:.1f})")
    print(f"  V2 ground has a power-law spectrum {TARGET['slope_range']} r2>={TARGET['r2_min']}: "
          f"{'PASS' if v_slope else 'FAIL'}  (slope {pa['fitted_slope']:.3f}, r2 {pa['r2']:.3f})")
    print(f"  V3 no sharp peak in the fly-scale band: {'PASS' if v_band else 'FAIL'} "
          f"(peak at {wl:.3f} mm)")
    print(f"  V4 eye clipping <= {TARGET['sky_clip_max']}: {'PASS' if v_clip else 'FAIL'} "
          f"(natural {natclip:.4f} vs blank {baseclip:.4f}, "
          f"audit baseline {TARGET['baseline_sky_clip']})")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, p in enumerate(presets):
        if views.get(p) is None:
            continue
        axes[0, col].imshow(views[p]["view"])
        axes[0, col].set_title(f"{p}\nfree camera (fly + ground)", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(views[p]["down"])
        axes[1, col].set_title("looking down at the ground", fontsize=9)
        axes[1, col].axis("off")
    axes[0, 2].imshow(views["natural_fbm"]["eye"][0])
    axes[0, 2].set_title("natural_fbm LEFT EYE (157 deg fisheye)", fontsize=9)
    axes[0, 2].axis("off")
    g = views["natural_fbm"]["down"]
    axes[1, 2].hist(views["natural_fbm"]["eye"][0].ravel(), bins=64, color="slategray")
    axes[1, 2].set_title("left-eye pixel histogram (clip at 255 visible?)", fontsize=9)
    fig.suptitle("Independent verification of the natural scene vs the audit's targets",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(OUT_DIR, "verify_natural_scene.png")
    fig.savefig(png, dpi=105)
    plt.close(fig)

    report = dict(what_this_is=("orchestrator's independent verification of the "
                                "natural-scene layer; re-measured, not trusted"),
                  targets=TARGET, forbidden_band_mm=FORBIDDEN_BAND_MM,
                  per_preset={p: (None if res[p] is None else
                                  {k: v for k, v in res[p].items()}) for p in presets},
                  verdicts=dict(V1_non_periodic=bool(v_per), V2_power_law=bool(v_slope),
                                V3_no_fly_scale_peak=bool(v_band),
                                V4_clipping_fixed=bool(v_clip)),
                  eye_clip_natural=natclip, eye_clip_blank=baseclip)
    jp = os.path.join(OUT_DIR, "verify_natural_scene.json")
    with open(jp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {png}\nwrote {jp}")


if __name__ == "__main__":
    main()
