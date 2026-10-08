"""Vision quicklook: do the fly's own eyes see anything, and does an object change it?

WHY THIS EXISTS
---------------
The embodied closed loop had NO visual pathway: the body was built without
``add_vision()``, so the compiled model had no eye cameras at all and
``Simulation.get_raw_vision()`` raised.  That is a capability gap, not a
numerical result, and it is why the behaviour acceptance task "notice and
respond to a salient object" is recorded as NOT RUN.

This script is a SELF-TEST / QUICKLOOK, hand-built for inspection.  It is not a
finished visual system.  It answers three narrow, checkable questions:

  Q1  Can the eye cameras produce a non-blank image at all?
  Q2  Does putting an object in front of the fly change what the eyes see?
  Q3  Is the CHANGE temporal (does the reading change over time), or is the
      scene merely spatially rich but temporally frozen?  A featureless scene
      fails in the SECOND way: spatial contrast can be large while temporal
      contrast is ~0, so nothing motion-like can be encoded.

WHAT THIS CANNOT SHOW
---------------------
FlyGym's own docstring states its fisheye ``Retina`` is calibrated for
FlyGym's OWN eye placement, so ommatidia readouts on the NeuroMechFly body are
APPROXIMATE optics -- not measured Drosophila optics.  There is also no
Drosophila visual-response reference dataset in this project, so nothing here
can say whether a response is "normal" or "abnormal".  Only
"degenerate vs informative" is measurable, and that is what is measured.

Run with:  MUJOCO_GL=egl venv_body/bin/python run_vision_quicklook.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.embodied.body_backend import BodyBackend, BodyConfig  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "embodied_body")

#: Object placed in front of the fly.  Sizes are in MILLIMETRES (model units).
#: The fly is ~2.5 mm long and ~1 mm tall, so a 1 mm-diameter sphere is a
#: prominent object at fly scale, not a landscape feature.
OBJECT_POS_MM = (4.0, 0.0, 0.5)
OBJECT_RADIUS_MM = 0.5

N_STEPS_SETTLE = 300          # 30 ms at dt = 1e-4 s
N_SAMPLES = 50                # number of visual samples
SAMPLE_EVERY = 200            # steps between samples -> 20 ms apart
DT_S = 1e-4


def look_at_xyaxes(pos, target):
    """MuJoCo wants a camera frame: X and Y axes; it looks along -Z (Z = X x Y)."""
    pos = np.asarray(pos, float)
    fwd = np.asarray(target, float) - pos
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return [float(v) for v in np.concatenate([x, y])]


@dataclass
class Condition:
    name: str
    with_object: bool
    blind: bool = False


def run_condition(cond: Condition, seed: int = 0, view_camera: bool = True):
    """Build one backend and record the visual readout over time.

    The object is supplied through BodyConfig.extra_geoms so it is created by
    BodyBackend itself, inside the normal construction path (cameras first,
    object before compile).  An earlier version of this script reconstructed a
    backend by hand, which broke on an internal attribute -- the config knob
    exists precisely so scene content never requires bypassing __init__.
    """
    cam_pos = (3.0, -9.0, 5.0)
    cam_xyaxes = tuple(look_at_xyaxes(cam_pos, (2.0, 0.0, 0.5)))
    geoms = ()
    if cond.with_object:
        geoms = (dict(name="target_object", type="sphere",
                      size=[OBJECT_RADIUS_MM, 0.0, 0.0],
                      pos=[float(v) for v in OBJECT_POS_MM],
                      rgba=[0.05, 0.05, 0.05, 1.0],
                      contype=0, conaffinity=0),)
    cfg = BodyConfig(add_vision=True, seed=seed, extra_geoms=geoms,
                     add_world_camera=view_camera, world_camera_name="viewcam",
                     world_camera_pos_mm=cam_pos,
                     world_camera_fovy=40.0,
                     world_camera_xyaxes=cam_xyaxes)
    backend = BodyBackend(cfg, gl_backend="egl")
    backend.attach_cpg_baseline()
    for _ in range(N_STEPS_SETTLE):
        backend.step()

    world_frames = []
    if view_camera:
        # FREE TRACKING camera, not a fixed model camera.  Measured bug: the fly
        # walks ~14 mm/s, so in the ~1.3 s recorded here it travels ~18 mm and
        # LEAVES a fixed camera's ~7 mm frame entirely -- the first version of this
        # figure rendered a blank white ground plane for exactly that reason.  The
        # camera therefore follows the thorax.
        #
        # It is sampled EARLY as well as late: by the last frame the fly has walked
        # well past the object at 4 mm, so a late-only view shows the fly and no
        # object and the two conditions render hundreds of pixels apart for reasons
        # that have nothing to do with vision.
        import mujoco
        _renderer = mujoco.Renderer(backend.model, height=360, width=480)

        def _render_world():
            thorax_mm = backend.observe().thorax_position_mm
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 6.5           # mm, model units
            cam.azimuth = 135.0
            # ELEVATION MUST BE POSITIVE.  Measured bug: elevation -22 deg with
            # distance 4.5 mm puts the camera at z = 0.5 - 4.5*sin(22 deg) = -1.2
            # mm, i.e. UNDERNEATH the infinite ground plane, so the whole frame is
            # the white underside of the floor and the fly is invisible.
            cam.elevation = 25.0
            cam.lookat[:] = np.asarray(thorax_mm, float)
            _renderer.update_scene(backend.data, cam)
            return _renderer.render().copy()

        world_frames.append(_render_world())

    frames, readouts, times = [], [], []
    for _ in range(N_SAMPLES):
        for _ in range(SAMPLE_EVERY):
            backend.step()
        frames.append(backend.raw_vision())
        readouts.append(backend.ommatidia_readouts())
        times.append(float(backend.observe().time_s))
    if view_camera:
        world_frames.append(_render_world())
        _renderer.close()

    return dict(name=cond.name, times_s=np.asarray(times),
                frames=np.asarray(frames), readouts=np.asarray(readouts),
                world_frames=world_frames,
                world_frame=world_frames[0] if world_frames else None,
                describe=backend.describe())


def temporal_statistics(readouts, frames=None):
    """The crux: TEMPORAL variation vs SPATIAL variation.

    A featureless scene can still be spatially rich (the fly's own legs, the
    near ground under the camera headlight) while being TEMPORALLY frozen.
    ``spatial_contrast`` here is the spread ACROSS ommatidia within one sample,
    averaged over time; ``temporal_std`` is the spread over TIME within one
    ommatidium, averaged over ommatidia.  They answer different questions and
    must not be conflated.

    ``frames`` is the RAW uint8 eye rendering.  It is reported separately and
    per eye, because an earlier version of this script named the ommatidia
    readout spread "frame_std" and then tested it against a frame-scale
    threshold -- the Q1 blank-frame test failed for the wrong reason.  The raw
    frame and the ommatidia readout live on different scales (uint8 vs ~0.6) and
    must never share a threshold.
    """
    r = np.asarray(readouts, float)              # (T, 2, n_ommatidia, 2)
    T = r.shape[0]
    # Collapse the mutually exclusive yellow/pale channels by summing.
    lum = r.sum(axis=3)                          # (T, 2, n)
    temporal_std = lum.std(axis=0)               # (2, n) over time
    spatial_contrast = lum.std(axis=2)           # (T, 2) across ommatidia
    per_eye_mean = lum.mean(axis=2)              # (T, 2)
    out = dict(
        n_samples=int(T),
        n_ommatidia=int(lum.shape[2]),
        temporal_std_mean_per_eye=[float(temporal_std[i].mean())
                                   for i in range(lum.shape[1])],
        temporal_std_median_per_eye=[float(np.median(temporal_std[i]))
                                     for i in range(lum.shape[1])],
        temporal_std_max=float(temporal_std.max()),
        spatial_contrast_mean_per_eye=[float(spatial_contrast[:, i].mean())
                                       for i in range(lum.shape[1])],
        mean_luminance_per_eye=[float(per_eye_mean[:, i].mean())
                                for i in range(lum.shape[1])],
        mean_luminance_range_per_eye=[(float(per_eye_mean[:, i].min()),
                                       float(per_eye_mean[:, i].max()))
                                      for i in range(lum.shape[1])],
        ommatidia_readout_std_per_eye=[float(np.std(r[0, i]))
                                       for i in range(lum.shape[1])],
    )
    if frames is not None:
        fr = np.asarray(frames)
        out["raw_frame_std_per_eye"] = [float(np.std(fr[0, i]))
                                        for i in range(fr.shape[1])]
        out["raw_frame_mean_per_eye"] = [float(np.mean(fr[0, i]))
                                         for i in range(fr.shape[1])]
    return out


def object_effect(a, b):
    """Per-eye effect of the object, compared at MATCHED sample times.

    The two conditions are separate simulations with the same seed and an object
    that cannot touch the fly, so any difference between them is caused by
    rendering alone.  Reported PER EYE on purpose: an earlier aggregate
    (averaging the two eyes) hid the fact that only one eye responds.
    """
    ra = np.asarray(a, float).sum(axis=3)   # (T, 2, n)
    rb = np.asarray(b, float).sum(axis=3)
    n = min(ra.shape[0], rb.shape[0])
    ra, rb = ra[:n], rb[:n]
    out = {"n_compared_samples": int(n), "per_eye": {}}
    for i, lab in enumerate(("left", "right")):
        d_mean = rb[:, i, :].mean(axis=1) - ra[:, i, :].mean(axis=1)
        d_pix = rb[:, i, :] - ra[:, i, :]
        identical = bool(np.array_equal(ra[:, i, :], rb[:, i, :]))
        out["per_eye"][lab] = dict(
            readout_bit_identical=identical,
            max_abs_delta_mean_luminance=float(np.abs(d_mean).max()),
            max_abs_delta_per_ommatidium=float(np.abs(d_pix).max()),
            frac_ommatidia_changed_any=float(np.mean(
                np.abs(d_pix).max(axis=0) > 0.0)),
        )
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("VISION QUICKLOOK -- hand-built self-test, NOT a finished visual system")
    print("=" * 78)

    blank = run_condition(Condition("no_object", with_object=False),
                          view_camera=True)
    withobj = run_condition(Condition("with_object", with_object=True),
                            view_camera=True)

    stats_blank = temporal_statistics(blank["readouts"], blank["frames"])
    stats_obj = temporal_statistics(withobj["readouts"], withobj["frames"])
    effect = object_effect(blank["readouts"], withobj["readouts"])

    print("\n-- Q1: are the RAW eye frames blank? (uint8, scale 0-255) --")
    print("  no_object   raw frame std per eye:", [round(v, 2) for v in
                                                   stats_blank["raw_frame_std_per_eye"]])
    print("  with_object raw frame std per eye:", [round(v, 2) for v in
                                                   stats_obj["raw_frame_std_per_eye"]])
    print("  (the ommatidia readout lives on a DIFFERENT scale -- ~0.4, not ~100 --")
    print("   so the two are reported separately and never share a threshold)")
    nonblank = all(v > 1.0 for v in stats_blank["raw_frame_std_per_eye"]) and \
        all(v > 1.0 for v in stats_obj["raw_frame_std_per_eye"])
    print(f"  Q1 non-blank raw frames: {'PASS' if nonblank else 'FAIL'}")

    print("\n-- Q2: does the object change the readout? (PER EYE, matched times) --")
    print("  the object has contype=conaffinity=0, so it cannot touch the fly:")
    print("  every difference below is caused by RENDERING alone")
    for lab, d in effect["per_eye"].items():
        print(f"    {lab} eye: bit-identical={d['readout_bit_identical']}  "
              f"max|delta mean luminance|={d['max_abs_delta_mean_luminance']:.6f}  "
              f"max|delta per ommatidium|={d['max_abs_delta_per_ommatidium']:.6f}  "
              f"frac ommatidia changed={d['frac_ommatidia_changed_any']:.4f}")
    n_eyes_responding = sum(1 for d in effect["per_eye"].values()
                            if not d["readout_bit_identical"])
    print(f"  Q2 eyes that respond to the object: {n_eyes_responding} of 2")
    if n_eyes_responding == 1:
        print("  NOTE, reported rather than averaged away: exactly one eye changed.")
        print("  A bit-identical eye is NOT a bug -- it means the object lies")
        print("  entirely OUTSIDE that eye's 157 deg field of view.  Averaging the")
        print("  two eyes would have hidden this.")

    print("\n-- Q3: temporal vs spatial variation in the ommatidia readout --")
    for nm, st in (("no_object", stats_blank), ("with_object", stats_obj)):
        print(f"  {nm}:")
        print(f"    temporal std (median/eye) = "
              f"{[round(v, 6) for v in st['temporal_std_median_per_eye']]}")
        print(f"    spatial contrast (mean/eye) = "
              f"{[round(v, 4) for v in st['spatial_contrast_mean_per_eye']]}")
        print(f"    mean luminance /eye = "
              f"{[round(v, 4) for v in st['mean_luminance_per_eye']]}")

    tb_eye = stats_blank["temporal_std_median_per_eye"]
    to_eye = stats_obj["temporal_std_median_per_eye"]
    print(f"\n  temporal std per eye  no_object  = {[round(v,6) for v in tb_eye]}")
    print(f"  temporal std per eye  with_obj   = {[round(v,6) for v in to_eye]}")
    q3 = all(o > b for o, b in zip(to_eye, tb_eye))
    q3_any = any(o > b for o, b in zip(to_eye, tb_eye))
    print(f"  Q3 object increases temporal variation, BOTH eyes: "
          f"{'PASS' if q3 else 'FAIL'}")
    print(f"  Q3 object increases temporal variation, AT LEAST ONE eye: "
          f"{'PASS' if q3_any else 'FAIL'}")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15, 9))
    for col, (rec, title) in enumerate(((blank, "scene: no object"),
                                        (withobj, "scene: 1 mm dark object at 4 mm"))):
        ax = fig.add_subplot(4, 2, 1 + col * 1)
        ax.imshow(rec["frames"][-1][0])
        ax.set_title(f"left eye raw (fisheye) -- {title}", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(4, 2, 3 + col * 1)
        ax.imshow(rec["frames"][-1][1])
        ax.set_title("right eye raw (fisheye)", fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(4, 2, 5 + col * 1)
        wf = rec.get("world_frames") or []
        if wf:
            ax.imshow(wf[0])
            ax.set_title(f"free camera (t={rec['times_s'][0] - 0.02:.2f} s, "
                         "early: fly still near the object)", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no world camera", ha="center")
        ax.axis("off")

    ax = fig.add_subplot(4, 2, 7)
    for rec, lab, c in ((blank, "no object", "tab:blue"),
                        (withobj, "with object", "tab:red")):
        lum = np.asarray(rec["readouts"], float).sum(axis=3)
        ax.plot(rec["times_s"], lum.mean(axis=(1, 2)), color=c, label=lab)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean luminance per ommatidium")
    ax.set_title("temporal trace of the visual readout", fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(4, 2, 8)
    for rec, lab, c in ((blank, "no object", "tab:blue"),
                        (withobj, "with object", "tab:red")):
        lum = np.asarray(rec["readouts"], float).sum(axis=3)
        ts = lum.std(axis=0).mean(axis=0)
        ax.hist(ts, bins=40, alpha=0.6, color=c, label=lab)
    ax.set_xlabel("per-ommatidium temporal std")
    ax.set_ylabel("count")
    ax.set_title("distribution of temporal variability", fontsize=9)
    ax.legend(fontsize=8)
    fig.suptitle("FlyGym eye cameras on the embodied fly -- APPROXIMATE optics "
                 "(Retina calibrated for FlyGym's own eye placement; 157 deg fovy, "
                 "721 ommatidia/eye)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(OUT_DIR, "vision_quicklook.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)

    report = dict(
        what_this_is=("hand-built quicklook/self-test of the eye-camera path, not "
                      "a finished visual system and not a biology result"),
        model=dict(n_cameras=int(blank["describe"]["vision"]["n_cameras_total"]),
                   eye_cameras=blank["describe"]["vision"]["eye_cameras"],
                   eye_fovy_deg_measured=blank["describe"]["vision"]["eye_fovy_deg_measured"],
                   n_ommatidia_per_eye=int(stats_blank["n_ommatidia"]),
                   nlight=int(blank["describe"]["scene"]["nlight"])),
        object_spec=dict(pos_mm=list(OBJECT_POS_MM),
                         radius_mm=OBJECT_RADIUS_MM,
                         note=("contype = conaffinity = 0, so it CANNOT touch the "
                               "fly: any difference is visual, not mechanical")),
        conditions=dict(no_object=stats_blank, with_object=stats_obj),
        object_effect=effect,
        world_frame_std_per_condition={
            rec["name"]: [None if f is None else float(np.std(f))
                          for f in rec.get("world_frames", [])]
            for rec in (blank, withobj)},
        camera_note=("free camera: distance 6.5 mm, azimuth 135 deg, elevation +25 "
                     "deg, lookat = thorax each frame.  Elevation must be positive: "
                     "a negative elevation puts the camera below the infinite "
                     "ground plane and the frame becomes blank white."),
        answers=dict(Q1_raw_frames_nonblank=bool(nonblank),
                     Q2_eyes_responding=int(n_eyes_responding),
                     Q3_object_increases_temporal_variation_both_eyes=bool(q3),
                     Q3_object_increases_temporal_variation_any_eye=bool(q3_any),
                     temporal_std_per_eye_no_object=tb_eye,
                     temporal_std_per_eye_with_object=to_eye),
        cannot_show=["no Drosophila visual-response reference dataset exists here, "
                     "so 'normal vs abnormal' is NOT measurable; only "
                     "'degenerate vs informative' is",
                     "FlyGym's Retina is calibrated for its own eye placement, so "
                     "these ommatidia readouts are approximate optics",
                     "the scene has nlight = 0, so the image is lit only by the "
                     "camera headlight -- there is no directional illumination, "
                     "shading or shadow cue"],
    )
    jp = os.path.join(OUT_DIR, "vision_quicklook.json")
    with open(jp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {png}")
    print(f"wrote {jp}")
    return report


if __name__ == "__main__":
    main()
