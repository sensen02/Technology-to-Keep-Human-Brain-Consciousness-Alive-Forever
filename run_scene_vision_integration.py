"""Scene x vision integration: does a NATURAL scene actually change what the eyes read?

WHY THIS EXISTS
---------------
``run_scene_selftest.py`` reported one honest ``EVIDENCE NOT FOUND``:

    "visual pathway: the scene objects are actually visible to the fly |
     the compiled model has ncam=0 and make_locomotion_fly never calls
     add_vision(), so there are no eye cameras and no retina"

That gap is closable now, because ``BodyBackend`` gained ``add_vision``.  This
script builds the SAME scene presets through ``BodyBackend`` WITH eye cameras
enabled and measures the visual readout under each.

It answers the user's actual question in its checkable form: the complaint was
that a featureless scene is degenerate for vision.  The measurable version of
that complaint is:

    does the visual readout become more informative (more TEMPORAL variation,
    more usable contrast) in a natural scene than in the featureless one?

WHAT THIS IS NOT
----------------
Hand-built integration test, not a finished visual system and not a biology
result.  There is NO Drosophila visual-response reference dataset in this
project, so "normal vs abnormal" is NOT measurable here -- only
"degenerate vs informative".  The eye optics are FlyGym's, which its own
docstring says are calibrated for FlyGym's OWN eye placement and are therefore
APPROXIMATE on the NeuroMechFly body.  Nothing here has been connected to the
connectome tier yet, so the attention acceptance task remains NOT RUN.

Run with:  MUJOCO_GL=egl venv_body/bin/python run_scene_vision_integration.py
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

PRESETS = ("blank", "natural_lit_no_objects", "natural")
N_SETTLE = 400
N_SAMPLES = 30
SAMPLE_EVERY = 200      # 20 ms apart
DT_S = 1e-4

#: PRE-REGISTERED before looking at the data.  A featureless scene still has
#: spatial structure (the fly's own legs under the camera headlight), so the
#: discriminating quantity is TEMPORAL variation, not spatial contrast.  The
#: threshold is deliberately a ratio rather than an absolute: it asks whether
#: the natural scene changes the readout over time substantially MORE than the
#: featureless control, on the same fly, same seed, same gait.
PRE_REGISTERED_RATIO = 1.10   # natural / blank temporal std must exceed this


def record(preset, seed=0):
    b = BodyBackend(BodyConfig(seed=seed, scene_preset=preset, add_vision=True),
                    gl_backend="egl")
    d = b.describe()
    b.attach_cpg_baseline()
    for _ in range(N_SETTLE):
        b.step()
    readouts, frames, times, legs = [], [], [], []
    for _ in range(N_SAMPLES):
        for _ in range(SAMPLE_EVERY):
            b.step()
        readouts.append(b.ommatidia_readouts())
        frames.append(b.raw_vision())
        obs = b.observe()
        times.append(float(obs.time_s))
        legs.append(int(np.sum(obs.contact_present)))
    info = dict(
        preset=preset,
        ngeom=int(b.model.ngeom), nlight=int(b.model.nlight),
        nhfield=int(getattr(b.model, "nhfield", 0)), npair=int(b.model.npair),
        ncam=int(b.model.ncam),
        scene=d["scene"]["applied"] if d["scene"]["applied"] else None,
        times_s=[float(t) for t in times],
        mean_legs_in_contact=float(np.mean(legs)),
        walk_mm=float(np.linalg.norm(b.observe().thorax_position_mm[:2])),
        readouts=np.asarray(readouts), frames=np.asarray(frames),
    )
    return info


def stats(rec):
    r = np.asarray(rec["readouts"], float)      # (T, 2, n, 2)
    lum = r.sum(axis=3)                          # (T, 2, n)
    temporal = lum.std(axis=0)                   # (2, n)
    spatial = lum.std(axis=2)                    # (T, 2)
    fr = np.asarray(rec["frames"], float)
    return dict(
        n_ommatidia=int(lum.shape[2]),
        temporal_std_median_per_eye=[float(np.median(temporal[i])) for i in range(2)],
        temporal_std_max=float(temporal.max()),
        spatial_contrast_mean_per_eye=[float(spatial[:, i].mean()) for i in range(2)],
        mean_luminance_per_eye=[float(lum[:, i, :].mean()) for i in range(2)],
        raw_frame_std_per_eye=[float(fr[0, i].std()) for i in range(2)],
        raw_frame_mean_per_eye=[float(fr[0, i].mean()) for i in range(2)],
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("SCENE x VISION INTEGRATION -- hand-built test, not a finished system")
    print("=" * 78)

    recs, sts = {}, {}
    for p in PRESETS:
        try:
            rec = record(p)
        except Exception as exc:
            print(f"  preset {p}: BUILD FAILED -- {type(exc).__name__}: {exc}")
            recs[p] = None
            continue
        recs[p] = rec
        sts[p] = stats(rec)
        s = sts[p]
        print(f"\n-- preset {p}")
        print(f"   ngeom={rec['ngeom']} nlight={rec['nlight']} "
              f"nhfield={rec['nhfield']} npair={rec['npair']} ncam={rec['ncam']}")
        print(f"   raw eye frame std  = {[round(v,1) for v in s['raw_frame_std_per_eye']]}"
              f"  mean = {[round(v,1) for v in s['raw_frame_mean_per_eye']]}")
        print(f"   temporal std median/eye = "
              f"{[round(v,6) for v in s['temporal_std_median_per_eye']]}")
        print(f"   spatial contrast /eye   = "
              f"{[round(v,4) for v in s['spatial_contrast_mean_per_eye']]}")
        print(f"   mean luminance /eye     = "
              f"{[round(v,4) for v in s['mean_luminance_per_eye']]}")
        print(f"   mean legs in contact = {rec['mean_legs_in_contact']:.3f}")

    ok = all(recs.get(p) is not None for p in PRESETS)
    if not ok:
        print("\nVERDICT: EVIDENCE NOT FOUND -- a preset failed to build")
        return

    # ---- pre-registered predictions -------------------------------------
    def med(p):
        return float(np.mean(sts[p]["temporal_std_median_per_eye"]))

    blank_t, lit_t, nat_t = med("blank"), med("natural_lit_no_objects"), med("natural")

    print("\n-- PRE-REGISTERED PREDICTIONS --")
    p1 = nat_t > blank_t * PRE_REGISTERED_RATIO
    print(f"  P1 natural temporal std > {PRE_REGISTERED_RATIO}x blank: "
          f"{'PASS' if p1 else 'FAIL'}   ({nat_t:.6f} vs {blank_t:.6f}, "
          f"ratio {nat_t / blank_t if blank_t else float('nan'):.3f})")
    p2 = lit_t > blank_t * PRE_REGISTERED_RATIO
    print(f"  P2 light alone (no objects) also raises it: "
          f"{'PASS' if p2 else 'FAIL'}   ({lit_t:.6f} vs {blank_t:.6f}, "
          f"ratio {lit_t / blank_t if blank_t else float('nan'):.3f})")
    nonblank = all(v > 1.0 for p in PRESETS
                   for v in sts[p]["raw_frame_std_per_eye"])
    print(f"  P3 every condition renders a non-blank frame: "
          f"{'PASS' if nonblank else 'FAIL'}   "
          f"(std per preset: "
          f"{[round(min(sts[p]['raw_frame_std_per_eye']), 2) for p in PRESETS]})")
    obj_visible = (recs["natural"]["readouts"].mean() !=
                   recs["natural_lit_no_objects"]["readouts"].mean())
    print(f"  P4 the scene OBJECTS change the readout beyond the lighting change: "
          f"{'PASS' if obj_visible else 'FAIL'}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(14, 11))
    for col, p in enumerate(PRESETS):
        axes[0, col].imshow(recs[p]["frames"][-1][0])
        axes[0, col].set_title(f"{p}\nleft eye raw (fisheye)", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(recs[p]["frames"][-1][1])
        axes[1, col].set_title("right eye raw", fontsize=9)
        axes[1, col].axis("off")
        lum = np.asarray(recs[p]["readouts"], float).sum(axis=3)
        for eye, c in ((0, "tab:blue"), (1, "tab:red")):
            axes[2, col].plot(recs[p]["times_s"], lum[:, eye, :].mean(axis=1),
                              color=c, label=f"eye {eye}")
        axes[2, col].set_title("mean luminance over time", fontsize=9)
        axes[2, col].set_xlabel("time (s)")
        axes[2, col].legend(fontsize=7)
        tstd = lum.std(axis=0).mean(axis=0)
        axes[3, col].hist(tstd, bins=40, color="slategray")
        axes[3, col].axvline(sts[p]["temporal_std_median_per_eye"][0],
                             color="tab:blue", ls="--", label="median")
        axes[3, col].set_title("per-ommatidium temporal std", fontsize=9)
        axes[3, col].set_xlabel("std")
        axes[3, col].legend(fontsize=7)
    fig.suptitle(
        "Scene x vision integration -- FlyGym eye cameras (APPROXIMATE optics: "
        "Retina calibrated for FlyGym's own eye placement)\n"
        "blank (no lights, flat) vs lit/no-objects vs natural (lights + "
        "micro-relief + objects).  Units: mm model.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(OUT_DIR, "scene_vision_integration.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)

    report = dict(
        what_this_is=("hand-built integration test of scene x vision; not a "
                      "finished visual system and not a biology result"),
        pre_registered_ratio=PRE_REGISTERED_RATIO,
        per_preset={p: dict(model=dict(ngeom=recs[p]["ngeom"],
                                       nlight=recs[p]["nlight"],
                                       nhfield=recs[p]["nhfield"],
                                       npair=recs[p]["npair"],
                                       ncam=recs[p]["ncam"],
                                       mean_legs_in_contact=recs[p]["mean_legs_in_contact"]),
                           stats=sts[p]) for p in PRESETS},
        temporal_std_medians=dict(blank=blank_t, natural_lit_no_objects=lit_t,
                                  natural=nat_t),
        predictions=dict(P1_natural_beats_blank=bool(p1),
                         P2_light_alone_helps=bool(p2),
                         P3_frames_nonblank=bool(nonblank),
                         P4_objects_change_readout=bool(obj_visible)),
        cannot_show=[
            "no Drosophila visual-response reference dataset exists here, so "
            "'normal vs abnormal' is NOT measurable; only 'degenerate vs "
            "informative' is",
            "FlyGym's Retina is calibrated for FlyGym's own eye placement, so "
            "these ommatidia readouts are approximate optics",
            "the visual drive is NOT yet connected to the connectome tier, so the "
            "'notice a salient object' acceptance task remains NOT RUN",
            "scene objects are contype=conaffinity=0, i.e. purely visual: they can "
            "never be touched, so this says nothing about tactile salience",
        ],
    )
    jp = os.path.join(OUT_DIR, "scene_vision_integration.json")
    with open(jp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {png}\nwrote {jp}")


if __name__ == "__main__":
    main()
