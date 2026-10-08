"""See-through render of the fly with the MEASURED touch pathway glowing inside.

WHAT IS REAL IN THESE PICTURES
------------------------------
* The body is a REAL MuJoCo render with the fly's geoms made translucent, posed at
  poses RECORDED during a walk (``truth/body_positions_mm``, ``truth/joint_angles_rad``
  from ``loop_episode_cpg_baseline_seed0.npz``).  No physics is invented for the image.
* Each glowing marker sits on that leg's TARSUS: positioned from the recorded 3D body
  position and projected through the renderer's own camera matrix, so it lands on the
  rendered leg instead of floating in a separate scatter plot.
* Marker SIZE and COLOUR encode a MEASURED firing rate: that leg's mechanosensory
  neuron group in the bounded BANC conductance tier, driven through the project's own
  mechanoreceptor bank with the RECORDED contact forces and joint angles -- the same
  transduction path the closed loop uses (``adapters.MechanoReceptorBank.transduce``
  then ``NeuralTier.advance``).
* A small inset shows the same six numbers as bars, because that is the only way to
  read exact values off a picture; it is a chart and is labelled as one.

HONEST LIMITS (printed on each figure)
--------------------------------------
* The tier is a bounded SELECTION of a dataset annotation, not a brain.
* Edges are UNSIGNED (``nt_pair`` is -1 on every cached row) and the cached table is
  thresholded at >=5 synapses, so edge counts are bandwidth, not excitation.
* Receptor kinetics and the force->drive scale are ILLUSTRATIVE; the contact-force
  channel's ABSOLUTE calibration is unresolved, so only its relative pattern is used,
  against a reference pose recorded in the same run.
* Marker POSITION is the leg the sensory neurons are annotated to innervate
  (nerve-based).  Single-neuron soma coordinates do not exist in this dataset, so
  individual neurons are NOT placed in space.
* A glowing marker means a neuron group fired.  No experience claim.

Run: ./venv_body/bin/python render_touch_xray.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs" / "embodied_body"
DEST = OUT / "stills_touch"
EPISODE = OUT / "loop_episode_cpg_baseline_seed0.npz"

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
TARSUS = {leg: f"{leg}_tarsus5" for leg in LEGS}
DT_NEURAL_MS = 5.0
WINDOW_S = 0.4

CAPTION = ("SEE-THROUGH RENDER — real MuJoCo geometry at a RECORDED pose; the six markers are "
           "per-leg mechanosensory firing measured in the bounded BANC tier.\n"
           "Tier = bounded SELECTION of an annotation; edges UNSIGNED and thresholded at >=5 "
           "synapses (bandwidth, not excitation); receptor kinetics ILLUSTRATIVE; contact-force "
           "absolute calibration UNRESOLVED (relative use only).\n"
           "Marker position = the ANNOTATED leg nerve, NOT single-neuron somata (which do not "
           "exist here). A marker means a neuron group fired — not evidence of experience.")


def to_u8(a):
    a = np.asarray(a)
    return a if a.dtype == np.uint8 else (255 * np.clip(a, 0, 1)).astype("uint8")


class Obs:
    """Minimal stand-in for BodyObservation, built from RECORDED episode arrays."""
    def __init__(self, contact_forces, joint_angles_rad):
        self.contact_forces = contact_forces
        self.joint_angles_rad = joint_angles_rad


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ep = {}
    with np.load(EPISODE, allow_pickle=False) as z:
        for k in z.files:
            ep[k] = z[k]
    t = np.asarray(ep["truth/time_s"], float)
    dt_body = float(t[1] - t[0])
    contacts = np.asarray(ep["truth/contact_forces"], float)      # (n,6,3)
    angles = np.asarray(ep["truth/joint_angles_rad"], float)      # (n,66)
    positions = np.asarray(ep["truth/body_positions_mm"], float)  # (n,69,3)
    print(f"episode {t.size} samples, {t[0]:.2f}..{t[-1]:.2f} s, dt {dt_body*1000:.1f} ms")

    # ---- body + tier -----------------------------------------------------
    import mujoco
    from engine.embodied import BodyBackend, BodyConfig
    import engine.embodied.adapters as ad

    # ONE MuJoCo context only: an earlier version built two backends and the
    # offscreen renderer produced a blank frame (changing geom alpha gave a
    # BYTE-IDENTICAL png, proving the fly was not in the rendered frame at all).
    be2 = BodyBackend(BodyConfig(seed=0, world_half_size_mm=1000.0), gl_backend=None)
    for gi in range(be2.model.ngeom):
        if str(be2.model.body(be2.model.geom_bodyid[gi]).name).startswith(be2.cfg.fly_name):
            be2.model.geom_rgba[gi][3] = 0.45
    be2.model.mat_reflectance[:] = 0.0
    print("one MuJoCo context; fly geoms translucent")

    built = ad.build_neural_tier(ad.TierConfig())
    ledger = ad.OwnershipLedger()
    tier = ad.NeuralTier(built, ad.TierConfig(), ledger)
    maps = ad.leg_rows_from_dof_order(be2.dof_order, be2.model)
    bank = ad.MechanoReceptorBank(built["sensory"], maps["angle_rows"], built["n"],
                                 ad.ReceptorConfig(), dt_ms=DT_NEURAL_MS,
                                 ledger=ad.OwnershipLedger(), seed=0)
    groups = built["sensory"]
    print("tier n =", built["n"], "| per-leg sensory group sizes:",
          {k: int(np.size(v)) for k, v in groups.items()})

    # ---- measure per-leg firing in windows -------------------------------
    n_sub = max(1, int(round(DT_NEURAL_MS / 5.0)))      # tier dt_ms is 5 ms
    win = int(round(WINDOW_S / dt_body))
    moments = [int(round(x / dt_body)) for x in (1.0, 2.0, 3.0, 4.0)]
    results = []
    for m in moments:
        i0 = max(0, m - win // 2)
        i1 = min(t.size, i0 + win)
        bank.set_reference_pose(angles[i0])
        counts = {leg: 0 for leg in LEGS}
        for k in range(i0, i1):
            cur, drive = bank.transduce(Obs(contacts[k], angles[k]))
            sp = tier.advance(n_sub, cur)
            for leg in LEGS:
                counts[leg] += int(np.count_nonzero(sp[:, groups[leg]]))
        rates = {leg: counts[leg] / (float((i1 - i0) * dt_body) * float(np.size(groups[leg])))
                 for leg in LEGS}
        results.append({"t_s": float(t[m]), "index": m, "rates_hz": rates,
                        "counts": counts})
        print(f"t={t[m]:.2f}s  " + "  ".join(f"{leg}={rates[leg]:7.2f}Hz" for leg in LEGS))

    # ---- render each moment with projected markers ------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rates = np.array([[r["rates_hz"][leg] for leg in LEGS] for r in results])
    vmax = float(all_rates.max()) or 1.0
    cmap = matplotlib.colormaps["inferno"]   # cm.get_cmap was removed in matplotlib 3.11

    made = []
    # The episode was produced by THIS configuration with seed 0, so stepping a fresh
    # simulator to the same times reproduces the same poses.  An earlier version wrote
    # the recorded qpos directly, which put the fly at a pose the tracking camera did
    # not follow and produced a blank frame with markers off the body.
    from flygym_demo.complex_terrain.common import apply_locomotion_action
    from flygym_demo.complex_terrain.cpg_controller import (
        CPGController, make_tripod_cpg_network)
    from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
    from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
    be2.attach_cpg_baseline()
    names2 = [str(n) for n in be2.bodyseg_names]
    def find2(want):
        for i, nm in enumerate(names2):
            if nm.endswith(f"name='{want}')"):
                return i
        return None
    tarsus_i = {leg: find2(TARSUS[leg]) for leg in LEGS}
    tarsus_i = {k: v for k, v in tarsus_i.items() if v is not None}
    print("tarsus bodies (render sim):", tarsus_i)

    step_of_t = {}
    cur_step = 0
    rend = be2.sim.set_renderer(be2.camera_spec, camera_res=(700, 900),
                                playback_speed=1.0, output_fps=1000, buffer_frames=True)
    for res in results:
        target = int(round(res["t_s"] / dt_body)) * int(round(dt_body / 1e-4))
        while cur_step < target:
            be2.step(); cur_step += 1
        # the fly must be in the frame before rendering
        for _ in range(3):
            be2.sim.render_as_needed()
        frame = np.asarray(rend.frames[be2.camera_spec][-1])
        print(f"   frame std={float(frame.std()):.1f} mean={float(frame.mean()):.1f} "
              f"nonblank={float(frame.std())>1}")
        M = np.asarray(rend.get_camera_matrix(be2.camera_spec, be2.sim.mj_data, be2.model))
        live = np.asarray(be2.sim.get_body_positions(be2.fly.name), float)
        homo = np.concatenate([live, np.ones((live.shape[0], 1))], axis=1)
        uv = homo @ M[:3, :].T
        uv = uv[:, :2] / np.where(np.abs(uv[:, 2:3]) < 1e-9, 1e-9, uv[:, 2:3])

        fig, ax = plt.subplots(figsize=(8.4, 6.6), layout="constrained")
        ax.imshow(to_u8(frame))
        ax.set_xticks([]); ax.set_yticks([])
        for leg in LEGS:
            ti = tarsus_i.get(leg)
            if ti is None:
                continue
            x, y = float(uv[ti, 0]), float(uv[ti, 1])
            r = 260.0 * float(res["rates_hz"][leg]) / vmax + 60.0
            c = cmap(float(res["rates_hz"][leg]) / vmax)
            ax.scatter([x], [y], s=r, facecolors="none", edgecolors=[c],
                       linewidths=3.4, zorder=5)
            ax.annotate(f"{res['rates_hz'][leg]:.0f} Hz", (x, y),
                        textcoords="offset points", xytext=(10, -8), color="white",
                        fontsize=10, zorder=6,
                        bbox=dict(fc="black", alpha=0.6, pad=1.6, lw=0))
        ax.set_title(CAPTION, fontsize=7.0)
        p = DEST / f"xray_touch_t{res['t_s']:.1f}s.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))
        print("wrote", p)

    # ---- one combined figure with the numeric inset ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained",
                             gridspec_kw={"width_ratios": [3, 1]})
    rend = be2.sim.set_renderer(be2.camera_spec, camera_res=(700, 900),
                                playback_speed=1.0, output_fps=1000, buffer_frames=True)
    for _ in range(3):
        be2.sim.render_as_needed()
    axes[0].imshow(to_u8(np.asarray(rend.frames[be2.camera_spec][-1])))
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("last recorded pose, translucent body", fontsize=9)
    x = np.arange(len(LEGS))
    for j, res in enumerate(results):
        axes[1].bar(x + (j - 1.5) * 0.2, [res["rates_hz"][l] for l in LEGS], width=0.2,
                    label=f"t={res['t_s']:.0f}s")
    axes[1].set_xticks(x, LEGS)
    axes[1].set(ylabel="mechanosensory group firing (Hz)",
                title="CHART (not a render): the same six numbers, read exactly")
    axes[1].legend(fontsize=8)
    fig.suptitle(CAPTION, fontsize=7.5)
    p = DEST / "xray_touch_combined.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))

    (DEST / "xray_touch_report.json").write_text(json.dumps({
        "what_this_is": "see-through render at recorded poses + measured per-leg "
                        "mechanosensory firing in the bounded BANC tier",
        "episode": str(EPISODE), "dt_body_ms": dt_body * 1000,
        "tier_neurons": int(built["n"]),
        "per_leg_group_sizes": {k: int(np.size(v)) for k, v in groups.items()},
        "window_s": WINDOW_S, "neural_dt_ms": DT_NEURAL_MS,
        "results": results,
        "honesty": CAPTION.split("\n"),
        "marker_position_basis": "annotated leg nerve (LEG_NERVE); NOT soma coordinates",
        "no_experience_claim": True,
    }, indent=2, default=str))
    print("wrote", len(made), "images + report")
    be2.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
