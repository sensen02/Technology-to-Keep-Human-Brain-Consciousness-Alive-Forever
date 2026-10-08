"""Vision non-degeneracy: is the embodied fly's visual input degenerate or informative?

WHAT IS MEASURED
----------------
The embodied fly closed loop had NO visual pathway: the locomotion fly was built without
``add_vision()``, so the compiled model had no eye cameras and ``get_raw_vision()``
raised; the scene had ``nlight == 0`` and no objects.  The consequence recorded before
now was that any visually guided test was NOT RUN.

This experiment replaces "not run" with a MEASURED, falsifiable statement.  It builds one
world per condition -- the same fly, the same seeds, the same tripod-CPG drive, differing
only in whether a small high-contrast object stands in front-left of the head -- renders
the compound eyes, reduces them to ``VisualFeatures``, and applies a PRE-REGISTERED
threshold to decide DEGENERATE vs INFORMATIVE.

Three conditions (separate simulations, identical seeds):
    object_present          dark 1.4 mm sphere 2.2 mm ahead and 1.4 mm to the left of the
                            head, at eye height (fly is ~2.5 mm long, so this is fly-scale)
    object_absent           the same world with no object (the historical scene)
    object_present_blind    the object is there, the eye is blind: the drive is forced to
                            zero at the readout level, so this is a real control

PRE-REGISTERED PREDICTIONS (declared before the run; failures are reported as failures)
    P1  object-present temporal std > object-absent temporal std
    P2  the object-absent verdict is DEGENERATE, or its median temporal std is "clearly
        closer to the threshold" -- pre-registered as ratio-to-threshold < 2.0
    P3  the blind drive is EXACTLY zero and its statistics are degenerate by construction

WHAT CANNOT BE CLAIMED
----------------------
Whether a fly's visual system is NORMAL is not testable here: this project has no
Drosophila visual-response reference dataset.  What IS testable is degenerate vs
informative.  FlyGym's fisheye Retina is calibrated for FlyGym's OWN eye placement, so
these ommatidia readouts are APPROXIMATE optics on a body the Retina was not calibrated
for.  Every gain and threshold is either derived from the readout quantisation geometry
(stated in the module) or marked ASSUMED with a sweep range.

OUTPUTS (both under outputs/embodied_body/)
    vision_nondegeneracy.json    every number, every verdict, every caveat, wall clock
    vision_nondegeneracy.png     sample eye frame per condition + temporal-std histograms
                                 with the pre-registered threshold drawn as a labelled line
    vision_nondegeneracy_series.npz   the full per-ommatidium arrays behind the figure

RUN (BODY environment; needs MUJOCO_GL, the script selects and reports the backend):
    ./venv_body/bin/python run_vision_nondegeneracy.py
    ./venv_body/bin/python run_vision_nondegeneracy.py --gl glfw      # force a backend
    ./venv_body/bin/python run_vision_nondegeneracy.py --quick        # shorter run

EXIT CODE
    0 = every pre-registered prediction PASSED
    1 = at least one pre-registered prediction FAILED (the results are still written)
    2 = no GL backend produced a frame, so no condition could be measured
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_DIR = os.path.join(HERE, "outputs", "embodied_body")
JSON_PATH = os.path.join(OUT_DIR, "vision_nondegeneracy.json")
PNG_PATH = os.path.join(OUT_DIR, "vision_nondegeneracy.png")
NPZ_PATH = os.path.join(OUT_DIR, "vision_nondegeneracy_series.npz")

from engine.embodied.vision import (                                    # noqa: E402
    ASSUMED_PARAMETERS, CAVEATS, HONESTY, MIN_SERIES_LENGTH,
    NONDEGENERACY_MIN_SPATIAL_CONTRAST, NONDEGENERACY_MIN_TEMPORAL_STD,
    OmmatidiaFrontEnd, THRESHOLD_DERIVATION, VisionConfig,
    ideal_observer_quantisation_floor, non_degeneracy_statistics, visual_drive_nA,
)
from engine.receptors import to_nA                                     # noqa: E402


# ===========================================================================
# PRE-REGISTERED  (declared here, before any measurement; not changed afterwards)
# ===========================================================================
FLY_NAME = "nmf"
TIMESTEP_S = 1e-4
CPG_INTRINSIC_FREQUENCY_HZ = 12.0
SEED = 0                       # identical seed in every condition
DURATION_S = 1.5               # ~1.5 s of simulated time per condition
DT_READ_S = 0.02               # VisualFeatures recorded every 20 ms -> 75 samples
FRAME_SAMPLE_EVERY = 5         # raw frames kept for the blank-frame check
#: POST-HOC ONLY (never used for a P1-P3 verdict): samples dropped when reporting the
#: "after settling" version of the same statistic, i.e. 0.2 s at 20 ms per sample.
SETTLE_SKIP_SAMPLES = 10

#: The object: 1.4 mm diameter dark sphere, 0.67 mm ahead / 1.0 mm left of the eye,
#: at eye height (eye at ~(0.80, +0.38, 1.02) mm).  Fly-scale, high contrast, and
#: verified to fall inside the LEFT eye's field only (right-eye readout changes by 0.0).
OBJECT = {
    "shape": "sphere", "radius_mm": 0.7, "diameter_mm": 1.4,
    "position_mm": (2.2, 1.4, 1.25), "rgba": (0.03, 0.03, 0.03, 1.0),
    "contype": 0, "conaffinity": 0, "group": 0,
    "note": ("contype=conaffinity=0: the object is VISUAL ONLY and cannot touch the fly, "
             "so any physical difference between conditions would be a bug; the measured "
             "thorax trajectories of the object_present and object_absent runs are "
             "compared in the report to prove they are identical"),
}
GROUND_HALF_SIZE_MM = 1000.0   # same scene as the historical runs

#: drive scale: ASSUMED (no measured Drosophila transduction gain exists).  A sweep is
#: reported so no single assumed value is presented as a result.
DRIVE_SCALE_ASSUMED = 1.0
DRIVE_SCALE_SWEEP = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

#: P2's pre-registered meaning of "clearly closer to the threshold".
P2_MARGIN_RATIO = 2.0

PREDICTIONS = {
    "P1": ("object-present left-eye median per-ommatidium temporal std > object-absent "
           "left-eye median (the object is placed on the left; the right eye is a "
           "within-run control and is expected to be unchanged)"),
    "P2": (f"object-absent verdict is DEGENERATE, or its median temporal std is within a "
           f"factor of {P2_MARGIN_RATIO} of the threshold "
           f"({NONDEGENERACY_MIN_TEMPORAL_STD:.1e}) -- pre-registered definition of "
           "'clearly closer to the threshold'"),
    "P3": ("blind=True drive is EXACTLY zero in every channel at every sample, and the "
           "blind condition's statistics are DEGENERATE"),
}

CONDITIONS = (
    ("object_present", True, False),
    ("object_absent", False, False),
    ("object_present_blind", True, True),
)

#: Post-hoc diagnostics.  NOT pre-registered, and they do NOT change the P1-P3 verdicts.
#: They answer the follow-up question "then what IS degenerate?": a static eye in a
#: static scene, and how much of the object-absent temporal contrast comes from the
#: ground texture rather than from the fly's own legs.
POST_HOC_DIAGNOSTICS = (
    ("static_object_absent", dict(object_present=False, walk=False, uniform_ground=False)),
    ("static_object_present", dict(object_present=True, walk=False, uniform_ground=False)),
    ("walk_uniform_ground_object_absent",
     dict(object_present=False, walk=True, uniform_ground=True)),
)

GL_CANDIDATES = ("egl", "osmesa", "glfw", None)

#: LITERATURE CONTEXT -- cited numbers from real Drosophila experiments, provided so the
#: reader can see the physical scale this prototype is nowhere near.  They are NOT a
#: reference dataset and are NOT used as a criterion anywhere in this file: no threshold,
#: gain or verdict below is taken from them.  A behaviour threshold measured on a real
#: fly under a specific stimulus is not a "normal response" baseline for a whole-brain
#: model, and this project has no way to compare against one.
LITERATURE_CONTEXT = {
    "not_a_reference_dataset": (
        "These are threshold numbers from specific behavioural and electrophysiological "
        "experiments. They are NOT a reference dataset, they were NOT compared against "
        "anything measured here, and no threshold or gain in this experiment is taken "
        "from them."),
    "optomotor_thresholds_50lx": {
        "spatial_acuity_cycles_per_degree": 0.10,
        "temporal_acuity_hz": 50,
        "contrast_threshold": 0.10,
        "at_0.3_lx": {"spatial_acuity_cycles_per_degree": 0.06,
                      "temporal_acuity_hz": 10, "contrast_threshold": 0.29},
        "source": ("Palavalli-Nettimi & Theobald 2020, Vision Research 169:33-40, "
                   "PMID 32163744, doi:10.1016/j.visres.2020.02.007"),
        "relevance_here": (
            "the recording interval of this experiment is 20 ms = 50 Hz, i.e. exactly the "
            "measured temporal acuity of the fly's optomotor response. This readout "
            "therefore cannot resolve anything the real fly could not, and it cannot be "
            "claimed to under-sample the fly either; the ASSUMED 20 ms high-pass corner "
            "sits at that same edge, which is why its sweep range (5-50 ms) matters."),
    },
    "photoreceptor_R1_R6": {
        "info_capacity_saturates": "200-300 bits/s at 3e5 photons/s",
        "adaptation_range_photons_per_s": (3e2, 3e6),
        "input_resistance_MOhm": "700 +/- 540 dark, falling to 320 +/- 100 on light adaptation",
        "steady_state_depolarisation_fraction": "0.39 +/- 0.09",
        "source": "Juusola & Hardie 2001, J Gen Physiol 117:3-25",
        "relevance_here": ("none of this is modelled: this module has no phototransduction "
                           "cascade, no adaptation and no quantum bumps."),
    },
    "dark_noise": {
        "rate_per_s": 2, "amplitude_pA": 2, "duration_ms": 10,
        "evidence_level": ("ABSTRACT LEVEL ONLY: the full text could not be fetched "
                           "(HTTP 500). Treat as unverified beyond the abstract."),
        "relevance_here": ("not used. Note that the pre-registered threshold used here is "
                           "a QUANTISATION floor of the renderer, not a photon-noise floor, "
                           "and the two are not comparable."),
    },
    "body_mass_mg": {"value": 0.983, "source": "Vaxenburg et al., Nature 643 (2025)"},
}


# ===========================================================================
# GL backend selection (must happen BEFORE mujoco is imported)
# ===========================================================================
GL_PROBE_SRC = r"""
import os, sys
import numpy as np
from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import make_locomotion_fly
f = make_locomotion_fly("probe")
f.add_vision(draw_sensor_markers=False)          # MUST precede add_fly
w = FlatGroundWorld(name="w", half_size=200.0)
w.add_fly(f, spawn_position=[0, 0, 0.5], spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]))
w.compile()
s = Simulation(w, timestep=1e-4)
s.warmup(0.01)
raw = np.asarray(s.get_raw_vision("probe"))
o = np.asarray(s.get_ommatidia_readouts("probe"))
ok = raw.shape == (2, 512, 450, 3) and o.shape[1:] == (721, 2)
print("PROBE_OK" if (ok and raw.std() > 1.0) else "PROBE_BLANK",
      float(raw.std()), tuple(raw.shape), tuple(o.shape))
"""


def select_gl_backend(forced=None, candidates=GL_CANDIDATES, timeout_s=900):
    """Return the first GL backend that actually renders here.  Measured, not assumed."""
    order = list(candidates)
    env_gl = os.environ.get("MUJOCO_GL")
    if forced is not None:
        order = [forced] + [c for c in order if c != forced]
    elif env_gl:
        order = [env_gl] + [c for c in order if c != env_gl]
    tried = []
    for gl in order:
        env = dict(os.environ)
        if gl is None:
            env.pop("MUJOCO_GL", None)
        else:
            env["MUJOCO_GL"] = gl
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, "-c", GL_PROBE_SRC], capture_output=True,
                                  text=True, env=env, timeout=timeout_s, cwd=HERE)
            ok = "PROBE_OK" in proc.stdout
            detail = (proc.stdout.strip().splitlines()[-1] if proc.stdout.strip()
                      else (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip()
                            else "no output"))
        except subprocess.TimeoutExpired:
            ok, detail = False, f"timeout after {timeout_s}s"
        tried.append({"backend": gl or "default(unset MUJOCO_GL)", "rendered": bool(ok),
                      "detail": detail, "seconds": round(time.time() - t0, 1)})
        print(f"  GL probe {str(gl):>8s}: {'OK  ' if ok else 'FAIL'} {detail}")
        if ok:
            return {"winner": gl, "winner_name": gl or "default(unset MUJOCO_GL)",
                    "tried": tried}
    return {"winner": None, "winner_name": None, "tried": tried}


# ===========================================================================
# Scene + recording
# ===========================================================================
def _build(object_present, uniform_ground, seed=SEED, walk=True):
    """Build fly + world + simulation.  Mirrors engine.embodied.body_backend (read-only
    reference) with the two additions this experiment needs: eye cameras and a scene
    object.  Returns (sim, fly, controller, apply_action).

    ``walk=False`` freezes the CPG phase (intrinsic frequency 0) but STILL writes the
    preprogrammed stance every step.  That matters: leaving the actuators at ctrl == 0
    would drive every joint toward zero angle and the fly would collapse over the run,
    which would look like "visual change in a static scene" when it is really a broken
    controller.  A frozen-phase CPG holds the stance instead."""
    from flygym.compose import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym.utils.mjcf import GEOM_TYPES, add_material, add_texture
    from flygym_demo.complex_terrain.common import (
        apply_locomotion_action, get_default_locomotion_dof_order, make_locomotion_fly)
    from flygym_demo.complex_terrain.cpg_controller import (
        CPGController, make_tripod_cpg_network)
    from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

    fly = make_locomotion_fly(FLY_NAME, colorize=False)
    # ORDER MATTERS (verified both ways): add_vision() resolves parent bodies through
    # self.mjcf_root.body(name), which returns None once the fly has been re-parented
    # into the world, so calling it after world.add_fly() fails with a misleading
    # AttributeError.  add_vision() takes NO fovy argument for this fly class and
    # returns None; the angle is fixed at 157 deg by the shipped vision.yaml asset.
    fly.add_vision(draw_sensor_markers=False)

    world = FlatGroundWorld(name="vision_world", half_size=GROUND_HALF_SIZE_MM)
    if uniform_ground:
        add_texture(world.mjcf_root, name="flatmat_tex", type="2d", builtin="flat",
                    width=8, height=8, rgb1=(0.35, 0.35, 0.35), rgb2=(0.35, 0.35, 0.35))
        add_material(world.mjcf_root, name="flatmat", texture="flatmat_tex",
                     texrepeat=(1, 1), reflectance=0.2)
        world.ground_geom.material = "flatmat"
    if object_present:
        world.mjcf_root.worldbody.add_geom(
            type=GEOM_TYPES[OBJECT["shape"]], name="salient_object",
            size=[float(OBJECT["radius_mm"]), 0, 0],
            pos=[float(v) for v in OBJECT["position_mm"]],
            rgba=list(OBJECT["rgba"]),
            contype=int(OBJECT["contype"]), conaffinity=int(OBJECT["conaffinity"]),
            group=int(OBJECT["group"]))
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    world.compile()
    sim = Simulation(world, timestep=TIMESTEP_S)
    sim.warmup()

    net = make_tripod_cpg_network(
        timestep=sim.timestep,
        intrinsic_frequency=(CPG_INTRINSIC_FREQUENCY_HZ if walk else 0.0), seed=seed)
    ctrl = CPGController(net, PreprogrammedSteps(),
                         output_dof_order=get_default_locomotion_dof_order())
    return sim, fly, ctrl, apply_locomotion_action


def run_condition(label, object_present, blind, walk=True, uniform_ground=False,
                  duration_s=DURATION_S, dt_read_s=DT_READ_S, seed=SEED,
                  drive_scale=DRIVE_SCALE_ASSUMED, frame_sample_every=FRAME_SAMPLE_EVERY,
                  verbose=True):
    """Record ``VisualFeatures`` for one condition and return (record, flat array dict)."""
    sim, fly, ctrl, apply_action = _build(object_present, uniform_ground, seed=seed,
                                          walk=walk)
    config = VisionConfig(enabled=True, fovy=145.0,
                          temporal_tau_ms=ASSUMED_PARAMETERS["temporal_tau_ms"]["value"],
                          contrast_gain=ASSUMED_PARAMETERS["contrast_gain"]["value"],
                          motion_gain=ASSUMED_PARAMETERS["motion_gain"]["value"],
                          n_channels=2, drive_scale_nA_per_unit=float(drive_scale),
                          seed=seed, blind=blind, scramble=False)
    fe = OmmatidiaFrontEnd(config)
    fe.bind(sim, fly.name)
    fe.reset()

    n_steps = max(1, int(round(dt_read_s / sim.timestep)))
    n_reads = max(2, int(round(duration_s / dt_read_s)))
    series, drives, frames, frame_times, thorax = [], [], [], [], []
    sim_time = 0.0
    t_wall = time.time()
    for k in range(n_reads):
        if k % frame_sample_every == 0:
            # Diagnostic render, kept for the blank-frame check.  It runs in EVERY
            # condition including blind (where read() itself does not query the camera),
            # so the blind condition's render pipeline is proven alive too.
            raw = np.asarray(sim.get_raw_vision(fly.name))
            frames.append(raw.copy())
            frame_times.append(sim_time)
        feats = fe.read(sim, fly.name, sim_time)
        series.append(feats)
        drives.append(visual_drive_nA(feats, config, to_nA))
        thorax.append(np.asarray(sim.get_body_positions(fly.name)[0], float).copy())
        for _ in range(n_steps):
            if walk:
                apply_action(sim, fly.name, ctrl.step())
            sim.step()
            sim_time += sim.timestep

    wall = time.time() - t_wall
    mask = np.asarray(fe.retina.ommatidia_id_map) > 0
    frame_stats = []
    for raw, t in zip(frames, frame_times):
        per_eye = []
        for e in range(raw.shape[0]):
            f = raw[e].astype(np.float64)
            per_eye.append({
                "full_frame_std_8bit": float(f.std()),
                "full_frame_mean_8bit": float(f.mean()),
                "interior_std_8bit": float(f[mask].std()),
                "interior_mean_8bit": float(f[mask].mean()),
                "min": float(f.min()), "max": float(f.max()),
            })
        frame_stats.append({"time_s": float(t), "per_eye": per_eye})
    all_frames_same = bool(all(np.array_equal(frames[0], fr) for fr in frames[1:]))
    any_frame_constant = bool(any(s["per_eye"][e]["min"] == s["per_eye"][e]["max"]
                                  for s in frame_stats for e in range(2)))

    stats = non_degeneracy_statistics(series)
    # POST-HOC, not pre-registered and NOT used for any P1-P3 verdict: the same statistic
    # over the series with the first SETTLE_SKIP_SAMPLES dropped, i.e. after the initial
    # stance/physics transition.  This separates "the scene keeps changing" from "the fly
    # was still settling when the recording started".
    settled = non_degeneracy_statistics(series[SETTLE_SKIP_SAMPLES:])
    settled_exact0 = float(np.mean(np.ptp(
        np.stack([f.eye_luminance for f in series[SETTLE_SKIP_SAMPLES:]]), axis=0) == 0.0))
    drive_nA = np.stack([d["drive_nA"] for d in drives], axis=0)      # (T, n_ch)
    lum = np.stack([f.eye_luminance for f in series], axis=0)         # (T, 2, n)
    tstd = lum.std(axis=0)                                            # (2, n)
    thorax = np.asarray(thorax)

    record = {
        "label": label,
        "condition": {"object_present": bool(object_present), "blind": bool(blind),
                      "walked": bool(walk), "uniform_ground": bool(uniform_ground)},
        "vision_config": config.as_dict(),
        "bind_report": fe.bind_report,
        "n_samples": len(series),
        "duration_s": float(n_reads * dt_read_s),
        "dt_read_s": float(dt_read_s),
        "wall_seconds": round(wall, 2),
        "statistics": stats,
        "statistics_after_settling_transient__post_hoc": {
            "disclaimer": (
                "NOT pre-registered, NOT used for the P1-P3 verdicts. Same statistic over "
                f"samples [{SETTLE_SKIP_SAMPLES}:] only, i.e. after "
                f"{SETTLE_SKIP_SAMPLES * dt_read_s:.1f} s of stance/physics transition. "
                "It separates 'the scene keeps changing' from 'the fly was still settling'."),
            "settle_skip_samples": int(SETTLE_SKIP_SAMPLES),
            "verdict": settled["verdict"],
            "n_samples": settled["n_samples"],
            "temporal_std_median_per_eye": settled["temporal_std_median_per_eye"],
            "fraction_ommatidia_above_threshold": settled["fraction_ommatidia_above_threshold"],
            "fraction_ommatidia_exactly_constant": settled_exact0,
        },
        "prediction_relevant": {
            "left_eye_median_temporal_std": float(stats["temporal_std_median_per_eye"][0]),
            "right_eye_median_temporal_std": float(stats["temporal_std_median_per_eye"][1]),
            "ratio_to_threshold_left": float(
                stats["temporal_std_median_per_eye"][0] / NONDEGENERACY_MIN_TEMPORAL_STD),
            "ratio_to_threshold_right": float(
                stats["temporal_std_median_per_eye"][1] / NONDEGENERACY_MIN_TEMPORAL_STD),
        },
        "drive": {
            "channel_names": drives[0]["channel_names"],
            "unit_contract": drives[0]["unit_contract"],
            "conversion_helper": drives[0]["conversion_helper"],
            "drive_scale_nA_per_unit": float(drive_scale),
            "drive_scale_provenance": drives[0]["drive_scale_provenance"],
            "nA_mean_per_channel": [float(v) for v in drive_nA.mean(axis=0)],
            "nA_max_per_channel": [float(v) for v in drive_nA.max(axis=0)],
            "nA_at_first_sample": [float(v) for v in drive_nA[0]],
            "nA_at_last_sample": [float(v) for v in drive_nA[-1]],
            "total_nA_mean": float(np.abs(drive_nA).sum(axis=1).mean()),
            "exactly_zero_at_every_sample": bool(np.all(drive_nA == 0.0)),
            "drive_mV_at_last_sample": [float(v) for v in drives[-1]["drive_mV"]],
            "caveats": drives[0]["caveats"],
        },
        "frame_check": {
            "n_frames_kept": len(frames),
            "frames": frame_stats,
            "all_kept_frames_bit_identical": all_frames_same,
            "any_frame_constant_over_pixels": any_frame_constant,
            "blank_frame_conclusion": (
                "INVALID CONDITION: at least one rendered frame is constant over all "
                "pixels (min == max), so its statistics cannot be interpreted"
                if any_frame_constant else
                "frames are not blank: every kept frame varies across pixels "
                "(min < max) in both eyes"),
            "note": ("frame std here is over the raw fisheye-corrected 512x450x3 frame, so "
                     "it includes the black region outside the fisheye disc and the "
                     "uniform white skybox.  It proves the renderer produced content; it "
                     "is NOT the temporal statistic."),
        },
        "locomotion": {
            "thorax_start_mm": [float(v) for v in thorax[0]],
            "thorax_end_mm": [float(v) for v in thorax[-1]],
            "displacement_mm": [float(v) for v in (thorax[-1] - thorax[0])],
            "thorax_height_mean_mm": float(thorax[:, 2].mean()),
            "controller": ("flygym_demo tripod CPG (ENGINEERING BASELINE, not the "
                           "connectome); intrinsic frequency "
                           f"{CPG_INTRINSIC_FREQUENCY_HZ if walk else 0.0} Hz, seed {seed}"
                           + ("" if walk else
                              "; FROZEN PHASE: the stance is still written every step but "
                              "the fly does not walk")),
        },
        "warnings": list(series[-1].warnings),
        "caveats": list(CAVEATS),
    }
    if verbose:
        s = stats
        print(f"  [{label}] verdict={s['verdict']}  "
              f"L med={s['temporal_std_median_per_eye'][0]:.6f} "
              f"R med={s['temporal_std_median_per_eye'][1]:.6f}  "
              f"frac>thr={s['fraction_ommatidia_above_threshold']:.3f}  "
              f"spatial={s['spatial_contrast_mean_abs']:.4f}  "
              f"drive_nA_mean={record['drive']['total_nA_mean']:.4g}  "
              f"frame_std_L={frame_stats[0]['per_eye'][0]['full_frame_std_8bit']:.1f}  "
              f"post-hoc settled L med="
              f"{settled['temporal_std_median_per_eye'][0]:.6f} "
              f"(exactly-constant "
              f"{100 * settled_exact0:.1f}%)  "
              f"({wall:.1f}s)")
    sim.close()
    arrays = {
        "temporal_std": tstd,
        "temporal_std_eye0": tstd[0],
        "temporal_std_eye1": tstd[1],
        "eye_luminance_series_L": lum[:, 0, :],
        "eye_luminance_series_R": lum[:, 1, :],
        "drive_nA_series": drive_nA,
        "thorax_mm_series": thorax,
        "frame_L": frames[0][0],
        "frame_R": frames[0][1],
    }
    return record, arrays


# ===========================================================================
# Report
# ===========================================================================
def make_figure(records, arrays, path, threshold, config_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flygym.vision.retina import Retina

    retina = Retina()
    order = ["object_present", "object_absent", "object_present_blind"]
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13.2),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.72]})
    for col, key in enumerate(order):
        rec = records[key]
        ax = axes[0, col]
        ax.imshow(arrays[key]["frame_L"])
        ax.set_title(f"{key}\nLEFT eye frame @ t=0, frame std="
                     f"{rec['frame_check']['frames'][0]['per_eye'][0]['full_frame_std_8bit']:.1f}"
                     f"/255", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, col]
        tstd = arrays[key]["temporal_std"]
        all_dead = bool(np.all(tstd == 0.0))
        for e, (style, ecolor, lbl) in enumerate(
                [("-", "#1f77b4", "left eye"), ("--", "#d62728", "right eye")]):
            if all_dead:
                continue
            vals = np.maximum(tstd[e], 1e-12)
            ax.hist(vals, bins=np.logspace(-7, 0.3, 46), histtype="step", linestyle=style,
                    color=ecolor, linewidth=1.8, label=f"{lbl} (median "
                    f"{np.median(tstd[e]):.2e})")
        ax.axvline(threshold, color="k", linewidth=2.0, linestyle="-.")
        ax.text(threshold * 1.25, ax.get_ylim()[1] * 0.62,
                f"pre-registered threshold\n{threshold:.1e}", fontsize=9.5,
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.85))
        if all_dead:
            ax.text(0.5, 0.5,
                    f"all {tstd.shape[1]} ommatidia in BOTH eyes:\ntemporal std EXACTLY 0.0\n"
                    "(nothing is plotted because there is nothing to plot)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    bbox=dict(facecolor="mistyrose", edgecolor="k"))
        ax.set_xscale("log")
        ax.set_xlim(1e-7, 2.0)
        s = rec["statistics"]
        ax.set_title(f"{key}: {s['verdict']}\nL med {s['temporal_std_median_per_eye'][0]:.2e}, "
                     f"R med {s['temporal_std_median_per_eye'][1]:.2e}, "
                     f"{100 * s['fraction_ommatidia_above_threshold']:.0f}% of ommatidia above",
                     fontsize=10.5)
        ax.set_xlabel("per-ommatidium temporal std of the raw readout\n"
                      "(readout units, 0..1 of full 8-bit scale)", fontsize=9.5)
        ax.set_ylabel("ommatidia" if col == 0 else "", fontsize=9.5)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(alpha=0.25)

        # where in the eye the temporal variation actually is, left eye, same ommatidia
        ax = axes[2, col]
        hexmap = retina.hex_pxls_to_human_readable(tstd[0].astype(np.float64),
                                                   default_value=np.nan)
        ax.set_facecolor("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(hexmap),
                       vmin=0.0, vmax=max(0.05, float(np.nanmax(tstd[0]))))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("LEFT eye: per-ommatidium temporal std\n"
                     "(black = that ommatidium never changed; grey = no ommatidium)",
                     fontsize=10.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                     label="temporal std")
    fig.suptitle("Embodied fly visual input: degenerate or informative?  "
                 "(approximate FlyGym optics; no Drosophila reference dataset)", fontsize=14)
    fig.text(0.5, 0.008, config_note, ha="center", fontsize=9.0)
    fig.tight_layout(rect=(0, 0.020, 1, 0.972))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_predictions(records):
    a, b, c = records["object_present"], records["object_absent"], records["object_present_blind"]
    la = a["prediction_relevant"]["left_eye_median_temporal_std"]
    lb = b["prediction_relevant"]["left_eye_median_temporal_std"]
    ra = a["prediction_relevant"]["right_eye_median_temporal_std"]
    rb = b["prediction_relevant"]["right_eye_median_temporal_std"]

    p1_pass = bool(la > lb)
    p1 = {
        "prediction": PREDICTIONS["P1"],
        "object_present_left_median": la, "object_absent_left_median": lb,
        "ratio_present_over_absent": float(la / lb) if lb > 0 else None,
        "control_right_eye_present": ra, "control_right_eye_absent": rb,
        "control_right_eye_identical": bool(ra == rb),
        "result": "PASS" if p1_pass else "FAIL",
    }

    vb = b["statistics"]["verdict"]
    ratio_b = b["prediction_relevant"]["ratio_to_threshold_left"]
    p2_pass = bool(vb == "DEGENERATE" or ratio_b < P2_MARGIN_RATIO)
    p2 = {
        "prediction": PREDICTIONS["P2"],
        "object_absent_verdict": vb,
        "object_absent_ratio_to_threshold": ratio_b,
        "pre_registered_margin_ratio": P2_MARGIN_RATIO,
        "object_present_ratio_to_threshold": a["prediction_relevant"]["ratio_to_threshold_left"],
        "result": "PASS" if p2_pass else "FAIL",
        "note": ("the raw medians are reported above and are NOT re-weighted; if this "
                 "reads FAIL, the object-absent scene still carried temporal contrast"),
    }

    zero = bool(c["drive"]["exactly_zero_at_every_sample"])
    vc = c["statistics"]["verdict"]
    p3_pass = bool(zero and vc == "DEGENERATE")
    p3 = {
        "prediction": PREDICTIONS["P3"],
        "blind_drive_exactly_zero_all_channels_all_samples": zero,
        "blind_verdict": vc,
        "blind_left_median_temporal_std": c["prediction_relevant"]["left_eye_median_temporal_std"],
        "result": "PASS" if p3_pass else "FAIL",
    }
    n_pass = sum(1 for p in (p1, p2, p3) if p["result"] == "PASS")
    return {"P1": p1, "P2": p2, "P3": p3,
            "tally": f"{n_pass}/3 pre-registered predictions PASS",
            "n_pass": n_pass, "n_total": 3}


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gl", default=None, help="force a GL backend (egl/osmesa/glfw/unset)")
    ap.add_argument("--quick", action="store_true", help="shorter recording (0.5 s)")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="skip the post-hoc diagnostics")
    args = ap.parse_args(argv)

    duration = 0.5 if args.quick else DURATION_S
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 100)
    print("VISION NON-DEGENERACY -- pre-registered criteria:")
    for k, v in PREDICTIONS.items():
        print(f"  {k}: {v}")
    print(f"  threshold NONDEGENERACY_MIN_TEMPORAL_STD = {NONDEGENERACY_MIN_TEMPORAL_STD:.1e} "
          f"(readout units); quantisation floor recomputed here = "
          f"{ideal_observer_quantisation_floor():.3e}; ratio = "
          f"{NONDEGENERACY_MIN_TEMPORAL_STD / ideal_observer_quantisation_floor():.2f}x")
    print(f"  {THRESHOLD_DERIVATION}")
    print("=" * 100)

    print("selecting GL backend (a frame must actually render):")
    gl = select_gl_backend(args.gl)
    if gl["winner"] is None and "default(unset MUJOCO_GL)" not in [
            t["backend"] for t in gl["tried"] if t["rendered"]]:
        print("FATAL: no GL backend produced a frame -- every condition would be blank.")
        print(json.dumps(gl, indent=2))
        return 2
    # The winner must be exported before any flygym/mujoco import below.
    if gl["winner"] is None:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ["MUJOCO_GL"] = str(gl["winner"])
    print(f"  -> using MUJOCO_GL={gl['winner_name']}")

    records, arrays = {}, {}
    for label, obj, blind in CONDITIONS:
        print(f"running condition {label} (object={obj}, blind={blind}, "
              f"{duration}s @ {DT_READ_S}s, seed={SEED})")
        rec, arr = run_condition(label, obj, blind, duration_s=duration)
        records[label], arrays[label] = rec, arr

    preds = evaluate_predictions(records)

    diagnostics = {}
    if not args.no_diagnostics:
        print("post-hoc diagnostics (NOT pre-registered; verdicts above are unchanged):")
        for label, kw in POST_HOC_DIAGNOSTICS:
            rec, arr = run_condition(label, kw["object_present"], False,
                                     walk=kw["walk"], uniform_ground=kw["uniform_ground"],
                                     duration_s=duration)
            diagnostics[label] = {
                "what": label, "kwargs": kw,
                "verdict": rec["statistics"]["verdict"],
                "left_eye_median_temporal_std":
                    rec["prediction_relevant"]["left_eye_median_temporal_std"],
                "right_eye_median_temporal_std":
                    rec["prediction_relevant"]["right_eye_median_temporal_std"],
                "fraction_ommatidia_above_threshold":
                    rec["statistics"]["fraction_ommatidia_above_threshold"],
                "spatial_contrast_mean_abs": rec["statistics"]["spatial_contrast_mean_abs"],
                "frame_std_L": rec["frame_check"]["frames"][0]["per_eye"][0]["full_frame_std_8bit"],
                "statistics": rec["statistics"],
                "statistics_after_settling_transient__post_hoc":
                    rec["statistics_after_settling_transient__post_hoc"],
            }
            arrays[label] = arr

    # --- object actually visible?  the two runs share seed and initial state, so the
    # difference at the FIRST frame is caused by the object alone.
    d0 = (arrays["object_present"]["eye_luminance_series_L"][0]
          - arrays["object_absent"]["eye_luminance_series_L"][0])
    visibility = {
        "mean_abs_left_eye_readout_difference_at_t0": float(np.mean(np.abs(d0))),
        "max_abs_left_eye_readout_difference_at_t0": float(np.max(np.abs(d0))),
        "fraction_left_ommatidia_changed_gt_1e-3": float(np.mean(np.abs(d0) > 1e-3)),
        "right_eye_readout_identical_between_conditions": bool(
            np.array_equal(arrays["object_present"]["eye_luminance_series_R"],
                           arrays["object_absent"]["eye_luminance_series_R"])),
        "object_position_mm": list(OBJECT["position_mm"]),
        "object_diameter_mm": OBJECT["diameter_mm"],
        "verdict": ("object is inside the LEFT eye's field and changes a measurable "
                    "fraction of its ommatidia"
                    if float(np.mean(np.abs(d0))) > 1e-3 else
                    "OBJECT NOT DETECTED in the eye readout -- the object condition is "
                    "not a valid test"),
    }
    thorax_delta = float(np.max(np.abs(
        np.asarray(records["object_present"]["locomotion"]["thorax_end_mm"])
        - np.asarray(records["object_absent"]["locomotion"]["thorax_end_mm"]))))

    fig_note = (f"threshold {NONDEGENERACY_MIN_TEMPORAL_STD:.1e} readout units = "
                f"{NONDEGENERACY_MIN_TEMPORAL_STD * 255:.2f} grey levels of 255 "
                f"(3.7x the quantisation allowance of a constant image); "
                f"approximate FlyGym optics, MUJOCO_GL={gl['winner_name']}")
    make_figure({k: records[k] for k in records}, arrays, PNG_PATH,
                NONDEGENERACY_MIN_TEMPORAL_STD, fig_note)

    np.savez_compressed(NPZ_PATH, **{f"{k}__{arr_k}": v
                                     for k, d in arrays.items()
                                     for arr_k, v in d.items()})

    report = {
        "title": "Embodied fly visual input: degenerate or informative?",
        "generated_by": os.path.abspath(__file__),
        "module": os.path.join(HERE, "engine", "embodied", "vision.py"),
        "wall_seconds_total": round(time.time() - t_start, 1),
        "peak_ram_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "gl_backend": gl,
        "pre_registration": {
            "predictions": PREDICTIONS,
            "threshold_readout_units": NONDEGENERACY_MIN_TEMPORAL_STD,
            "threshold_derivation": THRESHOLD_DERIVATION,
            "quantisation_floor_readout_units": ideal_observer_quantisation_floor(),
            "threshold_over_floor": NONDEGENERACY_MIN_TEMPORAL_STD
            / ideal_observer_quantisation_floor(),
            "spatial_contrast_floor": NONDEGENERACY_MIN_SPATIAL_CONTRAST,
            "min_series_length": MIN_SERIES_LENGTH,
            "conditions": [{"label": c[0], "object_present": c[1], "blind": c[2]}
                           for c in CONDITIONS],
            "seed": SEED, "duration_s": duration, "dt_read_s": DT_READ_S,
            "p2_margin_ratio": P2_MARGIN_RATIO,
            "threshold_was_fixed_before_analysis": True,
            "note": ("P2's FAIL branch is reported as a failure. The prediction text is "
                     "not redefined after the fact anywhere in this file."),
        },
        "scene": {
            "world": "flygym.compose.FlatGroundWorld, half_size = "
                     f"{GROUND_HALF_SIZE_MM} mm (the historical scene, unchanged)",
            "fly": "flygym_demo.complex_terrain.common.make_locomotion_fly('nmf')",
            "vision": ("fly.add_vision(draw_sensor_markers=False) called BEFORE "
                       "world.add_fly(...); NeuroMechFly.add_vision takes no fovy "
                       "argument and returns None"),
            "nlight": records["object_present"]["bind_report"].get("nlight", "not recorded"),
            "object": OBJECT,
            "lights_added": False,
            "nlight_compiled_model": records["object_present"]["bind_report"]["model_nlight"],
            "ncam_compiled_model": records["object_present"]["bind_report"]["model_ncam"],
            "lighting_note": ("the scene still contains nlight == 0 lights; the image is "
                              "lit by MuJoCo's camera headlight, which is why frames are "
                              "not black even with no lights in the scene"),
            "scene_fallback_used": False,
            "scene_fallback_note": ("no fallback was needed: a sphere geom WAS added to "
                                    "the worldbody before compile and is visible in the "
                                    "left eye's readout"),
        },
        "object_visibility_check": visibility,
        "physics_identity_check": {
            "max_abs_thorax_end_difference_mm_between_object_present_and_absent":
                thorax_delta,
            "note": ("the object has contype=conaffinity=0 (visual only) and both runs "
                     "use seed 0, so a non-zero difference would be a bug in the "
                     "comparison"),
        },
        "conditions": records,
        "predictions": preds,
        "post_hoc_diagnostics": {
            "disclaimer": ("NOT pre-registered, added AFTER the pre-registered verdicts, "
                           "and they change none of them. They answer 'then what IS "
                           "degenerate?'. Read them as follow-up evidence only."),
            "runs": diagnostics,
        },
        "assumed_parameters": ASSUMED_PARAMETERS,
        "literature_context_not_a_reference_dataset": LITERATURE_CONTEXT,
        "honesty": HONESTY,
        "caveats": list(CAVEATS),
    }

    # drive scale sweep: linear in the factor, so it can be reported exactly from the
    # recorded mV drive without re-running anything.  The factor itself is ASSUMED.
    mv = np.asarray(records["object_present"]["drive"]["drive_mV_at_last_sample"], float)
    report["drive_scale_sweep_at_last_sample"] = {
        "note": ("the drive in nA is linear in this factor and the factor is ASSUMED, so "
                 "the sweep is reported instead of a single 'result' current. mV drive at "
                 "the last sample is factor-independent."),
        "drive_mV": [float(v) for v in mv],
        "nA_per_scale": {f"{s:g}": [float(v) for v in to_nA(mv, s, "A" * 10 + " sweep")]
                         for s in DRIVE_SCALE_SWEEP},
    }

    with open(JSON_PATH, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)

    # ------------------------------------------------------------------ printout
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    for label, _, _ in CONDITIONS:
        r = records[label]
        s = r["statistics"]
        print(f"{label}:")
        print(f"   verdict              : {s['verdict']}")
        print(f"   why                  : {s['reason']}")
        print(f"   n_samples            : {s['n_samples']}   n_ommatidia/eye: {s['n_ommatidia']}")
        print(f"   temporal std per eye : L median {s['temporal_std_median_per_eye'][0]:.6e} "
              f"R median {s['temporal_std_median_per_eye'][1]:.6e}  "
              f"(mean L {s['temporal_std_mean_per_eye'][0]:.6e} R {s['temporal_std_mean_per_eye'][1]:.6e}, "
              f"p90 L {s['temporal_std_p90_per_eye'][0]:.6e} R {s['temporal_std_p90_per_eye'][1]:.6e})")
        print(f"   ratio to threshold   : L {r['prediction_relevant']['ratio_to_threshold_left']:.2f}x "
              f"R {r['prediction_relevant']['ratio_to_threshold_right']:.2f}x")
        print(f"   fraction > threshold : {s['fraction_ommatidia_above_threshold']:.3f} "
              f"(per eye {[round(v, 3) for v in s['fraction_ommatidia_above_threshold_per_eye']]})")
        print(f"   spatial contrast mean: {s['spatial_contrast_mean_abs']:.4e} "
              f"(floor {NONDEGENERACY_MIN_SPATIAL_CONTRAST:.1e})   "
              f"motion_energy mean: {s['motion_energy_mean']:.4e}")
        print(f"   mean luminance       : L {s['mean_luminance_per_eye'][0]:.4f} "
              f"R {s['mean_luminance_per_eye'][1]:.4f}")
        print(f"   frame std (t=0)      : L {r['frame_check']['frames'][0]['per_eye'][0]['full_frame_std_8bit']:.2f} "
              f"R {r['frame_check']['frames'][0]['per_eye'][1]['full_frame_std_8bit']:.2f} "
              f"(of 255)   bit-identical frames: "
              f"{r['frame_check']['all_kept_frames_bit_identical']}   blank: "
              f"{r['frame_check']['any_frame_constant_over_pixels']}")
        print(f"   drive nA mean/chan   : {[round(v, 6) for v in r['drive']['nA_mean_per_channel']]}"
              f"  exactly zero everywhere: {r['drive']['exactly_zero_at_every_sample']}")

    print("-" * 100)
    print("PRE-REGISTERED PREDICTIONS")
    for k in ("P1", "P2", "P3"):
        p = preds[k]
        print(f"  {k}: {p['result']}  -- {p['prediction']}")
        for key, val in p.items():
            if key not in ("prediction", "result"):
                print(f"        {key} = {val}")
    print(f"  TALLY: {preds['tally']}")

    print("-" * 100)
    print("OBJECT VISIBILITY / PHYSICS IDENTITY")
    print(f"  left-eye readout change at t=0 caused by the object: "
          f"mean|d|={visibility['mean_abs_left_eye_readout_difference_at_t0']:.4f}, "
          f"{100 * visibility['fraction_left_ommatidia_changed_gt_1e-3']:.1f}% of ommatidia "
          f"changed; right eye identical: "
          f"{visibility['right_eye_readout_identical_between_conditions']}")
    print(f"  thorax end-position difference between object_present and object_absent: "
          f"{thorax_delta:.2e} mm")

    print("-" * 100)
    print("WHAT I COULD NOT VERIFY (carried into every report)")
    for c in CAVEATS:
        print(f"  * {c}")
    print("  * LITERATURE CONTEXT IS NOT A CRITERION: measured Drosophila optomotor "
          "thresholds exist (0.10 cycles/deg, 50 Hz, 10% contrast at 50 lx; Palavalli-Nettimi "
          "& Theobald 2020, PMID 32163744) but they are behaviour thresholds from a specific "
          "experiment, NOT a reference dataset, and nothing here was compared against them.")
    print("=" * 100)
    print(f"json : {JSON_PATH}")
    print(f"png  : {PNG_PATH}")
    print(f"npz  : {NPZ_PATH}")
    print(f"total wall clock: {report['wall_seconds_total']}s, "
          f"peak RAM {report['peak_ram_mb']} MB")
    print(f"exit code {0 if preds['n_pass'] == preds['n_total'] else 1}: "
          f"0 = every pre-registered prediction PASSED, 1 = at least one FAILED "
          f"(the results are written either way)")
    return 0 if preds["n_pass"] == preds["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
