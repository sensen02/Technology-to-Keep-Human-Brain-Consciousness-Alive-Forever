"""Tests for the embodied body backend (milestone 1: a fly that walks).

Runs in the BODY environment (venv_body), because it needs MuJoCo:
    ./venv_body/bin/python run_body_selftest.py

WHAT THESE TESTS ESTABLISH, AND WHAT THEY DO NOT
------------------------------------------------
They establish that the backend is wired correctly and that its numbers are what
they claim: units, determinism, contact/actuator sanity, gate arithmetic, and
input validation.  They do NOT establish that a real fly walks this way; the
controller is FlyGym's demo CPG (an engineering baseline) and every threshold is
an engineering gate for this model, not a biological norm.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs", "embodied_body")

from engine.embodied import (BodyBackend, BodyConfig, summarise_walk,  # noqa: E402
                             MM, MM_PER_S, GEOMETRY_NOTE)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(("PASS " if ok else "FAIL ") + name + (" | " + str(detail)[:160] if detail else ""),
          flush=True)
    return bool(ok)


def main():
    t0 = time.perf_counter()
    cfg = BodyConfig(seed=0, timestep_s=1e-4)
    be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
    desc = be.describe()

    # ---- 1. units are millimetres, established from the model itself --------
    g = desc["model"]["gravity_mm_s2"]
    check("units: gravity is the mm/s^2 value the model YAML states",
          abs(g[2] + 9810.0) < 1e-6, f"gravity={g}")
    check("units: the units table says mm and the note says nothing is rescaled",
          desc["units"]["spawn_position_mm"] == MM and "millimetres" in GEOMETRY_NOTE,
          desc["units"]["spawn_position_mm"])

    # ---- 2. the body is the expected size (a fly is ~1 mm high, not ~1000) ---
    obs0 = be.observe()
    h0 = float(obs0.thorax_position_mm[2])
    check("geometry: thorax height is O(1) mm, not O(1000) -- catches a metre/mm mix-up",
          0.1 < h0 < 10.0, f"thorax z at spawn = {h0:.4f} mm")
    # MEASURED, not assumed: the compiled model reports 48 actuators, not 42.
    # 42 are the position actuators driving the locomotion DOFs (the CPG's output
    # order) and 6 more are the per-leg ADHESION actuators.  An earlier version of
    # this test asserted 42 because that is the CPG's DOF count, and it failed --
    # which is exactly what the test is for.
    check("geometry: 69 body segments and 48 actuators (42 position + 6 adhesion)",
          desc["n_bodysegs"] == 69 and desc["model"]["n_actuator"] == 48
          and len(desc["locomotion_dof_order"]) == 42,
          f"segs={desc['n_bodysegs']} nu={desc['model']['n_actuator']} "
          f"position_dofs={len(desc['locomotion_dof_order'])}")
    # The camera's name form is NOT stable enough to assert a shape on: a
    # standalone world compile produced 'nmf/trackcam', while THIS backend's
    # compiled model exposes it as 'trackcam'.  An earlier version of this test
    # demanded a "/" and failed on that difference.  What actually matters is
    # that the exposed name works, so this is a FUNCTIONAL check instead.
    try:
        _r = be.sim.set_renderer(desc["tracking_camera"], camera_res=(96, 128),
                                 playback_speed=1.0, output_fps=5, buffer_frames=True)
        for _ in range(3):
            be.step()
        _rendered = bool(be.sim.render_as_needed())
        _frames = _r.frames.get(desc["tracking_camera"], [])
        _std = float(np.asarray(_frames[-1]).std()) if _frames else 0.0
        func_ok, func_detail = (_rendered and _std > 1.0), \
            f"name={desc['tracking_camera']!r} in {desc['cameras']}, frames={len(_frames)}, std={_std:.1f}"
    except Exception as exc:                                     # noqa: BLE001
        func_ok, func_detail = False, f"{type(exc).__name__}: {exc}"
    check("backend: the exposed camera name actually renders a non-blank frame "
          "(name form 'trackcam' vs 'nmf/trackcam' is backend-dependent and NOT asserted)",
          func_ok, func_detail)
    check("backend: the site_positions limitation is recorded, not hidden",
          "0,3" in desc["site_positions_note"], desc["site_positions_note"][:60])

    # ---- 3. observation shapes and finiteness -------------------------------
    check("observe: joint angles (66,), velocities (66,), bodies (69,3), rotations (69,4)",
          obs0.joint_angles_rad.shape == (66,) and obs0.joint_velocities_rad_s.shape == (66,)
          and obs0.body_positions_mm.shape == (69, 3)
          and obs0.body_rotations_wxyz.shape == (69, 4),
          f"{obs0.joint_angles_rad.shape} {obs0.body_positions_mm.shape}")
    check("observe: all recorded quantities are finite at spawn",
          bool(np.isfinite(obs0.body_positions_mm).all()
               and np.isfinite(obs0.joint_angles_rad).all()),
          "finite")
    qn = float(np.linalg.norm(obs0.body_rotations_wxyz, axis=1).mean())
    check("observe: body rotations are unit quaternions (mean norm ~1)",
          abs(qn - 1.0) < 1e-6, f"mean |q| = {qn:.9f}")

    # ---- 4. the CPG actually drives a gait: legs cycle and contacts alternate -
    jl, cl, tl, xl = [], [], [], []
    for k in range(4000):
        be.step()
        if k % 20 == 0:
            o = be.observe()
            jl.append(o.joint_angles_rad.copy())
            cl.append(o.contact_present.copy())
            tl.append(o.time_s)
            xl.append(o.thorax_position_mm.copy())
    J = np.asarray(jl); C = np.asarray(cl)
    moved = float(np.linalg.norm(np.asarray(xl)[-1, :2] - np.asarray(xl)[0, :2]))
    per_joint_std = J.std(axis=0)
    n_moving_joints = int((per_joint_std > 1e-3).sum())
    check("gait: the CPG moves most actuated joints over 0.4 s",
          n_moving_joints >= 20, f"{n_moving_joints}/66 joints have sd > 1e-3 rad")
    check("gait: the fly translates horizontally",
          moved > 1.0, f"moved {moved:.3f} mm in {tl[-1]-tl[0]:.2f} s")
    legs_down = C.sum(axis=1)
    check("gait: at least one leg is in contact at every sampled instant",
          bool((legs_down >= 1).all()) and float(legs_down.max()) <= 6.0,
          f"legs down min={int(legs_down.min())} max={int(legs_down.max())} (must be <= 6)")
    raw_max = 0.0
    for _ in range(200):
        be.step()
        raw_max = max(raw_max, float(be.observe().contact_found_raw.max()))
    check("gait: the raw contact channel exceeds 1, proving it is NOT a leg count "
          "and that summing it would give impossible values",
          raw_max > 1.0,
          f"max raw found channel = {raw_max:.0f} (>1 observed); legs-in-contact is "
          f"therefore computed as (raw > 0)")
    check("gait: contact is not frozen (some legs lift and re-plant)",
          bool((C.std(axis=0) > 0).any()), f"per-leg contact sd = {np.round(C.std(axis=0),3)}")

    # ---- 5. determinism between two INDEPENDENT fresh instances ------------
    # Do NOT compare a fresh backend against the long-lived `be` used above: any
    # extra step taken elsewhere (for example by the camera check) offsets it and
    # this test then measures test ordering, not physics.  Two fresh instances,
    # stepped identically, are the correct comparison.
    def fresh_traj(seed, n=400, every=20):
        b = BodyBackend(BodyConfig(seed=seed, timestep_s=1e-4),
                        gl_backend=None).attach_cpg_baseline()
        xs = []
        for k in range(n):
            b.step()
            if k % every == 0:
                xs.append(b.observe().thorax_position_mm.copy())
        b.close()
        return np.asarray(xs)

    A = fresh_traj(0)
    B = fresh_traj(0)
    check("determinism: two independent fresh backends with the same seed agree exactly",
          np.array_equal(A, B), f"max |diff| = {float(np.abs(A - B).max()):.3e} mm")
    C = fresh_traj(7)
    check("determinism: a different seed gives a different trajectory",
          not np.array_equal(A, C),
          f"max |diff| = {float(np.abs(A - C).max()):.3e} mm")

    # ---- 7. gates are arithmetic, tested on synthetic cases ----------------
    def synthetic(disp_mm, h_end_mm):
        n = 11
        t = np.linspace(0, 1, n)
        p = np.zeros((n, 3))
        p[:, 0] = np.linspace(0, disp_mm, n)
        p[:, 2] = 1.0
        p[-1, 2] = h_end_mm
        c = np.ones((n, 6))
        return summarise_walk(t, p, c, cfg)
    ok_case = synthetic(50.0, 1.0)
    fail_move = synthetic(5.0, 1.0)
    fail_up = synthetic(50.0, 9.0)
    check("gates: a moving, upright case passes",
          ok_case["passed_engineering_gates"] is True, json.dumps(ok_case["gate_moved"]))
    check("gates: a barely-moving case fails on the movement gate",
          fail_move["passed_engineering_gates"] is False and fail_move["gate_moved"] is False,
          json.dumps({"moved": fail_move["gate_moved"]}))
    check("gates: an airborne/tipped case fails on the upright gate",
          fail_up["passed_engineering_gates"] is False and fail_up["gate_upright"] is False,
          json.dumps({"upright": fail_up["gate_upright"]}))
    check("metrics: units are attached to the numbers",
          ok_case["units"]["displacement"] == MM and ok_case["units"]["speed"] == MM_PER_S,
          json.dumps(ok_case["units"]))

    # ---- 8. invalid input is refused, not silently coerced -----------------
    bad = [dict(timestep_s=0.0), dict(timestep_s=float("nan")),
           dict(spawn_position_mm=(0.0, 0.0)), dict(world_half_size_mm=-1.0),
           dict(cpg_intrinsic_frequency_hz=0.0), dict(seed=-1), dict(seed=1.5)]
    refused = 0
    for kw in bad:
        try:
            BodyConfig(**kw).validate()
        except (ValueError, TypeError):
            refused += 1
    check("validation: every malformed config is refused", refused == len(bad),
          f"{refused}/{len(bad)} refused")
    try:
        be_noctrl = BodyBackend(BodyConfig(seed=0), gl_backend=None)
        be_noctrl.step()
        stepped = True
    except RuntimeError:
        stepped = False
    except Exception as exc:                                     # noqa: BLE001
        stepped = "wrong exception: %s" % type(exc).__name__
    check("validation: stepping without an action source raises RuntimeError",
          stepped is False, str(stepped))
    try:
        summarise_walk([0, 1], np.zeros((3, 3)), np.ones((3, 6)), cfg)
        mism = False
    except ValueError:
        mism = True
    check("validation: mismatched times/positions are refused", mism, "refused")

    be.close()
    n_pass = sum(1 for r in RESULTS if r["ok"])
    result = {"what_these_tests_establish": [
                  "the backend is wired correctly and its numbers are in the units it claims",
                  "the CPG baseline produces a gait: joints move, legs alternate, the body translates",
                  "the same seed reproduces and a different seed does not",
                  "engineering gates are arithmetic and were tested on synthetic cases",
                  "malformed configuration is refused rather than coerced"],
              "what_they_do_not_establish": [
                  "that a real fly walks this way: the controller is FlyGym's demo CPG",
                  "any biological norm for speed, height or gait; the gates are engineering thresholds",
                  "anything about perception, experience or consciousness"],
              "n_checks": len(RESULTS), "n_passed": n_pass,
              "n_failed": len(RESULTS) - n_pass,
              "passed": bool(n_pass == len(RESULTS)),
              "wall_seconds": time.perf_counter() - t0,
              "results": RESULTS}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "body_selftest.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed in {result['wall_seconds']:.1f}s")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
