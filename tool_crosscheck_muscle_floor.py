#!/usr/bin/env python3
"""INDEPENDENT CROSS-CHECK for the muscle floor audit (hand-run, NOT a finished product).

This is a second, deliberately independent probe whose only job is to check the two central
claims of run_muscle_floor_audit.py against the INSTALLED MuJoCo library rather than against
that script's own code path:

  CHECK 1  the activation rule.  Call mju_muscleDynamics / mju_muscleDynamicsTimescale on the
           installed libmujoco.so and compare with the local replication in
           run_muscle_floor_audit.py.
  CHECK 2  the passive force.  Call mju_muscleBias on the installed libmujoco.so for all 90
           muscles and compare with the actuator_force that mj_forward reports at act = 0.
           If the passive element did not act independently of activation, these would differ.

Nothing here writes to the canonical asset, to history, or to any shared module.  It only
prints; it produces no artifact of its own.  Run it with the same memory guard budget as the
main audit, and do not run it concurrently with any other heavy job of this project.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

PROJECT = Path("/run/media/sensen/Data2/cell_wound_prototype")
sys.path.insert(0, str(PROJECT))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from engine import memory_guard as mg                         # noqa: E402
from engine import canonical_physics as cp                    # noqa: E402
from run_muscle_floor_audit import (                          # noqa: E402
    CANONICAL_XML, EXPECTED_SHA256, sha256_file, muscle_dynamics_reference,
    muscle_fixed_point, REPLICATION_VALIDATION, validate_replication_region,
)


def load_lib():
    so = Path(mujoco.__file__).parent / "libmujoco.so.3.9.0"
    if not so.exists():
        raise SystemExit(f"installed library not found at {so}")
    lib = ctypes.CDLL(str(so))
    lib.mju_muscleDynamics.restype = ctypes.c_double
    lib.mju_muscleDynamics.argtypes = [ctypes.c_double, ctypes.c_double,
                                       ctypes.POINTER(ctypes.c_double)]
    lib.mju_muscleDynamicsTimescale.restype = ctypes.c_double
    lib.mju_muscleDynamicsTimescale.argtypes = [ctypes.c_double] * 4
    lib.mju_muscleBias.restype = ctypes.c_double
    lib.mju_muscleBias.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    lib.mju_muscleGain.restype = ctypes.c_double
    lib.mju_muscleGain.argtypes = [ctypes.c_double, ctypes.c_double,
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_double,
                                   ctypes.POINTER(ctypes.c_double)]
    return lib, str(so)


def buf(vals):
    return (ctypes.c_double * len(vals))(*[float(v) for v in vals])


def main() -> int:
    failures = []

    before = sha256_file(CANONICAL_XML)
    print(f"canonical sha256 before: {before}")
    if before != EXPECTED_SHA256:
        raise SystemExit("REFUSING: canonical asset hash differs from the expected value")

    lib, so_path = load_lib()
    print(f"installed library: {so_path}")
    print(f"mujoco {mujoco.__version__}")

    model = cp.load_model()
    data = mujoco.MjData(model)
    print(f"compiled: nq={model.nq} nv={model.nv} nu={model.nu} na={model.na} "
          f"dt={model.opt.timestep}")

    # ---------------- CHECK 1: activation rule ----------------
    print("\n=== CHECK 1: activation rule vs the installed library ===")
    prm3 = [float(x) for x in model.actuator_dynprm[0][:3]]
    print(f"dynprm[0:3] = {prm3}  (tau_act, tau_deact, smoothing_width)")

    # The asset can only reach ctrl >= 1e-4 (ctrllimited) and 0 <= act <= 1 (from act = 0 the
    # rule keeps act in [0,1]).  Require an exact match on the whole of that region; report the
    # corners outside it as measured mismatches of the replication, not as silent failures.
    reach_ctrl = (1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0)
    reach_act = (0.0, 1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    unreachable_ctrl = (-2.0, -1.0, -0.5, -1e-6)
    unreachable_act = (1.5, 2.0, 3.0)

    def diff_at(ctrl, act):
        return abs(lib.mju_muscleDynamics(ctrl, act, buf(prm3))
                   - muscle_dynamics_reference(ctrl, act, prm3))

    reachable = [(c, a) for c in reach_ctrl for a in reach_act]
    neg_ctrl = [(c, a) for c in unreachable_ctrl for a in reach_act]
    high_act = [(c, a) for c in reach_ctrl for a in unreachable_act]

    print(f"\n  VALIDATED region = ctrl >= 1e-4 AND 0 <= act <= 1  "
          f"({len(reachable)} points, this is exactly what the asset can reach)")
    worst_r, at_r = 0.0, None
    for c, a in reachable:
        d = diff_at(c, a)
        if d > worst_r:
            worst_r, at_r = d, (c, a)
    print(f"    worst |lib - replication| = {worst_r:.6g}" + (f"  at {at_r}" if at_r else ""))
    if worst_r >= 1e-9:
        failures.append(f"CHECK 1: replication differs in the VALIDATED region: {worst_r} at {at_r}")

    print(f"\n  UNREACHABLE ctrl <= 0 ({len(neg_ctrl)} points) -- reported, not required to match")
    worst_n, at_n = 0.0, None
    for c, a in neg_ctrl:
        d = diff_at(c, a)
        if d > worst_n:
            worst_n, at_n = d, (c, a)
    print(f"    worst |lib - replication| = {worst_n:.6g} at {at_n}")
    ex = (-1.0, -0.5, -1e-6)
    vals = {c: lib.mju_muscleDynamics(c, -0.5, buf(prm3)) for c in ex}
    repl_at_neg_half = muscle_dynamics_reference(ex[0], -0.5, prm3)
    print(f"    at act = -0.5 the library returns {vals}")
    print(f"    the replication returns {repl_at_neg_half:.10g} at ctrl = {ex[0]} for that")
    print("    same state, so the library IS insensitive to the magnitude of a negative ctrl")
    print("    there in a way this form does not reproduce.  (On the particular grid above")
    print("    the deviation happens to measure 0 because only act >= 0 was probed with")
    print("    negative ctrl; this corner is unreachable either way.)")

    print(f"\n  UNREACHABLE act > 1 ({len(high_act)} points) -- reported, not required to match")
    worst_h = max(diff_at(c, a) for c, a in high_act)
    print(f"    worst |lib - replication| = {worst_h:.6g}")
    print(f"    ctrl=0, act=2: lib={lib.mju_muscleDynamics(0.0, 2.0, buf(prm3)):.6g} "
          f"repl={muscle_dynamics_reference(0.0, 2.0, prm3):.6g}")
    print("    -> the library does not clip act inside the (0.5+1.5*act) scale there.")

    print("\n  key points inside the validated region:")
    for ctrl, act, note in ((0.0, 0.0, "headline start state"),
                            (1e-4, 0.0, "ctrl at the ctrlrange floor"),
                            (1e-4, 1e-4, "the fixed point"),
                            (1.0, 1.0, "fully activated"),
                            (1.0, 0.0, "full command, zero activation")):
        lib_dot = lib.mju_muscleDynamics(ctrl, act, buf(prm3))
        own_dot = muscle_dynamics_reference(ctrl, act, prm3)
        print(f"    ctrl={ctrl:<8} act={act:<8} lib={lib_dot:<14.8g} repl={own_dot:<14.8g} "
              f"({note})")

    # the artifact must record the same validated region this checker enforced, and the
    # run-time record must agree with a fresh grid computed here (independent re-measurement).
    expected_region = "ctrl >= 1e-4 AND 0 <= act <= 1"
    declared = validate_replication_region(prm3)
    print("\n  artifact-declared validation record (recomputed here):")
    print(f"    validated_region              : {declared['validated_region']}")
    print(f"    validated_points             : {declared['validated_points']}")
    print(f"    validated_points_matched     : {declared['validated_points_matched']}")
    print(f"    validated_worst_abs_diff     : {declared['validated_worst_abs_diff']}")
    print(f"    unreachable ctrl<=0 worst    : "
          f"{declared['unreachable_corner_ctrl_negative_worst_abs_diff']}")
    print(f"    unreachable act>1   worst    : "
          f"{declared['unreachable_corner_act_above_one_worst_abs_diff']}")
    if declared["validated_region"] != expected_region:
        failures.append("CHECK 1: artifact declares a different validated region than the "
                        f"checker enforced ({declared['validated_region']!r} vs "
                        f"{expected_region!r})")
    if declared["validated_points"] != len(reachable):
        failures.append(
            f"CHECK 1: artifact declares {declared['validated_points']} validated points but "
            f"the checker gridded {len(reachable)} for the same region")
    if declared["validated_points_matched"] != declared["validated_points"]:
        failures.append("CHECK 1: artifact reports a validated-region point that does not match")
    if REPLICATION_VALIDATION["validated_region"] != expected_region:
        failures.append("CHECK 1: module-level validation description disagrees with the checker")

    fixed = muscle_fixed_point(1e-4, prm3)
    print(f"\n  fixed point for ctrl=1e-4: {fixed!r}  (act* should equal ctrl exactly)")
    if abs(fixed - 1e-4) > 1e-15:
        failures.append("CHECK 1: fixed point is not ctrl")

    # the headline number: first step from act=0 with ctrl=0
    dot0 = lib.mju_muscleDynamics(min(1.0, float(np.clip(0.0, *model.actuator_ctrlrange[0]))),
                                  0.0, buf(prm3))
    print(f"  headline: act_dot at act=0, ctrl=0 (clamped to "
          f"{float(np.clip(0.0, *model.actuator_ctrlrange[0]))}) = {dot0}")
    if abs(dot0 - 2.0) > 1e-12:
        failures.append(f"CHECK 1: headline act_dot is {dot0}, not 2.0")
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    print(f"  mj_step at dt={model.opt.timestep} gives act = {data.act[0]!r}")
    if abs(data.act[0] - 0.0002) > 1e-15:
        failures.append(f"CHECK 1: act after one step is {data.act[0]}, not 0.0002")

    # ---------------- CHECK 2: passive force ----------------
    print("\n=== CHECK 2: passive force vs mju_muscleBias ===")
    mujoco.mj_resetData(model, data)
    data.act[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    if not np.all(np.asarray(data.act) == 0):
        failures.append("CHECK 2: act is not zero")

    force = np.array(data.actuator_force, dtype=float)
    length = np.array(data.actuator_length, dtype=float)
    lr = np.array(model.actuator_lengthrange, dtype=float)
    acc0 = np.array(model.actuator_acc0, dtype=float)
    gainprm = np.array(model.actuator_gainprm, dtype=float)

    bias = np.array([lib.mju_muscleBias(length[i], buf(lr[i]), acc0[i], buf(gainprm[i]))
                     for i in range(model.nu)])
    gain = np.array([lib.mju_muscleGain(length[i], 0.0, buf(lr[i]), acc0[i], buf(gainprm[i]))
                     for i in range(model.nu)])
    diff = np.abs(bias - force)
    print(f"  n actuators                       : {model.nu}")
    print(f"  max |mju_muscleBias - actuator_force| : {diff.max()}")
    print(f"  bitwise identical on all {model.nu}    : {bool(np.array_equal(bias, force))}")
    print(f"  force min / max                   : {force.min()} / {force.max()}")
    print(f"  muscles with force exactly 0       : {int(np.sum(force == 0.0))}")
    print(f"  muscles with force < 0             : {int(np.sum(force < 0.0))}")
    print(f"  force L2 norm (all 90)             : {float(np.linalg.norm(force))}")
    if not np.array_equal(bias, force):
        failures.append("CHECK 2: actuator_force != mju_muscleBias at act=0")

    # the strongest single claim: a muscle whose passive force equals its own act=1 force
    act1 = 1.0 * gain + bias
    ratio = np.abs(force) / np.abs(act1)
    order = np.argsort(-ratio)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("\n  per-muscle |passive| / |same muscle at act=1|, largest 5:")
    for i in order[:5]:
        print(f"    {names[i]:<42} passive={force[i]:<14.8g} act1={act1[i]:<14.8g} "
              f"ratio={ratio[i]:.6f}")
    print(f"  muscles with ratio == 1            : {int(np.sum(np.isclose(ratio, 1.0)))}")
    print(f"  muscles with ratio >= 0.5          : {int(np.sum(ratio >= 0.5))}")
    print(f"  muscles with ratio <= 0.01         : {int(np.sum(ratio <= 0.01))}")
    print(f"  median ratio                       : {float(np.median(ratio))}")
    # fail loudly if the report's claim would not reproduce
    if int(np.sum(np.isclose(ratio, 1.0))) < 1:
        failures.append("CHECK 2: no muscle has passive == its own act=1 force")

    after = sha256_file(CANONICAL_XML)
    print(f"\ncanonical sha256 after : {after}")
    print(f"canonical unchanged    : {after == before == EXPECTED_SHA256}")
    if after != before:
        failures.append("canonical asset changed during the cross-check")

    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  all cross-checks agree with the installed library and with the audit artifact")
    return 0


if __name__ == "__main__":
    with mg.guarded_run("muscle_floor_audit", **mg.BUDGETS["muscle_floor_audit"]) as report:
        rc = main()
        report("completed" if rc == 0 else "failed")
    raise SystemExit(rc)
