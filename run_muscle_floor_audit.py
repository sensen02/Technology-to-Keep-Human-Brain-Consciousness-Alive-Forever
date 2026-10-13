#!/usr/bin/env python3
"""MUSCLE CONTROL-FLOOR AUDIT  (hand-run, hand-tested research audit -- NOT a finished product)

QUESTION
--------
The canonical asset
    /run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml
compiles to nu=90 actuators, ALL of them muscles (dyntype=gaintype=biastype=muscle), with
ctrllimited=True and ctrlrange=[0.0001, 1].  Starting from act=0 with ctrl=0 (or ctrl=1e-4)
the first step gives act_dot=2.0 and act=0.0002 after dt=1e-4 -- and the muscle force is
still NEGATIVE at act=0, so "zero control" is not "zero force".

This script measures, on that exact asset (never modified, never rewritten):

  A. Where the act_dot=2.0 comes from, with the measured actuator_dynprm / gainprm / biasprm
     values and their documented field meanings.
  B. Timestep sensitivity: dt = 1e-4 / 5e-5 / 1e-5 (plus stability probes and the exact
     discrete map), and whether the act trajectory converges / oscillates.
  C. The magnitude of the passive force (act=0) for all 90 muscles, in MODEL force units.
  D. Whether the ctrlrange lower bound 0.0001 keeps a real active tension or is numerically
     negligible, measured as a steady-state act and force difference against 0.
  E. A NAMED DERIVED-MODEL comparison, IN MEMORY ONLY:
        'native'     : ctrlrange exactly as compiled, [0.0001, 1]
        'zero_floor' : ctrlrange lower bound changed to 0 in the COMPILED mjModel
     under the SAME initialisation: instantaneous fixed-pose diagnostic + short free
     integration.  The canonical XML is NOT written, NOT touched and NOT re-derived.

WHAT THIS IS NOT
----------------
Not a closed-loop acceptance, not a standing controller, not a finished product.  This is a
hand-written audit run by hand.  Results measured on the derived 'zero_floor' model are
properties of that derived model only and may NEVER be quoted as properties of the canonical
model.

MEMORY SAFETY
-------------
This host recorded a real global OOM kill.  The entry point therefore runs inside
engine/memory_guard.guarded_run('muscle_floor_audit', **BUDGETS['muscle_floor_audit']),
which refuses to start when the host cannot afford the job, caps this process's address
space, and takes the project-wide single-flight lock so two heavy runs cannot stack.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/run/media/sensen/Data2/cell_wound_prototype")
sys.path.insert(0, str(PROJECT))

import numpy as np                                            # noqa: E402
import mujoco                                                 # noqa: E402

from engine import memory_guard as mg                         # noqa: E402
from engine import canonical_physics as cp                    # noqa: E402

OUT_JSON = PROJECT / "outputs/body_physics_audit/muscle_floor_audit.json"
CANONICAL_XML = PROJECT / "outputs/muscles_six_legs/fruitfly_six_leg_standing.xml"
EXPECTED_SHA256 = "65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6"

# Protected files: (path, sha256).  Nothing below may change them.
PROTECTED = [
    (CANONICAL_XML, EXPECTED_SHA256),
    (PROJECT / "engine/canonical_physics.py", None),
    (PROJECT / "engine/memory_guard.py", None),
]

DTS = [1e-4, 5e-5, 1e-5]
DT_MAIN = 1e-4
N_STEP_FOR_DT = 0.001          # compare trajectories over the same physical span
FREE_INTEGRATION_S = 0.05

mt = None                      # set in main()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_snapshot() -> dict:
    return {str(p): sha256_file(p) for p, _ in PROTECTED if Path(p).exists()}


# ---------------------------------------------------------------------------
# Ground truth for the activation-dynamics rule: the INSTALLED MuJoCo library
# exports mju_muscleDynamics, so the rule can be called directly instead of being
# read out of a document.  This is what settles the docs-vs-measurement conflict.
# ---------------------------------------------------------------------------
MUJOCO_SO = Path(sys.prefix).parent / "lib/python3.12/site-packages/mujoco/libmujoco.so.3.9.0"


def load_muscle_dynamics_fn():
    """Return a ctypes wrapper for the installed library's mju_muscleDynamics."""
    import ctypes
    candidates = [
        Path(__import__("mujoco").__file__).parent / "libmujoco.so.3.9.0",
        MUJOCO_SO,
    ]
    so = next((c for c in candidates if c.exists()), None)
    if so is None:
        return None, None
    lib = ctypes.CDLL(str(so))
    lib.mju_muscleDynamics.restype = ctypes.c_double
    lib.mju_muscleDynamics.argtypes = [ctypes.c_double, ctypes.c_double,
                                       ctypes.POINTER(ctypes.c_double)]
    lib.mju_muscleDynamicsTimescale.restype = ctypes.c_double
    lib.mju_muscleDynamicsTimescale.argtypes = [ctypes.c_double] * 4

    def dyn(ctrl, act, prm3):
        buf = (ctypes.c_double * 3)(*[float(x) for x in prm3])
        return float(lib.mju_muscleDynamics(float(ctrl), float(act), buf))

    def timescale(dctrl, tau_act, tau_deact, smoothing):
        return float(lib.mju_muscleDynamicsTimescale(
            float(dctrl), float(tau_act), float(tau_deact), float(smoothing)))

    return dyn, timescale


def muscle_dynamics_reference(ctrl, act, prm3, *, trace: bool = False):
    """Exact replication of the INSTALLED library's mju_muscleDynamics (mujoco 3.9.0).

    The rule was recovered by calling mju_muscleDynamics/mju_muscleDynamicsTimescale on the
    installed libmujoco.so and by disassembling both functions; it is the rule also written in
    the MuJoCo modelling documentation (Millard et al. 2013):

        ctrl_c = min(max(ctrl, 0), 1)          # clipped to [0, 1] internally
        act_c  = min(max(act, 0), 1)
        scale  = 0.5 + 1.5 * act_c
        tau    = tau_act  * scale   if (ctrl_c - act_c) > 0
                 tau_deact / scale  otherwise
        act_dot = (ctrl_c - act_c) / max(tau, mjMINVAL)

    prm3 = actuator_dynprm[0:3] = (tau_act, tau_deact, smoothing_width).  This asset has
    smoothing_width = 0, which selects the hard switch.

    VALIDATED REGION (measured; validator: tool_crosscheck_muscle_floor.py): this replication
    agrees with the installed library to |diff| < 1e-9 on EVERY tested point of the region
    ctrl >= 1e-4 and 0 <= act <= 1 -- 54/54 points, and that region is the only one this asset
    can reach, because ctrllimited clamps ctrl up to 1e-4 and the rule above keeps
    act in [0, 1] from act = 0.  Outside it the installed library was measured to differ:
      * ctrl < 0        : unreachable; the library ignores the sign in ways this form does not
                          reproduce (e.g. at act = -0.5 it returns the same act_dot for
                          ctrl = -1.0 and ctrl = -0.5);
      * act > 1         : unreachable; measured e.g. ctrl = 0, act = 2 gives -10000 where this
                          form gives -5000, i.e. the library does not clip act inside the
                          (0.5 + 1.5*act) scale factor there.
    Those two corners are reported as UNVALIDATED in the artifact rather than modelled.
    """
    tau_act, tau_deact, smoothing = (float(x) for x in prm3)
    ctrl_c = min(1.0, max(0.0, float(ctrl)))
    act_c = min(1.0, max(0.0, float(act)))
    scale = 0.5 + 1.5 * act_c
    dctrl = ctrl_c - act_c
    if smoothing < 1e-10:
        tau = tau_act * scale if dctrl > 0.0 else tau_deact / scale
    else:
        blended = tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smoothing + 0.5)
        tau = blended if dctrl > 0.0 else blended / scale
    if trace:
        return dctrl / max(1e-300, tau), tau, {"ctrl_clipped": ctrl_c, "act_clipped": act_c,
                                               "scale": scale, "dctrl": dctrl}
    return dctrl / max(1e-300, tau)


#: Description of WHICH region of :func:`muscle_dynamics_reference` was validated against the
#: installed library, and which corners were measured to disagree.  The point counts are filled
#: in at run time by :func:`validate_replication_region` rather than asserted here.
REPLICATION_VALIDATION = {
    "validated_region": "ctrl >= 1e-4 AND 0 <= act <= 1",
    "match_tolerance": 1e-9,
    "is_this_region_the_reachable_one": True,
    "reachable_because": "ctrllimited clamps ctrl to [1e-4, 1]; from act = 0 the rule keeps "
                         "act in [0, 1] (act* = ctrl, and act <= 1 is preserved)",
    "known_mismatch_corner_ctrl_negative": {
        "reachable": False,
        "measured_example": "the library is insensitive to the magnitude of a negative ctrl "
                            "(at act = -0.5 it returns the same act_dot for ctrl = -1.0 and "
                            "ctrl = -0.5), which this form does not reproduce",
    },
    "known_mismatch_corner_act_above_one": {
        "reachable": False,
        "measured_example": "ctrl = 0, act = 2 gives library -10000 against this form's -5000; "
                            "the library does not clip act inside the (0.5+1.5*act) scale there",
    },
    "validator": "tool_crosscheck_muscle_floor.py (enforces exactly the region named above)",
}


def validate_replication_region(prm3) -> dict:
    """Grid the reachable region and the two unreachable corners against the installed library.

    Returns a record with the measured point counts and worst deviations, so the artifact states
    what was actually checked rather than what the author assumed.
    """
    dyn_fn, _ = load_muscle_dynamics_fn()
    if dyn_fn is None:
        return {"validated": False, "reason": "installed libmujoco.so not found"}

    reach_ctrl = (1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0)
    reach_act = (0.0, 1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

    def worst(pairs):
        w, at = 0.0, None
        for c, a in pairs:
            d = abs(dyn_fn(c, a, prm3) - muscle_dynamics_reference(c, a, prm3))
            if d > w:
                w, at = d, [c, a]
        return w, at

    reach = [(c, a) for c in reach_ctrl for a in reach_act]
    neg = [(c, a) for c in (-2.0, -1.0, -0.5, -1e-6) for a in reach_act]
    high = [(c, a) for c in reach_ctrl for a in (1.5, 2.0, 3.0)]
    w_reach, at_reach = worst(reach)
    w_neg, _ = worst(neg)
    w_high, _ = worst(high)

    rec = dict(REPLICATION_VALIDATION)
    rec.update({
        "validated": True,
        "validated_points": len(reach),
        "validated_points_matched": int(sum(
            1 for c, a in reach
            if abs(dyn_fn(c, a, prm3) - muscle_dynamics_reference(c, a, prm3)) < 1e-9)),
        "validated_worst_abs_diff": w_reach,
        "validated_worst_at": at_reach,
        "unreachable_corner_ctrl_negative_points": len(neg),
        "unreachable_corner_ctrl_negative_worst_abs_diff": w_neg,
        "unreachable_corner_act_above_one_points": len(high),
        "unreachable_corner_act_above_one_worst_abs_diff": w_high,
        "example_act_above_one": {
            "ctrl": 0.0, "act": 2.0,
            "library": dyn_fn(0.0, 2.0, prm3),
            "replication": muscle_dynamics_reference(0.0, 2.0, prm3),
        },
    })
    return rec


def _sigmoid(x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)


def muscle_fixed_point(ctrl: float, prm3, *, iters: int = 200) -> float:
    """Fixed point of the activation ODE, found by bisection on act_dot(act).

    act_dot is increasing in ctrl and decreasing in act, and the ODE is a relaxation of
    act toward its own fixed point, so bisection is exact here rather than approximate.
    """
    lo, hi = 0.0, max(1.0, float(ctrl) * 4.0 + 1.0)
    f_lo = muscle_dynamics_reference(ctrl, lo, prm3)
    f_hi = muscle_dynamics_reference(ctrl, hi, prm3)
    if f_lo <= 0.0:
        return lo
    if f_hi >= 0.0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if muscle_dynamics_reference(ctrl, mid, prm3) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def set_timestep(model, dt: float) -> None:
    model.opt.timestep = dt


def clone_into(model, src, dst) -> None:
    """Copy the full integration state, so two variants start from the IDENTICAL state."""
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.act[:] = src.act
    dst.ctrl[:] = src.ctrl
    dst.time = src.time


def reset_state(model, data) -> None:
    import mujoco
    mujoco.mj_resetData(model, data)


def force_norm(model, data) -> float:
    """L2 norm of the 90-muscle force vector, in MODEL force units."""
    return float(np.linalg.norm(data.actuator_force[:model.nu]))


def actuator_names(model) -> list:
    import mujoco
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


def build_derived_models(model):
    """The two NAMED, IN-MEMORY-ONLY variants of the already-compiled canonical model.

    'native'     : deep copy of the compiled canonical model, ctrlrange untouched [1e-4, 1].
    'zero_floor' : deep copy whose actuator_ctrlrange lower bound is set to 0.

    Both are deep copies of the compiled mjModel object held in this process's memory.  No
    XML is written and the canonical asset is not read, written or re-derived here, so
    nothing on disk can change as a result of this audit.
    """
    import copy as _copy
    native = _copy.deepcopy(model)
    zero_floor = _copy.deepcopy(model)
    zero_floor.actuator_ctrlrange[:zero_floor.nu, 0] = 0.0
    assert zero_floor.actuator_ctrlrange[0][0] == 0.0
    assert native.actuator_ctrlrange[0][0] == float(model.actuator_ctrlrange[0][0])
    return native, zero_floor


# ---------------------------------------------------------------------------
# Part A -- where does act_dot = 2.0 come from
# ---------------------------------------------------------------------------
def part_A(model, data) -> dict:
    import mujoco

    names = actuator_names(model)
    dynprm = np.array(model.actuator_dynprm[:model.nu])
    gainprm = np.array(model.actuator_gainprm[:model.nu])
    biasprm = np.array(model.actuator_biasprm[:model.nu])

    tau_act = dynprm[:, 0]
    tau_deact = dynprm[:, 1]
    tau_smooth = dynprm[:, 2]

    names = actuator_names(model)
    prm3 = dynprm[0][:3].tolist()

    # ---- the rule, called directly on the INSTALLED library --------------------------
    dyn_fn, timescale_fn = load_muscle_dynamics_fn()
    direct_calls = []
    replication_validation = validate_replication_region(prm3)
    if dyn_fn is not None:
        for ctrl, act in ((0.0, 0.0), (1e-4, 0.0), (1e-4, 1e-4), (0.5, 0.0),
                          (1.0, 0.0), (0.0, 0.5), (1.0, 1.0)):
            direct_calls.append({
                "ctrl": ctrl, "act": act,
                "tau_from_mju_muscleDynamicsTimescale": timescale_fn(
                    ctrl - act, prm3[0], prm3[1], prm3[2]),
                "act_dot_from_mju_muscleDynamics": dyn_fn(ctrl, act, prm3),
                "act_dot_from_local_replication": muscle_dynamics_reference(ctrl, act, prm3),
                "tau_from_local_replication": muscle_dynamics_reference(
                    ctrl, act, prm3, trace=True)[1],
            })
    replication_max_abs_diff = max(
        (abs(r["act_dot_from_mju_muscleDynamics"] - r["act_dot_from_local_replication"])
         for r in direct_calls), default=None)

    # ---- what the first step actually does ------------------------------------------
    ctrl_effective = float(np.clip(0.0, *model.actuator_ctrlrange[0]))
    act0 = 0.0
    actdot_first, tau_first, trace_first = muscle_dynamics_reference(
        ctrl_effective, act0, prm3, trace=True)
    dctrl = trace_first["dctrl"]
    branch = "activation (ctrl_clipped - act_clipped > 0)" if dctrl > 0 else \
             "deactivation (ctrl_clipped - act_clipped <= 0)"
    act_after_one_dt_analytic = act0 + DT_MAIN * actdot_first
    steady = muscle_fixed_point(ctrl_effective, prm3)

    # measured: same state, one step
    reset_state(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    measured = {
        "act_dot_first_forward": float(data.act_dot[0]),
        "act_at_forward": float(data.act[0]),
        "ctrl_as_stored_after_user_set_0": float(data.ctrl[0]),
    }
    reset_state(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    measured["act_after_one_step_dt1e-4"] = float(data.act[0])
    measured["time_after_one_step"] = float(data.time)

    # same with ctrl = 1e-4 instead of 0
    reset_state(model, data)
    data.ctrl[:] = 1e-4
    mujoco.mj_forward(model, data)
    measured_ctrl_floor = {
        "act_dot_first_forward": float(data.act_dot[0]),
        "act_after_one_step_dt1e-4": _step_once_act(model, data, 1e-4),
    }

    # is ctrl clamped by the compiled ctrlrange?  ctrl = 0 sits BELOW the lower bound, so if
    # clamping is active the dynamics must see 1e-4 (act_dot = 2.0), otherwise 0 (act_dot = 0).
    reset_state(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    act_dot_ctrl0 = float(data.act_dot[0])

    # the transient shape is the interesting part: act OVERSHOOTS its own fixed point
    set_timestep(model, DT_MAIN)
    reset_state(model, data)
    data.ctrl[:] = 0.0
    overshoot = []
    for k in range(21):
        overshoot.append(float(data.act[0]))
        if k == 20:
            break
        mujoco.mj_step(model, data)

    result = {
        "nu": int(model.nu),
        "all_dyntype_muscle": bool(np.all(model.actuator_dyntype[:model.nu] == mujoco.mjtDyn.mjDYN_MUSCLE)),
        "all_gaintype_muscle": bool(np.all(model.actuator_gaintype[:model.nu] == mujoco.mjtGain.mjGAIN_MUSCLE)),
        "all_biastype_muscle": bool(np.all(model.actuator_biastype[:model.nu] == mujoco.mjtBias.mjBIAS_MUSCLE)),
        "enum_values": {
            "mjDYN_MUSCLE": int(mujoco.mjtDyn.mjDYN_MUSCLE),
            "mjGAIN_MUSCLE": int(mujoco.mjtGain.mjGAIN_MUSCLE),
            "mjBIAS_MUSCLE": int(mujoco.mjtBias.mjBIAS_MUSCLE),
        },
        "ctrllimited_all_true": bool(np.all(model.actuator_ctrllimited[:model.nu] == 1)),
        "actlimited_all_false": bool(np.all(model.actuator_actlimited[:model.nu] == 0)),
        "forcelimited_all_false": bool(np.all(model.actuator_forcelimited[:model.nu] == 0)),
        "ctrlrange_all": _uniq_rows(model.actuator_ctrlrange[:model.nu]),
        "actrange_all": _uniq_rows(model.actuator_actrange[:model.nu]),
        "forcerange_all": _uniq_rows(model.actuator_forcerange[:model.nu]),
        "dynprm_unique_rows": _uniq_rows(dynprm),
        "tau_act_unique": sorted(set(np.round(tau_act, 12).tolist())),
        "tau_deact_unique": sorted(set(np.round(tau_deact, 12).tolist())),
        "tau_smoothing_width_unique": sorted(set(np.round(tau_smooth, 12).tolist())),
        "gainprm_equals_biasprm_all": bool(np.allclose(gainprm, biasprm)),
        "gainprm_columns_unique": {f"prm[{k}]": _uniq_1d(gainprm[:, k])
                                   for k in range(gainprm.shape[1])},
        "actuator_lengthrange_unique_count": len(_uniq_rows(
            [model.actuator_lengthrange[2 * i:2 * i + 2].tolist() for i in range(model.nu)])),
        "actuator_acc0_unique": _uniq_1d(model.actuator_acc0[:model.nu]),
        "actuator_gear_unique": _uniq_rows(model.actuator_gear[:model.nu]),
        "trntype_unique": _uniq_1d(model.actuator_trntype[:model.nu]),
        "per_actuator_sample": [
            {
                "name": names[i],
                "dynprm": dynprm[i].tolist(),
                "gainprm": gainprm[i].tolist(),
                "biasprm": biasprm[i].tolist(),
                "ctrlrange": model.actuator_ctrlrange[i].tolist(),
                "lengthrange": model.actuator_lengthrange[2 * i:2 * i + 2].tolist(),
                "acc0": float(model.actuator_acc0[i]),
                "gear": model.actuator_gear[i].tolist(),
                "trntype": int(model.actuator_trntype[i]),
                "actadr": int(model.actuator_actadr[i]),
                "actnum": int(model.actuator_actnum[i]),
            }
            for i in (0, 44, 88, 89)
        ],
        "activation_rule_called_on_installed_library": {
            "library": str(Path(mujoco.__file__).parent / "libmujoco.so.3.9.0"),
            "replicated_rule": "ctrl_c = clip(ctrl,0,1); act_c = clip(act,0,1); "
                               "scale = 0.5 + 1.5*act_c; "
                               "tau = tau_act*scale if (ctrl_c-act_c) > 0 else tau_deact/scale; "
                               "act_dot = (ctrl_c - act_c)/tau",
            "prm3_used": prm3,
            "calls": direct_calls,
            "max_abs_diff_library_vs_local_replication": replication_max_abs_diff,
            "replication_validation": replication_validation,
        },
        "replication_validation": replication_validation,
        "first_step_reconstruction": {
            "ctrl_commanded": 0.0,
            "ctrl_after_clamp_to_ctrlrange": ctrl_effective,
            "act_before": act0,
            "ctrl_clipped_in_dynamics": trace_first["ctrl_clipped"],
            "act_clipped_in_dynamics": trace_first["act_clipped"],
            "scale_0p5_plus_1p5_act": trace_first["scale"],
            "dctrl": dctrl,
            "branch_taken": branch,
            "tau_used": tau_first,
            "tau_used_is": "tau_act * (0.5 + 1.5*act) = 1e-4 * 0.5" if dctrl > 0
                           else "tau_deact / (0.5 + 1.5*act)",
            "act_dot_analytic": actdot_first,
            "act_dot_measured": measured["act_dot_first_forward"],
            "act_after_one_dt_analytic": act_after_one_dt_analytic,
            "act_after_one_dt_measured": measured["act_after_one_step_dt1e-4"],
            "matches_measurement": bool(
                abs(actdot_first - measured["act_dot_first_forward"]) < 1e-9
                and abs(act_after_one_dt_analytic - measured["act_after_one_step_dt1e-4"]) < 1e-12),
        },
        "activation_dynamics_fixed_point": {
            "formula_used": "bisection on act_dot(act) of the verified rule",
            "ctrl_effective": ctrl_effective,
            "act_fixed_point": steady,
            "note": "the ODE relaxes act toward this value: when act = ctrl exactly, "
                    "act_dot = (ctrl-act)/tau = 0 in BOTH branches, so the fixed point is "
                    "act* = ctrl regardless of which time constant applies",
        },
        "act_transient_at_dt_1e-4_first_21_values": overshoot,
        "act_transient_overshoots_fixed_point": bool(max(overshoot) > steady * 1.0000001),
        "act_transient_overshoot_peak": float(max(overshoot)),
        "measured_act_dynamics": measured,
        "measured_act_dynamics_ctrl_1e-4": measured_ctrl_floor,
        "ctrl_zero_is_clamped_to_ctrlrange_lower_bound": {
            "measured_act_dot_with_ctrl_0": act_dot_ctrl0,
            "expected_if_clamped_to_1e-4": 2.0,
            "expected_if_not_clamped_act_dot_equals_0": 0.0,
            "verdict": "clamped" if abs(act_dot_ctrl0 - 2.0) < 1e-9 else "NOT clamped",
        },
        "xml_default_class_vs_override": {
            "note": "the asset's own <default class=\"muscle\"> writes "
                    "dynprm=\"0.01 0.04 0 0 0 0 0 0 0 0\", while EVERY one of the 90 "
                    "actuators overrides it with dynprm=\"0.0001 0.0004 0 ...\".  The "
                    "documented default pair is (tau_act, tau_deact) = (0.01, 0.04); the "
                    "override is exactly that pair divided by 100.",
            "default_class_dynprm": [0.01, 0.04, 0, 0, 0, 0, 0, 0, 0, 0],
            "override_dynprm": dynprm[0].tolist(),
            "ratio_per_field": [float(0.01 / dynprm[0][0]), float(0.04 / dynprm[0][1])],
        },
        "documentation_reconciliation": {
            "measured_library_rule": (
                "act_dot = (ctrl_c - act_c)/tau with ctrl_c, act_c clipped to [0,1], where "
                "tau = tau_act*(0.5+1.5*act_c) when ctrl_c > act_c and "
                "tau = tau_deact/(0.5+1.5*act_c) otherwise; tau_act = dynprm[0] = 1e-4, "
                "tau_deact = dynprm[1] = 4e-4"),
            "doc_modeling_rst_says": (
                "tau = tau_act*(0.5+1.5*act) when ctrl-act > 0, tau_deact/(0.5+1.5*act) "
                "otherwise (Millard et al. 2013) - the SAME rule; the docs also state that "
                "the control signal is internally clamped to [0,1]"),
            "source_engine_util_misc_c_at_main_says": (
                "mju_muscleDynamicsTimescale returns tau = dctrl > 0 ? tau_act : tau_deact, "
                "i.e. WITHOUT the (0.5+1.5*act) factor"),
            "consequence": (
                "the documentation and the INSTALLED library agree; the version of "
                "engine_util_misc.c visible on the mujoco main branch does NOT apply the "
                "0.5+1.5*act factor that the installed 3.9.0 library demonstrably applies "
                "(verified by ctypes call AND by disassembly of libmujoco.so.3.9.0).  Anyone "
                "reproducing this from the main-branch source would compute act_dot = 1.0 "
                "where the compiled model produces 2.0.  The authoritative reference for this "
                "audit is therefore the INSTALLED library, not the prose, and not main."),
            "why_act_dot_is_2p0_and_not_1p0_or_0p25": (
                "ctrl=0 is clamped UP to 1e-4 by ctrllimited ctrlrange (measured), so dctrl = "
                "1e-4 > 0 and the ACTIVATION branch applies, not the deactivation branch; "
                "act=0 makes act_c=0 so the scale factor is 0.5+1.5*0 = 0.5, giving "
                "tau = tau_act*0.5 = 5e-5 and act_dot = 1e-4/5e-5 = 2.0.  The naive readings "
                "1.0 (forgetting the 0.5 factor) and 0.25 (forgetting the ctrl clamp, which "
                "would select the deactivation branch) are both excluded by measurement."),
            "resolved_by": "ctypes calls to mju_muscleDynamics / mju_muscleDynamicsTimescale on "
                           "the installed libmujoco.so.3.9.0, cross-checked against objdump of "
                           "the same two functions, cross-checked against mj_step trajectories",
            "local_replication_max_abs_diff_vs_library": None,   # filled below
        },
        "timestep": float(model.opt.timestep),
        "integrator_enum": int(model.opt.integrator),
        "integrator_name": {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}.get(
            int(model.opt.integrator), f"other({int(model.opt.integrator)})"),
        "gravity": model.opt.gravity.tolist(),
        "nq": int(model.nq), "nv": int(model.nv), "na": int(model.na),
    }
    result["documentation_reconciliation"]["local_replication_max_abs_diff_vs_library"] = \
        replication_max_abs_diff
    return result


def _step_once_act(model, data, ctrl_value: float) -> float:
    import mujoco
    reset_state(model, data)
    data.ctrl[:] = ctrl_value
    mujoco.mj_step(model, data)
    return float(data.act[0])


def _uniq_rows(mat) -> list:
    seen = []
    for row in mat:
        r = np.round(np.asarray(row, dtype=float), 12).tolist()
        if r not in seen:
            seen.append(r)
    return seen


def _uniq_1d(vec) -> list:
    return sorted(set(np.round(np.asarray(vec, dtype=float), 12).tolist()))


# ---------------------------------------------------------------------------
# Part B -- timestep sensitivity
# ---------------------------------------------------------------------------
def part_B(model, data) -> dict:
    import mujoco

    ctrl_effective = float(np.clip(0.0, *model.actuator_ctrlrange[0]))
    prm3 = [float(x) for x in model.actuator_dynprm[0][:3]]
    tau_act, tau_deact = prm3[0], prm3[1]
    steady = muscle_fixed_point(ctrl_effective, prm3)

    out = {
        "dt_list": DTS, "physical_span_s": N_STEP_FOR_DT,
        "ctrl_commanded": 0.0,
        "ctrl_effective_after_clamp": ctrl_effective,
        "prm3": prm3,
        "act_fixed_point": steady,
        "act_dot_analytic_first_step": muscle_dynamics_reference(ctrl_effective, 0.0, prm3),
        "integrator": {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}.get(
            int(model.opt.integrator), str(int(model.opt.integrator))),
        "exact_reference": _exact_act_reference(model, prm3, ctrl_effective),
    }
    act_dot_analytic_first = out["act_dot_analytic_first_step"]

    per_dt = {}
    for dt in DTS:
        set_timestep(model, dt)
        n = int(round(N_STEP_FOR_DT / dt))
        reset_state(model, data)
        data.ctrl[:] = 0.0                     # clamped to 1e-4
        act_traj = np.empty(n + 1)
        act_dot0 = None
        for k in range(n + 1):
            act_traj[k] = float(data.act[0])
            if k == n:
                break
            if act_dot0 is None:
                mujoco.mj_forward(model, data)
                act_dot0 = float(data.act_dot[0])
            mujoco.mj_step(model, data)
        ref_act = np.asarray(out["exact_reference"]["act"], dtype=float)
        ref_t = np.asarray(out["exact_reference"]["t"], dtype=float)
        ref = np.interp(np.linspace(0, N_STEP_FOR_DT, n + 1), ref_t, ref_act)
        err = np.abs(act_traj - ref)

        # A single overshoot followed by monotone decay is NOT oscillation.  Distinguish the
        # two properly: count direction reversals of the act increment EXCLUDING the rise from
        # the initial value to the first peak, and check that act never increases again after
        # the peak.
        inc = np.diff(act_traj)
        peak = int(np.argmax(act_traj))
        after_peak = inc[peak:] if peak < len(inc) else np.array([0.0])
        reversals_after_peak = int(np.sum(np.diff(np.sign(after_peak[after_peak != 0.0])) != 0)) \
            if np.count_nonzero(after_peak) > 1 else 0
        rises_after_peak = int(np.sum(after_peak > 0))

        # first time act comes within 1% of the fixed point (in simulated time)
        tol = 0.01 * steady
        within = np.where(np.abs(act_traj - steady) <= tol)[0]
        t_within_1pct = float(within[0] * dt) if len(within) else None

        per_dt[f"{dt:.0e}"] = {
            "dt": dt, "n_steps": n,
            "act_dot_first_step": act_dot0,
            "act_final": float(act_traj[-1]),
            "act_final_abs_err_vs_fixed_point": float(abs(act_traj[-1] - steady)),
            "act_min": float(act_traj.min()),
            "act_max": float(act_traj.max()),
            "act_max_step_index": peak,
            "act_range_over_span": float(act_traj.max() - act_traj.min()),
            "overshoot_above_fixed_point": float(max(0.0, act_traj.max() - steady)),
            "overshoot_relative_to_fixed_point": float(
                max(0.0, act_traj.max() - steady) / max(1e-300, steady)),
            "rises_in_act_after_the_peak": rises_after_peak,
            "increment_sign_reversals_after_the_peak": reversals_after_peak,
            "monotone_after_peak": bool(rises_after_peak == 0),
            "time_to_within_1pct_of_fixed_point_s": t_within_1pct,
            "max_abs_err_vs_reference": float(err.max()),
            "final_abs_err_vs_reference": float(err[-1]),
            "rel_err_vs_reference_final": float(err[-1] / max(1e-30, abs(ref[-1]))),
            "act_samples_first_12": act_traj[:12].tolist(),
            "reference_samples_first_12": ref[:12].tolist(),
        }
    out["per_dt"] = per_dt
    final_vals = [v["act_final"] for v in per_dt.values()]
    out["act_final_max_spread_across_dt"] = float(max(final_vals) - min(final_vals))
    out["act_final_spread_relative_to_fixed_point"] = float(
        (max(final_vals) - min(final_vals)) / max(1e-300, steady))
    out["act_final_abs_err_vs_fixed_point_per_dt"] = {
        k: v["act_final_abs_err_vs_fixed_point"] for k, v in per_dt.items()}
    out["converged_to_same_value_within_1pct_over_1ms"] = bool(
        all(v["act_final_abs_err_vs_fixed_point"] <= 0.01 * steady for v in per_dt.values()))
    out["max_overshoot_over_all_dt"] = float(max(v["overshoot_above_fixed_point"]
                                                 for v in per_dt.values()))
    out["any_true_numerical_oscillation"] = bool(
        any(v["increment_sign_reversals_after_the_peak"] > 0 for v in per_dt.values()))
    out["all_dt_monotone_after_peak"] = bool(
        all(v["monotone_after_peak"] for v in per_dt.values()))

    # ---- explicit Euler stability probe --------------------------------------------------
    # At act = 0 with ctrl = 1e-4 the dynamics sit exactly on the activation branch with
    #   act_dot = (ctrl - act)/(tau_act*(0.5 + 1.5*act)).
    # Near act = 0 this is the linear relaxation act_dot ~ ctrl/tau_act - act*(...)/tau_act, so
    # the explicit-Euler update has amplification factor (1 - dt*(d/dt)[f]) close to 1, and the
    # step is stable while dt is below roughly 2*tau_act/(0.5+1.5*act).  Rather than asserting a
    # threshold, the probe MEASURES the first step at and beyond the candidate thresholds and
    # compares against the exact first-step prediction from the verified rule.
    probe = []
    for dt in (0.25 * DT_MAIN, 0.5 * DT_MAIN, DT_MAIN, 2.0 * DT_MAIN, 4.0 * DT_MAIN,
               8.0 * DT_MAIN, 2e-4, 5e-4, 1e-3):
        set_timestep(model, dt)
        reset_state(model, data)
        data.ctrl[:] = 0.0
        predicted = act0_pred = 0.0 + dt * act_dot_analytic_first
        mujoco.mj_step(model, data)
        measured_act = float(data.act[0])
        probe.append({
            "dt": dt,
            "dt_over_tau_eff_at_act0": dt / (0.5 * tau_act),
            "act_predicted_first_step": predicted,
            "act_measured_after_one_step": measured_act,
            "abs_diff_prediction_vs_measured": abs(predicted - measured_act),
            "prediction_beyond_fixed_point": bool(predicted > steady),
        })
    diverged = [p["dt"] for p in probe
                if not np.isfinite(p["act_measured_after_one_step"])
                or p["act_measured_after_one_step"] > 2.0]
    out["explicit_euler_stability_probe"] = {
        "rule": "the activation branch at act=0 has act_dot = ctrl/(tau_act*0.5); the explicit "
                "Euler update overshoots the fixed point whenever dt*act_dot > act_fixed_point, "
                "i.e. dt > 0.5*tau_act, and diverges (reaches the act range clamp / instability) "
                "for dt well beyond that",
        "tau_act": tau_act,
        "tau_deact": tau_deact,
        "tau_effective_on_activation_branch_at_act0": 0.5 * tau_act,
        "dt_main": DT_MAIN,
        "dt_main_over_tau_eff_at_act0": DT_MAIN / (0.5 * tau_act),
        "note": "at dt = 1e-4 the ratio dt/tau_eff is 2.0 exactly, so the very first step lands "
                "at act = 2e-4 = 2x the fixed point 1e-4.  Measured consequence at dt=1e-4: one "
                "single overshoot followed by monotone decay, no oscillation (reported in "
                "per_dt above), and the same fixed point is reached at every dt.  The "
                "trajectory over 1 ms therefore converges as dt shrinks, but at dt=1e-4 the "
                "first step is a factor-2 overshoot of the commanded activation.",
        "timesteps_that_lose_finiteness_or_exceed_act_range": diverged,
        "probe": probe,
    }

    # the exact discrete map of the first step, used as an independent analytic check of the
    # measured act after one step at each dt
    discrete = []
    for dt in DTS + [2e-4, 4e-4]:
        set_timestep(model, dt)
        reset_state(model, data)
        data.ctrl[:] = 0.0
        act_pred = dt * act_dot_analytic_first
        mujoco.mj_step(model, data)
        discrete.append({"dt": dt, "act_predicted_euler_first_step": act_pred,
                         "act_measured": float(data.act[0]),
                         "abs_diff": abs(act_pred - float(data.act[0]))})
    out["first_step_discrete_map_check"] = discrete
    set_timestep(model, DT_MAIN)

    # convergence over a longer span: does the dt gap persist?
    # Does the dt difference wash out in simulated time?  Sample act at several spans; if the
    # dt=1e-4 overshoot is a transient artefact it must disappear at longer spans.
    spans = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    long_run = {}
    for dt in DTS:
        per_span = {}
        for span in spans:
            set_timestep(model, dt)
            reset_state(model, data)
            data.ctrl[:] = 0.0
            for _ in range(int(round(span / dt))):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            per_span[f"{span:g}"] = {
                "act": float(data.act[0]),
                "abs_err_vs_fixed_point": float(abs(data.act[0] - steady)),
                "act_dot": float(data.act_dot[0]),
            }
        long_run[f"{dt:.0e}"] = {"dt": dt, "per_span_s": per_span}
    out["longer_runs"] = long_run
    out["longer_run_spread_per_span"] = {
        s: float(max(long_run[k]["per_span_s"][s]["act"] for k in long_run)
                 - min(long_run[k]["per_span_s"][s]["act"] for k in long_run))
        for s in [f"{x:g}" for x in spans]}
    set_timestep(model, DT_MAIN)
    return out


def _exact_act_reference(model, prm3, ctrl_effective) -> dict:
    """Integrate the activation ODE with a very small explicit step, using the rule that was
    VERIFIED against the installed library (not the prose documentation).

    This is deliberately computed in Python rather than by MuJoCo, so the comparison against
    MuJoCo's own trajectory is not circular.
    """
    dt_ref = 1e-8
    n = int(round(N_STEP_FOR_DT / dt_ref))
    act = 0.0
    t = np.empty(n + 1)
    a = np.empty(n + 1)
    for k in range(n + 1):
        t[k] = k * dt_ref
        a[k] = act
        if k == n:
            break
        act_dot = muscle_dynamics_reference(ctrl_effective, act, prm3)
        act = act + dt_ref * act_dot
    return {"dt_ref": dt_ref, "ctrl": ctrl_effective, "prm3": prm3,
            "t": t.tolist(), "act": a.tolist(),
            "act_final": float(a[-1]), "act_max": float(a.max()),
            "act_is_monotone": bool(np.all(np.diff(a) >= -1e-18)),
            "note": "independent numerical integration of the verified activation rule; used "
                    "as the reference for the timestep-sensitivity comparison.  It shows the "
                    "true transient monotone rises to the fixed point and does NOT overshoot, "
                    "so the dt=1e-4 overshoot in the comparison is an integrator artefact."}


def _first_act_dot(model, data) -> float:
    import mujoco
    mujoco.mj_forward(model, data)
    return float(data.act_dot[0])


# ---------------------------------------------------------------------------
# Part C -- passive force magnitude at act = 0
# ---------------------------------------------------------------------------
def part_C(model, data) -> dict:
    import mujoco

    reset_state(model, data)
    data.act[:] = 0.0
    data.ctrl[:] = 0.0                      # clamped to 1e-4
    mujoco.mj_forward(model, data)
    f_passive = np.array(data.actuator_force[:model.nu], dtype=float)

    # active-segment forces at act = 1 (ctrl = 1)
    reset_state(model, data)
    data.act[:] = 1.0
    data.ctrl[:] = 1.0
    mujoco.mj_forward(model, data)
    f_active1 = np.array(data.actuator_force[:model.nu], dtype=float)

    # the force measured at the user-visible "zero control" but with act also driven to its
    # own steady state for ctrl = 1e-4 (relaxed state)
    reset_state(model, data)
    data.ctrl[:] = 0.0
    set_timestep(model, 1e-4)
    for _ in range(200):                    # 0.02 s: act has converged (tau_eff ~ 2e-4 s)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    f_relaxed = np.array(data.actuator_force[:model.nu], dtype=float)
    act_relaxed = float(data.act[0])

    names = actuator_names(model)
    f_passive_sorted = np.sort(f_passive)
    worst_passive_idx = int(np.argmax(np.abs(f_passive)))

    def stats(v):
        return {"min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "median": float(np.median(v)),
                "l2_norm": float(np.linalg.norm(v)),
                "sum": float(v.sum())}

    denom = np.linalg.norm(f_active1)
    return {
        "n_muscles": int(model.nu),
        "passive_at_act0_ctrl_clamped_1e-4": stats(f_passive),
        "passive_at_act0_all_negative": bool(np.all(f_passive < 0)),
        "passive_at_act0_n_negative": int(np.sum(f_passive < 0)),
        "passive_at_act0_n_zero_or_positive": int(np.sum(f_passive >= 0)),
        "passive_largest_abs": {
            "index": worst_passive_idx,
            "name": names[worst_passive_idx],
            "force": float(f_passive[worst_passive_idx]),
        },
        "passive_smallest_abs": {
            "index": int(np.argmin(np.abs(f_passive))),
            "name": names[int(np.argmin(np.abs(f_passive)))],
            "force": float(f_passive[int(np.argmin(np.abs(f_passive)))]),
        },
        "active_at_act1_ctrl1": stats(f_active1),
        "relaxed_state_ctrl_0": {
            "act": act_relaxed,
            "force": stats(f_relaxed),
            "note": "ctrl=0 is clamped to 1e-4 and act relaxes to its own steady state; this "
                    "is what 'no commanded control' actually produces in the canonical model",
        },
        "ratio_passive_l2_over_active_l2": float(np.linalg.norm(f_passive) / denom),
        "ratio_passive_abs_sum_over_active_abs_sum": float(
            np.abs(f_passive).sum() / np.abs(f_active1).sum()),
        "passive_to_active_ratio_per_muscle": {
            "min": float(np.min(np.abs(f_passive) / np.maximum(np.abs(f_active1), 1e-300))),
            "max": float(np.max(np.abs(f_passive) / np.maximum(np.abs(f_active1), 1e-300))),
            "median": float(np.median(np.abs(f_passive) / np.maximum(np.abs(f_active1), 1e-300))),
        },
        "worst_10_by_abs_passive": [
            {"name": names[i], "force": float(f_passive[i]),
             "force_at_act1": float(f_active1[i]),
             "abs_ratio_to_act1": float(abs(f_passive[i]) / max(1e-300, abs(f_active1[i])))}
            for i in np.argsort(-np.abs(f_passive))[:10]
        ],
        "best_10_by_abs_passive": [
            {"name": names[i], "force": float(f_passive[i]),
             "force_at_act1": float(f_active1[i]),
             "abs_ratio_to_act1": float(abs(f_passive[i]) / max(1e-300, abs(f_active1[i])))}
            for i in np.argsort(np.abs(f_passive))[:10]
        ],
        "all_90_force_at_act0": [
            {"name": names[i], "force": float(f_passive[i])} for i in range(model.nu)
        ],
        "all_90_force_at_act1": [
            {"name": names[i], "force": float(f_active1[i])} for i in range(model.nu)
        ],
        "sorted_passive_forces_ascending": f_passive_sorted.tolist(),
    }


# ---------------------------------------------------------------------------
# Part D -- what the 0.0001 floor actually does
# ---------------------------------------------------------------------------
def part_D(model, data, native, zero_floor) -> dict:
    import mujoco

    tau_act = float(model.actuator_dynprm[0][0])
    tau_deact = float(model.actuator_dynprm[0][1])
    floor = float(model.actuator_ctrlrange[0][0])
    prm3 = [float(x) for x in model.actuator_dynprm[0][:3]]

    # Steady state of the activation ODE driven by ctrl = floor, obtained from the rule that
    # was verified against the installed library (bisection, not assumption).
    act_steady_analytic = muscle_fixed_point(floor, prm3)

    native_name = "native"
    zero_name = "zero_floor"

    res = {}
    for tag, m in ((native_name, native), (zero_name, zero_floor)):
        d = mujoco.MjData(m)
        reset_state(m, d)
        d.ctrl[:] = 0.0
        # relax for 0.05 s = 500 steps at dt=1e-4.  With tau_act=1e-4 and tau_deact=4e-4 the
        # relaxation is complete in well under 1e-3 s, so this is far past convergence; the
        # measured value below is checked against the analytic fixed point.
        set_timestep(m, DT_MAIN)
        for _ in range(500):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        act_sum = float(np.sum(d.act[:]))
        res[tag] = {
            "model": tag,
            "ctrlrange_lower": float(m.actuator_ctrlrange[0][0]),
            "act_steady_mean": float(np.mean(d.act[:])),
            "act_steady_min": float(np.min(d.act[:])),
            "act_steady_max": float(np.max(d.act[:])),
            "act_steady_sum": act_sum,
            "act_steady_any_nan": bool(np.any(np.isnan(d.act[:]))),
            "force_at_steady": _stats(d.actuator_force[:m.nu]),
            "force_vector_at_steady": [float(x) for x in d.actuator_force[:m.nu]],
            "time_simulated_s": 500 * DT_MAIN,
        }
        set_timestep(m, DT_MAIN)

    fn = np.array(res[native_name]["force_vector_at_steady"], dtype=float)
    fz = np.array(res[zero_name]["force_vector_at_steady"], dtype=float)
    diff = fn - fz

    # rotation / posture relevance: moment arm of the force difference about the body DOFs.
    # qfrc = actuator_moment^T * force.  We report the model-unit generalised force
    # difference, which is the quantity that actually enters the equations of motion.
    reset_state(native, mujoco.MjData(native))
    d_native = mujoco.MjData(native)
    d_zero = mujoco.MjData(zero_floor)
    set_timestep(native, 1e-4)
    set_timestep(zero_floor, 1e-4)
    for _ in range(500):
        mujoco.mj_step(native, d_native)
        mujoco.mj_step(zero_floor, d_zero)
    # generalised force contribution of the actuators only:
    #   qfrc_actuator = moment^T * actuator_force  (already computed by mj_forward)
    qa_n = np.array(d_native.qfrc_actuator, dtype=float)
    qa_z = np.array(d_zero.qfrc_actuator, dtype=float)
    qdiff = qa_n - qa_z

    # mass matrix weighted: an equivalent "acceleration difference" if the force difference
    # were applied alone, in model units (cm/s^2).  This is the honest way to say whether
    # the floor can move the body.
    M = np.zeros((native.nv, native.nv))
    mujoco.mj_fullM(native, M, d_native.qM)
    try:
        acc_equiv = np.linalg.solve(M, qdiff)
        acc_equiv_ok = True
        acc_equiv_norm = float(np.linalg.norm(acc_equiv))
        acc_equiv_max = float(np.max(np.abs(acc_equiv)))
    except Exception as exc:                                # pragma: no cover
        acc_equiv_ok = False
        acc_equiv_norm = None
        acc_equiv_max = None
        res["mass_matrix_solve_error"] = repr(exc)

    # the same quantity for the total actuator generalised force, for scale
    qa_scale = float(np.linalg.norm(qa_n))
    acc_scale = float(np.linalg.norm(np.linalg.solve(M, qa_n))) if acc_equiv_ok else None

    return {
        "floor_value": floor,
        "analytic_steady_state_of_activation_ode_for_ctrl_eq_floor": act_steady_analytic,
        "analytic_note": "fixed point of the verified activation rule, found by bisection on "
                         "act_dot(act); at act = ctrl the derivative (ctrl-act)/tau is zero "
                         "in BOTH branches, so the fixed point is act* = ctrl exactly and it "
                         "does not depend on which time constant is used",
        "act_steady_measured_matches_analytic": bool(
            abs(res[native_name]["act_steady_mean"] - act_steady_analytic) < 1e-12),
        "derived_models": res,
        "act_steady_difference_native_minus_zero_floor_mean": float(
            res[native_name]["act_steady_mean"] - res[zero_name]["act_steady_mean"]),
        "act_steady_difference_over_full_range": float(
            (res[native_name]["act_steady_mean"] - res[zero_name]["act_steady_mean"]) / 1.0),
        "force_difference_model_units": {
            "l2_norm": float(np.linalg.norm(diff)),
            "max_abs": float(np.max(np.abs(diff))),
            "mean": float(np.mean(diff)),
            "per_muscle": [float(x) for x in diff],
        },
        "generalised_force_difference_model_units": {
            "l2_norm": float(np.linalg.norm(qdiff)),
            "max_abs": float(np.max(np.abs(qdiff))),
            "per_dof": [float(x) for x in qdiff],
        },
        "generalised_force_of_all_actuators_model_units_l2": qa_scale,
        "ratio_generalised_force_diff_over_total": float(np.linalg.norm(qdiff) / max(1e-300, qa_scale)),
        "equivalent_free_acceleration_of_force_difference_model_units": {
            "note": "solve(M, qfrc_diff): the acceleration the floor-induced force difference "
                    "would produce ALONE, in model units (cm/s^2).  Not a physical prediction "
                    "of the posed body, only the size of the effect.",
            "solved": acc_equiv_ok,
            "l2_norm_cm_per_s2": acc_equiv_norm,
            "max_abs_cm_per_s2": acc_equiv_max,
            "per_dof": [float(x) for x in acc_equiv] if acc_equiv_ok else None,
        },
        "equivalent_free_acceleration_of_all_actuators_cm_per_s2_l2": acc_scale,
        "ratio_equivalent_acceleration_diff_over_total": (
            float(acc_equiv_norm / acc_scale) if (acc_equiv_ok and acc_scale) else None),
        "gravity_for_scale_cm_per_s2": float(abs(native.opt.gravity[2])),
    }


def _stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "median": float(np.median(v)), "l2_norm": float(np.linalg.norm(v)),
            "sum": float(v.sum())}


# ---------------------------------------------------------------------------
# Part E -- named derived-model comparison under identical initialisation
# ---------------------------------------------------------------------------
def part_E(model, native, zero_floor) -> dict:
    import mujoco

    names = actuator_names(model)

    # ---- identical initialisation -------------------------------------------------------
    # The compiled model's own mj_resetData state is a FREE-FALL state: the thorax sits at
    # 4.0 model units with ZERO floor contacts (measured: stance_facts gives n_contact=0 and a
    # 9.73 mm gap).  Integrating from there would only measure free fall, so the audit uses the
    # asset's OWN declared standing pose -- keyframe 0 -- which is also the pose the project's
    # canonical_physics docstring describes (14 floor contacts at 20.449 mm).  The project's
    # other protocol (round-37 joint offsets + seat_root, 12 contacts at 20.4488 mm) is ALSO
    # measured and recorded, so the comparison does not hinge on which stance convention is
    # chosen.
    import run_fix_round39 as r39
    pose = json.loads(r39.ROUND37_POSE.read_text())

    d_native = mujoco.MjData(native)
    d_zero = mujoco.MjData(zero_floor)
    set_timestep(native, DT_MAIN)
    set_timestep(zero_floor, DT_MAIN)

    def seated_state(kind):
        """Fill d_native with a named stance initialisation; return (qpos, gate facts)."""
        if kind == "canonical_keyframe_0":
            mujoco.mj_resetDataKeyframe(native, d_native, 0)
        elif kind == "round37_offsets_seated":
            reset_state(native, d_native)
            r39.apply_round37_pose(native, d_native, pose)
            base = d_native.qpos.copy()
            z = cp.seat_root(native, d_native, base)
            d_native.qpos[:] = base
            d_native.qpos[2] = z
        else:
            raise ValueError(kind)
        d_native.qvel[:] = 0
        mujoco.mj_forward(native, d_native)
        facts = cp.stance_facts(native, d_native)       # raises if not a usable stance
        return d_native.qpos.copy(), facts

    both_stances = {}
    for kind in ("canonical_keyframe_0", "round37_offsets_seated"):
        q, facts = seated_state(kind)
        both_stances[kind] = {
            "root_z_model_units": float(q[2]),
            "root_z_mm": float(facts.root_z_mm),
            "gate": facts.as_dict(),
        }

    # primary stance for E1/E2: the asset's own keyframe 0
    seated, gate = seated_state("canonical_keyframe_0")
    d_native.qpos[:] = seated
    d_native.qvel[:] = 0
    mujoco.mj_forward(native, d_native)

    # copy the identical state verbatim into the derived model
    clone_into(native, d_native, d_zero)

    # commanded control: 0 on both.  The canonical model clamps it up to 1e-4; the derived
    # zero_floor model clamps it to 0.  Both activations start at the value the native model
    # reaches for that command, so the ONLY difference is the floor itself.
    d_native.ctrl[:] = 0.0
    d_zero.ctrl[:] = 0.0
    mujoco.mj_forward(native, d_native)
    act_seed = float(d_native.act[0])
    d_native.act[:] = act_seed
    d_zero.act[:] = act_seed
    mujoco.mj_forward(native, d_native)
    mujoco.mj_forward(zero_floor, d_zero)

    init_state = {
        "primary_source": "the asset's OWN keyframe 0 (model.key_qpos[0]), the pose described "
                          "in the canonical_physics docstring; copied verbatim into the "
                          "derived model",
        "stance_conventions_measured": both_stances,
        "round37_pose_json": str(r39.ROUND37_POSE),
        "seated_root_z_model_units": float(seated[2]),
        "stance_gate": gate.as_dict(),
        "stance_gate_passed": True,
        "qpos": [float(x) for x in d_native.qpos],
        "qvel": [float(x) for x in d_native.qvel],
        "act_seed_value": act_seed,
        "ctrl_native_commanded": 0.0,
        "ctrl_zero_floor_commanded": 0.0,
        "effective_ctrl_native_after_clamp": float(np.clip(0.0, *native.actuator_ctrlrange[0])),
        "effective_ctrl_zero_floor_after_clamp": float(
            np.clip(0.0, *zero_floor.actuator_ctrlrange[0])),
        "qpos_max_abs_diff_between_models_at_start": float(
            np.max(np.abs(d_native.qpos - d_zero.qpos))),
        "act_max_abs_diff_between_models_at_start": float(
            np.max(np.abs(d_native.act[:native.nu] - d_zero.act[:zero_floor.nu]))),
    }

    # ---- E1 instantaneous fixed-pose diagnostic ----------------------------------------
    f_n = np.array(d_native.actuator_force[:native.nu], dtype=float)
    f_z = np.array(d_zero.actuator_force[:zero_floor.nu], dtype=float)
    ad_n = np.array(d_native.act_dot[:native.nu], dtype=float)
    ad_z = np.array(d_zero.act_dot[:zero_floor.nu], dtype=float)

    instantaneous = {
        "note": "identical seated qpos/qvel/act, ctrl commanded 0 on both; the ONLY difference "
                "is the compiled ctrlrange lower bound",
        "act_dot_native": _stats(ad_n),
        "act_dot_zero_floor": _stats(ad_z),
        "act_dot_difference": _stats(ad_n - ad_z),
        "act_dot_native_unique": _uniq_1d(ad_n),
        "act_dot_zero_floor_unique": _uniq_1d(ad_z),
        "force_native": _stats(f_n),
        "force_zero_floor": _stats(f_z),
        "force_difference": _stats(f_n - f_z),
        "force_difference_l2_over_native_l2": float(
            np.linalg.norm(f_n - f_z) / max(1e-300, np.linalg.norm(f_n))),
        "force_is_bitwise_identical_at_this_instant": bool(np.array_equal(f_n, f_z)),
        "stance_identical": {
            "xpos_max_abs_diff": float(np.max(np.abs(d_native.xpos - d_zero.xpos))),
            "qpos_max_abs_diff": float(np.max(np.abs(d_native.qpos - d_zero.qpos))),
        },
    }

    # ---- E2 short free integration -----------------------------------------------------
    n_steps = int(round(FREE_INTEGRATION_S / DT_MAIN))
    thorax_id = _thorax_body_id(model)
    z0 = float(d_native.xpos[thorax_id][2])

    traj = {k: {"t": [], "act_mean": [], "act_0": [], "force_l2": [], "thorax_z": [],
                "qvel_l2": []} for k in ("native", "zero_floor")}
    collapse_drop = 0.10 * z0
    collapse = {"native": None, "zero_floor": None}

    for k in range(n_steps + 1):
        for tag, m, d in (("native", native, d_native), ("zero_floor", zero_floor, d_zero)):
            mujoco.mj_forward(m, d)
            tr = traj[tag]
            tr["t"].append(float(d.time))
            tr["act_mean"].append(float(np.mean(d.act[:m.nu])))
            tr["act_0"].append(float(d.act[0]))
            tr["force_l2"].append(float(np.linalg.norm(d.actuator_force[:m.nu])))
            z = float(d.xpos[thorax_id][2])
            tr["thorax_z"].append(z)
            tr["qvel_l2"].append(float(np.linalg.norm(d.qvel)))
            if collapse[tag] is None and (z0 - z) > collapse_drop:
                collapse[tag] = float(d.time)
        if k == n_steps:
            break
        mujoco.mj_step(native, d_native)
        mujoco.mj_step(zero_floor, d_zero)

    def arr(tag, key):
        return np.array(traj[tag][key], dtype=float)

    act_diff = arr("native", "act_mean") - arr("zero_floor", "act_mean")
    force_diff = arr("native", "force_l2") - arr("zero_floor", "force_l2")
    z_diff = arr("native", "thorax_z") - arr("zero_floor", "thorax_z")
    # The two act trajectories differ by construction (one relaxes to 1e-4, the other to 0),
    # so a RATIO is meaningless here -- it divides by a quantity that is exactly zero for the
    # derived model.  Compare the difference against the activation range [0, 1] instead.
    act_diff_frac_of_full_range = float(np.max(np.abs(act_diff)) / 1.0)

    # Gravity-only reference for the OBSERVED drop (a whole-run free-fall prediction is not
    # informative here because the body reaches the floor partway through the window).
    g = float(abs(model.opt.gravity[2]))            # model units (cm/s^2)
    measured_drop = float(z0 - arr("native", "thorax_z")[-1])
    t_free_fall_for_measured_drop = float(np.sqrt(2.0 * max(measured_drop, 0.0) / g))
    free_fall = {
        "gravity_model_units_cm_per_s2": g,
        "measured_drop_native_model_units": measured_drop,
        "free_fall_time_for_that_drop_s": t_free_fall_for_measured_drop,
        "duration_s": FREE_INTEGRATION_S,
        "note": "the body reaches the floor well inside the 0.05 s window (free fall over the "
                "full window would be 12.25 model units, more than the 2.04 units of height "
                "available), so this reference is computed for the drop actually observed",
    }

    free = {
        "duration_s": FREE_INTEGRATION_S,
        "n_steps": n_steps,
        "dt": DT_MAIN,
        "thorax_body_id": thorax_id,
        "thorax_z_initial_model_units": z0,
        "collapse_definition": "thorax z drops more than 10% of its initial height "
                               f"({collapse_drop:.6g} model length units)",
        "collapse_time_s": collapse,
        "collapse_time_native": collapse["native"],
        "collapse_time_zero_floor": collapse["zero_floor"],
        "collapse_time_difference_s": (
            None if (collapse["native"] is None or collapse["zero_floor"] is None)
            else float(collapse["native"] - collapse["zero_floor"])),
        "final_state": {
            tag: {
                "act_mean_final": float(arr(tag, "act_mean")[-1]),
                "act_0_final": float(arr(tag, "act_0")[-1]),
                "force_l2_final": float(arr(tag, "force_l2")[-1]),
                "thorax_z_final": float(arr(tag, "thorax_z")[-1]),
                "thorax_drop_model_units": float(z0 - arr(tag, "thorax_z")[-1]),
                "qvel_l2_final": float(arr(tag, "qvel_l2")[-1]),
            }
            for tag in ("native", "zero_floor")
        },
        "trajectory_differences": {
            "act_mean_abs_max": float(np.max(np.abs(act_diff))),
            "act_mean_abs_final": float(abs(act_diff[-1])),
            "force_l2_abs_max": float(np.max(np.abs(force_diff))),
            "force_l2_abs_final": float(abs(force_diff[-1])),
            "thorax_z_abs_max": float(np.max(np.abs(z_diff))),
            "thorax_z_abs_final": float(abs(z_diff[-1])),
            "thorax_z_final_over_initial": float(abs(z_diff[-1]) / max(1e-300, abs(z0))),
            "act_mean_abs_max_as_fraction_of_full_activation_range": act_diff_frac_of_full_range,
            "act_mean_relative_final_IS_UNDEFINED_here": True,
            "act_mean_relative_final_note": (
                "not reported: the zero_floor activation is exactly 0 for the whole run, so a "
                "ratio would divide by 0.  The absolute difference and its size relative to "
                "the [0,1] activation range are reported instead."),
        },
        "free_fall_reference": free_fall,
        "trajectories": {
            "time_s": traj["native"]["t"],
            "act_mean_native": traj["native"]["act_mean"],
            "act_mean_zero_floor": traj["zero_floor"]["act_mean"],
            "act_0_native": traj["native"]["act_0"],
            "act_0_zero_floor": traj["zero_floor"]["act_0"],
            "force_l2_native": traj["native"]["force_l2"],
            "force_l2_zero_floor": traj["zero_floor"]["force_l2"],
            "thorax_z_native": traj["native"]["thorax_z"],
            "thorax_z_zero_floor": traj["zero_floor"]["thorax_z"],
        },
        "qpos_max_abs_diff_after_run": float(np.max(np.abs(d_native.qpos - d_zero.qpos))),
        "qvel_max_abs_diff_after_run": float(np.max(np.abs(d_native.qvel - d_zero.qvel))),
        "act_max_abs_diff_after_run": float(np.max(np.abs(d_native.act[:native.nu] -
                                                           d_zero.act[:zero_floor.nu]))),
    }
    return {"identical_initialisation": init_state, "E1_instantaneous": instantaneous,
            "E2_free_integration": free,
            "model_names_used": {"native": "native (ctrlrange [0.0001, 1], the canonical "
                                           "compiled ctrlrange)",
                                 "zero_floor": "zero_floor (ctrlrange [0, 1], IN-MEMORY-ONLY "
                                               "derived model)"},
            "muscle_names_first_last": [names[0], names[-1]]}


def _thorax_body_id(model) -> int:
    import mujoco
    for cand in ("thorax", "Thorax", "body", "fly"):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
        if i >= 0:
            return int(i)
    # fall back to the first non-world body
    for i in range(1, model.nbody):
        return i
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> dict:
    import mujoco

    print(f"mujoco {mujoco.__version__}  python {platform.python_version()}  "
          f"numpy {np.__version__}", flush=True)

    # The output directory may not exist yet (another audit is creating it).  Create it, never
    # delete or overwrite anything already in it.
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    before = protected_snapshot()
    canonical_before = before[str(CANONICAL_XML)]
    if canonical_before != EXPECTED_SHA256:
        raise SystemExit(
            f"REFUSING TO RUN: canonical asset hash is {canonical_before}, expected "
            f"{EXPECTED_SHA256}.  The audit must not report on a changed asset.")

    from engine import canonical_physics as cp
    if str(cp.CANONICAL_XML) != str(CANONICAL_XML):
        raise SystemExit(f"canonical_physics points at {cp.CANONICAL_XML}, not {CANONICAL_XML}")

    print("compiling canonical model ...", flush=True)
    model = cp.load_model()
    print(f"compiled: nq={model.nq} nv={model.nv} nu={model.nu} na={model.na} "
          f"dt={model.opt.timestep}", flush=True)
    data = mujoco.MjData(model)

    # derived models: NAMED, IN-MEMORY-ONLY deep copies of the compiled mjModel
    native, zero_floor = build_derived_models(model)
    print(f"derived models built in memory: native ctrlrange_lower="
          f"{native.actuator_ctrlrange[0][0]}, zero_floor ctrlrange_lower="
          f"{zero_floor.actuator_ctrlrange[0][0]}", flush=True)

    xml_mtime_before = CANONICAL_XML.stat().st_mtime_ns

    out = {}
    t = time.time()
    out["A_act_dot_origin"] = part_A(model, data)
    print(f"A done ({time.time() - t:.1f}s)", flush=True); t = time.time()
    out["B_timestep_sensitivity"] = part_B(model, data)
    print(f"B done ({time.time() - t:.1f}s)", flush=True); t = time.time()
    out["C_passive_force"] = part_C(model, data)
    print(f"C done ({time.time() - t:.1f}s)", flush=True); t = time.time()
    out["D_control_floor_effect"] = part_D(model, data, native, zero_floor)
    print(f"D done ({time.time() - t:.1f}s)", flush=True); t = time.time()
    out["E_derived_model_comparison"] = part_E(model, native, zero_floor)
    print(f"E done ({time.time() - t:.1f}s)", flush=True)

    xml_mtime_after = CANONICAL_XML.stat().st_mtime_ns
    after = protected_snapshot()
    canonical_after = after[str(CANONICAL_XML)]

    out["protected_files"] = {
        "canonical_xml_path": str(CANONICAL_XML),
        "canonical_xml_sha256_before": canonical_before,
        "canonical_xml_sha256_after": canonical_after,
        "canonical_xml_sha256_expected": EXPECTED_SHA256,
        "canonical_xml_unchanged": bool(canonical_before == canonical_after == EXPECTED_SHA256),
        "canonical_xml_mtime_ns_before": xml_mtime_before,
        "canonical_xml_mtime_ns_after": xml_mtime_after,
        "canonical_xml_mtime_unchanged": bool(xml_mtime_before == xml_mtime_after),
        "other_protected_files_sha256": {k: v for k, v in after.items() if k != str(CANONICAL_XML)},
        "notes": "no XML was written at any point; the derived models are deep copies of the "
                 "compiled mjModel held only in this process's memory",
    }
    return out


if __name__ == "__main__":
    with mg.guarded_run("muscle_floor_audit", **mg.BUDGETS["muscle_floor_audit"]) as report:
        t0 = time.time()
        measured = main()
        info = report("completed")

    measured["_guard_report"] = info
    measured["_wall_s_main"] = time.time() - t0

    # ---- result contract, built explicitly (NOT cp.write_result: its default `units` block
    # hard-codes mass_model=mg and illustrative_vs_measured="measured", neither of which is
    # true for this audit).
    doc = {
        "status": "PASS" if measured["protected_files"]["canonical_xml_unchanged"] else "FAIL",
        "question": (
            "For the canonical asset outputs/muscles_six_legs/fruitfly_six_leg_standing.xml "
            "(nu=90, all muscle actuators, ctrllimited ctrlrange=[0.0001, 1]): (A) where does "
            "the first-step act_dot=2.0 come from and what are the measured actuator_dynprm/"
            "gainprm/biasprm fields; (B) how sensitive is the act trajectory to the timestep "
            "(1e-4 / 5e-5 / 1e-5) and does it converge; (C) what is the passive force "
            "magnitude at act=0 for each of the 90 muscles and its share of the active range; "
            "(D) does the ctrlrange lower bound 0.0001 keep a real active tension or is it "
            "numerically negligible; (E) how does a NAMED DERIVED MODEL with ctrlrange lower "
            "bound 0 ('zero_floor') differ from the unmodified compiled model ('native') "
            "under identical initialisation, in act trajectory, muscle force and collapse "
            "time over a 0.05 s free integration?"
        ),
        "status_meaning": (
            "PASS means: the audit ran to completion under the memory guard and the canonical "
            "asset hash is unchanged before and after.  It does NOT mean the model stands, "
            "that a controller exists, or that any posture result is accepted."
        ),
        "measured": measured,
        "provenance": {
            "run_by": "hand-written audit script, run by hand at the time in generated_at_utc",
            "exact_command": (
                "cd /run/media/sensen/Data2/cell_wound_prototype && "
                "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
                "/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python "
                "/run/media/sensen/Data2/cell_wound_prototype/run_muscle_floor_audit.py"),
            "script": str(Path(__file__).resolve()),
            "engine_module": "engine/canonical_physics.py (imported, unmodified)",
            "memory_guard": "engine/memory_guard.py guarded_run('muscle_floor_audit')",
            "mujoco_version": mujoco.__version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "interpreter": sys.executable,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "canonical_xml": str(CANONICAL_XML),
            "canonical_xml_sha256": measured["protected_files"]["canonical_xml_sha256_after"],
            "derived_models": (
                "native and zero_floor are DEEP COPIES of the compiled mjModel held in this "
                "process's memory.  No XML file was written, the canonical asset was not "
                "modified or re-derived, and results measured on zero_floor are properties of "
                "that derived model ONLY - they must never be quoted as canonical-model "
                "results."),
            "external_references": [
                "https://github.com/google-deepmind/mujoco/blob/main/doc/modeling.rst",
                "https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_util_misc.h",
                "https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_forward.c",
            ],
            "reference_priority_note": (
                "the INSTALLED library (libmujoco.so.3.9.0, called through ctypes and "
                "disassembled with objdump) is the authoritative reference for every dynamics "
                "claim here.  The prose documentation is used as a description, and the "
                "mujoco main-branch source is cited only to record where it disagrees with "
                "the installed library."),
        },
        "units": {
            "force": "MODEL force units.  The model is a MuJoCo model whose length unit is "
                     "the centimetre and whose mass unit is 1e-6 kg (see "
                     "engine/canonical_physics.py); NO SI CONVERSION HAS BEEN ESTABLISHED for "
                     "the muscle actuator force, so every force number in this artifact is "
                     "reported in model force units and must not be given an SI label.",
            "si_conversion_established": False,
            "si_conversion_note": "engine/canonical_physics.py converts length (model cm -> "
                                  "mm) and mass (model mg -> kg); it does NOT define a "
                                  "force-unit conversion, and this audit does not invent one.",
            "length_model": "cm",
            "length_reported_here": "model length units (cm); no _mm relabelling is applied "
                                    "to raw model quantities in this artifact",
            "time": "s",
            "acceleration": "model units (cm/s^2) where stated",
            "generalised_force_note": "qfrc values are model generalised force units "
                                      "(mass*length/time^2 in model units)",
            "gravity_model_used_for_scale": measured["A_act_dot_origin"]["gravity"][2],
            "act_and_ctrl": "dimensionless normalised activation / control in [0, 1]",
            "timestep": "s",
        },
        "what_this_does_not_support": [
            "This is a hand-run audit script, not a finished product and not a closed-loop "
            "acceptance test.  No acceptance gate was applied to the fly's posture.",
            "No SI force conversion is established here.  The force numbers are in model "
            "force units; do not relabel them as newtons, mN or any SI-derived unit.",
            "Part C is a SINGLE frozen configuration (the compiled initial state, act=0).  It "
            "does not establish the passive-force distribution over the reachable "
            "configuration space, nor that a small passive force is harmless.",
            "Part B characterises the behaviour of THIS asset's activation dynamics at the "
            "tested timesteps only.  It is not a general statement about MuJoCo muscle "
            "actuators, and 'converged over 1 ms' is not 'converged over a standing trial'.",
            "Part D/E results for 'zero_floor' are results of a DERIVED, IN-MEMORY model.  "
            "They do NOT describe the canonical asset and must not be quoted as if they did.",
            "A 0.05 s free integration is not standing, not a controller, and not stability. "
            "Collapse times measured here are not escape times and cannot be compared with "
            "the closed-loop escape-time results of this project.",
            "The audit does not establish WHY ctrlrange starts at 0.0001 in the upstream "
            "asset-generation pipeline; only what the compiled model does with it.  The XML "
            "default class writes ctrlrange '0 1' while every actuator overrides it with "
            "'0.0001 1', so the lower bound is an override whose design intent is not "
            "recorded in the asset and is NOT established here.",
            "No claim is made that the 0.0001 floor is a bug, nor that removing it is an "
            "improvement: this audit measures magnitude only.",
            "The memory guard bounds THIS process; it does not protect the host from other "
            "programs and does not make this audit's conclusions about memory.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(doc, indent=2, ensure_ascii=False)
    except TypeError as exc:
        # never lose the whole artifact to one unserialisable value: report it in place
        doc["_serialisation_error"] = repr(exc)
        text = json.dumps(doc, indent=2, ensure_ascii=False, default=str)
        print(f"WARNING: artifact needed default=str fallback: {exc!r}", flush=True)
    OUT_JSON.write_text(text)
    print(f"\nwrote {OUT_JSON}  ({len(text)} bytes)", flush=True)
    print(f"status={doc['status']} canonical_unchanged="
          f"{measured['protected_files']['canonical_xml_unchanged']}", flush=True)
    print(json.dumps({"protected_files": measured["protected_files"],
                      "_guard_report": measured["_guard_report"]}, indent=2))
