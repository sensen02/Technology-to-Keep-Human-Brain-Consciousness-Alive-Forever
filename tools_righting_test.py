#!/usr/bin/env python3
"""CAN THE CLOSED LOOP RIGHT THE FLY?  Same inverted start, both arms, measured.

THE QUESTION, POSED SO THE ANSWER MEANS SOMETHING.  The fly is STARTED UPSIDE DOWN in both
arms, with everything else identical.  The perturbation is an INITIAL CONDITION rather than a
mid-run poke, because a poke's answer depends on the poke; an initial condition is one number
both arms share.

WHAT EACH ARM IS (from engine.embodied.loop's own docstring, not my summary):
  * ``cpg_baseline``    -- the body runs FlyGym's tripod CPG and NOTHING reads the body.  It
    cannot right itself even in principle: it has no observation of its own orientation.
  * ``neural_modulated`` -- the SAME CPG generates the gait, and the neural tier's descending
    activity (over the real BANC connectome, driven by leg contact force and joint angle)
    modulates only the intrinsic frequency and a left/right asymmetry.  It does NOT generate
    the gait.  Whether that modulation happens to produce a righting torque is an EMPIRICAL
    question, and that is the whole point of running this.

WHAT IS MEASURED, and what would count as "recovered":
  * the thorax body's own UP AXIS projected on world +Z: +1 upright, -1 inverted.  This is the
    single number that says whether the fly is on its feet.
  * legs in contact, and the horizontal displacement, because "upright and walking" is the
    real target and "upright but inert" is not a recovery.
  A recovery counts only if up_z goes from about -1 to about +1 and STAYS there.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_righting_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

INVERTED = (0.0, 1.0, 0.0, 0.0)   # 180 deg about x, as a unit quaternion (w, x, y, z)


def trace(arm: str, quat, duration_s: float, seed: int = 0):
    """Run one arm and return the uprightness trace.  Uses the VALIDATED scheduler."""
    import mujoco
    from engine.embodied.loop import LoopConfig, MultirateScheduler

    cfg = LoopConfig(arm=arm, seed=seed, duration_s=duration_s,
                     spawn_position_mm=(0.0, 0.0, 0.6), spawn_quat_wxyz=quat,
                     gl_backend=None)
    sch = MultirateScheduler(cfg)
    ep = sch.run(want_frames=False)
    truth = ep.truth
    t = np.asarray(truth["time_s"], dtype=float)
    up = np.asarray(truth["thorax_up_z"], dtype=float) if "thorax_up_z" in truth else None
    return sch, ep, t, up


def thorax_up_z(rots_wxyz):
    """World +Z component of the thorax body's own up axis, per frame.  +1 = upright.

    The body's local +z in world coordinates is the THIRD COLUMN of its rotation matrix, which
    for a quaternion (w, x, y, z) is [2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)].  Written out
    because getting this from the wrong column is exactly the kind of sign error that would make
    an inverted fly look upright.
    """
    q = np.asarray(rots_wxyz, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return 1.0 - 2.0 * (x * x + y * y)


def analyse(ep, arm, label):
    t = np.asarray(ep.truth["time_s"], dtype=float)
    rots = np.asarray(ep.truth["body_rotations_wxyz"], dtype=float)   # (n, nbody, 4)
    pos = np.asarray(ep.truth["body_positions_mm"], dtype=float)
    cf = np.asarray(ep.truth["contact_found_raw"], dtype=float)
    # WHICH BODY INDEX IS THE THORAX IS FOUND, NOT ASSUMED.  MEASURED: the scheduler's
    # body arrays EXCLUDE the world body, so index 0 is the fly's root (the thorax), not index 1
    # as it is in run_arena_record's npz where index 0 is the world.  My first version assumed
    # index 1 and the assertion below caught it rather than silently tracing a wing.
    th = np.asarray(ep.truth["thorax_mm"], dtype=float)
    d = np.abs(pos - th[:, None, :]).max(axis=(0, 2))
    i_th = int(np.argmin(d))
    if float(d[i_th]) > 1e-3:
        raise RuntimeError("no body matches thorax_mm; the trace would be of the wrong body")
    up = thorax_up_z(rots[:, i_th, :])
    legs = (cf > 0).sum(axis=1)
    horiz = np.linalg.norm(pos[:, i_th, :2] - pos[0, i_th, :2], axis=1)
    half = len(up) // 2
    rec = {
        "arm": arm, "start": label, "n": int(up.size), "duration_s": float(t[-1] - t[0]),
        "thorax_body_index": i_th,
        "up_z_initial": float(up[0]), "up_z_final": float(up[-1]),
        "up_z_min": float(up.min()), "up_z_max": float(up.max()),
        "up_z_mean_first_half": float(up[:half].mean()),
        "up_z_mean_second_half": float(up[half:].mean()),
        "legs_down_mean": float(legs.mean()),
        "legs_down_second_half": float(legs[half:].mean()),
        "horizontal_travel_mm": float(horiz[-1]),
        "upright_fraction": float((up > 0.7).mean()),
    }
    rec["RECOVERED"] = bool(rec["up_z_final"] > 0.7 and rec["upright_fraction"] > 0.5)
    return rec, up, legs, horiz


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--arms", default="cpg_baseline,neural_modulated")
    ap.add_argument("--starts", default="upright,INVERTED")
    ap.add_argument("--invert-at", type=float, default=None,
                    help="walk normally, then invert the fly at this time (s). This is the "
                         "protocol that avoids the spawn-jam catapult.")
    ap.add_argument("--invert-height-mm", type=float, default=1.6)
    ap.add_argument("--reflex", action="store_true",
                    help="enable the state-dependent posture reflex (whole-body dorsal contact "
                         "drives a coxa-pitch swing, and switches off once upright)")
    ap.add_argument("--reflex-deg", type=float, default=120.0)
    a = ap.parse_args()
    want_arms = [x for x in a.arms.split(",") if x]
    if a.invert_at is not None:
        # MID-RUN PROTOCOL: always spawn upright (so the loop calibrates in a VALID walking
        # state), then turn the fly over deliberately.
        starts = [("walk-then-inverted", (1.0, 0.0, 0.0, 0.0))]
    else:
        starts = [("upright", (1.0, 0.0, 0.0, 0.0)), ("INVERTED", INVERTED)]
        starts = [x for x in starts if x[0] in a.starts.split(",")]
    print("=" * 96)
    print(f"RIGHTING TEST: same initial condition, both arms, duration {a.duration} s")
    print("=" * 96)
    out = []
    traces = {}
    for arm in want_arms:
        for label, quat in starts:
            import time as _t
            t0 = _t.perf_counter()
            try:
                sch = None
                from engine.embodied.loop import LoopConfig, MultirateScheduler
                cfg = LoopConfig(arm=arm, seed=0, duration_s=a.duration,
                                 spawn_position_mm=(0.0, 0.0, 0.6), spawn_quat_wxyz=quat,
                                 invert_at_s=a.invert_at,
                                 invert_height_mm=a.invert_height_mm,
                                 posture_reflex=bool(a.reflex),
                                 posture_reflex_deg=float(a.reflex_deg),
                                 gl_backend=None)
                ep = MultirateScheduler(cfg).run(want_frames=False)
                rec, up, legs, horiz = analyse(ep, arm, label)
                _dors = np.asarray(ep.truth.get("whole_body_dorsal_index", []), dtype=float)
                _nwb = np.asarray(ep.truth.get("whole_body_contact_n", []), dtype=float)
                if _dors.size == up.size:
                    rec["dorsal_index_mean"] = float(_dors.mean())
                    rec["dorsal_index_at_end"] = float(_dors[-1])
                    rec["whole_body_contact_fraction"] = float((_nwb > 0).mean())
                    rec["leg_contact_fraction"] = float(
                        (np.asarray(ep.truth["contact_found_raw"], float) > 0).any(axis=1).mean())
                rec["wall_s"] = round(_t.perf_counter() - t0, 1)
                traces[(arm, label)] = {
                    "t": [round(float(x), 4) for x in np.asarray(ep.truth["time_s"])],
                    "up_z": [round(float(x), 5) for x in up],
                    "dorsal_index": [round(float(x), 5) for x in
                                     np.asarray(ep.truth.get("whole_body_dorsal_index", []),
                                                dtype=float)]}
                out.append(rec)
                print(f"\n{arm:<18} start={label:<9} ({rec['wall_s']}s wall)")
                print(f"   up_z: start {rec['up_z_initial']:+.3f}  min {rec['up_z_min']:+.3f}  "
                      f"max {rec['up_z_max']:+.3f}  final {rec['up_z_final']:+.3f}")
                print(f"   up_z mean: 1st half {rec['up_z_mean_first_half']:+.3f}   "
                      f"2nd half {rec['up_z_mean_second_half']:+.3f}   "
                      f"upright fraction {rec['upright_fraction']*100:.0f}%")
                print(f"   legs down: overall {rec['legs_down_mean']:.2f}   "
                      f"2nd half {rec['legs_down_second_half']:.2f}")
                print(f"   horizontal travel {rec['horizontal_travel_mm']:.2f} mm")
                print(f"   ==> {'RECOVERED' if rec['RECOVERED'] else 'did NOT recover'}")
            except Exception as exc:
                print(f"\n{arm:<18} start={label:<9}: FAILED {type(exc).__name__}: {exc}")
                out.append({"arm": arm, "start": label, "error": f"{type(exc).__name__}: {exc}"})
    # A VACUOUS-TEST GUARD, added because the first version of this test WAS vacuous: a silent
    # no-op patch meant the spawn orientation never reached the body, so the inverted runs came
    # out bit-identical to the upright ones and the fly "recovered" from an inversion it had
    # never been put into.  The two starting conditions must produce DIFFERENT traces, or the
    # whole comparison is meaningless and must be refused rather than reported.
    good = [r for r in out if "error" not in r]
    for arm in want_arms:
        pair = [r for r in good if r["arm"] == arm]
        if len(pair) == 2:
            a, b = pair
            same = (abs(a["up_z_final"] - b["up_z_final"]) < 1e-9
                    and abs(a["horizontal_travel_mm"] - b["horizontal_travel_mm"]) < 1e-9
                    and abs(a["up_z_initial"] - b["up_z_initial"]) < 1e-9)
            if same and a.invert_at is None:
                raise SystemExit(
                    f"VACUOUS TEST: the upright and inverted runs for {arm} are IDENTICAL, so "
                    "the spawn orientation did not take effect and nothing about righting has "
                    "been measured.  Refusing to report a verdict.")
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    for r in out:
        if "error" in r:
            continue
        print(f"  {r['arm']:<18} from {r['start']:<9} -> "
              f"{'RECOVERED' if r['RECOVERED'] else 'NOT recovered':<14} "
              f"(final up_z {r['up_z_final']:+.3f}, travel {r['horizontal_travel_mm']:.2f} mm)")
    dest = HERE / "outputs" / "righting_test.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"question": "can the closed loop right an inverted fly",
                                "inverted_quaternion_wxyz": list(INVERTED),
                                "duration_s": a.duration, "results": out,
                                "traces": {f"{k[0]}|{k[1]}": v for k, v in traces.items()}},
                               indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
