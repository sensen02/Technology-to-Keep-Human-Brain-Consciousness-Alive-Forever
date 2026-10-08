#!/usr/bin/env python3
"""THREE HYPOTHESES ABOUT THE LEGS, ANSWERED WITH MEASUREMENTS.

The user asked whether the problem is (1) the leg POSTURE, (2) the leg LENGTH not reaching the ground,
or (3) the joint FORCE/TORQUE simulation being inadequate.  All three are measurable on the plant, with
no solver and no activation pattern anywhere:

  (2) LENGTH/REACH.  For each leg: its segment lengths, its fully extended reach, the reach required by
      the stance, and whether any joint is sitting at a limit -- a leg at its limit cannot push.
  (1) POSTURE / LOAD SHARING.  The measured vertical stiffness of each leg at the stance: lower the
      thorax by a small amount with zero activation and read how much each leg's load changes.  A leg
      whose vertical stiffness is near zero is not a strut, whatever its pose looks like.
  (3) TORQUE.  For each joint, the moment arm a vertical ground force has about that joint's axis, the
      torque that implies for the leg's share of the weight, and what the joint spring can supply
      (stiffness times the range still available to it).  If the spring cannot supply the torque, the
      joint simply rotates, and the muscle is the only thing that could hold it -- which is where the
      measured muscle capacity comes in.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
JR = (("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"), ("Trochanter", "yaw"),
      ("Trochanter", "pitch"), ("Trochanter", "roll"), ("Tibia", "pitch"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    z_adr = int(m.jnt_qposadr[jf]) + 2
    out = {"mw": mw}
    print(f"one body weight = {mw:.4f} force units (1 unit = 1 uN)")

    def load_per_leg():
        share = {leg: 0.0 for leg in LEGS}
        pt = {}
        for c in range(d.ncon):
            cc = d.contact[c]
            if cc.geom1 != fid and cc.geom2 != fid:
                continue
            other = cc.geom2 if cc.geom1 == fid else cc.geom1
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
            f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
            for leg in LEGS:
                if b.startswith(leg):
                    share[leg] += abs(f[0])
                    pt.setdefault(leg, np.asarray(cc.pos).copy())
                    break
        return share, pt

    def seat(dz=0.0):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[z_adr] += dz
        d.act[:] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)

    # ---------- (2) LENGTH AND REACH
    print("\n=== (2) LEG LENGTH AND REACH ===")
    seat()
    segs = ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
    print(f"  {'leg':<5}{'coxa->tarsus5 chain (mm)':>26}{'reach needed by stance':>25}"
          f"{'fraction':>10}")
    reach = {}
    for leg in LEGS:
        pts = []
        for sname in segs:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}{sname}")
            if b >= 0:
                pts.append(np.asarray(d.xpos[b]))
        L = float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
        coxa = np.asarray(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Coxa")])
        tip = pts[-1]
        need = float(np.linalg.norm(tip - coxa))
        reach[leg] = {"chain_mm": L, "need_mm": need, "fraction": need / L}
        print(f"  {leg:<5}{L:>26.3f}{need:>25.3f}{need/L:>10.3f}")
    out["reach"] = reach
    # which joints are at a limit, at the stance?
    print(f"\n  joints at or near a LIMIT at the stance (within 5% of the range):")
    lim = []
    for leg in LEGS:
        for seg, js in JR:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{seg}_{js}")
            q = float(d.qpos[int(m.jnt_qposadr[j])])
            lo, hi = (float(x) for x in m.jnt_range[j])
            span = hi - lo
            pos = (q - lo) / span if span > 0 else 0.5
            if pos < 0.05 or pos > 0.95:
                lim.append({"joint": f"{leg}{seg}_{js}", "qpos": q, "range": [lo, hi],
                            "position_in_range": pos})
                print(f"    {leg}{seg}_{js:<24} q={q:+.4f} range [{lo:+.3f},{hi:+.3f}] "
                      f"-> at {100*pos:.0f}% of range")
    if not lim:
        print("    none")
    out["joints_at_limit"] = lim

    # ---------- (1) VERTICAL STIFFNESS PER LEG (is the leg a strut?)
    print("\n=== (1) VERTICAL STIFFNESS OF EACH LEG AT THE STANCE ===")
    print(f"  moving the thorax down by delta with ZERO activation, and reading the load change")
    print(f"  {'leg':<5}{'load at dz=0 (bw)':>20}{'d(load)/d(dz) (bw/mm)':>24}{'share %':>10}")
    share0, _pt = load_per_leg()
    tot0 = sum(share0.values())
    stiff = {}
    for leg in LEGS:
        seat(0.05); s_up, _ = load_per_leg()
        seat(-0.05); s_dn, _ = load_per_leg()
        k = (s_dn[leg] - s_up[leg]) / 0.10 / mw
        stiff[leg] = k
        print(f"  {leg:<5}{share0[leg]/mw:>20.4f}{k:>24.4f}"
              f"{100*share0[leg]/max(tot0,1e-9):>10.1f}")
    out["vertical_stiffness_bw_per_mm"] = stiff
    print(f"  total tarsal load at the seated pose: {tot0/mw:.4f} bw")
    kk = np.array(list(stiff.values()))
    print(f"  stiffness spread: {(kk.max()/max(kk.min(),1e-12)):.1f}x between the stiffest and the "
          f"most compliant leg")

    # ---------- (3) TORQUE: what the ground force demands, and what the spring can supply
    print("\n=== (3) JOINT TORQUE DEMAND vs SPRING CAPACITY ===")
    seat()
    share0, pt = load_per_leg()
    print(f"  {'joint':<26}{'moment arm':>12}{'torque (uN*mm)':>16}{'spring can give':>17}"
          f"{'verdict':>12}")
    short = 0
    rows = []
    for leg in LEGS:
        if leg not in pt:
            continue
        c = pt[leg]
        Fz = share0[leg]                      # vertical force at the foot, in force units
        for seg, js in JR:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{seg}_{js}")
            p = np.asarray(d.xanchor[jid], float)
            a = np.asarray(d.xaxis[jid], float)
            r = c - p
            arm = abs(float(np.dot(a, np.cross(r, np.array([0.0, 0.0, -1.0])))))
            tau = Fz * arm
            q = float(d.qpos[int(m.jnt_qposadr[jid])])
            lo, hi = (float(x) for x in m.jnt_range[jid])
            ks = float(m.jnt_stiffness[jid])
            avail = (hi - q) if (q - lo) < (hi - q) else (q - lo)
            cap = ks * avail
            ok = cap >= tau
            if not ok:
                short += 1
            rows.append({"joint": f"{leg}{seg}_{js}", "arm_mm": arm, "torque": tau,
                         "spring_capacity": cap, "spring_holds": bool(ok)})
            print(f"  {f'{leg}{seg}_{js}':<26}{arm:>12.4f}{tau:>16.4f}{cap:>17.4f}"
                  f"{'ok' if ok else 'SPRING TOO WEAK':>12}")
    out["torque_rows"] = rows
    print(f"\n  joints whose spring cannot supply the torque demanded by the loaded foot: "
          f"{short} of {len(rows)}")
    print(f"  leg-joint stiffness in this model: {m.jnt_stiffness[1]:.3f} uN*mm/rad")
    (HERE / "outputs" / "leg_mechanics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'leg_mechanics.json'}")


if __name__ == "__main__":
    main()
