#!/usr/bin/env python3
"""DOES A STEADY MUSCLE PATTERN EXIST THAT BALANCES THE FLY AT ITS STANCE?

THE FLAW IN MY PREVIOUS PROBE, AND IT IS WHY EVERY EARLIER SEARCH FAILED.  I ranked muscles by the
VERTICAL acceleration they impart.  But the thorax carries a FREE JOINT, so holding a posture requires
balancing SIX quantities at once -- three force components and three torques -- and a pattern that
maximises lift alone can leave the body pitching or rolling until some other part hits the floor.
That is exactly what the measurements showed: at high joint stiffness the thorax rises to 1.48-1.70 mm
and the fly TIPS OVER (49 and 64 degrees, one to three legs down), while at low stiffness it sags.
The anti-gravity set optimised one of the six and ignored the other five.

So the question is posed properly here, as a linear balance problem measured at the stance:

    a_free(muscle)  =  the 6-component acceleration of the thorax's free joint  (3 linear, 3 angular)
    J[:, i]         =  a_free with muscle i at full activation, minus the passive value
    solve   min || a0 + J a ||   subject to   0 <= a <= 1

``a0`` is the passive acceleration (what gravity and the contacts do with no muscle active) and the
columns are MEASURED, one per muscle, on the model itself.  Nothing is hand-set: the pattern that
comes out is whatever balances the six components, and the residual says whether a steady pattern can
balance the fly AT ALL.

  small residual  -> a steady muscle pattern exists; tone is enough in principle, and the connectome's
                     job is to reach that pattern.
  large residual  -> no steady pattern can balance it, so balance must come from FEEDBACK, and that is
                     then a mathematical statement rather than an impression.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402

XML = Path(os.environ.get("STANDING_XML",
                          HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"))
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    v0, da0 = int(m.jnt_dofadr[jf]), int(m.jnt_dofadr[jf]) + 3
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print(f"one body weight = {mw:.4f}; free joint at dof {v0} (linear) / {da0} (angular)")
    for stiff in (float(os.environ.get("BAL_STIFF", "0.4")),):
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith("joint_"):
                m.jnt_stiffness[j] = stiff
    print(f"leg-joint stiffness in this solve: {m.jnt_stiffness[1]:.3f}")

    def accel():
        return np.array([d.qacc[v0], d.qacc[v0 + 1], d.qacc[v0 + 2],
                         d.qacc[da0], d.qacc[da0 + 1], d.qacc[da0 + 2]], float)

    SETTLE_S = float(os.environ.get("BAL_SETTLE", "0.03"))

    def reset():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.act[:] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        # SETTLE BRIEFLY FIRST.  MEASURED, and it matters: seating the tarsi leaves them about 0.003 mm
        # into the floor with 12-48 contacts, so the acceleration read at t = 0 is dominated by that
        # contact transient rather than by the postural imbalance.  Thirty milliseconds is long enough
        # for the transient to die and short enough (about 0.03 mm of thorax movement) that the fly is
        # still standing in its stance when the acceleration is read.
        for _ in range(int(SETTLE_S / m.opt.timestep)):
            mujoco.mj_step(m, d)

    reset()
    a0 = accel()
    z0 = float(d.xpos[th][2])
    print(f"\npassive acceleration at the stance (thorax z {z0:.4f} mm):")
    print(f"  linear  (x,y,z) = {np.round(a0[:3], 2)} mm/s^2   (free fall is -9801)")
    print(f"  angular (rx,ry,rz) = {np.round(a0[3:], 2)} rad/s^2")
    J = np.zeros((6, m.nu))
    for i in range(m.nu):
        reset()
        d.ctrl[i] = 1.0
        d.act[i] = 1.0
        mujoco.mj_forward(m, d)
        J[:, i] = accel() - a0
    # NORMALISE EACH ROW so the six components are commensurable: a component that gravity perturbs
    # by 1000 mm/s^2 should not be drowned out by one perturbed by 10.
    scale = np.maximum(np.abs(a0), np.abs(J).max(axis=1) * 0.5)
    scale = np.where(scale > 1e-9, scale, 1.0)
    Jn, a0n = J / scale[:, None], a0 / scale
    print(f"\nrow scales: {np.round(scale, 2)}")
    # is there any direction that can cancel a0?
    sol = lsq_linear(Jn, -a0n, bounds=(0.0, 1.0), max_iter=2000)
    resid = float(np.linalg.norm(Jn @ sol.x + a0n))
    passive_norm = float(np.linalg.norm(a0n))
    a_after = a0 + J @ sol.x
    print(f"\nnon-negative least squares over 90 muscles (each bounded to [0,1]):")
    print(f"  residual |a0 + J a| (normalised) = {resid:.6f}   passive |a0| = {passive_norm:.6f}")
    print(f"  fraction of the passive imbalance cancelled: "
          f"{100 * (1 - resid / passive_norm):.2f}%")
    print(f"  resulting acceleration: linear {np.round(a_after[:3], 2)} mm/s^2, "
          f"angular {np.round(a_after[3:], 2)} rad/s^2")
    n_on = int((sol.x > 0.02).sum())
    print(f"  muscles used above 0.02: {n_on} of {m.nu}; mean activation where used "
          f"{sol.x[sol.x > 0.02].mean() if n_on else 0:.3f}")
    top = np.argsort(-sol.x)[:10]
    print("  the pattern's largest entries: "
          + ", ".join(f"{names[i]}={sol.x[i]:.3f}" for i in top))
    # a fair baseline for comparison: how much can the BEST SINGLE muscle cancel?
    best1 = max(float(np.linalg.norm(a0n + Jn[:, i])) for i in range(m.nu))
    print(f"\n  for comparison, the best single muscle leaves residual "
          f"{min(np.linalg.norm(a0n + Jn[:, i]) for i in range(m.nu)):.6f}; "
          f"doing nothing leaves {passive_norm:.6f}")
    verdict = ("A STEADY PATTERN CAN BALANCE THE FLY" if resid < 0.1 * passive_norm else
               "NO STEADY PATTERN CAN BALANCE IT -- balance must come from feedback")
    print(f"\n=== VERDICT: {verdict} ===")
    (HERE / "outputs" / "balance_solve.json").write_text(json.dumps({
        "stiffness": float(m.jnt_stiffness[1]), "thorax_z": z0,
        "passive_accel": a0.tolist(), "row_scale": scale.tolist(),
        "residual": resid, "passive_norm": passive_norm,
        "cancelled_fraction": float(1 - resid / passive_norm),
        "accel_after": a_after.tolist(),
        "pattern": {names[i]: float(sol.x[i]) for i in range(m.nu) if sol.x[i] > 1e-6},
        "n_muscles_used": n_on, "verdict": verdict}, indent=2))
    print(f"wrote {HERE / 'outputs' / 'balance_solve.json'}")


if __name__ == "__main__":
    main()
