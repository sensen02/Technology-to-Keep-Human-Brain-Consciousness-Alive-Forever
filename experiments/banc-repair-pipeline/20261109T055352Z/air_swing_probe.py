"""Can each leg move its pad forward WHILE AIRBORNE, and if not, why?

The sign-based swing command (move every joint in the direction that raises foot x) does nothing
for the hind legs: their Coxa_roll sits on its joint limit almost every tick, so the joints that
would carry the foot forward cannot move. This instead solves, each tick, the one-step kinematic
problem in the horizontal plane with the joint limits projected out, and measures how far the pad
actually travels forward while it is off the ground.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

CLEAR = 0.02          # airborne threshold used for this probe (5x the placement height)


def joint_limits(core, leg):
    out = []
    for i in T.leg_joint_action_index(core, leg):
        j = core.plant.joint(int(core.actuator.joint_ids[i]))
        lo, hi = (float(j.range[0]), float(j.range[1])) if j.limited.any() else (-1e9, 1e9)
        out.append((i, lo, hi))
    return out


def swing_direction(core, leg, want=0.01, dz=0.0):
    """Two objective step: move the pad +x by `want` while changing height by `dz`.

    A horizontal-only step lowers the pad for the hind legs, which then touches down within a few
    ticks. Solving for x and z together keeps the pad airborne while it travels forward.
    """
    Jx = T.foot_horizontal_jacobian(core, leg)
    Jz = T.pad_vertical_jacobian(core, leg)
    joints = joint_limits(core, leg)
    dofs = np.array([int(core.actuator.dof_addresses[i]) for i, _, _ in joints])
    J = np.vstack([Jx[dofs], Jz[dofs]])
    rhs = np.array([want, dz])
    JJt = J @ J.T + 1e-8 * np.eye(2)
    try:
        dq = J.T @ np.linalg.solve(JJt, rhs)
    except np.linalg.LinAlgError:
        dq = np.zeros(len(dofs))
    a = np.zeros(rc.NJ)
    blocked = 0
    for (i, lo, hi), step in zip(joints, dq):
        q = float(core.d.qpos[int(core.actuator.qpos_addresses[i])])
        room = (hi - q) if step > 0 else (q - lo)
        allowed = max(0., min(abs(step), max(room, 0.))) * (1. if step > 0 else -1.)
        if abs(allowed) < abs(step) * 0.5:
            blocked += 1
        a[i] = np.clip(allowed / T.TRAVEL, -1., 1.)
    return a, blocked


def raise_leg(core, leg, max_ticks=220, clear=CLEAR):
    for k in range(max_ticks):
        if float(core.measure()['padz'][leg]) > clear:
            return True, k
        core.tick(-T.descend_direction(core, leg))
    return False, max_ticks


def probe(leg):
    core = rc.Core(); core.reset()
    ok, k_raise = raise_leg(core, leg)
    z_at = float(core.measure()['padz'][leg])
    x0 = float(core.measure()['foot'][leg][0])
    best = 0.; blocked_total = 0; ticks_air = 0
    for k in range(150):
        a, blocked = swing_direction(core, leg, want=0.02)
        blocked_total += blocked
        m = core.tick(a)
        padz = float(m['padz'][leg]); dx = float(m['foot'][leg][0]) - x0
        if padz > rc_thresh():
            ticks_air += 1
            best = max(best, dx)
    return dict(leg=rc.LEG_ORDER[leg], reached_clearance=ok, ticks_to_clear=k_raise,
                padz_at_clearance=z_at, airborne_advance=best, airborne_ticks=ticks_air,
                joint_limit_blocks=blocked_total,
                passes_advance=bool(best >= T.THRESH['advance']))


def rc_thresh():
    return T.THRESH['place_padz']


def main():
    out = dict(label='airborne forward reach per leg, joint limits projected out; no learning',
               clearance_target=CLEAR, per_leg={})
    for leg in range(rc.NL):
        r = probe(leg)
        out['per_leg'][r['leg']] = r
        print('%-2s clearance %-5s ticks %3d padz %+.4f  airborne advance %+.4f over %3d ticks, '
              'joint-limit blocks %d  %s' % (
                  r['leg'], r['reached_clearance'], r['ticks_to_clear'], r['padz_at_clearance'],
                  r['airborne_advance'], r['airborne_ticks'], r['joint_limit_blocks'],
                  'PASS' if r['passes_advance'] else 'FAIL'), flush=True)
    out['n_can_advance_airborne'] = sum(1 for r in out['per_leg'].values() if r['passes_advance'])
    (HERE / 'AIR_SWING_PROBE.json').write_text(json.dumps(out, indent=2) + '\n')
    print('legs that can advance airborne:', out['n_can_advance_airborne'], '/ 6')


if __name__ == '__main__':
    main()
