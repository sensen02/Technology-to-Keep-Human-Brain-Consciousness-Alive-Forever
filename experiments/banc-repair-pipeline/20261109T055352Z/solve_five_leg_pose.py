"""Standing pose on FIVE legs, with a chosen leg raised and held clear.

Measured problem this addresses: commanding the vertical Jacobian raise for the right hind leg
lets the trunk sag by 0.65 because the other five legs keep their stand-pose targets. Here the
five supporting legs are re-solved so that their pads stay coplanar on the floor, the raised pad
stays clear, and the body height and COM stay where the acceptance criteria want them. The result
is then verified by an actual static hold, not by the solver's own report.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc

import os
RAISED = os.environ.get('RAISED', 'RH')
COM_TARGET = float(os.environ.get('COM_TARGET', '1.45'))
CLEAR = float(os.environ.get('CLEAR', '0.04'))
HOLD_S = float(os.environ.get('HOLD_S', '0.30'))


def build():
    core = rc.Core()          # loads the same derived world the trainer uses
    m = core.plant; d = core.d
    jids = m.actuator_trnid[:, 0]
    qa = m.jnt_qposadr[jids]
    lo = m.jnt_range[jids, 0] + .01; hi = m.jnt_range[jids, 1] - .01
    return core, m, d, qa, lo, hi


def pad_points(m, d, core):
    return {L: np.mean([d.geom_xpos[g] for g in core.pads[L]], axis=0) for L in rc.LEG_ORDER}


def pad_lowest(m, d, core, leg):
    zs = []
    for g in core.pads[leg]:
        zs.append(float(d.geom_xpos[g][2] - m.geom_size[g][0]))
    return min(zs)


def nonfoot_low(m, d, core):
    low = 1e9
    for g in range(m.ngeom):
        if core.planes[g]:
            continue
        if core.geom_leg[g] >= 0:
            continue
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            b = int(m.geom_bodyid[g])
            mi = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
            R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
            vw = (v @ R.T + m.geom_pos[g]) @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
            low = min(low, float(vw[:, 2].min()))
        else:
            low = min(low, float(d.geom_xpos[g][2] - np.linalg.norm(m.geom_size[g])))
    return low


def main():
    core, m, d, qa, lo, hi = build()
    jids = m.actuator_trnid[:, 0]
    key = m.key_qpos[0].copy()
    q_std = core.pose_q.copy()
    raised = rc.LEG_ORDER.index(RAISED)
    x0 = np.concatenate([q_std, [core.pose_z]])

    def measure(x):
        q, rz = x[:42], x[42]
        d.qpos[:] = key; d.qpos[qa] = q; d.qpos[2] = rz
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        return q, rz

    def residual(x):
        q, rz = measure(x)
        r = []
        for li, L in enumerate(rc.LEG_ORDER):
            z = pad_lowest(m, d, core, L)
            if li == raised:
                r.append(20. * max(0., CLEAR - z))      # lifted pad must stay clear
            else:
                r.append(20. * z)                      # supporting pads on the floor
        r.append(6. * (d.subtree_com[0][2] - COM_TARGET))
        r.append(8. * max(0., .02 - nonfoot_low(m, d, core)))
        r += list(0.05 * (q - q_std))                  # stay near the nominal pose
        return np.array(r)

    sol = least_squares(residual, x0, bounds=(np.concatenate([lo, [0.8]]),
                                              np.concatenate([hi, [2.2]])),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
    q, rz = measure(sol.x)
    seats = {L: pad_lowest(m, d, core, L) for L in rc.LEG_ORDER}
    out = dict(label='five leg stance with %s raised; solver report' % RAISED,
               raised=RAISED, com_target=COM_TARGET, clearance_target=CLEAR,
               q_rad=[float(v) for v in q], root_z=float(rz),
               com=float(d.subtree_com[0][2]),
               pad_lowest={L: float(v) for L, v in seats.items()},
               nonfoot_low=float(nonfoot_low(m, d, core)),
               cost=float(sol.cost), optimality=float(sol.optimality),
               max_joint_excursion=float(np.abs(q - q_std).max()))
    # verified statically: hold the solved joint targets and let the physics run
    core2 = rc.Core()
    core2.reset()
    core2.d.qpos[core2.a_qpos] = q
    core2.d.qpos[2] = rz
    mj.mj_forward(core2.plant, core2.d)
    core2.actuator.reset(core2.d)         # re-project the incremental targets onto this pose
    core2._padz_prev = np.asarray([float(core2.measure()['padz'][l]) for l in range(rc.NL)])
    core2.buf[:] = core2.actuator.target_rad[None]
    z0 = float(core2.d.qpos[2])
    com0 = float(np.mean([core2.measure()['com']]))
    hold = []
    n = int(HOLD_S / rc.PDT)
    for k in range(n):
        m2 = core2.tick_rate(np.zeros(rc.NJ))
        hold.append((float(m2['com']), float(m2['loads'].sum() / rc.WEIGHT), float(m2['nonfoot'])))
    out['static_hold'] = dict(seconds=HOLD_S, root_drop=float(core2.d.qpos[2] - z0),
                              com_start=com0, com_min=min(h[0] for h in hold),
                              support_min=min(h[1] for h in hold),
                              nonfoot_max=max(h[2] for h in hold))
    tag = '%s_%s' % (RAISED, str(COM_TARGET).replace('.', ''))
    (HERE / ('STAND_FIVE_LEG_%s.json' % tag)).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('raised', 'com', 'root_z', 'nonfoot_low',
                                          'max_joint_excursion', 'cost')}, indent=1))
    print('pad lowest:', {k: round(v, 4) for k, v in out['pad_lowest'].items()})
    print('static hold:', {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in out['static_hold'].items()})


if __name__ == '__main__':
    main()
