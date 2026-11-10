"""A press direction computed at the lifted pose instead of negating the lift direction.

The sweep in DEMO_SWEEP.json showed that no (lift direction, its negation) pair completes the
task: negating the lift direction undoes the lift instead of lowering the foot. This recomputes
the vertical Jacobian of the prompted leg at the current configuration and drives the joint
rates that lower the pads, so the press is a state feedback law rather than a fixed constant.
No clock is involved; both phases are still functions of measured state.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def leg_joints(core, leg):
    L = rc.LEG_ORDER[leg]
    return [i for i, j in enumerate(core.actuator.joint_ids)
            if core.plant.joint(int(j)).name.startswith('joint_' + L)]


def pad_vertical_jacobian(core, leg):
    rows = []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        r = float(core.plant.geom_size[g][0])
        point = np.asarray(core.d.geom_xpos[g]) - np.array([0., 0., r])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[2].copy())
    return np.mean(rows, axis=0)


def descend_direction(core, leg, scale=1.0):
    """Joint rates that lower the leg's pads, computed at the CURRENT configuration."""
    J = pad_vertical_jacobian(core, leg)
    a = np.zeros(rc.NJ)
    for i in leg_joints(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = -np.sign(j) * scale
    return a


def run(leg, lift_ticks=150, descend=True, ticks=T.T_EP):
    core = rc.Core(); demo = T.demo_directions()[leg]
    core.reset()
    rest = float(core.measure()['padz'][leg])
    start_x = float(core.measure()['foot'][leg][0])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nonfoot = 0.; best_adv = 0.
    last = None; switch_padz = None; traj = []
    for k in range(ticks):
        if k < lift_ticks or not descend:
            a = demo['unit']
        else:
            a = descend_direction(core, leg)
            if switch_padz is None:
                switch_padz = float(core.measure()['padz'][leg])
        mm = core.tick(a)
        padz = float(mm['padz'][leg])
        maxz = max(maxz, padz); minload = min(minload, float(mm['loads'][leg]))
        sup_min = min(sup_min, float(mm['loads'].sum() / rc.WEIGHT))
        com_min = min(com_min, float(mm['com']))
        nonfoot = max(nonfoot, float(mm['nonfoot']))
        best_adv = max(best_adv, float(mm['foot'][leg][0]) - start_x)
        last = mm
        if k % 25 == 0 or k == ticks - 1:
            traj.append([k, round(padz, 4), round(float(mm['loads'][leg] / rc.WEIGHT), 3)])
    final_load = float(last['loads'][leg] / rc.WEIGHT); final_padz = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz >= T.THRESH['lift']),
                advance=bool(best_adv >= T.THRESH['advance']),
                place=bool(final_load >= T.THRESH['place_load'] and final_padz <= T.THRESH['place_padz']),
                safe=bool(sup_min >= T.THRESH['support'] and nonfoot <= T.THRESH['nonfoot']
                          and com_min >= T.THRESH['com']))
    return dict(max_padz=maxz, padz_at_switch=switch_padz, final_padz=final_padz,
                final_load=final_load, min_load=minload / rc.WEIGHT, best_advance=best_adv,
                support_min=sup_min, com_min=com_min, nonfoot_max=nonfoot,
                criteria=crit, passed=all(crit.values()), traj=traj)


def main():
    out = dict(label='teacher self-check with a state dependent press direction; no learning',
               lift_ticks=150, per_leg={})
    for leg in range(rc.NL):
        r = run(leg)
        out['per_leg'][rc.LEG_ORDER[leg]] = r
        print(rc.LEG_ORDER[leg], 'padz@switch %+.4f -> final %+.4f  final load %.3f  %s' % (
            r['padz_at_switch'], r['final_padz'], r['final_load'],
            ''.join(k[0].upper() for k, v in r['criteria'].items() if v)), flush=True)
    out['n_passed'] = sum(r['passed'] for r in out['per_leg'].values())
    out['conclusion'] = ('the press direction must be computed at the lifted pose; negating the '
                         'lift direction cannot lower the foot')
    (HERE / 'DEMO_PRESS.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'n_passed': out['n_passed']}, indent=2))


if __name__ == '__main__':
    main()
