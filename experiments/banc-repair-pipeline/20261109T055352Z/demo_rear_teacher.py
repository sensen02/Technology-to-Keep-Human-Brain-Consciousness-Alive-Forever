"""A teacher whose swing starts from a MEASURED clearance, scored with the acceptance definition.

Findings that shape it: the hind legs have plenty of kinematic forward room (60 ticks of full
authority could move the pad +1.8), but their airborne window is short because the trunk sags when
they are raised (root drops 0.47 for LH and 0.65 for RH). So the teacher raises until the pad is
measured clearly airborne, then swings forward while holding height, then lowers. The trigger is a
measured pad height, not a clock, so the demonstration is state triggered.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T


def two_objective(core, leg, want=0.02, dz=0.01):
    """Joint step that moves the pad +x by `want` while raising it by `dz`, limits projected."""
    Jx = T.foot_horizontal_jacobian(core, leg)
    Jz = T.pad_vertical_jacobian(core, leg)
    idx = T.leg_joint_action_index(core, leg)
    dofs = np.array([int(core.actuator.dof_addresses[i]) for i in idx])
    J = np.vstack([Jx[dofs], Jz[dofs]])
    try:
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(2), np.array([want, dz]))
    except np.linalg.LinAlgError:
        return np.zeros(rc.NJ)
    a = np.zeros(rc.NJ)
    for i, step in zip(idx, dq):
        j = core.plant.joint(int(core.actuator.joint_ids[i]))
        lo, hi = (float(j.range[0]), float(j.range[1])) if j.limited.any() else (-1e9, 1e9)
        q = float(core.d.qpos[int(core.actuator.qpos_addresses[i])])
        room = (hi - q) if step > 0 else (q - lo)
        allowed = max(0., min(abs(step), max(room, 0.))) * (1. if step > 0 else -1.)
        a[i] = np.clip(allowed / T.TRAVEL, -1., 1.)
    return a


def run(leg, clear=.06, swing_ticks=90, ticks=None, trace=False):
    ticks = ticks or T.T_EP
    core = rc.Core(); core.reset()
    rest = float(core.measure()['padz'][leg])
    x0 = float(core.measure()['foot'][leg][0])
    maxz = rest; minload = 1e9; sup = 1e9; com = 1e9; nf = 0.
    air_adv = 0.; whole_adv = 0.; air = 0; lifted = False; swing_left = 0; last = None
    for k in range(ticks):
        padz = float(core.measure()['padz'][leg])
        if not lifted:
            a = -T.descend_direction(core, leg)
            if padz >= clear:
                lifted = True; swing_left = swing_ticks
        elif swing_left > 0:
            a = two_objective(core, leg, want=0.03, dz=0.005)
            swing_left -= 1
        else:
            a = T.descend_direction(core, leg)
        m = core.tick(a)
        padz = float(m['padz'][leg]); dx = float(m['foot'][leg][0]) - x0
        maxz = max(maxz, padz); minload = min(minload, float(m['loads'][leg]))
        sup = min(sup, float(m['loads'].sum() / rc.WEIGHT))
        com = min(com, float(m['com'])); nf = max(nf, float(m['nonfoot']))
        whole_adv = max(whole_adv, dx)
        if padz > T.THRESH['place_padz']:
            air += 1; air_adv = max(air_adv, dx)
        last = m
    fl = float(last['loads'][leg] / rc.WEIGHT); fp = float(last['padz'][leg])
    crit = dict(unload=bool(minload / rc.WEIGHT <= T.THRESH['unload']),
                lift=bool(maxz - rest >= T.THRESH['lift']),
                advance=bool(air_adv >= T.THRESH['advance']),
                place=bool(fl >= T.THRESH['place_load'] and fp <= T.THRESH['place_padz']),
                safe=bool(sup >= T.THRESH['support'] and nf <= T.THRESH['nonfoot']
                          and com >= T.THRESH['com']))
    return dict(rise=maxz - rest, air_advance=air_adv, whole_advance=whole_adv,
                airborne_ticks=air, final_load=fl, final_padz=fp,
                support_min=sup, com_min=com, nonfoot_max=nf,
                criteria=crit, passed=all(crit.values()))


def main():
    grid = [(c, s) for c in (.04, .08, .15) for s in (60, 90, 120)]
    out = dict(label='state triggered teacher (raise to measured clearance, swing holding height)',
               metric='acceptance definition: advance counted only while the pad is above the placement height',
               per_config={})
    best = None
    for clear, swing in grid:
        res = {rc.LEG_ORDER[l]: run(l, clear, swing) for l in range(rc.NL)}
        n = sum(r['passed'] for r in res.values())
        key = 'clear=%.2f,swing=%d' % (clear, swing)
        out['per_config'][key] = dict(n_passed=n,
                                      passed_legs=[L for L, r in res.items() if r['passed']],
                                      per_leg=res)
        print('%-18s -> %d/6  %s' % (key, n, {L: '%s a=%.3f' % (
            ''.join(k[0].upper() for k, v in r['criteria'].items() if v), r['air_advance'])
            for L, r in res.items()}), flush=True)
        if best is None or n > best[1]:
            best = (key, n, clear, swing)
    out['best'] = best[0]; out['best_n'] = best[1]
    out['best_config'] = dict(clear=best[2], swing_ticks=best[3])
    (HERE / 'DEMO_REAR_TEACHER.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best:', out['best'], out['best_n'])


if __name__ == '__main__':
    main()
