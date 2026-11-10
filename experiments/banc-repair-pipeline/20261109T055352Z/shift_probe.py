"""What actually unloads a rear leg? The three rear legs fail the step task for different
reasons (LH cannot lift, RM places without lifting, RH lifts and places but never advances), and
an earlier round established that lifting a rear leg tips the body back into the ground. This
probes candidate "weight shift" commands and measures whether they unload a chosen rear leg
WITHOUT losing support, body height or non-foot clearance.

No learning, no acceptance change: this is a capability probe to design the pre-phase of a
demonstration, and every candidate is reported whether it works or not.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

REAR = ['LH', 'RH', 'RM']


def leg_joint_actions(core, leg):
    return T.leg_joint_action_index(core, leg)


def foot_horizontal_jacobian(core, leg):
    """Mean d(foot x position)/d(q) over the leg's pad spheres."""
    rows = []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        point = np.asarray(core.d.geom_xpos[g])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[0].copy())
    return np.mean(rows, axis=0)


def push_direction(core, leg, scale=1.0, backwards=True):
    """Joint rates that move this leg's foot backwards (+x is forward), i.e. push the body
    forward. Computed at the current pose, same style as descend_direction."""
    J = foot_horizontal_jacobian(core, leg)
    a = np.zeros(rc.NJ)
    for i in leg_joint_actions(core, leg):
        j = float(J[int(core.actuator.dof_addresses[i])])
        if abs(j) > 1e-9:
            a[i] = (-np.sign(j) if backwards else np.sign(j)) * scale
    return a


def run_shift(ticks, legs_to_push, scale, target_leg):
    """Command a coordinated push with `legs_to_push` for `ticks`, then measure."""
    core = rc.Core()
    core.reset()
    start_com = float(core.measure()['com'])
    start_foot = float(core.measure()['foot'][target_leg][0])
    sup_min = 1e9; com_min = 1e9; nf_max = 0.
    for k in range(ticks):
        a = np.zeros(rc.NJ)
        for leg in legs_to_push:
            a = np.maximum(a * 0, a) + push_direction(core, leg, scale)
        m = core.tick(np.clip(a, -1., 1.))
        sup_min = min(sup_min, float(m['loads'].sum() / rc.WEIGHT))
        com_min = min(com_min, float(m['com']))
        nf_max = max(nf_max, float(m['nonfoot']))
    m = core.measure()
    return dict(ticks=ticks, push=list(legs_to_push), scale=scale,
                rear_load={L: float(m['loads'][rc.LEG_ORDER.index(L)] / rc.WEIGHT) for L in REAR},
                front_load={L: float(m['loads'][rc.LEG_ORDER.index(L)] / rc.WEIGHT)
                            for L in ('LF', 'LM', 'RF')},
                target_load=float(m['loads'][target_leg] / rc.WEIGHT),
                target_foot_shift=float(m['foot'][target_leg][0]) - start_foot,
                com_end=float(m['com']), com_shift=float(m['com']) - start_com,
                support_min=sup_min, com_min=com_min, nonfoot_max=nf_max,
                safe=bool(sup_min >= .90 and nf_max <= .20 * rc.WEIGHT and com_min >= 1.00))


def main():
    all_legs = list(range(rc.NL))
    front_mid = [rc.LEG_ORDER.index(L) for L in ('LF', 'LM', 'RF')]
    same_side = {L: [rc.LEG_ORDER.index(x) for x in ({'LH': ['LF', 'LM'], 'RH': ['RF', 'RM'],
                                                      'RM': ['RF', 'RH']}[L])] for L in REAR}
    out = dict(label='weight-shift capability probe; no learning, acceptance unchanged',
               baseline={}, candidates=[])
    core = rc.Core(); core.reset()
    m0 = core.measure()
    out['baseline'] = dict(rear_load={L: float(m0['loads'][rc.LEG_ORDER.index(L)] / rc.WEIGHT)
                                      for L in REAR},
                           front_load={L: float(m0['loads'][rc.LEG_ORDER.index(L)] / rc.WEIGHT)
                                       for L in ('LF', 'LM', 'RF')},
                           com=float(m0['com']))
    print('baseline rear loads', {k: round(v, 3) for k, v in out['baseline']['rear_load'].items()},
          'com', round(out['baseline']['com'], 3), flush=True)
    for L in REAR:
        tl = rc.LEG_ORDER.index(L)
        for push, tag in ((front_mid, 'front_mid'), (same_side[L], 'same_side'),
                          (all_legs, 'all_legs')):
            for scale in (.25, .5, .8):
                for ticks in (60, 120):
                    r = run_shift(ticks, push, scale, tl)
                    r.update(target=L, push_tag=tag)
                    out['candidates'].append(r)
                    print('%-2s push %-10s s%.2f t%3d -> rear load %s  com %.3f sup %.3f nf %.1f %s' % (
                        L, tag, scale, ticks,
                        {k: round(v, 3) for k, v in r['rear_load'].items()},
                        r['com_end'], r['support_min'], r['nonfoot_max'],
                        'SAFE' if r['safe'] else ''), flush=True)
    best = {}
    for L in REAR:
        rows = [r for r in out['candidates'] if r['target'] == L and r['safe']]
        best[L] = min(rows, key=lambda r: r['target_load']) if rows else None
    out['best_safe_per_target'] = {
        L: (None if v is None else dict(push=v['push_tag'], scale=v['scale'],
                                        ticks=v['ticks'], target_load=v['target_load']))
        for L, v in best.items()}
    (HERE / 'SHIFT_PROBE.json').write_text(json.dumps(out, indent=2) + '\n')
    print('best safe:', json.dumps(out['best_safe_per_target']))


if __name__ == '__main__':
    main()
