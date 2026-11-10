"""Find a lift direction for each leg that raises its pads without losing the body.

lift_kinematics.py shows the pads can be moved far up on all six legs, but the direction that
maximises pad rise on LH and RH also collapses the body (support 0.07 and 0.00). This script
searches a small declared family of directions per leg and keeps the ones that satisfy the
task's safety side: pad rise >= 0.02 while support stays >= 0.90, COM >= 1.10 and no non-foot
contact. Those directions are then stored as a teacher reference; they are not a controller
and not a policy result.
"""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc

W = rc.WEIGHT
TICKS = 300
RATE_TRAVEL = rc.MAX_RATE * rc.PDT * TICKS
DISTAL = ('Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')


def leg_joints(core, leg):
    L = rc.LEG_ORDER[leg]
    out = []
    for i, j in enumerate(core.actuator.joint_ids):
        n = core.plant.joint(int(j)).name
        if n.startswith('joint_' + L):
            out.append((i, n.replace('joint_' + L, '')))
    return out


def jac_rows(core, leg):
    rows = []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        r = float(core.plant.geom_size[g][0])
        point = np.asarray(core.d.geom_xpos[g]) - np.array([0., 0., r])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[2].copy())
    return np.mean(rows, axis=0)


def best_direction(core, leg, joints, scale=1.0):
    """Unit-clipped direction that maximises pad rise, restricted to the chosen joints."""
    rows = jac_rows(core, leg)
    qi = core.actuator.qpos_addresses
    rng = np.asarray(core.plant.jnt_range[[int(core.actuator.joint_ids[i]) for i, _ in joints]], float)
    q = core.d.qpos[qi[[i for i, _ in joints]]]
    up = np.minimum(rng[:, 1] - q, RATE_TRAVEL)
    dn = np.minimum(q - rng[:, 0], RATE_TRAVEL)
    a = np.zeros(rc.NJ)
    for k, (i, name) in enumerate(joints):
        j = float(rows[int(core.actuator.dof_addresses[i])])
        if abs(j) < 1e-9:
            continue
        travel = up[k] if j > 0 else dn[k]
        a[i] = np.sign(j) * min(1., travel / RATE_TRAVEL) * scale
    return a


def run(leg, direction):
    core = rc.Core()
    core.reset()
    rest = float(core.measure()['padz'][leg])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nonfoot = 0.
    traj = []
    for _ in range(TICKS):
        m = core.tick(direction)
        maxz = max(maxz, float(m['padz'][leg]))
        minload = min(minload, float(m['loads'][leg]))
        sup_min = min(sup_min, float(m['loads'].sum() / W))
        com_min = min(com_min, m['com'])
        nonfoot = max(nonfoot, float(m['nonfoot']))
        traj.append([float(m['padz'][leg]), float(m['loads'][leg] / W)])
    return dict(rest_padz=rest, max_padz=float(maxz), rise=float(maxz - rest),
                min_load_over_body_weight=float(minload / W), support_min=float(sup_min),
                com_min=float(com_min), nonfoot_max=float(nonfoot), traj=traj)


def main():
    probe = rc.Core()
    out = {'label': 'declared lift-direction family per leg, safety-screened; teacher reference only',
           'window_ticks': TICKS, 'acceptance': dict(rise=0.02, support_min=0.90, com_min=1.10,
                                                     nonfoot_max=0.20 * W),
           'per_leg': {}}
    chosen = {}
    for leg in range(rc.NL):
        joints_all = leg_joints(probe, leg)
        joints_distal = [(i, n) for i, n in joints_all if n in DISTAL]
        cands = {
            'all7': best_direction(probe, leg, joints_all, 1.0),
            'all7_half': best_direction(probe, leg, joints_all, .5),
            'distal': best_direction(probe, leg, joints_distal, 1.0),
            'distal_half': best_direction(probe, leg, joints_distal, .5),
            'distal_quarter': best_direction(probe, leg, joints_distal, .25),
        }
        rows = {}
        idx = [i for i, _ in joints_all]
        for name, d in cands.items():
            r = run(leg, d)
            r['unit_command'] = np.round(d[idx], 4).tolist()
            r['joints'] = [n for _, n in joints_all]
            r['accepted'] = bool(r['rise'] >= 0.02 and r['support_min'] >= 0.90
                                 and r['com_min'] >= 1.10 and r['nonfoot_max'] <= 0.20 * W)
            rows[name] = r
        out['per_leg'][rc.LEG_ORDER[leg]] = rows
        ok = [(n, r) for n, r in rows.items() if r['accepted']]
        best = max(ok, key=lambda kv: kv[1]['rise']) if ok else None
        chosen[rc.LEG_ORDER[leg]] = dict(candidate=best[0] if best else None,
                                         rise=best[1]['rise'] if best else None,
                                         support_min=best[1]['support_min'] if best else None,
                                         direction=best[1]['unit_command'] if best else None)
        print(json.dumps({rc.LEG_ORDER[leg]: {n: (round(r['rise'], 3), round(r['support_min'], 3),
                                                   round(r['com_min'], 3), r['accepted'])
                                              for n, r in rows.items()}}, default=str), flush=True)
    out['chosen'] = chosen
    out['all_legs_have_safe_lift_direction'] = all(v['candidate'] is not None for v in chosen.values())
    (HERE / 'LIFT_DEMO.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'all_legs_have_safe_lift_direction': out['all_legs_have_safe_lift_direction'],
                      'chosen': {k: (v['candidate'], None if v['rise'] is None else round(v['rise'], 3))
                                 for k, v in chosen.items()}}, indent=2))


if __name__ == '__main__':
    main()
