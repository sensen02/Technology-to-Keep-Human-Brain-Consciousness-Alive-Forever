"""Where does the lift of a leg actually break? Kinematics first, then a simulation of the
best kinematically achievable direction.

For each leg this computes the vertical Jacobian of every one of its three pads with respect
to that leg's seven joints, converts it into the maximum pad rise reachable inside the
remaining joint travel and the actuator's own rate limit (2 rad/s over the 300 ms task
window), and then commands exactly that direction to measure what the body really does.
The gap between the kinematic bound and the measured rise separates "the pose cannot lift
this leg" from "the pose can, but something else stops it".
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
RATE_TRAVEL = rc.MAX_RATE * rc.PDT * TICKS      # rad of joint travel available in the window


def leg_joint_indices(core, leg):
    L = rc.LEG_ORDER[leg]
    return [i for i, j in enumerate(core.actuator.joint_ids)
            if core.plant.joint(int(j)).name.startswith('joint_' + L)]


def pad_vertical_jacobian(core, leg):
    """d(pad bottom height)/d(q) for the leg's joints, one row per pad."""
    rows, names = [], []
    for g in core.pads[rc.LEG_ORDER[leg]]:
        r = float(core.plant.geom_size[g][0])
        point = np.asarray(core.d.geom_xpos[g]) - np.array([0., 0., r])
        jacp = np.zeros((3, core.plant.nv))
        mj.mj_jac(core.plant, core.d, jacp, None, point, int(core.plant.geom_bodyid[g]))
        rows.append(jacp[2].copy())
        names.append(core.plant.geom(g).name)
    return np.array(rows), names


def analyse(leg):
    core = rc.Core()
    jidx = leg_joint_indices(core, leg)
    dofs = [int(core.actuator.dof_addresses[i]) for i in jidx]
    jids = [int(core.actuator.joint_ids[i]) for i in jidx]
    q = core.d.qpos[core.actuator.qpos_addresses[jidx]]
    rng = core.plant.jnt_range[jids]
    up = np.minimum(rng[:, 1] - q, RATE_TRAVEL)
    dn = np.minimum(q - rng[:, 0], RATE_TRAVEL)
    J, gnames = pad_vertical_jacobian(core, leg)
    Jl = J[:, dofs]                                   # (3 pads, 7 joints)
    # optimistic bound: each joint moves as far as it can in whichever direction raises the pad
    best = np.where(Jl > 0, Jl * up[None, :], -Jl * dn[None, :])
    bound_per_pad = best.sum(axis=1)
    # the leg is only as liftable as its lowest pad, and all three must clear together
    kinematic_bound = float(bound_per_pad.min())
    direction = np.zeros(rc.NJ)
    for k, i in enumerate(jidx):
        j = float(Jl[:, k].sum())
        if abs(j) < 1e-9:
            continue
        travel = up[k] if j > 0 else dn[k]
        direction[i] = np.sign(j) * min(1., travel / RATE_TRAVEL)
    # simulate that direction
    core.reset()
    rest = float(core.measure()['padz'][leg])
    maxz = rest; minload = 1e9; sup_min = 1e9; com_min = 1e9; nonfoot = 0.
    for _ in range(TICKS):
        m = core.tick(direction)
        maxz = max(maxz, float(m['padz'][leg]))
        minload = min(minload, float(m['loads'][leg]))
        sup_min = min(sup_min, float(m['loads'].sum() / W))
        com_min = min(com_min, m['com'])
        nonfoot = max(nonfoot, float(m['nonfoot']))
    # did the commanded joints actually move?
    q_after = core.d.qpos[core.actuator.qpos_addresses[jidx]]
    moved = q_after - q
    names = [core.plant.joint(j).name.replace('joint_' + rc.LEG_ORDER[leg], '') for j in jids]
    return dict(
        joints=names,
        q_start=q.tolist(), travel_up_rad=up.tolist(), travel_down_rad=dn.tolist(),
        pad_names=gnames,
        vertical_jacobian_per_pad=np.round(Jl, 5).tolist(),
        kinematic_bound_min_pad=kinematic_bound,
        direction_units=np.round(direction[jidx], 4).tolist(),
        measured_max_padz=float(maxz), measured_rise=float(maxz - rest), rest_padz=rest,
        joint_displacement_rad=np.round(moved, 4).tolist(),
        min_load_over_body_weight=float(minload / W),
        support_min=float(sup_min), com_min=float(com_min), nonfoot_max=float(nonfoot),
        reaches_task_lift=bool(maxz >= 0.015))


def main():
    out = {'label': 'kinematic lift bound and its simulation; diagnostic, no learning',
           'window_ticks': TICKS, 'rate_travel_rad': RATE_TRAVEL,
           'task_lift_requirement': 0.015, 'per_leg': {}}
    for leg in range(rc.NL):
        r = analyse(leg)
        out['per_leg'][rc.LEG_ORDER[leg]] = r
        print(json.dumps({rc.LEG_ORDER[leg]: {
            'bound': round(r['kinematic_bound_min_pad'], 4),
            'measured_rise': round(r['measured_rise'], 4),
            'reaches': r['reaches_task_lift'],
            'joint_move_max': round(float(np.abs(r['joint_displacement_rad']).max()), 4)}}),
            flush=True)
    bl = out['per_leg']
    out['summary'] = {
        'legs_whose_pose_cannot_lift': [L for L, v in bl.items()
                                        if v['kinematic_bound_min_pad'] < 0.015],
        'legs_whose_pose_can_lift_but_body_does_not': [
            L for L, v in bl.items()
            if v['kinematic_bound_min_pad'] >= 0.015 and not v['reaches_task_lift']],
        'legs_reaching': [L for L, v in bl.items() if v['reaches_task_lift']],
        'interpretation': 'kinematic_bound_min_pad is the largest pad rise available inside the '
                          'remaining joint travel and the 2 rad/s rate limit over the task '
                          'window, for the direction that raises the lowest pad of that leg. '
                          'If it is below 0.015 the pose itself rules the lift out.'}
    (HERE / 'LIFT_KINEMATICS.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['summary'], indent=2))


if __name__ == '__main__':
    main()
