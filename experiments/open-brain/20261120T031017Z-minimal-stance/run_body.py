"""Standalone MuJoCo muscle-body bench. Input is a frozen initial pose."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / 'body/fruitfly_six_leg_standing.xml'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def names(model, object_type, count):
    return [mj.mj_id2name(model, object_type, index) for index in range(count)]


def moment_matrix(model, data):
    matrix = np.zeros((model.nu, model.nv))
    for arm in range(model.nu):
        start = data.moment_rowadr[arm]
        end = start + data.moment_rownnz[arm]
        matrix[arm, data.moment_colind[start:end]] = data.actuator_moment[start:end]
    return matrix


def load_pose(model, pose):
    data = mj.MjData(model)
    data.qpos[:] = pose
    mj.mj_forward(model, data)
    return data


def ground_loads(model, data, floor, foot_by_geom):
    loads = np.zeros(len(LEGS) + 1)
    local = np.zeros(6)
    for index, contact in enumerate(data.contact):
        if floor in contact.geom:
            other = contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
            mj.mj_contactForce(model, data, index, local)
            loads[foot_by_geom.get(other, len(LEGS))] += max(0., local[0])
    return loads


def rollout(model, pose, floor, foot_by_geom, thorax, weight):
    data = load_pose(model, pose)
    steps = round(.2 / model.opt.timestep)
    qpos = np.empty((steps + 1, model.nq))
    qvel = np.empty((steps + 1, model.nv))
    activation = np.empty((steps + 1, model.na))
    loads = np.empty((steps + 1, len(LEGS) + 1))
    torso = np.empty((steps + 1, 2))
    times = np.empty(steps + 1)
    for step in range(steps + 1):
        times[step] = data.time
        qpos[step], qvel[step], activation[step] = data.qpos, data.qvel, data.act
        loads[step] = ground_loads(model, data, floor, foot_by_geom) / weight
        torso[step] = data.subtree_com[0, 2], data.xmat[thorax, 8]
        if step < steps:
            mj.mj_step(model, data)
            mj.mj_forward(model, data)
    foot = loads[:, :-1].sum(axis=1)
    carrying = (loads[:, :-1] > .01).sum(axis=1)
    tail = times >= .1 - 1e-12
    nonfoot = np.flatnonzero(loads[:, -1] > 0.)
    warnings = data.warning.number.tolist()
    assert not np.any(data.ctrl) and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    arrays = dict(time=times, qpos=qpos, qvel=qvel, activation=activation,
                  ground_load_w=loads, torso=torso)
    assert all(np.isfinite(value).all() for value in arrays.values())
    gates = dict(three_loaded_legs=bool(np.all(carrying[tail] >= 3)),
                 foot_load_08_12=bool(np.all((foot[tail] >= .8) & (foot[tail] <= 1.2))),
                 zero_nonfoot=bool(np.all(loads[tail, -1] == 0.)),
                 com_retention=bool(np.all(torso[tail, 0] >= .9 * torso[0, 0])),
                 upright=bool(np.all(torso[tail, 1] >= .95)))
    result = dict(steps=steps, warnings=warnings, standing=all(gates.values()) and not any(warnings),
                  gates=gates, initial_leg_load_w=loads[0, :-1].tolist(),
                  first_nonfoot_s=float(times[nonfoot[0]]) if nonfoot.size else None,
                  tail_mean_foot_w=float(foot[tail].mean()), peak_nonfoot_w=float(loads[:, -1].max()),
                  min_com_mm=float(torso[:, 0].min()), initial_com_mm=float(torso[0, 0]),
                  last_com_mm=float(torso[-1, 0]), activation_peak=float(activation.max()))
    return result, arrays


def muscle_probe(model, pose, actuator_names, joint_names):
    data = load_pose(model, pose)
    hinge_ids = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)
    dofs = model.jnt_dofadr[hinge_ids]
    qids = model.jnt_qposadr[hinge_ids]
    tol = 1e-8
    moment = moment_matrix(model, data)
    length = data.actuator_length.copy()
    length_range = model.actuator_lengthrange.copy()
    passive = data.actuator_force.copy()
    diff = np.empty((model.nu, len(qids)))
    for column, qid in enumerate(qids):
        data.qpos[qid] += 1e-6
        mj.mj_fwdPosition(model, data)
        plus = data.actuator_length.copy()
        data.qpos[qid] -= 2e-6
        mj.mj_fwdPosition(model, data)
        diff[:, column] = (plus - data.actuator_length) / 2e-6
        data.qpos[qid] = pose[qid]
    data = load_pose(model, pose)
    response_force = np.empty(model.nu)
    activated = np.empty(model.nu)
    response_torque = np.empty((model.nu, model.nv))
    initial_torque = data.qfrc_actuator.copy()
    steps = round(.02 / model.opt.timestep)
    for actuator, act_id in enumerate(model.actuator_actadr):
        data.ctrl[:] = 0.
        data.act[:] = 0.
        data.ctrl[actuator] = 1.
        for _ in range(steps):
            mj.mj_fwdActuation(model, data)
            data.act[act_id] += model.opt.timestep * data.act_dot[act_id]
        mj.mj_fwdActuation(model, data)
        activated[actuator] = data.act[act_id]
        response_force[actuator] = data.actuator_force[actuator] - passive[actuator]
        response_torque[actuator] = data.qfrc_actuator - initial_torque
    np.testing.assert_array_equal(data.qpos, pose)
    np.testing.assert_array_equal(data.qvel, 0.)
    error = np.abs(response_torque - response_force[:, None] * moment).max()
    derivative_error = np.abs(diff - moment[:, dofs]).max()
    assert error < tol and derivative_error < 1e-6
    assert np.all(response_force <= tol)
    tendon_ids = model.actuator_trnid[:, 0]
    tendon_names = names(model, mj.mjtObj.mjOBJ_TENDON, model.ntendon)
    active = response_torque[:, dofs]
    compatible = (length >= length_range[:, 0] - tol) & (length <= length_range[:, 1] + tol)
    lower_exceeded = pose[qids] < model.jnt_range[hinge_ids, 0] - tol
    upper_exceeded = pose[qids] > model.jnt_range[hinge_ids, 1] + tol
    return dict(
        moment_rank=int(np.linalg.matrix_rank(moment[:, dofs])),
        joint_range_violations=[joint_names[j] for j in hinge_ids[lower_exceeded | upper_exceeded]],
        activation_steps_per_muscle=steps, max_torque_reconstruction_error=float(error),
        max_length_derivative_error=float(derivative_error),
        zero_active_force_count=int(np.count_nonzero(np.abs(response_force) <= tol)),
        outside_lengthrange_count=int(np.count_nonzero(~compatible)),
        one_sided_active_joints=[joint_names[j] for j, torque in zip(hinge_ids, active.T)
                                 if not (torque.min() < -tol and torque.max() > tol)],
        muscles=[dict(actuator=name, tendon=tendon_names[tendon_ids[i]], length_mm=float(length[i]),
                      lengthrange_mm=length_range[i].tolist(), within_lengthrange=bool(compatible[i]),
                      passive_force=float(passive[i]), active_force_delta=float(response_force[i]),
                      activation_20ms=float(activated[i]), hinge_torque={joint_names[j]: float(value)
                          for j, value in zip(hinge_ids, active[i]) if abs(value) > tol})
                 for i, name in enumerate(actuator_names)])


def main(out):
    model = mj.MjModel.from_xml_path(str(MODEL))
    joint_names = names(model, mj.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = names(model, mj.mjtObj.mjOBJ_ACTUATOR, model.nu)
    geoms = names(model, mj.mjtObj.mjOBJ_GEOM, model.ngeom)
    floor = int(np.flatnonzero(model.geom_type == mj.mjtGeom.mjGEOM_PLANE).item())
    thorax = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Thorax')
    assert thorax >= 0
    foot_by_geom = {i: LEGS.index(name[:2]) for i, name in enumerate(geoms)
                    if name and name[:2] in LEGS and 'Tarsus' in name}
    assert set(foot_by_geom.values()) == set(range(len(LEGS)))
    assert model.nu == model.na == model.ntendon == 90 and model.nmesh == 71 and model.neq == 0
    assert np.all(model.actuator_dyntype == mj.mjtDyn.mjDYN_MUSCLE)
    assert np.all(model.actuator_gaintype == mj.mjtGain.mjGAIN_MUSCLE)
    assert np.all(model.actuator_biastype == mj.mjtBias.mjBIAS_MUSCLE)
    assert np.all(model.actuator_trntype == mj.mjtTrn.mjTRN_TENDON)
    assert set(model.actuator_trnid[:, 0]) == set(range(model.ntendon))
    assert model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE and joint_names[0]
    with np.load(HERE / 'body/archived_pose.npz', allow_pickle=False) as saved:
        poses = dict(keyframe=model.key_qpos[0].copy(), archived_six_feet=saved['qpos'])
    assert all(q.shape == (model.nq,) for q in poses.values())
    weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
    floor_controls = model.actuator_ctrllimited.astype(bool) & (model.actuator_ctrlrange[:, 0] > 0.)
    result = dict(mujoco=mj.__version__, nq=model.nq, nv=model.nv, nbody=model.nbody,
                  nu=model.nu, ntendon=model.ntendon, nmesh=model.nmesh, root_joint=joint_names[0],
                  n_hinges=int(np.count_nonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)),
                  timestep_s=model.opt.timestep, gravity=model.opt.gravity.tolist(), weight_native=weight,
                  floor_geom=geoms[floor], native_ctrl_floors={a: float(v) for a, v, present in
                    zip(actuator_names, model.actuator_ctrlrange[:, 0], floor_controls) if present},
                  actuator_names=actuator_names, trials={}, muscle_probes={})
    for mode in ('native', 'zero_floor_control'):
        if mode == 'zero_floor_control':
            model.actuator_ctrlrange[floor_controls, 0] = 0.
        for label, pose in poses.items():
            tag = f'{mode}_{label}'
            result['trials'][tag], arrays = rollout(model, pose, floor, foot_by_geom, thorax, weight)
            np.savez_compressed(out / f'{tag}.npz', **arrays)
            if mode == 'zero_floor_control':
                result['muscle_probes'][label] = muscle_probe(model, pose, actuator_names, joint_names)
    (out / 'BODY.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main(args.output)
