"""Native contact and joint observations, without a sensory current generator."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / 'baseline'
if not BASE.is_dir():
    BASE = HERE.parents[2] / 'experiments/open-brain/20261120T031017Z-minimal-stance'
sys.path.insert(0, str(BASE))
from run_body import LEGS, MODEL, ground_loads, load_pose, names


class BodyObservations:
    def __init__(self, model):
        self.model = model
        self.floor = int(np.flatnonzero(model.geom_type == mj.mjtGeom.mjGEOM_PLANE).item())
        self.thorax = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Thorax')
        geom_names = names(model, mj.mjtObj.mjOBJ_GEOM, model.ngeom)
        self.foot_by_geom = {i: LEGS.index(name[:2]) for i, name in enumerate(geom_names)
                             if name and name[:2] in LEGS and 'Tarsus' in name}
        hinges = [model.joint(f'joint_{leg}Tibia_pitch').id for leg in LEGS]
        assert np.all(model.jnt_type[hinges] == mj.mjtJoint.mjJNT_HINGE)
        self.tibia_qpos = model.jnt_qposadr[hinges]
        self.tibia_dof = model.jnt_dofadr[hinges]
        self.weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))

    def measure(self, data):
        forces = ground_loads(self.model, data, self.floor, self.foot_by_geom)
        contacts = np.zeros(len(LEGS), dtype=np.int64)
        for contact in data.contact:
            first, second = map(int, contact.geom)
            other = second if first == self.floor else first if second == self.floor else None
            if other in self.foot_by_geom:
                contacts[self.foot_by_geom[other]] += 1
        return dict(tibia_angle_rad=data.qpos[self.tibia_qpos].copy(),
                    tibia_velocity_rad_s=data.qvel[self.tibia_dof].copy(),
                    foot_contacts=contacts, foot_normal_native=forces[:-1],
                    ground_load_w=forces / self.weight,
                    torso=np.array([data.subtree_com[0, 2], data.xmat[self.thorax, 8]]))


def run(model, measure, qpos, command=None):
    data = load_pose(model, qpos)
    if command is not None:
        data.act[:] = command
        data.ctrl[:] = command
        mj.mj_forward(model, data)
    steps = round(.2 / model.opt.timestep)
    samples = []
    for tick in range(steps + 1):
        samples.append(dict(time=data.time, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            activation=data.act.copy(), native_hinge_actuator_torque=data.qfrc_actuator[6:].copy(),
            **measure.measure(data)))
        if tick != steps:
            mj.mj_step(model, data)
            mj.mj_forward(model, data)
    expected = np.zeros(model.nu) if command is None else command
    np.testing.assert_array_equal(data.ctrl, expected)
    assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
    series = {k: np.asarray([row[k] for row in samples]) for k in samples[0]}
    assert all(np.isfinite(value).all() for value in series.values())
    return series


def first_time(time, mask):
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if indices.size else None


def summary(a):
    time, force, torso = a['time'], a['ground_load_w'], a['torso']
    feet = force[:, :-1].sum(axis=1)
    carrying = (force[:, :-1] > .01).sum(axis=1)
    tail = time >= .1 - 1e-12
    gates = dict(three_loaded=np.all(carrying[tail] >= 3), zero_nonfoot=np.all(force[tail, -1] == 0),
                 foot_range=np.all((feet[tail] >= .8) & (feet[tail] <= 1.2)),
                 com_retained=np.all(torso[tail, 0] >= .9 * torso[0, 0]), upright=np.all(torso[tail, 1] >= .95))
    return dict(initial_foot_weight=float(feet[0]), first_nonfoot_s=first_time(time, force[:, -1] > 0),
        first_under_08_foot_weight_s=first_time(time, feet < .8),
        first_under_three_loaded_feet_s=first_time(time, carrying < 3), first_com_drop_10pct_s=first_time(time, torso[:, 0] < .9 * torso[0, 0]),
        mean_tail_foot_weight=float(feet[tail].mean()), lowest_com_mm=float(torso[:, 0].min()),
        gates={k: bool(v) for k, v in gates.items()}, stand=bool(all(gates.values())))


def main(out):
    model = mj.MjModel.from_xml_path(str(MODEL))
    observations = BodyObservations(model)
    with np.load(BASE / 'body/archived_pose.npz', allow_pickle=False) as a:
        keyframe, archived = model.key_qpos[0].copy(), a['qpos']
    lp_path = BASE / 'results/static_archived_six_feet_all_muscles.npz'
    with np.load(lp_path, allow_pickle=False) as a:
        command = a['solution'][:model.nu]
    pose_trials = dict(keyframe=(keyframe, None), archived_six_feet=(archived, None),
                       archived_six_feet_constant_LP=(archived, command))
    reports = {}
    for case, (pose, u) in pose_trials.items():
        arrays = run(model, observations, pose, u)
        if u is None:
            with np.load(BASE / 'results' / f'native_{case}.npz', allow_pickle=False) as old:
                for k in old.files:
                    np.testing.assert_array_equal(arrays[k], old[k], err_msg=f'{case} {k}')
        reports[case] = summary(arrays)
        np.savez_compressed(out / f'BODY_{case}.npz', **arrays)
    (out / 'BODY_OBSERVATIONS.json').write_text(json.dumps(dict(legs=LEGS,
        units={'foot_normal_native': 'MuJoCo native units', 'angle': 'rad', 'speed': 'rad/s'},
        exact_previous_arrays=12, native_steps=6000, trials=reports), indent=2) + '\n')
    print(reports)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
