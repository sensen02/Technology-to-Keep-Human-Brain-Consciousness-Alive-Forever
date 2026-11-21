"""Drive the one-command actuator from a virtual droplet at the moving oral mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from engine.embodied.feeding import FeedingBody, FeedingState, Food


RADIUS_MM = .08
METABOLIC_PERIOD_S = .01
DURATION_S = 1.2


def advance_mouth(model, data):
    geometry = model.geom('nmf/c_haustellum').id
    mesh = model.geom_dataid[geometry]
    assert mesh >= 0
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    verts = model.mesh_vert[first:first + count]
    base = data.body('nmf/c_rostrum').xpos
    world = verts @ data.geom_xmat[geometry].reshape(3, 3).T + data.geom_xpos[geometry]
    tip_local = verts[np.argmax(np.linalg.norm(world - base, axis=1))].copy()
    return geometry, tip_local


def mouth_xyz(data, geometry, point):
    return data.geom_xmat[geometry].reshape(3, 3) @ point + data.geom_xpos[geometry]


def run(body_archive, output):
    model_path = body_archive / 'BODY48.mjb'
    pose_path = body_archive / 'CANONICAL_INIT.npz'
    model = mj.MjModel.from_binary_path(str(model_path))
    data = mj.MjData(model)
    with np.load(pose_path, allow_pickle=False) as pose:
        for field in ('qpos', 'qvel', 'act', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied',
                      'eq_active', 'userdata', 'mocap_pos', 'mocap_quat'):
            getattr(data, field)[:] = pose[field]
        data.time = float(pose['time'])
        positional = np.flatnonzero(model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT)
        assert len(positional) == 42
        data.ctrl[positional] = pose['ctrl_joint']
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    mj.mj_forward(model, data)
    geom, point = advance_mouth(model, data)
    origin = mouth_xyz(data, geom, point)
    centers = np.array([origin, origin + np.array([.5, 0., 0.]), origin])
    names = ('near', 'far', 'near_blocked')
    states = [FeedingState(Food(.2, .2)) for _ in names]
    traces = {name: [] for name in ('time', 'tip', 'distance', 'contact', 'ingested', 'crop', 'command')}
    body = FeedingBody()
    every = round(METABOLIC_PERIOD_S / model.opt.timestep)
    ticks = round(DURATION_S / model.opt.timestep)
    assert np.isclose(every * model.opt.timestep, METABOLIC_PERIOD_S)
    for tick in range(1, ticks + 1):
        mj.mj_step(model, data)
        if tick % every:
            continue
        # Refresh world geometry; mj_step has completed integration of qpos.
        mj.mj_kinematics(model, data)
        tip = mouth_xyz(data, geom, point)
        distances = np.linalg.norm(centers - tip, axis=1)
        contact = distances <= RADIUS_MM
        time = tick * model.opt.timestep
        feed = bool(.2 <= time < .8)
        samples = [body.advance(state, METABOLIC_PERIOD_S, feed=feed,
                                contact=bool(near), blocked=i == 2)
                   for i, (state, near) in enumerate(zip(states, contact, strict=True))]
        states = [sample.state for sample in samples]
        values = (time, tip, distances, contact, [sample.ingested.volume for sample in samples],
                  [sample.crop_stretch.value for sample in samples], feed)
        for key, value in zip(traces, values, strict=True):
            traces[key].append(value)
    arrays = {key: np.array(values) for key, values in traces.items()}
    assert not data.warning.number.any()
    assert np.isfinite(arrays['tip']).all()
    assert not arrays['ingested'][~arrays['command']].any()
    assert arrays['ingested'][:, 0].sum() > .1
    assert not arrays['ingested'][:, 1:].any()
    assert np.array_equal(arrays['contact'][:, 0], arrays['contact'][:, 2])
    assert arrays['contact'][arrays['command'], 0].all()
    output.mkdir()
    np.savez_compressed(output / 'TRACE.npz', **arrays)
    result = dict(
        status='REGISTERED_MOUTH_POINT_CONTACT_PASS', conditions=names, native_steps=ticks,
        body_bench_inputs='one archived stance; held archived leg targets; commanded intake',
        droplet_radius_mm=RADIUS_MM, intake_by_condition=arrays['ingested'].sum(axis=0).tolist(),
        largest_mouth_displacement_mm=float(np.linalg.norm(arrays['tip'] - origin, axis=1).max()),
        body_ascending_neural_delivery=False,
        hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (model_path, pose_path, Path(__file__).resolve(),
                             Path(__file__).resolve().parents[3] / 'engine/embodied/feeding.py')})
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('body_archive', type=Path)
    p.add_argument('output', type=Path)
    params = p.parse_args()
    run(params.body_archive, params.output)
