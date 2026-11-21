"""Verify rendered fruit visibility, wall geometry, odor and the oral gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

from fruit_field import FRUIT, FRUIT_RADIUS, DX, OdorField, Wall, teacher_route

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get('P1_ARCHIVE_ROOT', '/run/media/sensen/Data2/cell_wound_prototype'))
BODY_DIR = ARCHIVE / 'experiments/open-brain/20261117T184709Z-body-terrain-repair'
PREVIOUS = HERE.parent / '20261118T195801Z-feeding-interoception'
sys.path[:0] = [str(BODY_DIR), str(HERE.parents[2]), str(PREVIOUS)]
import body_model
from engine.embodied.feeding import FeedingBody, FeedingState, Food
from mouth_bench import advance_mouth, mouth_xyz


def build_scene(wall):
    base_terrain = body_model.Terrain
    make_original = body_model.make_locomotion_fly

    def make_fly(**options):
        fly = make_original(**options)
        fly.add_vision()
        return fly

    class FruitTerrain(base_terrain):
        def __init__(self, height):
            super().__init__(height)
            world = self.mjcf_root.worldbody
            world.add_geom(name='fruit', type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                           pos=FRUIT, size=(FRUIT_RADIUS, .64, .78),
                           rgba=(.86, .12, .052, 1.), contype=0, conaffinity=0)
            if wall is not None:
                world.add_geom(name='fruit_wall', type=mj.mjtGeom.mjGEOM_BOX,
                               pos=(wall.x, 0., wall.height / 2),
                               size=(wall.half_width, wall.half_length, wall.height / 2),
                               rgba=(.32, .40, .44, 1.), contype=0, conaffinity=0)

        def add_fly(self, fly, *args, **kwargs):
            result = super().add_fly(fly, *args, **kwargs)
            if wall is not None:
                for geoms in fly.bodyseg_to_mjcfgeom.values():
                    for g in geoms:
                        self.mjcf_root.add_pair(geomname1=g.name, geomname2='fruit_wall',
                                                friction=(1., 1., .02, .0001, .0001))
            return result

    body_model.Terrain = FruitTerrain
    body_model.make_locomotion_fly = make_fly
    try:
        return body_model.Body(adhesion=True)
    finally:
        body_model.Terrain = base_terrain
        body_model.make_locomotion_fly = make_original


def place(body, snapshot, xy, yaw):
    body.d.qpos[:] = snapshot['qpos']
    body.d.qpos[:2] = xy
    rotation = np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
    orientation = np.empty(4)
    mj.mju_mulQuat(orientation, rotation, body.d.qpos[3:7])
    body.d.qpos[3:7] = orientation
    body.d.qvel[:] = snapshot['qvel']
    body.d.ctrl[body.actuators] = snapshot['ctrl_joint']
    body.d.ctrl[body.adhesion_ids] = 0.
    mj.mj_forward(body.m, body.d)


def antennae(body):
    return np.stack([body.d.body(f'nmf/{side}_funiculus').xpos.copy() for side in ('l', 'r')])


def render_pair(body):
    fruit = body.m.geom('fruit').id
    original = body.m.geom_group[fruit]
    visible = body.sim.get_raw_vision(body.name)
    try:
        body.m.geom_group[fruit] = 3  # Default eye rendering excludes this group.
        blank = body.sim.get_raw_vision(body.name)
    finally:
        body.m.geom_group[fruit] = original
    changed = (np.max(abs(visible.astype(int) - blank.astype(int)), axis=-1) > 5).sum(axis=(1, 2))
    return visible, blank, changed.tolist()


def run(output):
    wall = Wall()
    fields = {'open': OdorField(), 'wall': OdorField(wall),
              'sealed': OdorField(Wall(half_length=7.)),
              'no_odor': OdorField(wall, source_strength=0.)}
    configurations = {
        'open': (False, (0., 0.), 0.),
        'behind_wall': (True, (0., 0.), 0.),
        'past_left_edge': (True, (5.9, 3.8), -.9),
        'past_right_edge': (True, (5.9, -3.8), .9),
    }
    output.mkdir()
    models = {has_wall: build_scene(wall if has_wall else None) for has_wall in (False, True)}
    for flag, body in models.items():
        mj.mj_saveModel(body.m, str(output / ('wall.mjb' if flag else 'open.mjb')), None)
    with np.load(BODY_DIR / 'CANONICAL_INIT.npz', allow_pickle=False) as source:
        snapshot = {k: source[k] for k in ('qpos', 'qvel', 'ctrl_joint')}
    cameras = {}
    rows = {}
    for name, (has_wall, xy, yaw) in configurations.items():
        body = models[has_wall]
        place(body, snapshot, xy, yaw)
        sensors = antennae(body)
        image, blank, changed = render_pair(body)
        cameras[name], cameras[name + '_without_fruit'] = image, blank
        rows[name] = dict(
            xy_mm=list(xy), yaw_rad=yaw, antennae_xyz_mm=sensors.tolist(),
            visibly_changed_pixels=changed,
            odor=fields['wall' if has_wall else 'open'].sample(sensors).tolist())
    for body in models.values():
        body.sim.eye_renderer.close()
    assert sum(rows['open']['visibly_changed_pixels']) > 20
    assert sum(rows['behind_wall']['visibly_changed_pixels']) == 0
    assert all(sum(rows[name]['visibly_changed_pixels']) > 20
               for name in ('past_left_edge', 'past_right_edge'))
    blocked_start = np.asarray(rows['behind_wall']['antennae_xyz_mm'])
    assert np.all(fields['wall'].sample(blocked_start) > 0.)
    assert not fields['sealed'].sample(blocked_start).any()
    assert not fields['no_odor'].sample(blocked_start).any()
    for field in fields.values():
        assert field.residual < 1e-10 and field.balance_error < 1e-10
    # Test actual collidable wall construction without integrating a trajectory.
    body = models[True]
    wall_index = body.m.geom('fruit_wall').id
    wall_contacts = 0
    for x in np.linspace(3.4, 4.3, 10):
        place(body, snapshot, (x, 0.), 0.)
        wall_contacts = max(wall_contacts, sum(
            wall_index in (pair.geom1, pair.geom2) for pair in body.d.contact[:body.d.ncon]))
    assert wall_contacts > 0
    place(body, snapshot, (0., 0.), 0.)
    geom, tip = advance_mouth(body.m, body.d)
    mouth_offset = mouth_xyz(body.d, geom, tip)
    feeding_xy = FRUIT[:2] - mouth_offset[:2] - np.array([.20, 0.])
    routes = {name: teacher_route((0., 0.), feeding_xy, barrier)
              for name, barrier in [('open', None), ('wall', wall), ('sealed', Wall(half_length=7.))]}
    assert routes['open'] is not None and routes['wall'] is not None and routes['sealed'] is None
    assert len(routes['wall']) > len(routes['open'])
    # Use a grid-aligned mouth target that agrees with the geometry route end.
    place(body, snapshot, routes['wall'][-1], 0.)
    distance = float(np.linalg.norm(mouth_xyz(body.d, geom, tip) - FRUIT))
    assert distance <= FRUIT_RADIUS
    feeding = FeedingBody()
    state = FeedingState(Food(.2, .2))
    intake = {
        'near': feeding.advance(state, .01, feed=True, contact=distance <= FRUIT_RADIUS).ingested.volume,
        'far': feeding.advance(state, .01, feed=True, contact=False).ingested.volume,
        'blocked': feeding.advance(state, .01, feed=True, contact=True, blocked=True).ingested.volume,
    }
    assert intake['near'] > 0. and intake['far'] == intake['blocked'] == 0.
    np.savez_compressed(output / 'SENSORS.npz', **cameras,
                        route_open=routes['open'], route_wall=routes['wall'],
                        odor_open=fields['open'].value, odor_wall=fields['wall'].value,
                        odor_sealed=fields['sealed'].value)
    inputs = [Path(__file__), HERE/'fruit_field.py', BODY_DIR/'CANONICAL_INIT.npz',
              BODY_DIR/'body_model.py', BODY_DIR/'JOINT_LIMITS.json', PREVIOUS/'mouth_bench.py',
              HERE.parents[2]/'engine/embodied/feeding.py']
    result = dict(status='FRUIT_SCENE_SENSOR_GATE_PASS', sensor_stations=rows,
                  route_length_mm={name: (len(route)-1)*DX if route is not None else None
                                   for name, route in routes.items()},
                  wall_contact_probe_max=wall_contacts, oral_distance_mm=distance,
                  commanded_intake_probe=intake, odor_open_end=fields['wall'].sample(blocked_start).tolist(),
                  odor_sealed=fields['sealed'].sample(blocked_start).tolist(),
                  odor_off=fields['no_odor'].sample(blocked_start).tolist(),
                  odor_numerical_residual={key: f.residual for key,f in fields.items()},
                  physics_steps=0, eon_steps=0, training_updates=0,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
