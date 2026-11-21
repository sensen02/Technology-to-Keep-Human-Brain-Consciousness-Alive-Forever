"""Learn fruit bearing and range from the same FlyGym eyes and antennae used online."""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'20261118T223412Z-fruit-cues'
sys.path.insert(0, str(PRIOR))
from preflight import BODY_DIR, FRUIT, antennae, build_scene, place
from fruit_field import OdorField


class Arena:
    """Zero-wind open arena translated with the fruit. Shape and odor remain co-located."""
    def __init__(self, fruit):
        self.fruit = np.asarray(fruit, dtype=float)
        self.shift = FRUIT - self.fruit
        self.field = OdorField()

    def smell(self, points):
        return self.field.sample(points + self.shift)


def vision_features(rgb):
    small = rgb[:, ::3, ::3].astype(np.int16)
    red, green, blue = small.transpose((3, 0, 1, 2))
    mask = (red - np.maximum(green, blue) > 38) & (red > 60)
    count = mask.sum(axis=(1, 2))
    if not count.any():
        return np.zeros(18), False
    h, w = red.shape[1:]
    cx = mask.sum(axis=1) @ np.linspace(-1., 1., w) / np.maximum(count, 1)
    cy = mask.sum(axis=2) @ np.linspace(-1., 1., h) / np.maximum(count, 1)
    area = np.sqrt(count / (h*w)) * 12.
    return np.r_[1., (count > 0), cx, cy, area, cx*area, cy*area, cx*cx,
                 area*area, np.log1p(count.sum())], True


def odor_features(raw):
    left, right = raw
    total = left+right
    d = (left-right)/(total + 1e-12)
    power = np.log1p(total/.025)
    return np.array([1., d, power, d*power, power*power, d*power*power, d**3])


def estimate(model, vf, visible, chemical):
    visual = vf @ np.array(model['vision_weights']) if visible else None
    if chemical.sum() > 1e-8:
        smell = odor_features(chemical) @ np.array(model['odor_weights'])
        if visual is None:
            return float(np.clip(smell[0], -.95, .95)), float(np.clip(smell[1], .10, 1.1)), True
    if visual is None:
        return 0., 0., False
    if chemical.sum() > 1e-8:
        visual[0] = .91*visual[0] + .09*smell[0]
    return float(np.clip(visual[0], -1.1, 1.1)), float(np.clip(visual[1], .10, 1.1)), True


def fit_ridge(x, y, l2):
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def collect(seed, frames, body, snapshot, field, root):
    place(body, snapshot, (0., 0.), 0.)
    body.motor.reset(seed=seed)
    rng = np.random.default_rng(seed)
    visual, chemical, teachers, images = [], [], [], []
    for row in range(frames):
        # Sample a moving body, using the existing physical teacher. Privileged
        # geometry generates labels only. Fruit geometry has no contact pairs.
        wing = np.sin(row*.31)*.26
        command = np.array([.56-wing, .56+wing])
        for _ in range(55):
            control, pads = body.teacher(command)
            body.apply(control, pads)
        mj.mj_forward(body.m, body.d)
        thorax = body.d.xpos[body.thorax].copy()
        orientation = body.d.xmat[body.thorax].reshape(3, 3)
        heading = np.arctan2(orientation[1, 0], orientation[0, 0])
        target_range = rng.uniform(1.02, 6.1)
        target_bearing = rng.uniform(-.90, .90)
        heading += target_bearing
        location = np.array([thorax[0]+target_range*np.cos(heading),
                             thorax[1]+target_range*np.sin(heading), FRUIT[2]])
        body.m.geom_pos[root] = location
        mj.mj_forward(body.m, body.d)
        rgb = body.sim.get_raw_vision(body.name)
        vf, seen = vision_features(rgb)
        sniff = field.sample(antennae(body) + FRUIT - location)
        if seen:
            visual.append(vf)
            chemical.append(odor_features(sniff))
            teachers.append([target_bearing, 1./target_range])
            if len(visual) <= 4:
                images.append(rgb[:, ::3, ::3])
    return np.asarray(visual), np.asarray(chemical), np.asarray(teachers), np.asarray(images)


def run(out):
    out.mkdir()
    body = build_scene(None)
    fruit = body.m.geom('fruit').id
    field = OdorField()
    with np.load(BODY_DIR/'CANONICAL_INIT.npz', allow_pickle=False) as src:
        snapshot = {k: src[k] for k in ('qpos', 'qvel', 'ctrl_joint')}
    a = collect(41, 150, body, snapshot, field, fruit)
    b = collect(937, 65, body, snapshot, field, fruit)
    body.sim.eye_renderer.close()
    vw = fit_ridge(a[0], a[2], .002)
    ow = fit_ridge(a[1], a[2], .001)
    metrics = {}
    for mode, data in [('train', a), ('holdout', b)]:
        pred = data[0] @ vw
        olf = data[1] @ ow
        metrics[mode] = dict(samples=len(data[0]),
             vision_angle_mae_deg=float(np.rad2deg(abs(pred[:,0]-data[2][:,0]).mean())),
             vision_inverse_range_mae=float(abs(pred[:,1]-data[2][:,1]).mean()),
             odor_angle_mae_deg=float(np.rad2deg(abs(olf[:,0]-data[2][:,0]).mean())))
    result = dict(vision_weights=vw.tolist(), odor_weights=ow.tolist(), metrics=metrics,
                  training_algorithm='two ridge regressions from labeled moving-eye and antenna samples',
                  teacher_physics_steps=(150+65)*55)
    (out/'FIT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(out/'TRAIN.npz', train_vision=a[0], train_odor=a[1], train_teacher=a[2],
                    held_vision=b[0], held_odor=b[1], held_teacher=b[2], held_images=b[3])
    assert metrics['holdout']['samples'] >= 45
    assert metrics['holdout']['vision_angle_mae_deg'] < 14.
    print(metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
