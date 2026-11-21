"""Same-body physical reachability control using privileged geometric waypoints."""
import argparse
import json
from pathlib import Path

import numpy as np

from preflight import (
    BODY_DIR, FRUIT, FRUIT_RADIUS, FeedingBody, FeedingState, Food, Wall,
    advance_mouth, build_scene, mouth_xyz, place,
)
from fruit_field import teacher_route


def simplify(route, wall):
    if wall is None:
        return route[-1:]
    targets = []
    i = 0
    while i < len(route) - 1:
        for j in range(len(route) - 1, i, -1):
            line = np.linspace(route[i], route[j], 120)
            if not wall.contains_xy(line, clearance=1.0).any():
                targets.append(route[j])
                i = j
                break
        else:
            raise ValueError("point sequence cannot be shortened with clearance")
    return np.array(targets)


def run(mode, preflight, out, seeds, clearance, mirror):
    wall = Wall() if mode == 'wall' else None
    with np.load(preflight/'SENSORS.npz', allow_pickle=False) as data:
        route = data['route_' + mode]
    if wall is not None:
        planner_wall = Wall(wall.x, wall.half_width + clearance, wall.half_length + clearance, wall.height)
        route = teacher_route(route[0], route[-1], planner_wall)
    else:
        planner_wall = None
    targets = simplify(route, planner_wall)
    if mirror:
        targets[:, 1] *= -1
    with np.load(BODY_DIR/'CANONICAL_INIT.npz', allow_pickle=False) as data:
        snapshot = {k: data[k] for k in ('qpos', 'qvel', 'ctrl_joint')}
    out.mkdir()
    all_results = []
    for seed in seeds:
        body = build_scene(wall)
        place(body, snapshot, (0., 0.), 0.)
        body.motor.reset(seed=seed)
        geom, point = advance_mouth(body.m, body.d)
        intake_state = FeedingState(Food(1., 1.))
        feeder = FeedingBody()
        dt = body.m.opt.timestep
        count = round(5.0 / dt)
        wall_g = body.m.geom('fruit_wall').id if wall is not None else -1
        target_idx = 0
        drive = np.zeros(2)
        consumed = 0.
        first_intake = None
        contact_steps = 0
        rows = []
        wall_contact_native = 0
        warning = False
        for i in range(count):
            if i % 100 == 0:
                position = body.d.xpos[body.thorax].copy()
                rotation = body.d.xmat[body.thorax].reshape((3, 3))
                tip = mouth_xyz(body.d, geom, point)
                dist = float(np.linalg.norm(tip - FRUIT))
                mouth_contact = dist <= FRUIT_RADIUS
                contact_steps += mouth_contact
                if target_idx < len(targets) - 1 and np.linalg.norm(position[:2] - targets[target_idx]) <= .45:
                    target_idx += 1
                delta = targets[target_idx] - position[:2]
                distance = float(np.linalg.norm(delta))
                yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                error = float((np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
                steer = np.clip(.85 * error, -.69, .69)
                forward = np.clip(.58 * distance, .22, .76) * max(0., np.cos(error))
                drive[:] = (np.clip(forward - steer, -1., 1.), np.clip(forward + steer, -1., 1.))
                if mouth_contact or i < 2000:
                    drive.fill(0.)
                receipt = feeder.advance(intake_state, dt * 100, feed=bool(mouth_contact),
                                         contact=bool(mouth_contact))
                intake_state = receipt.state
                consumed += receipt.ingested.volume
                if first_intake is None and receipt.ingested.volume > 0.:
                    first_intake = i * dt
                rows.append(np.r_[i * dt, position, tip, dist, consumed, drive,
                                  target_idx, rotation[2, 2]])
            control, pads = body.teacher(drive)
            body.apply(control, pads)
            if wall_g >= 0:
                wall_contact_native += any(wall_g in (c.geom1, c.geom2) and not c.exclude
                                           for c in body.d.contact[:body.d.ncon])
            if body.d.warning.number.any() or not np.isfinite(body.d.qpos).all():
                warning = True
                break
        a = np.asarray(rows)
        np.savez_compressed(out/f'TRACE{seed}.npz', trace=a, waypoints=targets)
        metrics = dict(
            seed=seed, steps=i+1, mode=mode, first_intake_s=first_intake, intake_units=consumed,
            min_distance_mm=float(a[:, 7].min()), last_position_mm=a[-1, 1:4].tolist(),
            last_waypoint=target_idx, mouth_contact_samples=int(contact_steps),
            min_upright=float(a[:, -1].min()), wall_contact_native=int(wall_contact_native),
            reached_and_ate=bool(consumed >= .04 and not warning and a[:, -1].min() > .8
                                 and not wall_contact_native),
            privileged_world_positions=True, uses_eon=False, uses_banc=False)
        metrics['planner_clearance_mm'] = 1. + clearance if wall is not None else None
        metrics['mirrored_waypoints'] = mirror
        all_results.append(metrics)
        print(json.dumps(metrics), flush=True)
    (out/'RESULT.json').write_text(json.dumps(all_results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('open', 'wall'))
    p.add_argument('preflight', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('seeds', type=int, nargs='+')
    p.add_argument('--clearance', type=float, default=.65)
    p.add_argument('--mirror', action='store_true')
    cfg = p.parse_args()
    run(cfg.mode, cfg.preflight, cfg.output, cfg.seeds, cfg.clearance, cfg.mirror)
