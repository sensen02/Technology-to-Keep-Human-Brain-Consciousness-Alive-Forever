"""Render the same courses used by the proxy and run the live P1 neural loop."""
import argparse
from pathlib import Path

import numpy as np

from terrain_spec import CLOSED, STAGES, make_terrain
from common import FRUIT_RADIUS, FoodBank, Resource, build_env, detail, init_course, solid
import run_course_docking


COLOR = (.49, .45, .32, 1.)
Y_LIMIT = 1.40
BOTTOM = -.26
PRISM_FACES = (0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
               1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7)


def ground_segment(arena, name, a, b, low, high, za, zb):
    if abs(zb - za) < 1e-12:
        solid(arena.worldbody, name, 'box',
              pos=((a+b)/2, (low+high)/2, (za+BOTTOM)/2),
              size=((b-a)/2, (high-low)/2, (za-BOTTOM)/2), rgba=COLOR)
        return
    vertices = (a, low, BOTTOM, b, low, BOTTOM, b, high, BOTTOM, a, high, BOTTOM,
                a, low, za, b, low, zb, b, high, zb, a, high, za)
    mesh = arena.asset.add('mesh', name=name, vertex=vertices, face=PRISM_FACES)
    solid(arena.worldbody, name, 'mesh', mesh=mesh, rgba=COLOR)


def make_board(seed, challenge, side, *, food=True):
    if challenge in ('flat', 'wide_bridge'):
        return init_course(seed, challenge, side, food=food)
    cfg = make_terrain(seed, challenge, side)
    env = build_env()
    env.random_state.seed(29)
    arena = env.task._arena.mjcf_model
    world = arena.worldbody
    plane = arena.find('geom', 'groundplane')
    plane.material = None
    plane.pos = (0., 0., BOTTOM)
    plane.rgba = (.115, .025, .50, 1.)
    world.add('light', name='sky', directional=True, pos=(0, 0, 5), dir=(.15, .25, -1),
              ambient=(.49, .49, .48), diffuse=(.45, .46, .47), castshadow=False)
    cuts = sorted({x for x, _ in cfg.height_profile} |
                  {x for gate in cfg.gates for x in (gate.near, gate.far)})
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        middle = (a+b)/2
        gate = next((g for g in cfg.gates if g.near < middle < g.far), None)
        lo, hi = -Y_LIMIT, Y_LIMIT
        if gate:
            lo, hi = (gate.edge, hi) if gate.side == 1 else (lo, -gate.edge)
        ground_segment(arena, f'course_ground_{i}', a, b, lo, hi, cfg.height(a), cfg.height(b))
    for i, g in enumerate(cfg.gates):
        lo, hi = (-Y_LIMIT, g.edge) if g.side == 1 else (-g.edge, Y_LIMIT)
        for j, x in enumerate((g.near-.024, g.far+.024)):
            detail(world, f'course_lip_{i}_{j}', 'box',
                   pos=(x, (lo+hi)/2, cfg.height(x)+.002),
                   size=(.023, (hi-lo)/2, .002), rgba=(.095, .015, .75, 1.))
    for eye in ('left', 'right'):
        world.add('camera', name='orchard_eye_'+eye, fovy=56)
    resources = []
    if food:
        top = float(cfg.height(cfg.goal_x))+.004
        fruit = Resource('trial_fruit', 'ground', cfg.goal_x, cfg.goal_y, top, FRUIT_RADIUS, .9)
        detail(world, fruit.geom, 'cylinder', pos=(fruit.x, fruit.y, top-.002),
               size=(fruit.radius, .002), rgba=(.84, .065, .024, 1.))
        resources.append(fruit)
    return env, cfg, FoodBank(resources)


def run(args):
    args.profile, args.no_food, args.early, args.neural_seed = 'strong', False, True, 29
    args.vision = CLOSED/'VISION/VISION.npz'
    run_course_docking.init_course = make_board
    run_course_docking.run(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--challenge', choices=STAGES, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--side', type=int, choices=(-1, 1), required=True)
    p.add_argument('--duration', type=float, required=True)
    run(p.parse_args())
