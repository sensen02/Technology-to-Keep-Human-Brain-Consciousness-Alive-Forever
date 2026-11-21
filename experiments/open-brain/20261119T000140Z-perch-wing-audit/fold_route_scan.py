"""Trace static wing-to-abdomen clearance across candidate stow paths."""
import json

import mujoco as mj
import numpy as np

from inspect_pose import FLIGHT, build_scene


def main():
    case = FLIGHT / 'NEAR'
    data = json.loads((case / 'RESULT.json').read_text())
    poses = np.load(case / 'QPOS.npy', allow_pickle=False)
    env = build_scene(data['geometry']['x_cm'])
    env.reset()
    phy = env.physics
    m, d = phy.model, phy.data
    names = [f'walker/wing_{a}_{s}' for s in ('left', 'right') for a in ('yaw', 'roll', 'pitch')]
    wing = [m.name2id(name, 'joint') for name in names]
    dofs = m.jnt_qposadr[wing]
    geoms = [(m.name2id(f'walker/wing_{side}_{surface}_collision', 'geom'),
              m.name2id(f'walker/abdomen_{i}_collision', 'geom'))
             for side in ('left', 'right') for surface in ('membrane', 'brown') for i in range(4, 8)]
    t = data['fold_s']
    root = poses[round(t / .0002) - 1].copy()
    launch = root[dofs].copy()
    parked = m.qpos_spring[dofs].copy()
    print('initial:', launch.round(4), 'final:', parked.round(4))
    d.qpos[:] = root
    phy.forward()
    for name, radians in [
        ('direct', []), ('lift', [(1.23, -.45, .25), (1.5, .05, -1.)]),
        ('dorsal', [(1.25, -.30, -.85), (1.50, .25, -1.)]),
        ('pitch', [(1.10, -.10, -1.20), (1.5, .1, -1.20)]),
        ('spread', [(.45, -.45, .40), (.45, -.40, -1.1), (1.45, .15, -1.05)]),
        ('out', [(.72, .0, 1.5), (.72, -.55, -1.0), (1.48, .15, -1.0)]),
        ('yaw_first', [(1.5, .04, 1.4), (1.5, .05, -.8)]),
    ]:
        points = [launch, *(np.tile(x, 2) for x in radians), parked]
        gaps = []
        for a, b in zip(points, points[1:]):
            gap = []
            for frac in np.linspace(0, 1, 25):
                d.qpos[dofs] = (1 - frac) * a + frac * b
                mj.mj_kinematics(m.ptr, d.ptr)
                gap.append(min(mj.mj_geomDistance(m.ptr, d.ptr, x, y, .4, None)
                               for x, y in geoms) * 10.)
            gaps.append([round(min(gap), 3), round(gap[-1], 3)])
        print(name, gaps)


if __name__ == '__main__':
    main()
