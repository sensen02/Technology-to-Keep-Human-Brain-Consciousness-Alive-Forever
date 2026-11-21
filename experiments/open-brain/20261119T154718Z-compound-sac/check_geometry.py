"""Confirm compound-course collision surface against analytic replay terrain."""
import json
from pathlib import Path

from dm_control.mujoco import engine, wrapper
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
from PIL import Image

from terrain_spec import HERE, dry_check
from live_compound import BOTTOM, make_board
from common import OrchardEyes, start_pose


def ray(physics, x, y):
    z = 1.
    gid = np.zeros(1, dtype=np.int32)
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    distance = mjlib.mj_ray(physics.model.ptr, physics.data.ptr, np.array([x, y, z]),
                            np.array([0., 0., -1.]), group, 1, -1, gid)
    return z - distance, physics.model.id2name(int(gid[0]), 'geom')


def survey(show=False):
    dry_check()
    cases = [('slope_bridge', 27261), ('staggered', 27263), ('staggered_slope', 27267)]
    result = []
    for challenge, seed in cases:
        for side in (1, -1):
            env, cfg, stock = make_board(seed, challenge, side)
            env.reset()
            start_pose(env)
            physics = env.physics
            samples = []
            for (a, _), (b, _) in zip(cfg.height_profile, cfg.height_profile[1:]):
                for fraction in (.07, .41, .88):
                    x = a + (b-a)*fraction
                    if x < .18:
                        continue
                    expected = float(cfg.height(x))
                    active = next((g for g in cfg.gates if g.near < x < g.far), None)
                    y = active.side*.48 if active else .85
                    actual, geom = ray(physics, x, y)
                    samples.append(dict(x=x, y=y, expected=expected, got=actual, geom=geom))
            for g in cfg.gates:
                for y in (g.side*.48, 0., -g.side*.53):
                    x = (g.near+g.far)/2
                    expected = BOTTOM if cfg.blocked((x,y)) else float(cfg.height(x))
                    actual, geom = ray(physics, x, y)
                    samples.append(dict(x=x, y=y, expected=expected, got=actual, geom=geom))
            fruit, = stock.resources
            assert abs(fruit.top-cfg.height(fruit.x)-.004)<1e-12
            assert np.isfinite(physics.data.qpos).all()
            max_error = max(abs(r['expected']-r['got']) for r in samples)
            print(challenge, seed, side, 'samples', len(samples), 'max error cm', max_error, flush=True)
            assert max_error < .0007, max(samples, key=lambda r: abs(r['expected']-r['got']))
            result.append(dict(challenge=challenge, seed=seed, side=side,
                               max_error_cm=max_error, samples=samples))
            if show and challenge == 'staggered_slope' and side == 1:
                option = wrapper.MjvOption()
                option.sitegroup[:]=0
                camera = engine.MovableCamera(physics, height=520, width=830)
                camera.set_pose((1.55,0.,-.06), 4.5, 89., -67.)
                left, right, *_ = OrchardEyes(physics).sample()
                Image.fromarray(camera.render(scene_option=option)).save(HERE/'COURSE_PREVIEW.png')
                width, height = left.shape[1], left.shape[0]
                board = Image.new('RGB', (2*width, height))
                board.paste(Image.fromarray(left), (0, 0))
                board.paste(Image.fromarray(right), (width, 0))
                board.save(HERE/'COURSE_EYES.png')
            env.close()
    (HERE/'GEOMETRY_AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    survey(show=True)
