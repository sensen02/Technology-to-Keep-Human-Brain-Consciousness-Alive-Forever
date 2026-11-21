"""Static camera probe for terrain occlusion at the same spawn, before any control."""
from dataclasses import asdict
import json

import numpy as np
from PIL import Image

from terrain_spec import HERE
from common import OrchardEyes, observation, start_pose
from live_compound import make_board

SCENES = (('wide_bridge',15671,-1), ('slope_bridge',27261,1), ('slope_bridge',27261,-1),
          ('staggered_slope',27267,1), ('staggered_slope',27267,-1))


def probe():
    from terrain_spec import CLOSED
    with np.load(CLOSED/'VISION/VISION.npz', allow_pickle=False) as archive:
        weights = archive['weights']
    rows = []
    for stage,seed,side in SCENES:
        env, cfg, stock = make_board(seed,stage,side)
        env.reset()
        start_pose(env)
        phy = env.physics
        eyes = OrchardEyes(phy)
        origin, forward, up = eyes.head_pose()
        result = []
        for elevation in (0., .025, .050, .10):
            left,right,items,_ = eyes.sample((origin+elevation*up,forward,up))
            red,green,blue = left.astype(np.int32).transpose(2,0,1)
            red_pixels = (red>=80) & (red-green>=31) & (red*100>=green*195) & (red-blue>=37)
            coords = np.argwhere(red_pixels)
            row_count = int(np.count_nonzero(red_pixels.any(axis=1)))
            area = int(red_pixels.sum())
            raw_box = (tuple(coords.min(axis=0).tolist())+tuple(coords.max(axis=0).tolist())) if area else None
            result.append(dict(probe_elevation_mm=10*elevation,raw_red_pixels=area,
                               red_rows=row_count,raw_yx_bounds=raw_box,
                               candidates=[asdict(item) for item in items]))
            if stage == 'staggered_slope' and side == 1 and elevation in (0., .10):
                Image.fromarray(left).save(HERE/f'VIS_STATIC_{round(elevation*100):02d}.png')
        baseline, *_ = observation(eyes,weights,(0.,0.),0.,.23)
        rows.append(dict(stage=stage,seed=seed,side=side,visible_feature=float(baseline[0]),
                         actual_hazard=baseline[4:10].tolist(),probe=result))
        print(stage, seed, side, [(point['red_rows'],len(point['candidates'])) for point in result], flush=True)
        env.close()
    (HERE/'VISION_PREFLIGHT.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    probe()
