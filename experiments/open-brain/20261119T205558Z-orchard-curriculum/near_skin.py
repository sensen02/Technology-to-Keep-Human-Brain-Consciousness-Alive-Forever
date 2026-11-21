"""Third-stage curriculum using a checked start at the fruit surface."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from rollout import HERE,tr,main

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--spawn',required=True,choices=['bite','approach'])
    extra,args=parser.parse_known_args()
    sys.argv[1:]=args
    site=sys.argv[1]
    folder=Path(sys.argv[2])
    rows=json.loads((HERE/'SPAWNS.json').read_text())[site][extra.spawn]
    if not rows:
        raise ValueError('No inspected pose meets this gate')
    pose=rows[0]
    registration=tr.fit_spawn
    def fit(crop,env,base_start):
        original=crop.floor
        crop.floor=lambda yx: original(np.asarray(yx)+[pose['y'],pose['x']])
        try:
            q=registration(crop,env,base_start)
        finally:
            crop.floor=original
        q[:2]+=[pose['x'],pose['y']]
        rot=tr.Rotation.from_rotvec([0.,0.,pose['yaw']])*tr.Rotation.from_quat(q[[4,5,6,3]])
        q[3:7]=rot.as_quat()[[3,0,1,2]]
        with env.physics.reset_context():
            env.physics.data.qpos[:]=q
        return q
    tr.fit_spawn=fit
    main()
    cfg=json.loads((folder/'CONFIG.json').read_text())
    cfg.update(start_x=pose['x'],start_y=pose['y'],start_yaw=pose['yaw'],spawn_condition=extra.spawn)
    (folder/'CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
