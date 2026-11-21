import json
from pathlib import Path
import sys

import numpy as np

from common_eye import scene, TOP_CM

if __name__=='__main__':
    run=Path(sys.argv[1])
    result=json.loads((run/'RESULT.json').read_text())
    env=scene(result['geometry']['x_cm'],result['mode']=='elevated')
    env.reset()
    phy=env.physics
    p=np.load(run/'QPOS.npy')
    for sec in (.68,.8,1.15,1.55):
        phy.data.qpos[:]=p[min(round(sec/.0002)-1,len(p)-1)]
        phy.forward()
        names=[]
        for con in phy.data.contact[:phy.data.ncon]:
            a,b=[phy.model.id2name(int(g),'geom') for g in con.geom]
            if 'fruit' in (a,b):
                names.append((a,b,round(con.pos[2],6)))
        for name in ['walker/labrum_left_lower_collision','walker/labrum_right_lower_collision']:
            gid=phy.model.name2id(name,'geom')
            print(sec,name,phy.data.geom_xpos[gid].tolist())
        print(names,flush=True)
