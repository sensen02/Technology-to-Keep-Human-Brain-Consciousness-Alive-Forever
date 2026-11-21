"""Independent held-out mechanical contacts and single-probe withdrawal."""
import json
import numpy as np
from floor import Floor,HERE,CX,CY
from acquire import image
import sys
merged='--merged' in sys.argv
if merged:
    xy=[(CX[7,17]+d,CY[7,17]) for d in (-.05,.05)]
else:
    xy=[(CX[7,col]+.07,CY[7,col]-.045) for col in (17,19)]
f=Floor('probes',xy)
schedule=[(12,[0,0]),(15,[1.4,0]),(18,[1.4,4.8]),(16,[1.4,0]),
          (12,[0,0]),(15,[0,3.2]),(12,[0,0])]
tracks=[];known=[]
for count,loads in schedule:
    for _ in range(count):
        for _ in range(20):
            for i,force in enumerate(loads):
                joint=f.m.joint(f'weight_{i}')
                v=joint.dofadr[0];q=joint.qposadr[0]
                f.d.qfrc_applied[v]=-force if force else -330*f.d.qpos[q]-.22*f.d.qvel[v]
            f.step()
        tracks.append(f.depth().copy())
        known.append([[px,py,float(value)] for (px,py),value in zip(xy,loads) if value])
folder=HERE/('merged' if merged else 'punches')
image(tracks,folder,711)
(folder/'private.json').write_text(json.dumps(known))
