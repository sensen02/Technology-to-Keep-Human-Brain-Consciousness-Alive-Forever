"""Online frozen brain with only received electrode AN events at its body afferent boundary."""
import sys,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'20261119T001532Z-locomotion-choice'))
import live_brain as live
# Use 5 ms exchanges to avoid seeing events from the next body frame.
live.BLOCK=50
cfg=json.loads(sys.stdin.readline())
model=live.Brain(dict(seed=29,ticks=cfg['ticks'],adapter=False,cut=False))
boundary=type(model.neck.an_boundary)(2)
original=model.neck.update
rates=model.head.rates
intent=(0.,0.)
model.head.rates=lambda pace,turn,odor:rates(*intent,odor) if walking else (0.,0.)
print(json.dumps(dict(roots=model.neck.root_names,an=[row['banc_root'] for row in model.inputs.an])),flush=True)
for line in sys.stdin:
    data=json.loads(line);intent=tuple(data['intent'])
    walking=data['circulating']<.295 and (intent[0]>.008 or abs(intent[1])>.015)
    start=model.tick;source=[];brain_input=[]
    received=np.asarray(data['arrivals'],dtype=bool)
    def update(i,dn,front):
        source.append(dn.copy())
        output=original(i,dn,np.zeros(2))
        model.neck.an_inputs=boundary.step(received[i-start])
        brain_input.append(model.neck.an_inputs.copy())
        return output
    model.neck.update=update;model.ground_taste=0.
    packet=model.advance(dict(height_probability=0.,landing_eye=False,seen=walking,contact=data['contact'],center=0.,front=[0.,0.]))
    print(json.dumps(dict(source=np.asarray(source).astype(int).tolist(),encoded=np.asarray(brain_input).astype(int).tolist(),feed=packet['feed'])),flush=True)
