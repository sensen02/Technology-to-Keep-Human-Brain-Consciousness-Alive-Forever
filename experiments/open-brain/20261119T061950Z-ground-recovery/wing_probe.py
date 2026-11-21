"""Probe whether native limb and wing actuators can generate righting leverage."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import numpy as np
from recover import Integrator,start
from fixture import HERE,Plant,mj

p=Plant()
r=Integrator(p)
pool=ThreadPoolExecutor(max_workers=8)
positions=p.model.jnt_qposadr[p.wing_joint]
print('wings',p.base[positions].tolist(),p.model.jnt_range[p.wing_joint].tolist(),flush=True)
groups=([0],[1],[2],[3],[4],[5],[0,3],[1,4],[2,5],[0,1],[3,4],[1,2],[4,5])
for case in ['left_side','back','orchard_fall','right_side']:
    state,up,_=start(p,case)
    neutral=p.ctrl_neutral
    opts=[]
    for group,x,dwell in product(groups,(-.11,-.038,.038,.11),(1,3)):
        a=neutral.copy()
        a[p.wing_act[np.asarray(group)]]=x
        opts.append((a,dwell,(group,x,dwell)))
    gen=np.random.default_rng(890)
    for j in range(64):
        a=neutral.copy()
        a[p.wing_act]=gen.uniform(-.18,.18,size=6)
        a[p.act]=np.clip(gen.uniform(-.9,.9,size=p.act.shape),p.min[p.act],p.max[p.act])
        opts.append((a,int(gen.choice([1,2,3])),('rnd',j)))
    def eval_action(opt):
        a,d,tag=opt
        x=r.run(state,[a]*d+[neutral]*12,seconds=.008)
        if x is None:
            return None
        z=x[0]
        return float(z[0]*3.5+z[1]*7+z[4]*.2-z[5]*.034-z[3]*.003+z[6]*.3),tag,z.tolist(),a.tolist(),d
    got=sorted((x for x in pool.map(eval_action,opts) if x is not None),key=lambda x:x[0],reverse=True)
    (HERE/f'WING_{case}.json').write_text(json.dumps(dict(initial=up,top=got[:15]),indent=2)+'\n')
    for score,tag,y,_,_ in got[:5]:
        print(case,tag,np.round(y[:7],4),round(score,4),flush=True)
pool.shutdown()
p.env.close()
