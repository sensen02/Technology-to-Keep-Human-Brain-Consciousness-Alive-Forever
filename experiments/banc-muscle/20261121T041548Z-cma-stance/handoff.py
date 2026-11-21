"""Measure one native-step foot-load sensitivity to the 90 individual muscle commands."""
from bench import Bench, HERE, copy, mj, np
import json
b=Bench()
b.cache=b.prepare(b.weights)
trace=b.cache[1]
entry=[]
for label,w in [('original',b.weights['motor_gain']),('fit',np.load(HERE/'training/warm_motor.npz')['motor_gain'])]:
    g=np.zeros_like(b.io.motor_gain);g[b.motor_rows,b.motor_cols]=w
    entry.append((label,np.clip(g@trace,0,1)))
entry.append(('tonic',b.target))
def first(cmd):
    d=copy.copy(b.handoff)
    d.ctrl[:]=cmd
    mj.mj_step(b.model,d);mj.mj_forward(b.model,d)
    return b.obs.measure(d)['ground_load_w'][:-1].sum()
result={}
for name,u in entry:
    y=first(u)
    grad=np.array([first(np.r_[u[:i],max(0,min(1,u[i]+(.0001 if u[i]<.9999 else -.0001))),u[i+1:]]) for i in range(len(u))])
    grad=(grad-y)/np.where(u<.9999,.0001,-.0001)
    result[name]=dict(foot=y,error=float(np.linalg.norm(u-b.target)),
                      slope=grad.tolist(),commands=u.tolist())
(HERE/'HANDOFF.json').write_text(json.dumps(result,indent=2)+'\n')
for k,r in result.items(): print(k,r['foot'],r['error'])
