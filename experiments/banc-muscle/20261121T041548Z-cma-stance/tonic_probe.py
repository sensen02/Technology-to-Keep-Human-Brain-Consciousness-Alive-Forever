"""Isolate an ideal fixed-command body control for interpreting trainable readouts."""
from bench import Bench, HERE, json, mj, np, support
b=Bench()
d=b.handoff
initial=d.qpos.copy()
first_gate=first_nonfoot=None
for tick in range(2500):
    d.ctrl[:]=b.target
    mj.mj_step(b.model,d)
    mj.mj_forward(b.model,d)
    z=b.obs.measure(d)
    depth=min((c.dist for c in d.contact if b.obs.floor in c.geom),default=0.)
    if first_gate is None and not support(z,depth,b.height):
        first_gate=(tick+1)*.1
    if first_nonfoot is None and z['ground_load_w'][-1]>0:
        first_nonfoot=(tick+1)*.1
result=dict(first_gate_ms=first_gate,first_nonfoot_ms=first_nonfoot,final_root_delta_mm=(d.qpos[:3]-initial[:3]).tolist())
(HERE/'FIXED_TONIC.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
