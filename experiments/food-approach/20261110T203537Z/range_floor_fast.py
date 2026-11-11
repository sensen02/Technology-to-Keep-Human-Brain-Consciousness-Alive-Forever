"""Read-only reachable-action envelope lower bound; no optimization or physics."""
import json
import torch
from fast_motor import HERE,FastMotor,load_base,sha256
from fast_data import prepare

torch.set_num_threads(1);model=FastMotor(load_base());episodes,_=prepare(model)
results={}
for e in episodes:
    with torch.no_grad():
        y=e['y'][960:];lo=torch.tanh(e['mean'][960:]-e['gate'][960:]);hi=torch.tanh(e['mean'][960:]+e['gate'][960:])
        error=torch.relu(lo-y)+torch.relu(y-hi)
        results[e['name']]=dict(outside_fraction=float((error>0).double().mean()),irreducible_bounded_action_MSE=float(error.square().mean()),max_unavoidable_action_error=float(error.max()),gate_max=float(e['gate'][960:].max()))
report=dict(kind='declared latent residualbound1 reachable-action envelope',cases=results,physical_steps=0,optimizer_updates=0,
            sources={str(HERE/n):sha256(HERE/n) for n in ('range_floor_fast.py','fast_motor.py','fast_data.py')},
            interpretation='algebraic fitting lower bound, not proof of physical success/failure or body infeasibility')
with (HERE/'FAST_RANGE_FLOOR.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report,indent=2))
