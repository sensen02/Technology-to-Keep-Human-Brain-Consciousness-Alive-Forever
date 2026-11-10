"""Read-only constructor/forward compatibility smoke; zero integration steps."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
import final_native_audit as a
sys.path.insert(0,str(a.RECORDER))
import physics
import student_wave as sw
from identity import verify_runtime
import torch
torch.set_num_threads(1)
core=physics.Core(adhesion=False)
run=a.ROOT/'20261110T125712Z/EVAL_ROLE_2000'
identity=a.read_json(run/'IDENTITY.json')
verify_runtime(identity,core)
mesh=core.plant.geom_type==7
a.require(core.plant.nmesh==71 and int(mesh.sum())==71 and np.all(core.plant.geom_rgba[mesh,3]==1),'full71opaque failure')
model,payload=sw.load_student(identity['config']['cli']['checkpoint'],core)
with np.load(run/'decisions_5ms.npz',allow_pickle=False) as z:
    x,v,actions=a.bounded(z,'X238'),a.bounded(z,'V255'),a.bounded(z,'action')
state=model.actor.initial_state(1)
with torch.no_grad():
    for i in range(2880):
        output=model.act(x[i],v[i],state=state,deterministic=True)
        state=output['state']
        if i>=960:a.require(np.array_equal(output['action'][0].numpy(),actions[i]),'actual existing2000 ANN action differs row '+str(i))
a.require(core.d.time==0.,'physical time changed')
report={'status':'PASS','physical_steps':0,'existing_run':str(run),'rows':2880,'actual_neural_rows':1920,
    'checkpoint':identity['config']['cli']['checkpoint'],'checkpoint_sha256':a.sha(identity['config']['cli']['checkpoint']),
    'all_recorded_actual_neural_actions_bitexact':True,'compiled_runtime_verified':True,'opaque_mesh_count':71,
    'initial_heading':core.initial_heading.tolist(),'scope':'existing failed2000 compatibility evidence; no final candidate verdict'}
path=Path(__file__).parent/'RUNTIME_FORWARD_CHECK.json'
with path.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'status':'PASS','output':str(path),'physical_steps':0,'actual_actions_bitexact':1920}),flush=True)
