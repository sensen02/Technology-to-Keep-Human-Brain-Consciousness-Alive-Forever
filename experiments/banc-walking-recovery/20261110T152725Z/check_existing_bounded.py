"""Read-only real evidence smoke: one 25.6ms native chunk and matched prefix.
No native/neural final verdict is generated for legacy300 (no neural archive).
"""
import json
from pathlib import Path
import tempfile
import numpy as np
import final_native_audit as a

run=a.ROOT/'20261110T114148Z/EVAL_GRU_ROLE_NATIVE'
fast=a.ROOT/'20261110T125712Z/EVAL_ROLE_2000'
zero=a.ROOT/'20261110T111240Z/EVAL_ZERO_MATCHED_RETRY'
result=a.read_json(run/'RESULT.json')
entry=result['native']['chunks'][0]
a.require(a.sha(entry['path'])==entry['sha256'],'first native hash mismatch')
a.verify_identity(a.read_json(run/'IDENTITY.json'))
a.verify_identity(a.read_json(fast/'IDENTITY.json'))
a.verify_identity(a.read_json(zero/'IDENTITY.json'))
with np.load(entry['path'],allow_pickle=False) as z:
    end=float(a.bounded(z,'time_after_s')[-1])
    for key in z.files:a.array_header(z,key)
with tempfile.TemporaryDirectory(prefix='final-audit-real-slice-') as td:
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
        keep=a.bounded(z,'time')<=end+1e-10
        trace={key:a.bounded(z,key)[keep] for key in ('time','qpos','qvel','com','action','target','actualctrl')}
    path=Path(td)/'slice_trace.npz';np.savez_compressed(path,**trace)
    a.WINDOWS={'auxiliary':(0.,.01),'autonomous':(.01,end)}
    diagnostics=a.extra_native({'chunks':[entry],'native_samples':entry['rows']},.002494271478*9801.,path)
rows={}
with np.load(fast/'trace_1ms.npz',allow_pickle=False) as x,np.load(zero/'trace_1ms.npz',allow_pickle=False) as y:
    for key in sorted(set(x.files)&set(y.files)):
        rows[key]=bool(np.array_equal(a.bounded(x,key)[:5800],a.bounded(y,key)[:5800]))
a.require(all(rows.values()),'latest existing2000 and zero prefixes differ')
report={'status':'PASS','physical_steps':0,'historical_writes':0,
    'native_source':str(run),'native_chunk':entry,'bounded_native_diagnostics':diagnostics,
    'zero_control_prefix_comparison_source':str(fast),'zero_control':str(zero),
    'prefix_samples':5800,'prefix_arrays_exact':rows,
    'scope':'one native chunk and actual existing2000 prefix smoke; NOT a final candidate/full native/neural acceptance'}
path=Path(__file__).parent/'EXISTING_BOUNDED_CHECK.json'
with path.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'output':str(path),'status':report['status'],'native_rows':entry['rows'],'prefix_exact':all(rows.values()),'physical_steps':0}),flush=True)
