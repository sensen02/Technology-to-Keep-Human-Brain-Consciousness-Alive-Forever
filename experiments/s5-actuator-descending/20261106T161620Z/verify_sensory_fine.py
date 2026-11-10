import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'claw_direction/fine/20261106T191739Z';p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());cells=p['flex']+p['ext'];data={};events={};drives={};checks=[]
for r in s['records']:
 dt=r['dt_ms'];delta=r['delta_nA'];sp=np.unpackbits(np.load(O/(r['tag']+'.npz'))['packed'],axis=1)[:,:12000];counts=sp[round(50/dt):].sum(0)
 assert counts[p['flex']].tolist()==r['flex_counts'];assert counts[p['ext']].tolist()==r['ext_counts'];assert int(counts.sum())==r['network_spikes'];assert int(counts[p['all_background_cells']].sum())==r['sensory_spikes']
 motor=sp[:,cells];rate=np.zeros(len(cells));cmd=[];decay=np.exp(-dt/5);n=round(1/dt)
 for k,x in enumerate(motor):
  if k%n==0:cmd.append(np.clip([rate[:5].mean()/100,rate[5:].mean()/100],0,1))
  rate=decay*rate+(1-decay)*x/(dt/1000)
 drives[dt,delta]=np.array(cmd);data[dt,delta]=sp;events[dt,delta]=[(np.flatnonzero(motor[:,j])+1)*dt for j in range(7)]
for dt in p['dt_ms']:
 for delta in p['deltas_nA']:assert np.array_equal(data[dt,delta][:round(50/dt)],data[dt,0][:round(50/dt)])
a,b=p['dt_ms']
for delta in p['deltas_nA']:
 ea,eb=events[a,delta],events[b,delta];equal=all(len(x)==len(y) for x,y in zip(ea,eb));timing=max([float(abs(x-y).max()) for x,y in zip(ea,eb) if len(x)] or [0]) if equal else None
 mae=float(abs(drives[a,delta]-drives[b,delta]).mean());response_a=(drives[a,delta]-drives[a,0]).sum(0)*.001;response_b=(drives[b,delta]-drives[b,0]).sum(0)*.001;response_err=float(abs(response_a-response_b).max())
 checks.append(dict(delta_nA=delta,target_event_counts_a=[len(x) for x in ea],target_event_counts_b=[len(x) for x in eb],counts_equal=equal,max_ordinal_event_error_ms=timing,drive_mean_absolute_difference=mae,response_integral_a_s=response_a.tolist(),response_integral_b_s=response_b.tolist(),response_error_s=response_err,pass_screen=bool(equal and timing<=1 and mae<=.01 and response_err<=.001)))
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['hashes'].items())
report=dict(importance='核心必需',verified=True,neural_runs=6,physics_runs=0,dt_pair_ms=p['dt_ms'],checks=checks,convergence_screen_passed=all(c['pass_screen'] for c in checks),scope='finite engineering local-input screen, not global numerical or physical convergence')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
