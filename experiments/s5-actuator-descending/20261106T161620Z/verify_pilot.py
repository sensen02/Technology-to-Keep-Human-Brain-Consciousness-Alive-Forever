"""Independent checks of spike chain, physical errors, and ablation equality."""
import json
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/joint_pilot')
s=json.loads((O/'SUMMARY.json').read_text());checks={};results=[];q0=1.2246235578147273;lim=10.550299302726808
for r in s['runs']:
 z=np.load(O/(r['tag']+'.npz'));a=z['rows'];cmd=z['commands'];sp=np.unpackbits(z['spikes_packed'],axis=1)[:,:12000]
 rates=np.zeros(12000);decoded=[];decay=np.exp(-.0005/.005)
 for k in range(len(cmd)):
  u=np.clip((rates[z['flex']].mean()-rates[z['ext']].mean())/100,-1,1)
  decoded.append(0 if r['condition']=='zero_output' else u)
  for t in range(2*k,2*k+2):rates=decay*rates+(1-decay)*sp[t]/.0005
 assert np.array_equal(np.array(decoded),cmd[:,1])
 assert int(sp.sum())==r['spikes']
 iae=float(np.abs(a[:,1]-q0).sum()*.0000125);assert abs(iae-r['angle_iae'])<1e-15
 qprev=np.r_[q0,a[:-1,1]];vprev=np.r_[r['velocity'],a[:-1,2]]
 if r['arm']=='torque':raw=lim*np.repeat(cmd[:,1],80)
 elif r['arm']=='incremental':raw=lim/.2*(a[:,4]-qprev)-lim*.005*vprev
 else:raw=lim*(a[:,5]-a[:,6])+.1*(lim/.2)*(q0-qprev)-lim*.005*vprev
 err=float(np.max(np.abs(np.clip(raw,-lim,lim)-a[:,3])))
 assert err<1e-12
 results.append(dict(tag=r['tag'],command_error=err,spikes=int(sp.sum()),max_input_nA=float(cmd[:,4].max()),iae=iae))
for arm in ['torque','incremental','muscle']:
 for direction in ['neg','pos']:
  normal=np.load(O/f'{arm}_normal_{direction}.npz')['rows']
  for condition in ['zero_output','sensory_cut']:
   checks[f'{arm}_{direction}_{condition}_same_physics']=bool(np.array_equal(normal,np.load(O/f'{arm}_{condition}_{direction}.npz')['rows']))
checks['all_spike_and_command_chains_verified']=True
checks['all_runs_zero_spikes']=all(r['spikes']==0 for r in results)
report=dict(importance='核心必需',passed=all(checks.values()),checks=checks,results=results,conclusion='No neural contribution: normal and both ablations identical for each actuator. Holding/recovery is mechanical/servo, not neural.')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=report['passed'],runs=len(results),max_command_error=max(r['command_error'] for r in results),max_input_nA=max(r['max_input_nA'] for r in results))))
