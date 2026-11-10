import json,hashlib,sys
from pathlib import Path
import numpy as np
O=Path(sys.argv[1]);p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());l0=np.array(p['l0']);fmax=np.array(p['fmax']);checks=[]
for r in s['records']:
 z=np.load(O/(r['tag']+'.npz'));a=z['rows'];cmd=z['commands'];sp=np.unpackbits(z['spikes_packed'],axis=1)[:,:12000];rates=np.zeros(12000);n=round(1/r['dt_ms']);decay=np.exp(-r['dt_ms']/p['readout_tau_ms']);targets=[]
 for k in range(250):
  t=np.clip([rates[p['flex']].mean()/100,rates[p['ext']].mean()/100],0,1)
  if r['condition']=='zero_output':t[:]=0
  targets.append(t)
  for step in range(k*n,(k+1)*n):rates=decay*rates+(1-decay)*sp[step]/(r['dt_ms']/1000)
 assert np.array_equal(targets,cmd[:,1:3]);assert int(sp.sum())==r['spikes']
 repeated=np.repeat(targets,80,axis=0);act=np.zeros(2);acts=[]
 for t in repeated:
  tau=np.where(t>act,.01,.03);act=t+(act-t)*np.exp(-.0000125/tau);acts.append(act.copy())
 assert np.array_equal(acts,a[:,3:5])
 if r['arm']=='direct_tendon':force=fmax*repeated
 else:
  lam=a[:,5:7]/l0;v=a[:,7:9]/(10*l0);fl=np.exp(-((lam-1)/.25)**2);fv=np.ones_like(v);short=v<0;fv[short]=np.maximum(0,1+v[short])/(1-4*v[short]);fv[~short]=1+.5*v[~short]/(v[~short]+.25)
  force=np.clip(fmax*(a[:,3:5]*fl*fv+.05*(np.maximum(lam-1,0)/.1)**2+.02*np.maximum(v,0)),0,fmax)
 err=float(abs(force-a[:,9:11]).max());assert err<1e-12
 within=bool((a[:,1]>=.4789).all() and (a[:,1]<=2.502).all());assert within==r['within_range'];assert abs(a[-1,1]-1.2246235578147273-r['final_shift'])<1e-12
 checks.append(dict(tag=r['tag'],force_error=err,within_range=within))
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['hashes'].items())
comparisons=[]
for arm in p['arms']:
 for pulse in p['pulse_ms']:
  rr=[r for r in s['records'] if r['arm']==arm and r['pulse_ms']==pulse and r['condition']=='normal'];comparisons.append(dict(arm=arm,pulse_ms=pulse,normal_off_shifts=[r['shift_at_off'] for r in rr],normal_final_shifts=[r['final_shift'] for r in rr],all_in_range=all(r['within_range'] for r in rr)))
report=dict(importance='核心必需',verified=True,runs=len(checks),max_force_error=max(r['force_error'] for r in checks),checks=checks,comparisons=comparisons,inputs_unchanged=True)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(verified=True,runs=len(checks),comparisons=comparisons)))
