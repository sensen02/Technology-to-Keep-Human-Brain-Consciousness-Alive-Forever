import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
ndt=float(sys.argv[1]);O=ROOT/('joint_pulse_'+str(ndt));s=json.loads((O/'SUMMARY.json').read_text());p=json.loads((O/'PREREGISTRATION.json').read_text())
q0=1.2246235578147273;lim=10.550299302726808;checks=[]
for r in s['runs']:
 z=np.load(O/(r['tag']+'.npz'));a=z['rows'];c=z['commands'];sp=np.unpackbits(z['spikes_packed'],axis=1)[:,:12000];rates=np.zeros(12000);queue=[];out=[];n=round(.001/ndt);decay=np.exp(-ndt/.005)
 for k in range(250):
  raw=np.clip((rates[z['flex']].mean()-rates[z['ext']].mean())/100,-1,1);queue.append(raw)
  u=(queue[k-20] if k>=20 else 0) if r['condition']=='delayed_output' else raw
  out.append(0 if r['condition']=='zero_output' else u)
  for t in range(n*k,n*(k+1)):rates=decay*rates+(1-decay)*sp[t]/ndt
 assert np.array_equal(out,c[:,1]);assert int(sp.sum())==r['spikes'];assert a.shape==(20000,8)
 ref=q0;refs=[]
 for u in out:ref=np.clip(ref+.002*u,.4789,2.502);refs.extend([ref]*80)
 assert np.array_equal(refs,a[:,4])
 qprev=np.r_[q0,a[:-1,1]];vprev=np.r_[0,a[:-1,2]]
 if r['arm']=='torque':raw=lim*np.repeat(out,80)
 elif r['arm']=='incremental':raw=lim/.2*(a[:,4]-qprev)-lim*.005*vprev
 else:
  act=np.zeros(2);saved=[]
  for u in np.repeat(out,80):
   target=np.array([max(u,0),max(-u,0)]);act=target+(act-target)*np.exp(-.0000125/.01);saved.append(act.copy())
  assert np.array_equal(saved,a[:,5:7]);raw=lim*(a[:,5]-a[:,6])+.1*(lim/.2)*(q0-qprev)-lim*.005*vprev
 err=float(abs(np.clip(raw,-lim,lim)-a[:,3]).max());assert err<1e-12
 tail=a[-4000:];hold=bool((a[:,1]>=.4789).all() and (a[:,1]<=2.502).all() and np.isfinite(a).all() and r['warnings']==0 and abs(c[-50:,1]).max()<1e-3 and np.ptp(tail[:,4])<1e-4 and abs(tail[:,1]-tail[:,4]).max()<1e-3)
 assert hold==r['holds_reference']
 checks.append(dict(tag=r['tag'],command_error=err,hold=hold,last_spike_ms=float((np.flatnonzero(sp.any(1))[-1]+1)*ndt*1000) if sp.any() else None,tail_motor_spikes=int(sp[round(.2/ndt):,np.r_[z['flex'],z['ext']]].sum())))
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['hashes'].items())
report=dict(importance='核心必需',verified=True,runs=len(checks),checks=checks,source_inputs_unchanged=True)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
