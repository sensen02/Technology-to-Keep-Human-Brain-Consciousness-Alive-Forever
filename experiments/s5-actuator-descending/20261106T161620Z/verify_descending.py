import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
P=ROOT/'descending_probe';pr=json.loads((P/'PREREGISTRATION.json').read_text());s=json.loads((P/'SUMMARY.json').read_text());audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());checks={}
for r in s['records']:
 z=np.load(P/(r['tag']+'.npz'));dt=r['dt_ms'];sp=np.unpackbits(z['packed'],axis=1)[:,:12000];counts=sp[round(50/dt):].sum(0)
 assert counts.sum()==r['total_spikes']
 for key,label in [('1','flex'),('-1','ext')]:
  assert np.array_equal(counts[audit['targets'][key]],r[label+'_counts'])
  assert abs((counts[audit['targets'][key]]/.15).mean()-r[label+'_Hz'])<1e-12
 assert not sp[:round(50/dt)].any()
checks['probe_16_spike_records_verified']=True
checks['probe_equal_injected_count']=len(pr['candidate'])==len(pr['matched'])==19 and not set(pr['candidate'])&set(pr['matched'])
for dt in [.5,.0625]:
 for name in ['matched','candidate_cut']:
  rr=[r for r in s['records'] if r['dt_ms']==dt and r['arm']==name]
  checks[f'no_motor_response_{name}_{dt}']=all(r['flex_Hz']==r['ext_Hz']==0 for r in rr)
checks['probe_inputs_unchanged']=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in pr['hashes'].items())
(P/'VERIFICATION.json').write_text(json.dumps(dict(importance='核心必需',checks=checks,passed=all(checks.values())),indent=2)+'\n')
if len(sys.argv)>1:
 ndt=float(sys.argv[1]);O=ROOT/('joint_descending_'+str(ndt));s=json.loads((O/'SUMMARY.json').read_text());pr=json.loads((O/'PREREGISTRATION.json').read_text());q0=1.2246235578147273;lim=10.550299302726808;errs=[]
 for r in s['runs']:
  z=np.load(O/(r['tag']+'.npz'));a=z['rows'];cmd=z['commands'];sp=np.unpackbits(z['spikes_packed'],axis=1)[:,:12000];rates=np.zeros(12000);decoded=[];decay=np.exp(-ndt/.005);n=round(.001/ndt)
  for k in range(len(cmd)):
   u=np.clip((rates[z['flex']].mean()-rates[z['ext']].mean())/100,-1,1);decoded.append(0 if r['condition']=='zero_output' else u)
   for t in range(n*k,n*(k+1)):rates=decay*rates+(1-decay)*sp[t]/ndt
  assert np.array_equal(decoded,cmd[:,1]);assert int(sp.sum())==r['spikes'];assert abs(np.abs(a[:,1]-q0).sum()*.0000125-r['angle_iae'])<1e-15
  qprev=np.r_[q0,a[:-1,1]];vprev=np.r_[r['velocity'],a[:-1,2]]
  if r['arm']=='torque':raw=lim*np.repeat(cmd[:,1],80)
  elif r['arm']=='incremental':raw=lim/.2*(a[:,4]-qprev)-lim*.005*vprev
  else:raw=lim*(a[:,5]-a[:,6])+.1*(lim/.2)*(q0-qprev)-lim*.005*vprev
  errs.append(float(abs(np.clip(raw,-lim,lim)-a[:,3]).max()));assert errs[-1]<1e-12
 effects={arm:all(next(r for r in s['runs'] if r['arm']==arm and r['velocity']==v and r['condition']=='normal')['angle_iae']<next(r for r in s['runs'] if r['arm']==arm and r['velocity']==v and r['condition']==c)['angle_iae'] for v in [-.1,.1] for c in ['zero_output','sensory_cut']) and all(r['within_range'] and r['finite'] and r['warnings']==0 for r in s['runs'] if r['arm']==arm) for arm in ['torque','incremental','muscle']}
 assert effects==s['neural_contribution_screen'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in pr['hashes'].items())
 report=dict(importance='核心必需',passed=True,runs=len(errs),max_command_error=max(errs),neural_contribution_screen=effects,source_and_inputs_unchanged=True)
 (O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
else:print(json.dumps(checks))
