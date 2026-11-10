import json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/pulse_timing')
p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());verified=[]
for r in s['records']:
 z=np.load(O/(r['tag']+'.npz'));sp=np.unpackbits(z['packed'],axis=1)[:,:12000];counts=sp.sum(0);dt=r['dt_ms'];n=round(1/dt);rates=np.zeros(12000);decay=np.exp(-dt/5);u=[]
 for k in range(len(sp)):
  if k%n==0:u.append(float(np.clip((rates[p['flex']].mean()-rates[p['ext']].mean())/100,-1,1)))
  rates=decay*rates+(1-decay)*sp[k]/(dt/1000)
 assert np.array_equal(u,z['commands'][:,1]);shift=0
 for val in u:shift+=.002*val
 assert shift==r['target_shift_rad'];assert counts[p['flex']].tolist()==r['flex_counts'];assert counts[p['ext']].tolist()==r['ext_counts']
 for cell,times in r['motor_spike_times_ms'].items():assert ((np.flatnonzero(sp[:,int(cell)])+1)*dt).tolist()==times
 verified.append(r['tag'])
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['hashes'].items())
summary={str(d):[r['target_shift_rad'] for r in s['records'] if r['pulse_ms']==d] for d in p['pulse_ms']}
report=dict(importance='核心必需',passed=True,verified_runs=len(verified),signed_target_by_duration=summary,sign_consistent={k:len(set(np.sign(v)))==1 for k,v in summary.items()},inputs_unchanged=True)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
