import json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/sensory_transfer');p=json.loads((O/'PREREGISTRATION.json').read_text());s=json.loads((O/'SUMMARY.json').read_text());comparisons=[]
for r in s['records']:
 sp=np.unpackbits(np.load(O/(r['tag']+'.npz'))['packed'],axis=1)[:,:12000];start=round(50/r['dt_ms']);counts=sp[start:].sum(0)
 assert counts[p['flex']].tolist()==r['flex_counts'] and counts[p['ext']].tolist()==r['ext_counts'];assert int(counts.sum())==r['network_spikes'];assert int(counts[p['channel_ids'][r['channel']]].sum())==r['sensory_spikes']
 baseline=next(b for b in s['records'] if b['channel']==r['channel'] and b['dt_ms']==r['dt_ms'] and b['base_nA']==r['base_nA'] and b['cut']==r['cut'] and b['delta_nA']==0)
 ref=np.unpackbits(np.load(O/(baseline['tag']+'.npz'))['packed'],axis=1)[:,:12000];assert np.array_equal(sp[:start],ref[:start]);cells=p['flex']+p['ext'];a=sp[:,cells].astype(float);b=ref[:,cells].astype(float)
 # Reconstruct two population targets independently, at command boundaries.
 rate=np.zeros(7);rr=np.zeros(7);err=[];n=round(1/r['dt_ms']);decay=np.exp(-r['dt_ms']/5)
 for k in range(len(sp)):
  if k%n==0:
   target=np.clip([rate[:5].mean()/100,rate[5:].mean()/100],0,1);tref=np.clip([rr[:5].mean()/100,rr[5:].mean()/100],0,1);err.append(target-tref)
  rate=decay*rate+(1-decay)*a[k]/(r['dt_ms']/1000);rr=decay*rr+(1-decay)*b[k]/(r['dt_ms']/1000)
 comparisons.append(dict(tag=r['tag'],motor_spike_bit_changes=int(np.count_nonzero(a!=b)),max_target_change=float(abs(np.array(err)).max()),integrated_target_difference_s=(np.array(err).sum(0)*.001).tolist()))
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['hashes'].items())
report=dict(importance='核心必需',verified=True,runs=len(comparisons),pre_input_pairs_identical=True,inputs_unchanged=True,comparisons=comparisons)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
