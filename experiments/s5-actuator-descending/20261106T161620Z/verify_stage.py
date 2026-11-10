"""Independent raw-array and graph-path checks; no simulation."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A')
checks={}
p=json.loads((O/'descending/PATH_AUDIT.json').read_text()); w=[sparse.load_npz(A/f'ACTUAL_{k}.npz').tocsr() for k in ['We','Wi']]
checked=0
for r in p['records']:
 for sign,path in r['paths'].items():
  if path['distance']<0: continue
  nodes=path['local_indices'];assert len(nodes)-1==path['distance']<=4
  assert nodes[0]==r['local_index'] and nodes[-1] in p['targets'][sign]
  for pre,post,edge in zip(nodes[:-1],nodes[1:],path['edges']):
   assert float(w[0][post,pre])==edge['We'] and float(w[1][post,pre])==edge['Wi']
   assert sum(float(x[post,pre]) for x in w)>0
   checked+=1
checks['reported_path_edges_exist']=True
# Recompute <=4-step reachability with vector propagation, independent of BFS.
reach=[]
for target in p['targets'].values():
 mask=np.zeros(12000,dtype=bool);mask[target]=True
 for _ in range(4):mask |= ((w[0]+w[1]).T@mask.astype(float))>0
 reach.append(mask)
candidates=sorted(r['local_index'] for r in p['records'] if all(x[r['local_index']] for x in reach))
checks['candidate_set_independently_matches']=candidates==sorted(p['ranked_candidate_indices'])
# Command clock is 1 ms independent of neural substep.
refs=[]
for ndt in [.0005,.0000625]:
 q=1.2246235578147273
 for t in range(round(.1/ndt)):
  if t%round(.001/ndt)==0:q+=.001*2*(.002 if t*ndt<.03 else 0)
 refs.append(q)
checks['increment_clock_invariant']=refs[0]==refs[1] and abs(refs[0]-(1.2246235578147273+.00012))<1e-12
C=O/'calibration_v3'
results=json.loads((C/'RESULTS.json').read_text())
for arm in ['torque','incremental','muscle']:
 xs=[np.load(C/(arm+s+'.npz'))['rows'] for s in ['_neg','_zero','_pos']]
 checks[arm+'_pulse_direction']=bool(xs[0][2399,1]<xs[1][2399,1]<xs[2][2399,1])
 checks[arm+'_zero_drift']=bool(abs(xs[1][-1,1]-1.2246235578147273)<1e-6)
 checks[arm+'_within_limits']=bool(all((x[:,1]>=.4789).all() and (x[:,1]<=2.502).all() for x in xs))
 checks[arm+'_finite']=bool(all(np.isfinite(x).all() for x in xs))
 if arm=='incremental':checks['incremental_holds_target']=all(abs(x[-1,1]-x[-1,4])<1e-5 for x in xs)
 if arm=='muscle':checks['muscle_recovers']=all(abs(x[-1,1]-1.2246235578147273)<abs(x[2399,1]-1.2246235578147273) for x in [xs[0],xs[2]])
checks['declared_checks_match']=all(checks[k]==v for k,v in results['checks'].items() if k in checks)
# Record verification failures rather than hiding behind a successful exit.
report=dict(importance='核心必需',checks=checks,passed=all(checks.values()),path_edges_checked=checked,neural_dt_clock_refs=refs,scope='isolated engineering calibration and topology, not neural support')
(O/'STAGE_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
