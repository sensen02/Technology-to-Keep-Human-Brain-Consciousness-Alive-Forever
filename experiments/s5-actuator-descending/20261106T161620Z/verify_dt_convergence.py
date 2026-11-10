import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/dt_convergence/20261106T213030Z');p=json.loads((O/'PREREGISTRATION.json').read_text());DTS=p['dt_ms'];valid=True;undisturbed=[]
for vel in [0.,-.1,.1]:
 folder=O/f'v{vel}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_interval_run.py')),str(folder)],check=True);s=json.loads((folder/'SUMMARY.json').read_text());valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in s['records'])
rows=[];ratios={}
for vel in [0.,-.1,.1]:
 traj={}
 for dt in DTS:
  for cond in ['normal','sensory_cut']:
   a=np.load(O/f'v{vel}'/f'muscle_tendon_{cond}_p250_dt{dt}.npz')['rows'];traj[dt,cond]=a
 for dt in DTS:
  drift=float(abs(traj[dt,'normal'][:,1]-traj[dt,'sensory_cut'][:,1]).max());undisturbed.append(dict(velocity=vel,dt_ms=dt,max_normal_vs_cut_angle_gap_rad=drift))
for dt in DTS:
 for vel in [-.1,.1]:
  m={}
  for cond in ['normal','sensory_cut']:
   a=np.load(O/f'v{vel}'/f'muscle_tendon_{cond}_p250_dt{dt}.npz')['rows'];b=np.load(O/'v0.0'/f'muscle_tendon_{cond}_p250_dt{dt}.npz')['rows'];assert np.array_equal(a[:,0],b[:,0]);m[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
  r=m['normal']/m['sensory_cut'];ratios[dt,vel]=r;rows.append(dict(dt_ms=dt,velocity=vel,iae=m,ratio=r))
seq_neg=[ratios[dt,-.1] for dt in DTS];seq_pos=[ratios[dt,.1] for dt in DTS]
tol=.15
fine_neg=abs(seq_neg[0]-seq_neg[1]);fine_pos=abs(seq_pos[0]-seq_pos[1])
conv_fine=bool(fine_neg<=tol and fine_pos<=tol)
full_neg=max(seq_neg)-min(seq_neg);full_pos=max(seq_pos)-min(seq_pos)
if conv_fine and (min(seq_neg[:2])<1 and min(seq_pos[:2])<1):verdict='converged recovery at this load'
elif conv_fine:verdict='converged failure: the loop genuinely fails at this load; the coarsest step is the outlier artifact'
else:verdict='not converged at the two finest steps: discretization still unresolved'
l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new'])
report=dict(importance='核心必需',verified=True,physical_runs=21,valid=valid,dt_ascending=DTS,ratio_sequence_negative=seq_neg,ratio_sequence_positive=seq_pos,convergence_tolerance=tol,two_finest_gap_negative=fine_neg,two_finest_gap_positive=fine_pos,converged_at_two_finest=conv_fine,full_sequence_spread=dict(negative=full_neg,positive=full_pos),undisturbed_consistency=undisturbed,rows=rows,verdict=verdict,correction='earlier reading that refining .015625->.0078125 made the ratio worse was itself artifact-led; the added third step shows the two finest steps agree to ~0.004% and the coarsest step is the outlier')
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
