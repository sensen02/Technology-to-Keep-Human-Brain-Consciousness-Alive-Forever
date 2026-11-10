import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/gravity_body/20261106T201513Z');p=json.loads((O/'PREREGISTRATION.json').read_text());configs=[]
for g in p['gravity_fractions']:
 valid=True
 for vel in [0.,-.1,.1]:
  folder=O/f'g{g}_v{vel}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_interval_run.py')),str(folder)],check=True);s=json.loads((folder/'SUMMARY.json').read_text());valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in s['records']);protocol=json.loads((folder/'PREREGISTRATION.json').read_text());assert np.array_equal(protocol['gravity'],np.array(p['full_gravity'])*g)
 ratios=[];rows=[];zero_rows=[];baselines=[]
 for dt in p['dt_ms']:
  for cond in ['normal','sensory_cut','zero_output']:
   z=np.load(O/f'g{g}_v0.0'/f'muscle_tendon_{cond}_p250_dt{dt}.npz');a=z['rows'];baselines.append(dict(dt_ms=dt,condition=cond,final_shift_rad=float(a[-1,1]-1.2246235578147273),max_abs_shift_rad=float(abs(a[:,1]-1.2246235578147273).max())))
   if cond=='zero_output':
    assert not z['commands'][:,1:3].any() and not a[:,3:5].any();zero_rows.append(a)
  for vel in [-.1,.1]:
   metrics={}
   for cond in p['conditions']:
    name=f'muscle_tendon_{cond}_p250_dt{dt}.npz';a=np.load(O/f'g{g}_v{vel}'/name)['rows'];b=np.load(O/f'g{g}_v0.0'/name)['rows'];assert np.array_equal(a[:,0],b[:,0]);metrics[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
   ratio=metrics['normal']/metrics['sensory_cut'];ratios.append(ratio);rows.append(dict(dt_ms=dt,velocity=vel,iae=metrics,ratio=ratio))
 assert np.array_equal(*zero_rows)
 configs.append(dict(gravity_fraction=g,valid=valid,ratios=ratios,recovery_gate_passed=bool(valid and all(r<1 for r in ratios)),rows=rows,baselines=baselines,zero_output_physics_identical_across_neural_dt=True))
l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new'])
report=dict(importance='核心必需',verified=True,physical_runs=42,configs=configs);(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
