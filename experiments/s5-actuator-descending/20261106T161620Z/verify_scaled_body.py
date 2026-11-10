import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/scaled_body/20261106T193236Z');p=json.loads((O/'PREREGISTRATION.json').read_text());rows=[];valid=True
for vel in [0.,-.1,.1]:
 folder=O/f'v{vel}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_sensory_bias_run.py')),str(folder)],check=True);s=json.loads((folder/'SUMMARY.json').read_text());valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in s['records'])
ratios=[];zero_shifts=[]
for dt in p['dt_ms']:
 z=np.load(O/'v0.0'/f'muscle_tendon_zero_output_p250_dt{dt}.npz');assert not z['commands'][:,1:3].any();zero_shifts.append(float(abs(z['rows'][:,1]-1.2246235578147273).max()))
 for vel in [-.1,.1]:
  metrics={}
  for cond in ['normal','sensory_cut','zero_output']:
   name=f'muscle_tendon_{cond}_p250_dt{dt}.npz';a=np.load(O/f'v{vel}'/name)['rows'];b=np.load(O/'v0.0'/name)['rows'];assert np.array_equal(a[:,0],b[:,0]);metrics[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
  ratio=metrics['normal']/metrics['sensory_cut'];ratios.append(ratio);rows.append(dict(dt_ms=dt,velocity=vel,iae=metrics,normal_over_cut=ratio))
consistency=[]
for vel in [0.,-.1,.1]:
 for cond in p['conditions']:
  a=np.load(O/f'v{vel}'/f'muscle_tendon_{cond}_p250_dt{p["dt_ms"][0]}.npz');b=np.load(O/f'v{vel}'/f'muscle_tendon_{cond}_p250_dt{p["dt_ms"][1]}.npz');consistency.append(dict(velocity=vel,condition=cond,max_angle_difference_rad=float(abs(a['rows'][:,1]-b['rows'][:,1]).max()),command_mae=float(abs(a['commands'][:,1:3]-b['commands'][:,1:3]).mean())))
l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new']);assert max(zero_shifts)<1e-8
report=dict(importance='核心必需',verified=True,physical_runs=18,valid=valid,ratios=ratios,recovery_gate_passed=bool(valid and all(r<1 for r in ratios)),zero_output_undisturbed_max_shifts_rad=zero_shifts,rows=rows,dt_consistency_diagnostic=consistency)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
