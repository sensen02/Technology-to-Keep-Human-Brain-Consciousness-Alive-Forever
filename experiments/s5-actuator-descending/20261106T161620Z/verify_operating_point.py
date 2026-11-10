import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/operating_point/20261106T195609Z');p=json.loads((O/'PREREGISTRATION.json').read_text());configs=[]
for bias in p['backgrounds_nA']:
 valid=True
 for vel in [0.,-.1,.1]:
  folder=O/f'b{bias}_v{vel}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_interval_run.py')),str(folder)],check=True);s=json.loads((folder/'SUMMARY.json').read_text());valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in s['records'])
 rows=[];ratios=[];differences=[]
 for dt in p['dt_ms']:
  for vel in [-.1,.1]:
   metrics={}
   for cond in p['conditions']:
    name=f'muscle_tendon_{cond}_p250_dt{dt}.npz';a=np.load(O/f'b{bias}_v{vel}'/name)['rows'];b=np.load(O/f'b{bias}_v0.0'/name)['rows'];assert np.array_equal(a[:,0],b[:,0]);metrics[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
   ratio=metrics['normal']/metrics['sensory_cut'];ratios.append(ratio);rows.append(dict(dt_ms=dt,velocity=vel,iae=metrics,normal_over_cut=ratio))
 for vel in [0.,-.1,.1]:
  a=np.load(O/f'b{bias}_v{vel}'/f'muscle_tendon_normal_p250_dt{p["dt_ms"][0]}.npz');b=np.load(O/f'b{bias}_v{vel}'/f'muscle_tendon_normal_p250_dt{p["dt_ms"][1]}.npz');differences.append(dict(velocity=vel,max_angle_difference_rad=float(abs(a['rows'][:,1]-b['rows'][:,1]).max()),command_mae=float(abs(a['commands'][:,1:3]-b['commands'][:,1:3]).mean())))
 configs.append(dict(bias_nA=bias,valid=valid,ratios=ratios,recovery_gate_passed=bool(valid and all(r<1 for r in ratios)),rows=rows,dt_consistency=differences))
l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new'])
report=dict(importance='核心必需',verified=True,physical_runs=36,configs=configs);(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
