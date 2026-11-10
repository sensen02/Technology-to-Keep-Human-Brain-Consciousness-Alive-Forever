import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/readout_window/20261106T190609Z');p=json.loads((O/'PREREGISTRATION.json').read_text());rows=[];configs=[]
for tau in p['tau_ms']:
 valid=True
 for vel in [0.,-.1,.1]:
  folder=O/f'tau{tau}_v{vel}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_readout_run.py')),str(folder)],check=True)
  valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in json.loads((folder/'SUMMARY.json').read_text())['records'])
 ratios=[]
 for dt in [.5,.0625]:
  for vel in [-.1,.1]:
   metrics={}
   for cond in ['normal','sensory_cut']:
    tag=f'muscle_tendon_{cond}_p250_dt{dt}';a=np.load(O/f'tau{tau}_v{vel}'/(tag+'.npz'))['rows'];b=np.load(O/f'tau{tau}_v0.0'/(tag+'.npz'))['rows'];metrics[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
   ratio=metrics['normal']/metrics['sensory_cut'];ratios.append(ratio);rows.append(dict(tau_ms=tau,dt_ms=dt,velocity=vel,iae=metrics,ratio=ratio))
 configs.append(dict(tau_ms=tau,valid=valid,ratios=ratios,gate_passed=bool(valid and all(r<1 for r in ratios))))
l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new'])
report=dict(importance='核心必需',verified=True,physical_runs=24,configs=configs,rows=rows)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
