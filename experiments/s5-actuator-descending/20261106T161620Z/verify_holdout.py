import subprocess,sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'holdout/20261106T200439Z';p=json.loads((O/'PREREGISTRATION.json').read_text());h=p['holdout'];valid=True;rows=[];ratios=[]
for vel in h['velocity_rad_s']:
 folder=O/f'v{float(vel)}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_interval_run.py')),str(folder)],check=True);s=json.loads((folder/'SUMMARY.json').read_text());valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in s['records'])
for dt in h['dt_ms']:
 for vel in [v for v in h['velocity_rad_s'] if v]:
  metrics={}
  for cond in h['conditions']:
   name=f'muscle_tendon_{cond}_p250_dt{dt}.npz';a=np.load(O/f'v{float(vel)}'/name)['rows'];b=np.load(O/'v0.0'/name)['rows'];assert np.array_equal(a[:,0],b[:,0]);metrics[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
  ratio=metrics['normal']/metrics['sensory_cut'];ratios.append(ratio);rows.append(dict(dt_ms=dt,velocity=vel,iae=metrics,ratio=ratio,passed=ratio<1))
repeated=[]
for cond in h['conditions']:
 name=f'muscle_tendon_{cond}_p250_dt0.015625.npz';a=np.load(R/'operating_point/20261106T195609Z/b0.03_v0.0'/name);b=np.load(O/'v0.0'/name);assert all(np.array_equal(a[k],b[k]) for k in ['rows','commands','spikes_packed']);repeated.append(dict(condition=cond,all_saved_arrays_bitwise_identical=True))
f=Path(p['frozen_candidate_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==p['frozen_candidate_sha256'];frozen=json.loads(f.read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in frozen['hashes'].items());l=p['change_ledger'];assert all(hashlib.sha256(Path(l[k+'_file']).read_bytes()).hexdigest()==l[k+'_sha256'] for k in ['old','new'])
report=dict(importance='核心必需',verified=True,physical_runs=20,valid=valid,ratios=ratios,holdout_gate_passed=bool(valid and all(r<1 for r in ratios)),rows=rows,repeated_baselines=repeated,frozen_candidate_unchanged=True)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
