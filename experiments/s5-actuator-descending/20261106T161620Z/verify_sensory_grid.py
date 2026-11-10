import sys,subprocess,json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/sensory_grid/20261106T185758Z');p=json.loads((O/'PREREGISTRATION.json').read_text());rows=[];configs=[]
for bias in p['background_nA']:
 for gain in p['dynamic_gain']:
  valid=True
  for vel in [0.,-.1,.1]:
   folder=O/f'b{bias}_g{gain}_v{vel}';subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_sensory_bias_run.py')),str(folder)],check=True)
   valid=valid and all(r['within_range'] and r['forces_bounded'] and r['finite'] and r['warnings']==0 for r in json.loads((folder/'SUMMARY.json').read_text())['records'])
  ratios=[]
  for dt in [.5,.0625]:
   for vel in [-.1,.1]:
    metrics={}
    for cond in ['normal','sensory_cut']:
     tag=f'muscle_tendon_{cond}_p250_dt{dt}';a=np.load(O/f'b{bias}_g{gain}_v{vel}'/(tag+'.npz'))['rows'];b=np.load(O/f'b{bias}_g{gain}_v0.0'/(tag+'.npz'))['rows']
     metrics[cond]=float(abs(a[:,1]-b[:,1]).sum()*.0000125)
    ratio=metrics['normal']/metrics['sensory_cut'];ratios.append(ratio);rows.append(dict(bias_nA=bias,gain=gain,dt_ms=dt,velocity=vel,iae=metrics,ratio=ratio))
  configs.append(dict(bias_nA=bias,gain=gain,range_force_finite_warning_checks=valid,ratios=ratios,worst_ratio=max(ratios),gate_passed=bool(valid and all(r<1 for r in ratios))))
ledger=p['change_ledger'];assert hashlib.sha256(Path(ledger['old_file']).read_bytes()).hexdigest()==ledger['old_sha256'];assert hashlib.sha256(Path(ledger['new_file']).read_bytes()).hexdigest()==ledger['new_sha256']
for bias in p['background_nA']:
 for vel in [0.,-.1,.1]:
  for dt in [.5,.0625]:
   tag=f'muscle_tendon_sensory_cut_p250_dt{dt}.npz';a=np.load(O/f'b{bias}_g0.1_v{vel}'/tag);b=np.load(O/f'b{bias}_g0.3_v{vel}'/tag)
   assert np.array_equal(a['rows'],b['rows']) and np.array_equal(a['spikes_packed'],b['spikes_packed'])
report=dict(importance='核心必需',verified=True,physical_runs=48,configs=configs,rows=rows,all_configs_retained=True,repeated_cut_controls_bitwise_identical=True)
(O/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
