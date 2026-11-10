import sys,subprocess,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');rows=[];gates={}
for bias in [0.,.02]:
 for vel in [0.,-.1,.1]:
  folder=ROOT/('sensory_bias_body_b'+str(bias)+'_v'+str(vel));subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('verify_sensory_bias_run.py')),str(folder)],check=True)
 for dt in [.5,.0625]:
  for vel in [-.1,.1]:
   metrics={}
   for cond in ['normal','sensory_cut','zero_output']:
    tag=f'muscle_tendon_{cond}_p250_dt{dt}';folder=ROOT/('sensory_bias_body_b'+str(bias)+'_v'+str(vel));ref=ROOT/('sensory_bias_body_b'+str(bias)+'_v0.0')
    a=np.load(folder/(tag+'.npz'))['rows'];b=np.load(ref/(tag+'.npz'))['rows'];r=next(r for r in json.loads((folder/'SUMMARY.json').read_text())['records'] if r['tag']==tag)
    metrics[cond]=dict(disturbance_iae=float(abs(a[:,1]-b[:,1]).sum()*.0000125),final_difference=float(a[-1,1]-b[-1,1]),within_range=r['within_range'],finite=r['finite'],warnings=r['warnings'])
   rows.append(dict(bias_nA=bias,dt_ms=dt,velocity=vel,metrics=metrics,normal_better_than_cut=metrics['normal']['disturbance_iae']<metrics['sensory_cut']['disturbance_iae']))
 gates[str(bias)]=all(r['normal_better_than_cut'] and all(m['within_range'] and m['finite'] and m['warnings']==0 for m in r['metrics'].values()) for r in rows if r['bias_nA']==bias)
report=dict(importance='核心必需',verified=True,physical_runs=36,rows=rows,feedback_recovery_gate=gates,definition='paired disturbance error relative to same-condition undisturbed movement; not q0 return under descending drive')
(ROOT/'SENSORY_BIAS_COMPARISON.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
