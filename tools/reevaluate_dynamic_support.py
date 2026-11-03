"""Re-judge saved runs with the dynamic criteria (native 12.5 us resolution). Pure arrays."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import dynamic_support_metrics as dsm
O=Path('/home/sensen/Desktop/engine修复验收/s4-dynamic-support-criteria/20261102T234000Z')
SA=Path('/home/sensen/Desktop/engine修复验收/s4-stage-a-pool-modes/20261102T203810Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
RUNS=[
 ('baseline_standard_handover',BASE/'RUN/BODY.npz',(1.1,1.3),None),
 ('stageA_best_pool24',SA/'DIAGNOSE_R2/RUN/BODY.npz',(1.1,1.3),SA/'DIAGNOSE_R2/RELEASE_NATIVE.npz'),
 ('stageA_cand0_200ms',SA/'VALIDATION_R2/CAND0_repeatA_release200ms/RUN/BODY.npz',(1.1,1.3),SA/'VALIDATION_R2/CAND0_repeatA_release200ms/RELEASE_NATIVE.npz'),
 ('stageA_cand0_400ms',SA/'VALIDATION_R2/CAND0_extended_release400ms/RUN/BODY.npz',(1.1,1.3),SA/'VALIDATION_R2/CAND0_extended_release400ms/RELEASE_NATIVE.npz'),
 ('stageA_cand1_200ms',SA/'VALIDATION_R2/CAND1_repeatA_release200ms/RUN/BODY.npz',(1.1,1.3),SA/'VALIDATION_R2/CAND1_repeatA_release200ms/RELEASE_NATIVE.npz'),
]
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'RESULT.json').exists()
 rows=[]
 for name,body_p,aw,nat_p in RUNS:
  body=ar(body_p);assist=dsm.window_metrics(body,*aw,'assist')
  if nat_p is None:
   res=dsm.evaluate(body,aw,(1.3,1.4))
   res['resolution_note']='this run keeps its release records only at 0.5 ms and its native episodes are chunked, so the balance check is instrument-limited here; every other row uses the 12.5 us native records'
  else:
   res=dsm.evaluate_native(assist,ar(nat_p)['rows']);res['resolution_note']='12.5 us native release records'
  rows.append(dict(run=name,passed=res['passed'],checks=res['checks'],assist=assist,release=res['release'],tolerance=res['tolerance'],sources=dict(body=str(body_p),native=(str(nat_p) if nat_p else 'chunked under s5-banc888-physical-handover/NATIVE/'))))
  print(name,'pass',res['passed'],'failed',[k for k,v in res['checks'].items() if not v],flush=True)
 out=dict(passed_any=any(r['passed'] for r in rows),runs=rows,
  source_sha256={str(Path(__file__)):sha(Path(__file__)),'dynamic_support_metrics':sha(Path(__file__).parent/'dynamic_support_metrics.py')},
  static_criterion=dsm.retired_static_reference(),
  scope='re-judgement of already saved trajectories; no new simulation',
  metric_history=['v1 wrongly counted non-foot contact as support (foot 0.10-0.20 W appeared as 1.0 W); corrected to foot-only support with a separate non-foot share check','acceleration is differentiated from COM at native 12.5 us and low-passed over 9 samples before the balance check; at 0.5 ms sampling the balance check is unusable and must not be used'],
  calls=dict(model_load=0,forward=0,body_step=0,neural_step=0))
 (O/'RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print('passed_any',out['passed_any'])
if __name__=='__main__':main()
