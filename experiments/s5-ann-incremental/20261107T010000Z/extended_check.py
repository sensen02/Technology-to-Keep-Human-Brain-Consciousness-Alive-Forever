"""Preregistered longer standing and true cold ANN controls, no retraining."""
from pathlib import Path
import json,numpy as np
from learn_stand import Runtime,O,RATE,dump,sha
from ann_translator import NumpyFullBrainTranslator

def main():
 root=O/'learning';modelpath=root/'TRANSLATOR.npz';model=NumpyFullBrainTranslator(modelpath)
 previous=json.loads((root/'EVALUATION.json').read_text())
 if not all(r['validation']['passed'] for r in previous if r['mode']=='ann'):
  raise RuntimeError('Both initial ANN time steps must pass before extended standing')
 protocol=dict(importance='核心必需',stage='independent followup after initial ANN standing',model=str(modelpath),model_sha256=sha(modelpath),source_sha256=sha(Path(__file__)),runtime_sha256=sha(Path(__file__).with_name('learn_stand.py')),conditions=['6s total:1s teacher+5s ANN, same heldout seed55 and .0625ms brain','3s true cold ANN from initial physical pose:0s teacher, .125ms brain','3s frozen brain history: use first post-warm history for every ANN call, keep live target state'],no_training=True,criterion='all same D support gates over post1s, no nonfoot support throughout cold run; quantify all output drift',cold_definition='override teacher() with ANN inference so even first1s commands come from ANN; run() still stores a state checkpoint at1s, not an actual controller switch')
 dump(root/'EXTENDED_PREREGISTRATION.json',protocol)
 results=[Runtime(.0625).run('extended_ann_5s','ann',55,model,seconds=6.)]
 class Cold(Runtime):
  def teacher(self,s):
   state=(s['a'].target_rad-self.p.q0[self.p.qa]).astype(np.float32)
   return RATE*model(s['history'][None],state[None])[0]
 results.append(Cold(.125).run('cold_ann','ann',66,model,seconds=3.))
 class FrozenHistory:
  def __init__(self):self.h=None
  def __call__(self,h,t):
   if self.h is None:self.h=h.copy()
   return model(self.h,t)
 results.append(Runtime(.125).run('frozen_brain_history','ann',55,FrozenHistory(),seconds=3.))
 for r in results:
  with np.load(Path(r['folder'])/'NATIVE.npz') as z:r['whole_run_nonfoot_max']=float(z['rows'][:,7].max())
 dump(root/'EXTENDED_RESULTS.json',results)
 assert sha(modelpath)==protocol['model_sha256']
 print(json.dumps(dict(extended_pass=results[0]['validation']['passed'],cold_pass=results[1]['validation']['passed'])),flush=True)
if __name__=='__main__':main()
