"""Bind actual final400 only after complete calibration/storage/evaluation evidence; stdlib."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-calibration',action='store_true');p.add_argument('--brain-sha256');a=p.parse_args()
 if not a.bind_completed_calibration:print('CODE_ONLY_UNBOUND no admission');return
 digest=a.brain_sha256
 if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest) or sha(BRIDGE)!=digest:raise ValueError('Actualfinal400SHArequired')
 if (ROOT/'INPUT_BINDINGS.json').exists() or (ROOT/'FROZEN.json').exists():raise FileExistsError('No blind rebind')
 jobs=read(CAL/'JOBS.json');r=read(CAL/'EVALUATION/RESULT.json');v=read(CAL/'EVALUATION/VERIFY.json');saved=read(CAL/'TRAIN/VERIFY_000400.json')
 if len(jobs)!=7 or not all(x['exit_code']==0 for x in jobs) or r['checkpoint_sha256']!=digest or v['checkpoint_sha256']!=digest or saved['sha256']!=digest or not v['integrity_all_pass'] or not all(v['checks'].values()) or not saved['all_passed'] or not all(saved['checks'].values()) or v['result_sha256']!=sha(CAL/'EVALUATION/RESULT.json'):raise ValueError('Complete actualcalibration400independentproofneeded')
 if sha(VISION)!=VISION_SHA or sha(ODOR)!=ODOR_SHA or sha(MOTOR)!=MOTOR_SHA or sha(BINDING)!=BINDING_SHA:raise ValueError('Frozen source identity')
 proof={}
 for item in (read(CAL/'FROZEN.json'),r,v,saved):
  for key in ('source_hashes','data_hashes'):
   for path,h in item.get(key,{}).items():
    if sha(path)!=h:raise ValueError('Calibration ancestor drift '+path)
    proof[path]=h
 checked=read(ROOT/'CODEONLY_CHECK_FINAL.json')
 if not checked['all_passed'] or not all(checked['checks'].values()):raise ValueError('Sourcecodecontractsrequired')
 for path,h in checked['source_hashes'].items():
  if sha(path)!=h:raise ValueError('Bodycodechanged '+path)
  proof[path]=h
 names=('PREREG.json','runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json','CODEONLY_CHECK_FINAL.json');paths=[ROOT/n for n in names]+[CAL/n for n in ('common.py','model.py','FROZEN.json','JOBS.json','TRAIN/VERIFY_000400.json','EVALUATION/RESULT.json','EVALUATION/VERIFY.json')]+[BRIDGE,VISION,ODOR,MOTOR,BINDING,PEAK_SOURCE,HEAD_AUDIT]+[PREV/n for n in ('runtime.py','runner.py','audit.py','summary.py','FROZEN.json','NEURAL_VISUAL_BODY_SUMMARY.json','PARENT_READY_REVIEW.json')]+list(BASE.glob('*.py'))+list((BASE/'sources').glob('*.py'))+[NEUTRAL/n for n in ('IDENTITY.json','RESULT.json','trace_1ms.npz','neural_decisions.npz')]
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
  proof[str(path)]=sha(path)
 save(ROOT/'INPUT_BINDINGS.json',dict(kind=KIND,summary_kind=SUMMARY_KIND,brain_checkpoint=str(BRIDGE),brain_sha256=digest,brain_final_step=400,brain_format=FORMAT,visual_checkpoint=str(VISION),visual_sha256=VISION_SHA,odor_checkpoint=str(ODOR),odor_sha256=ODOR_SHA,motor_sha256=MOTOR_SHA,neck_sha256=BINDING_SHA,unknown_SHA_not_admitted=False,functional_pass_claim=False,new_physics_steps=0));proof[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json');save(ROOT/'FROZEN.json',dict(kind='MULTISENSORY_BODY_BOUND_BEFORE_PHYSICS',source_hashes=proof,model_loads=0,Core_constructions=0,new_physics_steps=0));print('ACTUAL_FUSION400_AND_FROZEN_SENSORY_NECK_MOTOR_BOUND_NO_PHYSICS')
if __name__=='__main__':main()
