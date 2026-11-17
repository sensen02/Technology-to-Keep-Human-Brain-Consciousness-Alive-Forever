"""Stdlib-only admission from actual completed final300 identity, never guessed weights."""
import argparse,ast
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-calibration',action='store_true');p.add_argument('--brain-sha256');a=p.parse_args()
 if not a.bind_completed_calibration:print('CODE_ONLY_UNBOUND no models/physics');return
 digest=a.brain_sha256
 if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest) or sha(BRIDGE)!=digest:raise ValueError('Actual final300 bytes/digest required')
 if any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json')):raise FileExistsError('Inspect prior admission, no blindrebind')
 declared=[ast.literal_eval(n.value) for n in ast.parse((CAL/'common.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in n.targets)]
 if declared!=[FORMAT]:raise ValueError('Actual strict final300 loader FORMAT required')
 result=read(CAL/'EVALUATION/RESULT.json');audit=read(CAL/'EVALUATION/VERIFY.json');saved=read(CAL/'TRAIN/VERIFY_000300.json');jobs=read(CAL/'JOBS.json')
 if len(jobs)!=7 or not all(j['exit_code']==0 for j in jobs):raise ValueError('Actual seven calibration stages must finish')
 if result['checkpoint_sha256']!=digest or audit['checkpoint_sha256']!=digest or saved['sha256']!=digest or audit['result_sha256']!=sha(CAL/'EVALUATION/RESULT.json'):raise ValueError('Actual model/storage/result identity')
 if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or not saved['all_passed'] or not all(saved['checks'].values()):raise ValueError('Independent calibration replay/metric proof required')
 if sha(VISION)!=VISION_SHA or sha(MOTOR)!=MOTOR_SHA or sha(BINDING)!=BINDING_SHA:raise ValueError('Frozen vision/motor/source drift')
 visual_acceptance=read(V7/'FINAL_ACCEPTANCE.json');parent=PROJECT/'experiments/neck-real-structure/20261113T000915Z';parent_acceptance=read(parent/'FINAL_SCOPED_ACCEPTANCE.json');hashes={}
 for proof in (read(CAL/'FROZEN.json'),result,audit,saved):
  for key in ('source_hashes','data_hashes'):
   for path,h in proof.get(key,{}).items():
    if sha(path)!=h:raise ValueError('Calibration/parent source drift '+path)
    hashes[path]=h
 for evidence in (visual_acceptance['source_evidence_hashes'],parent_acceptance['evidence_hashes']):
  for path,h in evidence.items():
   if sha(path)!=h:raise ValueError('Prior completed evidence drift '+path)
   hashes[path]=h
 names=('PREREG.json','runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','review.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json')
 paths=[ROOT/n for n in names]+[CAL/n for n in ('common.py','model.py','CALIBRATION_PREREG.json','FROZEN.json','JOBS.json','EVALUATION/RESULT.json','EVALUATION/VERIFY.json','TRAIN/VERIFY_000300.json')]+[BRIDGE,VISION,MOTOR,BINDING,PEAK_SOURCE,NEUTRAL/'IDENTITY.json',NEUTRAL/'RESULT.json',NEUTRAL/'trace_1ms.npz',NEUTRAL/'neural_decisions.npz',VIS/'camera.py',VIS/'PREREG_CAMERA_PROBE.json',PROJECT/'experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json',V7/'FINAL_ACCEPTANCE.json',parent/'FINAL_SCOPED_ACCEPTANCE.json',SCOPE/'visual_source_audit/NEURAL_VISUAL_SOURCE_REVIEW.json',SCOPE/'visual_source_audit/OUTPUT_VERIFICATION.json']
 for folder in (BASE,BASE/'sources',MOTOR_DIR,POLICY,VIS/'visual_body_v1',VIS/'visual_body_v2',VIS/'landmark_far_v3/body',VIS/'landmark_coverage_v5/body',V7/'body',PORTS):paths.extend(folder.glob('*.py'))
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
  hashes[str(path)]=sha(path)
 for row in read(BINDING)['source_files']:
  if sha(row['path'])!=row['sha256']:raise ValueError('Source fibre ledger drift')
  hashes[row['path']]=row['sha256']
 save(ROOT/'INPUT_BINDINGS.json',dict(kind=KIND,brain_checkpoint=str(BRIDGE),brain_sha256=digest,brain_final_step=300,brain_format=FORMAT,visual_checkpoint=str(VISION),visual_sha256=VISION_SHA,motor_checkpoint=str(MOTOR),motor_sha256=MOTOR_SHA,neck_binding=str(BINDING),neck_sha256=BINDING_SHA,offline_integrity_verified=True,new_body_qualified=False,full_neck_qualified=False,physical_steps=0))
 hashes[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json');save(ROOT/'FROZEN.json',dict(kind='ACTUAL_RGB_NECK_BODY_INPUTS_FROZEN',source_hashes=hashes,Core_constructions=0,model_loads=0,physics_steps=0));print('ACTUAL_FINAL300_THREE_MODELS_AND_NECK_BOUND_NO_PHYSICS')
if __name__=='__main__':main()
