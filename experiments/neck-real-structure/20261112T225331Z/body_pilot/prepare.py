"""Stdlib fixed-checkpoint/artifact admission; calibration integrity not all-task functional PASS."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-calibration',action='store_true');p.add_argument('--brain-sha256');a=p.parse_args()
 if not a.bind_completed_calibration:print('CODE_ONLY no admission');return
 if a.brain_sha256!=BRIDGE_SHA or sha(BRIDGE)!=BRIDGE_SHA:raise ValueError('Actual final200 calibration SHA required')
 if (ROOT/'FROZEN.json').exists() or (ROOT/'INPUT_BINDINGS.json').exists():raise FileExistsError('Do not rebind completed or partial body admission')
 result=read(CAL/'EVALUATION/RESULT.json');audit=read(CAL/'EVALUATION/VERIFY.json');saved=read(CAL/'TRAIN/VERIFY_000200.json');jobs=read(CAL/'JOBS.json')
 if result['checkpoint_sha256']!=BRIDGE_SHA or audit['checkpoint_sha256']!=BRIDGE_SHA or saved['sha256']!=BRIDGE_SHA:raise ValueError('Actual calibration identity mismatch')
 if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=sha(CAL/'EVALUATION/RESULT.json') or not saved['all_passed'] or not all(saved['checks'].values()):raise ValueError('Actual calibrated storage/replay evidence required')
 if len(jobs)!=7 or not all(j['exit_code']==0 for j in jobs):raise ValueError('Actual completed fixed200 pipeline required')
 if sha(BINDING)!=BINDING_SHA or sha(MOTOR)!=MOTOR_SHA:raise ValueError('Source-neck/motor identity changed')
 hashes={}
 for proof in (read(CAL/'FROZEN.json'),result,audit,saved):
  for key in ('source_hashes','data_hashes'):
   for path,h in proof.get(key,{}).items():
    if sha(path)!=h:raise ValueError('Calibration source drift '+path)
    hashes[path]=h
 names=('PREREG.json','runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json');paths=[ROOT/n for n in names]+[CAL/n for n in ('model.py','common.py','FROZEN.json','PREREG.json','JOBS.json','TRAIN/VERIFY_000200.json','EVALUATION/RESULT.json','EVALUATION/VERIFY.json')]+[BRIDGE,BINDING,MOTOR,PEAK_SOURCE]+[PORTS/n for n in ('api.py','source_transport.py')]+[MOTOR_DIR/n for n in ('model.py','model_io.py','USER_VISUAL_ACCEPTANCE.json')]+list(BASE.glob('*.py'))+list((BASE/'sources').glob('*.py'))+[NEUTRAL/n for n in ('IDENTITY.json','RESULT.json','trace_1ms.npz','neural_decisions.npz')]
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
  hashes[str(path)]=sha(path)
 binding=read(BINDING)
 for row in binding['source_files']:
  if sha(row['path'])!=row['sha256']:raise ValueError('Real source binding changed')
  hashes[row['path']]=row['sha256']
 save(ROOT/'INPUT_BINDINGS.json',dict(kind='ARTIFICIAL_BRAIN_SOURCE_NECK_BODY_PILOT_V1',brain_checkpoint=str(BRIDGE),brain_sha256=BRIDGE_SHA,brain_final_step=200,neck_binding=str(BINDING),neck_sha256=BINDING_SHA,motor_checkpoint=str(MOTOR),motor_sha256=MOTOR_SHA,offline_integrity_verified=True,offline_all_functional_PASS_claim=False,body_function_qualified=False,whole_neck_qualified=False,physical_steps=0))
 hashes[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json');save(ROOT/'FROZEN.json',dict(kind='BODY_PILOT_INPUTS_FROZEN_BEFORE_PHYSICS',source_hashes=hashes,model_loads=0,Core_constructions=0,physics_steps=0));print('ACTUAL_CALIBRATION200_SOURCE_AND_MOTOR_BOUND_NO_PHYSICS')
if __name__=='__main__':main()
