"""Stdlib actual offline-PASS admission; no guessed visual weight or Core construction."""
from runtime import *
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-offline-pass',action='store_true');p.add_argument('--visual-sha256');a=p.parse_args()
 if not a.bind_completed_offline_pass:print('CODE_ONLY no admission bound');return
 digest=a.visual_sha256
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('explicit actual visual final500 SHA required')
 if (ROOT/'INPUT_BINDINGS.json').exists() or (ROOT/'FROZEN.json').exists():raise FileExistsError('no rebind existing bodyinputs')
 ck=POLICY/'TRAIN/checkpoint_000500.pt';result=POLICY/'VALIDATION/RESULT.json';va=POLICY/'VALIDATION/VERIFY.json';sa=POLICY/'TRAIN/VERIFY_000500.json';r=read(result);audit=read(va);saved=read(sa)
 if sha(ck)!=digest or r['checkpoint_sha256']!=digest or audit['checkpoint_sha256']!=digest or saved['sha256']!=digest:raise ValueError('actual visualweight evidence binding required')
 if r['all_targets_pass'] is not True or not all(row['all_targets_pass'] is True and all(v is True for v in row['flags'].values()) for row in r['rows'] if row['functional_required']):raise ValueError('offline visual functional gate failed')
 if audit['integrity_all_pass'] is not True or audit['all_targets_pass'] is not True or not all(v is True for v in audit['checks'].values()) or audit['result_sha256']!=sha(result):raise ValueError('independent final visual validation proof required')
 if not saved['all_passed'] or not all(v is True for v in saved['checks'].values()):raise ValueError('saved actual CNN/GRU replay proof required')
 hashes={}
 for proof in (read(POLICY/'FROZEN.json'),r,audit,saved):
  for path,h in proof['source_hashes'].items():
   if sha(path)!=h:raise ValueError('offline source drift '+path)
   hashes[path]=h
 camera_probe=read(VIS/'CAMERA_PROBE/PREFLIGHT.json')
 for path,h in camera_probe['source_hashes'].items():
  if sha(path)!=h:raise ValueError('camera proof drift '+path)
  hashes[path]=h
 acceptance=MOTOR_DIR/'USER_VISUAL_ACCEPTANCE.json';accept=read(acceptance)
 if accept['checkpoint_sha256']!=MOTOR_SHA or accept['visual_acceptance'] is not True or accept['engineering_qualified'] is not False:raise ValueError('motor experimental acceptance identity')
 names=('PREREG_PROPOSAL.json','runtime.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py')
 paths=[ROOT/n for n in names]+[POLICY/'common.py',POLICY/'model.py',POLICY/'FROZEN.json',ck,result,va,sa,MOTOR,MOTOR_DIR/'model.py',MOTOR_DIR/'model_io.py',acceptance,PEAK_SOURCE,GEOMETRY,VIS/'camera.py',VIS/'PREREG_CAMERA_PROBE.json',VIS/'CAMERA_PROBE/PREFLIGHT.json']+list(BASE.glob('*.py'))+list((BASE/'sources').glob('*.py'))+[NEUTRAL/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz')]
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
  hashes[str(path)]=sha(path)
 save(ROOT/'INPUT_BINDINGS.json',dict(kind='VISUAL_BODY_WORLD_LANDMARK_V1',visual_checkpoint=str(ck),visual_sha256=digest,visual_loader=str(POLICY/'common.py'),visual_format='VISION_STATIC_HEAD_POLICY_V1',motor_checkpoint=str(MOTOR),motor_sha256=MOTOR_SHA,offline_result=str(result),offline_result_sha256=sha(result),offline_audit=str(va),offline_audit_sha256=sha(va),body_response_qualified=False,physical_steps=0))
 hashes[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json');save(ROOT/'FROZEN.json',dict(kind='VISUAL_BODY_INPUTS_BEFORE_ANY_PHYSICS',source_hashes=hashes,model_loads=0,Core_constructions=0,physics_steps=0));print('ACTUAL_OFFLINE_VISUAL_PASS_BOUND_NO_PHYSICS_EXECUTED')
if __name__=='__main__':main()
