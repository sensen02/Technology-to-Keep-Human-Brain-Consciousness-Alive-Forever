"""Stdlib gate: actual far final500 offline PASS before new complete-body admission."""
from runtime import *
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-offline-pass',action='store_true');p.add_argument('--visual-sha256');a=p.parse_args()
 if not a.bind_completed_offline_pass:print('CODE_ONLY_UNBOUND no admission or physics');return
 digest=a.visual_sha256
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('explicit actual completed far final500 SHA required')
 if (ROOT/'INPUT_BINDINGS.json').exists() or (ROOT/'FROZEN.json').exists():raise FileExistsError('no overwrite existing far body admission')
 ck=POLICY/'TRAIN/checkpoint_000500.pt';rp=POLICY/'VALIDATION/RESULT.json';ap=POLICY/'VALIDATION/VERIFY.json';sp=POLICY/'TRAIN/VERIFY_000500.json'
 r=read(rp);a=read(ap);saved=read(sp)
 if sha(ck)!=digest or Path(r['checkpoint']).resolve()!=ck or r['checkpoint_sha256']!=digest or a['checkpoint_sha256']!=digest or saved['sha256']!=digest:raise ValueError('actual far final500 source evidence mismatch')
 if r['all_targets_pass'] is not True or not r['rows'] or not all(row['all_targets_pass'] is True and all(v is True for v in row['flags'].values()) for row in r['rows'] if row['functional_required']):raise ValueError('far offline functional gates failed')
 if a['integrity_all_pass'] is not True or a['all_targets_pass'] is not True or not a['checks'] or not all(v is True for v in a['checks'].values()) or a['result_sha256']!=sha(rp):raise ValueError('far independent validation proof required')
 if saved['all_passed'] is not True or not saved['checks'] or not all(v is True for v in saved['checks'].values()):raise ValueError('far independent saved production replay required')
 # Exact actual loader source must declare the new format. No old near policy admitted by SHA alone.
 import ast
 tree=ast.parse((POLICY/'common.py').read_text());formats=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in node.targets)]
 if formats!=['VISION_FAR_LANDMARK_POLICY_V3']:raise ValueError('actual far common FORMAT declaration mismatch')
 hashes={}
 for proof in (read(POLICY/'FROZEN.json'),r,a,saved):
  for path,h in proof['source_hashes'].items():
   if sha(path)!=h:raise ValueError('far offline source drift '+path)
   hashes[path]=h
 for value in r['outputs'].values():
  if sha(value['path'])!=value['sha256']:raise ValueError('far offline output drift')
  hashes[value['path']]=value['sha256']
 optical=read(VIS/'CAMERA_PROBE/PREFLIGHT.json')
 for path,h in optical['source_hashes'].items():
  if sha(path)!=h:raise ValueError('camera source drift '+path)
  hashes[path]=h
 acceptance=MOTOR_DIR/'USER_VISUAL_ACCEPTANCE.json';accepted=read(acceptance)
 if accepted['checkpoint_sha256']!=MOTOR_SHA or accepted['visual_acceptance'] is not True or accepted['engineering_qualified'] is not False:raise ValueError('frozen accepted experimental motor required')
 names=('PREREG_PROPOSAL.json','runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json')
 paths=[ROOT/n for n in names]+[BODY2/'adapter.py']+[BODY1/n for n in ('runtime.py','runner.py','audit.py','summary.py')]+[POLICY/'common.py',POLICY/'model.py',POLICY/'FROZEN.json',ck,rp,ap,sp,VIS/'landmark_far_v3/PREREG.json',MOTOR,MOTOR_DIR/'model.py',MOTOR_DIR/'model_io.py',acceptance,PEAK_SOURCE,GEOMETRY,VIS/'camera.py',VIS/'PREREG_CAMERA_PROBE.json',VIS/'CAMERA_PROBE/PREFLIGHT.json']+list(BASE.glob('*.py'))+list((BASE/'sources').glob('*.py'))+[NEUTRAL/n for n in ('RESULT.json','IDENTITY.json','trace_1ms.npz','neural_decisions.npz')]
 for path in paths:
  if not path.is_file():raise FileNotFoundError(path)
  hashes[str(path)]=sha(path)
 save(ROOT/'INPUT_BINDINGS.json',dict(kind='VISUAL_FAR_WORLD_LANDMARK_BODY_V3',visual_checkpoint=str(ck),visual_sha256=digest,visual_loader=str(POLICY/'common.py'),visual_format=formats[0],motor_checkpoint=str(MOTOR),motor_sha256=MOTOR_SHA,offline_result=str(rp),offline_result_sha256=sha(rp),offline_audit=str(ap),offline_audit_sha256=sha(ap),far_only=True,near_domain_repaired=False,body_response_qualified=False,physical_steps=0))
 hashes[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json')
 save(ROOT/'FROZEN.json',dict(kind='FAR_BODY_INPUTS_BEFORE_ANY_PHYSICS',source_hashes=hashes,model_loads=0,Core_constructions=0,physics_steps=0))
 print('ACTUAL_OFFLINE_FAR_VISUAL_PASS_BOUND_NO_PHYSICS_EXECUTED')
if __name__=='__main__':main()
