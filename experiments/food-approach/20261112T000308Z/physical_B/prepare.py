"""Stdlib-only gate. Explicit final SHA required; never deploys or creates a Core."""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parent;CANDIDATE=ROOT.parent
AB=CANDIDATE.parent/'20261111T224234Z'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-offline-pass',action='store_true');p.add_argument('--behavior-sha256');a=p.parse_args()
 if not a.bind_completed_offline_pass:print('CODE_ONLY_UNBOUND: no admission written');return
 digest=a.behavior_sha256;require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'explicit actual finalSHA required')
 require(not (ROOT/'INPUT_BINDINGS.json').exists() and not (ROOT/'FROZEN.json').exists(),'immutable bindings already exist; no blind rebind')
 ck=CANDIDATE/'TRAIN/checkpoint_000500.pt';rp=CANDIDATE/'VALIDATION/RESULT.json';ap=CANDIDATE/'VALIDATION/VERIFY.json';saved=CANDIDATE/'TRAIN/VERIFY_000500.json'
 r=read(rp);v=read(ap);s=read(saved)
 require(sha(ck)==digest and Path(r['checkpoint']).resolve()==ck and r['checkpoint_sha256']==v['checkpoint_sha256']==s['sha256']==digest,'actual final500 checkpoint mismatch')
 require(r['all_targets_pass'] is True and len(r['rows'])==12 and all(row['all_targets_pass'] is True and all(x is True for x in row['flags'].values()) for row in r['rows']),'all12 offline functional targets required')
 require(v['integrity_all_pass'] is True and v['all_targets_pass'] is True and all(x is True for x in v['checks'].values()) and v['result_sha256']==sha(rp),'independent validation proof required')
 require(s['all_passed'] is True and all(x is True for x in s['checks'].values()),'independent final saved replay required')
 require(r['motor_sha256']==MOTOR_SHA,'offline motor mismatch')
 hashes={}
 for freeze in (AB/'FROZEN.json',CANDIDATE/'FROZEN.json'):
  for path,h in read(freeze)['source_hashes'].items():require(sha(path)==h,'inherited frozen dependency drift '+path);hashes[path]=h
 for proof in (r,v,s):
  for path,h in proof.get('source_hashes',{}).items():require(sha(path)==h,'offline source drift '+path);hashes[path]=h
 for path in (ck,rp,ap,saved,CANDIDATE/'common.py',CANDIDATE/'FROZEN.json',CANDIDATE/'PREREG.json',CANDIDATE/'VALIDATION/OUTPUTS.npz'):
  require(path.is_file(),'missing proof '+str(path));hashes[str(path)]=sha(path)
 require(sha(r['outputs']['path'])==r['outputs']['sha256'] and Path(r['outputs']['path']).resolve()==CANDIDATE/'VALIDATION/OUTPUTS.npz','offline outputs identity')
 gate=read(AB/'A_GATE.json');require(gate['all_A_integrity_and_functional_pass'] is True,'old A gate failed')
 for row in gate['rows']:
  name=row['condition'];folder=AB/'trials'/name;rr=read(folder/'RESULT.json');aa=read(folder/'VERIFY.json')
  require(rr['checkpoint_sha256']==MOTOR_SHA and rr['functional_pass'] is True and aa['integrity_all_pass'] is True and all(x is True for x in aa['checks'].values()),'same-motor A proof invalid')
  require(sha(folder/'RESULT.json')==row['result_sha256']==aa['result_sha256'] and sha(folder/'VERIFY.json')==row['audit_sha256'],'old A evidence drift')
  for path in (folder/'RESULT.json',folder/'VERIFY.json'):hashes[str(path)]=sha(path)
 for name in ('A_GATE.json','CLAMP_LEVELS.json'):
  source=AB/name;destination=ROOT/name
  if destination.exists():require(destination.read_bytes()==source.read_bytes(),'inherited local copy differs '+name)
  else:
   with destination.open('xb') as f:f.write(source.read_bytes())
  hashes[str(destination)]=sha(destination)
 for name in ('runtime.py','runner.py','audit.py','summarize_B.py','PREREG.json','CLAMP_LEVELS.json','A_GATE.json','FROZEN.json'):
  path=AB/name;hashes[str(path)]=sha(path)
 for name in ('PREREG.json','prepare.py','runtime.py','runner.py','audit.py','summarize_B.py','orchestrate.py','preflight.py','check_codeonly.py'):
  path=ROOT/name;require(path.is_file(),'new wrapper missing '+str(path));hashes[str(path)]=sha(path)
 fresh(ROOT/'INPUT_BINDINGS.json',dict(kind='CANDIDATE2_NEW_BEHAVIOR_PHYSICAL_B6_V1',behavior_checkpoint=str(ck),behavior_sha256=digest,behavior_loader=str(CANDIDATE/'common.py'),behavior_format='ODOR_SEGMENT_RECOVERY_V1',validation_result=str(rp),validation_result_sha256=sha(rp),validation_audit=str(ap),validation_audit_sha256=sha(ap),motor_sha256=MOTOR_SHA,A_gate_reference=str(AB/'A_GATE.json'),A_gate_sha256=sha(AB/'A_GATE.json'),A_trials_reused=3,new_B_trials=6,physical_executed=False))
 hashes[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json')
 fresh(ROOT/'FROZEN.json',dict(kind='CANDIDATE2_PHYSICAL_B6_INPUTS',source_hashes=hashes,physical_executed=False))
 print('OFFLINE_FINAL500_PASS_BOUND_NO_PHYSICS_EXECUTED')
if __name__=='__main__':main()
