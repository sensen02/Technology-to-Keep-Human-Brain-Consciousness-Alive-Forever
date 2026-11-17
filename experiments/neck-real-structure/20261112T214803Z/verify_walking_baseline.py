"""Read-only archived walking identity registration; no imports of numerical/physical backends."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 statepath=PROJECT/'WALKING_CURRENT_STATE.json';state=read(statepath);acceptpath=Path(state['final_acceptance']);a=read(acceptpath);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Walking baseline '+k)
 check('archived_acceptance_PASS',a['status']=='PASS')
 check('checkpoint_state_and_acceptance_same',state['selected_model']==a['selected_checkpoint'] and state['selected_model_sha256']==a['checkpoint_sha256'])
 evidence={str(statepath):sha(statepath),str(acceptpath):sha(acceptpath)}
 for key,digestkey in [('selected_checkpoint','checkpoint_sha256'),('native_audit','native_audit_sha256'),('video','video_sha256'),('video_evidence','video_evidence_sha256')]:
  p=Path(a[key]);actual=sha(p);check(key+'_bytes_match_archived_hash',actual==a[digestkey]);evidence[str(p)]=actual
 check('original_engineering5_PASS',len(a['native_engineering'])==5 and all(v['status']=='PASS' for v in a['native_engineering'].values()))
 check('native_slip_not_rounded',a['native_engineering']['slip']['value']==0.49411203866010717 and a['native_engineering']['slip']['value']<=.5)
 check('duration_full',a['test_duration']['settle_s']==1 and a['test_duration']['teacher_s']==4.8 and a['test_duration']['autonomous_s']==9.6)
 check('neural_chain_verified_in_archive',a['neural_synchronization']['hchain_exact'] and a['neural_synchronization']['all_checkpoint_outputs_bitexact'])
 for row in a['perturbations']:
  p=Path(row['run'])/'RESULT.json';check('perturb_'+str(row['seed'])+'_saved_result_hash',sha(p)==row['result_sha256']);evidence[str(p)]=sha(p)
 report=dict(kind='REGISTERED_WALKING_BASELINE_NOT_NEW_ARCHITECTURE_ACCEPTANCE',all_checks_pass=all(checks.values()),checks=checks,brain_type='trained_substitute_ANN',neck_real_structure_qualified=False,new_three_part_architecture_pass=False,checkpoint=a['selected_checkpoint'],checkpoint_sha256=a['checkpoint_sha256'],native_engineering=a['native_engineering'],complete_cycles=state['native_metrics']['complete_cycles'],duration=a['test_duration'],native_steps=state['native_integrity']['steps'],video=a['video'],desktop_video=state['desktop_video'],future_use='Immutable teacher/reference only; no direct brain-to-joint bypass. New architecture must pass full original walking and neural-interface interventions anew.',current_verification='Read saved evidence and stream hashes only; no new physical replay or video decode performed in this registration.',source_hashes=evidence,registration_script_sha256=sha(__file__),new_training_updates=0,new_physics_steps=0,Core_constructions=0)
 with (ROOT/'WALKING_BASELINE.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(all_checks_pass=True,checks=len(checks),native_slip=a['native_engineering']['slip']['value'],new_architecture_pass=False,new_physics_steps=0)))
if __name__=='__main__':main()
