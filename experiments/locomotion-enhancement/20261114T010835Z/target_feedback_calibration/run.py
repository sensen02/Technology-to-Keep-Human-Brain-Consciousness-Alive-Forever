"""Parentonefixed600 conditionalfeedbackpipeline withthreefreshsaveaudits."""
import argparse,datetime,os,subprocess,sys
from sources import ROOT,read,save,sha,verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-calibration',action='store_true');a=p.parse_args()
 if not a.execute_parent_calibration:p.error('Parentonefinitecandidate release required')
 b=read(ROOT/'INPUT_BINDINGS.json');verify(b['source_hashes']);verify(b['data_hashes']);pre=read(ROOT/'PREFLIGHT.json')
 if not pre['all_passed'] or pre['timing_status']!='READY_FIXED600_5400S' or pre['source_hashes']!=b['source_hashes'] or pre['data_hashes']!=b['data_hashes']:raise ValueError('Actualconditionalfullpreflightrequired')
 path=ROOT/'JOBS.json'
 if path.exists():raise FileExistsError('No overwrite/retry')
 jobs=[]
 def journal():
  temp=path.with_suffix('.tmp')
  with temp.open('x') as f:
   import json
   json.dump(jobs,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(temp,path)
 journal()
 for stage,script,flag in (('TRAIN_FIXED600_CONDITIONAL_FEEDBACK','train.py','--execute-parent-training'),('FINAL600_ACTUALX_AND_CONDITIONAL_METRICS','evaluate.py','--execute-parent-evaluation'),('INDEPENDENT_SAVED_METRIC_AUDIT','audit_eval.py','--execute-parent-saved-audit')):
  verify(b['source_hashes']);verify(b['data_hashes']);command=[sys.executable,'-B',str(ROOT/script),flag];row=dict(stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(stage+'.log')));jobs.append(row);journal();print(stage,flush=True)
  with open(row['log'],'x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
  row.update(status='COMPLETED' if r.returncode==0 else 'FAILED',exit_code=r.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal()
  if r.returncode!=0:raise RuntimeError('Finiteconditionalstagefailed,noautomaticextension')
 tr=read(ROOT/'TRAIN/RESULT.json');v=read(ROOT/'EVALUATION_VERIFY.json');verify(b['source_hashes']);verify(b['data_hashes'])
 if tr['updates']!=600 or not v['integrity_all_pass']:raise ValueError('Actualfinal600proofrequired')
 save(ROOT/'STAGE_VERIFICATION.json',dict(kind='TARGET_FEEDBACK_FIXED600_COMPLETED_V1',all_stages_exit0=True,train_updates=600,checkpoint=tr['checkpoint'],checkpoint_sha256=tr['sha256'],evaluation_sha256=sha(ROOT/'EVALUATION.json'),independent_audit_sha256=sha(ROOT/'EVALUATION_VERIFY.json'),physical_steps=0,body_behavior_pass=None,new_neck_deployed=False,conditional_notphysicalcounterfactual=True))


if __name__=='__main__':main()
