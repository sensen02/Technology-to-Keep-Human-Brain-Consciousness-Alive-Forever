"""Fixed parent collection cohort:4physics+4NN+3teacher0step+summary=12."""
import argparse
import datetime
import subprocess
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-fourcases',action='store_true');args=parser.parse_args()
    if not args.execute_parent_fourcases:parser.error('Explicit parent finite teacher-assisted collection required')
    cpu1();sources=frozen();verify(sources);pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Same-source actual oneCore0step preflight required')
    path=ROOT/'JOBS.json'
    if path.exists():raise FileExistsError('No overwrite/no automatic rerun')
    jobs=[]
    def journal():
        temp=path.with_suffix('.tmp')
        with temp.open('x') as stream:json.dump(jobs,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    def stage(case,label,script,flag):
        verify(sources);command=[sys.executable,'-B',str(ROOT/script),'--case',case,flag]
        row=dict(case=case,stage=label,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(case+'_'+label+'.log')));jobs.append(row);journal();print(json.dumps(row),flush=True)
        with open(row['log'],'x') as log:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
        row.update(status='COMPLETED' if completed.returncode==0 else 'FAILED',exit_code=completed.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal()
        if completed.returncode!=0:raise RuntimeError('Finite collection stage failed; preserve partials, no retry: '+case+' '+label)
    journal()
    for case in CASES:
        stage(case,'PHYSICS_COLLECTION','collector.py','--execute-parent-physical')
        stage(case,'FRESH_NN_SAVED_AUDIT','audit_nn.py','--execute-parent-saved-audit')
    for case in CASES:
        if case!='old_walk_control':stage(case,'TEACHER_LABEL_0INTEGRATION_AUDIT','audit_teacher.py','--execute-parent-teacher-audit')
    row=dict(stage='SUMMARY',status='RUNNING',start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());jobs.append(row);journal();rows=[];steps=0
    for case in CASES:
        folder=ROOT/'trials'/case;r=read(folder/'RESULT.json');nn=read(folder/'NN_VERIFY.json')
        if not nn['integrity_all_pass'] or nn['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actual fresh NN saved proof required')
        proof=None
        if case!='old_walk_control':
            proof=read(folder/'TEACHER_VERIFY.json')
            if not proof['integrity_all_pass'] or proof['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('Actual independent current-state teacher label replay proof required')
        steps+=r['native']['actual_native_steps']
        rows.append(dict(case=case,condition=CASES[case][0],command=CASES[case][1],actual_native_steps=r['native']['actual_native_steps'],teacher_assisted=case!='old_walk_control',full_duration=r['full_duration'],early_stop=r['early_stop'],qualified_teacher_labels=r['qualified_teacher_labels'],original_five=r['original_five'],descriptive_engineering_metrics=r['candidate_metrics'],result=str(folder/'RESULT.json'),result_sha256=sha(folder/'RESULT.json'),NN_VERIFY_sha256=sha(folder/'NN_VERIFY.json'),TEACHER_VERIFY_sha256=None if proof is None else sha(folder/'TEACHER_VERIFY.json'),label_replay_integrity_verified=proof is not None and proof['integrity_all_pass'],neural_acceptance=False))
    if steps>4928000 or len(jobs)!=12:raise ValueError('Fixed cohort native/stage budget exceeded')
    verify(sources)
    save(ROOT/'SUMMARY.json',dict(kind=FORMAT+'_SUMMARY',label=LABEL,scope=SCOPE,stages=12,actual_new_native_steps=steps,max_native_budget=4928000,rows=rows,checkpoint_sha256=ACTUAL_SHA,source_hashes=sources,training_updates=0,neural_acceptance=False,new_neck_deployed=False,data_qualified_for_training=False,parent_review_before_any_training=True,interpretation='Teacher-assisted collection and matched old256 control only. Saved NN and teacher-label integrity are separate from physical stability and autonomous acceptance. All invalid mask rows and unsafe early stops retained.'))
    row.update(status='COMPLETED',exit_code=0,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary_sha256=sha(ROOT/'SUMMARY.json'));journal();print('FIXED_ASSISTED_COLLECTION_AND_SAVED_AUDITS_COMPLETED_NOT_NEURAL_ACCEPTANCE',flush=True)


if __name__=='__main__':main()
