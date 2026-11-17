"""Parent finite release: only2 new pivots+2saved/paired audits, no old reruns."""
import argparse
import datetime
import os
import subprocess
import sys
from common import *


def journal(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-twocases',action='store_true');a=p.parse_args()
    if not a.execute_parent_twocases:p.error('Parent finite two-case release required')
    cpu1();sources=frozen();verify(sources);pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Actual same-source0step preflight required')
    path=ROOT/'JOBS.json'
    if path.exists():raise FileExistsError('No overwrite/retry or ancestor reruns')
    jobs=[];journal(path,jobs)
    for condition in ('left_pivot','right_pivot'):
        for stage,script,flag in (('PHYSICS','runner.py','--execute-parent-physical'),('SAVED_PAIRED_AUDIT','audit_saved.py','--execute-parent-saved-audit')):
            verify(sources);command=[sys.executable,'-B',str(ROOT/script),'--condition',condition,flag]
            row=dict(condition=condition,stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(condition+'_'+stage+'.log')))
            jobs.append(row);journal(path,jobs);print(json.dumps(row),flush=True)
            with open(row['log'],'x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
            row.update(status='COMPLETED' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal(path,jobs)
            if result.returncode!=0:raise RuntimeError('Finite stage failed, no retry: '+condition+' '+stage)
    rows=[];steps=0
    for condition in ('left_pivot','right_pivot'):
        r=read(ROOT/'trials'/condition/'RESULT.json');v=read(ROOT/('MATCHED_'+condition+'.json'))
        if not v['integrity_all_pass']:raise ValueError('Same-direction paired saved audit required')
        steps+=r['actual_native_steps'];rows.append(dict(condition=condition,actual_native_steps=r['actual_native_steps'],actual_ms=r['actual_ms'],candidate_metrics=r['candidate_metrics'],original_five=r['original_five'],early_stop=r['early_stop'],comparison=str(ROOT/('MATCHED_'+condition+'.json')),comparison_sha256=sha(ROOT/('MATCHED_'+condition+'.json')),teacher_pivot_task_pass=v['teacher_pivot_task_pass']))
    if steps>2464000:raise ValueError('Finite2case native budget exceeded')
    verify(sources)
    save(ROOT/'SUMMARY.json',dict(kind='TWO_CASE_FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_SUMMARY_V2',new_physical_cases=2,new_native_steps=steps,budget_native=2464000,ancestor_physics_rerun=False,rows=rows,noANN=True,no_optimizer=True,source_hashes=sources,scope='Only fixedconfig engineeringteacher pivotfeasibility; expectedyaw is not achievedyaw, no ANN/neck/nativefly conclusion.'))
    print(json.dumps(dict(summary=str(ROOT/'SUMMARY.json'),new_native_steps=steps,ancestor_physics_rerun=False)),flush=True)


if __name__=='__main__':main()
