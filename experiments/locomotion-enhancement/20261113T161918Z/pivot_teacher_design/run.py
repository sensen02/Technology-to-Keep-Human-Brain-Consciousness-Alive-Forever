"""Parent opt-in serial finite3 physical cases + saved-only audits, no retries."""
import argparse
import datetime
import os
import subprocess
import sys
import time
from common import *


def journal(path, value):
    temp=path.with_suffix('.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-threecases',action='store_true');a=p.parse_args()
    if not a.execute_parent_threecases:p.error('Parent finite three-case release flag required')
    cpu1(); sources=frozen(); verify(sources)
    pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Actual same-source PREFLIGHT required')
    path=ROOT/'JOBS.json'
    if path.exists() or (ROOT/'SUMMARY.json').exists():raise FileExistsError('No overwrite/retry of finite experiment')
    jobs=[];journal(path,jobs);started=time.monotonic()
    for condition in ('stop','left_pivot','right_pivot'):
        for stage, script, flag in (('PHYSICS','runner.py','--execute-parent-physical'),('AUDIT','audit_saved.py','--execute-parent-saved-audit')):
            verify(sources)
            command=[sys.executable,'-B',str(ROOT/script),'--condition',condition,flag]
            row=dict(condition=condition,stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(condition+'_'+stage+'.log')))
            jobs.append(row);journal(path,jobs)
            print(json.dumps(row),flush=True)
            with open(row['log'],'x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
            row.update(status='COMPLETED' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal(path,jobs)
            if result.returncode!=0:raise RuntimeError('Finite pipeline stage failed; no automatic retry: '+condition+' '+stage)
    rows=[];steps=0
    for condition in ('stop','left_pivot','right_pivot'):
        trial=ROOT/'trials'/condition;r=read(trial/'RESULT.json');v=read(trial/'VERIFY.json')
        if not v['integrity_all_pass'] or v['result_sha256']!=sha(trial/'RESULT.json'):raise ValueError('Independent saved audit required')
        steps+=r['actual_native_steps'];rows.append(dict(condition=condition,actual_native_steps=r['actual_native_steps'],early_stop=r['early_stop'],full_duration=r['full_duration'],candidate_metrics=r['candidate_metrics'],original_five=r['original_five'],result=str(trial/'RESULT.json'),result_sha256=sha(trial/'RESULT.json'),audit=str(trial/'VERIFY.json'),audit_sha256=sha(trial/'VERIFY.json')))
    if steps>3696000:raise ValueError('Fixed physical budget exceeded')
    verify(sources)
    summary=dict(kind='THREE_CASE_ALGORITHM_TEACHER_PIVOT_FEASIBILITY_SUMMARY_V1',completed_stages=len(jobs),all_stage_exit0=all(j['exit_code']==0 for j in jobs),total_actual_native_steps=steps,max_native_budget=3696000,rows=rows,scope='Teacher-only finite diagnostic. Stop success threshold pending; no neural/neck/pivot generalization or fullgoalcompletion claim.',source_hashes=sources,wall_s=time.monotonic()-started)
    save(ROOT/'SUMMARY.json',summary);print(json.dumps(dict(summary=str(ROOT/'SUMMARY.json'),total_actual_native_steps=steps,stages=len(jobs))),flush=True)


if __name__=='__main__':main()
