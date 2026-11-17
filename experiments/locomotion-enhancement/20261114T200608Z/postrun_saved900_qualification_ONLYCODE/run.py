"""Conditional postrun three-stage saved-model qualification, shared3600s cap."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from metadata import ROOT,ORIGINAL,PYTHON,read,save,sha,verify,prepare,qualify
from runtime import STAGES


def main():
    p=argparse.ArgumentParser();p.add_argument('--sha256',required=True);p.add_argument('--execute-parent-postrun-saved900',action='store_true');a=p.parse_args()
    if not a.execute_parent_postrun_saved900:p.error('Explicit parent postrun saved900 qualification required')
    if sys.executable!=str(PYTHON):raise ValueError('Use original qualified125712Python')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(name+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if any((ROOT/name).exists() for name in ('JOBS.json','FAILURE.json','RESOURCE_COMPLETION.json')):raise FileExistsError('No retry/extension/overwrite')
    started=time.monotonic();end=started+3600.
    def cap(*unused):
        if time.monotonic()>=end or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.>1800:raise RuntimeError('Postrun shared3600/RSS1800 hard cap exceeded')
    signal.signal(signal.SIGALRM,cap);signal.setitimer(signal.ITIMER_REAL,1.,1.)
    meta=prepare(a.sha256);jobs=[];path=ROOT/'JOBS.json'
    def journal():
        temp=path.with_suffix('.tmp')
        with temp.open('x') as stream:json.dump(jobs,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    journal()
    for stage in STAGES:
        for key in ('source_hashes','data_hashes','proof_hashes'):verify(meta[key])
        remaining=end-time.monotonic()
        if remaining<=0:raise RuntimeError('Shared3600 deadline expired before next stage; no extension')
        command=[sys.executable,'-B',str(ROOT/'runtime.py'),'--stage',stage,'--sha256',a.sha256,'--deadline-monotonic',str(end),'--execute-parent-postrun-stage']
        row=dict(stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(stage+'.log')));jobs.append(row);journal()
        with open(row['log'],'x') as log:
            process=None;oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
            try:
                # Install cleanup before exposing the new processgroup to the
                # parent's periodic deadline handler. Child restores its mask.
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
                signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
                try:code=process.wait(timeout=max(0.,end-time.monotonic()))
                except subprocess.TimeoutExpired:code=124
            except BaseException:
                row.update(status='FAILED',exit_code=124,pipeline_elapsed_s=time.monotonic()-started,reason='POSTRUN_SHARED_RESOURCE_OR_INTERRUPT');journal();raise
            finally:
                # A second periodic cap must not interrupt kill/wait cleanup.
                signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
                try:
                    if process is not None and process.poll() is None:
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait()
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        row.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,pipeline_elapsed_s=time.monotonic()-started,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal()
        if code!=0:raise RuntimeError('Postrun saved qualification stage failed, original failure and partial newoutputs retained; no retry')
    final=qualify(a.sha256)
    for key in ('source_hashes','data_hashes','proof_hashes'):verify(final[key])
    fresh=read(ROOT/'TRAIN/VERIFY_000900.json');evaluation=read(ROOT/'EVALUATION.json');audit=read(ROOT/'EVALUATION_VERIFY.json');tr=read(ROOT/'TRAIN/RESULT.json')
    if len(jobs)!=3 or any(row['exit_code']!=0 for row in jobs) or not fresh['integrity_all_pass'] or not all(fresh['checks'].values()) or not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['evaluation_sha256']!=sha(ROOT/'EVALUATION.json') or evaluation['checkpoint_sha256']!=a.sha256 or tr['sha256']!=a.sha256 or tr['qualification_new_updates']!=0:raise ValueError('Actual allthree new saved900 proofs required')
    outputs={str(p):sha(p) for p in (ROOT/'TRAIN/VERIFY_000900.json',ROOT/'TRAIN/CONDITIONAL_000900.npz',ROOT/'TRAIN/RESULT.json',ROOT/'EVALUATION.json',ROOT/'EVALUATION_ARRAYS.npz',ROOT/'EVALUATION_VERIFY.json',ROOT/'JOBS.json',ROOT/'SOURCE_LEDGER.json')}
    for stage in STAGES:outputs[str(ROOT/(stage+'_COMPLETION.json'))]=sha(ROOT/(stage+'_COMPLETION.json'))
    verify(outputs)
    elapsed=time.monotonic()-started
    if elapsed>=3600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.>1800:raise RuntimeError('No success commit after postrun resource deadline')
    save(ROOT/'STAGE_VERIFICATION.json',dict(kind='POSTRUN_SAVED900_QUALIFICATION_NOT_ORIGINAL_PIPELINE_SUCCESS',qualification_mode='POSTRUN_SAVED_MODEL_ONLY',all_postrun_stages_exit0=True,original_pipeline_all3exit0=False,original_deadline_failure=final['original_deadline_failure'],checkpoint=final['checkpoint'],checkpoint_sha256=a.sha256,completed_original_total_updates=900,completed_original_new_updates=300,postrun_training_updates=0,optimizer_operations=0,physical_steps=0,new_neck_deployed=False,body_behavior_pass=None,outputs=outputs,wall_s=elapsed,source_hashes=final['source_hashes'],data_hashes=final['data_hashes'],proof_hashes=final['proof_hashes']))
    # Commit resource completion only after all success outputs have been written
    # and independently read/hash-checked; original failure remains immutable.
    stage=read(ROOT/'STAGE_VERIFICATION.json');verify(stage['outputs']);artifacts=dict(outputs);artifacts[str(ROOT/'STAGE_VERIFICATION.json')]=sha(ROOT/'STAGE_VERIFICATION.json')
    elapsed=time.monotonic()-started
    if elapsed>=3600:raise RuntimeError('Shared deadline expired before final resource completion')
    save(ROOT/'RESOURCE_COMPLETION.json',dict(**{'pass':True},status='PASS',qualification_mode='POSTRUN_SAVED_MODEL_ONLY',wall_s=elapsed,wall_limit_s=3600,RSS_limit_MiB=1800,stage_RSS_MiB={stage:read(ROOT/(stage+'_COMPLETION.json'))['peak_RSS_MiB'] for stage in STAGES},artifacts=artifacts,all_artifact_hashes_reverified=True,original_failure_preserved=True,no_false_original_pipeline_PASS=True,science_and_success_output_included=True,failure_bookkeeping_never_qualifies=True,requires_no_FAILURE_and_actual_run_exit0=True,postrun_training_updates=0,optimizer_operations=0,Core_constructions=0,physical_steps=0,body_behavior_pass=None,new_neck_deployed=False))
    verify(read(ROOT/'RESOURCE_COMPLETION.json')['artifacts'])
    if time.monotonic()>=end:raise RuntimeError('Shared deadline exceeded during final completion verification')
    print('POSTRUN_SAVED900_QUALIFIED_NO_NEW_UPDATES_OR_ORIGINAL_PIPELINE_PASS',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0.)
        if not (ROOT/'FAILURE.json').exists():save(ROOT/'FAILURE.json',dict(kind='POSTRUN_SAVED900_QUALIFICATION_FAILED',error_type=type(exc).__name__,error=repr(exc),qualification_pass=False,partial_outputs_preserved=True,any_completion_requires_no_FAILURE_and_actual_process_exit0=True,new_training_updates=0,optimizer_operations=0,Core_constructions=0,physical_steps=0,no_retry_or_extension=True))
        raise
