"""Parent release: ONE new physicalstop + saved paired audit, archivedstop not rerun."""
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
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-singlecase',action='store_true');a=p.parse_args()
    if not a.execute_parent_singlecase:p.error('Parent onecase release required')
    cpu1();sources=frozen();verify(sources);pre=read(ROOT/'PREFLIGHT.json')
    if not pre['all_passed'] or pre['source_hashes']!=sources:raise ValueError('Same-source actual0step preflight required')
    path=ROOT/'JOBS.json'
    if path.exists():raise FileExistsError('No overwrite or archivedcontrol rerun')
    jobs=[];journal(path,jobs)
    for stage,script,flag in (('PHYSICS','runner.py','--execute-parent-physical'),('SAVED_PAIRED_AUDIT','audit_saved.py','--execute-parent-saved-audit')):
        verify(sources);command=[sys.executable,'-B',str(ROOT/script),'--condition','stop',flag]
        row=dict(stage=stage,status='RUNNING',command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(ROOT/(stage+'.log')))
        jobs.append(row);journal(path,jobs);print(json.dumps(row),flush=True)
        with open(row['log'],'x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False)
        row.update(status='COMPLETED' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());journal(path,jobs)
        if result.returncode!=0:raise RuntimeError('Singlecase stage failed; no retry: '+stage)
    result=read(ROOT/'trials/stop/RESULT.json');comparison=read(ROOT/'MATCHED_COMPARISON.json')
    if result['actual_native_steps']>1232000 or not comparison['integrity_all_pass']:raise ValueError('Budget or pairedaudit failed')
    verify(sources)
    save(ROOT/'SUMMARY.json',dict(kind='SINGLE_NEW_FIXED_PRIVATE_SHADOW_STOP_CONTROL_SUMMARY_V1',new_physical_cases=1,new_native_steps=result['actual_native_steps'],ancestor_physics_rerun=False,comparison=str(ROOT/'MATCHED_COMPARISON.json'),comparison_sha256=sha(ROOT/'MATCHED_COMPARISON.json'),new_support_full_window_qualified=comparison['new_support_full_window_qualified'],stop_speed_PASS=None,noANN=True,no_optimizer=True,source_hashes=sources))
    print(json.dumps(dict(summary=str(ROOT/'SUMMARY.json'),new_native_steps=result['actual_native_steps'],archived_physics_rerun=False)),flush=True)


if __name__=='__main__':main()
