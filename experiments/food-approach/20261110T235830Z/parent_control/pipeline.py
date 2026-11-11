"""Exactly two preregistered parent500 controls and saved trajectory audits."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import evaluate_parent as ep

HERE=ep.HERE

def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    ready=json.loads((HERE/'READINESS.json').read_text())
    if not ready['all_passed']:raise ValueError('control offline checks failed')
    for p,h in ready['source_hashes'].items():
        if ep.sha256(p)!=h:raise ValueError('readiness source drift '+p)
    if ep.sha256(ep.CHECKPOINT)!=ep.SHA:raise ValueError('checkpoint drift')
    summary=HERE/'SUMMARY.json'
    if summary.exists() or any((HERE/('EVAL_'+c)).exists() for c in ep.COMMANDS):raise FileExistsError('fresh controls required')
    frozen={str(p):ep.sha256(p) for p in ep.source_paths()};results={};started=time.monotonic()
    def persist():dump(summary,dict(status='COMPLETE' if len(results)==2 else 'RUNNING',checkpoint=str(ep.CHECKPOINT),checkpoint_sha256=ep.SHA,conditions=results,all_conditions_eligible=bool(len(results)==2 and all(r['eligible'] for r in results.values())),saved_trajectory_verification=bool(len(results)==2 and all(r['integrity'] for r in results.values())),source_hashes=frozen,elapsed_s=time.monotonic()-started,scope='Matched parent500 ramp control, no optimization or food qualification'))
    def command(label,argv):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (HERE/(label+'.log')).open('x') as log:
            process=subprocess.run([sys.executable,'-B',*map(str,argv)],env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        return dict(returncode=process.returncode,argv=[sys.executable,'-B',*map(str,argv)],log=str(HERE/(label+'.log')))
    def one(c):
        out=HERE/('EVAL_'+c);audit=HERE/('VERIFY_'+c+'.json')
        execution=command('EVAL_'+c,[HERE/'evaluate_parent.py','--checkpoint',ep.CHECKPOINT,'--sha256',ep.SHA,'--condition',c,'--out',out,'--training-complete'])
        value=dict(condition=c,execution=execution,eligible=False,integrity=False)
        if execution['returncode']==0:
            verification=command('VERIFY_'+c,[HERE/'verify_actual.py','--run',out,'--out',audit]);value['verification']=verification
            r=json.loads((out/'RESULT.json').read_text());value.update(result=str(out/'RESULT.json'),result_sha256=ep.sha256(out/'RESULT.json'),autonomous=r['autonomous'],cycles=r['autonomous_cycles'],yaw_change_rad=r['yaw_change_rad'],criteria=r['criteria'])
            if verification['returncode']==0:
                a=json.loads(audit.read_text());value.update(eligible=bool(a['integrity_all_pass'] and a['engineering_pass'] and r['engineering_pass']),integrity=bool(a['integrity_all_pass']),audit=str(audit),audit_sha256=ep.sha256(audit),audit_checks=len(a['checks']))
        return value
    persist()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,c):c for c in ep.COMMANDS}
        for future in as_completed(futures):
            c=futures[future]
            try:results[c]=future.result()
            except BaseException as exc:results[c]=dict(condition=c,error=repr(exc),eligible=False,integrity=False)
            persist();print(json.dumps(dict(condition=c,eligible=results[c]['eligible'],integrity=results[c]['integrity'],metrics=results[c].get('autonomous'),yaw=results[c].get('yaw_change_rad'))),flush=True)
    for p,h in frozen.items():
        if ep.sha256(p)!=h:raise ValueError('control source changed '+p)
    if any('error' in r or r['execution']['returncode']!=0 or r.get('verification',{}).get('returncode',0)!=0 for r in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
