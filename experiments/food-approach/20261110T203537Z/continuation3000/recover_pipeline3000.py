"""Resume interrupted orchestration with immutable evaluator sources and fresh output paths."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import subprocess
import time
from pipeline3000 import HERE,PYTHON,CHECKPOINT,CONDITIONS,sha256,dump,readiness

SHA='17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8'


def main():
    p=argparse.ArgumentParser();p.add_argument('--restart-incomplete-neutral',action='store_true');a=p.parse_args()
    if not a.restart_incomplete_neutral:raise ValueError('explicit incomplete-neutral restart authorization required')
    old=json.loads((HERE/'FINAL3000_SUMMARY.json').read_text())
    frozen=readiness()
    if old['source_freeze']!=frozen or sha256(CHECKPOINT)!=SHA:raise ValueError('source/checkpoint drift')
    if old['conditions'] or (HERE/'EVAL_neutral/RESULT.json').exists():raise ValueError('unexpected completed original condition')
    if any((HERE/('EVAL_'+c)).exists() for c in CONDITIONS[1:]):raise ValueError('unexpected already-started other condition')
    outputs={c:HERE/('RECOVERY_EVAL_'+c) for c in CONDITIONS}
    report=HERE/'RECOVERY_FINAL3000_SUMMARY.json'
    if report.exists() or any(x.exists() for x in outputs.values()):raise FileExistsError('fresh recovery outputs required')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    recovery_hash=sha256(__file__);started=time.time();results={}
    interruption=dict(original_job='bash-219',job_handle='unknown after session recovery',processes_surviving=False,original_summary=str(HERE/'FINAL3000_SUMMARY.json'),original_summary_sha256=sha256(HERE/'FINAL3000_SUMMARY.json'),incomplete_neutral=str(HERE/'EVAL_neutral'),saved_artifacts={str(x):sha256(x) for x in (HERE/'EVAL_neutral').iterdir() if x.is_file()},saved_trajectory=False,restart_reason='No endpoint/neural arrays persisted; cannot continue or audit interrupted in-memory trajectory. Original artifacts preserved.')
    def persist():
        complete=set(results)==set(CONDITIONS)
        dump(report,dict(status='COMPLETE' if complete else 'RUNNING',checkpoint=str(CHECKPOINT),checkpoint_step=3000,checkpoint_sha256=SHA,
            all_conditions_eligible=bool(complete and all(v['eligible'] for v in results.values())),saved_trajectory_verification=bool(complete and all(v['saved_trajectory_verified'] for v in results.values())),
            conditions=results,source_freeze=frozen,recovery_source={str(Path(__file__).resolve()):recovery_hash},interruption=interruption,elapsed_s=time.time()-started,
            scope='fixed3000 replacementANN engineering only; no food behavior or full native certification'))
    def run(label,argv):
        with (HERE/(label+'.log')).open('x') as log:
            done=subprocess.run([str(PYTHON),'-B',*map(str,argv)],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        return dict(returncode=done.returncode,log=str(HERE/(label+'.log')),argv=[str(PYTHON),'-B',*map(str,argv)])
    def evaluate(c):
        out=outputs[c];audit=HERE/('RECOVERY_VERIFY_ACTUAL_'+c+'.json')
        argv=[HERE/'eval3000.py','--checkpoint',CHECKPOINT,'--sha256',SHA,'--condition',c,'--out',out]
        if c.startswith('speed'):argv+=['--neutral-reference',outputs['neutral']/'RESULT.json']
        dump(HERE/('RECOVERY_ASSOCIATION_'+c+'.json'),dict(condition=c,output=str(out),checkpoint=str(CHECKPOINT),checkpoint_sha256=SHA,recovery_source={str(Path(__file__).resolve()):recovery_hash},frozen_evaluation_sources=frozen,command=[str(PYTHON),'-B',*map(str,argv)],interruption=interruption))
        execution=run('RECOVERY_EVAL_'+c,argv)
        value=dict(condition=c,eligible=False,saved_trajectory_verified=False,execution=execution)
        if execution['returncode']==0:
            r=json.loads((out/'RESULT.json').read_text());auditing=run('RECOVERY_AUDIT_'+c,[HERE/'verify_actual3000.py','--run',out,'--out',audit])
            value.update(result=str(out/'RESULT.json'),result_sha256=sha256(out/'RESULT.json'),audit=auditing,criteria=r['criteria'],autonomous=r['autonomous'],yaw_change_rad=r['yaw_change_rad'],checkpoint_sha256=r['checkpoint_sha256'])
            if auditing['returncode']==0:
                v=json.loads(audit.read_text());value.update(saved_trajectory_verified=bool(v['integrity_all_pass']),audit_path=str(audit),audit_sha256=sha256(audit),audit_checks=len(v['checks']))
                value['eligible']=bool(r['engineering_pass'] and v['engineering_pass'] and v['integrity_all_pass'] and r['checkpoint_sha256']==SHA and v['checkpoint_sha256']==SHA)
        return value
    def group(names):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={c:pool.submit(evaluate,c) for c in names}
            for c,future in futures.items():
                try:results[c]=future.result()
                except BaseException as exc:results[c]=dict(condition=c,eligible=False,saved_trajectory_verified=False,error=repr(exc))
                persist();print(json.dumps(dict(condition=c,eligible=results[c]['eligible'],saved_trajectory_verified=results[c]['saved_trajectory_verified'])),flush=True)
    persist();group(('neutral',));group(('left','right'))
    if results['neutral']['eligible']:group(('speed085','speed090'))
    else:
        for c in ('speed085','speed090'):results[c]=dict(condition=c,eligible=False,saved_trajectory_verified=False,status='NOT_RUN_MATCHED_NEUTRAL_FAILED')
        persist()
    if readiness()!=frozen or sha256(CHECKPOINT)!=SHA or sha256(__file__)!=recovery_hash:raise ValueError('source/checkpoint drift during recovery')
    print(json.dumps(dict(summary=str(report),all_conditions_eligible=all(v['eligible'] for v in results.values()),saved_trajectory_verification=all(v['saved_trajectory_verified'] for v in results.values()))),flush=True)
    if any('error' in v or v.get('execution',{}).get('returncode',0)!=0 or v.get('audit',{}).get('returncode',0)!=0 for v in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
