"""Bounded final3000 physical evaluation and independent saved-actual audit runner.
Run only on explicit training-complete authorization and final checkpoint SHA.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
CHECKPOINT=HERE/'TRAIN/checkpoint_003000.pt'
CONDITIONS=('neutral','left','right','speed085','speed090')
FROZEN=('PREREG.json','train3000.py','verify3000.py','eval3000.py','verify_actual3000.py','pipeline3000.py','EVAL_PREREG.json')


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def readiness():
    for name in FROZEN:
        p=HERE/name
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        else:json.loads(p.read_text())
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    if prereg['checkpoint_step']!=3000 or tuple(prereg['conditions'])!=CONDITIONS:raise ValueError('prereg drift')
    sources={str(HERE/n):sha256(HERE/n) for n in FROZEN}
    original={str(ROOT/n):sha256(ROOT/n) for n in ('fast_motor.py','evaluate_fast.py','verify_fast_actual.py','EVAL_PREREG.json','PREREG.json')}
    return dict(status='SOURCES_READY_NO_PHYSICS',checkpoint=str(CHECKPOINT),interpreter=str(PYTHON),sources=sources,original_sources=original,max_concurrent_processes=2,cpu_threads_per_process=1,physics_integration_steps=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sha256')
    parser.add_argument('--training-complete',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    ready=readiness()
    if args.prepare_only:
        print(json.dumps(ready,indent=2),flush=True);return
    if not args.training_complete or args.sha256 is None:raise ValueError('explicit --training-complete and final checkpoint --sha256 required')
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise ValueError('exact interpreter required')
    if len(args.sha256)!=64 or sha256(CHECKPOINT)!=args.sha256:raise ValueError('exact final checkpoint SHA mismatch')
    summary_path=HERE/'FINAL3000_SUMMARY.json'
    if summary_path.exists() or any((HERE/('EVAL_'+c)).exists() or (HERE/('VERIFY_ACTUAL_'+c+'.json')).exists() for c in CONDITIONS):raise FileExistsError('fresh pipeline outputs required')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    results={};started=time.time()
    def persist():
        complete=set(results)==set(CONDITIONS)
        dump(summary_path,dict(status='COMPLETE' if complete else 'RUNNING',checkpoint=str(CHECKPOINT),checkpoint_sha256=args.sha256,checkpoint_step=3000,
             all_conditions_eligible=bool(complete and all(r.get('eligible',False) for r in results.values())),
             saved_trajectory_verification=bool(complete and all(r.get('saved_trajectory_verified',False) for r in results.values())),
             conditions={c:results[c] for c in CONDITIONS if c in results},source_freeze=ready,elapsed_s=time.time()-started,
             scope='replacementANN fixed-condition engineering only; no food behavior optimization or full native certification'))
    def command(label,argv):
        log=HERE/(label+'.log');tick=time.time()
        with log.open('x') as handle:
            process=subprocess.run([str(PYTHON),'-B',*map(str,argv)],cwd=str(HERE),env=env,stdout=handle,stderr=subprocess.STDOUT,check=False)
        return dict(argv=[str(PYTHON),'-B',*map(str,argv)],returncode=process.returncode,log=str(log),elapsed_s=time.time()-tick)
    def condition(c):
        out=HERE/('EVAL_'+c);audit=HERE/('VERIFY_ACTUAL_'+c+'.json')
        argv=[HERE/'eval3000.py','--checkpoint',CHECKPOINT,'--sha256',args.sha256,'--condition',c,'--out',out]
        if c.startswith('speed'):argv+=['--neutral-reference',HERE/'EVAL_neutral/RESULT.json']
        execution=command('EVAL_'+c,argv)
        value=dict(condition=c,execution=execution,eligible=False,saved_trajectory_verified=False)
        if execution['returncode']==0:
            auditing=command('AUDIT_'+c,[HERE/'verify_actual3000.py','--run',out,'--out',audit]);value['audit']=auditing
            r=json.loads((out/'RESULT.json').read_text());value.update(result=str(out/'RESULT.json'),result_sha256=sha256(out/'RESULT.json'),checkpoint_sha256=r['checkpoint_sha256'],criteria=r['criteria'],autonomous=r['autonomous'],yaw_change_rad=r['yaw_change_rad'])
            if auditing['returncode']==0:
                a=json.loads(audit.read_text())
                value.update(audit_path=str(audit),audit_sha256=sha256(audit),saved_trajectory_verified=bool(a['integrity_all_pass']),audit_checks=len(a['checks']))
                value['eligible']=bool(a['integrity_all_pass'] and r['engineering_pass'] and a['engineering_pass'] and r['checkpoint_sha256']==args.sha256 and a['checkpoint_sha256']==args.sha256)
        return value
    def group(conditions):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={c:pool.submit(condition,c) for c in conditions}
            for c,future in futures.items():
                try:results[c]=future.result()
                except BaseException as exc:results[c]=dict(condition=c,eligible=False,saved_trajectory_verified=False,error=repr(exc))
                persist();print(json.dumps(dict(condition=c,eligible=results[c]['eligible'],saved_trajectory_verified=results[c]['saved_trajectory_verified'])),flush=True)
    persist()
    group(('neutral',))
    group(('left','right'))
    if results['neutral']['eligible']:group(('speed085','speed090'))
    else:
        for c in ('speed085','speed090'):results[c]=dict(condition=c,eligible=False,saved_trajectory_verified=False,status='NOT_RUN_MATCHED_NEUTRAL_FAILED')
        persist()
    if sha256(CHECKPOINT)!=args.sha256 or readiness()!=ready:raise ValueError('checkpoint or source drift during pipeline')
    print(json.dumps(dict(summary=str(summary_path),all_conditions_eligible=all(r['eligible'] for r in results.values()),saved_trajectory_verification=all(r['saved_trajectory_verified'] for r in results.values()))),flush=True)
    if any('error' in r or r.get('execution',{}).get('returncode',0)!=0 or r.get('audit',{}).get('returncode',0)!=0 for r in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
