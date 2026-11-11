"""Parent-authorized fixed1500 three-condition evaluation and saved-actual audits."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from evaluate_motor import HERE,CHECKPOINT,COMMANDS,sha256


def readiness():
    files=[HERE/n for n in ('pipeline.py','evaluate_motor.py','verify_actual.py','EVAL_PREREG.json','check_prepared.py')]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        else:json.loads(p.read_text())
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    if prereg['conditions']!=COMMANDS or prereg['checkpoint_step']!=1500:raise ValueError('fixed3 prereg drift')
    return dict(status='CODE_READY_NO_PHYSICAL_EXECUTION',source_hashes={str(p):sha256(p) for p in files},conditions=list(COMMANDS),checkpoint_step=1500,max_concurrent_processes=2,cpu_threads_per_process=1,physics_integration_steps=0)


def dump(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sha256');p.add_argument('--training-complete',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    ready=readiness()
    if a.prepare_only:print(json.dumps(ready,indent=2));return
    if not a.training_complete or not a.sha256 or len(a.sha256)!=64:raise ValueError('explicit parent --training-complete and final SHA required')
    preparation=json.loads((HERE/'READINESS.json').read_text())
    if not preparation['all_passed']:raise ValueError('offline preparation checks required')
    for path,digest in preparation['source_hashes'].items():
        if sha256(path)!=digest:raise ValueError('offline checked source drift '+path)
    ready['offline_readiness_sha256']=sha256(HERE/'READINESS.json')
    if sha256(CHECKPOINT)!=a.sha256:raise ValueError('final selected checkpoint digest mismatch')
    summary=HERE/'FINAL_SUMMARY.json'
    if summary.exists() or any((HERE/('EVAL_'+c)).exists() or (HERE/('VERIFY_'+c+'.json')).exists() for c in COMMANDS):raise FileExistsError('fresh outputs required')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    results={};started=time.monotonic()
    def persist():
        complete=len(results)==3
        dump(summary,dict(status='COMPLETE' if complete else 'RUNNING',checkpoint=str(CHECKPOINT),checkpoint_sha256=a.sha256,conditions=results,
             all_conditions_eligible=bool(complete and all(r.get('engineering_pass',False) and r.get('integrity_all_pass',False) for r in results.values())),
             saved_trajectory_verification=bool(complete and all(r.get('integrity_all_pass',False) for r in results.values())),readiness=ready,elapsed_s=time.monotonic()-started,
             scope='three fixed conditions replacement ANN/derived body engineering; no speed/food/fullnative certification'))
    def command(label,argv):
        log=HERE/(label+'.log')
        with log.open('x') as stream:rc=subprocess.run([sys.executable,'-B',*map(str,argv)],env=env,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
        return dict(returncode=rc,argv=[sys.executable,'-B',*map(str,argv)],log=str(log))
    def one(c):
        out=HERE/('EVAL_'+c);audit=HERE/('VERIFY_'+c+'.json')
        execution=command('EVAL_'+c,[HERE/'evaluate_motor.py','--checkpoint',CHECKPOINT,'--sha256',a.sha256,'--condition',c,'--out',out,'--training-complete'])
        value=dict(engineering_pass=False,integrity_all_pass=False,execution=execution)
        if execution['returncode']==0:
            auditing=command('VERIFY_'+c,[HERE/'verify_actual.py','--run',out,'--out',audit]);value['verification']=auditing
            r=json.loads((out/'RESULT.json').read_text());value.update(result=str(out/'RESULT.json'),result_sha256=sha256(out/'RESULT.json'),criteria=r['criteria'],autonomous=r['autonomous'],cycles=r['autonomous_cycles']['complete_per_leg'],yaw_change_rad=r['yaw_change_rad'])
            if auditing['returncode']==0:
                v=json.loads(audit.read_text());value.update(engineering_pass=bool(r['engineering_pass'] and v['engineering_pass']),integrity_all_pass=bool(v['integrity_all_pass']),audit=str(audit),audit_sha256=sha256(audit),audit_checks=len(v['checks']))
        return value
    persist();results['neutral']=one('neutral');persist();print(json.dumps(dict(condition='neutral',engineering_pass=results['neutral']['engineering_pass'],integrity=results['neutral']['integrity_all_pass'])),flush=True)
    if results['neutral']['engineering_pass'] and results['neutral']['integrity_all_pass']:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(one,c):c for c in ('left','right')}
            for f in as_completed(futures):
                c=futures[f]
                try:results[c]=f.result()
                except BaseException as exc:results[c]=dict(error=repr(exc),engineering_pass=False,integrity_all_pass=False)
                persist();print(json.dumps(dict(condition=c,engineering_pass=results[c]['engineering_pass'],integrity=results[c]['integrity_all_pass'])),flush=True)
    else:
        for c in ('left','right'):results[c]=dict(engineering_pass=False,integrity_all_pass=False,status='NOT_RUN_NEUTRAL_FAILED')
        persist()
    for path,h in ready['source_hashes'].items():
        if sha256(path)!=h:raise ValueError('evaluation source drift '+path)
    if sha256(CHECKPOINT)!=a.sha256:raise ValueError('checkpoint drift')
    if any('error' in r or r.get('execution',{}).get('returncode',0)!=0 or r.get('verification',{}).get('returncode',0)!=0 for r in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
