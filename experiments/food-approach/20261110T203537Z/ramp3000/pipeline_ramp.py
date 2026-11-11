"""Prepared4case matchedramp diagnostic; CPU1 processes, at most2 concurrent.
No automatic launch; parent must finish current step tests and explicitly authorize.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
from eval_ramp import HERE,CONT,COMMANDS,sha256

PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
CHECKPOINT=CONT/'TRAIN/checkpoint_003000.pt'


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sha256');p.add_argument('--neutral-reference');p.add_argument('--neutral-audit');p.add_argument('--launch-authorized',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    ready=json.loads((HERE/'READINESS.json').read_text())
    for path,digest in ready['frozen_sources'].items():
        if sha256(path)!=digest:raise ValueError('readiness source drift '+path)
    if a.prepare_only:print(json.dumps(ready,indent=2));return
    if not a.launch_authorized or not a.sha256 or not a.neutral_reference or not a.neutral_audit:raise ValueError('explicit parent launch, SHA, neutral result and independent audit required')
    if sha256(CHECKPOINT)!=a.sha256:raise ValueError('checkpoint digest mismatch')
    for path in (a.neutral_reference,a.neutral_audit):
        if not Path(path).is_absolute() or not Path(path).is_file():raise ValueError('existing absolute neutral paths required')
    summary=HERE/'RAMP3000_SUMMARY.json'
    if summary.exists() or any((HERE/('EVAL_'+c)).exists() or (HERE/('VERIFY_'+c+'.json')).exists() for c in COMMANDS):raise FileExistsError('fresh outputs required')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    started=time.time();results={}
    def command(label,argv):
        with (HERE/(label+'.log')).open('x') as f:r=subprocess.run([str(PYTHON),'-B',*map(str,argv)],env=env,cwd=str(HERE),stdout=f,stderr=subprocess.STDOUT)
        return dict(returncode=r.returncode,argv=[str(x) for x in argv],log=str(HERE/(label+'.log')))
    def case(condition):
        out=HERE/('EVAL_'+condition);audit=HERE/('VERIFY_'+condition+'.json')
        execute=command('EVAL_'+condition,[HERE/'eval_ramp.py','--checkpoint',CHECKPOINT,'--sha256',a.sha256,'--condition',condition,'--out',out,'--neutral-reference',a.neutral_reference,'--neutral-audit',a.neutral_audit,'--launch-authorized'])
        result=dict(execution=execute,integrity_all_pass=False,engineering_pass=False)
        if execute['returncode']==0:
            verification=command('VERIFY_'+condition,[HERE/'verify_ramp.py','--run',out,'--out',audit]);result['verification']=verification
            r=json.loads((out/'RESULT.json').read_text());result.update(result=str(out/'RESULT.json'),result_sha256=sha256(out/'RESULT.json'),criteria=r['criteria'],autonomous=r['autonomous'],yaw_change_rad=r['yaw_change_rad'])
            if verification['returncode']==0:
                v=json.loads(audit.read_text());result.update(integrity_all_pass=v['integrity_all_pass'],engineering_pass=bool(v['engineering_pass'] and r['engineering_pass']),audit=str(audit),audit_sha256=sha256(audit))
        return result
    def persist():
        dump(summary,dict(status='COMPLETE' if len(results)==4 else 'RUNNING',checkpoint=str(CHECKPOINT),checkpoint_sha256=a.sha256,conditions=results,
             diagnostic_only=True,food_qualification=False,all_original_engineering_conditions_pass=bool(len(results)==4 and all(r['engineering_pass'] and r['integrity_all_pass'] for r in results.values())),
             matched_neutral_result=a.neutral_reference,matched_neutral_audit=a.neutral_audit,readiness=ready,elapsed_s=time.time()-started))
    persist()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(case,c):c for c in COMMANDS}
        for future in as_completed(futures):
            c=futures[future]
            try:results[c]=future.result()
            except BaseException as exc:results[c]=dict(error=repr(exc),integrity_all_pass=False,engineering_pass=False)
            persist();print(json.dumps(dict(condition=c,engineering_pass=results[c]['engineering_pass'],integrity=results[c]['integrity_all_pass'])),flush=True)
    for path,digest in ready['frozen_sources'].items():
        if sha256(path)!=digest:raise ValueError('source changed during diagnostic')
    if any('error' in r or r['execution']['returncode']!=0 or r.get('verification',{}).get('returncode',0)!=0 for r in results.values()):raise SystemExit(1)

if __name__=='__main__':main()
