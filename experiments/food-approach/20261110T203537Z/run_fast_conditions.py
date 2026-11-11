"""Managed neutral then four condition processes (CPU1 each,max2 concurrently)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from fast_motor import HERE,sha256


def command(argv,label):
    print(json.dumps(dict(phase=label,command=[str(x) for x in argv])),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (HERE/(label+'.log')).open('x') as stream:
        rc=subprocess.run([sys.executable,'-B']+[str(x) for x in argv],stdout=stream,stderr=subprocess.STDOUT,env=env).returncode
    print(json.dumps(dict(phase=label,exit_code=rc)),flush=True)
    return rc


def main(args):
    ck=Path(args.checkpoint).resolve();digest=sha256(ck)
    def one(condition):
        out=HERE/f'EVAL_FAST_{condition}'
        argv=[HERE/'evaluate_fast.py','--checkpoint',ck,'--sha256',digest,'--condition',condition,'--out',out]
        if condition.startswith('speed'):argv+=['--neutral-reference',HERE/'EVAL_FAST_neutral/RESULT.json']
        rc=command(argv,'eval_fast_'+condition)
        if rc:return dict(condition=condition,path=str(out),exit_code=rc,eligible=False)
        vp=HERE/f'VERIFY_ACTUAL_{condition}.json'
        verify_rc=command([HERE/'verify_fast_actual.py','--run',out,'--out',vp],'verify_actual_'+condition)
        r=json.loads((out/'RESULT.json').read_text())
        return dict(condition=condition,path=str(out),exit_code=rc,verification_exit_code=verify_rc,
                    eligible=bool(r['engineering_pass'] and verify_rc==0),autonomous=r['autonomous'],yaw_change_rad=r['yaw_change_rad'],
                    actual_verification=str(vp),result_sha256=sha256(out/'RESULT.json'))
    results=[one('neutral')]
    if results[0]['eligible']:
        with ThreadPoolExecutor(max_workers=2) as pool:results+=list(pool.map(one,['left','right','speed085','speed090']))
    else:
        print(json.dumps(dict(skipped='speed reference requires qualifiedneutral; no additional configurations launched')),flush=True)
    summary=dict(checkpoint=str(ck),checkpoint_sha256=digest,conditions=results,
                 all_conditions_eligible=len(results)==5 and all(r['eligible'] for r in results),
                 scope='fixed5 actual physical engineering conditions + independent savedarrayverification; not food or fullnativecertification')
    (HERE/'FAST_CONDITIONS_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(all_conditions_eligible=summary['all_conditions_eligible'],conditions=[dict(condition=r['condition'],eligible=r['eligible']) for r in results])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);main(p.parse_args())
