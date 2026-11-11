"""Authorized fixed budget orchestration; run only after CACHE_CHECKS pass."""
import json
from pathlib import Path
import subprocess
import sys
from residual_motor import HERE,sha256


def main():
    out=HERE/'TRAIN_RESIDUAL_500'
    with (HERE/'TRAIN_FIXED500.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(HERE/'cached_training.py'),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    ck=out/'checkpoint_000500.pt';results={}
    for condition in ('neutral','left','right'):
        target=HERE/('EVAL_RESIDUAL500_'+condition)
        with (HERE/('EVAL_'+condition+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-B',str(HERE/'evaluate_candidate.py'),'--checkpoint',str(ck),'--expected-sha256',sha256(ck),'--out',str(target),'--condition',condition],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((target/'RESULT.json').read_text());results[condition]=dict(candidate_engineering_pass=result['candidate_engineering_pass'],slip=result['autonomous']['slip_ratio_max'],yaw=result['yaw_change_rad'],gates=result['autonomous']['gates'],cycles=result['autonomous_cycles'])
        print(json.dumps(dict(condition=condition,result=results[condition])),flush=True)
    report=dict(all3_pass=all(r['candidate_engineering_pass'] for r in results.values()),checkpoint=str(ck),checkpoint_sha256=sha256(ck),results=results,scope='fixed500 alternate candidate; no behavior learning, no extension, no native certification')
    (HERE/'FIXED500_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
