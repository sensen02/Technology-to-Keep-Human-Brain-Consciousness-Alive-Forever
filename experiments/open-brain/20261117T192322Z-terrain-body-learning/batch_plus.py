import argparse
import time
from shared import *
from rollout_plus import run
p=argparse.ArgumentParser();p.add_argument('group',choices=('train','dev_teacher','dev','held','dagger','replay'))
p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--beta',type=float,default=0.)
a=p.parse_args()
seeds=TRAIN_SEEDS if a.group=='train' else DEV_SEEDS if a.group in ('dev_teacher','dev','dagger') else HELD_SEEDS
if a.group=='replay':seeds=HELD_SEEDS[:1]
mode='teacher' if a.group in ('train','dev_teacher') else 'dagger' if a.group=='dagger' else 'student'
a.out.mkdir(exist_ok=False,parents=True)
config=dict(group=a.group,heights=HEIGHTS,seeds=seeds,mode=mode,beta=a.beta,checkpoint=str(a.checkpoint),
 source_sha256={str(HERE/n):sha(HERE/n) for n in ('batch_plus.py','shared.py','rollout_plus.py','policy.py','policy_plus.py','proprioception.py')})
write(a.out/'CONFIG.json',config)
results=[];started=time.monotonic()
for height in HEIGHTS:
 for seed in seeds:
    if time.monotonic()-started>900:raise TimeoutError('bounded batch budget')
    label='flat' if height==0 else 'block015' if height==.15 else 'block030'
    results.append(run(height,seed,mode,a.out/f'{label}_{seed}',a.checkpoint,a.beta))
    write(a.out/'PROGRESS.json',dict(results=results))
write(a.out/'RESULT.json',dict(results=results,all_pass=all(r['pass_'] for r in results),steps=sum(r['steps'] for r in results)))
