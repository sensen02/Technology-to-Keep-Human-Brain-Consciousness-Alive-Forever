import argparse,time
from common import *
from rollout_repair import run
p=argparse.ArgumentParser();p.add_argument('group',choices=('dev','held','replay'));p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
seeds=(181,263,1151,1723) if a.group=='dev' else (2039,2711,3253) if a.group=='held' else (1723,263)
a.out.mkdir(parents=True,exist_ok=False);write(a.out/'CONFIG.json',dict(group=a.group,seeds=seeds,heights=(0,.15,.30),checkpoint=str(a.checkpoint),sha256=sha(a.checkpoint)))
results=[];start=time.monotonic()
for height in (0.,.15,.30):
 for seed in seeds:
  assert time.monotonic()-start<900
  label='flat' if height==0 else 'block015' if height==.15 else 'block030'
  results.append(run(height,seed,'student',a.out/f'{label}_{seed}',a.checkpoint))
  write(a.out/'PROGRESS.json',results)
write(a.out/'RESULT.json',dict(results=results,all_pass=all(r['pass_'] for r in results),steps=sum(r['steps'] for r in results)))
