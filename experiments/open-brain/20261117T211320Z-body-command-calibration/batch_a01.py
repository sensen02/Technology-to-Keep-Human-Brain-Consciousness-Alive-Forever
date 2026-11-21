from scope import *
from run_body_v3 import run
from metrics_v3 import compare
import argparse
p=argparse.ArgumentParser();p.add_argument('split',choices=('dev','held'));p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path,default=WEIGHT);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rows=[]
for v in ((0,1) if a.split=='dev' else (2,)):
 for side in ('left','right'):
  f='a01_'+side
  t=run(f,v,'teacher',a.out/f'{f}_{v}_teacher',a.checkpoint);s=run(f,v,'student',a.out/f'{f}_{v}_student',a.checkpoint)
  rows.append(compare(t,s));write(a.out/'PROGRESS.json',rows)
write(a.out/'RESULT.json',dict(pairs=rows,all_relative=all(r['relative_pass'] for r in rows),steps=len(rows)*40000))
