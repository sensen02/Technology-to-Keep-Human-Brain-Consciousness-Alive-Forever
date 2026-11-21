from scope import *
from run_body import run
from metrics import compare
import argparse
p=argparse.ArgumentParser();p.add_argument('split',choices=('dev','held','replay'));p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path,default=WEIGHT);a=p.parse_args()
variants=(0,1) if a.split=='dev' else (2,);a.out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();rows=[]
for variant in variants:
 for family in FAMILIES:
  assert time.monotonic()-start<900
  t=run(family,variant,'teacher',a.out/f'{family}_{variant}_teacher',a.checkpoint)
  s=run(family,variant,'student',a.out/f'{family}_{variant}_student',a.checkpoint)
  pair=compare(t,s);rows.append(pair);write(a.out/'PROGRESS.json',rows)
write(a.out/'RESULT.json',dict(pairs=rows,all_relative=all(r['relative_pass'] for r in rows),steps=len(rows)*40000))
