from scope import *
from run_body_v2 import run
from metrics import compare
import argparse
p=argparse.ArgumentParser();p.add_argument('split',choices=('dev','held'));p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path,default=WEIGHT);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);rows=[];start=time.monotonic()
for v in ((0,1) if a.split=='dev' else (2,)):
 for f in FAMILIES:
  assert time.monotonic()-start<900
  t=run(f,v,'teacher',a.out/f'{f}_{v}_teacher',a.checkpoint);s=run(f,v,'student',a.out/f'{f}_{v}_student',a.checkpoint)
  rows.append(compare(t,s));write(a.out/'PROGRESS.json',rows)
write(a.out/'RESULT.json',dict(pairs=rows,all_relative=all(r['relative_pass'] for r in rows)))
