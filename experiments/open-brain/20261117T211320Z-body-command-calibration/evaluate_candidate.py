"""Rerun only the student; reuse immutable complete same-body teacher references."""
from scope import *
from run_body_v3 import run
from metrics_v3 import compare
import argparse
p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
rows=[];began=time.monotonic()
for variant in (0,1):
 for family in FAMILIES+('a01_left','a01_right'):
  assert time.monotonic()-began<900
  reference=HERE/('DEV_A01' if family.startswith('a01') else 'DEV')/f'{family}_{variant}_teacher'
  t=json.loads((reference/'RESULT.json').read_text());assert sha(reference/'TRACE.npz')==t['trace_sha256']
  s=run(family,variant,'student',a.out/f'{family}_{variant}_student',a.checkpoint)
  row=compare(t,s);row.update(teacher_reference=str(reference),teacher_result_sha256=sha(reference/'RESULT.json'))
  rows.append(row);write(a.out/'PROGRESS.json',rows)
write(a.out/'RESULT.json',dict(pairs=rows,all_relative=all(r['relative_pass'] for r in rows),steps=len(rows)*20000,teacher_reuse='same immutable full independent baseline episodes, not shadow labels'))
