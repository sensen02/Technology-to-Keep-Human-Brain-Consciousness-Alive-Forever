"""Frozen teacher qualification with richer read-only sensory/force recording."""
import argparse
from body_model import *
from run_teacher_final import run

p=argparse.ArgumentParser();p.add_argument('phase',choices=('parity','formal','replay'));args=p.parse_args()
if args.phase=='parity':jobs=[(.15,119)]
elif args.phase=='formal':
    chosen=json.loads((HERE/'SELECTION.json').read_text());assert chosen['kind']=='terminal'
    jobs=[(h,s) for h in (0.,.15,.30) for s in (383,941)]
else:jobs=[(h,383) for h in (0.,.15,.30)]
config=dict(phase=args.phase,jobs=jobs,teacher_kind='terminal',steps=17000*len(jobs),
       source_sha256={str(HERE/n):sha(HERE/n) for n in ('qualify.py','run_teacher_final.py','pad_logging.py','body_model.py','JOINT_LIMITS.json','CANONICAL_INIT.npz')})
write(HERE/f'QUALIFY_{args.phase.upper()}.json',config)
results=[]
for height,seed in jobs:
    label='flat' if height==0 else 'block015' if height==.15 else 'block030'
    results.append(run(height,seed,'terminal',HERE/args.phase.upper()/f'terminal_{label}_{seed}'))
assert all(sha(p)==v for p,v in config['source_sha256'].items())
write(HERE/f'{args.phase.upper()}_RESULT.json',dict(results=results,all_pass=all(r['pass_'] for r in results),steps=sum(r['steps'] for r in results)))
