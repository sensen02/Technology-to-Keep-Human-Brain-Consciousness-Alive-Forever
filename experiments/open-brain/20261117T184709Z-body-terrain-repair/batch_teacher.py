"""Bounded declared development/formal/replay groups; no parameter adaptation."""
import argparse
from run_teacher import run
from body_model import HERE,write,sha,Path,json

p=argparse.ArgumentParser();p.add_argument('phase',choices=('dev','formal','replay'));args=p.parse_args()
if args.phase=='dev':jobs=[(0.,119),(0.,227),(.15,227),(.30,119),(.30,227)]
elif args.phase=='formal':
    selected=json.loads((HERE/'SELECTION.json').read_text());assert selected['kind']=='terminal'
    jobs=[(h,s) for h in (0.,.15,.30) for s in (383,941)]
else:jobs=[(h,383) for h in (0.,.15,.30)]
config=dict(phase=args.phase,kind='terminal',jobs=jobs,expected_steps=17000*len(jobs),
            code_sha256=sha(Path(__file__)),teacher_sha256=sha(HERE/'run_teacher.py'))
write(HERE/f'BATCH_{args.phase.upper()}.json',config)
results=[]
for height,seed in jobs:
    prefix='flat' if height==0 else 'block015' if height==.15 else 'block030'
    out=HERE/args.phase.upper()/f'terminal_{prefix}_{seed}'
    results.append(run(height,seed,'terminal',out))
    write(HERE/f'{args.phase.upper()}_PROGRESS.json',dict(results=results))
assert config['teacher_sha256']==sha(HERE/'run_teacher.py')
write(HERE/f'{args.phase.upper()}_RESULT.json',dict(results=results,all_pass=all(r['pass_'] for r in results),
             steps=sum(r['steps'] for r in results),teacher_only=True))
