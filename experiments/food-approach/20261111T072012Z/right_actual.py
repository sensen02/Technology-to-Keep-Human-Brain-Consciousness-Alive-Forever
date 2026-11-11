from pathlib import Path
import sys, shutil, json
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'20261111T043404Z'
sys.path.insert(0,str(OLD))
import evaluate_motor as ev
for name in ('candidate.pt','READINESS.json','EVAL_PREREG.json'):
 shutil.copyfile(OLD/'g04_c00'/name,HERE/name)
a=ev.old.parser().parse_args(['--checkpoint',str(HERE/'candidate.pt'),'--sha256','6a6c23feb9a4b36fcf761f62f9d63309b5c81763f950afe520f42c09a5855e2f','--condition','right','--out',str(HERE/'EVAL_right'),'--training-complete'])
ev.run(a)
import verify_actual
r=verify_actual.verify(str(HERE/'EVAL_right'))
with (HERE/'VERIFY_right.json').open('x') as f:json.dump(ev.jsonable(r),f,indent=2)
assert r['integrity_all_pass'] and all(r['checks'].values())
print(json.dumps(dict(independent_audit=True,checks=len(r['checks']),engineering_pass=r['engineering_pass'])),flush=True)
