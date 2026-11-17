"""Strict minimal inherited-source adapters; called only by parent actual entrypoints."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261111T234150Z'
def replace_once(source,old,new):
 if source.count(old)!=1:raise ValueError('frozen inherited source shape mismatch '+old)
 return source.replace(old,new)
def build(name):
 source=(OLD/name).read_text()
 if name in ('train.py','verify_saved.py'):
  source=re.sub(r'\b(1001|1000|500)\b',lambda m:{'1001':'501','1000':'500','500':'250'}[m.group(0)],source)
  source=source.replace('checkpoint_001000.pt','checkpoint_000500.pt')
 elif name=='evaluate.py':
  source=source.replace('checkpoint_001000.pt','checkpoint_000500.pt').replace('VERIFY_001000.json','VERIFY_000500.json').replace('final1000','final500')
 elif name=='preflight.py':
  source=replace_once(source,"assert all(checks.values()),checks","checks.update(read(HERE/'DATASET.json')['reuse_checks'])\n assert all(checks.values()),checks")
 elif name=='run_candidate.py':
  source=replace_once(source,"('verify_saved.py',['--step','500'],'VERIFY_500'),('verify_saved.py',['--step','1000'],'VERIFY_1000')","('verify_saved.py',['--step','250'],'VERIFY_250'),('verify_saved.py',['--step','500'],'VERIFY_500')")
 else:
  if name!='audit_validation.py':raise ValueError('unregistered inherited executable')
 return source
def run(name):
 namespace={'__name__':'_segmented_actual_'+name.replace('.','_'),'__file__':str(HERE/name)}
 exec(compile(build(name),str(HERE/name)+'::frozen_inherited_'+name,'exec'),namespace)
 namespace['main']()
