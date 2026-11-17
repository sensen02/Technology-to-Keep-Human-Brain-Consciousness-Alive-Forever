"""Singleexactfrozenv1 algorithm adapter; actualcoverage/new1000budget explicit."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent;V1=HERE.parents[1]/'visual_policy_v1'
def once(src,old,new):
 if src.count(old)!=1:raise ValueError('coverage source shape '+old)
 return src.replace(old,new)
def build(name):
 src=(V1/name).read_text()
 if name in ('train.py','verify_saved.py'):
  # Simultaneousmapping avoids cascading250->500->1000 or500->2000.
  src=re.sub(r'\b(501|500|250)\b',lambda m:{'501':'1001','500':'1000','250':'500'}[m.group(0)],src)
  src=src.replace('checkpoint_000500.pt','checkpoint_001000.pt')
 if name=='preflight.py':
  src=once(src,"check('actual112RGB_library',rgb.shape==(112,2,32,32,3))","check('actual616RGB_library',rgb.shape==(616,2,32,32,3))")
  src=once(src,"check('12pairs24unique',sorted(v for pair in table for v in pair)==list(range(24)))","check('360pairs720unique',sorted(v for pair in table for v in pair)==list(range(720)))")
  src=once(src,'torch.manual_seed(20261112);m=VisualPolicy();a=deepcopy(m);','torch.manual_seed(20261112);m,parent=initialize();a=deepcopy(m);')
  src=once(src,'expand([0,4])','expand([0,80])')
  marker=' hashes={str(p):sha(p) for p in sources()+'
  src=once(src,marker,' from extra_checks import extra_checks\n checks.update(extra_checks(m,parent))\n'+marker)
 elif name=='train.py':
  src=once(src,'torch.manual_seed(20261112);m=VisualPolicy();opt=','torch.manual_seed(20261112);m,parent=initialize();opt=')
  src=once(src,'payload=dict(format_version=FORMAT,','payload=dict(format_version=FORMAT,parent_sha256=PARENT_SHA,pair_coverage=pair_coverage(pairs),')
  src=once(src,'gen.choice(12,2,replace=False)','gen.choice(360,2,replace=False)')
  src=once(src,'expand([0,4])','expand([0,80])')
  src=once(src,"status='INITIAL_RANDOM_VISUAL_POLICY_NOT_BEHAVIOR_QUALIFIED'","status='EXACT_PARENT79db_NEW_COVERAGE_NOT_YET_QUALIFIED'")
  src=once(src,"status='TRAINED_NOT_VALIDATED_OR_PHYSICAL',","status='TRAINED_NOT_VALIDATED_OR_PHYSICAL',pair_coverage=pair_coverage(pairs),")
 elif name=='verify_saved.py':
  src=once(src,'from data import *','from data import *')
  src=once(src,'generator.choice(12,2,replace=False)','generator.choice(360,2,replace=False)')
  src=once(src,"checks['pair_table_exact']=p['pair_table']==pair_table();","checks['pair_table_exact']=p['pair_table']==pair_table();checks['actual_pair_coverage_counts_exact']=p['pair_coverage']==pair_coverage(ids);")
 elif name=='evaluate.py':
  src=src.replace('checkpoint_000500.pt','checkpoint_001000.pt').replace('VERIFY_000500.json','VERIFY_001000.json').replace('final500','final1000')
 elif name=='run_candidate.py':
  src=once(src,"('verify_saved.py',['--step','250'],'VERIFY_250'),('verify_saved.py',['--step','500'],'VERIFY_500')","('verify_saved.py',['--step','500'],'VERIFY_500'),('verify_saved.py',['--step','1000'],'VERIFY_1000')")
 elif name!='audit_validation.py':raise ValueError('unregisteredentry')
 return src
def run(name):
 ns={'__name__':'_coverage_frozen_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(build(name),str(HERE/name)+'::explicit1000coverage_adapter','exec'),ns);ns['main']()
