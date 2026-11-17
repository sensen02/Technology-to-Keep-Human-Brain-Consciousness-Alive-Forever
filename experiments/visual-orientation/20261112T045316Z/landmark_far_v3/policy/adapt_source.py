"""One exact adapter over frozen v1 algorithms; far-domain data/lineage remain explicit."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;V1=HERE.parents[1]/'visual_policy_v1'
def once(source,old,new):
 if source.count(old)!=1:raise ValueError('far source adaptation mismatch '+old)
 return source.replace(old,new)
def build(name):
 src=(V1/name).read_text()
 if name=='preflight.py':
  src=once(src,"check('actual112RGB_library',rgb.shape==(112,2,32,32,3))","check('actual42_farRGB_library',rgb.shape==(42,2,32,32,3))")
  src=once(src,"check('12pairs24unique',sorted(v for pair in table for v in pair)==list(range(24)))","check('18pairs36unique',sorted(v for pair in table for v in pair)==list(range(36)))")
  src=once(src,'torch.manual_seed(20261112);m=VisualPolicy();a=deepcopy(m);','torch.manual_seed(20261112);m,parent=initialize();a=deepcopy(m);')
  marker=' hashes={str(p):sha(p) for p in sources()+'
  src=once(src,marker,' from extra_checks import extra_checks\n checks.update(extra_checks(m,parent))\n'+marker)
 elif name=='train.py':
  src=once(src,'torch.manual_seed(20261112);m=VisualPolicy();opt=','torch.manual_seed(20261112);m,parent=initialize();opt=')
  src=once(src,'payload=dict(format_version=FORMAT,','payload=dict(format_version=FORMAT,parent_sha256=PARENT_SHA,')
  src=once(src,'gen.choice(12,2,replace=False)','gen.choice(18,2,replace=False)')
  src=once(src,"status='INITIAL_RANDOM_VISUAL_POLICY_NOT_BEHAVIOR_QUALIFIED'","status='EXACT_PARENT79db_FAR_DOMAIN_NOT_YET_QUALIFIED'")
 elif name=='verify_saved.py':src=once(src,'generator.choice(12,2,replace=False)','generator.choice(18,2,replace=False)')
 elif name not in ('evaluate.py','audit_validation.py','run_candidate.py'):raise ValueError('unregistered inherited entry')
 return src
def run(name):
 ns={'__name__':'_far_visual_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(build(name),str(HERE/name)+'::exact_frozen_v1_adapter','exec'),ns);ns['main']()
