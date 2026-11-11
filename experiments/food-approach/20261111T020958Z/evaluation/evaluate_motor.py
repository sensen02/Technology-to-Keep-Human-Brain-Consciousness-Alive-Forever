"""Fixed5000 nonlinear readout evaluation, SHA-pinned physical/scoring predecessor."""
from pathlib import Path
import hashlib
import types

HERE=Path(__file__).resolve().parent
PREDECESSOR=HERE.parent.parent/'20261111T010218Z/evaluation'
FROZEN={
 'evaluate_motor.py':'254c25cc8123ff1e8238f39931293296cb6d5e068557ef9a2fa20b1a8eb1a0d3',
 'verify_actual.py':'ff5ca02d2285b79a3bbf2bc5a9b7c91987830af51163bb2734e96a8a1cf9e644',
 'pipeline.py':'9231625880f903e0fd5229289f56fee85474897e1ad5081190b904befc310cf9'}

def predecessor_source(name):
 p=PREDECESSOR/name
 if hashlib.sha256(p.read_bytes()).hexdigest()!=FROZEN[name]:raise ValueError('immutable predecessor source drift '+name)
 return p.read_text()

# Rebinding __file__ explicitly fixes new HERE/ROOT/TRAIN while the c6b1500
# passed-right reference remains the original, separately identified ancestor.
source=predecessor_source('evaluate_motor.py').replace('2000','5000')
source=source.replace("kind='fixed5000 positive-neural-readout actual physical evaluation'","kind='fixed5000 nonlinear-positive-neural-readout actual physical evaluation'")
old="return list(dict.fromkeys(original_paths()+parent_paths()+[Path(p) for p in prereg['frozen_primary_sources']]))"
new="paths=list(dict.fromkeys(original_paths()+parent_paths()+[Path(p) for p in prereg['frozen_primary_sources']]))\n        for path in paths:\n            if not path.is_file():raise FileNotFoundError('pre-Core source path missing '+str(path))\n        return paths"
if source.count(old)!=1:raise ValueError('predecessor source_paths shape changed')
source=source.replace(old,new)
_module=types.ModuleType('_nonlinear_positive_actual_evaluation')
_module.__dict__.update(__file__=str(HERE/'evaluate_motor.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'evaluate_motor.py')+'::immutable-derived','exec'),_module.__dict__)
for name in ('ROOT','OLD','PRIMARY','TRAIN','CHECKPOINT','PARENT_CHECKPOINT','PARENT_SHA','PARENT_RIGHT','PARENT_AUDIT','BASE','ORIGINAL','COMMANDS','HEAD_AUDIT','BASE_CHECKPOINT','jsonable','command_at','score','pin','flatten','checked_load','parser','evaluate','source_paths','verify_primary','verify_parent_right','parent_paths','replace_once','derived_source','sha256','arrays'):
 globals()[name]=getattr(_module,name)

if __name__=='__main__':evaluate(parser().parse_args())
