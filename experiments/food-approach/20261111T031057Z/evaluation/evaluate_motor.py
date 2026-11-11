"""Fixed8000 weighted-readout counterpart from immutable positive-head evaluator."""
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
 if hashlib.sha256(p.read_bytes()).hexdigest()!=FROZEN[name]:raise ValueError('immutable predecessor drift '+name)
 return p.read_text()


# Only fixed-step/path metadata, source capture and pre-Core artifact checks differ.
# The measured c6b1500 parent-right reference remains the original qualified motor.
source=predecessor_source('evaluate_motor.py').replace('2000','8000')
source=source.replace("kind='fixed8000 positive-neural-readout actual physical evaluation'","kind='fixed8000 RH-load-weighted positive-neural-readout actual physical evaluation'")
old="return list(dict.fromkeys(original_paths()+parent_paths()+[Path(p) for p in prereg['frozen_primary_sources']]))"
new="paths=list(dict.fromkeys(original_paths()+parent_paths()+[Path(p) for p in prereg['frozen_primary_sources']]))\n        for path in paths:\n            if not path.is_file():raise FileNotFoundError('pre-Core source path missing '+str(path))\n        return paths"
if source.count(old)!=1:raise ValueError('predecessor source path contract changed')
source=source.replace(old,new)
# Bind all authenticated training inputs, including the before-action loss mask,
# into actual run identity; these inputs are never appended to actor observations.
_module=types.ModuleType('_weighted8000_positive_readout_evaluator')
_module.__dict__.update(__file__=str(HERE/'evaluate_motor.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'evaluate_motor.py')+'::pinned-predecessor','exec'),_module.__dict__)
_original_derived=_module.derived_source

def derived_source():
 text=_original_derived()
 old="extra_paths=paths+[Path(args.checkpoint).resolve()])"
 new="extra_paths=paths+[Path(args.checkpoint).resolve()]+[Path(p) for p in payload['config']['source_hashes']]+[Path(p) for p in payload['config']['data_hashes']])"
 if text.count(old)!=1:raise ValueError('identity capture contract changed')
 return text.replace(old,new)

# Rebuild evaluate with identical physics and score using the expanded identity.
rebuilt=types.ModuleType('_weighted8000_physical_body')
rebuilt.__dict__.update(__file__=str(HERE/'evaluate_motor.py'),__name__=rebuilt.__name__)
exec(compile(derived_source(),str(HERE/'evaluate_motor.py')+'::frozen-body','exec'),rebuilt.__dict__)
# Reuse the fully derived source-path and right-preservation functions; their
# own immutable globals point to this same new ROOT through the first binding.
rebuilt.source_paths=_module.source_paths
rebuilt.verify_parent_right=_module.verify_parent_right
_original_checked=rebuilt.checked_load

def checked_load(io,path,digest):
 model,payload=_original_checked(io,path,digest)
 for category in ('source_hashes','data_hashes'):
  for filename in payload['config'][category]:
   if not Path(filename).is_file():raise FileNotFoundError('authenticated training artifact missing '+filename)
 return model,payload

rebuilt.checked_load=checked_load
for name in ('ROOT','OLD','PRIMARY','TRAIN','CHECKPOINT','PARENT_CHECKPOINT','PARENT_SHA','PARENT_RIGHT','PARENT_AUDIT','verify_primary','verify_parent_right','parent_paths','replace_once','sha256','arrays'):
 if hasattr(_module,name):globals()[name]=getattr(_module,name)
for name in ('BASE','ORIGINAL','COMMANDS','HEAD_AUDIT','BASE_CHECKPOINT','jsonable','command_at','score','pin','flatten','parser','evaluate','source_paths'):
 globals()[name]=getattr(rebuilt,name)

if __name__=='__main__':evaluate(parser().parse_args())
