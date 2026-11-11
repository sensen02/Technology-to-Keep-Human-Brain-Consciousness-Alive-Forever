"""Final10000 counterpart from SHA-pinned positive-head evaluator; no physics on import."""
from pathlib import Path
import hashlib
import types

HERE=Path(__file__).resolve().parent
PREDECESSOR=HERE.parent.parent/'20261111T010218Z/evaluation'
FROZEN_PREDECESSOR={
 'evaluate_motor.py':'254c25cc8123ff1e8238f39931293296cb6d5e068557ef9a2fa20b1a8eb1a0d3',
 'verify_actual.py':'ff5ca02d2285b79a3bbf2bc5a9b7c91987830af51163bb2734e96a8a1cf9e644',
 'pipeline.py':'9231625880f903e0fd5229289f56fee85474897e1ad5081190b904befc310cf9'}

def predecessor_source(name):
    path=PREDECESSOR/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_PREDECESSOR[name]:raise ValueError('frozen predecessor drift '+name)
    return path.read_text()

# The predecessor explicitly fixes right preservation to the original qualified
# c6b...1500 ancestor. Never substitute continuation cfg.parent_checkpoint here.
source=predecessor_source('evaluate_motor.py').replace('2000','10000').replace('checkpoint_0010000.pt','checkpoint_010000.pt')
_module=types.ModuleType('_final10000_positive_readout_evaluator')
_module.__dict__.update(__file__=str(HERE/'evaluate_motor.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'evaluate_motor.py')+'::pinned-predecessor','exec'),_module.__dict__)
for _name in ('ROOT','OLD','PRIMARY','TRAIN','CHECKPOINT','PARENT_CHECKPOINT','PARENT_SHA','PARENT_RIGHT','PARENT_AUDIT','BASE','ORIGINAL','COMMANDS','HEAD_AUDIT','BASE_CHECKPOINT','jsonable','command_at','score','pin','flatten','checked_load','parser','evaluate','source_paths','verify_primary','verify_parent_right','parent_paths','replace_once','derived_source','sha256','arrays'):
    globals()[_name]=getattr(_module,_name)

if __name__=='__main__':evaluate(parser().parse_args())
