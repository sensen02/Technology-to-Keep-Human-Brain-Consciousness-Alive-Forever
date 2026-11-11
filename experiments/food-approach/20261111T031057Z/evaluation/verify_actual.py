"""Independent fixed8000 actual checkpoint/trajectory audit; no Core construction."""
from pathlib import Path
import sys
import types
HERE=Path(__file__).resolve().parent
if 'evaluate_motor' in sys.modules and Path(sys.modules['evaluate_motor'].__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('conflicting evaluator import')
sys.path.insert(0,str(HERE))
import evaluate_motor as current
if Path(current.__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('wrong evaluator module')


def derived_source():
 current.verify_primary();source=(current.PRIMARY/'verify_actual.py').read_text()
 old='from evaluate_motor import HERE,BASE,ORIGINAL,COMMANDS,pin,checked_load,flatten,command_at,score,jsonable,sha256'
 source=current.replace_once(source,old,old+',verify_parent_right')
 old="    if r.get('native') is not None:"
 new="    if r['condition']=='right':\n        right_comparison=verify_parent_right(t,d,g,ident,identity,check)\n        check('right_parent_preservation_bound',right_comparison==r['right_parent_preservation'])\n    if r.get('native') is not None:"
 return current.replace_once(source,old,new)

_MODULE=types.ModuleType('_weighted8000_actual_audit')
_MODULE.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=_MODULE.__name__)
exec(compile(derived_source(),str(HERE/'verify_actual.py')+'::frozen-derived','exec'),_MODULE.__dict__)
verify=_MODULE.verify
main=_MODULE.main
if __name__=='__main__':main()
