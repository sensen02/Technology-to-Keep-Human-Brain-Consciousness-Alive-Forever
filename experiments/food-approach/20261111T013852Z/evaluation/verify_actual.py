"""Independent final10000 actual audit with immutable c6b...1500 right comparator."""
from pathlib import Path
import sys
import types
HERE=Path(__file__).resolve().parent
if 'evaluate_motor' in sys.modules and Path(sys.modules['evaluate_motor'].__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('conflicting evaluator import')
sys.path.insert(0,str(HERE))
import evaluate_motor as current
source=current.predecessor_source('verify_actual.py').replace('2000','10000')
_module=types.ModuleType('_final10000_actual_audit')
_module.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'verify_actual.py')+'::pinned-predecessor','exec'),_module.__dict__)
derived_source=_module.derived_source
verify=_module.verify
main=_module.main
if __name__=='__main__':main()
