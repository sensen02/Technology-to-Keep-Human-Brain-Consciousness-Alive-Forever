"""Own scoped independent auditor, reusing frozen primary audit equations unchanged."""
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
if 'evaluate_motor' in sys.modules and Path(sys.modules['evaluate_motor'].__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('conflicting primary evaluator import')
sys.path.insert(0,str(HERE))
import evaluate_motor as control
if Path(control.__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('wrong control evaluator')
control.verify_primary()
_MODULE=types.ModuleType('_two_only_saved_actual_audit')
_MODULE.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=_MODULE.__name__)
exec(compile((control.PRIMARY/'verify_actual.py').read_text(),str(HERE/'verify_actual.py')+'::frozen-derived','exec'),_MODULE.__dict__)
verify=_MODULE.verify
main=_MODULE.main

if __name__=='__main__':main()
