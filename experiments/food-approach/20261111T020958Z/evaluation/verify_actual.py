"""Independent saved actual5000 audit; no Core construction or integration."""
from pathlib import Path
import types
import evaluate_motor as current
HERE=Path(__file__).resolve().parent
if Path(current.__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('wrong current evaluator import')
source=current.predecessor_source('verify_actual.py').replace('actual2000','actual5000')
_module=types.ModuleType('_nonlinear_positive_saved_actual_audit')
_module.__dict__.update(__file__=str(HERE/'verify_actual.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'verify_actual.py')+'::immutable-derived','exec'),_module.__dict__)
verify=_module.verify
main=_module.main
if __name__=='__main__':main()
