"""Own strict three-condition pipeline; frozen primary orchestration reused explicitly."""
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
if 'evaluate_motor' in sys.modules and Path(sys.modules['evaluate_motor'].__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('conflicting primary evaluator import')
sys.path.insert(0,str(HERE))
import evaluate_motor as control
if Path(control.__file__).resolve()!=HERE/'evaluate_motor.py':raise ValueError('wrong control evaluator')
control.verify_primary()
_MODULE=types.ModuleType('_two_only_physical_pipeline')
_MODULE.__dict__.update(__file__=str(HERE/'pipeline.py'),__name__=_MODULE.__name__)
exec(compile((control.PRIMARY/'pipeline.py').read_text(),str(HERE/'pipeline.py')+'::frozen-derived','exec'),_MODULE.__dict__)
readiness=_MODULE.readiness
main=_MODULE.main

if __name__=='__main__':main()
