"""Fixed final5000 three-condition execution, only after explicit parent launch."""
from pathlib import Path
import types
from evaluate_motor import predecessor_source
HERE=Path(__file__).resolve().parent
source=predecessor_source('pipeline.py').replace('2000','5000').replace('positive neural readout','nonlinear positive neural readout')
_module=types.ModuleType('_nonlinear_positive_three_conditions')
_module.__dict__.update(__file__=str(HERE/'pipeline.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'pipeline.py')+'::immutable-derived','exec'),_module.__dict__)
main=_module.main
readiness=_module.readiness
if __name__=='__main__':main()
