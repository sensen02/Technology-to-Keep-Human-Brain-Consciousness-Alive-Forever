"""Fixed8000 three-condition runner; no execution without explicit final SHA authorization."""
from pathlib import Path
import types
import evaluate_motor as current
HERE=Path(__file__).resolve().parent
text=current.predecessor_source('pipeline.py').replace('2000','8000')
_module=types.ModuleType('_weighted8000_actual_pipeline')
_module.__dict__.update(__file__=str(HERE/'pipeline.py'),__name__=_module.__name__)
exec(compile(text,str(HERE/'pipeline.py')+'::pinned-predecessor','exec'),_module.__dict__)
main=_module.main
readiness=_module.readiness
if __name__=='__main__':main()
