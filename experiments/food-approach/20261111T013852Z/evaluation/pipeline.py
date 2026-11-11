"""Final10000 three-condition pipeline; guarded, no physical launch on import."""
from pathlib import Path
import types
from evaluate_motor import predecessor_source
HERE=Path(__file__).resolve().parent
source=predecessor_source('pipeline.py').replace('2000','10000')
_module=types.ModuleType('_final10000_pipeline')
_module.__dict__.update(__file__=str(HERE/'pipeline.py'),__name__=_module.__name__)
exec(compile(source,str(HERE/'pipeline.py')+'::pinned-predecessor','exec'),_module.__dict__)
readiness=_module.readiness
main=_module.main
if __name__=='__main__':main()
