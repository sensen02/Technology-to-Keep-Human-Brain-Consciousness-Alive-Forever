"""Fresh-path recovery only: frozen physical/scoring evaluator with explicit namespace."""
from pathlib import Path
import importlib.util,types,json
HERE=Path(__file__).resolve().parent
ORIGINAL_EVAL=HERE.parent/'evaluation'
spec=importlib.util.spec_from_file_location('_precore_interrupted_evaluator',ORIGINAL_EVAL/'evaluate_motor.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
for name in dir(original):
 if not name.startswith('__'):globals()[name]=getattr(original,name)
HERE=Path(__file__).resolve().parent

def source_paths():
 paths=original.source_paths()+[HERE/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','READINESS.json')]
 for p in paths:
  if not p.is_file():raise FileNotFoundError('pre-Core source path missing '+str(p))
 return list(dict.fromkeys(paths))

g=dict(original.evaluate.__globals__);g.update(HERE=HERE,source_paths=source_paths)
evaluate=types.FunctionType(original.evaluate.__code__,g,original.evaluate.__name__,original.evaluate.__defaults__,original.evaluate.__closure__)

if __name__=='__main__':evaluate(parser().parse_args())
