"""Explicit frozen evaluator binding; physical/scoring code remains unchanged."""
from pathlib import Path
import importlib.util,sys,json,types
import model_io as io
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'20261111T010218Z/evaluation'
spec=importlib.util.spec_from_file_location('_calibrated_eval_base',OLD/'evaluate_motor.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
for n in ('BASE','ORIGINAL','COMMANDS','HEAD_AUDIT','BASE_CHECKPOINT','jsonable','command_at','score','flatten','verify_parent_right','arrays','sha256','PRIMARY','verify_primary','replace_once'):globals()[n]=getattr(old,n)
def pin():
 sys.path.insert(0,str(BASE));import student_wave as sw;import identity
 return sw,identity,io
def checked_load(module,path,digest):return io.load_checked(path,digest)
def source_paths():
 files=[HERE/n for n in ('PREREG.json','model_io.py','evaluate_motor.py','verify_actual.py','prepare.py','pipeline.py','PREPARATION.json')]
 files+=list(OLD.glob('*.py'))+[OLD/'EVAL_PREREG.json',OLD/'READINESS.json',io.PARENT,HEAD_AUDIT]
 for p in files:
  if not p.is_file():raise FileNotFoundError(p)
 return files

def bind(case):
 case=Path(case).resolve()
 original=old.build()
 def actual_paths():return source_paths()+[case/'EVAL_PREREG.json',case/'READINESS.json',case/'candidate.pt']
 g=dict(original.evaluate.__globals__);g.update(HERE=case,ROOT=HERE,TRAIN=HERE,CHECKPOINT=case/'candidate.pt',pin=pin,checked_load=checked_load,source_paths=actual_paths,verify_parent_right=verify_parent_right)
 return types.FunctionType(original.evaluate.__code__,g,original.evaluate.__name__,original.evaluate.__defaults__,original.evaluate.__closure__)

def run(args):return bind(Path(args.out).parent)(args)
if __name__=='__main__':run(old.parser().parse_args())
