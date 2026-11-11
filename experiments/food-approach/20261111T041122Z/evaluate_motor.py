"""Frozen original physical loop with explicit half-command test input only."""
from pathlib import Path
import types,sys,importlib.util,json
import numpy as np
import model_io as io
HERE=Path(__file__).resolve().parent
PRIMARY=HERE.parent/'20261110T235830Z/evaluation'
SOURCE=PRIMARY/'evaluate_motor.py'
if io.sha256(SOURCE)!='2e91e11d71013a993cfdb6bcda7920d36a57ba5b3d82105bb05ece503609c012':raise ValueError('frozen physical loop changed')
mod=types.ModuleType('_half_frozen_original_evaluator');mod.__dict__.update(__file__=str(SOURCE),__name__=mod.__name__)
_source=SOURCE.read_text().replace("prereg['checkpoint_step']!=1500","prereg['checkpoint_step']!=2000").replace("kind='round2 fixed1500 slow residual actual physical evaluation'","kind='unchanged eeced2000 fixed half-command physical evaluation'")
exec(compile(_source,str(SOURCE)+'::half-profile-metadata', 'exec'),mod.__dict__)
BASE=mod.BASE;ORIGINAL=mod.ORIGINAL;HEAD_AUDIT=mod.HEAD_AUDIT
COMMANDS={'left':[0.,.5],'right':[0.,-.5]}
for key in ('jsonable','score','flatten','sha256'):globals()[key]=getattr(mod,key)
def command_at(condition,decision):
 if condition not in COMMANDS or not isinstance(decision,(int,np.integer)) or not 0<=decision<2880:raise ValueError('fixed half profile required')
 u=min(max(0.,(int(decision)-960)*.005),1.);blend=u*u*(3.-2.*u)
 return np.asarray(COMMANDS[condition],dtype=np.float64)*blend

def pin():
 sys.path.insert(0,str(BASE));import student_wave as sw;import identity
 return sw,identity,io

def checked_load(module,path,digest):return io.load_checked(path,digest)
def source_paths():
 files=[HERE/n for n in ('PREREG.json','EVAL_PREREG.json','model_io.py','evaluate_motor.py','verify_actual.py','prepare.py','pipeline.py','PREPARATION.json')]+[SOURCE,PRIMARY/'verify_actual.py',io.PATH,HEAD_AUDIT]
 for p in files:
  if not p.is_file():raise FileNotFoundError('preCore source missing '+str(p))
 return files

def run(args):
 case=Path(args.out).parent.resolve();g=dict(mod.evaluate.__globals__)
 def paths():return source_paths()+[case/'EVAL_PREREG.json',case/'READINESS.json']
 g.update(HERE=case,ROOT=HERE,TRAIN=HERE,CHECKPOINT=io.CHECKPOINT,COMMANDS=COMMANDS,command_at=command_at,pin=pin,checked_load=checked_load,source_paths=paths)
 return types.FunctionType(mod.evaluate.__code__,g,mod.evaluate.__name__,mod.evaluate.__defaults__,mod.evaluate.__closure__)(args)

if __name__=='__main__':run(mod.parser().parse_args())
