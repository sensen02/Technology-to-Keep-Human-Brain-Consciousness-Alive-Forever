"""ONLYstd globals/API/branchisolation staticcanaries, no backendimport."""
from pathlib import Path
import ast,builtins,contextlib,dis,json,sys,types
ROOT=Path(__file__).resolve().parent

def unresolved(function):
 missing=[];seen=set()
 def codevisit(code,namespace,label):
  for op in dis.get_instructions(code):
   if op.opname in ('LOAD_GLOBAL','LOAD_NAME') and op.argval not in namespace and op.argval not in builtins.__dict__:missing.append(label+':'+str(op.argval))
  for c in code.co_consts:
   if isinstance(c,types.CodeType):codevisit(c,namespace,label+'/<code>')
 def visit(f):
  if id(f) in seen:return
  seen.add(id(f));codevisit(f.__code__,f.__globals__,f.__qualname__)
  w=getattr(f,'__wrapped__',None)
  if isinstance(w,types.FunctionType):visit(w)
 visit(function);return missing

def main():
 for path in ROOT.glob('*.py'):compile(ast.parse(path.read_text()),str(path),'exec')
 import common,learning,sources,data,utils
 assert common.SAVE_STEPS==(600,750,900) and common.FORMAT=='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
 assert common.load_checkpoint.__globals__ is common.g and common.save_checkpoint.__globals__ is common.g
 assert common._preserve_rng.__globals__ is contextlib.__dict__ and not unresolved(common._preserve_rng)
 manager=common._preserve_rng();assert type(manager).__name__=='_GeneratorContextManager'
 for f in (common.load_checkpoint,common.save_checkpoint,learning.tf_objective,learning.conditional_objective,learning.tf_forward,learning.forward):assert not unresolved(f),unresolved(f)
 tree=ast.parse((ROOT/'learning.py').read_text());conditional=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='conditional_objective')
 assert not any(isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='data' and isinstance(n.slice,ast.Constant) and n.slice.value in ('y','valid') for n in ast.walk(conditional))
 assert learning.resource_gate.__defaults__==(9000.,) and 1800 in learning.resource_gate.__code__.co_consts
 train=ast.parse((ROOT/'train.py').read_text());calls=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
 assert any(n.func.id=='range' and len(n.args)==2 and all(isinstance(a,ast.Constant) for a in n.args) and [a.value for a in n.args]==[601,901] for n in calls)
 assert not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym'))
 print(json.dumps(dict(ASTall=True,strict900_namespace=True,contextmanager_originalglobals=True,TFonly_actionlabels_static=True,training1800wall9000=True,updates601_to900_only=True,numericimports=0,arrayloads=0,modelcalls=0,Core=0,physical=0),indent=2))
if __name__=='__main__':main()
