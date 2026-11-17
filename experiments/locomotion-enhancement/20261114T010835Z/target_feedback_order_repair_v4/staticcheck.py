"""Stdlib imported/decorated helper global resolution, no numericbackend."""
from pathlib import Path
import ast,builtins,contextlib,dis,hashlib,inspect,json,sys,types
ROOT=Path(__file__).resolve().parent

def global_check(function):
 missing=[];seen=set()
 def visit(code,namespace,label):
  for op in dis.get_instructions(code):
   if op.opname in ('LOAD_GLOBAL','LOAD_NAME'):
    if op.argval not in namespace and op.argval not in builtins.__dict__:missing.append(label+':'+str(op.argval))
  for c in code.co_consts:
   if isinstance(c,types.CodeType):visit(c,namespace,label+'/<code>')
 def functionvisit(f):
  if id(f) in seen:return
  seen.add(id(f));visit(f.__code__,f.__globals__,f.__qualname__)
  wrapped=getattr(f,'__wrapped__',None)
  if isinstance(wrapped,types.FunctionType):functionvisit(wrapped)
 functionvisit(function);return missing

def main():
 for p in ROOT.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
 import common
 preserved=common._original._preserve_rng
 assert common._preserve_rng is preserved
 assert common._preserve_rng.__globals__ is preserved.__globals__
 assert '_GeneratorContextManager' in common._preserve_rng.__globals__
 assert not global_check(common._preserve_rng)
 manager=common._preserve_rng()
 assert type(manager).__name__=='_GeneratorContextManager'
 # Do not enter this real RNG context: itsbody imports numericalbackends.
 for name in common.REBIND_SCIENCE:
  assert common.g[name].__globals__ is common.g and not global_check(common.g[name])
 for name in ('_validate_rng','restore_rng','_validate_optimizer_state','_validate_live_optimizer','_digest','sha','_read','verify_hashes','_require_neural_only','_module'):
  assert name in common._original.__dict__,name
  assert common.g[name] is common._original.__dict__[name]
 @contextlib.contextmanager
 def fixture():
  yield 'stdlibcontextwrapperexecution'
 with fixture() as value:assert value=='stdlibcontextwrapperexecution'
 assert fixture.__globals__ is contextlib.__dict__ and not global_check(fixture)
 assert not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym'))
 print(json.dumps(dict(AST=True,scientific_clone_whitelist=list(common.REBIND_SCIENCE),preserved_original_contextmanager_globals=True,recursive_function_globals_resolved=True,stdlib_contextmanager_fixture_executed=True,numeric_imports=0,model_loads=0,physical_steps=0),indent=2))
if __name__=='__main__':main()
