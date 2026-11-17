"""Reuse frozen direct loops as AST with one far-scope kind change; explicit globals."""
from pathlib import Path
import ast,importlib.util,sys
HERE=Path(__file__).resolve().parent
VIS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z')
BODY1=VIS/'visual_body_v1';BODY2=VIS/'visual_body_v2'
s=importlib.util.spec_from_file_location('_far_inherited_body2_ast',BODY2/'adapter.py');_adapter=importlib.util.module_from_spec(s);sys.modules[s.name]=_adapter;s.loader.exec_module(_adapter)
class FarKind(ast.NodeTransformer):
 def visit_Constant(self,node):
  if node.value=='VISUAL_BODY_OCCLUSION_FIXEDFRAME_V2':return ast.copy_location(ast.Constant('VISUAL_FAR_WORLD_LANDMARK_BODY_V3'),node)
  return node

def source(name):
 if name not in ('runner.py','audit.py','summary.py'):raise ValueError('only registered direct-loop reuse')
 tree=FarKind().visit(_adapter.source(name));ast.fix_missing_locations(tree);return tree

def namespace(name):
 import runtime
 ns={'__name__':'_far_direct_'+name.replace('.','_'),'__file__':str(HERE/name)}
 exec(compile(source(name),str(HERE/name)+'::far_same_physics','exec'),ns)
 ns.update({key:value for key,value in runtime.__dict__.items() if not key.startswith('_')})
 if name=='summary.py':ns['IDS']=runtime.NORMAL_IDS
 return ns

def run(name):namespace(name)['main']()
