"""Thin explicit namespace reuse of immutable FAR body loops; no runtime image augmentation."""
from pathlib import Path
import ast,importlib.util,sys
HERE=Path(__file__).resolve().parent
VIS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z')
FAR_BODY=VIS/'landmark_far_v3/body'
_s=importlib.util.spec_from_file_location('_mirror_body_far_loop_adapter',FAR_BODY/'adapter.py');_far_adapter=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_far_adapter;_s.loader.exec_module(_far_adapter)
class MirrorScope(ast.NodeTransformer):
 def __init__(self,name):self.name=name
 def visit_Constant(self,node):
  mapping={'VISUAL_FAR_WORLD_LANDMARK_BODY_V3':'VISUAL_FAR_MIRROR_POLICY_BODY_V4','VISION_FAR_LANDMARK_POLICY_V3':'VISION_FAR_MIRROR_POLICY_V4'}
  if isinstance(node.value,str) and node.value in mapping:return ast.copy_location(ast.Constant(mapping[node.value]),node)
  return node
 def visit_Assign(self,node):
  self.generic_visit(node)
  if self.name=='prepare.py' and any(isinstance(t,ast.Name) and t.id=='paths' for t in node.targets):
   extra=ast.parse("[FAR_BODY/n for n in ('PREREG_PROPOSAL.json','runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json')]").body[0].value
   node.value=ast.BinOp(left=node.value,op=ast.Add(),right=extra)
  return node

def _compile_namespace(tree,name):
 import runtime
 tree=MirrorScope(name).visit(tree);ast.fix_missing_locations(tree)
 ns={'__name__':'_mirror_body_'+name.replace('.','_'),'__file__':str(HERE/name)}
 exec(compile(tree,str(HERE/name)+'::mirror_policy_actual_RGB','exec'),ns)
 ns.update({key:value for key,value in runtime.__dict__.items() if not key.startswith('_')});ns['FAR_BODY']=FAR_BODY
 return ns

def namespace(name):
 # Summary here is the inherited normal6 helper, called by the separate full summary entry.
 ns=_compile_namespace(_far_adapter.source(name),name)
 if name=='summary.py':ns['IDS']=ns['NORMAL_IDS']
 return ns

def entry_namespace(name):
 if name not in ('prepare.py','preflight.py','summary.py','orchestrate.py'):raise ValueError('unregistered mirror body entry')
 return _compile_namespace(ast.parse((FAR_BODY/name).read_text()),name)

def run(name):namespace(name)['main']()
def run_entry(name):entry_namespace(name)['main']()
