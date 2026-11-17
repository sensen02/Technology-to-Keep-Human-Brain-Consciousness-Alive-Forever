"""Targeted format/finalcheckpoint AST only; all physics/sensor/control constants unchanged."""
from pathlib import Path
import ast,importlib.util,sys
HERE=Path(__file__).resolve().parent
VIS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z');FAR_BODY=VIS/'landmark_far_v3/body'
_s=importlib.util.spec_from_file_location('_coverage_body_far_adapter',FAR_BODY/'adapter.py');_far_adapter=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_far_adapter;_s.loader.exec_module(_far_adapter)
class CoverageScope(ast.NodeTransformer):
 def __init__(self,name):self.name=name;self.step_changes=0
 def visit_Constant(self,node):
  mapping={'VISUAL_FAR_WORLD_LANDMARK_BODY_V3':'VISUAL_COVERAGE_POLICY_FAR_BODY_V5','VISION_FAR_LANDMARK_POLICY_V3':'VISION_COVERAGE_POLICY_V5'}
  if self.name=='prepare.py':mapping.update({'checkpoint_000500.pt':'checkpoint_001000.pt','VERIFY_000500.json':'VERIFY_001000.json','TRAIN/checkpoint_000500.pt':'TRAIN/checkpoint_001000.pt','TRAIN/VERIFY_000500.json':'TRAIN/VERIFY_001000.json'})
  if self.name=='preflight.py':mapping['strict_actual_far_final500']='strict_actual_coverage_final1000'
  if isinstance(node.value,str) and node.value in mapping:return ast.copy_location(ast.Constant(mapping[node.value]),node)
  return node
 def visit_Compare(self,node):
  self.generic_visit(node)
  if self.name=='preflight.py' and isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='vp' and isinstance(node.left.slice,ast.Constant) and node.left.slice.value=='step' and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==500:
   node.comparators[0]=ast.Constant(1000);self.step_changes+=1
  return node
 def visit_Assign(self,node):
  self.generic_visit(node)
  if self.name=='prepare.py' and any(isinstance(t,ast.Name) and t.id=='paths' for t in node.targets):
   extra=ast.parse("[FAR_BODY/n for n in ('PREREG_PROPOSAL.json','runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json')]").body[0].value;node.value=ast.BinOp(left=node.value,op=ast.Add(),right=extra)
  return node

def _compile_namespace(tree,name):
 import runtime
 transform=CoverageScope(name);tree=transform.visit(tree);ast.fix_missing_locations(tree)
 if name=='preflight.py' and transform.step_changes!=1:raise ValueError('exact one preflight vp final1000 comparison expected')
 ns={'__name__':'_coverage_body_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(tree,str(HERE/name)+'::targetedfinal1000_samephysics','exec'),ns);ns.update({key:value for key,value in runtime.__dict__.items() if not key.startswith('_')});ns['FAR_BODY']=FAR_BODY;return ns

def namespace(name):
 ns=_compile_namespace(_far_adapter.source(name),name)
 if name=='summary.py':ns['IDS']=ns['NORMAL_IDS']
 return ns

def entry_namespace(name):
 if name not in ('prepare.py','preflight.py','summary.py','orchestrate.py'):raise ValueError('unregistered coverage entry')
 return _compile_namespace(ast.parse((FAR_BODY/name).read_text()),name)

def run(name):namespace(name)['main']()
def run_entry(name):entry_namespace(name)['main']()
