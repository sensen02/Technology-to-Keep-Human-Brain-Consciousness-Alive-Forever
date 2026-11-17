"""New policy namespace only over immutable coverage/far code; actual RGB unchanged."""
from pathlib import Path
import ast,importlib.util,sys
HERE=Path(__file__).resolve().parent;VIS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z');COVERAGE_BODY=VIS/'landmark_coverage_v5/body'
_s=importlib.util.spec_from_file_location('_opponent_body_coverage_adapter',COVERAGE_BODY/'adapter.py');_coverage_adapter=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_coverage_adapter;_s.loader.exec_module(_coverage_adapter)
class OpponentScope(ast.NodeTransformer):
 def __init__(self,name):self.name=name
 def visit_Constant(self,node):
  mapping={'VISUAL_FAR_WORLD_LANDMARK_BODY_V3':'VISUAL_OPPONENT_FAR_BODY_V7','VISUAL_COVERAGE_POLICY_FAR_BODY_V5':'VISUAL_OPPONENT_FAR_BODY_V7','VISION_FAR_LANDMARK_POLICY_V3':'VISION_OPPONENT_POLICY_V7','VISION_COVERAGE_POLICY_V5':'VISION_OPPONENT_POLICY_V7'}
  if isinstance(node.value,str) and node.value in mapping:return ast.copy_location(ast.Constant(mapping[node.value]),node)
  return node
 def visit_Assign(self,node):
  self.generic_visit(node)
  if self.name=='prepare.py' and any(isinstance(t,ast.Name) and t.id=='paths' for t in node.targets):
   extra=ast.parse("[COVERAGE_BODY/n for n in ('runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json','PREREG_PROPOSAL.json')]").body[0].value;node.value=ast.BinOp(left=node.value,op=ast.Add(),right=extra)
  return node

def _compile(tree,name):
 import runtime
 tree=OpponentScope(name).visit(tree);ast.fix_missing_locations(tree)
 ns={'__name__':'_opponent_body_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(tree,str(HERE/name)+'::new_opponent_actualRGB','exec'),ns);ns.update({key:value for key,value in runtime.__dict__.items() if not key.startswith('_')});ns['COVERAGE_BODY']=COVERAGE_BODY;return ns

def _coverage_tree(name):
 # Apply originalcoverage targetedfinal1000 mapping without invoking its actualruntime.
 if name in ('runner.py','audit.py'):tree=_coverage_adapter._far_adapter.source(name)
 else:tree=ast.parse((_coverage_adapter.FAR_BODY/name).read_text())
 transform=_coverage_adapter.CoverageScope(name);tree=transform.visit(tree);ast.fix_missing_locations(tree)
 if name=='preflight.py' and transform.step_changes!=1:raise ValueError('original strictfinal1000 mapping drift')
 return tree

def namespace(name):
 if name not in ('runner.py','audit.py','summary.py'):raise ValueError('unregistered actual loop')
 if name=='summary.py':tree=_coverage_adapter._far_adapter.source(name)
 else:tree=_coverage_tree(name)
 ns=_compile(tree,name)
 if name=='summary.py':ns['IDS']=ns['NORMAL_IDS']
 return ns

def entry_namespace(name):
 if name not in ('prepare.py','preflight.py','summary.py','orchestrate.py'):raise ValueError('unregistered entry')
 return _compile(_coverage_tree(name),name)

def run(name):namespace(name)['main']()
def run_entry(name):entry_namespace(name)['main']()
