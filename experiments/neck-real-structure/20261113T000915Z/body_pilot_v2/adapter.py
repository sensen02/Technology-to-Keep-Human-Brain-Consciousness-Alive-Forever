"""One AST namespace layer over immutable first-body direct loops; only identity/finalstep changes."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent;OLD_BODY=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_pilot')
class Adapt(ast.NodeTransformer):
 def __init__(self,name):self.name=name;self.step_changes=0;self.bindingchanges=0
 def visit_Constant(self,node):
  mapping={'ARTIFICIAL_BRAIN_SOURCE_NECK_BODY_PILOT_V1':'SOURCE_NECK_FEEDBACK_AGGREGATION_BODY_PILOT_V2','ENGINEERING_SOURCE_NECK_OFFLINE_CALIBRATION_V1':'SOURCE_NECK_FEEDBACK_AGGREGATION_V2','TRAIN/VERIFY_000200.json':'TRAIN/VERIFY_000400.json','actual_final200_bridge':'actual_final400_bridge','ARTIFICIAL_BRAIN_SOURCE_NECK_SIX_CASE_BODY_SUMMARY':'SOURCE_NECK_AGGREGATION_SIX_CASE_BODY_SUMMARY_V2','Actual final200 calibration SHA required':'Actual final400 calibration SHA required','Actual completed fixed200 pipeline required':'Actual completed fixed400 pipeline required','ACTUAL_CALIBRATION200_SOURCE_AND_MOTOR_BOUND_NO_PHYSICS':'ACTUAL_CALIBRATION400_SOURCE_AND_MOTOR_BOUND_NO_PHYSICS'}
  if isinstance(node.value,str) and node.value in mapping:return ast.copy_location(ast.Constant(mapping[node.value]),node)
  return node
 def visit_Compare(self,node):
  self.generic_visit(node)
  if self.name=='preflight.py' and isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='bp' and isinstance(node.left.slice,ast.Constant) and node.left.slice.value=='step' and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==200:
   node.comparators[0]=ast.Constant(400);self.step_changes+=1
  return node
 def visit_keyword(self,node):
  self.generic_visit(node)
  if self.name=='prepare.py' and node.arg=='brain_final_step' and isinstance(node.value,ast.Constant) and node.value.value==200:node.value=ast.Constant(400);self.step_changes+=1
  return node
 def visit_Call(self,node):
  self.generic_visit(node)
  if self.name=='prepare.py' and isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='brain_final_step' for k in node.keywords):node.keywords.append(ast.keyword(arg='brain_format',value=ast.Constant('SOURCE_NECK_FEEDBACK_AGGREGATION_V2')));self.bindingchanges+=1
  return node
 def visit_Assign(self,node):
  self.generic_visit(node)
  if self.name=='prepare.py' and any(isinstance(t,ast.Name) and t.id=='paths' for t in node.targets):
   extra=ast.parse("[ROOT/'adapter.py']+[OLD_BODY/n for n in ('PREREG.json','runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json')]").body[0].value;node.value=ast.BinOp(left=node.value,op=ast.Add(),right=extra)
  return node

def tree(name):
 if name not in ('runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py'):raise ValueError('Unregisteredbodyentry')
 transformer=Adapt(name);value=transformer.visit(ast.parse((OLD_BODY/name).read_text()));ast.fix_missing_locations(value)
 if name=='preflight.py' and transformer.step_changes!=1:raise ValueError('Expectedonefixed400preflightcomparison')
 if name=='prepare.py' and (transformer.step_changes!=1 or transformer.bindingchanges!=1):raise ValueError('Expectednew400sourcebindingfields')
 return value

def namespace(name):
 import runtime
 ns={'__name__':'_bodyv2_explicit_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(tree(name),str(HERE/name)+'::samephysicaldirectloops','exec'),ns);ns.update({k:v for k,v in runtime.__dict__.items() if not k.startswith('_')});ns['OLD_BODY']=OLD_BODY;return ns

def run(name):namespace(name)['main']()
