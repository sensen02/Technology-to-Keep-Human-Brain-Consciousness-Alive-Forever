"""One AST adaptation of frozenbody1 direct loops; full namespace is explicit newruntime."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent;BODY1=HERE.parent/'visual_body_v1'
class Adapt(ast.NodeTransformer):
 def __init__(self,name):self.name=name;self.query_changes=0;self.actor_changes=0;self.clock_changes=0;self.artifact_changes=0;self.audit_changes=0
 def visit_Constant(self,node):
  if node.value=='VISUAL_BODY_WORLD_LANDMARK_V1':return ast.copy_location(ast.Constant('VISUAL_BODY_OCCLUSION_FIXEDFRAME_V2'),node)
  if self.name=='prepare.py' and node.value=='VISION_STATIC_HEAD_POLICY_V1':return ast.copy_location(ast.Constant('VISION_STATIC_HEAD_OCCLUSION_RECOVERY_V2'),node)
  if self.name=='summary.py' and node.value=='VISUAL_BODY_SUMMARY.json':return ast.copy_location(ast.Constant('NORMAL6_SUMMARY.json'),node)
  return node
 def visit_Assign(self,node):
  self.generic_visit(node)
  if self.name=='audit.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='expectedids':
   node.value=ast.Call(func=ast.Name(id='expected_actor_ids',ctx=ast.Load()),args=[ast.Attribute(value=ast.Name(id='args',ctx=ast.Load()),attr='condition',ctx=ast.Load()),node.value],keywords=[]);self.actor_changes+=1
  if self.name=='prepare.py' and any(isinstance(t,ast.Name) and t.id=='paths' for t in node.targets):
   extra=ast.parse("[ROOT/'adapter.py']+[BODY1/n for n in ('runtime.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py')]").body[0].value;node.value=ast.BinOp(left=node.value,op=ast.Add(),right=extra)
  return node
 def visit_Compare(self,node):
  self.generic_visit(node)
  if self.name=='audit.py' and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq) and isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='cap' and isinstance(node.left.slice,ast.Constant) and node.left.slice.value in ('capture_time_s','release_time_s'):
   self.clock_changes+=1;return ast.copy_location(ast.Compare(left=ast.Call(func=ast.Name(id='abs',ctx=ast.Load()),args=[ast.BinOp(left=node.left,op=ast.Sub(),right=node.comparators[0])],keywords=[]),ops=[ast.LtE()],comparators=[ast.Constant(1e-10)]),node)
  return node
 def visit_keyword(self,node):
  self.generic_visit(node)
  if self.name=='orchestrate.py' and node.arg=='physical_trial_budget':node.value=ast.Constant(8)
  return node
 def visit_Call(self,node):
  self.generic_visit(node)
  if self.name=='runner.py' and isinstance(node.func,ast.Attribute) and node.func.attr=='query' and isinstance(node.func.value,ast.Attribute) and node.func.value.attr=='queue':
   node.func=ast.Attribute(value=ast.Name(id='stream',ctx=ast.Load()),attr='actor_query',ctx=ast.Load());self.query_changes+=1
  if self.name=='runner.py' and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='rows' and node.func.attr=='append':
   dictionary=node.args[0]
   for key,expr in [('raw_released_image_capture_id',"obs['raw_capture_id']"),('raw_released_capture_time_s',"obs['raw_capture_time_s']"),('raw_released_release_time_s',"obs['raw_release_time_s']"),('actor_frame_frozen',"obs['actor_frame_frozen']")]:dictionary.keywords.append(ast.keyword(arg=key,value=ast.parse(expr).body[0].value))
  return node
 def visit_Expr(self,node):
  self.generic_visit(node)
  if self.name=='runner.py' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='stream' and node.value.func.attr=='close':
   self.artifact_changes+=1;return [node,*ast.parse('artifacts.update(stream.additional_artifacts)').body]
  if self.name=='audit.py' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='save' and isinstance(node.value.args[0],ast.BinOp) and isinstance(node.value.args[0].right,ast.Constant) and node.value.args[0].right.value=='VERIFY.json':
   self.audit_changes+=1;return [*ast.parse('audit_controls(check,out,r,d,rgb,captures)').body,node]
  return node

def source(name):
 tree=ast.parse((BODY1/name).read_text());adapt=Adapt(name);tree=adapt.visit(tree);ast.fix_missing_locations(tree)
 if name=='runner.py' and (adapt.query_changes!=1 or adapt.artifact_changes!=1):raise ValueError('exactrunnerASTshapechanged')
 if name=='audit.py' and (adapt.actor_changes!=1 or adapt.clock_changes!=2 or adapt.audit_changes!=1):raise ValueError('exactauditASTshapechanged')
 return tree

def namespace(name):
 import runtime
 ns={'__name__':'_body2_exact_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(source(name),str(HERE/name)+'::explicitbody2namespace','exec'),ns)
 ns.update({k:v for k,v in runtime.__dict__.items() if not k.startswith('_')});ns['BODY1']=BODY1
 if name=='summary.py':ns['IDS']=runtime.NORMAL_IDS
 return ns

def run(name):namespace(name)['main']()
