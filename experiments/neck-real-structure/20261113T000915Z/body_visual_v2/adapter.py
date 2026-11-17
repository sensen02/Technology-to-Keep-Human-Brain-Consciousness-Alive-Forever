"""One explicitnamespace rebind over actualverified v1renderer/plot, candidatecaption only."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
OLD_VISUAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/body_visual')
class CandidateCaption(ast.NodeTransformer):
 def __init__(self,name):self.name=name;self.choice_changes=0
 def visit_Tuple(self,node):
  if self.name=='render.py' and all(isinstance(x,ast.Constant) for x in node.elts) and [x.value for x in node.elts]==['P_left_pulse','P_left_DN_cut']:
   self.choice_changes+=1;return ast.copy_location(ast.Tuple(elts=[ast.Constant('P_empty'),*node.elts],ctx=node.ctx),node)
  return self.generic_visit(node)
 def visit_Constant(self,node):
  if isinstance(node.value,str):
   text=node.value.replace('Source-qualified 2 DNp09 / 1 AN08B018 ports | experimental instruction, NOT visual input','Feedback aggregation v2 | source2 DNp09 / 1 AN08B018 | instruction, NOT vision').replace('ACTUAL ARTIFICIAL BRAIN16 ->','FEEDBACK AGGREGATION V2: ARTIFICIAL BRAIN16 ->').replace('MATCHED ACTUAL PHYSICAL COMMUNICATION INTERVENTIONS','FEEDBACK AGGREGATION V2: MATCHED ACTUAL PHYSICAL INTERVENTIONS')
   if text!=node.value:return ast.copy_location(ast.Constant(text),node)
  return node

def source(name):
 if name not in ('render.py','plot_sync.py'):raise ValueError('Registeredoldvisualentryonly')
 change=CandidateCaption(name);tree=change.visit(ast.parse((OLD_VISUAL/name).read_text()));ast.fix_missing_locations(tree)
 if name=='render.py' and change.choice_changes!=1:raise ValueError('Exactly one pre-render Pempty CLI extension expected')
 return tree

def namespace(name):
 import contracts
 ns={'__name__':'_aggregation_v2_savedvisual_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(source(name),str(HERE/name)+'::oldverifiedrender_newcandidate_namespace','exec'),ns);ns.update({k:v for k,v in contracts.__dict__.items() if not k.startswith('_')});return ns

def run(name):namespace(name)['main']()
