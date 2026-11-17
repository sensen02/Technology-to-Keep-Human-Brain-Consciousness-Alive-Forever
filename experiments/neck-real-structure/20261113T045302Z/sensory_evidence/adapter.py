"""One explicit AST layer over immutable actualverified RGB/source renderer; no heavy import."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent;OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T033622Z/visual_evidence/render.py')

def expression(text):return ast.parse(text,mode='eval').body
class Multisensory(ast.NodeTransformer):
 def __init__(self):self.changes={k:0 for k in ('baseline','frozen','ANcut','details','header','odor_decor')}
 def visit_Constant(self,node):
  if isinstance(node.value,str):
   value=node.value.replace('ACTUAL RGB BRAIN -> NECK -> OTHER','ACTUAL RGB + ODOR -> BRAIN -> NECK -> OTHER').replace('ACTUAL_RGB_BRAIN_NECK_BODY.mp4','ACTUAL_MULTISENSORY_BRAIN_NECK_BODY.mp4').replace('minus matched V_empty','minus matched O_noodor').replace('304 memories','336 memories').replace('16CNNcache / AN1 / DN2 separate','16cache/filterpending/AN1DN2 separate').replace('NOT full-neck/native retina/general navigation validation','NOT full-neck/native sensory/food arrival validation').replace('Source-cut','Source-cut')
   if value!=node.value:return ast.copy_location(ast.Constant(value),node)
  return node
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='load' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value=='V_empty':node.args[0]=ast.Constant('O_noodor');self.changes['baseline']+=1
  return node
 def visit_Subscript(self,node):
  self.generic_visit(node)
  if isinstance(node.value,ast.Name) and node.value.id=='d' and isinstance(node.slice,ast.Constant) and node.slice.value=='actor_frame_frozen':self.changes['frozen']+=1;return ast.copy_location(expression('np.zeros(3080,dtype=bool)'),node)
  if isinstance(node.value,ast.Name) and node.value.id=='d' and isinstance(node.slice,ast.Constant) and node.slice.value=='AN_cut':self.changes['ANcut']+=1;return ast.copy_location(expression('np.zeros(3080,dtype=bool)'),node)
  return node
 def visit_Assign(self,node):
  self.generic_visit(node)
  if any(isinstance(t,ast.Name) and t.id=='details' for t in node.targets):
   node.value=expression("[f'Vision NN {vu:+.5f}  Odor NN {float(d[\"odor_raw_turn\"][qi]):+.5f}',f'Fusionbrain16 norm {brainnorm:.5f}',f'ENV receptors L/R {np.round(d[\"receptors_used\"][qi],4).tolist()}',f'ACTOR receptors L/R {np.round(d[\"actor_receptors\"][qi],4).tolist()}',f'Concentration(dimless) {np.round(d[\"concentration\"][qi],4).tolist()}',f'Filter before {np.round(d[\"filter_before\"][qi],4).tolist()}',f'Filter after {np.round(d[\"filter_after\"][qi],4).tolist()}',f'Pending before {np.round(d[\"pending_before\"][qi],4).tolist()}',f'Pending after {np.round(d[\"pending_after\"][qi],4).tolist()}',f'Odor cap/consume {int(d[\"odor_capture_step\"][qi])}/{int(d[\"odor_consume_step\"][qi])} ms',f'Odor cut={bool(d[\"odor_actor_cut\"][qi])} swap={bool(d[\"odor_actor_swap\"][qi])}',f'DN held {np.round(d[\"DN_held\"][qi],5).tolist()}',f'AN held {np.round(d[\"AN_held\"][qi],5).tolist()} decoded {turn:+.5f}',f'DN cap/rel/use {int(d[\"DN_capture_step\"][qi])}/{int(d[\"DN_release_step\"][qi])}/{int(d[\"physical_step_ms\"][qi])}',f'AN cap/rel/use {int(d[\"AN_capture_step\"][qi])}/{int(d[\"AN_release_step\"][qi])}/{int(d[\"physical_step_ms\"][qi])}','336 memories;cache/filter/pending separate','Odor marks: spectator ENV only, not eyeRGB']")
   self.changes['details']+=1
  if any(isinstance(t,ast.Name) and t.id=='lines' for t in node.targets):
   node.value=expression("[a.condition+' | '+phase,f'Actual {now:.3f}/15.400s | '+(f'ANN {elapsed:.3f}s' if elapsed>=0 else 'PRE-ANN')+' | 1.00x /50fps',f'Actual vision {vu:+.4f} + odor {float(d[\"odor_raw_turn\"][qi]):+.4f} -> brain16 -> DN -> turn {turn:+.4f}',f'336 memories; independent visual/odor world sources; scene {capture[\"stimulus\"]}']")
   self.changes['header']+=1
  if any(isinstance(t,ast.Name) and t.id=='body' for t in node.targets):
   extra=ast.parse("""
# Spectator-only odor marks are NEVER inserted into savedactorRGB or policy inputs.
if a.condition not in ('V_empty','V_center','V_left_pulse','V_right_pulse','V_left_to_right','V_right_to_left','O_noodor') and bool(g['odor_source_valid'][i]) and physical_ms>=6400:
 center=np.r_[g['odor_source_xy'][i],.02]
 camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_CYLINDER,[.07,.01,0],center,np.eye(3),[.1,.8,.8,.8])
 for site in g['sensor_xyz'][i]:camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_SPHERE,[.035,0,0],site,np.eye(3),[.1,.8,.8,1.])
""").body
   self.changes['odor_decor']+=1;return extra+[node]
  return node
 def visit_keyword(self,node):
  self.generic_visit(node)
  if node.arg=='neural_state_total' and isinstance(node.value,ast.Constant) and node.value.value==304:node.value=ast.Constant(336)
  return node
 def visit_BinOp(self,node):
  self.generic_visit(node)
  if isinstance(node.op,ast.Add) and isinstance(node.left,ast.Constant) and node.left.value==362 and isinstance(node.right,ast.BinOp) and isinstance(node.right.op,ast.Mult) and isinstance(node.right.left,ast.Constant) and node.right.left.value==22 and isinstance(node.right.right,ast.Name) and node.right.right.id=='j':return ast.copy_location(expression('350+18*j'),node)
  return node

def tree():
 change=Multisensory();value=change.visit(ast.parse(OLD.read_text()));ast.fix_missing_locations(value)
 expected={'baseline':1,'frozen':1,'ANcut':1,'details':1,'header':1,'odor_decor':1}
 if change.changes!=expected:raise ValueError('Exact renderer adaptations drift '+str(change.changes))
 return value,change.changes

def namespace():
 import contracts
 value,changes=tree();ns=dict(__name__='_sensory_actualsaved_render',__file__=str(HERE/'render.py'));exec(compile(value,str(HERE/'render.py')+'::immutableRGBcore_multisensory_sidebar','exec'),ns);ns.update({k:v for k,v in contracts.__dict__.items() if not k.startswith('_')});return ns
