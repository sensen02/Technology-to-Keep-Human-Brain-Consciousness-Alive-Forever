"""Completeexplicitnewglobals over frozenrenderer/plot, onlyfixedclosedcoverdecoration changes."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent;ORIGINAL=HERE.parents[1]/'visual_body_v2/visual'
class Adapt(ast.NodeTransformer):
 def __init__(self,name):self.name=name;self.cover_setup=0;self.cover_draw=0
 def visit_If(self,node):
  text=ast.unparse(node.test)
  if self.name=='render.py' and text=="a.condition == 'V_left_occluded'" and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='geometry_path' for t in n.targets) for n in node.body):
   self.cover_setup+=1;return ast.parse("if a.condition=='V_left_occluded':\n cover_world_centers,cover_world_basis=recorded_covers(g,cap)").body[0]
  if self.name=='render.py' and text=="wall_center is not None and capture.get('counterfactual_available', False)":
   self.cover_draw+=1;return ast.parse("if a.condition=='V_left_occluded' and capture.get('counterfactual_available',False):\n for center in cover_world_centers:\n  require(capture['cover_rotation']==cover_world_basis.tolist(),'recordedcoverrotationdrift')\n  camera.add_decor(renderer.scene,mj,mj.mjtGeom.mjGEOM_BOX,[.18,.18,1.4],center,cover_world_basis,[.12,.12,.12,1.])").body[0]
  return self.generic_visit(node)
 def visit_Constant(self,node):
  if isinstance(node.value,str):
   text=node.value.replace('ACTUAL MOVING-HEAD VISION','ACTUAL FAR RGB / FIXED NEURAL OPPONENT').replace('NOT biological vision validation','engineering red-opponent / NOT biological retina').replace('wall3.60','fixed double covers3.60')
   if text!=node.value:return ast.copy_location(ast.Constant(text),node)
  return node
def source(name):
 tree=ast.parse((ORIGINAL/name).read_text());change=Adapt(name);tree=change.visit(tree);ast.fix_missing_locations(tree)
 if name=='render.py' and (change.cover_setup!=1 or change.cover_draw!=1):raise ValueError('exactfrozennearwallrenderadaptation expected')
 return tree
def recorded_covers(geometry,captures):
 import numpy as np
 valid=np.flatnonzero(geometry['source_anchor_valid']);
 if not len(valid):raise ValueError('savedworldanchorrequired')
 centers=np.asarray(geometry['target_centers_xyz'][int(valid[0])],dtype=float)
 if centers.shape!=(3,3):raise ValueError('savedorderedleft/right/centeranchorsrequired')
 # Actualsourcearray order is fixed in frozenrunner: LEFT,RIGHT,CENTER. No laterhead recomputation.
 after=np.asarray(geometry['target_centers_xyz'][valid]);
 if not np.array_equal(after,np.broadcast_to(centers,after.shape)):raise ValueError('worldanchorchanged')
 boxes=centers[:2].copy();boxes[:,2]=1.4;covered=[c for c in captures if c.get('counterfactual_available',False)]
 if not covered:raise ValueError('actualcovermetadatarequired')
 basis=np.asarray(covered[0]['cover_rotation'],dtype=float)
 if basis.shape!=(3,3) or not all(np.array_equal(c['cover_rotation'],basis) for c in covered):raise ValueError('actualfixedcoverbasisrequired')
 return boxes,basis
def namespace(name):
 import contracts
 ns={'__name__':'_opponent_delivery_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(source(name),str(HERE/name)+'::savedactual_far_opponent_delivery','exec'),ns);ns.update({k:v for k,v in contracts.__dict__.items() if not k.startswith('_')});ns['recorded_covers']=recorded_covers;return ns
def run(name):namespace(name)['main']()
