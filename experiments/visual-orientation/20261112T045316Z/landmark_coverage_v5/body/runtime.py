"""Same actual far RGB body, explicit new coverage policy roots and strictfinal1000 only."""
from pathlib import Path
import importlib.util,sys,types,ast
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z'
ROOT=VIS/'landmark_coverage_v5/body';POLICY=VIS/'landmark_coverage_v5/policy';FAR_BODY=VIS/'landmark_far_v3/body'
_s=importlib.util.spec_from_file_location('_coverage_body_frozen_far_runtime',FAR_BODY/'runtime.py');_far=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_far;_s.loader.exec_module(_far)
_g=dict(_far.__dict__);_g.update(ROOT=ROOT,POLICY=POLICY,VIS=VIS,PROJECT=PROJECT,FAR_BODY=FAR_BODY)
for _key,_value in list(_g.items()):
 if isinstance(_value,types.FunctionType):_g[_key]=types.FunctionType(_value.__code__,_g,_value.__name__,_value.__defaults__,_value.__closure__)
for _name in ('_OpticalProxy','CameraStream'):
 _cls=getattr(_far,_name);_attrs={key:(types.FunctionType(value.__code__,_g,value.__name__,value.__defaults__,value.__closure__) if isinstance(value,types.FunctionType) else value) for key,value in _cls.__dict__.items() if key not in ('__dict__','__weakref__')};_attrs['__module__']=__name__;_g[_name]=type(_name,_cls.__bases__,_attrs)

class _FinalStep(ast.NodeTransformer):
 def __init__(self):self.changed=0
 def visit_Compare(self,node):
  self.generic_visit(node)
  if isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='vp' and isinstance(node.left.slice,ast.Constant) and node.left.slice.value=='step' and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==500:
   node.comparators[0]=ast.Constant(1000);self.changed+=1
  return node
_tree=ast.parse((FAR_BODY/'runtime.py').read_text());_fn=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_policies');_change=_FinalStep();_fn=_change.visit(_fn)
if _change.changed!=1:raise ValueError('exactly one far policy finalstep comparison expected')
_only=ast.Module(body=[_fn],type_ignores=[]);ast.fix_missing_locations(_only);exec(compile(_only,str(ROOT/'runtime.py')+'::targetedvpstep1000','exec'),_g)
_g['_g']=_g;globals().update({key:value for key,value in _g.items() if not key.startswith('__')})
POLICY_FORMAT='VISION_COVERAGE_POLICY_V5';FINAL_STEP=1000
_inherited_load_policies=load_policies

def load_policies():
 if read(ROOT/'INPUT_BINDINGS.json')['visual_format']!=POLICY_FORMAT:raise ValueError('new coverage policy format required')
 visual,motor,vp,mp=_inherited_load_policies()
 if vp['format_version']!=POLICY_FORMAT or vp['step']!=FINAL_STEP:raise ValueError('actual coverage final1000 required')
 return visual,motor,vp,mp
_g.update(load_policies=load_policies,POLICY_FORMAT=POLICY_FORMAT,FINAL_STEP=FINAL_STEP)
