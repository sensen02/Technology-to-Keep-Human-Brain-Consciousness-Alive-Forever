"""Same far actual-RGB physics/control functions with all globals rebound to mirror-policy body."""
from pathlib import Path
import importlib.util,sys,types
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z'
ROOT=VIS/'landmark_mirror_v4/body';POLICY=VIS/'landmark_mirror_v4/policy';FAR_BODY=VIS/'landmark_far_v3/body'
_s=importlib.util.spec_from_file_location('_mirror_body_frozen_far_runtime',FAR_BODY/'runtime.py');_far=importlib.util.module_from_spec(_s);sys.modules[_s.name]=_far;_s.loader.exec_module(_far)
_g=dict(_far.__dict__)
_g.update(ROOT=ROOT,POLICY=POLICY,VIS=VIS,PROJECT=PROJECT,FAR_BODY=FAR_BODY)
# Every inherited helper and original loop method receives one complete new namespace.
for _key,_value in list(_g.items()):
 if isinstance(_value,types.FunctionType):_g[_key]=types.FunctionType(_value.__code__,_g,_value.__name__,_value.__defaults__,_value.__closure__)
for _name in ('_OpticalProxy','CameraStream'):
 _cls=getattr(_far,_name);_attrs={key:(types.FunctionType(value.__code__,_g,value.__name__,value.__defaults__,value.__closure__) if isinstance(value,types.FunctionType) else value) for key,value in _cls.__dict__.items() if key not in ('__dict__','__weakref__')}
 _attrs['__module__']=__name__;_g[_name]=type(_name,_cls.__bases__,_attrs)
_g['_g']=_g
globals().update({key:value for key,value in _g.items() if not key.startswith('__')})
# Never apply policy training augmentation here: images pass unchanged through original capture/queue.
POLICY_FORMAT='VISION_FAR_MIRROR_POLICY_V4'
_inherited_load_policies=load_policies
def load_policies():
 if read(ROOT/'INPUT_BINDINGS.json')['visual_format']!=POLICY_FORMAT:raise ValueError('new mirror-policy format required')
 visual,motor,vp,mp=_inherited_load_policies()
 if vp['format_version']!=POLICY_FORMAT or vp['step']!=500:raise ValueError('actual mirror final500 required')
 return visual,motor,vp,mp
_g.update(load_policies=load_policies,POLICY_FORMAT=POLICY_FORMAT)
