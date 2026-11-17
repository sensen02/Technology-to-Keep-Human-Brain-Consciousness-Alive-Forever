"""V2 extends inherited real camera capture guard only; preserves failed V1."""
from pathlib import Path
import ast
import importlib.util
import inspect
import sys
import textwrap
import types

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
spec = importlib.util.spec_from_file_location('_rough_long_v1_frozen_runtime',V1/'runtime.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
g = dict(vars(base))
g.update(HERE=HERE,KIND='SOURCE_NECK_ROUGH_TERRAIN_LONG19P2_CAPTURE_REPAIR_V2')
for key,value in list(g.items()):
    if isinstance(value,types.FunctionType) and value.__module__==base.__name__:
        cloned = types.FunctionType(value.__code__,g,value.__name__,value.__defaults__,value.__closure__)
        cloned.__kwdefaults__ = None if value.__kwdefaults__ is None else dict(value.__kwdefaults__)
        cloned.__annotations__ = dict(value.__annotations__)
        g[key] = cloned
_original_adapted_upstream = g['adapted_upstream']
_original_verify = g['verify']

def camera_namespace(old_camera_runtime):
    original = old_camera_runtime()
    capture = original['_capture']
    source = textwrap.dedent(inspect.getsource(capture))
    tree,changes = g['scheduled_tree'](source,inspect.getsourcefile(capture))
    if len(changes)!=1 or changes[0]['old_value']!=15400 or changes[0]['new_expression']!='TOTAL_MS':
        raise ValueError('Exactly inherited camera15400 bound must be changed')
    capture_globals = dict(capture.__globals__,**g['SCHEDULE'])
    exec(compile(tree,inspect.getsourcefile(capture)+'::capture_horizon_only_v2','exec'),capture_globals)
    adapted = capture_globals['capture']
    namespace = dict(original)
    namespace['_capture'] = adapted
    cls = original['CameraStream']
    attrs = {}
    for key,value in vars(cls).items():
        if key in ('__dict__','__weakref__'):
            continue
        if isinstance(value,types.FunctionType):
            value = types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
        attrs[key] = value
    namespace['CameraStream'] = type(cls.__name__,cls.__bases__,attrs)
    namespace['_g'] = namespace
    return namespace

def adapted_upstream(old):
    namespace = vars(_original_adapted_upstream(old)).copy()
    original_camera_runtime = namespace['camera_runtime']
    namespace['camera_runtime'] = lambda: camera_namespace(original_camera_runtime)
    return types.SimpleNamespace(**namespace)

def verify():
    value = _original_verify()
    proof = g['read'](HERE/'CAPTURE_HORIZON_PROOF.json')
    for path,digest in proof['extra_source_hashes'].items():
        if g['sha'](path)!=digest:
            raise ValueError('V1 partial/camera lineage drift '+path)
    return value

g.update(adapted_upstream=adapted_upstream,verify=verify,verify_prepared=verify)
globals().update({key:value for key,value in g.items() if not key.startswith('__')})
# Keep V2 wrappers visible after cloning V1 helpers.
globals().update(camera_namespace=camera_namespace,adapted_upstream=adapted_upstream,verify=verify)
