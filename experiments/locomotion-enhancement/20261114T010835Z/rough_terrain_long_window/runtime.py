"""Long-window isolated namespace; stdlib only until explicit parent execution."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import textwrap
import types

HERE = Path(__file__).resolve().parent
SOURCE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
REPAIR = SOURCE/'terrain_start_recovery_v1'
NEURAL = SOURCE/'terrain_neural_baseline'
LOOP = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body')
CASE = 'mild_blocks_seed17_start_repair_v1_long19p2'
OLD_CASE = 'mild_blocks_seed17_start_repair_v1'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_LONG19P2_V1'
ANN_MS = 19200  # The only independently set new duration.
SETTLE_MS, WARM_MS, HANDOVER_MS = 1000, 4800, 5800
TOTAL_MS = HANDOVER_MS + ANN_MS
DECISIONS = TOTAL_MS // 5
FRAMES = TOTAL_MS // 20
POST_NATIVE = ANN_MS * 80
TOTAL_NATIVE = TOTAL_MS * 80
SCHEDULE = dict(ANN_MS=ANN_MS,TOTAL_MS=TOTAL_MS,TOTAL_MS_PLUS_ONE=TOTAL_MS+1,
    DECISIONS=DECISIONS,FRAMES=FRAMES,POST_NATIVE=POST_NATIVE,TOTAL_NATIVE=TOTAL_NATIVE,
    ANN_MS_PLUS_ONE=ANN_MS+1)
SYMBOLS = {9600:'ANN_MS',15400:'TOTAL_MS',15401:'TOTAL_MS_PLUS_ONE',
    3080:'DECISIONS',770:'FRAMES',768000:'POST_NATIVE',1232000:'TOTAL_NATIVE',9601:'ANN_MS_PLUS_ONE'}
sys.dont_write_bytecode = True

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')

def scheduled_tree(source,path):
    """No changed numeric constants: replace registered duration nodes with names.

    Inverse substitution restores the complete original AST, including all numbers.
    Strings are retained as historical diagnostic labels, not numerical gates.
    """
    original = ast.parse(source)
    changes = []
    class Symbols(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in SYMBOLS:
                symbol = SYMBOLS[node.value]
                changes.append(dict(source=str(path),line=node.lineno,column=node.col_offset,
                    old_value=node.value,new_expression=symbol,new_value=SCHEDULE[symbol]))
                result = ast.copy_location(ast.Name(id=symbol,ctx=ast.Load()),node)
                result._old_duration_value = node.value
                return result
            return node
    tree = Symbols().visit(copy.deepcopy(original))
    restored = copy.deepcopy(tree)
    class Inverse(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.copy_location(ast.Constant(node._old_duration_value),node) if hasattr(node,'_old_duration_value') else node
    restored = Inverse().visit(restored)
    if ast.dump(original,include_attributes=True)!=ast.dump(restored,include_attributes=True):
        raise ValueError('Non-reversible duration adaptation '+str(path))
    ast.fix_missing_locations(tree)
    return tree,changes

def load(name,path,*,scheduled=False):
    path = Path(path)
    if scheduled:
        tree,_ = scheduled_tree(path.read_text(),path)
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__dict__.update(SCHEDULE)
        sys.modules[name] = module
        exec(compile(tree,str(path)+'::long19p2','exec'),module.__dict__)
        return module
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def verify():
    proof = read(HERE/'FROZEN.json')
    for path,digest in proof['source_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('Frozen source drift '+path)
    return proof

def verify_prepared():
    return verify()

def scene_metadata():
    old = read(REPAIR/'PREPARED/MODEL_LEDGER.json')['cases'][OLD_CASE]
    new = read(HERE/'PREPARED/MODEL_LEDGER.json')['cases'][CASE]
    return dict(kind=KIND,scene_version=CASE,case=CASE,first_block_front_world_x=1.35,
        new_scene_xml=new['xml'],new_scene_xml_sha256=new['xml_sha256'],
        source_scene_xml=old['xml'],source_scene_xml_sha256=old['xml_sha256'],
        XML_bytes_identical=sha(old['xml'])==sha(new['xml']),
        only_scientific_change='autonomous window 9.6 to 19.2 seconds',
        schedule=dict(SCHEDULE),archived_control_physics_repeated=False,
        original_rough_archive=str(REPAIR/'trials'/OLD_CASE/'RESULT.json'))

def adapt_function(function):
    source = textwrap.dedent(inspect.getsource(function))
    tree,_ = scheduled_tree(source,inspect.getsourcefile(function))
    namespace = dict(function.__globals__,**SCHEDULE)
    exec(compile(tree,inspect.getsourcefile(function)+'::long19p2','exec'),namespace)
    return namespace[function.__name__]

def adapted_upstream(old):
    # Keep load_policies object/globals exactly: old064313 weights and identities.
    namespace = dict(vars(old))
    namespace['stable'] = adapt_function(old.stable)
    cls = old.BrainNeckLoop
    attrs = {key:value for key,value in vars(cls).items() if key not in ('__dict__','__weakref__')}
    attrs['finalize'] = adapt_function(cls.finalize)
    namespace['BrainNeckLoop'] = type(cls.__name__,cls.__bases__,attrs)
    return types.SimpleNamespace(**namespace)

def routed_load(name,path):
    path = Path(path)
    if path.resolve() in ((NEURAL/'runtime.py').resolve(),(NEURAL/'metrics.py').resolve()):
        module = load(name,path,scheduled=True)
        if path.name=='runtime.py':
            original_upstream = module.upstream
            module.upstream = lambda: adapted_upstream(original_upstream())
            module.module = routed_load
        return module
    return load(name,path)

def context(*,audit=False):
    repair = load('_long_frozen_repair',REPAIR/'runtime.py')
    functions = repair.clone_functions(repair,dict(HERE=HERE,CASE=CASE,KIND=KIND,
        load=routed_load,verify=verify,verify_prepared=verify_prepared,scene_metadata=scene_metadata))
    namespace,state = functions['context'](audit=audit)
    namespace.update(SCHEDULE)
    namespace['SUMMARY_KIND'] = KIND+'_SUMMARY'
    return namespace,state

def loop_tree(name):
    adapter = load('_long_frozen_adapter',NEURAL/'adapter.py')
    tree,route_changes = adapter.tree(name)
    # Preserve the old adapter's routing; duration symbols are a second explicit layer.
    source = ast.unparse(tree)
    adapted,changes = scheduled_tree(source,LOOP/name)
    # Ledger uses actual source line positions, rather than unparse line positions.
    _,source_changes = scheduled_tree((LOOP/name).read_text(),LOOP/name)
    if [(r['old_value'],r['new_expression']) for r in changes] != [(r['old_value'],r['new_expression']) for r in source_changes]:
        raise ValueError('Routing altered duration-site traversal')
    return adapted,source_changes,route_changes
