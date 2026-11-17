"""Stdlib single-layer immutable original AST, metadata identity substitutions only."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import types

ROOT=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component')
CAL=ROOT.parent/'fastpath_motor_calibration'
FORMAT='INDEPENDENT_FASTPATH_MOTOR_COMPONENT_BODY_V1'
MOTOR_FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1'
CORE=('PREREG.json','common.py','prepare.py','preflight.py','runner.py','audit.py','run.py','adapter.py','OLD_SOURCE_SNAPSHOT.json','static_check.py','DESIGN.zh-CN.md')
ENTRIES=('common.py','preflight.py','runner.py','audit.py','run.py')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def numeric(tree):
    return collections.Counter((type(n.value).__name__,repr(n.value)) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float,complex))


def transformed(name):
    if name not in ENTRIES:raise ValueError('Original entry whitelist')
    snapshot=json.loads((ROOT/'OLD_SOURCE_SNAPSHOT.json').read_text())
    for path,digest in snapshot['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Immutable original source drift: '+path)
    original=ast.parse((OLD/name).read_text());before=numeric(original)
    class Identity(ast.NodeTransformer):
        def visit_Assign(self,node):
            if name=='common.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                values={'ROOT':repr(str(ROOT)),'CAL':repr(str(CAL)),'TEACHER':repr(str(OLD.parent/'pivot_teacher_design')),'FORMAT':repr(FORMAT),'FILES':repr(CORE)}
                key=node.targets[0].id
                if key in values:
                    text=('Path('+values[key]+')') if key in ('ROOT','CAL','TEACHER') else values[key]
                    node.value=ast.parse(text,mode='eval').body
                    return node
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value=='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1':return ast.copy_location(ast.Constant(MOTOR_FORMAT),node)
            return node
    tree=Identity().visit(original);ast.fix_missing_locations(tree)
    if numeric(tree)!=before:raise ValueError('Numerical literal multiset changed')
    return tree


def namespace(name):
    ns={'__name__':'_fastpath_body_original_'+name.replace('.','_'),'__file__':str(ROOT/name)}
    exec(compile(transformed(name),str(OLD/name)+'::single_layer_new_identity','exec'),ns)
    if ns.get('ROOT')!=ROOT or ns.get('FORMAT')!=FORMAT:raise ValueError('Wrong new ROOT/FORMAT')
    def inspect(value):
        if isinstance(value,types.FunctionType) and value.__module__==ns['__name__']:
            if value.__globals__ is not ns or value.__globals__.get('ROOT')!=ROOT:raise ValueError('Stale function globals')
        elif isinstance(value,type) and value.__module__==ns['__name__']:
            for method in vars(value).values():
                inspect(method.__func__ if isinstance(method,(classmethod,staticmethod)) else method)
    for value in list(ns.values()):inspect(value)
    return ns


def run(name):namespace(name)['main']()
