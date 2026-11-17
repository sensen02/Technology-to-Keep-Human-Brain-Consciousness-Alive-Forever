"""Direct original body AST; only identities and four finalstep guards600->900."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import types

ROOT=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component')
CAL=ROOT.parent/'state_aligned_action_calibration/wall_replan_9000'
FORMAT='STATE_ALIGNED_ACTION_MOTOR_COMPONENT_BODY_V1'
MOTOR_FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
FINAL_STEP=900
SOURCE_ID=3
CORE=('PREREG.json','common.py','prepare.py','preflight.py','runner.py','audit.py','run.py','adapter.py','OLD_SOURCE_SNAPSHOT.json','static_check.py','DESIGN.zh-CN.md')
ENTRIES=('common.py','preflight.py','runner.py','audit.py','run.py')
FINAL_GUARD_COUNTS={'common.py':1,'preflight.py':2,'runner.py':1,'audit.py':0,'run.py':0}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def numeric(tree):
    return collections.Counter((type(n.value).__name__,repr(n.value)) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float,complex))


def transformed(name):
    if name not in ENTRIES:raise ValueError('Original entry whitelist')
    snapshot=json.loads((ROOT/'OLD_SOURCE_SNAPSHOT.json').read_text())
    for path,digest in snapshot['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('Immutable original source drift: '+path)
    tree=ast.parse((OLD/name).read_text());original_numeric=numeric(tree);changed=[]
    class Identity(ast.NodeTransformer):
        def visit_Assign(self,node):
            if name=='common.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                replacements={'ROOT':repr(str(ROOT)),'CAL':repr(str(CAL)),'TEACHER':repr(str(OLD.parent/'pivot_teacher_design')),'FORMAT':repr(FORMAT),'FILES':repr(CORE)}
                key=node.targets[0].id
                if key in replacements:
                    text='Path('+replacements[key]+')' if key in ('ROOT','CAL','TEACHER') else replacements[key]
                    node.value=ast.parse(text,mode='eval').body;return node
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==600:
                changed.append(node.lineno);return ast.copy_location(ast.Constant(FINAL_STEP),node)
            if node.value=='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1':return ast.copy_location(ast.Constant(MOTOR_FORMAT),node)
            if isinstance(node.value,str) and 'final600' in node.value:return ast.copy_location(ast.Constant(node.value.replace('final600','final900')),node)
            return node
    tree=Identity().visit(tree);ast.fix_missing_locations(tree)
    if len(changed)!=FINAL_GUARD_COUNTS[name]:raise ValueError('Only explicitly registered finalstep guards may change')
    expected=original_numeric.copy()
    if changed:
        expected[('int','600')]-=len(changed)
        if expected[('int','600')]==0:del expected[('int','600')]
        expected[('int','900')]+=len(changed)
    if numeric(tree)!=expected:raise ValueError('Unexpected nonmetadata numerical change')
    return tree


def namespace(name):
    ns={'__name__':'_statealigned_original_body_'+name.replace('.','_'),'__file__':str(ROOT/name)}
    exec(compile(transformed(name),str(OLD/name)+'::single_entry_new_identity_final900','exec'),ns)
    if name=='common.py':ns['SOURCE_CANDIDATE']=SOURCE_ID
    if ns.get('ROOT')!=ROOT or ns.get('FORMAT')!=FORMAT or ns.get('SOURCE_CANDIDATE')!=SOURCE_ID:raise ValueError('Wrong new body/source namespace')
    def check(value):
        if isinstance(value,types.FunctionType) and value.__module__==ns['__name__']:
            if value.__globals__ is not ns or value.__globals__.get('ROOT')!=ROOT or value.__globals__.get('SOURCE_CANDIDATE')!=SOURCE_ID:raise ValueError('Stale function globals')
        elif isinstance(value,type) and value.__module__==ns['__name__']:
            for method in vars(value).values():check(method.__func__ if isinstance(method,(classmethod,staticmethod)) else method)
    for value in list(ns.values()):check(value)
    return ns


def run(name):namespace(name)['main']()
