"""Explicit AST reuse: same v1 numerical loop; new factory and provenance only."""
from pathlib import Path
import ast
import hashlib
from common import ROOT, OLD

NAMES=('preflight.py','runner.py','audit_saved.py')


class LabelsAndCLI(ast.NodeTransformer):
    def __init__(self,name): self.name=name
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            if self.name=='preflight.py' and node.value=='left_pivot':return ast.copy_location(ast.Constant(value='stop'),node)
            if node.value=='FULL_BODY_ALGORITHM_TEACHER_PIVOT_FEASIBILITY_V1':return ast.copy_location(ast.Constant(value='STOP_FIXED_SHADOW_MATCHED_CONTROL_V1'),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        for keyword in node.keywords:
            if keyword.arg=='choices' and isinstance(keyword.value,(ast.Tuple,ast.List)) and [getattr(x,'value',None) for x in keyword.value.elts]==['stop','left_pivot','right_pivot']:
                keyword.value=ast.copy_location(ast.Tuple(elts=[ast.Constant(value='stop')],ctx=ast.Load()),keyword.value)
        return node


def namespace(name):
    if name not in NAMES:raise ValueError('Only registered v1 entrypoints')
    original=ast.parse((OLD/name).read_text());original_numbers=[n.value for n in ast.walk(original) if isinstance(n,ast.Constant) and type(n.value) in (int,float,bool)]
    tree=LabelsAndCLI(name).visit(original);ast.fix_missing_locations(tree)
    if original_numbers != [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float,bool)]:
        raise ValueError('V1 numerical AST changed')
    if name=='preflight.py':
        probe=ast.parse("reference_actual=teacher.old.make_teacher(c,'stop',order_indices=order)\nreference_action,reference_diag=reference_actual.action(c,0.)\nchecks['same_initial_actual_vs_fixed_action_exact']=np.array_equal(action,reference_action)\nchecks['same_initial_actual_vs_fixed_private_qdes_exact']=np.array_equal(candidate.q_des,reference_actual.q_des)\nchecks['same_initial_actual_vs_fixed_target_exact']=np.array_equal(diag['desired_motor_target'],reference_diag['desired_motor_target'])").body
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        position=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='candidate' and n.value.func.attr=='action')
        main.body[position+1:position+1]=probe
        ast.fix_missing_locations(tree)
    ns=dict(__name__='_stopshadow_same_'+name.replace('.','_'),__file__=str(ROOT/name))
    exec(compile(tree,str(ROOT/name)+'::frozen_v1_exact_numeric_AST','exec'),ns)
    ns['_old_numeric_AST_unchanged']=True
    return ns


def run(name):return namespace(name)['main']()
