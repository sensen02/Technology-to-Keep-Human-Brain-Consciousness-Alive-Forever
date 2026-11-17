"""One AST layer directly over v1 numerical loops; no nested stoprunner import."""
import ast
from common import ROOT,OLD,STOP

NAMES=('preflight.py','runner.py','audit_saved.py')


def fixed_action_AST_equal():
    def action(path):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='fixed_action')
    return ast.dump(action(ROOT/'teacher.py'),include_attributes=False)==ast.dump(action(STOP/'teacher.py'),include_attributes=False)


class LabelsAndCLI(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=='FULL_BODY_ALGORITHM_TEACHER_PIVOT_FEASIBILITY_V1':return ast.copy_location(ast.Constant(value='FIXED_PRIVATE_SHADOW_PIVOT_TEACHER_V2'),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        for kw in node.keywords:
            if kw.arg=='choices' and isinstance(kw.value,(ast.Tuple,ast.List)) and [getattr(x,'value',None) for x in kw.value.elts]==['stop','left_pivot','right_pivot']:
                kw.value=ast.copy_location(ast.Tuple(elts=[ast.Constant(value='left_pivot'),ast.Constant(value='right_pivot')],ctx=ast.Load()),kw.value)
        return node


def namespace(name):
    if name not in NAMES:raise ValueError('Only fixed v1 entries')
    if not fixed_action_AST_equal():raise ValueError('Numerical fixed_action differs from actual-qualified stopfactor')
    tree=ast.parse((OLD/name).read_text());numbers=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float,bool)]
    tree=LabelsAndCLI().visit(tree);ast.fix_missing_locations(tree)
    if numbers!=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float,bool)]:raise ValueError('V1 numerical AST changed')
    if name=='preflight.py':
        probe=ast.parse("reference_actual=teacher.old.make_teacher(c,'left_pivot',order_indices=order)\nreference_action,reference_diag=reference_actual.action(c,0.)\nchecks['same_initial_actual_vs_fixed_action_exact']=np.array_equal(action,reference_action)\nchecks['same_initial_actual_vs_fixed_private_qdes_exact']=np.array_equal(candidate.q_des,reference_actual.q_des)\nchecks['same_initial_actual_vs_fixed_target_exact']=np.array_equal(diag['desired_motor_target'],reference_diag['desired_motor_target'])\nchecks['fixed_action_AST_same_as_actual_qualified_stop']=True").body
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        position=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='candidate' and n.value.func.attr=='action')
        main.body[position+1:position+1]=probe;ast.fix_missing_locations(tree)
    ns=dict(__name__='_pivotfixed_same_'+name.replace('.','_'),__file__=str(ROOT/name))
    exec(compile(tree,str(ROOT/name)+'::frozen_v1_exact_numeric_AST','exec'),ns)
    ns['_old_numeric_AST_unchanged']=True
    return ns


def run(name):return namespace(name)['main']()
