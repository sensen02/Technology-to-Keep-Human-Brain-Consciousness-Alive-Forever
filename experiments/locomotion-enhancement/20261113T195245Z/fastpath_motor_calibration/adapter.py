"""Single immutable entry AST reuse, explicit newdirectory globals/imports."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parent
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration')
NAMES=('train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py')


def namespace(name):
    if name not in NAMES:raise ValueError('Registeredentryonly')
    tree=ast.parse((PARENT/name).read_text())
    class Metadata(ast.NodeTransformer):
        def visit_Constant(self,node):
            if name=='verify_saved.py' and type(node.value) is float and node.value==1e-10:
                return ast.copy_location(ast.Constant(value=1e-12),node)
            if isinstance(node.value,str):
                value=node.value.replace('INDEPENDENT_MOTOR','FASTPATH_MOTOR').replace('independent_body_command_calibration','fastpath_joint_repair_calibration')
                if value!=node.value:return ast.copy_location(ast.Constant(value=value),node)
            return node
    tree=Metadata().visit(tree);ast.fix_missing_locations(tree)
    ns={'__name__':'_fastpath_same_'+name.replace('.','_'),'__file__':str(ROOT/name)}
    exec(compile(tree,str(ROOT/name)+'::immutable_parent_entry_AST','exec'),ns)
    return ns


def run(name):namespace(name)['main']()
