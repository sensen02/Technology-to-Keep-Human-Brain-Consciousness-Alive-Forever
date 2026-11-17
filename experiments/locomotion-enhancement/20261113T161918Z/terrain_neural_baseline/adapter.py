"""One explicit AST namespace over actual validated simultaneous sensory loops.

Alterations are structural artifact routing, derived identity/measurement proxies
and terrain-specific warm comparison applicability. All numerical loop bodies,
NN calls, source queue timing and physical action schedules remain unchanged.
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOOP = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body')


def tree(name):
    if name not in ('runner.py','audit.py'):
        raise ValueError(name)
    original = ast.parse((LOOP/name).read_text())
    route = ast.dump(ast.parse("ROOT/'trials'/a.condition",mode='eval').body)
    changes = {'route':0,'runtime_import':0,'old_warm_loops':0}

    class DerivedNamespace(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module == 'runtime':
                changes['runtime_import'] += 1
                return None
            return node

        def visit_BinOp(self,node):
            if ast.dump(node) == route:
                changes['route'] += 1
                return ast.copy_location(ast.parse("ROOT/'trials'/TERRAIN_CASE",mode='eval').body,node)
            return self.generic_visit(node)

        def visit_For(self,node):
            node = self.generic_visit(node)
            labels = [item.value for item in ast.walk(node) if isinstance(item,ast.Constant) and isinstance(item.value,str)]
            if any(text.startswith(('physicalwarmunchanged:','motorprimeunchanged:')) for text in labels):
                changes['old_warm_loops'] += 1
                return ast.copy_location(ast.If(test=ast.Compare(left=ast.Name(id='TERRAIN_CASE',ctx=ast.Load()),
                    ops=[ast.Eq()],comparators=[ast.Constant('plane_control')]),body=[node],orelse=[]),node)
            return node

    result = DerivedNamespace().visit(original)
    expected = dict(route=1,runtime_import=1,old_warm_loops=2 if name=='audit.py' else 0)
    if changes != expected:
        raise ValueError('Exact original AST contract changed '+repr((changes,expected)))
    ast.fix_missing_locations(result)
    return result,changes


def run(name,case):
    import runtime
    namespace,state = runtime.context(case,audit=name=='audit.py')
    namespace.update(__name__='_terrain_neural_frozen_'+name.replace('.','_'),
                     __file__=str(LOOP/name))
    parsed,changes = tree(name)
    exec(compile(parsed,str(LOOP/name)+'::explicit_terrain_namespace_v1','exec'),namespace)
    # The frozen numerical loop always runs V_empty; external case controls only
    # physical terrain and artifact routing. No terrain label enters neural input.
    previous = list(sys.argv)
    sys.argv = [str(HERE/name),'--condition','V_empty','--parent-resource-released']
    try:
        namespace['main']()
    finally:
        sys.argv = previous
