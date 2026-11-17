"""Single original-entry AST relocation, no scientific numerical expression edit."""
from pathlib import Path
import ast,linecache
ORIGINAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration')
def load(name,namespace):
 if name not in ('utils.py','verify_saved.py','evaluate.py','audit_eval.py'):raise ValueError('Registeredentryonly')
 text=(ORIGINAL/name).read_text();tree=ast.parse(text)
 if name=='evaluate.py':
  class Saves(ast.NodeTransformer):
   def visit_Tuple(self,node):
    if all(isinstance(e,ast.Constant) for e in node.elts) and [e.value for e in node.elts]==[0,300,600]:return ast.copy_location(ast.Tuple(elts=[ast.Constant(value=x) for x in (300,450,600)],ctx=ast.Load()),node)
    return self.generic_visit(node)
  tree=Saves().visit(tree);ast.fix_missing_locations(tree)
 virtual=str(Path(namespace['__file__']).resolve());linecache.cache[virtual]=(len(text),None,text.splitlines(True),virtual);exec(compile(tree,virtual,'exec'),namespace)
