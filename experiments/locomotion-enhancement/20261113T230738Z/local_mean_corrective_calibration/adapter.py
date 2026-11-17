"""Stdlib exact immutable state-comparison helper extraction; no numericimport."""
from pathlib import Path
import ast
import hashlib
import importlib.util

PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration')
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration')
g={'__name__':'_localmean_saved_state_comparators','hashlib':hashlib}
tree=ast.parse((OLD/'preflight.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tensor_digest','max_error')],type_ignores=[]),str(OLD/'preflight.py')+'::state_comparator_only','exec'),g)
max_error=g['max_error'];tensor_digest=g['tensor_digest']

def module(name,path):
    import sys
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
