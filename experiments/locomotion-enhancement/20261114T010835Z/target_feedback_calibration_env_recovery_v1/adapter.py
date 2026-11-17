"""Stdlib one-layer immutable original AST, output-root relocation only."""
from pathlib import Path
import ast
ORIGINAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration')
EXPECTED_PYTHON=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')

def load(name,namespace):
 if name not in ('model.py','common.py','sources.py','data.py','scan.py','learning.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_eval.py','run.py','utils.py'):raise ValueError('Unregisteredimmutableentry')
 source=ORIGINAL/name;text=source.read_text();tree=ast.parse(text)
 # TorchScript/inspect must see exact original definitions at the relocated
 # module filename; mtime=None denotes a virtual immutable source cache.
 import linecache
 virtual=str(Path(namespace['__file__']).resolve())
 linecache.cache[virtual]=(len(text),None,text.splitlines(keepends=True),virtual)
 exec(compile(tree,virtual,'exec'),namespace)
