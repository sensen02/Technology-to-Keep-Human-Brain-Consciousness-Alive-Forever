"""Exact frozen original visual class/production; mirror only in offline batch adapter."""
from common import module,V1
_original=module('_mirror_original_visual_architecture',V1/'model.py')
VisualPolicy=_original.VisualPolicy
Scan=_original.Scan
make=_original.make
production=_original.production
objective=_original.objective
