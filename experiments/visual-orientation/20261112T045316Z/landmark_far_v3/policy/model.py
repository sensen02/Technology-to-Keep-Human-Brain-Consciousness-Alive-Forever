"""Exact frozen original CNN/GRU class and production path; no new online gates."""
from common import module,V1
_original=module('_far_frozen_visual_architecture',V1/'model.py')
VisualPolicy=_original.VisualPolicy
Scan=_original.Scan
make=_original.make
production=_original.production
objective=_original.objective
