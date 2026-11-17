"""Exact original neural architecture/loss/scan/accumulation, no scientific computation changed."""
from common import module,ORIGINAL
_core=module('_fusion_recovery_exact_original_model',ORIGINAL/'model.py')
NeuralBridge=_core.NeuralBridge;Scan=_core.Scan;make_scan=_core.make_scan;python_rollout=_core.python_rollout;objective=_core.objective;accumulated_backward=_core.accumulated_backward;torch=_core.torch
