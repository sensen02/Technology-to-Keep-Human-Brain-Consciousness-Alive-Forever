"""Same frozen v1 architecture/production paths; only trainable weights change."""
from common import module,OLD
_original=module('_wallvision_original_network',OLD/'model.py')
VisualPolicy=_original.VisualPolicy;Scan=_original.Scan;make=_original.make;production=_original.production;objective=_original.objective
