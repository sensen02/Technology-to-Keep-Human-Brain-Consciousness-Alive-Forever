"""Use exact frozen old full1920 shared-parameter scan and objective, no new runtime logic."""
from common import module,OLD
legacy=module('_segmented_frozen_scan',OLD/'engine.py')
Scan=legacy.Scan;make=legacy.make;production=legacy.production;objective=legacy.objective
