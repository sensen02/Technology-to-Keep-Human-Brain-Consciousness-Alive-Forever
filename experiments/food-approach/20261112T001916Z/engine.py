"""Frozen original shared-parameter scan, production path and loss exactly reused."""
from common import module,ORIGINAL
legacy=module('_paired_frozen_full1920_engine',ORIGINAL/'engine.py')
Scan=legacy.Scan;make=legacy.make;production=legacy.production;objective=legacy.objective
