"""Bind the exact archived front-touch timing module despite same-name modules."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


repair = Path(__file__).resolve().parent.parent / "20261118T020425Z-front-touch-timing-repair"
spec = spec_from_file_location("p1_front_touch_timing_repair", repair / "timing.py")
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

FrontTouchNeck = module.FrontTouchNeck
CanonicalEvents = module.CanonicalEvents
