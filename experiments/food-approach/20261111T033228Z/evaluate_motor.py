"""New root binding of frozen calibration evaluator; no physics/scoring alteration."""
from pathlib import Path
import hashlib
_PREDECESSOR=Path(__file__).resolve().parent.parent/'20261111T024558Z/evaluate_motor.py'
_EXPECTED=''
# Source identity is frozen by preparation and included in every actual run.
exec(compile(_PREDECESSOR.read_text(),str(Path(__file__).resolve())+'::frozen-predecessor','exec'),globals())
