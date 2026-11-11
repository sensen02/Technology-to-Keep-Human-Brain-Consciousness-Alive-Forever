"""Frozen full physical rollout; only candidate neural parameter identity differs."""
from pathlib import Path
_PREDECESSOR=Path(__file__).resolve().parent.parent/'20261111T024558Z/evaluate_motor.py'
exec(compile(_PREDECESSOR.read_text(),str(Path(__file__).resolve())+'::frozen-predecessor','exec'),globals())
