"""Exactly four preregistered bias conditions; same frozen orchestration/scoring."""
from pathlib import Path
_PREDECESSOR=Path(__file__).resolve().parent.parent/'20261111T024558Z/pipeline.py'
exec(compile(_PREDECESSOR.read_text(),str(Path(__file__).resolve())+'::frozen-predecessor','exec'),globals())
