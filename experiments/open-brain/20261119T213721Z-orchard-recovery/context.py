from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PREV = HERE.parent / '20261119T205558Z-orchard-curriculum'
sys.path.insert(0, str(PREV))
import rollout
from rollout import tr
