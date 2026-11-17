"""Fixed experiment locations and the source-qualified, nonplastic neck."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
NECK=HERE.parent/'20261117T164207Z-fixed-neck'
GROUND=HERE.parent/'20261117T071007Z-ground-dn-response'
EON=HERE.parent/'20261117T064507Z-eon-overlap'
sys.path.insert(0,str(NECK))
import adapters
adapters.install(True)
from adapters import phase
import body_run as first
import expansion as study
from fixed_neck import hash_file, FixedCircuit
sys.path.insert(0,str(GROUND))
import ground_response as ground
sys.path.insert(0,str(EON))
import overlap

OLD_WEIGHT=study.HERE/'PHASE/FIT/epoch015.pt'
