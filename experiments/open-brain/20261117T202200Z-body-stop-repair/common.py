from pathlib import Path
import sys,hashlib,json,platform
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'20261117T192322Z-terrain-body-learning'
sys.path.insert(0,str(PARENT))
from shared import *
from policy_plus import read as legacy_read
from proprioception import extra_live
HERE=Path(__file__).resolve().parent
CHECKPOINT=PARENT/'FIT2/epoch030.pt'

def manifest(extra=()):
    paths=set(HERE.glob('*.py'))|{HERE/'drive.sh',CHECKPOINT,TEACHER/'CANONICAL_INIT.npz',COMMAND_PATH}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).is_file() and str(file).startswith(str(ROOT)):paths.add(Path(file).resolve())
    assets=json.loads((TEACHER/'PREPARE_INPUTS.json').read_text())['source_sha256']
    assert all(sha(p)==v for p,v in assets.items())
    result={str(p):sha(p) for p in paths|set(map(Path,extra))};result.update(assets)
    return result

def verify_sources(values):
    assert all(sha(p)==v for p,v in values.items())
