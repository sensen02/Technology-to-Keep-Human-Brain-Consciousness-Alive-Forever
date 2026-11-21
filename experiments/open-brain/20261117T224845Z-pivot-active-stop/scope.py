from pathlib import Path
import sys,json,copy,time
HERE=Path(__file__).resolve().parent
STOP=HERE.parent/'20261117T202200Z-body-stop-repair'
sys.path.insert(0,str(STOP))
from common import *
from policy_repair import read
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'20261117T211320Z-body-command-calibration'
WEIGHT=PREVIOUS/'FIT/epoch040.pt'
FAMILIES=('pivot_left','pivot_right','stop60','stop74a','stop74b')
SEEDS=(7013,7523,8861)

def freeze(extra=()):
 paths=set(HERE.glob('*.py'))|{HERE/'drive.sh',HERE/'PROTOCOL.zh-CN.md',WEIGHT}
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if file and Path(file).is_file() and str(file).startswith(str(ROOT)):paths.add(Path(file).resolve())
 assets=json.loads((TEACHER/'PREPARE_INPUTS.json').read_text())['source_sha256']
 assert all(sha(p)==v for p,v in assets.items())
 result={str(p):sha(p) for p in paths|set(map(Path,extra))};result.update(assets);return result
