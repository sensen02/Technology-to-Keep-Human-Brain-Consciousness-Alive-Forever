"""Strict alias to unchanged qualified-parent ANN, no tensor or class edits."""
from pathlib import Path
import importlib.util,sys
ROOT=Path(__file__).resolve().parent
PATH=ROOT.parent/'20261111T010218Z/train/model_io.py'
CHECKPOINT=ROOT.parent/'20261111T010218Z/train/TRAIN/checkpoint_002000.pt'
SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'
spec=importlib.util.spec_from_file_location('_half_unchanged_parent',PATH);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
sha256=old.sha256
def load_checked(path,digest,require_final=True):
 if Path(path).resolve()!=CHECKPOINT or digest!=SHA:raise ValueError('exact unchanged eeced2000 required')
 return old.load_checked(path,digest,require_final=True)
