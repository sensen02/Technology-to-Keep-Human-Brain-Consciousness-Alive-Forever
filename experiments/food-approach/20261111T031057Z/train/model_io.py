"""Strict new storage for conditional RH-stance weighted readout continuation."""
from pathlib import Path
import importlib.util,sys
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT.parent/'20261111T010218Z/train'
PARENT=OLD/'TRAIN/checkpoint_002000.pt'
PARENT_SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'
spec=importlib.util.spec_from_file_location('_rh_weighted_original_io',OLD/'model_io.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
ReadoutMotor=legacy.ReadoutMotor
sha256=legacy.sha256
exact=legacy.exact
verify_hashes=legacy.verify_hashes
BASE=legacy.BASE
REFERENCE=legacy.REFERENCE
FROZEN_PARENT=legacy.PARENT
FROZEN_SHA=legacy.PARENT_SHA

def load_parent():return legacy.load_checked(PARENT,PARENT_SHA,require_final=True)

def save(path,model,opt,config,curve,rng):
 path=Path(path).resolve()
 if not path.is_relative_to(HERE) or path.exists():raise ValueError('new train-scoped path required')
 torch.save(dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
 return sha256(path)

def load_checked(checkpoint,sha,require_final=True):
 if sha256(checkpoint)!=sha:raise ValueError('checkpoint digest mismatch')
 p=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if p.get('format_version')!=1 or p['config']['kind']!='fixed_RHstance_weighted_readout8000':raise ValueError('weighted readout format mismatch')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 model,_=load_parent()
 if p['identity']!=model.identity():raise ValueError('architecture changed')
 model.load_state_dict(p['state_dict'],strict=True)
 frozen,_=load_parent()
 if not exact(model.parent.state_dict(),frozen.parent.state_dict()):raise ValueError('frozen194878 motor changed')
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite model weights')
 if p['config']['physical_eval_only_step']!=8000 or p['curve'][-1]['step'] not in (5000,8000):raise ValueError('registered fixed budget changed')
 if require_final and p['curve'][-1]['step']!=8000:raise ValueError('physical evaluation final8000 only')
 return model.eval(),p
