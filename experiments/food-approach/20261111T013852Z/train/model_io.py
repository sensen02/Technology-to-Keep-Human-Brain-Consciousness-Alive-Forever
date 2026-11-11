"""Exact bounded readout continuation storage; immutable original class."""
from pathlib import Path
import importlib.util,sys
import torch
HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/'20261111T010218Z/train'
spec=importlib.util.spec_from_file_location('_continued_positive_original_io',OLD/'model_io.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
sha256=legacy.sha256;verify_hashes=legacy.verify_hashes;exact=legacy.exact
ReadoutMotor=legacy.ReadoutMotor;BASE=legacy.BASE;REFERENCE=legacy.REFERENCE
PARENT=OLD/'TRAIN/checkpoint_002000.pt'
PARENT_SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'
FROZEN_PARENT=legacy.PARENT;FROZEN_SHA=legacy.PARENT_SHA

def load_parent():
 model,p=legacy.load_checked(PARENT,PARENT_SHA)
 if p['identity']['parent_checkpoint']!=str(FROZEN_PARENT) or p['identity']['parent_sha256']!=FROZEN_SHA:raise ValueError('actual frozen motor ancestry mismatch')
 if [r['step'] for r in p['curve']]!=list(range(1,2001)):raise ValueError('parent curve1..2000 required')
 return model,p

def save(path,model,opt,config,curve,rng):
 path=Path(path).resolve()
 if not path.is_relative_to(HERE) or path.exists():raise ValueError('fresh new train output required')
 torch.save(dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
 return sha256(path)

def load_checked(path,sha,require_final=True):
 if sha256(path)!=sha:raise ValueError('checkpoint SHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=1 or p['config']['kind']!='exact_positive_readout10000':raise ValueError('new continuation format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 frozen,_=legacy.load_parent();model=ReadoutMotor(frozen)
 if model.identity()!=p['identity']:raise ValueError('model identity changed')
 model.load_state_dict(p['state_dict'],strict=True)
 verified,_=legacy.load_parent()
 if not exact(model.parent.state_dict(),verified.state_dict()):raise ValueError('frozen194878 motor changed')
 if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite weights')
 step=p['curve'][-1]['step']
 if step not in (4000,7000,10000) or p['config']['physical_eval_only_step']!=10000:raise ValueError('fixed step mismatch')
 if require_final and step!=10000:raise ValueError('only final10000 physical evaluation')
 if [r['step'] for r in p['curve']]!=list(range(1,step+1)):raise ValueError('curve must be1..N')
 return model.eval(),p
