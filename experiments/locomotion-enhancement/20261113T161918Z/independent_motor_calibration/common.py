"""Strict storage API. STDLIB ONLY on import; numerical work is parent-call-only.
initialize()->(model,oldpayload); load_checkpoint(path,digest,final=True)->(m,p).
No physical backend, network execution, optimizer operation or file writes on import.
"""
from pathlib import Path
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
OLD_DIR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
OLD_CHECKPOINT=OLD_DIR/'COMBINED.pt'
OLD_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'
SAVE_STEPS=(0,300,600)
FIELDS={'format_version','step','model_state_dict','optimizer_state_dict','frozen_old_state_digest','config','rng','curve','identity'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def verify_hashes(hashes):
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Nonempty absolute source/input hash mapping required')
    for filename,digest in hashes.items():
        if not isinstance(filename,str) or not Path(filename).is_absolute() or not _digest(digest):raise ValueError('Absolute path and SHA256 required')
        if not Path(filename).is_file() or sha(filename)!=digest:raise ValueError('Source/input identity drift: '+filename)
    return True


def _check_config(config):
    if not isinstance(config,dict):raise ValueError('Config dict required')
    for key in ('source_hashes','data_hashes'):verify_hashes(config.get(key))
    required=(ROOT/'model.py',ROOT/'common.py',OLD_DIR/'model.py',OLD_DIR/'model_io.py')
    if any(config['source_hashes'].get(str(p))!=sha(p) for p in required):raise ValueError('New architecture/storage and strict old source hashes required')
    if config['data_hashes'].get(str(OLD_CHECKPOINT))!=OLD_SHA:raise ValueError('Frozen original789f must be a declared input')
    return True


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def _backend():
    name='_independent_speed_yaw_numeric_model'
    cached=sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve()!=ROOT/'model.py':raise ValueError('Numeric backend namespace collision')
        return cached
    return _module(name,ROOT/'model.py')


@contextmanager
def _preserve_rng():
    # Lazy, parent-only. Loading/checking never consumes the live training RNG.
    import random
    import numpy as np
    import torch
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    try:yield
    finally:
        random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])


def _require_neural_only():
    if any(name in sys.modules for name in ('physics','mujoco','student_wave','flygym')):raise ValueError('Strict motor loading must precede physical backend imports')


def _load_old():
    _require_neural_only()
    if sha(OLD_CHECKPOINT)!=OLD_SHA:raise ValueError('Frozen old789f checkpoint drift')
    sentinel=object();names=('model','common','_independent_old789f_arch','_independent_old789f_io')
    aliases={name:sys.modules.get(name,sentinel) for name in names};paths=list(sys.path)
    try:
        sys.path.insert(0,str(OLD_DIR))
        arch=_module('_independent_old789f_arch',OLD_DIR/'model.py');sys.modules['model']=arch
        loader=_module('_independent_old789f_io',OLD_DIR/'model_io.py')
        old,payload=loader.load_checked(OLD_CHECKPOINT,OLD_SHA,require_final=True)
    finally:
        sys.path[:]=paths
        for name,previous in aliases.items():
            if previous is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=previous
    if any(sys.modules.get(name,sentinel) is not value for name,value in aliases.items()) or sys.path!=paths:raise ValueError('model/common/path namespace restoration failed')
    return old,payload


def initialize():
    """Fresh zero-head adapter; old constructor RNG effects are restored.
    New adapter initialization consumes the caller's normal seeded Torch RNG.
    No old model.step / forward is performed by this function.
    """
    _require_neural_only()
    backend=_backend()
    with _preserve_rng():old,payload=_load_old()
    model=backend.IndependentMotor(old)
    model.verify_old();model.validate_architecture()
    return model,payload


def _validate_rng(rng):
    import random
    import numpy as np
    import torch
    if not isinstance(rng,dict) or set(rng)!={'torch','numpy','python'}:raise ValueError('Exact Torch/NumPy/Python RNG restoration state required')
    if not torch.is_tensor(rng['torch']) or rng['torch'].dtype!=torch.uint8 or rng['torch'].ndim!=1 or rng['torch'].device.type!='cpu':raise ValueError('CPU Torch uint8 RNG vector required')
    # Isolated generators validate encoding without touching live RNG.
    torch.Generator(device='cpu').set_state(rng['torch'])
    local_np=np.random.RandomState();local_np.set_state(rng['numpy'])
    local_python=random.Random();local_python.setstate(rng['python'])
    return True


def restore_rng(rng):
    """EXPLICIT resume operation; load_checkpoint itself preserves live RNG."""
    _validate_rng(rng)
    import random
    import numpy as np
    import torch
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])


def _validate_optimizer_state(state,model,step):
    import torch
    if not isinstance(state,dict) or set(state)!={'state','param_groups'}:raise ValueError('Complete Adam optimizer state_dict required')
    groups=state['param_groups']
    if not isinstance(groups,list) or not groups:raise ValueError('Adam param_groups required')
    order=[pid for group in groups for pid in group['params']]
    parameters=model.adapter_parameters()
    if len(order)!=8 or len(set(order))!=8 or any(type(pid) is not int for pid in order):raise ValueError('Exactly8 unique ordered adapter optimizer parameters')
    for group in groups:
        if 'betas' not in group or 'eps' not in group or not math.isfinite(float(group.get('lr',float('nan')))):raise ValueError('Adam hyperparameters required')
    slots=state['state']
    if not isinstance(slots,dict) or set(slots)-set(order):raise ValueError('Optimizer contains undeclared/frozen parameters')
    if step==0 and not slots:return True
    if set(slots)!=set(order):raise ValueError('All8 Adam moments required after updates')
    for pid,p in zip(order,parameters):
        slot=slots[pid]
        if set(slot) not in ({'step','exp_avg','exp_avg_sq'},{'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}):raise ValueError('Exact Adam moment slots required')
        counter=slot['step']
        if torch.is_tensor(counter):
            if counter.numel()!=1 or counter.device.type!='cpu' or not bool(torch.isfinite(counter).all()):raise ValueError('Finite CPU Adam step scalar')
            counter=float(counter)
        if float(counter)!=step:raise ValueError('All8 Adam counters must equal saved update')
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key in slot:
                value=slot[key]
                if not torch.is_tensor(value) or value.shape!=p.shape or value.dtype!=p.dtype or value.device.type!='cpu' or not bool(torch.isfinite(value).all()):raise ValueError('Adam moment shape/dtype/finite mismatch')
    return True


def _validate_live_optimizer(opt,model):
    flattened=[p for group in opt.param_groups for p in group['params']]
    expected=model.adapter_parameters()
    if len(flattened)!=8 or any(p is not q for p,q in zip(flattened,expected)):raise ValueError('Optimizer must own exactly8 adapter parameters in canonical order, no oldweights')


def _validate_payload(payload,model,final):
    import torch
    if not isinstance(payload,dict) or set(payload)!=FIELDS or payload['format_version']!=FORMAT:raise ValueError('Strict independent motor payload fields/format required')
    step=payload['step']
    if type(step) is not int or step not in SAVE_STEPS or (final and step!=600):raise ValueError('Final600 or explicit nonfinal0/300/600 only')
    _check_config(payload['config'])
    if payload['identity']!=model.identity() or payload['frozen_old_state_digest']!=model.frozen_old_state_digest:raise ValueError('Model architecture/oldfrozen identity mismatch')
    state=payload['model_state_dict'];canonical=model.state_dict()
    if not isinstance(state,dict) or state.keys()!=canonical.keys():raise ValueError('Exact old+new model_state_dict keys required')
    for name,tensor in state.items():
        expected=canonical[name]
        if not torch.is_tensor(tensor) or tensor.shape!=expected.shape or tensor.dtype!=expected.dtype or tensor.device.type!='cpu' or not bool(torch.isfinite(tensor).all()):raise ValueError('Saved model tensor shape/dtype/finite mismatch: '+name)
    _validate_optimizer_state(payload['optimizer_state_dict'],model,step);_validate_rng(payload['rng'])
    curve=payload['curve']
    if not isinstance(curve,list) or not curve or not isinstance(curve[-1],dict) or curve[-1].get('step')!=step:raise ValueError('Saved curve final step must bind checkpoint update')
    return True


def load_checkpoint(path,digest,final=True):
    """Strict loader; returns eval model with adapter trainability retained and payload.
    Preserves caller RNG during constructors/storage validation; resume requires
    explicit opt.load_state_dict(payload['optimizer_state_dict']) + restore_rng.
    """
    _require_neural_only();path=Path(path).resolve()
    if not _digest(digest) or sha(path)!=digest:raise ValueError('Actual checkpoint SHA256 required')
    import torch
    with _preserve_rng():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if not isinstance(payload,dict) or set(payload)!=FIELDS or payload.get('format_version')!=FORMAT:raise ValueError('Strict payload format before constructors')
        _check_config(payload['config'])
        model,_=initialize();_validate_payload(payload,model,final)
        model.load_state_dict(payload['model_state_dict'],strict=True)
        model.verify_old();model.validate_architecture();model.eval()
        if sha(path)!=digest:raise ValueError('Checkpoint changed during load')
        _check_config(payload['config'])
    return model,payload


def save_checkpoint(path,model,opt,config,curve,step,rng):
    """Exclusive atomic publication via same-directory temp+hardlink (no overwrite).
    Saves requested0/300/600 only; does not claim numerical independent replay.
    """
    _require_neural_only()
    import torch
    if type(step) is not int or step not in SAVE_STEPS:raise ValueError('Only0/300/600 save points')
    model.verify_old();model.validate_architecture();_validate_live_optimizer(opt,model)
    payload=dict(format_version=FORMAT,step=step,model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state_dict=copy.deepcopy(opt.state_dict()),frozen_old_state_digest=model.frozen_old_state_digest,config=copy.deepcopy(config),rng=copy.deepcopy(rng),curve=copy.deepcopy(curve),identity=copy.deepcopy(model.identity()))
    _validate_payload(payload,model,False)
    path=Path(path).resolve()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name);torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        model.verify_old();_check_config(config)
        # link fails atomically if another publisher created the final path.
        os.link(temp,path)
        return sha(path)
    finally:
        if temp is not None and temp.exists():temp.unlink()
