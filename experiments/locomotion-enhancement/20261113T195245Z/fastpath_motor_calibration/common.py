"""Strict fastpath storage, STDLIB ONLY at import. No numerical execution on import."""
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
ANCESTOR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
PARENT_DIR=ANCESTOR/'independent_motor_calibration'
PARENT_CHECKPOINT=PARENT_DIR/'TRAIN/checkpoint_000600.pt'
PARENT_SHA='454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398'
OLD_DIR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
OLD_CHECKPOINT=OLD_DIR/'COMBINED.pt'
OLD_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1'
SAVE_STEPS=(0,300,600)
FIELDS={'format_version','step','model_state_dict','optimizer_state_dict','frozen_old_state_digest','config','rng','curve','identity'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def _digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _read(path):return json.loads(Path(path).read_text())
def verify_hashes(hashes):
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Nonemptyabsolute source/inputhashes')
    for filename,digest in hashes.items():
        if not isinstance(filename,str) or not Path(filename).is_absolute() or not _digest(digest) or not Path(filename).is_file() or sha(filename)!=digest:raise ValueError('Source/inputidentitydrift: '+str(filename))
    return True


def parent_qualification_hashes():
    """Stdlib actualcompleted parent600/storage/A175/floor178 qualification only."""
    if sha(PARENT_CHECKPOINT)!=PARENT_SHA or sha(OLD_CHECKPOINT)!=OLD_SHA:raise ValueError('Actualparent600/original789fSHA required')
    tr=_read(PARENT_DIR/'TRAIN/RESULT.json');stage=_read(PARENT_DIR/'STAGE_VERIFICATION.json');jobs=_read(PARENT_DIR/'JOBS.json');evaluation=_read(PARENT_DIR/'EVALUATION.json');ev=_read(PARENT_DIR/'EVALUATION_VERIFY.json')
    if tr['updates']!=600 or tr['checkpoint']!=str(PARENT_CHECKPOINT) or tr['sha256']!=PARENT_SHA or not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint_sha256']!=PARENT_SHA:raise ValueError('Actualparentfixed600 completedpipeline required')
    if len(jobs)!=3 or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in jobs):raise ValueError('Actual3parentstagesexit0 required')
    if not ev['integrity_all_pass'] or not all(ev['checks'].values()) or ev['evaluation_sha256']!=sha(PARENT_DIR/'EVALUATION.json') or evaluation['checkpoint_sha256']!=PARENT_SHA:raise ValueError('Actualparentfinalindependentmetrics required')
    paths=[PARENT_DIR/'model.py',PARENT_DIR/'common.py',PARENT_DIR/'TRAIN/RESULT.json',PARENT_DIR/'STAGE_VERIFICATION.json',PARENT_DIR/'JOBS.json',PARENT_DIR/'EVALUATION.json',PARENT_DIR/'EVALUATION_VERIFY.json',PARENT_CHECKPOINT,OLD_CHECKPOINT]
    for step in SAVE_STEPS:
        path=PARENT_DIR/'TRAIN'/f'VERIFY_{step:06d}.json';v=_read(path)
        if not v['integrity_all_pass'] or not all(v['checks'].values()) or v['step']!=step or sha(v['checkpoint'])!=v['checkpoint_sha256']:raise ValueError('Actualparent0/300/600 storageproof required')
        paths.extend((path,Path(v['checkpoint'])))
    diagnostic=ANCESTOR/'independent_motor_failure_diagnostic';a=_read(diagnostic/'STAT.json');av=_read(diagnostic/'VERIFY.json');floor=_read(diagnostic/'REPRESENTABILITY.json');fv=_read(diagnostic/'REPRESENTABILITY_VERIFY.json')
    if a['checkpoint_sha256']!=PARENT_SHA or not a['baseline_replay_all4_exact'] or not av['all_passed'] or not all(av['checks'].values()) or av['STAT_sha256']!=sha(diagnostic/'STAT.json') or av['report_sha256']!=sha(diagnostic/'REPORT.zh-CN.md'):raise ValueError('ActualstageA all4baselineproof required')
    if floor['checkpoint_sha256']!=PARENT_SHA or floor['global_equal8_action_floor']['action_MSE_lower_bound']!=0. or floor['global_equal8_action_floor']['strict_out_of_range_fraction']!=0. or len(floor['training_sequences'])!=8:raise ValueError('Actual8sourcezero amplitudefloor required; boundunchanged')
    if not fv['all_passed'] or not all(fv['checks'].values()) or fv['result_sha256']!=sha(diagnostic/'REPRESENTABILITY.json') or fv['report_sha256']!=sha(diagnostic/'REPRESENTABILITY.zh-CN.md') or fv['cache_sha256']!=sha(floor['cache']['path']):raise ValueError('Actualrepresentationindependentaudit required')
    paths += [diagnostic/n for n in ('STAT.json','VERIFY.json','REPORT.zh-CN.md','REPRESENTABILITY.json','REPRESENTABILITY_VERIFY.json','REPRESENTABILITY.zh-CN.md')]+[Path(floor['cache']['path'])]
    hashes={str(path):sha(path) for path in paths}
    for key in ('source_hashes','data_hashes'):verify_hashes(tr[key])
    return hashes


def _check_config(config):
    if not isinstance(config,dict):raise ValueError('Configdict required')
    for key in ('source_hashes','data_hashes'):verify_hashes(config.get(key))
    required=(ROOT/'model.py',ROOT/'common.py',PARENT_DIR/'model.py',PARENT_DIR/'common.py',OLD_DIR/'model.py',OLD_DIR/'model_io.py')
    if any(config['source_hashes'].get(str(p))!=sha(p) for p in required):raise ValueError('Fast/parent/original strictsourcehashes required')
    if config['data_hashes'].get(str(PARENT_CHECKPOINT))!=PARENT_SHA or config['data_hashes'].get(str(OLD_CHECKPOINT))!=OLD_SHA:raise ValueError('Actualparent600+original789f inputbinding')
    proof=parent_qualification_hashes()
    if any(config['data_hashes'].get(p)!=h for p,h in proof.items()):raise ValueError('Actualparentstorage/A/floorproof hashes required indata_hashes')
    return True


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def _backend():
    name='_independent_fastpath_repair_numeric_model';cached=sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve()!=ROOT/'model.py':raise ValueError('Fastpathbackendnamespacecollision')
        return cached
    return _module(name,ROOT/'model.py')

@contextmanager
def _preserve_rng():
    import random,numpy as np,torch
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    try:yield
    finally:random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])

def _require_neural_only():
    if any(name in sys.modules for name in ('physics','mujoco','student_wave','flygym')):raise ValueError('StrictNNload must precede physicalbackend')


def _load_parent():
    _require_neural_only();parent_qualification_hashes();sentinel=object();names=('model','common','sources','_fastpath_parent_strict_io');prior={k:sys.modules.get(k,sentinel) for k in names};paths=list(sys.path)
    try:
        loader=_module('_fastpath_parent_strict_io',PARENT_DIR/'common.py');parent,payload=loader.load_checkpoint(PARENT_CHECKPOINT,PARENT_SHA,final=True)
    finally:
        sys.path[:]=paths
        for name,value in prior.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    if sys.path!=paths or any(sys.modules.get(k,sentinel) is not v for k,v in prior.items()):raise ValueError('Parentloaderaliases/pathrestorefailed')
    return parent,payload


def initialize():
    """Strictparent600 -> exactparameter migration, no forward, callerRNG preserved."""
    _require_neural_only();backend=_backend()
    with _preserve_rng():parent,payload=_load_parent();model=backend.IndependentMotor(parent);model.verify_old();model.validate_architecture()
    return model,payload


def _validate_rng(rng):
    import random,numpy as np,torch
    if not isinstance(rng,dict) or set(rng)!={'torch','numpy','python'}:raise ValueError('Complete3RNGstate required')
    if not torch.is_tensor(rng['torch']) or rng['torch'].dtype!=torch.uint8 or rng['torch'].ndim!=1 or rng['torch'].device.type!='cpu':raise ValueError('CPUtorchuint8 RNG')
    torch.Generator(device='cpu').set_state(rng['torch']);np.random.RandomState().set_state(rng['numpy']);random.Random().setstate(rng['python']);return True

def restore_rng(rng):
    _validate_rng(rng)
    import random,numpy as np,torch
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])


def _validate_optimizer_state(state,model,step):
    import torch
    if not isinstance(state,dict) or set(state)!={'state','param_groups'}:raise ValueError('CompleteAdamstate required')
    groups=state['param_groups']
    if not isinstance(groups,list) or not groups:raise ValueError('Adamgroups required')
    order=[pid for group in groups for pid in group['params']];parameters=model.adapter_parameters()
    if len(order)!=8 or len(set(order))!=8 or any(type(pid) is not int for pid in order):raise ValueError('Canonical8uniqueadapter Adamparams')
    for group in groups:
        if 'betas' not in group or 'eps' not in group or not math.isfinite(float(group.get('lr',float('nan')))):raise ValueError('Adamhyperparameters')
    slots=state['state']
    if not isinstance(slots,dict) or set(slots)-set(order):raise ValueError('Undeclared/frozenoptimizerparameters')
    if step==0 and not slots:return True
    if set(slots)!=set(order):raise ValueError('All8Adam momentsrequired')
    for pid,p in zip(order,parameters):
        slot=slots[pid]
        if set(slot) not in ({'step','exp_avg','exp_avg_sq'},{'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}):raise ValueError('ExactAdamslots')
        counter=slot['step']
        if torch.is_tensor(counter):
            if counter.numel()!=1 or counter.device.type!='cpu' or not bool(torch.isfinite(counter).all()):raise ValueError('FiniteCPUstepcounter')
            counter=float(counter)
        if float(counter)!=step:raise ValueError('8Adamcountersmustequalsavedupdate')
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key in slot:
                value=slot[key]
                if not torch.is_tensor(value) or value.shape!=p.shape or value.dtype!=p.dtype or value.device.type!='cpu' or not bool(torch.isfinite(value).all()):raise ValueError('Adamshape/dtype/finite')
    return True

def _validate_live_optimizer(opt,model):
    flat=[p for group in opt.param_groups for p in group['params']];expected=model.adapter_parameters()
    if len(flat)!=8 or any(p is not q for p,q in zip(flat,expected)):raise ValueError('Optimizerexactordered8newtensors only')


def _validate_payload(payload,model,final):
    import torch
    if not isinstance(payload,dict) or set(payload)!=FIELDS or payload['format_version']!=FORMAT:raise ValueError('Strictfastpathpayloadfields/FORMAT')
    step=payload['step']
    if type(step) is not int or step not in SAVE_STEPS or (final and step!=600):raise ValueError('Only0/300/600, final600')
    _check_config(payload['config'])
    if payload['identity']!=model.identity() or payload['frozen_old_state_digest']!=model.frozen_old_state_digest:raise ValueError('Architecture/lineage/originalfrozenidentity')
    state=payload['model_state_dict'];canonical=model.state_dict()
    if not isinstance(state,dict) or state.keys()!=canonical.keys():raise ValueError('Exactold/newstatekeys')
    for name,tensor in state.items():
        expected=canonical[name]
        if not torch.is_tensor(tensor) or tensor.shape!=expected.shape or tensor.dtype!=expected.dtype or tensor.device.type!='cpu' or not bool(torch.isfinite(tensor).all()):raise ValueError('Savedmodelshape/dtype/finite: '+name)
    _validate_optimizer_state(payload['optimizer_state_dict'],model,step);_validate_rng(payload['rng']);curve=payload['curve']
    if not isinstance(curve,list) or not curve or not isinstance(curve[-1],dict) or curve[-1].get('step')!=step:raise ValueError('Savedcurve末step mustbindcheckpoint')
    return True


def load_checkpoint(path,digest,final=True):
    _require_neural_only();path=Path(path).resolve()
    if not _digest(digest) or sha(path)!=digest:raise ValueError('ActualcheckpointSHA required')
    import torch
    with _preserve_rng():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if not isinstance(payload,dict) or set(payload)!=FIELDS or payload.get('format_version')!=FORMAT:raise ValueError('Strictfastpathformatbeforeconstructors')
        _check_config(payload['config']);model,_=initialize();_validate_payload(payload,model,final);model.load_state_dict(payload['model_state_dict'],strict=True);model.verify_old();model.validate_architecture();model.eval()
        if sha(path)!=digest:raise ValueError('Checkpointchangedwhileloading')
        _check_config(payload['config'])
    return model,payload


def save_checkpoint(path,model,opt,config,curve,step,rng):
    """Same-directory temporaryfsync+exclusiveatomic hardlink; nooverwrite."""
    _require_neural_only()
    import torch
    if type(step) is not int or step not in SAVE_STEPS:raise ValueError('Only0/300/600save')
    model.verify_old();model.validate_architecture();_validate_live_optimizer(opt,model)
    payload=dict(format_version=FORMAT,step=step,model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer_state_dict=copy.deepcopy(opt.state_dict()),frozen_old_state_digest=model.frozen_old_state_digest,config=copy.deepcopy(config),rng=copy.deepcopy(rng),curve=copy.deepcopy(curve),identity=copy.deepcopy(model.identity()))
    _validate_payload(payload,model,False);path=Path(path).resolve()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:temp=Path(stream.name);torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        model.verify_old();_check_config(config);os.link(temp,path);return sha(path)
    finally:
        if temp is not None and temp.exists():temp.unlink()
