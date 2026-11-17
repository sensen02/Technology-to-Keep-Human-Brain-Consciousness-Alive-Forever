"""Strict local-mean storage. STDLIB ONLY on import; parent-call-only numerics.
initialize()->(model,parentpayload), load_checkpoint(path,digest,final=True)->(m,p).
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
PARENT_DIR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration')
PARENT_CHECKPOINT=PARENT_DIR/'TRAIN/checkpoint_000600.pt'
PARENT_SHA='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd'
COLLECTION=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection')
OLD_DIR=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/combined_turning')
OLD_CHECKPOINT=OLD_DIR/'COMBINED.pt'
OLD_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
FORMAT='INDEPENDENT_MOTOR_LOCAL_MEAN_CORRECTIVE_V1'
SAVE_STEPS=(0,300,600)
FIELDS={'format_version','step','model_state_dict','optimizer_state_dict','frozen_old_state_digest','config','rng','curve','identity'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def _digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _read(path):return json.loads(Path(path).read_text())
def verify_hashes(hashes):
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Nonemptyabsolute source/inputhashes required')
    for filename,digest in hashes.items():
        if not isinstance(filename,str) or not Path(filename).is_absolute() or not _digest(digest) or not Path(filename).is_file() or sha(filename)!=digest:raise ValueError('Source/inputidentitydrift: '+str(filename))
    return True


def parent_qualification_hashes():
    """Actual007d600 proof plus eligible MASKED assistedcollection; stdlib only.
    Does not load weights/arrays. Eligibility is not all-raw-label eligibility or
    neural acceptance. Callers must apply assisted label_qualified to their loss.
    """
    if sha(PARENT_CHECKPOINT)!=PARENT_SHA or sha(OLD_CHECKPOINT)!=OLD_SHA:raise ValueError('Actualparent007d600/original789fSHA required')
    tr=_read(PARENT_DIR/'TRAIN/RESULT.json');stage=_read(PARENT_DIR/'STAGE_VERIFICATION.json');jobs=_read(PARENT_DIR/'JOBS.json');evaluation=_read(PARENT_DIR/'EVALUATION.json');ev=_read(PARENT_DIR/'EVALUATION_VERIFY.json')
    if tr['updates']!=600 or tr['checkpoint']!=str(PARENT_CHECKPOINT) or tr['sha256']!=PARENT_SHA or not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint_sha256']!=PARENT_SHA:raise ValueError('Actualparent007d fixed600 qualification required')
    if len(jobs)!=3 or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in jobs):raise ValueError('Actual3parentstagesexit0 required')
    if not ev['integrity_all_pass'] or not all(ev['checks'].values()) or ev['evaluation_sha256']!=sha(PARENT_DIR/'EVALUATION.json') or evaluation['checkpoint_sha256']!=PARENT_SHA or evaluation['step']!=600:raise ValueError('Actualparentindependentfinalmetrics required')
    paths=[PARENT_DIR/'model.py',PARENT_DIR/'common.py',PARENT_DIR/'TRAIN/RESULT.json',PARENT_DIR/'STAGE_VERIFICATION.json',PARENT_DIR/'JOBS.json',PARENT_DIR/'EVALUATION.json',PARENT_DIR/'EVALUATION_VERIFY.json',PARENT_CHECKPOINT,OLD_CHECKPOINT]
    inherited={}
    def merge(hashes):
        if not isinstance(hashes,dict):raise ValueError('Sourceproof mapping required')
        for path,digest in hashes.items():
            if path in inherited and inherited[path]!=digest:raise ValueError('Conflicting archived identity: '+path)
            inherited[path]=digest
    merge(tr['source_hashes']);merge(tr['data_hashes'])
    for step in SAVE_STEPS:
        path=PARENT_DIR/'TRAIN'/f'VERIFY_{step:06d}.json';proof=_read(path)
        if not proof['integrity_all_pass'] or not all(proof['checks'].values()) or proof['step']!=step or sha(proof['checkpoint'])!=proof['checkpoint_sha256']:raise ValueError('Actualparent0/300/600 savedverification required')
        devpath=PARENT_DIR/'TRAIN'/f'DEV_VERIFY_{step:06d}.json';dev=_read(devpath)
        if not dev['integrity_all_pass'] or not all(dev['checks'].values()) or dev['step']!=step or dev['checkpoint_sha256']!=proof['checkpoint_sha256']:raise ValueError('Actualparent4unlabeledDEV savedverification required')
        paths.extend((path,devpath,Path(proof['checkpoint'])))
    if sha(evaluation['arrays']['path'])!=evaluation['arrays']['sha256']:raise ValueError('Actualparentfinalmetricsartifact drift')
    paths.append(Path(evaluation['arrays']['path']))
    qpath=COLLECTION/'PARENT_DATASET_QUALIFICATION.json';qualification=_read(qpath);summary=_read(COLLECTION/'SUMMARY.json');collection_jobs=_read(COLLECTION/'JOBS.json')
    if qualification['eligible_for_masked_supervised_learning'] is not True or qualification['all_raw_labels_eligible'] is not False or qualification['neural_acceptance'] is not False or qualification['new_brain_neck_deployed'] is not False:raise ValueError('Only maskedcorrective eligibility, never rawlabel orneural acceptance')
    if qualification['all12stages_exit0'] is not True or qualification['all_sources_reverified'] is not True or qualification['old_control_original5_cycle_PASS'] is not True or qualification['actual_new_native_steps']!=4928000 or qualification['student_checkpoint_sha256']!=PARENT_SHA:raise ValueError('Actualqualifiedmixedcollection source required')
    if qualification['student_mix']!=.25 or qualification['teacher_mix']!=.75 or qualification['training_updates']!=0:raise ValueError('Fixedcollection mixture/history identity required')
    if len(collection_jobs)!=12 or any(j.get('exit_code')!=0 or j.get('status')!='COMPLETED' for j in collection_jobs) or summary['stages']!=12 or summary['actual_new_native_steps']!=4928000:raise ValueError('Actual12collectionstagesexit0 required')
    paths.extend((qpath,COLLECTION/'SUMMARY.json',COLLECTION/'JOBS.json',COLLECTION/'INPUT_BINDINGS.json'))
    expected={'old_walk_control':False,'assisted_stop':True,'assisted_left_pivot':True,'assisted_right_pivot':True}
    rows=qualification['rows']
    if len(rows)!=4 or {r['case'] for r in rows}!=set(expected):raise ValueError('Exactly4qualifiedsourcecases required')
    nn_count=teacher_count=valid_count=0
    for row in rows:
        case=row['case'];folder=COLLECTION/'trials'/case;resultpath=folder/'RESULT.json'
        if row['result']!=str(resultpath) or sha(resultpath)!=row['result_sha256']:raise ValueError('Actualcollection RESULT binding')
        result=_read(resultpath)
        if row['teacher_assisted'] is not expected[case] or result['teacher_assisted'] is not expected[case] or result['neural_acceptance'] is not False:raise ValueError('Collectionassistance is not neuralacceptance')
        if row['qualified_teacher_labels']!=result['qualified_teacher_labels']:raise ValueError('Actualqualifiedlabelcount binding')
        if expected[case]:
            if row['supervision_kind']!='unmixed_full_teacher_action':raise ValueError('Correctiveteacherlabels must not be mixedappliedactions')
            valid_count+=int(row['qualified_teacher_labels'])
        elif row['supervision_kind']!='direct_frozen_oldmean_tanh':raise ValueError('Positiveoldwalking source kind')
        proof_names=set()
        for proof_row in row['proofs']:
            proofpath=Path(proof_row['path'])
            if not proofpath.is_absolute() or proofpath.parent!=folder or sha(proofpath)!=proof_row['sha256']:raise ValueError('Actualcollection auditSHA/path binding')
            proof=_read(proofpath)
            if not proof['integrity_all_pass'] or not all(proof['checks'].values()) or proof['result_sha256']!=row['result_sha256'] or int(proof_row['checks'])!=len(proof['checks']):raise ValueError('Independentcollection numericalproof required')
            if proof.get('neural_acceptance') is not False:raise ValueError('Assistedproof must not implyneuralacceptance')
            if proofpath.name=='NN_VERIFY.json':nn_count+=1
            elif proofpath.name=='TEACHER_VERIFY.json':
                teacher_count+=1
                if not expected[case] or proof['physical_integration_steps']!=0 or proof['NN_forward_calls']!=0:raise ValueError('Teacherlabelaudits must be offline zeroNN/zerointegration')
            else:raise ValueError('Unexpectedcollection audit artifact')
            proof_names.add(proofpath.name);paths.append(proofpath);merge(proof['source_hashes'])
        if proof_names!=({'NN_VERIFY.json','TEACHER_VERIFY.json'} if expected[case] else {'NN_VERIFY.json'}):raise ValueError('FourNNandthreeTeacher sourceproof coverage required')
        merge(result['source_hashes']);paths.append(resultpath)
        for key in ('trace','neural_decisions','geometry'):
            item=result[key]
            if sha(item['path'])!=item['sha256']:raise ValueError('Actualcollection artifactSHA')
            paths.append(Path(item['path']))
        if 'teacher_diagnostics' in result:
            item=result['teacher_diagnostics']
            if sha(item['path'])!=item['sha256']:raise ValueError('ActualteachermetadataartifactSHA')
            paths.append(Path(item['path']))
    if nn_count!=4 or teacher_count!=3 or valid_count!=qualification['assisted_teacher_valid_labels']:raise ValueError('Actual4NN/3teacher/qualifiedcount requirement')
    verify_hashes(inherited)
    hashes={str(path):sha(path) for path in paths}
    # Return proof identities; the immutable ancestral input graphs are verified
    # above, while caller DATASETmanifest independently binds actual sample files.
    return hashes


def _check_config(config):
    if not isinstance(config,dict):raise ValueError('Configdict required')
    for key in ('source_hashes','data_hashes'):verify_hashes(config.get(key))
    required=(ROOT/'model.py',ROOT/'common.py',PARENT_DIR/'model.py',PARENT_DIR/'common.py',OLD_DIR/'model.py',OLD_DIR/'model_io.py')
    if any(config['source_hashes'].get(str(path))!=sha(path) for path in required):raise ValueError('Localmean/parent/originalstrictsourcehashes required')
    if config['data_hashes'].get(str(PARENT_CHECKPOINT))!=PARENT_SHA or config['data_hashes'].get(str(OLD_CHECKPOINT))!=OLD_SHA:raise ValueError('Actualparent007d600/original789f data binding')
    proof=parent_qualification_hashes()
    if any(config['data_hashes'].get(path)!=digest for path,digest in proof.items()):raise ValueError('Actualparentstorage+maskedcollection proof hashes required indata_hashes')
    return True


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def _backend():
    name='_independent_local_mean_corrective_numeric_model';cached=sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve()!=ROOT/'model.py':raise ValueError('Localmeanbackendnamespacecollision')
        return cached
    return _module(name,ROOT/'model.py')

@contextmanager
def _preserve_rng():
    import random,numpy as np,torch
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    try:yield
    finally:random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])

def _require_neural_only():
    if any(name in sys.modules for name in ('physics','mujoco','student_wave','flygym')):raise ValueError('StrictNNload beforephysicalbackend required')

def _load_parent():
    _require_neural_only();parent_qualification_hashes();sentinel=object();names=('model','common','sources','_localmean_parent_strict_io');prior={k:sys.modules.get(k,sentinel) for k in names};paths=list(sys.path)
    try:
        loader=_module('_localmean_parent_strict_io',PARENT_DIR/'common.py');parent,payload=loader.load_checkpoint(PARENT_CHECKPOINT,PARENT_SHA,final=True)
    finally:
        sys.path[:]=paths
        for name,value in prior.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    if sys.path!=paths or any(sys.modules.get(k,sentinel) is not value for k,value in prior.items()):raise ValueError('Parentloaderaliases/pathrestore failed')
    return parent,payload

def initialize():
    """Strict007d parent -> deterministic migration; no forward; preserve callerRNG."""
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
    if set(slots)!=set(order):raise ValueError('All8Adam moments required')
    for pid,p in zip(order,parameters):
        slot=slots[pid]
        if set(slot) not in ({'step','exp_avg','exp_avg_sq'},{'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}):raise ValueError('ExactAdamslots')
        counter=slot['step']
        if torch.is_tensor(counter):
            if counter.numel()!=1 or counter.device.type!='cpu' or not bool(torch.isfinite(counter).all()):raise ValueError('FiniteCPUstepcounter')
            counter=float(counter)
        if float(counter)!=step:raise ValueError('8Adamcounters must equal savedupdate')
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
    if not isinstance(payload,dict) or set(payload)!=FIELDS or payload['format_version']!=FORMAT:raise ValueError('Strictlocalmeanpayloadfields/FORMAT')
    step=payload['step']
    if type(step) is not int or step not in SAVE_STEPS or (final and step!=600):raise ValueError('Only0/300/600, final600')
    _check_config(payload['config'])
    if payload['identity']!=model.identity() or payload['frozen_old_state_digest']!=model.frozen_old_state_digest:raise ValueError('Architecture/lineage/frozenoldidentity')
    state=payload['model_state_dict'];canonical=model.state_dict()
    if not isinstance(state,dict) or state.keys()!=canonical.keys():raise ValueError('Exactold/newstatekeys')
    for name,tensor in state.items():
        expected=canonical[name]
        if not torch.is_tensor(tensor) or tensor.shape!=expected.shape or tensor.dtype!=expected.dtype or tensor.device.type!='cpu' or not bool(torch.isfinite(tensor).all()):raise ValueError('Savedmodelshape/dtype/finite: '+name)
    _validate_optimizer_state(payload['optimizer_state_dict'],model,step);_validate_rng(payload['rng']);curve=payload['curve']
    if not isinstance(curve,list) or not curve or not isinstance(curve[-1],dict) or curve[-1].get('step')!=step:raise ValueError('Savedcurve finalstep must bindcheckpoint')
    return True


def load_checkpoint(path,digest,final=True):
    _require_neural_only();path=Path(path).resolve()
    if not _digest(digest) or sha(path)!=digest:raise ValueError('ActualcheckpointSHA required')
    import torch
    with _preserve_rng():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if not isinstance(payload,dict) or set(payload)!=FIELDS or payload.get('format_version')!=FORMAT:raise ValueError('Strictlocalmeanformat before constructors')
        _check_config(payload['config']);model,_=initialize();_validate_payload(payload,model,final);model.load_state_dict(payload['model_state_dict'],strict=True);model.verify_old();model.validate_architecture();model.eval()
        if sha(path)!=digest:raise ValueError('Checkpoint changed while loading')
        _check_config(payload['config'])
    return model,payload


def save_checkpoint(path,model,opt,config,curve,step,rng):
    """Same-directory tempfsync+exclusiveatomic hardlink; never overwrite."""
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
