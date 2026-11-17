"""Stdlib-only helpers. Parent runtime numerical paths are explicit functions."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
import types

ROOT=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z')
PREVIOUS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z')
CAL=PREVIOUS/'fastpath_motor_calibration'
TEACHER=OLD/'pivot_teacher_design'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
QUERY=PREVIOUS/'teacher_student_state_query'
RECOVERY=PREVIOUS/'teacher_query_report_recovery'
FORMAT='TEACHER_ASSISTED_COLLECTION_FIXED4_V1'
LABEL='TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE'
SCOPE='BodyANNcomponent_NOTnewbrainneckdeployment'
ACTUAL_SHA='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd'
MOTOR_FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1'
CASES={'old_walk_control':('walk',[1.,0.]),'assisted_stop':('stop',[0.,0.]),'assisted_left_pivot':('left_pivot',[0.,1.]),'assisted_right_pivot':('right_pivot',[0.,-1.])}
CONDITIONS={case:value[1] for case,value in CASES.items()}
MASK_NAMES=('actual_COM','actual_NF_1ms','IK','unbacked_plannedload','teacher_supported','qdot_clip','rate_clip','target_bounds','tau_ff_clip','finite42','actual_restore')
FILES=('PREREG.json','common.py','prepare.py','preflight.py','collector.py','audit_nn.py','audit_teacher.py','run.py','static_check.py','DESIGN.zh-CN.md')
SOURCE_CANDIDATE=7

# Single-layer compile of actual original helpers, no old namespace retained.
_HELPER=OLD/'independent_motor_body_component/common.py'
with _HELPER.open('rb') as _stream:
    if hashlib.file_digest(_stream,'sha256').hexdigest()!='788577b88d44c79f206e3ef2f003c677537809ec35511f9043048b2598e7d1d1':raise ValueError('Original helper source drift')
_SELECTED=('module','sha','read','jsonable','save','verify','cpu1','physical_stack','snapshot','assert_same','yaw','body_length','score_from_arrays')
_tree=ast.parse(_HELPER.read_text())
_nodes=[n for n in _tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in _SELECTED]
if {n.name for n in _nodes}!=set(_SELECTED):raise ValueError('Missing original helper')
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(_HELPER)+'::selected_original_functions_newglobals','exec'),globals())


def load_motor():
    if any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave')):raise ValueError('Strict007dNNload before physical backend')
    binding=read(ROOT/'INPUT_BINDINGS.json')
    for key in ('proof_hashes','source_hashes','data_hashes'):verify(binding[key])
    if binding['kind']!=FORMAT or binding['checkpoint_sha256']!=ACTUAL_SHA or binding['motor_format']!=MOTOR_FORMAT or binding['final_step']!=600 or binding['state_dims']!=[128,64,64,96]:raise ValueError('Strict actual007d final600 binding')
    sentinel=object();names=('common','model','sources');prior={name:sys.modules.get(name,sentinel) for name in names};paths=list(sys.path)
    try:
        io=module('_assisted_actual007d_strictio',CAL/'common.py');motor,payload=io.load_checkpoint(binding['checkpoint'],ACTUAL_SHA,final=True)
    finally:
        sys.path[:]=paths
        for name,value in prior.items():
            if value is sentinel:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    if any(sys.modules.get(name,sentinel) is not value for name,value in prior.items()):raise ValueError('Strict loader aliases not restored')
    motor.requires_grad_(False)
    return motor,payload


def shadow_details(m,x,u,state):
    """One direct frozen-old step, exact fastpath formula; no mean-res old readback."""
    import torch
    m._validate_inputs(x,u)
    if len(state)!=4 or any(s.shape!=(len(x),n) for s,n in zip(state,(128,64,64,96))):raise ValueError('Explicit352 state')
    oldmean,features,oldafter=m.cached_features(x,u,state[:3])
    backend=sys.modules[type(m).__module__]
    encoded=torch.tanh(m.encoder(features));candidate=m.gru(encoded,state[3]);h=(1.-backend.ALPHA)*state[3]+backend.ALPHA*candidate
    residual=backend.BOUND*torch.tanh(m.head(torch.cat((h,encoded),dim=-1)))
    return oldmean+residual,(*oldafter,h),residual,oldmean


def frozen():
    binding=read(ROOT/'INPUT_BINDINGS.json')
    for key in ('proof_hashes','source_hashes','data_hashes'):verify(binding[key])
    hashes={**binding['source_hashes'],**binding['data_hashes'],**binding['proof_hashes']}
    hashes[str(ROOT/'INPUT_BINDINGS.json')]=sha(ROOT/'INPUT_BINDINGS.json')
    for name in FILES:
        path=str(ROOT/name)
        if hashes.get(path)!=sha(path):raise ValueError('Frozen collection source drift: '+path)
    return hashes


def make_teacher(c,condition):
    order=read(OLD/'pivot_fixed_shadow_v2/PREFLIGHT.json')['order_indices']
    if sorted(order)!=list(range(6)):raise ValueError('Qualified original teacher leg order required')
    path=OLD/('stop_shadow_control/teacher.py' if condition=='stop' else 'pivot_fixed_shadow_v2/teacher.py')
    factory=module('_assisted_fixed_teacher_'+condition,path)
    return factory.make_teacher(c,condition,order_indices=order)


def restore_api():return module('_assisted_frozen_offline_restore',QUERY/'restore.py')


def state_hashes(c):return {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in snapshot(c).items()}


def teacher_mask(c,action,diag,source_exact=True,actual_com=None,actual_nf=None):
    import numpy as np
    measured=c.measure() if actual_nf is None else None
    z=float(c.d.subtree_com[c.th,2]) if actual_com is None else float(np.asarray(actual_com)[2])
    nf=float(measured['nonfoot']) if actual_nf is None else float(actual_nf)
    unbacked=sum(float(diag['per_leg_fz_over_W'][i]) for i in diag['unbacked_support_indices'])
    mask=dict(actual_COM=z>=1.2,actual_NF_1ms=nf==0.,IK=float(max(diag['ik_error_norm_per_leg']))<=.01,unbacked_plannedload=unbacked<=.01,teacher_supported=not diag['unsupported'],qdot_clip=diag['qdot_des_clip_count']==0,rate_clip=diag['rate_clip_count']==0,target_bounds=len(diag['motor_target_bound_indices'])==0,tau_ff_clip=len(diag['tau_ff_native_clip_indices'])==0,finite42=action.shape==(42,) and bool(np.isfinite(action).all()),actual_restore=bool(source_exact))
    if tuple(mask)!=MASK_NAMES:raise ValueError('Mask schema order')
    return mask,unbacked


def score(t,g,boundary,L,W,condition,native,source):
    namespace=dict(score_from_arrays.__globals__);namespace['SOURCE_CANDIDATE']=source
    function=types.FunctionType(score_from_arrays.__code__,namespace,score_from_arrays.__name__,score_from_arrays.__defaults__,score_from_arrays.__closure__)
    return function(t,g,boundary,L,W,condition,native)
