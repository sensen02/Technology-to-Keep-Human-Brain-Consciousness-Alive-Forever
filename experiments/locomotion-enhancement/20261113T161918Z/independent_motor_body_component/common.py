"""Explicit component-only helpers; stdlib import boundary, no ghost sensoryports."""
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
CAL=ROOT.parent/'independent_motor_calibration'
TEACHER=ROOT.parent/'pivot_teacher_design'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
FORMAT='INDEPENDENT_MOTOR_COMPONENT_BODY_DIAGNOSTIC_V1'
CONDITIONS={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
SOURCE_CANDIDATE=6
FILES=('PREREG.json','common.py','prepare.py','preflight.py','runner.py','audit.py','run.py')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jsonable(v):
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
    if isinstance(v,Path):return str(v)
    if type(v).__module__.startswith('numpy'):return v.tolist() if hasattr(v,'tolist') else v.item()
    return v

def save(p,value):
    with Path(p).open('x') as stream:json.dump(jsonable(value),stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def verify(hashes):
    for p,digest in hashes.items():
        if sha(p)!=digest:raise ValueError('Source/input drift: '+p)

def cpu1():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(name+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import torch
    torch.set_num_threads(1)

def load_motor():
    if any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave')):raise ValueError('Strictfinal600NNload beforephysicalbackend')
    b=read(ROOT/'INPUT_BINDINGS.json');verify(b['proof_hashes']);verify(b['source_hashes']);verify(b['data_hashes'])
    if b['kind']!=FORMAT or b['motor_format']!='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1' or b['final_step']!=600 or b['state_dims']!=[128,64,64,96]:raise ValueError('Actual352final600binding required')
    paths=list(sys.path);sentinel=object();prior={k:sys.modules.get(k,sentinel) for k in ('common','model')}
    try:
        ci=module('_component_strict_motor_io',CAL/'common.py');m,payload=ci.load_checkpoint(b['checkpoint'],b['checkpoint_sha256'],final=True)
    finally:
        sys.path[:]=paths
        for k,v in prior.items():
            if v is sentinel:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    if any(sys.modules.get(k,sentinel) is not v for k,v in prior.items()):raise ValueError('Restorecommon/modelaliases')
    m.requires_grad_(False);return m,payload


def frozen():
    b=read(ROOT/'INPUT_BINDINGS.json');verify(b['proof_hashes']);verify(b['source_hashes']);verify(b['data_hashes'])
    paths=[ROOT/n for n in FILES]+[ROOT/'INPUT_BINDINGS.json',TEACHER/'common.py',TEACHER/'teacher.py',BASE/'cycle_reward.py']
    hashes={**b['source_hashes'],**b['data_hashes'],**b['proof_hashes']};hashes.update({str(p):sha(p) for p in paths});return hashes


def physical_stack():
    import numpy as np
    helpers=module('_component_frozen_teacher_helpers',TEACHER/'common.py')
    t=module('_component_frozen_bodylength_teacher',TEACHER/'teacher.py');t.pin_baseline()
    import physics,identity,static_teacher
    ns=dict(np=np,Path=Path,sha256=identity.sha256,MAX_RATE=physics.MAX_RATE,summarise=physics.summarise)
    trace_ast=helpers.selected_ast(BASE/'student_wave.py',('SOURCE','teacher_config','no_external','bounded_action','Trace','settle'),ns)
    peakns=dict(np=np);peak_ast=helpers.selected_ast(helpers.PEAK_SOURCE,('PeakRecorder',),peakns)
    return types.SimpleNamespace(**ns),physics,identity,static_teacher,peakns['PeakRecorder'],dict(legacy_trace_selected_AST_sha256=trace_ast,legacy_peak_selected_AST_sha256=peak_ast,neural_component_Torch_allowed=True)


def snapshot(c):
    return {k:getattr(c.d,k).copy() for k in ('qpos','qvel','ctrl','xfrc_applied','qfrc_applied')}|dict(target=c.actuator.target_rad.copy(),buffer=c.buf.copy())
def assert_same(a,b):
    import numpy as np
    if a.keys()!=b.keys() or any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('Actualstate/target/buffer reset orwrite')
def yaw(c):
    import numpy as np
    r=c.d.xmat[c.th].reshape(3,3);return float(np.arctan2(r[1,0],r[0,0]))
def body_length(c):return module('_component_exact_sevenbody_extent',TEACHER/'teacher.py').body_length_from_compiled_mesh(c)


def score_from_arrays(t,g,boundary,L,W,condition,native):
    import numpy as np
    mask=t['control_source']==SOURCE_CANDIDATE;gm=g['control_source']==SOURCE_CANDIDATE
    if not np.any(mask):return None
    com=t['com'][mask];loads=t['loads'][mask];nf=t['nonfoot'][mask];times=t['time'][mask]
    yy=np.unwrap(np.r_[boundary['yaw'],g['thorax_yaw'][gm]]);dyaw=float(yy[-1]-yy[0]);duration=float(times[-1]-boundary['time'])
    offset=com[:,:2]-np.asarray(boundary['com'])[:2];drift=float(np.linalg.norm(offset,axis=1).max()/L)
    velocity=np.diff(np.vstack([boundary['com'],com]),axis=0)/.001
    body_v=np.stack([r.T@v for r,v in zip(g['thorax_rotation'][gm],velocity)])/L
    steady=times>=boundary['time']+.5-1e-8;rms=float(np.sqrt(np.mean(np.sum(body_v[steady,:2]**2,axis=1)))) if np.any(steady) else None
    helpers=module('_component_measurement_cycles_only',TEACHER/'common.py');cycles=helpers.PivotCycles(W)
    first=int(np.flatnonzero(mask)[0])
    for i in range(max(0,first-10),len(times)+first):cycles.update(float(t['time'][i]),t['loads'][i],t['padz'][i])
    cr=module('_component_old_walking_cyclemetric',BASE/'cycle_reward.py');observer=cr.CycleTracker(cr.CycleConfig(timeout=4.)).reset(start_time=boundary['time']);events=[]
    for ti,load,z,foot in zip(times,loads,t['padz'][mask],t['relativefoot'][mask]):events.extend(observer.update(dict(time=float(ti),loads=load,padz=z,relativefoot=foot))['events'])
    events.extend(observer.finalize()['events']);walkcounts=[sum(e['type']=='complete' and e['leg']==leg for e in events) for leg in cr.LEG_ORDER]
    shared=dict(support=float(loads.sum(1).mean()/W)>=.9,nonfoot_1ms=float(nf.max())==0.,com=float(com[:,2].min())>=1.2,native_nonfoot=native['post_nonfoot_peak_native']==0.,native_extra_force=native['extra_force_max']==0.,native_COM=native['post_COM_min']>=1.2,full_window=native['post_native_steps']==768000 and native['actual_native_steps']==1232000)
    if condition=='walk':
        if len(times)<2:forward=False;slip=False
        else:
            delta=com[-1,:2]-com[0,:2];dt=float(times[-1]-times[0]);net=float(np.linalg.norm(delta)/dt);vt=(loads*t['speed'][mask]).sum(0)/np.maximum(loads.sum(0),1e-12);forward=float(delta[0]/dt)>=.05;slip=float((vt/max(net,1e-9)).max())<=.5
        shared.update(forward=forward,legacy_slip=slip,all6walkingcycles=all(x>=1 for x in walkcounts))
    elif condition=='stop':shared.update(drift=drift<=.1,post_transition_XY_RMS=rms is not None and rms<=.01,yaw_abs=abs(math.degrees(dyaw))<=5.)
    else:shared.update(drift=drift<=.5,signed_yaw=math.degrees(dyaw if condition=='left_pivot' else -dyaw)>=45.,all6pivotcycles=all(x>=1 for x in cycles.counts))
    denominator=loads.sum(0);tangent=(loads*t['speed'][mask]).sum(0)/np.maximum(denominator,1e-12)/L
    return dict(duration_s=duration,yaw_change_rad=dyaw,yaw_change_deg=math.degrees(dyaw),max_COM_xy_drift_bodylength=drift,final_COM_xy_drift_bodylength=float(np.linalg.norm(offset[-1])/L),COM_xy_path_bodylength=float(np.linalg.norm(np.diff(np.vstack([boundary['com'],com]),axis=0)[:,:2],axis=1).sum()/L),support_mean_over_weight=float(loads.sum(1).mean()/W),COM_min_native=float(com[:,2].min()),nonfoot_1ms_peak=float(nf.max()),post_transition_body_XY_velocity_RMS_bodylength_per_s=rms,pivot_cycle_counts=list(cycles.counts),walk_cycle_counts=walkcounts,per_leg_loaded_tangent_speed_bodylength_per_s=[float(v) if d>1e-12 else None for v,d in zip(tangent,denominator)],per_leg_load_integral_native_force_s=(denominator*.001).tolist(),gates=shared,task_pass_candidate=bool(all(shared.values())),gate_version='COMPONENT_'+condition.upper()+'_V1',stop_cycles_applicable=False if condition=='stop' else None)
