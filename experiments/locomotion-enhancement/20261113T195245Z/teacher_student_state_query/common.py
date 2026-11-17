"""Offline statequery algorithm only, no neural model or physical integration."""
from pathlib import Path
import importlib.util
import json
import math
import os
import resource
import sys
import time
from metadata import ROOT,OLD,BASE,read,save,sha,verify,factory_path


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def cpu1():
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError(k+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})

def resources(start):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800 or time.monotonic()-start>300:raise RuntimeError('Queryresourcecap exceeded, noextension')
    return rss

def stack():
    teacher=module('_statequery_frozen_teacherbase',OLD/'pivot_teacher_design/teacher.py');teacher.pin_baseline()
    import physics,identity,static_teacher,mujoco as mj
    # Do not import student_wave/models/torch calibration modules.
    return physics,identity,mj

def load_arrays(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def snapshot(c):
    return {name:getattr(c.d,name).copy() for name in ('qpos','qvel','ctrl','xfrc_applied','qfrc_applied')}|dict(target=c.actuator.target_rad.copy(),buffer=c.buf.copy())

def snapshot_equal(a,b):
    import numpy as np
    return a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)

def make_teacher(c,condition):
    order=read(OLD/'pivot_fixed_shadow_v2/PREFLIGHT.json')['order_indices']
    if sorted(order)!=list(range(6)):raise ValueError('Sourcequalified fixed6legorder')
    factory=module('_statequery_fixed_'+condition,factory_path(condition));return factory.make_teacher(c,condition,order_indices=order)

def legacy_diagnostics(path):
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['candidate_phase']:yield row

def jacobian_probe(c,teacher):
    import numpy as np
    original=teacher.q_des.copy();errors=[]
    for li in range(6):
        teacher._fk();position,jac=teacher._leg_kinematics(c,li)
        qi=teacher.qa[teacher.leg_act[li][0]];epsilon=1e-7
        teacher.q_des[qi]+=epsilon;teacher._fk();plus=teacher._leg_kinematics(c,li,False)
        teacher.q_des[qi]-=2*epsilon;teacher._fk();minus=teacher._leg_kinematics(c,li,False)
        teacher.q_des[:]=original;errors.append(float(np.max(np.abs((plus[:2]-minus[:2])/(2*epsilon)-jac[:2,0]))))
    teacher._fk();return errors

class QueryInvariantError(RuntimeError):
    """Fatal source/restore/steptrap/statewrite failure; never a missinglabel."""


class TeacherLocalQueryError(RuntimeError):
    """Only an explicit local numerical/allocation infeasibility may be retained."""


def query_one(c,teacher,trace,initial,ms,elapsed,restore_api,mj):
    import numpy as np
    try:
        restored=restore_api.restore(c,trace,initial,ms,mujoco_module=mj)
    except Exception as exc:
        raise QueryInvariantError('Fatal offline restore/forward/guard failure: '+repr(exc)) from exc
    before=snapshot(c);source_index=ms-1
    record=dict(time_s=float(trace['time'][source_index]),physical_ms=ms,teacher_elapsed_s=elapsed,actual_COM=trace['com'][source_index].copy(),actual_NF_1ms=float(trace['nonfoot'][source_index]),actual_yaw=float(np.arctan2(c.d.xmat[c.th].reshape(3,3)[1,0],c.d.xmat[c.th].reshape(3,3)[0,0])),actual_target=c.actuator.target_rad.copy(),actual_ctrl=c.d.ctrl.copy(),restore=restored)
    try:
        action,diag=teacher.action(c,elapsed)
    except np.linalg.LinAlgError as exc:
        if not snapshot_equal(before,snapshot(c)):raise QueryInvariantError('Teacherstatewrite during failed solve') from exc
        raise TeacherLocalQueryError('Local teacher linear solve infeasible: '+repr(exc)) from exc
    except RuntimeError as exc:
        if 'allocation failed' in str(exc).lower():
            if not snapshot_equal(before,snapshot(c)):raise QueryInvariantError('Teacherstatewrite during failed allocation') from exc
            raise TeacherLocalQueryError('Local teacher load allocation infeasible: '+repr(exc)) from exc
        raise QueryInvariantError('Fatal teacher runtime/forbidden-call failure: '+repr(exc)) from exc
    except Exception as exc:
        raise QueryInvariantError('Fatal teacher API/source failure: '+repr(exc)) from exc
    if not snapshot_equal(before,snapshot(c)):raise QueryInvariantError('Teacherchanged restoredactualstate/target/buffer')
    source_ok=bool(restored.get('exact_state',False))
    if not source_ok:raise QueryInvariantError('Actualrestorefailed, no labelqueryadmission')
    unbacked=sum(float(diag['per_leg_fz_over_W'][i]) for i in diag['unbacked_support_indices'])
    mask=dict(actual_COM=float(trace['com'][source_index,2])>=1.2,actual_NF_1ms=float(trace['nonfoot'][source_index])==0.,IK=float(max(diag['ik_error_norm_per_leg']))<=.01,unbacked_plannedload=unbacked<=.01,teacher_supported=not diag['unsupported'],qdot_clip=diag['qdot_des_clip_count']==0,rate_clip=diag['rate_clip_count']==0,target_bounds=len(diag['motor_target_bound_indices'])==0,tau_ff_clip=len(diag['tau_ff_native_clip_indices'])==0,finite42=action.shape==(42,) and bool(np.isfinite(action).all()),actual_restore=source_ok)
    record.update(action=action.copy(),diag=diag,label_mask=mask,label_qualified=bool(all(mask.values())),unbacked_planned_load_over_W=unbacked)
    return record

def array_record(record,NNaction):
    import numpy as np
    d=record['diag'];mask42=lambda indices:np.isin(np.arange(42),np.asarray(indices,dtype=int))
    return dict(time=record['time_s'],physical_ms=record['physical_ms'],teacher_elapsed_s=record['teacher_elapsed_s'],actual_target=record['actual_target'],actual_ctrl=record['actual_ctrl'],query_action=record['action'],actual_action=np.asarray(NNaction).copy(),action_difference=record['action']-NNaction,desired_motor_target=np.asarray(d['desired_motor_target']),q_des_act=np.asarray(d['q_des_act']),IK_error=np.asarray(d['ik_error_norm_per_leg']),qdot_raw=np.asarray(d['qdot_des_unclipped']),rate=np.asarray(d['rate_rad_s']),tau_ff=np.asarray(d['tau_ff']),raw_PD_torque_at_desired_target=np.asarray(d['actual_state_raw_torque_at_desired_target']),raw_PD_torque_at_next_target=np.asarray(d['actual_state_raw_torque_at_next_target']),raw_PD_desired_exceed_mask=mask42(d['actual_state_raw_torque_native_exceed_indices']),raw_PD_next_exceed_mask=mask42(d['next_target_torque_native_exceed_indices']),qdot_clip_count=d['qdot_des_clip_count'],rate_clip_count=d['rate_clip_count'],target_bound_mask=mask42(d['motor_target_bound_indices']),tau_clip_mask=mask42(d['tau_ff_native_clip_indices']),planned_load_over_W=np.asarray(d['per_leg_fz_over_W']),actual_contacts=np.asarray(d['actual_contact_counts']),unsupported=d['unsupported'],unbacked_planned_load_over_W=record['unbacked_planned_load_over_W'],actual_COM=record['actual_COM'],actual_NF_1ms=record['actual_NF_1ms'],actual_yaw=record['actual_yaw'],label_qualified=record['label_qualified'],label_mask=np.asarray(list(record['label_mask'].values()),dtype=bool))

def summarized(rows):
    import numpy as np
    if not rows:return dict(rows=0,label_qualified_count=0,label_qualified_fraction=None)
    good=np.asarray([r['label_qualified'] for r in rows],dtype=bool);diff=np.asarray([r['action_difference'] for r in rows]);finite=np.isfinite(diff).all(1)
    desired=np.asarray([r['desired_motor_target'] for r in rows]);actual=np.asarray([r['actual_target'] for r in rows]);target_delta=desired-actual
    def corr(a,b):
        if a.size<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or float(a.std())==0. or float(b.std())==0.:return None
        return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    relation=dict(teacher_desiredtarget_minus_actual_integrator_RMS=float(np.sqrt(np.mean(target_delta[finite]**2))) if finite.any() else None,querycorrection_vs_teacher_targeterror_Pearson_all=corr(diff[finite],target_delta[finite]),querycorrection_vs_teacher_targeterror_Pearson_qualified=corr(diff[good],target_delta[good]),interpretation='Descriptiveconditional relation; teacher law explicitly uses targeterror, correlation isnot causalidentification or proof of actorunobservability')
    return dict(rows=len(rows),label_qualified_count=int(good.sum()),label_qualified_fraction=float(good.mean()),all_finite_action_difference_RMSE=float(np.sqrt(np.mean(diff[finite]**2))) if finite.any() else None,qualified_action_difference_RMSE=float(np.sqrt(np.mean(diff[good]**2))) if good.any() else None,all_action_difference_absmax=float(np.max(np.abs(diff[finite]))) if finite.any() else None,teacher_actual_register_relation=relation,label_mask_names=['actual_COM','actual_NF_1ms','IK','unbacked_plannedload','teacher_supported','qdot_clip','rate_clip','target_bounds','tau_ff_clip','finite42','actual_restore'],label_mask_pass_counts=np.asarray([r['label_mask'] for r in rows]).sum(0).tolist(),scope='Labelquerymask only, no demonstratedphysicalrecovery')
