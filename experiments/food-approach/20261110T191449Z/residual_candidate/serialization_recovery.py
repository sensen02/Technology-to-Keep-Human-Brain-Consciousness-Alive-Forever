"""Recover completed left report without integration; scalar-safe wrapper for right.

Frozen training/evaluator source and checkpoint hashes are unchanged.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from residual_motor import HERE,BASE,sha256
sys.path.insert(0,str(BASE))
import student_wave as sw
import identity
from evaluate_candidate import evaluate,load_checked


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    return value


def recover_left():
    run=HERE/'EVAL_RESIDUAL500_left'
    if (run/'RESULT.json').exists():raise FileExistsError('do not overwrite completed report')
    ident=json.loads((run/'IDENTITY.json').read_text());identity.verify(ident)
    with np.load(run/'trace_1ms.npz',allow_pickle=False) as z,np.load(HERE/'EVAL_RESIDUAL500_neutral/trace_1ms.npz',allow_pickle=False) as ref:
        z={k:z[k] for k in z.files};ref={k:ref[k] for k in ref.files}
        if len(z['time'])!=15400 or not all(np.array_equal(z[k][:5800],ref[k][:5800]) for k in z):raise ValueError('left incomplete or prefix differs')
        select=z['control_source']==sw.SOURCE['student'];idx=np.flatnonzero(select)
        if len(idx)!=9600:raise ValueError('actual ANN source count')
        rec=[(float(z['time'][j]),z['com'][j],z['loads'][j],float(z['nonfoot'][j]),z['speed'][j]) for j in idx]
        auto=sw.summarise(rec,len(rec)*80,initial_heading=np.array([1.,0.]))
        auto['window']=dict(first_sample_s=rec[0][0],last_sample_s=rec[-1][0],sample_count=len(rec))
        observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(z['time'][idx[0]-1]));events=[]
        for j in idx:events.extend(observer.update(dict(time=float(z['time'][j]),loads=z['loads'][j],padz=z['padz'][j],relativefoot=z['relativefoot'][j],com=z['com'][j]))['events'])
        cycles=sw.cycle_report(observer,events)
        extra=bool(np.any(z['external_body_force_max']) or np.any(z['external_generalized_force_max']))
    with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:
        yaw=np.unwrap(z['yaw_after_each_control_ms']);delta=float(yaw[-1]-yaw[4799])
        if len(z['time'])!=2880 or not np.array_equal(z['hidden_before'][1:],z['hidden_after'][:-1]):raise ValueError('actual hidden chain differs')
        handover_time=float(z['time'][960])
    sign_ok=bool(delta>=.01)
    result=dict(candidate_engineering_pass=bool(all(auto['gates'].values()) and cycles['all_legs_completed'] and sign_ok),yaw_sign_and_magnitude_pass=sign_ok,neutral_parity=None,status='RECOVERED_COMPLETED_RUN_AFTER_JSON_NUMPY_BOOL_FAILURE',autonomous=auto,autonomous_cycles=cycles,transition=dict(time_s=handover_time,no_reset=True,exact={'completed_original_evaluator_continuity_check_before_save':True},both_hidden_states_included=True,note='original evaluator checked all snapshots before completion; original snapshot hashes lost on JSON serialization exception; actual saved hidden chain and matched physical prefix rechecked'),yaw_change_rad=delta,command=[0.,1.],trace=dict(path=str(run/'trace_1ms.npz'),sha256=sha256(run/'trace_1ms.npz'),samples=15400),identity_sha256=ident['identity_sha256'],checkpoint_sha256=sha256(HERE/'TRAIN_RESIDUAL_500/checkpoint_000500.pt'),physical_steps=1232000,teacher_calls_after_handover=0,extra_forces=extra,odor_or_food_success=False,state_layout='base128,residual64',source_frame='yaw world positiveCCW originalworldX gates unchanged',recovery=dict(source=str(HERE/'serialization_recovery.py'),source_sha256=sha256(HERE/'serialization_recovery.py'),physical_steps=0,method='same frozen summary and cycle tracker replay from saved actual arrays; initial heading verified identical prefix to baseline',failure_log=str(HERE/'EVAL_left.log')))
    identity.atomic_json(run/'RESULT.json',native(result));print(json.dumps(dict(condition='left',pass_all=result['candidate_engineering_pass'],slip=auto['slip_ratio_max'],yaw=delta)),flush=True)


def run_right():
    original=identity.atomic_json
    def safe(path,value):return original(path,native(value))
    identity.atomic_json=safe
    original_capture=identity.capture
    def capture(core,config,extra_paths=()):
        config=dict(config,serialization_wrapper=str(HERE/'serialization_recovery.py'),serialization_only_numpy_scalar_conversion=True)
        return original_capture(core,config,extra_paths=list(extra_paths)+[HERE/'serialization_recovery.py'])
    identity.capture=capture
    ck=HERE/'TRAIN_RESIDUAL_500/checkpoint_000500.pt'
    evaluate(SimpleNamespace(checkpoint=str(ck),expected_sha256=sha256(ck),out=str(HERE/'EVAL_RESIDUAL500_right'),condition='right'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['recover_left','run_right']);a=p.parse_args()
    if a.mode=='recover_left':recover_left()
    else:run_right()
