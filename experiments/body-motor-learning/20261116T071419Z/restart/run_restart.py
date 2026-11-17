"""Replay the verified brake prefix and test an explicit continued physical restart."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parents[1]/'20261116T063132Z'/'teacher'
sys.path.insert(0,str(ORIGINAL))
import physical
from physical import RF,Local,FloorRecord,build_scene,bench,read,stan,W
from gap_weight import Weight

PREFIX=ORIGINAL/'BRAKE_AS_FIXED_WEIGHT'
SAFE=ORIGINAL/'SAFE_STOP_WEIGHT_EVIDENCE.json'
AS=ORIGINAL/'AS.json'
PLAN=ORIGINAL/'PLAN_BRAKE_RF_ONLY.json'
RELEASE=6450
WALK_MS=9600
TERMINAL=1200
END=RELEASE+WALK_MS+TERMINAL
SOURCE=physical.ARCHIVE

TRACE_KEYS=('time','qpos','qvel','com','loads','nonfoot','padz','speed','foot','upright',
            'target','ctrl','applied','correction','rf_pad_xyz')
DECISION_KEYS=('ms','x','intent','local','base','action')

def archived():
    certified=read(SAFE)
    result=read(PREFIX/'RESULT.json')
    audit=read(PREFIX/'NATIVE_AUDIT.json')
    assert certified['full']['safe_stance_pass']
    assert certified['full']['audited_native_steps']==RELEASE*80
    assert result['actual_ms']==RELEASE
    files={'TRACE.npz':result['trace_sha256'],'DECISIONS.npz':result['decision_sha256'],
           'NATIVE.npz':audit['native_sha256'],'NATIVE_AUDIT.json':result['native_audit_sha256']}
    for name,digest in files.items():
        if stan.sha(PREFIX/name)!=digest:
            raise ValueError(f'Original safe-stop evidence changed: {name}')
    return files


def check_prefix(trace,decisions,rec,minimum):
    result={}
    for key,value in zip(TRACE_KEYS,trace,strict=True):
        assert np.isfinite(value).all(),key
    with np.load(PREFIX/'TRACE.npz',allow_pickle=False) as prefix:
        for key,value in zip(TRACE_KEYS,trace,strict=True):
            np.testing.assert_array_equal(value[:minimum+1],prefix[key][:minimum+1],
                                          err_msg='Stored same-episode safe-stop '+key)
        result['all_trace_fields_bitwise']=list(TRACE_KEYS)
    with np.load(PREFIX/'DECISIONS.npz',allow_pickle=False) as prefix:
        stop=minimum//5
        for key,value in zip(DECISION_KEYS,decisions,strict=True):
            np.testing.assert_array_equal(value[:stop],prefix[key][:stop],
                                          err_msg='Stored same-episode safe-stop '+key)
        result['all_decision_fields_bitwise']=list(DECISION_KEYS)
    if rec is not None:
        with np.load(PREFIX/'NATIVE.npz',allow_pickle=False) as n:
            for key,value in [('forces',rec.forces),('pose',rec.poses),('rf_floor',rec.rf_contact),
                              ('rf_pad_bottom_z',rec.fk_z)]:
                np.testing.assert_array_equal(value[:minimum*80],n[key][:minimum*80],
                                              err_msg='Stored safe-stop native '+key)
            result['all_native_fields_bitwise']=['forces','pose','rf_floor','rf_pad_bottom_z']
    result['prefix_initial_sample_and_ms']=minimum
    result['prefix_native_steps']=minimum*80 if rec is not None else None
    return result


def main(output,mode,probe_ms,native):
    torch.set_num_threads(1)
    evidence=archived()
    known_elapsed=time.monotonic()
    total=min(END,probe_ms) if probe_ms is not None else END
    assert total>=RELEASE
    output.mkdir()
    scenario=read(AS)
    initial=bench.STARTS/'nominal.npz'
    initial_sha=read(bench.STARTS/'BANK.json')['nominal']['sha256']
    c,delta=build_scene(scenario,initial,initial_sha)
    local=Local(c,WALK_MS)
    base=physical.frozen_policy()
    brake=Weight(local,read(PLAN))
    record=FloorRecord(c,total) if native else None
    c.recorder=record
    extras=[Path(__file__).resolve(),AS,PLAN,SAFE,initial,SOURCE/'RESULT.json',SOURCE/'TRACE.npz',
            physical.BASE/'prototype.py',physical.BASE/'walking_bench.py',physical.NATIVE_LIB,
            ORIGINAL/'floor_contacts.cpp',PREFIX/'RESULT.json',
            *(PREFIX/f for f in evidence)]
    ident=stan.identity.capture(c,dict(scope='verified same-event AS brake then explicit renewed propulsion',
          restart_mode=mode,restarted_at_ms=RELEASE,scenario=scenario,initial_snapshot_sha256=initial_sha),
          extra_paths=extras)
    stan.put(output/'RUN_IDENTITY.json',ident)
    tr=[]; dc=[]
    m=c.measure()
    a=np.zeros(42)
    corr=a.copy()
    sustained=np.zeros(total+1,dtype=np.int8)
    def snap():
        pad=c.d.geom_xpos[local.pad_ids].mean(0).copy()
        tr.append((float(c.d.time),c.d.qpos.copy(),c.d.qvel.copy(),c.d.subtree_com[c.th].copy(),
                    m['loads'].copy(),float(m['nonfoot']),m['padz'].copy(),m['speed'].copy(),
                    m['foot'].copy(),float(c.d.xmat[c.th,8]),c.actuator.target_rad.copy(),c.d.ctrl.copy(),
                    a.copy(),corr.copy(),pad))
    snap()
    for t in range(total):
        if t%5==0:
            intent=int((500<=t<RELEASE) or (mode=='immediate' and RELEASE<=t<RELEASE+WALK_MS))
            x=bench.features(c,m)
            local_value=local.sense(m)
            b=np.asarray(base(x,intent),dtype=np.float64)
            if t<RELEASE:
                a=brake(t,x,intent,local_value,b,m)
            else:
                # The requested event releases the already latched brake.
                # Baseline controller now sees the actual recovered posture.
                a=b.copy()
            a=bench.legacy.bounded_action(a)
            corr=a-b
            dc.append((t,x.copy(),intent,local_value.copy(),b.copy(),a.copy()))
        m=c.tick(a)
        if np.any(c.d.qfrc_applied) or np.any(c.d.xfrc_applied):
            raise RuntimeError('External force appeared')
        snap()
        sustained[t+1]=0 if t+1<RELEASE else int(mode=='immediate' and t+1<RELEASE+WALK_MS)
    trace=tuple(np.asarray(v) for v in zip(*tr,strict=True))
    decisions=tuple(np.asarray(v) for v in zip(*dc,strict=True))
    prefix_check=check_prefix(trace,decisions,record,RELEASE)
    samples=dict(zip(TRACE_KEYS,trace,strict=True))
    samples['renewed_propulsion']=sustained
    np.savez_compressed(output/'TRACE.npz',**samples)
    np.savez_compressed(output/'DECISIONS.npz',**dict(zip(DECISION_KEYS,decisions,strict=True)))
    test=bench.metrics(samples,RELEASE,WALK_MS,END)
    prestand=bench.standing(samples,RELEASE-260,RELEASE)
    interim=bench.standing(samples,0,min(500,total))
    first=test['first_step']
    wm=RELEASE*80
    resume_movement=float(samples['com'][min(RELEASE+WALK_MS,total),0]-samples['com'][RELEASE,0])
    initial_brake=physical.event_metrics({key:value[:RELEASE+1] for key,value in samples.items()},scenario)
    restart_status=dict(prefix=prefix_check,static_before_restart=prestand['passed'],
         commanded_restart_ms=RELEASE if mode=='immediate' else None,first_resume_strict_step=first,
         first_resume_latency_ms=(first['time']-samples['time'][RELEASE])*1.e3 if first else None,
         pre_restart_original_actuating_ms=brake.onset,post_gate_forward_displacement_native=resume_movement,
         no_root_writes_after_initial_restore=True,zero_external_force=True)
    test.update(scope='same continuous AS safe stop and renewed propulsion',mode=mode,
         start='nominal',scenario=scenario,scenario_source=str(AS.resolve()),initial_snapshot_sha256=initial_sha,
         walk_ms=WALK_MS,walk_begin_ms=RELEASE,actual_ms=total,local_stop_original=initial_brake,
         restarted_propulsion_ms=RELEASE if mode=='immediate' else None,expected_final_ms=END,
         elapsed_wall_s=time.monotonic()-known_elapsed,restart=restart_status,complete=total==END,
         original_initial_standing=interim,prefix_stance_before_release=prestand,
         trace_sha256=stan.sha(output/'TRACE.npz'),decision_sha256=stan.sha(output/'DECISIONS.npz'),
         physical_native_iterations=total*80)
    # Use the exact stock all-native measurement with the post-stop window.
    if record is not None:
        before=physical.WARM
        physical.WARM=RELEASE
        try:
            audit=physical.native_metrics(record,output,test,samples)
        finally:
            physical.WARM=before
        test['native_audit_sha256']=stan.sha(output/'NATIVE_AUDIT.json')
        test['native_joint_pass']=audit['native_joint_pass']
    stan.identity.verify_runtime(ident,c)
    stan.put(output/'RESULT.json',test)
    print(json.dumps({'mode':mode,'actual_ms':total,'joint_sampled_pass':test['joint_sampled_pass'],
                      'native_joint_pass':test.get('native_joint_pass'),'restart':restart_status,'cycles':test['cycles']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['immediate','no_restart'])
    p.add_argument('output',type=Path)
    p.add_argument('--native',action='store_true')
    p.add_argument('--probe-ms',type=int)
    a=p.parse_args()
    main(a.output.resolve(),a.mode,a.probe_ms,a.native)
