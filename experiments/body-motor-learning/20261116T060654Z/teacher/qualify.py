"""Pair same-scene runs to verify a physical, isolated recovery effect."""
import argparse
from pathlib import Path

import numpy as np

from physical import W, read, stan


def file_index(run):
    result = read(run/'RESULT.json')
    hashes = {name:stan.sha(run/name) for name in ('RESULT.json','TRACE.npz','DECISIONS.npz','NATIVE_AUDIT.json')}
    assert hashes['TRACE.npz']==result['trace_sha256']
    assert hashes['DECISIONS.npz']==result['decision_sha256']
    assert hashes['NATIVE_AUDIT.json']==result['native_audit_sha256']
    audit=read(run/'NATIVE_AUDIT.json')
    assert stan.sha(run/'NATIVE.npz') == audit['native_sha256']
    return result,audit,hashes


def qualify(run, control, output, absent=None):
    r, raw, files=file_index(run)
    c, off, cfiles=file_index(control)
    assert c['mode']=='control' and r['mode'] in ('teacher','student')
    assert r['scenario'] == c['scenario'] and r['scenario'] is not None
    assert r['initial_snapshot_sha256']==c['initial_snapshot_sha256']
    assert r['walk_ms']==c['walk_ms']==9600
    assert r['recipient_leg']==c['recipient_leg']=='RF'
    first, end = r['event']['obstacle_first_contact_ms'], r['event']['obstacle_last_contact_ms']
    event = r['event']
    reload = event.get('first_target_floor_reload_ms')
    bound = c['event']['obstacle_first_contact_ms']
    prefix = min(first if first is not None else 11300,bound if bound is not None else 11300)
    scene = r['scenario']
    past = float(scene['center'][0]+scene['half_size'][0])
    floor_and_beyond = bool(reload is not None and end is not None and end < reload
        and event['target_pad_xyz_at_reload'][0]>=past+.02)
    with np.load(run/'TRACE.npz',allow_pickle=False) as a, np.load(control/'TRACE.npz',allow_pickle=False) as b, \
            np.load(run/'DECISIONS.npz',allow_pickle=False) as d:
        change = d['action'] - d['base']
        driven=np.flatnonzero(np.any(change!=0.,axis=1)&(d['intent']==1))
        trained_onset = int(d['ms'][driven[0]]) if len(driven) else None
        axes=('qpos','qvel','com','loads','padz','target','ctrl','applied')
        exact=all(np.array_equal(a[key][:prefix],b[key][:prefix]) for key in axes)
        no_pretouch_change = bool(trained_onset is not None and first is not None and trained_onset>=first)
        lift_gain=event.get('target_pad_peak_during_recovery',0.)-c['event'].get('target_pad_peak_during_recovery',0.)
        # Touched snapshots are independent of actuation onset.
        else_changed=bool(change[:,:21].any() or change[:,28:].any())
        absent_exact=None
        if absent is not None:
            clean,cleanraw,cleanh=file_index(absent)
            assert clean['mode']=='control' and clean['scenario'] is None
            assert clean['initial_snapshot_sha256']==r['initial_snapshot_sha256']
            with np.load(absent/'TRACE.npz',allow_pickle=False) as z:
                absent_exact=all(np.array_equal(b[key][:prefix],z[key][:prefix]) for key in axes)

    check=dict(complete_frames=r['actual_ms']==11300 and raw['native_integrations']==904000,
               native_joint=raw['native_joint_pass'], sampled_joint=r['joint_sampled_pass'],
               preceding_obstacle_control_exact=exact,
               sampled_and_native_touch=bool(r['event']['triggered'] and raw['obstacle_contact_native_steps']>0),
               no_wrong_leg_or_segment=raw['obstacle_wrong_geom_native_steps']==0,
               no_extra_forces=raw['zero_extra_force'],
               unloaded_front_leg_at_touch=event.get('target_leg_load_at_touch_W',1.)<.08,
               only_RF_correction=not else_changed,
               acts_after_contact=no_pretouch_change,
               floor_reload_beyond_obstacle=floor_and_beyond,
               brief_obstacle_event=bool(first is not None and end is not None and end-first<=400),
               support_reserved=event.get('minimum_other_ground_support_W',0.)>.75,
               deliberate_lift=lift_gain>=.020)
    if absent is not None:
        check['preceding_absent_control_exact']=absent_exact
    out=dict(passed=all(check.values()), checks=check, role=r['mode'], run=str(run), control=str(control),
             same_scene_controller_baseline_sha256=cfiles['RESULT.json'],
             result_sha256=files['RESULT.json'],trace_sha256=files['TRACE.npz'],
             decision_sha256=files['DECISIONS.npz'], native_audit_sha256=files['NATIVE_AUDIT.json'],
             native_samples=raw['native_integrations'], affected_leg='RF', around_touch_bitwise_ms=prefix,
             intervention_onset_ms=trained_onset, stable_ground_reload_ms=reload, pad_clearance_gain=lift_gain,
             resisting_impulses_Ws=[off['obstacle_resisting_impulse_Ws'],raw['obstacle_resisting_impulse_Ws']],
             contact_native_steps=[off['obstacle_contact_native_steps'],raw['obstacle_contact_native_steps']],
             full_D1_D8=False,
             scope='Frozen nominal 0.5s stand, 9.6s walk and 1.2s stand with one physical box at RF swing')
    stan.put(output,out)
    print(out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('control',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--absent',type=Path)
    args=parser.parse_args()
    qualify(args.run.resolve(),args.control.resolve(),args.output.resolve(),
            args.absent.resolve() if args.absent else None)
