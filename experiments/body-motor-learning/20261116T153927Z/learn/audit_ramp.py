"""Trace-matched overlay incline and level geometric control."""
import argparse
from pathlib import Path

import numpy as np

from slope import STAND, W, bench


def load(folder):
    result=bench.read(folder/'RESULT.json')
    for f,expected in result['files'].items():
        if STAND.sha(folder/f)!=expected:
            raise ValueError(f'Raw archive changed at {folder/f}')
    return result


def ending(folder,settings):
    with np.load(folder/'TRACE.npz',allow_pickle=False) as data:
        begin=data['com'][0]
        end=data['com'][-1]
        x=float(data['pad_min_x'][-1].min())
        net=end-begin
    with np.load(folder/'NATIVE.npz',allow_pickle=False) as data:
        forces=data['forces']
        ensure=data['pose'][:,7]
        assert not ensure.any()
        load=data['ramp_normal'][:,:6]
        last=forces[-min(16000,len(forces)):]
        supported=np.flatnonzero(np.any(load>=.02*W,axis=1))
    return dict(ms=settings['actual_ms'],native_steps=settings['native_steps'],complete=settings['complete'],
                qualified=settings['passed'],net_com=net.tolist(),terminal_back_foot_x=x,
                first_incline_load_ms=float(supported[0]/80.) if len(supported) else None,
                terminal_mean_support_W=float(last[:,:6].sum(1).mean()/W),
                terminal_nonfoot_max_W=float(last[:,12].max()/W))


def compare_prefixes(left_path,right_path,interest_ms):
    fields=('qpos','qvel','com','loads')
    with np.load(left_path/'TRACE.npz',allow_pickle=False) as left, np.load(right_path/'TRACE.npz',allow_pickle=False) as right:
        cutoff=min(len(left['qpos']),len(right['qpos']),interest_ms+1)
        differ=np.any(left['qpos'][:cutoff]!=right['qpos'][:cutoff],axis=1)
        first=np.flatnonzero(differ)
        equal=int(first[0]) if len(first) else cutoff
        state_prefix={field:bool(np.array_equal(left[field][:equal],right[field][:equal])) for field in fields}
    with np.load(left_path/'DECISIONS.npz',allow_pickle=False) as left, np.load(right_path/'DECISIONS.npz',allow_pickle=False) as right:
        decisions=min(len(left['ms']),len(right['ms']),max((equal-1)//5,0))
        decision_prefix={field:bool(np.array_equal(left[field][:decisions],right[field][:decisions]))
                         for field in ('x','command','action')}
    with np.load(left_path/'NATIVE.npz',allow_pickle=False) as left, np.load(right_path/'NATIVE.npz',allow_pickle=False) as right:
        steps=max((equal-1)*80,0)
        native_prefix={name:bool(np.array_equal(left[name][:steps],right[name][:steps])) for name in left.files}
    return dict(interest_ms=interest_ms,first_different_qpos_ms=equal if len(first) else None,
                exact_endpoint_stamps=equal,same_state_prefix=state_prefix,same_decisions_before=decision_prefix,
                native_steps_before=steps,native_prefix_bitwise=native_prefix)


def run(candidate,control,output,short=None):
    candidate_report,control_report=map(load,(candidate,control))
    setting1=bench.read(candidate/'RUN_IDENTITY.json')['config']
    setting0=bench.read(control/'RUN_IDENTITY.json')['config']
    signatures={key:setting0[key]==setting1[key] for key in ('initial_snapshot_sha256','ground')}
    def compare_time(a,b):
        fields=('time','qpos','qvel','com','loads')
        with np.load(a/'TRACE.npz',allow_pickle=False) as left, np.load(b/'TRACE.npz',allow_pickle=False) as right:
            initial={key:bool(np.array_equal(left[key][0],right[key][0])) for key in fields}
            horizon=min(len(left['time']),len(right['time']))
            a1,b1=left['qpos'][:horizon],right['qpos'][:horizon]
            div=np.flatnonzero(np.any(a1!=b1,axis=1))
        return initial,int(div[0]) if len(div) else None
    initial,first=compare_time(candidate,control)
    slope=ending(candidate,candidate_report)
    plain=ending(control,control_report)
    static_match=all(initial.values()) and all(signatures.values())
    same_schedule=candidate_report['plan']['course_ms']==control_report['plan']['course_ms']
    status=bool(static_match and same_schedule and candidate_report['passed'] and control_report['passed'])
    terrain_start=slope['first_incline_load_ms']
    before_contact=compare_prefixes(candidate,control,int(terrain_start)) if terrain_start is not None else None
    earlier={}
    if short is not None:
        probe=load(short)
        crossing=probe['first_com_x1_ms']
        earlier=dict(short_source=str(short),short_crossing_ms=crossing,
                    short_prefix=compare_prefixes(candidate,short,crossing))
    STAND.put(output,dict(source_candidate=str(candidate),source_control=str(control),same_initial_fields=initial,
                  same_identity_essentials=signatures,initial_archived_sensor_delta=[setting1['initial_observation_max_abs_delta'],
                   setting0['initial_observation_max_abs_delta']],same_declared_course_duration=same_schedule,
                  first_dynamic_state_difference_ms=first,verified_signed_elevation_difference=slope['net_com'][2]-plain['net_com'][2],
                  candidate=slope,control=plain,matched_level_prefix=before_contact,passed=status,**earlier))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path)
    p.add_argument('control',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--short',type=Path)
    a=p.parse_args()
    run(a.candidate.resolve(),a.control.resolve(),a.output.resolve(),a.short.resolve() if a.short else None)
