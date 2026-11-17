"""Check exact prefixes, raw trajectories, and independent arc repeats."""
import argparse
from pathlib import Path

import numpy as np

from rollout import HERE,stan,read,UNITS


def bound(run):
    result=read(run/'RESULT.json')
    for key,leaf in (('trace_sha256','TRACE.npz'),('decisions_sha256','DECISIONS.npz'),('native_sha256','NATIVE.npz')):
        if stan.sha(run/leaf)!=result[key]:
            raise ValueError(f'Hash mismatch {run/leaf}')
    return result


def one(candidate,direct,repeat=None):
    r=bound(candidate)
    d=bound(direct)
    assert r['direction']==d['direction'] and r['bounds']==d['bounds']
    assert r['command_units']==d['command_units']==UNITS
    entry,arc,coast,finish=r['bounds']
    before={}
    for filename,index in [('TRACE.npz',arc+1),('DECISIONS.npz',arc//5),('NATIVE.npz',arc*80)]:
        with np.load(candidate/filename,allow_pickle=False) as x,np.load(direct/filename,allow_pickle=False) as y:
            for key in x.files:
                np.testing.assert_array_equal(x[key][:index],y[key][:index],err_msg=f'{filename}/{key}')
            before[filename]=list(x.files)
    exact=None
    if repeat is not None:
        u=bound(repeat)
        assert u['passed']==r['passed']
        exact={}
        for name in ('TRACE.npz','DECISIONS.npz','NATIVE.npz'):
            exact[name]=bool(stan.sha(repeat/name)==stan.sha(candidate/name))
        assert all(exact.values()),exact
    known=r['legacy_exit_window']
    switch=d['legacy_exit_window']
    return dict(direction=r['direction'],candidate=str(candidate.resolve()),direct_control=str(direct.resolve()),
                all_pre_exit_bitwise=before,independent_whole_bitwise=exact,
                native_integrations=r['native_integrations'],all_gates=r['checks'],passed=r['passed'],
                prior_archive_ratio=r['original_direct_switch_reference'],
                matched_abrupt_ratio=switch['max_normalized_slip'],
                native_continuous_legacy_exit_ratio=known['max_normalized_slip'],
                native_full_exit_ratio=r['full_exit_window']['max_normalized_slip'],
                native_straight_ratio=r['straight_window']['max_normalized_slip'],
                signed_forward_native_s=known['forward_native_s'],straight_yaw_residual_deg=r['straight_window']['net_yaw_deg'],
                completed_exit_cycles=r['full_exit_window']['cycles'],terminal_stand=r['original_arc']['tail'],
                terminal_yaw_residual_deg=r['terminal_residual_yaw_deg'],
                before_switch_ms=arc,continued_straight_measured_from_ms=r['straight_begin_ms'],
                arc_net_yaw=r['original_arc']['actual_turn_deg'],coast_ms=coast-arc,full_D1_D8=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('blended',type=Path)
    p.add_argument('ablation',type=Path)
    p.add_argument('--repeat',type=Path)
    a=p.parse_args()
    result=one(a.blended,a.ablation,a.repeat)
    stan.put(a.output,result)
    print(result)
