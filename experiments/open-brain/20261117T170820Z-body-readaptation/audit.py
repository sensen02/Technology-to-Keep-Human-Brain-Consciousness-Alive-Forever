"""Verify actual weight updates, fixed neck inputs and teacher-free physical replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from context import HERE,NECK,OLD_WEIGHT,first,hash_file
from tasks import EXTRA
from fixed_neck import parameters,full_catalog


def read(path):
    return json.loads(path.read_text())


def load(path):
    entry=read(path/'RESULT.json')
    trace=path/'TRACE.npz'
    assert entry['trace_sha256']==hash_file(trace)
    with np.load(trace,allow_pickle=False) as archive:
        arrays=dict(archive)
    return entry,arrays


def verify_transit(entry,arrays):
    order=entry.get('arrival_roots',entry.get('ports'))
    explicit=set(entry.get('cut_roots',entry.get('cut',[])))
    if 'task' in entry:
        case=entry['task']
        if case['mode']=='cut':
            target={'stride_left':('walking_outer_stride','left'),'stride_right':('walking_outer_stride','right'),
                    'bluebell':('halt_bluebell',None),'web_b':('halt_web_b',None)}[case['family']]
            explicit.update(root for root in order if full_catalog()['ports'][root]['group']==target[0]
                            and (target[1] is None or full_catalog()['ports'][root]['side']==target[1]))
    else:
        case=entry['case']
    counts=len(arrays['arrived'])
    expected=np.zeros_like(arrays['arrived'])
    for i,root in enumerate(order):
        if root in explicit:continue
        delay=parameters()['ports'][root]['delay_ticks']
        expected[delay:,i]=arrays['emitted'][:-delay,i]
        if case.get('cut_end') and root in {r for r,p in full_catalog()['ports'].items() if p['direction']=='DN'}:
            expected[case['cut_start']:min(counts,case['cut_end']+delay),i]=False
    assert np.array_equal(expected,arrays['arrived'])


def weight_digest(entry):
    return entry.get('weight_sha256',entry.get('body_weight_sha',entry.get('body_weight_sha256')))


def verify_pair(path,replay,checkpoint):
    result,a=load(path)
    repeated,b=load(replay)
    assert weight_digest(result)==weight_digest(repeated)==hash_file(checkpoint)
    assert result['gates']==repeated['gates']
    identical={key:bool(np.array_equal(value,b[key])) for key,value in a.items()}
    assert all(identical.values()),(path,identical)
    verify_transit(result,a)
    return dict(pass_=result['pass_'],source=result,exact_replay=identical,
                actual_physical_steps=2*len(a['trace']))


def summarize(checkpoint):
    expected=[line.split() for line in (NECK/'FORMAL_JOBS.txt').read_text().splitlines()]
    checks={}
    comparisons={}
    for suite,name in expected:
        key=suite+'_'+name
        checked=verify_pair(HERE/'FORMAL'/key,HERE/'REPLAY'/key,checkpoint)
        checks[key]=checked
        old=read(NECK/'FORMAL'/key/'RESULT.json')
        new=checked['source']
        comparisons[key]=dict(baseline_pass=old['pass_'],retrained_pass=new['pass_'],
            baseline_speed=old.get('active_speed_mm_s'),retrained_speed=new.get('active_speed_mm_s'),
            baseline_yaw=old.get('yaw_degrees',old.get('yaw_deg')),
            retrained_yaw=new.get('yaw_degrees',new.get('yaw_deg')),
            baseline_halt_rms=old.get('braking',{}).get('rms_mm_s'),
            retrained_halt_rms=new.get('braking',{}).get('rms_mm_s'))
    extra={}
    extra_baseline={}
    for task in EXTRA:
        key=task.name
        extra[key]=verify_pair(HERE/'EXTRA'/key,HERE/'EXTRA_REPLAY'/key,checkpoint)
        old,old_arrays=load(HERE/'EXTRA_BASELINE'/key)
        new,new_arrays=load(HERE/'EXTRA'/key)
        assert weight_digest(old)==hash_file(OLD_WEIGHT)
        assert old['task']==new['task'] and old['parameter_sha256']==new['parameter_sha256']
        ports=new['ports']
        dn=[i for i,root in enumerate(ports) if full_catalog()['ports'][root]['direction']=='DN']
        assert np.array_equal(old_arrays['emitted'][:,dn],new_arrays['emitted'][:,dn])
        assert np.array_equal(old_arrays['arrived'][:,dn],new_arrays['arrived'][:,dn])
        verify_transit(old,old_arrays)
        extra_baseline[key]=dict(pass_=old['pass_'],source=old,actual_physical_steps=len(old_arrays['trace']))
    original=torch.load(OLD_WEIGHT,weights_only=True,map_location='cpu')['state']
    new=torch.load(checkpoint,weights_only=True,map_location='cpu')['state']
    updates={}
    for name,value in new.items():
        difference=(value-original[name]).double()
        updates[name]=dict(l2=float(difference.norm()),maximum=float(difference.abs().max()))
    assert updates['fourier']['l2']>0
    assert all(updates[key]['l2']>0 for key in updates if key.startswith('reflex.'))
    identity=read(HERE/'TRAINING_INPUTS.json')
    assert all(hash_file(Path(path))==digest for path,digest in identity['sha256'].items())
    steps=sum(row['actual_physical_steps'] for row in checks.values())+sum(
        row['actual_physical_steps'] for row in extra.values())+sum(row['actual_physical_steps'] for row in extra_baseline.values())
    result=dict(pass_=all(c['pass_'] for c in checks.values()) and all(c['pass_'] for c in extra.values()),
       checkpoint=str(checkpoint.resolve()),weight_sha256=hash_file(checkpoint),original_weight_sha256=hash_file(OLD_WEIGHT),
       cases=checks,extra=extra,baseline_extra=extra_baseline,matched_comparisons=comparisons,
       parameter_updates=updates,actual_physical_steps=steps,independent_replay_count=len(checks)+len(extra),
       neck_parameters_sha=hash_file(NECK/'PARAMETERS.json'),neck_readout_sha=hash_file(NECK/'READOUT.json'))
    first.write(HERE/'FINAL.json',result)
    print('Regression',sum(c['pass_'] for c in checks.values()),'/',len(checks),
          'new conditions',sum(c['pass_'] for c in extra.values()),'/',len(extra),
          'baseline extra',sum(c['pass_'] for c in extra_baseline.values()),'/',len(extra_baseline),
          'independent replays',result['independent_replay_count'],
          'steps',steps,flush=True)
    for name,item in {**checks,**extra}.items():
        if not item['pass_']:
            print('FAILED',name,[k for k,v in item['source']['gates'].items() if not v],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    summarize(args.checkpoint)
