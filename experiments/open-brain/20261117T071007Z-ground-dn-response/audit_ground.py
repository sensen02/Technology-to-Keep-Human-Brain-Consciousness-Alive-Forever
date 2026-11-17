"""Check per-axon arrival, foot measurements, stop conditions and independent replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from ground_response import BASE, CATALOG, DT, HERE, PROBES, TARGETS, TICKS, WEIGHT, case_for, first, motion_window, score
from circuit import TRANSMISSION_TICKS


def read(path):
    return json.loads(path.read_text())


def load(directory):
    result=read(directory/'RESULT.json')
    path=directory/'TRACE.npz'
    assert result['trace_sha256']==first.sha(path)
    assert result['body_weight_sha']==first.sha(WEIGHT)
    assert result['frozen_init_sha']==first.sha(first.MODEL_FILE)
    with np.load(path,allow_pickle=False) as arrays:
        values=dict(arrays)
    assert len(values['trace'])==TICKS and values['ctrl'].shape==(TICKS//10,42)
    kernel=np.full(51,1./51)
    support=np.column_stack([np.convolve(load,kernel,mode='same')>=.02 for load in values['normal_load'].T])
    assert np.array_equal(values['trace'][:,9:15],support)
    return result,values


def pairing(name,variant,directory,replays=None):
    base=f'{name}_{variant}'
    opened,a=load(directory/(base+'_open'))
    cut,b=load(directory/(base+'_cut'))
    case=case_for(name,variant)
    assert opened['case']==cut['case']
    assert opened['calibration_sha256']==cut['calibration_sha256']
    assert opened['ports']==cut['ports']
    catalog=read(CATALOG)['ports']
    order=opened['ports']
    affected=[order.index(root) for root in cut['cut_roots']]
    forward=[i for i,root in enumerate(order) if catalog[root]['group']=='forward']
    uncut=[i for i,root in enumerate(order) if catalog[root]['direction']=='DN' and i not in affected]
    assert affected and set(forward)<=set(uncut)
    first_arrival=int(np.flatnonzero(a['arrived'][:,affected].any(axis=1))[0])
    expected=np.zeros_like(a['arrived'][:,affected])
    expected[TRANSMISSION_TICKS:]=a['emitted'][:-TRANSMISSION_TICKS,affected]
    checks=dict(same_dn_emission=np.array_equal(a['emitted'][:,affected+uncut],b['emitted'][:,affected+uncut]),
                physical_prefix_same=np.array_equal(a['trace'][:first_arrival],b['trace'][:first_arrival]),
                exact_crossing=np.array_equal(expected,a['arrived'][:,affected]),
                cut_zero=not b['arrived'][:,affected].any(),
                spared_exact=np.array_equal(a['arrived'][:,uncut],b['arrived'][:,uncut]),
                open_behavior=opened['pass_'],cut_behavior=cut['pass_'])
    extra={}
    if case.stride:
        yaw=opened['signed_yaw_deg']-cut['signed_yaw_deg']
        stride=opened['outer_minus_inner_mm']-cut['outer_minus_inner_mm']
        checks.update(stride_change=stride>=.025,yaw_change=yaw>=12.,cut_heading=abs(cut['yaw_deg'])<=12.)
        extra.update(paired_signed_yaw_deg=yaw,paired_stride_delta_mm=stride)
        legs=list(opened['leg_stride_mm'])
        rebuilt=score(case,'open',a['trace'],a['feet'],legs)
        assert rebuilt['leg_stride_mm']==opened['leg_stride_mm'] and rebuilt['gates']==opened['gates']
    else:
        block=slice(case.action+2600,case.action+5400)
        rates=int(a['arrived'][block,forward].sum())
        stop_pulses=int(a['arrived'][block,affected].sum())
        stopped=opened['stable_start_tick']
        checks.update(drive_still_arriving=rates>0,brake_still_arriving=stop_pulses>0,
                      sustained_stop_found=stopped is not None)
        extra.update(forward_arrivals_while_standing=rates,brake_arrivals_while_standing=stop_pulses,
                     first_still_after_arrival_ms=(stopped-first_arrival)*DT*1e3 if stopped is not None else None)
        assert opened['braking']==motion_window(a['trace'],(block.start,block.stop))
    if replays:
        for mode,original,array in (('open',opened,a),('cut',cut,b)):
            replay,repeated=load(replays/(base+'_'+mode))
            assert replay['gates']==original['gates']
            assert all(np.array_equal(value,repeated[field]) for field,value in array.items()), (name,variant,mode)
        extra['bit_exact_replay']=True
    return dict(pass_=bool(all(checks.values())),gates={k:bool(v) for k,v in checks.items()},
                first_arrival_tick=first_arrival,dn_emitted=int(a['emitted'][:,affected].sum()),
                dn_arrived=int(a['arrived'][:,affected].sum()),dn_arrived_cut=int(b['arrived'][:,affected].sum()),
                open=opened,cut=cut,**extra)


def audit(directory,split,out,replays):
    variants=('dev',) if split=='dev' else ('held0','held1')
    arms={f'{name}_{variant}':pairing(name,variant,directory,replays) for name in TARGETS for variant in variants}
    first.write(out,dict(pass_=all(arm['pass_'] for arm in arms.values()),arms=arms,
               raw_physical_steps=len(arms)*2*TICKS*(2 if replays else 1)))
    for name,arm in arms.items():
        print(name,arm['pass_'],'fail:',[k for k,v in arm['gates'].items() if not v],
              'yaw',arm.get('paired_signed_yaw_deg'),'stride',arm.get('paired_stride_delta_mm'),
              'halt',arm.get('first_still_after_arrival_ms'),
              flush=True)


def calibration(dev,out):
    telemetry = {'stride_left':(2,3),'stride_right':(3,2),'bluebell':(4,5),'web_b':(6,7)}
    samples={}
    sources={}
    for name,(i,j) in telemetry.items():
        result, arrays=load(dev/(name+'_dev_open'))
        onset=result['case']['action']
        window=arrays['levels'][onset+2600:onset+5400]
        if name.startswith('stride'):
            samples[name]=float(np.quantile(window[:,i]-window[:,j],.10))
        else:
            samples[name]=float(np.quantile(window[:,[i,j]].mean(axis=1),.10))
        sources[str((dev/(name+'_dev_open')/'TRACE.npz').resolve())]=result['trace_sha256']
    assert min(samples.values())>.005,samples
    first.write(out,dict(stride_mv={'left':samples['stride_left'],'right':samples['stride_right']},
               bluebell_mv=.88*samples['bluebell'],web_b_mv=.88*samples['web_b'],
               source_files=sources))
    print(samples,flush=True)


def freeze(out,selected):
    required=[HERE/name for name in ('PROTOCOL.zh-CN.md','MEASUREMENT_REVISION.zh-CN.md',
             'ground_response.py','audit_ground.py','batch.sh','SELECTION.txt','NATIVE_MEASUREMENT.zh-CN.md')]
    required+= [WEIGHT,CATALOG,PROBES,first.MODEL_FILE,first.HERE/'BODY.mjb',selected]
    parent=read(BASE/'PHASE/IDENTITY.json')['sha256']
    assert all(first.sha(name)==digest for name,digest in parent.items())
    sources={str(file.resolve()):first.sha(file) for file in required}
    sources.update(parent)
    catalog=read(CATALOG)['ports']
    ports={}
    groups={group for group,side in TARGETS.values()}|{'forward'}
    for root,row in catalog.items():
        if row['group'] not in groups:
            continue
        swc=first.PARENT/'geometry'/row['object']['filename']
        digest=first.sha(swc)
        assert digest==row['object']['sha256']
        sources[str(swc.resolve())]=digest
        ports[root]={'name':row['cell_type'],'side':row['side'],'swc':str(swc),
                     'crossing':row['geometry']['crossing_children'],'sign':row['sign']}
    assert len(ports)==8
    first.write(out,dict(sha256=sources,ports=ports))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','calibration','freeze'))
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--split',choices=('dev','held'),default='held')
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args()
    if args.action=='audit':
        audit(args.source,args.split,args.out,args.replay)
    elif args.action=='calibration':
        calibration(args.source,args.out)
    else:
        freeze(args.out,args.source)
