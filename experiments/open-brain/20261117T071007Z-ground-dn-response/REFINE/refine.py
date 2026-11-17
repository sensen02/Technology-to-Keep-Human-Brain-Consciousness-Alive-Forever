"""Recalibrate DNg13 from development VNC signals and repeat with unseen seeds."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from audit_ground import load, pairing, read
from ground_response import CATALOG, WEIGHT, case_for, first, run as rollout

TARGETS=('stride_left','stride_right')
VARIANTS=('held0','held1')
PARAMS=ROOT/'CALIBRATION.json'


def trial(name,variant):
    idx=VARIANTS.index(variant)
    seed=(441,532)[idx]+24*TARGETS.index(name)
    return replace(case_for(name,variant),seed=seed,period=(53,69)[idx],push=idx)


def calibrate():
    refs={}
    parameters={}
    for name in TARGETS:
        folder=ROOT.parent/'DEV_FINAL'/f'{name}_dev_open'
        result,arrays=load(folder)
        onset=result['case']['action']
        levels=arrays['levels'][onset+2600:onset+5400,2:4]
        delta=float(np.median(np.abs(levels[:,0]-levels[:,1])))
        assert delta>.005
        parameters[name.rsplit('_',1)[-1]]=delta*.25/.15
        refs[str(folder/'TRACE.npz')]=result['trace_sha256']
    first.write(PARAMS,dict(stride_mv=parameters,bluebell_mv=1.2,web_b_mv=1.2,source_files=refs))
    sources=[ROOT/'PREREG.zh-CN.md',ROOT/'refine.py',ROOT/'batch.sh',PARAMS,WEIGHT,CATALOG,
             ROOT.parent/'IDENTITY.json',ROOT.parent/'ground_response.py',ROOT.parent/'audit_ground.py']
    first.write(ROOT/'IDENTITY.json',dict(sha256={str(path):first.sha(path) for path in sources},
                cases={f'{name}_{variant}':asdict(trial(name,variant)) for name in TARGETS for variant in VARIANTS}))


def verify():
    manifest=read(ROOT/'IDENTITY.json')
    for path,digest in manifest['sha256'].items():
        assert first.sha(path)==digest,path
    for path,digest in read(PARAMS)['source_files'].items():
        assert first.sha(path)==digest,path
    formal=ROOT/'FORMAL'
    replay=ROOT/'REPLAY'
    arms={}
    for name in TARGETS:
        for variant in VARIANTS:
            case=manifest['cases'][f'{name}_{variant}']
            legacy=asdict(case_for(name,variant))
            assert all(case[k]==legacy[k] for k in ('name','on','action','off'))
            assert case['seed']!=legacy['seed'] and case['period']!=legacy['period']
            for location in (formal,replay):
                for mode in ('open','cut'):
                    record=read(location/f'{name}_{variant}_{mode}'/'RESULT.json')
                    assert record['case']==case
                    assert record['calibration_sha256']==first.sha(PARAMS)
            result=pairing(name,variant,formal,replay)
            arms[f'{name}_{variant}']=result
            print(name,variant,result['pass_'],result['open']['yaw_deg'],
                  result['paired_stride_delta_mm'],[k for k,v in result['gates'].items() if not v],flush=True)
    first.write(ROOT/'FINAL.json',dict(pass_=all(r['pass_'] for r in arms.values()),arms=arms,
                raw_physical_steps=320000))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('calibrate','run','verify'))
    parser.add_argument('name',nargs='?',choices=TARGETS)
    parser.add_argument('variant',nargs='?',choices=VARIANTS)
    parser.add_argument('mode',nargs='?',choices=('open','cut'))
    parser.add_argument('out',nargs='?',type=Path)
    args=parser.parse_args()
    if args.action=='calibrate':
        calibrate()
    elif args.action=='run':
        rollout(trial(args.name,args.variant),args.mode,args.out,PARAMS)
    else:
        verify()
