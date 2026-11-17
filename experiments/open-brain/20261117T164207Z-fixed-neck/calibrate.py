"""Calibrate a static receiver on held-out neural stimuli, with no body fitting."""
import argparse
from collections import defaultdict
import json

import numpy as np
import torch
from scipy.optimize import isotonic_regression

from fixed_neck import FixedCircuit, HERE, LegacyCircuit, PARENT, RECEPTORS, full_catalog, write
from physical import Receiver
from build_catalog import hash_file

PERIODS=(180,120,90,70,60,50,40,30)
TICKS=3000


def make(group,side):
    catalog=full_catalog()['ports']
    ports={root:port for root,port in catalog.items()
           if port['group']==group and (side is None or port['side']==side)
           or port['group'] in ('front_touch','front_load')}
    return {'ports':ports}


def probes(receiver,circuit):
    anchors=json.loads((PARENT/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json').read_text())['probes']
    for key,(_,_,sign) in receiver.probes.items():
        roots=anchors['/'.join(key)]
        receiver.probes[key]=roots,[circuit.index[r] for r in roots],sign


def pair(group,side,out):
    torch.set_num_threads(1)
    catalog=make(group,side)
    old=LegacyCircuit(catalog,batch=len(PERIODS))
    new=FixedCircuit(catalog,batch=len(PERIODS))
    old_receiver=Receiver(old)
    new_receiver=Receiver(new)
    probes(old_receiver,old)
    probes(new_receiver,new)
    side_names=sorted({p['side'] for p in catalog['ports'].values() if p['group']==group})
    targets={s:old_receiver.probes[group,s][1] for s in side_names}
    old_read=np.zeros((TICKS,len(PERIODS),len(side_names)))
    new_read=np.zeros_like(old_read)
    old_state=np.zeros((len(PERIODS),len(old.roots)),dtype=np.float32)
    new_state=np.zeros((len(PERIODS),len(new.roots)),dtype=np.float32)
    source_roots=[root for root,port in catalog['ports'].items() if port['group']==group]
    assert source_roots
    direction=catalog['ports'][source_roots[0]]['direction']
    if direction=='AN':
        source_roots=sorted({pre for r in source_roots for pre,n in catalog['ports'][r]['incoming_receptors']})
    old_columns=old.validate_inputs(source_roots,'brain' if direction=='DN' else 'body')
    new_columns=new.validate_inputs(source_roots,'brain' if direction=='DN' else 'body')
    mask=np.zeros(len(old.ports),dtype=bool)
    new_targets={s:new_receiver.probes[group,s][1] for s in side_names}
    source_counts=np.zeros((2,len(PERIODS)),dtype=int)
    sign=catalog['ports'][next(r for r in catalog['ports'] if catalog['ports'][r]['group']==group)]['sign']
    for tick in range(TICKS):
        old_state.fill(0.)
        new_state.fill(0.)
        if tick>=400:
            active=[lane for lane,period in enumerate(PERIODS) if (tick-400)%period==0]
            old_state[np.ix_(active,old_columns)]=68.75
            new_state[np.ix_(active,new_columns)]=68.75
        emitted,_,_,v=old.step(old_state,mask)
        sent,_,_,w=new.step(new_state,mask)
        source_counts[0]+=emitted[:,[old.port_index[r] for r,p in old.ports.items() if p['group']==group]].sum(1)
        source_counts[1]+=sent[:,[new.port_index[r] for r,p in new.ports.items() if p['group']==group]].sum(1)
        for j,s in enumerate(side_names):
            old_read[tick,:,j]=((v[:,targets[s]]+52)*sign).mean(1)
            new_read[tick,:,j]=((w[:,new_targets[s]]+52)*sign).mean(1)
    out.mkdir()
    np.savez_compressed(out/'CALIBRATION.npz',old=old_read,new=new_read,counts=source_counts)
    write(out/'INPUT.json',dict(group=group,side=side,period_ticks=PERIODS,ticks=TICKS,
          static_windows=[1500,2900],parameter_sha=hash_file(HERE/'PARAMETERS.json'),
          catalog_sha=hash_file(PARENT/'CATALOG.json'),code_sha=hash_file(HERE/'calibrate.py')))
    write(out/'RESULT.json',dict(source_counts=source_counts.tolist(),sides=side_names,
              old_steady=old_read[1500:2900].mean(0).tolist(),new_steady=new_read[1500:2900].mean(0).tolist()))
    print(group,side,'counts',source_counts.tolist(),flush=True)


def collect(folder):
    maps={}
    for file in sorted(folder.glob('*/RESULT.json')):
        result=json.loads(file.read_text())
        info=json.loads((file.parent/'INPUT.json').read_text())
        for column,side in enumerate(result['sides']):
            pairs=[(0.,0.)]+[(float(n[column]),float(o[column]))
                    for n,o in zip(result['new_steady'],result['old_steady'],strict=True)]
            pairs.sort()
            xs=np.array([x for x,y in pairs])
            ys=np.array([y for x,y in pairs])
            ys=np.asarray(isotonic_regression(ys).x)
            # Merge equal levels. Extrapolation uses only the fitted maximum.
            unique,indices=np.unique(xs,return_index=True)
            maps[info['group']+'/'+side]={'new':unique.tolist(),'old':ys[indices].tolist(),
                   'source':str(file.parent/'CALIBRATION.npz'),
                   'source_sha':hash_file(file.parent/'CALIBRATION.npz')}
    expected={p['group']+'/'+p['side'] for p in full_catalog()['ports'].values() if p['direction']=='DN'}
    assert expected<=maps.keys(),expected-maps.keys()
    write(HERE/'READOUT.json',dict(maps=maps,rule='monotone static matched-rate mean voltage; no temporal inversion',
          parameters=hash_file(HERE/'PARAMETERS.json'),model_training=False))
    print('maps',len(maps))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','collect'))
    parser.add_argument('folder')
    parser.add_argument('--group')
    parser.add_argument('--side',choices=('left','right'))
    args=parser.parse_args()
    from pathlib import Path
    if args.action=='run':
        pair(args.group,args.side,Path(args.folder))
    else:
        collect(Path(args.folder))
