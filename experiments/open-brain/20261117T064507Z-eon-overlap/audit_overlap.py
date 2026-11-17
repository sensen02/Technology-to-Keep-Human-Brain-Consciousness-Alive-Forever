"""Independently check per-axon delivery, body metrics and exact replay."""
import argparse
from pathlib import Path

import numpy as np

from overlap import HERE, NAMES, WEIGHT, brain_record, digest, read, write
from circuit import TRANSMISSION_TICKS


def load(folder):
    part = read(folder/'RESULT.json')
    trace=folder/'TRACE.npz'
    assert digest(trace)==part['trace_sha256']
    assert part['body_weight_sha256']==digest(WEIGHT)
    with np.load(trace,allow_pickle=False) as data:
        content=dict(data)
    assert len(content['trace'])==17000 and content['ctrl'].shape==(1700,42)
    return part,content


def compare_delivery(result,trace):
    info, source = brain_record(Path(result['neural_input']))
    assert digest(Path(result['neural_input'])/'BRAIN.npz')==result['brain_sha256']
    operated = set(result['cut'])
    indices = [result['ports'].index(root) for root in info['roots']]
    data = trace['arrived'][:,indices]
    assert np.array_equal(trace['emitted'][:,indices],source)
    expected = np.zeros_like(source)
    expected[TRANSMISSION_TICKS:]=source[:-TRANSMISSION_TICKS]
    for i,root in enumerate(info['roots']):
        if root in operated:
            expected[:,i]=False
    assert np.array_equal(expected,data)
    return dict(emitted=int(source.sum()),arrived=int(data.sum()),
                exact_named_delays=True)


def compare_cut(records,key,cut_mode):
    intact, before = records[key+'_open']
    cut, after = records[key+'_'+cut_mode]
    assert intact['brain_sha256']==cut['brain_sha256'] and intact['ports']==cut['ports']
    ports=cut['ports']
    operated=[ports.index(root) for root in cut['cut']]
    immune=[ports.index(root) for root in cut['dn_ports'] if root not in cut['cut']]
    assert operated
    sent=int(after['emitted'][:,operated].sum())
    reached=int(before['arrived'][:,operated].sum())
    cut_arrival=int(after['arrived'][:,operated].sum())
    assert sent and reached and cut_arrival==0
    assert np.array_equal(before['emitted'][:,operated+immune],after['emitted'][:,operated+immune])
    assert np.array_equal(before['arrived'][:,immune],after['arrived'][:,immune])
    summary=dict(affected_roots=cut['cut'], emitted=sent, arrived_open=reached,
                 arrived_cut=cut_arrival, other_dn_exact=True, intact_body=intact['pass_'], cut_body=cut['pass_'])
    if cut_mode=='steer':
        angle=intact['yaw_degrees']
        new_angle=cut['yaw_degrees']
        summary.update(yaw_open=angle,yaw_cut=new_angle, proper_body_effect=(
            cut['pass_'] and abs(new_angle)<=.4*abs(angle)))
    else:
        summary.update(proper_body_effect=cut['pass_'] and intact['pass_'])
    return summary


def audit(formal,replay,out):
    records, same, delivery = {}, {}, {}
    for path in sorted(formal.glob('*/RESULT.json')):
        name=path.parent.name
        original, values=load(path.parent)
        records[name]=(original,values)
        delivery[name]=compare_delivery(original,values)
        if replay is not None:
            second, repeated=load(replay/name)
            assert second['gates']==original['gates']
            assert original['cut']==second['cut']
            same[name]={key:bool(np.array_equal(values[key],repeated[key])) for key in values}
            assert all(same[name].values()), same[name]
    assert len(records)==23
    cuts={f'{seed}_{name}':compare_cut(records,f'{seed}_{name}','steer')
          for seed in (271,362) for name in NAMES[2:]}
    cuts.update(oDN1=compare_cut(records,'271_cruise','speed'),
                all=compare_cut(records,'271_a02_left','all'))
    speeds={}
    for seed in (271,362):
        low=records[f'{seed}_cruise_open'][0]['active_speed_mm_s']
        high=records[f'{seed}_fast_open'][0]['active_speed_mm_s']
        speeds[str(seed)]=dict(cruise=low,fast=high,pass_=bool(low>0 and high>=1.25*low))
    behaviors={k:r['pass_'] for k,(r,_) in records.items()}
    write(out,dict(behavior_pass=behaviors,
          behaviors={k:{key:r[key] for key in r if key not in ('ports','dn_ports')} for k,(r,_) in records.items()},
          delivery=delivery,replays=same,speed_pair=speeds,cuts=cuts,
          actual_native_steps=17000*(len(records)+len(same)),
          pass_=all(behaviors.values()) and all(c['proper_body_effect'] for c in cuts.values())
                and all(v['pass_'] for v in speeds.values())))
    print('behavior',sum(behaviors.values()),'/',len(behaviors),'speed',speeds,'cuts',
          {k:v['proper_body_effect'] for k,v in cuts.items()},'replays',len(same))
    for item in ('a01','a02'):
        for side in ('left','right'):
            for seed in (271,362):
                name=f'{seed}_{item}_{side}'
                print(name,cuts[name]['yaw_open'],cuts[name]['yaw_cut'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('formal',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args()
    audit(args.formal,args.replay,args.out)
