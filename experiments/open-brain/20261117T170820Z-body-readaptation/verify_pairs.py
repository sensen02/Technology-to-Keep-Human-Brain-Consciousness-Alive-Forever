"""Check matched source spikes, turn/stride changes and continued drive during halts."""
import json
from pathlib import Path
import numpy as np
from context import HERE,NECK,first,hash_file
from fixed_neck import full_catalog


def load(folder):
    record=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'TRACE.npz',allow_pickle=False) as a:
        return record,dict(a)


def main():
    catalog=full_catalog()['ports']
    pairs={}
    for kind in ('ground','eon','extra'):
        if kind=='ground':
            keys=[f'{target}__{variant}' for target in ('stride_left','stride_right','bluebell','web_b')
                   for variant in ('held0','held1')]
        elif kind=='eon':
            keys=[f'{seed}_{target}' for seed in (271,362) for target in ('a01_left','a01_right','a02_left','a02_right')]
        else:
            keys=[f'extra_{target}' for target in ('stride_left','stride_right','bluebell','web_b')]
        for key in keys:
            if kind=='extra':
                opened=HERE/'EXTRA'/(key+'_open')
                operated=HERE/'EXTRA'/(key+'_cut')
            else:
                opened=HERE/'FORMAL'/f'{kind}_{key}__open'
                operated=HERE/'FORMAL'/f'{kind}_{key}__{"steer" if kind=="eon" else "cut"}'
            normal,a=load(opened)
            cut,b=load(operated)
            order=normal['ports']
            if kind=='extra':
                target=normal['task']['family']
                group={'stride_left':'walking_outer_stride','stride_right':'walking_outer_stride',
                       'bluebell':'halt_bluebell','web_b':'halt_web_b'}[target]
                side=target.rsplit('_',1)[1] if target.startswith('stride') else None
                affected=[i for i,r in enumerate(order) if catalog[r]['group']==group and
                          (side is None or catalog[r]['side']==side)]
            else:
                affected=[order.index(root) for root in cut.get('cut_roots',cut.get('cut',[]))]
            dn=[i for i,r in enumerate(order) if catalog[r]['direction']=='DN']
            spared=[i for i in dn if i not in affected]
            assert affected
            same=np.array_equal(a['emitted'][:,dn],b['emitted'][:,dn]) and np.array_equal(a['arrived'][:,spared],b['arrived'][:,spared])
            checks=dict(source_and_spared_identical=bool(same),cut_arrival_zero=not b['arrived'][:,affected].any(),
                        source_active=bool(a['emitted'][:,affected].any()))
            if kind=='eon':
                checks['turn_removed']=abs(cut['yaw_degrees'])<=12 and abs(cut['yaw_degrees'])<=.4*abs(normal['yaw_degrees'])
            elif 'stride' in key:
                checks['yaw_change']=normal['signed_yaw_deg']-cut['signed_yaw_deg']>=12
                checks['stride_change']=normal['outer_minus_inner_mm']-cut['outer_minus_inner_mm']>=.025
                checks['cut_heading']=abs(cut['yaw_deg'])<=12
            else:
                case=normal.get('case',normal.get('task'))
                block=slice(case['action']+2600,case['action']+5400)
                advance=[i for i,r in enumerate(order) if catalog[r]['group']=='forward']
                checks['drive_continues']=bool(a['arrived'][block,advance].any())
                checks['brake_arrives']=bool(a['arrived'][block,affected].any())
            pairs[kind+'_'+key]=dict(pass_=all(checks.values()),checks=checks,
                emitted=int(a['emitted'][:,affected].sum()),intact_arrived=int(a['arrived'][:,affected].sum()))
    first.write(HERE/'PAIR_AUDIT.json',dict(pass_=all(x['pass_'] for x in pairs.values()),pairs=pairs))
    print('Paired causal checks',sum(x['pass_'] for x in pairs.values()),'of',len(pairs),flush=True)


if __name__=='__main__':main()
