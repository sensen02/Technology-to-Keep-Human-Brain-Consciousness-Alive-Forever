"""Audit saved body results, per-port physical arrival and independent replay."""
import json
from pathlib import Path

import numpy as np

from fixed_neck import HERE, PARENT, full_catalog, hash_file, parameters, write


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    result=read(path/'RESULT.json')
    runtime=read(path/'RUNTIME.json')
    trace=path/'TRACE.npz'
    assert runtime['original_result_sha256']==hash_file(path/'RESULT.json')
    assert result['trace_sha256']==hash_file(trace)
    assert runtime['parameters_sha256']==hash_file(HERE/'PARAMETERS.json')
    assert runtime['runtime_sha256']==hash_file(HERE/'fixed_neck.py')
    assert runtime['readout_sha256']==hash_file(HERE/'READOUT.json')
    with np.load(trace,allow_pickle=False) as saved:
        data=dict(saved)
    return result,data,runtime


def expected_arrival(emitted,order,result):
    value=np.zeros_like(emitted)
    n=len(emitted)
    explicit=set(result.get('cut_roots',result.get('cut',[])))
    case=result['case']
    catalog=full_catalog()['ports']
    for col,root in enumerate(order):
        delay=parameters()['ports'][root]['delay_ticks']
        if root in explicit:
            continue
        value[delay:,col]=emitted[:-delay,col]
        if case.get('cut_end') and catalog[root]['direction']=='DN':
            lo=case['cut_start']
            hi=min(n,case['cut_end']+delay)
            value[lo:hi,col]=False
    return value


def main():
    manifest=read(HERE/'IDENTITY.json')
    assert all(hash_file(Path(name))==digest for name,digest in manifest['sha256'].items())
    expected=[line.split() for line in (HERE/'FORMAL_JOBS.txt').read_text().splitlines()]
    results={}
    total_steps=0
    for suite,name in expected:
        folder=f'{suite}_{name}'
        source,first,identity=arrays(HERE/'FORMAL'/folder)
        replay,second,ridentity=arrays(HERE/'REPLAY'/folder)
        assert (HERE/'FORMAL'/f'{folder}.exit').read_text().strip()=='0'
        assert (HERE/'REPLAY'/f'{folder}.exit').read_text().strip()=='0'
        assert source['gates']==replay['gates'] and identity==ridentity
        equals={key:bool(np.array_equal(value,second[key])) for key,value in first.items()}
        assert all(equals.values()),(folder,equals)
        order=source.get('arrival_roots',source.get('ports'))
        expect=expected_arrival(first['emitted'],order,source)
        assert np.array_equal(expect,first['arrived']),folder
        total_steps+=2*len(first['trace'])
        results[folder]=dict(pass_=source['pass_'],gates=source['gates'],
                            source=source, replay_exact=equals,arrival_exact=True,source_order=order)
    # Each dedicated halt, turn, and Eon route retains the corresponding cut control.
    pairs={}
    catalog=full_catalog()['ports']
    for suite,name in expected:
        if suite=='ground' and name.endswith('__open'):
            key=name.removesuffix('__open')
            opened=HERE/'FORMAL'/f'ground_{name}'
            operated=HERE/'FORMAL'/f'ground_{key}__cut'
        elif suite=='eon' and name.endswith('__steer'):
            key=name.removesuffix('__steer')
            opened=HERE/'FORMAL'/f'eon_{key}__open'
            operated=HERE/'FORMAL'/f'eon_{name}'
        else:
            continue
        source,a,_=arrays(opened)
        cut,b,_=arrays(operated)
        order=cut['ports']
        affected=cut.get('cut_roots',cut.get('cut',[]))
        cols=[order.index(root) for root in affected]
        dn=[order.index(root) for root in order if catalog[root]['direction']=='DN']
        spare=[i for i in dn if i not in cols]
        same=bool(np.array_equal(a['emitted'][:,dn],b['emitted'][:,dn])
                  and np.array_equal(a['arrived'][:,spare],b['arrived'][:,spare]))
        assert same and not b['arrived'][:,cols].any()
        item=dict(same_dn_emitted_and_spared_arrival=same,source_sent=int(a['emitted'][:,cols].sum()),
                  open_arrivals=int(a['arrived'][:,cols].sum()),cut_arrivals=int(b['arrived'][:,cols].sum()))
        if suite=='ground' and 'stride_' in key:
            item['yaw_change_deg']=source['signed_yaw_deg']-cut['signed_yaw_deg']
            item['stride_change_mm']=source['outer_minus_inner_mm']-cut['outer_minus_inner_mm']
            item['pass_']=item['yaw_change_deg']>=12 and item['stride_change_mm']>=.025
        elif suite=='ground':
            case=source['case']
            block=slice(case['action']+2600,case['action']+5400)
            fwd=[i for i,root in enumerate(order) if catalog[root]['group']=='forward']
            item['forward_arrival_while_standing']=int(a['arrived'][block,fwd].sum())
            item['brake_arrival_while_standing']=int(a['arrived'][block,cols].sum())
            item['pass_']=min(item['forward_arrival_while_standing'],item['brake_arrival_while_standing'])>0
        else:
            item['pass_']=bool(abs(cut['yaw_degrees'])<=12 and
                               abs(cut['yaw_degrees'])<=.4*abs(source['yaw_degrees']))
        pairs[f'{suite}_{key}']=item
    speeds={}
    for suite,keys in (('body',('0','1')),('eon',('271','362'))):
        for key in keys:
            get=lambda speed:results[f'{suite}_held_{speed}_{key}']['source'] if suite=='body' else results[f'eon_{key}_{speed}__open']['source']
            low=get('cruise')['active_speed_mm_s']
            high=get('fast')['active_speed_mm_s']
            speeds[suite+'_'+key]=dict(low=low,high=high,pass_=bool(high>=1.25*low))
    write(HERE/'FINAL.json',dict(pass_=all(r['pass_'] for r in results.values()) and
             all(p['pass_'] for p in pairs.values()) and all(v['pass_'] for v in speeds.values()),
             cases=results,pairs=pairs,speeds=speeds,independent_replay_count=len(results),
             raw_physical_steps=total_steps,input_identity_count=len(manifest['sha256'])))
    print('body cases',sum(r['pass_'] for r in results.values()),'of',len(results),
          'pairs',sum(v['pass_'] for v in pairs.values()),'of',len(pairs),'speeds',speeds)
    for name,r in results.items():
        if not r['pass_']:
            print('FAILED',name,[k for k,v in r['gates'].items() if not v])


if __name__=='__main__':
    main()
