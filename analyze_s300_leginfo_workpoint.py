"""S300 analysis: leg-axis three-level decay chain, feasibility, arm selection (read-only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
LEGS=('lf','lm','lh','rf','rm','rh'); ARMS=('s_1.00','s_0.50','s_0.25','s_0.10','s_0.05','s_0.02')
GROUPS=Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json')
POOLS=Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_11_mn_pools.json')
def disp_leg(M):  # M: samples x 6 legs
    cm=M.mean(1,keepdims=True); R=M-cm
    d=R.std(0)/np.abs(M.mean(0)+1e-30)
    return float(d.mean()), {LEGS[i]: float(d[i]) for i in range(6)}
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    g=json.loads(GROUPS.read_text())['legs']; pool=json.loads(POOLS.read_text())
    res={}
    for arm in ARMS:
        d=out/arm
        sp=np.load(d/'SPIKES.npz',allow_pickle=True); shape=sp['shape']; packed=sp['packed']
        n=int(shape[1]); counts=sp['spike_counts']
        spikes=np.unpackbits(packed,axis=1)[:,:n].astype(bool) if packed.shape[0]==int(shape[0]) else np.unpackbits(packed,axis=1)[:,:n].astype(bool)
        ne=np.load(d/'NEURAL.npz',allow_pickle=True); rates=np.asarray(ne['rates_hz'],float); V=np.asarray(ne['decoder_motor_gains'],float); nz=np.flatnonzero(V!=0)
        post=np.asarray(ne['decoder_postclip'],float)
        duty=float(spikes.mean()); dt=.0005
        # input: per-leg mean of its 5 subtype channels
        sci=json.loads((d/'SENSORY_CHANNEL_INDEX.json').read_text()); groups=sci['groups']
        inp=np.zeros((400,6))
        sc=np.asarray(ne['sensory_current'],float)
        for j,leg in enumerate(LEGS):
            allidx=[int(x) for k,v in groups.items() if k.startswith(leg+':') for x in v]
            inp[:,j]=sc[:,allidx].mean(axis=1) if allidx else 0.0
        # pool: per-leg mean per-neuron rate
        po=np.load(d/'POOL_NEURONS.npz',allow_pickle=True); _ids=np.asarray(po['pool_root_ids']); _loc=np.asarray(po['local_index']); _keep=np.flatnonzero(_loc>=0); ids=_ids[_keep]; loc=_loc[_keep]
        poolleg={leg:[] for leg in LEGS}
        for r in pool['measured']['resolved']:
            for pid in r['pool_root_ids']:
                k=np.flatnonzero(ids==int(pid))
                if k.size==0: continue
                poolleg[r['legs'] if 'legs' in r else None] if False else None
                poolleg.setdefault(r.get('leg','').lower(), []).append(int(loc[int(k[0])]))
        # fall back: derive leg from actuator prefix
        poolleg={leg:[] for leg in LEGS}
        for r in pool['measured']['resolved']:
            leg=r['actuator'][:2].lower()
            if leg not in poolleg: continue
            for pid in r['pool_root_ids']:
                k=np.flatnonzero(ids==int(pid))
                if k.size: poolleg[leg].append(int(loc[int(k[0])]))
        poolrate=np.zeros((800,6))
        for j,leg in enumerate(LEGS):
            idx=sorted(set(poolleg[leg]))
            poolrate[:,j]=spikes[:,idx].mean(1) if idx else 0.0
        poolrate_ms=poolrate.reshape(400,2,6).mean(1)
        outp=np.stack([np.asarray(ne['decoder_postclip'],float)[:,g[leg]['minus']].mean(1) for leg in LEGS],1)
        m=(np.arange(1,401)>=20)&(np.arange(1,401)<400)
        D_in,din=disp_leg(inp[m]); D_pool,dp=disp_leg(poolrate_ms[m]); D_out,dout=disp_leg(outp[m])
        rmean=rates[:,nz].mean(0)
        D_time=float(np.median(rates[:,nz].std(0)/np.maximum(rates[:,nz].mean(0),1e-30)))
        CI=float(D_out/D_in) if D_in else None
        res[arm]=dict(duty=duty,post1_nz=float((post[:,nz]==1).mean()),post1_90=float((post==1).mean()),
                      D_in_leg=D_in,D_pool_leg=D_pool,D_out_leg=D_out,CI=D_out/D_in if D_in else None,D_time=D_time,
                      per_leg=dict(input=din,pool=dp,output=dout),rate_mean_range=[float(rmean.min()),float(rmean.max())],
                      firing=bool(spikes.any()),
                      feasible=bool(0.05<=duty<=0.50 and float((post[:,nz]==1).mean())<=0.30 and spikes.any()))
    json.dump(res,open(out/'LEG_AXIS_CHAIN.json','w'),indent=2)
    feas={k:v for k,v in res.items() if v['feasible']}
    sel=max(feas,key=lambda k:(feas[k]['D_out_leg'],feas[k]['D_time'])) if feas else None
    json.dump(dict(arms=res,feasible=list(feas),selected=sel),open(out/'SELECTION.json','w'),indent=2)
    # loss location
    base=res['s_1.00']; ratios=dict(in_to_pool=base['D_pool_leg']/base['D_in_leg'],pool_to_out=base['D_out_leg']/base['D_pool_leg'])
    worst=None
    legs=base['per_leg']
    for a in LEGS:
        for b in LEGS:
            if a<b:
                for lvl in ('input','pool','output'):
                    pass
    if ratios['in_to_pool']<0.2 and ratios['pool_to_out']<0.5: where='both levels collapse'
    elif ratios['in_to_pool']<0.2: where='network'
    elif ratios['pool_to_out']<0.5: where='pool readout'
    else: where='neither (no clear collapse)'
    json.dump(dict(arm='s_1.00',D_in_leg=base['D_in_leg'],D_pool_leg=base['D_pool_leg'],D_out_leg=base['D_out_leg'],
                   ratios=ratios,where=where,per_leg=legs,
                   note='leg-axis definition: per leg input=mean of its 5 subtype channels, pool=mean per-neuron rate of that leg resolved pool neurons (202 of 208 in tier), output=that leg minus group mean; common mode removed; std_leg/|mean_leg| averaged over 20-400 ms'),
              open(out/'LOSS_LOCATION.json','w'),indent=2)
    print(json.dumps(dict(arms={k:dict(duty=round(v['duty'],5),D_in=round(v['D_in_leg'],4),D_pool=round(v['D_pool_leg'],4),D_out=round(v['D_out_leg'],4),post1=round(v['post1_nz'],4),feas=v['feasible']) for k,v in res.items()},selected=sel,where=where,ratios=ratios),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
