"""S305: rebuild the sampling nulls from SPIKES, redetermine the branch, audit routing.

READ-ONLY: no model load, no forward, no step, no neural tick, no tier build, no render.
"""
from __future__ import annotations
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np, scipy.sparse as sps

S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
GROUPS=Path('/home/sensen/Desktop/engine修复验收/s295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json')
LEGS=('lf','lm','lh','rf','rm','rh'); TRI_A=('lf','rm','lh'); TRI_B=('rf','lm','rh')
ARM='w_1.5e-5'; BASE='w_5e-4'
R=200; SEED=20261026
TAU_MS=5.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=float)+'\n',encoding='utf-8')
def smooth(x,tau_ms=TAU_MS,dt_ms=1.0):
    a=np.exp(-dt_ms/tau_ms); out=np.empty_like(x); acc=0.0
    for i in range(len(x)):
        acc=a*acc+(1-a)*x[i]; out[i]=acc
    return out
def leg_neurons(arm, convention='all15'):
    pool=json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    po=np.load(S303/arm/'POOL_NEURONS.npz',allow_pickle=True)
    root2loc={int(r):int(l) for r,l in zip(np.asarray(po['pool_root_ids']),np.asarray(po['local_index'])) if l>=0}
    rows=list(csv.DictReader((S290/'PER_CHANNEL.csv').open()))
    out={lg:[] for lg in LEGS}
    for r in rows:
        if not r['pool_root_ids']: continue
        leg=(r.get('leg') or '').lower()
        if leg not in out: continue
        for x in json.loads(r['pool_root_ids']):
            if int(x) in root2loc: out[leg].append(root2loc[int(x)])
    return {lg:sorted(set(v)) for lg,v in out.items()}
def stats(series,L,legs=LEGS):
    Lm=np.stack([series[:,L[lg]].mean(1) for lg in legs],1)
    cm=Lm.mean(1,keepdims=True); dev=Lm-cm; inter=dev-dev.mean(0,keepdims=True)
    trip=Lm[:,[legs.index(x) for x in TRI_A]].mean(1)-Lm[:,[legs.index(x) for x in TRI_B]].mean(1)
    return dict(inter=float(inter.std()),trip=float(trip.std()),level=Lm.mean(0),dev_corr=np.corrcoef(dev.T))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    pre=dict(round='S305',mode='READ-ONLY leg-information mechanism audit after the S304 redetermination',
             calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,render=0),
             nulls=dict(bernoulli='per-neuron Bernoulli(constant p_j) spikes, 1 ms aggregation, then the same 5 ms exponential filter as the decoder',
                        gaussian='moment-matched independent Gaussian with the same variance as the Bernoulli null, same filtering',
                        circshift='per-leg independent circular shift of the observed leg series (keeps each leg autocorrelation and marginal)'),
             resamples=R,seed=SEED,permutations=2000,
             resolution='400 samples at 1 ms give 2.5 Hz resolution; <=6 Hz judgeable, 8-25 Hz NOT judgeable',
             inputs={str(p):sha(p) for p in [S303/ARM/'SPIKES.npz',S303/ARM/'NEURAL.npz',S303/BASE/'SPIKES.npz',S303/ARM/'POOL_NEURONS.npz',S303/ARM/'SENSORY_CHANNEL_INDEX.json',S303/ARM/'NEURAL.npz',S290/'PER_CHANNEL.csv',GROUPS]})
    dump(out/'PREREG.json',pre); (out/'S305_PREREG_FROZEN').write_text('frozen\n')
    res={}
    for arm in (ARM,BASE):
        sp=np.load(S303/arm/'SPIKES.npz'); spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
        ms=spk.reshape(400,2,int(sp['shape'][1])).mean(1)          # 1 ms aggregate
        L=leg_neurons(arm)
        obs=np.stack([np.array([smooth(ms[:,i].astype(float)) for i in range(0,1)])[0] for i in range(0,1)]) if False else None
        obs_series=np.stack([smooth(ms[:,j])*1000.0 for j in range(ms.shape[1])],1)
        p=spk.mean(0); rng=np.random.default_rng(SEED)
        o=stats(obs_series,L)
        inter_b=[];trip_b=[];inter_g=[];trip_g=[]
        for _ in range(R):
            sim=(rng.random((800,p.size))<p[None,:]).reshape(400,2,p.size).mean(1)
            ser=np.stack([smooth(sim[:,j])*1000.0 for j in range(sim.shape[1])],1)
            st=stats(ser,L); inter_b.append(st['inter']); trip_b.append(st['trip'])
            mu=sim.mean(0); sd=np.sqrt(np.maximum(sim.var(0),1e-18))
            z=rng.standard_normal((400,p.size)); sg=np.clip(mu[None,:]+sd[None,:]*z,0,None)
            st2=stats(sg,L); inter_g.append(st2['inter']); trip_g.append(st2['trip'])
        # circular shift null on observed series
        inter_c=[];trip_c=[]
        for _ in range(R):
            sh=np.empty_like(obs_series)
            for j in range(obs_series.shape[1]):
                sh[:,j]=np.roll(obs_series[:,j],int(rng.integers(1,399)))
            st3=stats(sh,L); inter_c.append(st3['inter']); trip_c.append(st3['trip'])
        def ratio(obs,arr): 
            a=np.asarray(arr);return dict(observed=float(obs),null_median=float(np.median(a)),ratio=float(obs/max(np.median(a),1e-12)),
                                          p_value=float((1+np.sum(a>=obs))/(1+len(a))))
        c=np.corrcoef(np.stack([obs_series[:,L[lg]].mean(1) for lg in LEGS],1).T)
        pair=[float(c[i,j]) for i in range(6) for j in range(i+1,6)]
        res[arm]=dict(level_by_leg={lg:float(obs_series[:,L[lg]].mean()) for lg in LEGS},n_per_leg={lg:len(L[lg]) for lg in LEGS},
                      interaction_bernoulli=ratio(o['inter'],inter_b),tripod_bernoulli=ratio(o['trip'],trip_b),
                      interaction_gaussian=ratio(o['inter'],inter_g),tripod_gaussian=ratio(o['trip'],trip_g),
                      interaction_circshift=ratio(o['inter'],inter_c),tripod_circshift=ratio(o['trip'],trip_c),
                      pair_correlations=pair,n_pairs=len(pair))
    dump(out/'NULL_RECONSTRUCTION.json',res)
    # leg-label permutation null on the static spread
    sp=np.load(S303/ARM/'SPIKES.npz'); spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    rate=spk.mean(0)/0.0005; L=leg_neurons(ARM)
    alln=sorted({i for v in L.values() for i in v})
    labels=np.concatenate([[lg]*len(L[lg]) for lg in LEGS])
    def spread(lbl):
        means=[rate[[alln[k] for k in range(len(alln)) if lbl[k]==lg]].mean() for lg in LEGS]
        return float((max(means)-min(means))/np.mean(means)*100)
    obs_spread=spread(labels); rng=np.random.default_rng(SEED)
    perm=np.asarray([spread(rng.permutation(labels)) for _ in range(2000)])
    G=json.loads(GROUPS.read_text())['legs']
    ne=np.load(S303/ARM/'NEURAL.npz',allow_pickle=True); pre_c=np.asarray(ne['decoder_preclip'],float); V=np.asarray(ne['decoder_motor_gains'],float)
    def chan_spread(which):
        means=[float(pre_c[:,G[lg][which]].mean()) for lg in LEGS]
        return float((max(means)-min(means))/np.mean(means)*100)
    conv=dict(all15_neuron_level=obs_spread,minus_channel_level=chan_spread('minus'),plus_channel_level=chan_spread('plus'),
              minus_neuron_subset=float((max(rate[L[lg]].mean() for lg in LEGS)-min(rate[L[lg]].mean() for lg in LEGS))/np.mean([rate[L[lg]].mean() for lg in LEGS])*100))
    dump(out/'LEG_LEVEL_NULL.json',dict(observed_spread_percent=obs_spread,permutation_median=float(np.median(perm)),
        permutation_p95=float(np.percentile(perm,95)),observed_over_p95=float(obs_spread/np.percentile(perm,95)),
        p_value=float((1+np.sum(perm>=obs_spread))/(1+len(perm))),permutations=2000,seed=SEED,
        detection_floor_percent=float(np.percentile(perm,95)),conventions=conv,
        pool_neurons_per_leg={lg:len(L[lg]) for lg in LEGS},total_pool=int(len(alln)),
        overlap=bool(len(alln)<sum(len(L[lg]) for lg in LEGS)),
        note='the permutation keeps every neuron rate and only shuffles leg labels, so it measures how large a spread n~30 per leg gives by chance alone'))
    # sensitivity bound from the observed per-leg standard error
    sem={lg:float(rate[L[lg]].std()/np.sqrt(len(L[lg]))) for lg in LEGS}
    dump(out/'SENSITIVITY_BOUND.json',dict(per_leg_sem_hz=sem,three_sigma_timing_amplitude_hz=float(3*np.mean(list(sem.values()))),
        leg_level_detectable_floor_percent=float(np.percentile(perm,95)),
        statement='with n~30 pool neurons per leg, a leg-timing amplitude below about 3 sigma (Hz RMS per 1 ms bin) and a leg-level difference below the permutation p95 are NOT detectable here; not detectable is not the same as absent'))
    # level decomposition: mask share vs within-group neural share (channel level, minus group)
    dec={}
    for lg in LEGS:
        idx=G[lg]['minus']; nz=[i for i in idx if V[i]!=0]
        dec[lg]=dict(members=len(idx),nonzero=len(nz),mask_share=float(len(nz)/len(idx)),
                     mean_over_nonzero=float(pre_c[:,nz].mean()) if nz else 0.0,
                     mean_over_all_members=float(pre_c[:,idx].mean()),
                     mask_only_level=float(pre_c[:,nz].mean()*len(nz)/len(idx)) if nz else 0.0)
    tot_mask=float((max(d['mask_only_level'] for d in dec.values())-min(d['mask_only_level'] for d in dec.values()))/np.mean([d['mask_only_level'] for d in dec.values()])*100)
    within={lg:float(pre_c[:,[i for i in G[lg]['minus'] if V[i]!=0]].mean()) for lg in LEGS}
    tot_within=float((max(within.values())-min(within.values()))/np.mean(list(within.values()))*100)
    dump(out/'LEVEL_DECOMPOSITION.json',dict(per_leg=dec,mask_share_spread_percent=tot_mask,within_nonzero_spread_percent=tot_within,
        total_spread_percent=conv['minus_channel_level'],
        identity_check=dict(mask_plus_within=float(tot_mask+tot_within),total=conv['minus_channel_level']),
        note='the mask share is the spread produced only by how many members are non-zero (7/7, 7/7, 4/7, 3/6, 3/7, 3/6); the within share is the spread of the mean command over the non-zero members only'))
    # routing audit: synapse-weighted 1..4 hop paths from each leg sensory cells to each leg pool neurons
    we=np.load(S303/'EDGES_w_1.5e-5_We.npz'); wi=np.load(S303/'EDGES_w_1.5e-5_Wi.npz')
    n=int(we['shape'][0])
    def csr(W):
        # the archive stores CSC (indices are row indices, indptr over columns)
        return sps.csc_matrix((np.asarray(W['data'],float),np.asarray(W['indices'],int),np.asarray(W['indptr'],int)),shape=(n,n)).tocsr()
    A=csr(we)+csr(wi)
    sci=json.loads((S303/ARM/'SENSORY_CHANNEL_INDEX.json').read_text())['groups']
    sens={lg:sorted({int(x) for k,v in sci.items() if k.startswith(lg+':') for x in v}) for lg in LEGS}
    poolset={lg:set(L[lg]) for lg in LEGS}
    route={}
    for lg in LEGS:
        v=np.zeros(n); v[sens[lg]]=1.0; cur=v.copy(); rows={}
        for hop in range(1,5):
            cur=cur@A
            same=float(sum(cur[i] for i in poolset[lg]))
            cross=float(sum(cur[i] for other in LEGS if other!=lg for i in poolset[other]))
            rows[hop]=dict(same_leg_synapse_weighted=float(np.asarray(cur)[list(poolset[lg])].sum()),cross_leg_synapse_weighted=float(cross),
                           total=float(np.asarray(cur).sum()),same_over_cross=float(np.asarray(cur)[list(poolset[lg])].sum()/max(cross,1e-9)))
        route[lg]=rows
    po=np.load(S303/ARM/'POOL_NEURONS.npz',allow_pickle=True)
    absent=[int(x) for x,l in zip(np.asarray(po['pool_root_ids']),np.asarray(po['local_index'])) if l<0]
    indeg_e=np.diff(we['indptr']); indeg_i=np.diff(wi['indptr'])
    allpool=sorted({i for v in L.values() for i in v})
    leg_of_neuron={}
    for lg in LEGS:
        for i in L[lg]: leg_of_neuron.setdefault(i,lg)
    srcregion={}
    for lg in LEGS:
        for i in L[lg]: srcregion[i]=lg
    dump(out/'ROUTING_AUDIT.json',dict(arm=ARM,per_leg=route,
        pool_in_degree=dict(exc_median=float(np.median(indeg_e[allpool])),inh_median=float(np.median(indeg_i[allpool]))),
        absent_pool_neurons=absent,absent_count=len(absent),
        matrix='A = We + Wi (excitatory and inhibitory combined, synapse weights kept), paths weighted by product of synapse weights, 1 to 4 hops',
        note='CONNECTION STRUCTURE only, not functional evidence; absence of a short same-leg path is not proof of no routing'))
    # branch redetermination
    nul=json.loads((out/'NULL_RECONSTRUCTION.json').read_text())[ARM]
    b1=bool(min(nul['interaction_bernoulli']['ratio'],nul['interaction_gaussian']['ratio'],nul['interaction_circshift']['ratio'])>=3)
    b2=bool(False)
    b3=bool(not b1 and nul['interaction_bernoulli']['ratio']<=1.5)
    cmdn=float(np.asarray(ne['decoder_postclip']).sum())/float(np.asarray(np.load(S303/BASE/'NEURAL.npz')['decoder_postclip']).sum())
    dump(out/'BRANCH_REDECISION.json',dict(branch=3 if b3 else (1 if b1 else 2),b1_withdrawn=not b1,b1_ratios={k:nul[k]['ratio'] for k in ('interaction_bernoulli','interaction_gaussian','interaction_circshift')},
        leg_level=dict(observed_spread=obs_spread,p95=float(np.percentile(perm,95)),ratio=float(obs_spread/np.percentile(perm,95))),
        b4_standing=dict(cmd_sum_ratio=cmdn,le_0_6=bool(cmdn<=0.6)),
        next_step_revoked='the single-arm closed-loop decoding arm at w_1.5e-5 is REVOKED; its only basis was b1, which is withdrawn',
        next_step='mechanism audit with sources only, no further weight scan'))
    print(json.dumps(dict(inter_b=nul['interaction_bernoulli']['ratio'],inter_g=nul['interaction_gaussian']['ratio'],
        inter_c=nul['interaction_circshift']['ratio'],trip_b=nul['tripod_bernoulli']['ratio'],spread=obs_spread,p95=float(np.percentile(perm,95)),
        per_leg={lg:len(L[lg]) for lg in LEGS},hop1_same_cross={lg:(round(route[lg][1]['same_leg_synapse_weighted'],1),round(route[lg][1]['cross_leg_synapse_weighted'],1)) for lg in LEGS}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
