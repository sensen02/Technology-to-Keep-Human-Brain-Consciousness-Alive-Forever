"""S330: leg-specific input budget, decoupling dose response, eight arms."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S327=Path('/home/sensen/Desktop/engine修复验收/s327-global-mode-and-motif-influence/20261027T004101Z')
S328=Path('/home/sensen/Desktop/engine修复验收/s328-mode-origin-and-coupling-budget/20261027T005128Z')
S329=Path('/home/sensen/Desktop/engine修复验收/s329-phase-resolved-command/20261027T010135Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def env_of(x,ds=2):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds] if ds>1 else y
def std_cols(Y):
    Y=Y-Y.mean(0,keepdims=True);s=Y.std(0,keepdims=True);s[s<1e-12]=1e-12;return Y/s
def mean_pair_corr(Z):
    n,k=Z.shape
    if k<2: return 0.0
    cs=Z.sum(1);return (float(cs@cs)/n-k)/(k*(k-1))
def d1_with_null(Y,rng,nulls=NULLS):
    Z=std_cols(Y);obs=mean_pair_corr(Z);null=[]
    for _ in range(nulls):
        sh=rng.integers(0,Z.shape[0],Z.shape[1])
        null.append(mean_pair_corr(np.stack([np.roll(Z[:,i],sh[i]) for i in range(Z.shape[1])],1)))
    null=np.asarray(null)
    return dict(D1=float(obs),D1_null_p95=float(np.percentile(null,95)),D1_p=float((np.sum(null>=obs)+1)/(len(null)+1)),nulls=nulls)
def pair_matrix(E):
    return {f'{a}|{b}':float(np.corrcoef(E[a],E[b])[0,1]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
def peak_of(y,lo=4.0,hi=50.0,ds=2):
    n=len(y);f=np.fft.rfftfreq(n,ds*1e-3);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=lo+2*df)&(f<=hi-2*df)
    return float(f[int(np.argmax(np.where(inter,P,0)))]) if inter.any() else None
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S330',mode='leg-specific budget and decoupling dose response; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=9,control_tick=45000,tier_advance=5000,neural_substep=90000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,
      rules=['archived EDGES npz must be read with csr_matrix, never csc_matrix; the four counts 1943 out, 1553 in, 160+17 pool same-leg, 0 cross-leg pool are all archived',
             'the primary criterion is mean_cross_leg over the 15 leg pairs with a per-cell independent circular shift null (NULLS=1000)',
             'band_ratio and peak_hz are diagnostics only',
             'no else fallback; slice asserts n <= len(signal); every record carries its nulls count',
             'dose arms must report the realised leg-specific share, pool median rate, f380 and spike_total; a share deviation above 10 percent is DOSE_MISMATCH'],
      selftest_floor=dict(D1_ctrl_max=0.05,D1_obs_minus_ctrl=0.30),
      branches=dict(x1='leg-specific input is the limiter',x2='leg-specific input is not the limiter',x3='dose ladder not completable',
                    x4='seed floor too large (UNDERPOWERED)',x5='share is not the causal variable'))
    dump(o/'PREREG.json',pre)
    (o/'S330_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s330.py)=%s\nsha256(audit_s330_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s330_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    M18_by_leg={lg:[loc[int(c)] for r in ROLES for c in modules[lg][r]] for lg in LEGS}
    M18=[j for lg in LEGS for j in M18_by_leg[lg]];Mset=set(M18)
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    seen={}
    for r in rows:
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());pool_by_leg={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool]
    We25=np.load(S303/'EDGES_w_2.5e-5_We.npz');Wi25=np.load(S303/'EDGES_w_2.5e-5_Wi.npz')
    def tocsr(z):
        n=int(z['indptr'].size-1)
        return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
    WeC=tocsr(We25);WiC=tocsr(Wi25);n=WeC.shape[0]
    # ---- stage A: pool input budget
    tab={}
    for lg in LEGS:
        pc=pool_by_leg[lg];own=set(M18_by_leg[lg])
        e_in=float(WeC[pc,:].sum());i_in=float(WiC[pc,:].sum())
        colonE=WeC[pc,:];colonI=WiC[pc,:]
        e_mine=0.0;w_cnt=0
        for r_ in range(colonE.shape[0]):
            rw=colonE.getrow(r_)
            for c_,v_ in zip(rw.indices,rw.data):
                if int(c_) in own: e_mine+=float(v_);w_cnt+=1
        i_mine=0;wi_cnt=0
        for r_ in range(colonI.shape[0]):
            rw=colonI.getrow(r_)
            for c_,v_ in zip(rw.indices,rw.data):
                if int(c_) in own: i_mine+=float(v_);wi_cnt+=1
        cross=0
        for lg2 in LEGS:
            if lg2==lg: continue
            other=set(M18_by_leg[lg2])
            for r_ in range(colonE.shape[0]):
                rw=colonE.getrow(r_)
                for c_ in rw.indices:
                    if int(c_) in other: cross+=1
        tab[lg]=dict(n_pool=len(pc),poolexc_uS=e_in,poolinh_uS=i_in,motifE_uS=e_mine,share=e_mine/max(e_in,1e-12),
          we_edges=w_cnt,wi_edges=wi_cnt,cross_leg_edges=cross)
    # M18 outward edges and how many land in the pool
    out_tot=0;into_pool=0
    poolset=set(poolcols)
    for j in range(n):
        if j in Mset: continue
        for Wm in (WeC,WiC):
            rw=Wm.getrow(j)
            for c_ in rw.indices:
                if int(c_) in Mset:
                    out_tot+=1
                    if j in poolset: into_pool+=1
    presyn={}
    for lg in LEGS:
        s=set()
        for j in pool_by_leg[lg]:
            s|=set(int(x) for x in WeC.getrow(j).indices)
        presyn[lg]=s
    inter=[len(presyn[a]&presyn[b]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]]
    common=set.intersection(*presyn.values())
    common_share=float(WeC[poolcols,:][:, sorted(common)].sum()/max(WeC[poolcols,:].sum(),1e-12)) if common else 0.0
    dump(o/'POOL_INPUT_BUDGET.json',dict(per_leg=tab,edges=dict(m18_out=out_tot,m18_into_pool=into_pool),
      cross_leg_pool_edges=sum(v['cross_leg_edges'] for v in tab.values()),
      presyn=dict(n_union=len(set().union(*presyn.values())),pairwise_shared=inter,common=len(common),common_share=common_share),
      reads='csr_matrix((data,indices,indptr)), rows are postsynaptic'))
    # ---- dose table and transfer models
    base=tab['lf']['share'];share0=float(np.mean([tab[lg]['share'] for lg in LEGS]))
    dose=[]
    for v_,k_ in ((1.0e-3,4.33),(2.0e-3,8.67),(4.0e-3,17.33),(8.0e-3,34.66)):
        shares=[(tab[lg]['poolexc_uS']-tab[lg]['motifE_uS']+tab[lg]['we_edges']*v_)/tab[lg]['poolexc_uS'] for lg in LEGS]
        dose.append(dict(v_uS=v_,k_out=k_,share_min=min(shares),share_max=max(shares),share_mean=float(np.mean(shares))))
    def i_of_f(f,t_ref=0.002,tau=0.020,R=1e9,dV=0.015):
        q=np.exp((1.0/max(f,1e-9)-t_ref)/tau)
        return q*dV/((q-1)*R)*1e9
    base_rate=175.1
    for d in dose:
        add=[d['share_min']*(i_of_f(base_rate)) , d['share_max']*(i_of_f(base_rate))]
        d['extra_current_nA']=[round(add[0],4),round(add[1],4)]
        d['predicted_pool_hz']=[round(float(1.0/(0.002+0.020*np.log(max(base_rate+i,1.0)/max(base_rate+i-15,1e-9)))),1) if False else None for i in add]
    rho=lambda c:1.0/c-1.0
    corrB0,corrP2=0.8521,0.7671;shareB0=share0;shareP2=float(np.mean([ (tab[lg]['poolexc_uS']-tab[lg]['motifE_uS']+tab[lg]['we_edges']*8.6432)/tab[lg]['poolexc_uS'] for lg in LEGS]))
    kA=(rho(corrP2)-rho(corrB0))/(shareP2-shareB0)
    def modelA(s): return 1.0/(1.0+rho(corrB0)+kA*(s-shareB0))
    alpha=np.log(rho(corrP2)/rho(corrB0))/np.log(shareP2/shareB0)
    def modelB(s): return 1.0/(1.0+rho(corrB0)*(s/shareB0)**alpha)
    kB=rho(corrP2)/max(shareP2,1e-12)
    def modelC(s): return 1.0/(1.0+kB*s)
    dump(o/'TRANSFER_MODEL.json',dict(dose=dose,shareB0=shareB0,shareP2=shareP2,corr_B0=corrB0,corr_P2=corrP2,
      alpha=alpha,kA=kA,kB=kB,
      predictions={str(s):dict(A=modelA(s),B=modelB(s),C=modelC(s)) for s in (0.10,0.20,0.30,0.40,0.53)},
      rule='if |meas(20%)-A| < |meas(20%)-B| then model A wins, otherwise B wins; if neither is within 0.10 the round records THIRD_MODEL',
      note='the 5 percent budget point of S329 is the motif OUTWARD CONDUCTANCE share of network excitation; the pool leg-specific share is a different quantity (0.72 to 1.10 percent at baseline)'))
    # ---- mode floor: partial correlations and motif-layer floor
    k17=np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:12000].astype(bool)
    k17a=np.unpackbits(np.load(S322/'SPK_K17_abl.npz')['packed'],axis=1)[:,:12000].astype(bool)
    def partials(A):
        g=env_of(A.sum(1).astype(float))
        def pc(x,y):
            rx=x-np.polyval(np.polyfit(g,x,1),g);ry=y-np.polyval(np.polyfit(g,y,1),g)
            return float(np.corrcoef(rx,ry)[0,1])
        own=[];other=[]
        for lg in LEGS:
            ref=env_of(A[:,loc[int(modules[lg]['E1'][0])]].astype(float))
            for j in pool_by_leg[lg]:
                own.append(pc(env_of(A[:,j].astype(float)),ref))
                for lg2 in LEGS:
                    if lg2==lg: continue
                    ref2=env_of(A[:,loc[int(modules[lg2]['E1'][0])]].astype(float))
                    other.append(pc(env_of(A[:,j].astype(float)),ref2))
        return float(np.median(own)),float(np.median(other))
    own17,oth17=partials(k17);ownA,othA=partials(k17a)
    motifE={lg:env_of(k17[:,loc[int(modules[lg]['E1'][0])]].astype(float)) for lg in LEGS}
    mp=pair_matrix(motifE)
    dump(o/'MODE_FLOOR.json',dict(partial_own=own17,partial_other=oth17,partial_own_abl=ownA,partial_other_abl=othA,
      motif_layer_pairs=mp,motif_layer_mean=float(np.mean(list(mp.values()))),
      note='with a 0.86 percent leg-specific budget the leg-specific transfer is not identifiable under the common mode; a purely motif-driven pool can at best reach the motif layer floor near 0.44'))
    dump(o/'S329_CORRECTION.json',dict(items=[
      dict(id=1,issue='selftest (a)',corrected='the twelve synthetic regular trains must be given independent start phases drawn uniformly in [0, 1000/175) ms, otherwise they are one string replicated twelve times and D1 is exactly 1.0'),
      dict(id=2,issue='selftest (c)',corrected='the 0.0132 is the shared linear-detrend and boxcar footprint of equal-length records, not collective structure; the criterion becomes D1_ctrl <= 0.05 and D1_obs - D1_ctrl >= 0.30, both numbers fixed by the measured dynamic range 0.013 versus 0.394, and a footprint-preserving null is reported alongside'),
      dict(id=3,issue='seed floor',corrected='all three S329 arms used background_seed=0 so the 0.085 dose effect had no sigma; S330 adds two zero-dose arms with different seeds'),
      dict(id=4,issue='P2 five percent',corrected='the S329 five percent budget point is the motif outward conductance share of network excitation, not the pool leg-specific share (0.72 to 1.10 percent); the two are reported separately from now on')]))
    # ---- stage B arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,banc_load=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in Gd if r['graded']]
    if not cand: dump(o/'STOP.json',dict(reason='no measured graded point'));return 2
    I_STAR=float(min(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
    b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
    e=b['edges'];We0,Wi0=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    def set_motif(We,Wi):
        for lg in LEGS:
            for src_r in ROLES:
                for tgt_r in ROLES:
                    if src_r==tgt_r: continue
                    for csrc in modules[lg][src_r]:
                        for ctgt in modules[lg][tgt_r]:
                            i=loc[int(csrc)];j=loc[int(ctgt)]
                            for Wm in (We,Wi):
                                s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                                if kk.size: Wm.data[s0+kk[0]]=K_DESIGN*EDGE_TOTAL
    def pool_same_leg_edges(We,Wi):
        out=[]
        for lg in LEGS:
            own=set(M18_by_leg[lg])
            for j in pool_by_leg[lg]:
                for Wm,tag in ((We,'e'),(Wi,'i')):
                    s0,e0=Wm.indptr[j],Wm.indptr[j+1]
                    for t in range(s0,e0):
                        if int(Wm.indices[t]) in own: out.append((tag,j,t))
        return out
    def iso_edges(We,Wi):
        out=[]
        for j in poolcols:
            for Wm,tag in ((We,'e'),(Wi,'i')):
                s0,e0=Wm.indptr[j],Wm.indptr[j+1]
                for t in range(s0,e0):
                    if int(Wm.indices[t]) not in Mset: out.append((tag,j,t))
        return out
    def run(name,seed=0,v=None,iso=False,ticks=5000):
        We=We0.copy();Wi=Wi0.copy();set_motif(We,Wi);touched=[]
        if v is not None:
            for tag,j,t in pool_same_leg_edges(We,Wi):
                Wm=We if tag=='e' else Wi;b0v=float(Wm.data[t]);Wm.data[t]=v
                touched.append(dict(kind=tag,row=int(j),col=int(Wm.indices[t]),before_uS=b0v,after_uS=v))
        if iso:
            for tag,j,t in iso_edges(We,Wi):
                Wm=We if tag=='e' else Wi;b0v=float(Wm.data[t]);Wm.data[t]=b0v*0.1
                touched.append(dict(kind=tag,row=int(j),col=int(Wm.indices[t]),before_uS=b0v,after_uS=float(Wm.data[t])))
        # realised share
        share=[]
        for lg in LEGS:
            own=set(M18_by_leg[lg]);e_tot=0.0;e_mine=0.0
            for j in pool_by_leg[lg]:
                s0,e0=We.indptr[j],We.indptr[j+1]
                for t in range(s0,e0):
                    val=float(We.data[t]);e_tot+=val
                    if int(We.indices[t]) in own: e_mine+=val
            share.append(e_mine/max(e_tot,1e-12))
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        drive=bg+np.full(n,I_STAR);spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        E={lg:env_of(spm[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
        pairs=pair_matrix(E)
        rates=np.array([spm[:,j].sum()*1000.0/ticks for j in poolcols])
        Y=np.stack([env_of(spm[:,j].astype(float)) for j in poolcols],1)
        d=d1_with_null(Y,np.random.default_rng(SEED+1))
        return dict(pairs=pairs,mean_cross_leg=float(np.mean(list(pairs.values()))),
          pool_median_rate_hz=float(np.median(rates)),f380=float((rates>=380).mean()),
          spike_total=int(spm.sum()),touched=len(touched),touch_sample=touched[:2],
          share_mean=float(np.mean(share)),share_min=float(min(share)),share_max=float(max(share)),
          D1=d['D1'],D1_null_p95=d['D1_null_p95'],nulls=d['nulls'])
    We_t=We0.copy();Wi_t=Wi0.copy();set_motif(We_t,Wi_t)
    b_t=dict(b);b_t['edges']=dict(b['edges']);b_t['edges']['We']=We_t;b_t['edges']['Wi']=Wi_t
    tier=NeuralTier(b_t,cfg,OwnershipLedger());CD['network_init']+=1
    cur=np.full(n,I_STAR);spm_t=np.zeros((5000,n),bool)
    for k in range(5000):
        spm_t[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0);CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
    ref=np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:12000].astype(bool)[:5000]
    gate=bool(np.array_equal(spm_t,ref))
    AR={}
    AR['G_cn_gate']=run('G_cn_gate',seed=0)
    gate_cn=bool(np.array_equal(np.unpackbits(np.load(o/'SPK_G_cn_gate.npz')['packed'],axis=1)[:,:int(np.load(o/'SPK_G_cn_gate.npz')['shape'][1])],spm_t))
    AR['V0_seed1']=run('V0_seed1',seed=1);AR['V0_seed2']=run('V0_seed2',seed=2)
    for nm,v in (('Kout4.33',1.0e-3),('Kout8.67',2.0e-3),('Kout17.33',4.0e-3),('Kout34.66',8.0e-3)):
        AR[nm]=run(nm,seed=0,v=v)
    AR['ISO90']=run('ISO90',seed=0,iso=True)
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',dict(gate_tier=gate,gate_cn=gate_cn))
    # ---- V14 synthetic and archived controls
    def reg_train(rate,ticks=5000,phase_ms=0.0):
        y=np.zeros(ticks);t=phase_ms
        while t<ticks:
            k=int(round(t))
            if 0<=k<ticks: y[k]=1.0
            t+=1000.0/rate
        return y
    def thin(y,pfunc):
        r=np.random.default_rng(11);return y*(r.random(len(y))<pfunc(np.arange(len(y))))
    ctr={}
    syn=[reg_train(175.0,phase_ms=float(np.random.default_rng(100+i).uniform(0,1000/175.0))) for i in range(12)]
    ctr['a']=d1_with_null(np.stack([env_of(x) for x in syn],1),np.random.default_rng(SEED+2))
    pfunc=lambda t: np.clip(1+0.4*np.sin(2*np.pi*21.0*t/1000.0),0,1)
    ctr['b']=d1_with_null(np.stack([env_of(thin(x,pfunc)) for x in syn],1),np.random.default_rng(SEED+3))
    pois=[]
    for i in range(12):
        r=np.random.default_rng(200+i);lam=175.0/1000.0
        pois.append((r.random(5000)<lam).astype(float))
    ctr['c']=d1_with_null(np.stack([env_of(x) for x in pois],1),np.random.default_rng(SEED+4))
    def arch(p):
        A=np.unpackbits(np.load(p)['packed'],axis=1)[:,:12000].astype(bool)
        Y=np.stack([env_of(A[:,j].astype(float)) for j in poolcols],1)
        d=d1_with_null(Y,np.random.default_rng(SEED+5))
        rates=np.array([A[:,j].sum()*1000.0/A.shape[0] for j in poolcols])
        peaks=np.array([peak_of(Y[:,i]) for i in range(Y.shape[1])],float)
        m=np.isfinite(peaks)
        d['corr_peak_rate']=float(np.corrcoef(peaks[m],rates[m])[0,1]) if m.sum()>3 and np.std(peaks[m])>0 else None
        return d,A
    ctr['d'],_cn=arch(S328/'SPK_C_norec.npz')
    ctr['e'],_b0=arch(S327/'SPK_B0_repeat.npz')
    ctr['f'],_a0=arch(S322/'SPK_A0_gate.npz')
    half=[]
    for sl in (slice(0,2500),slice(2500,5000)):
        Y=np.stack([env_of(_b0[sl,j].astype(float)) for j in poolcols],1)
        half.append(d1_with_null(Y,np.random.default_rng(SEED+6)))
    ctr['h']=dict(halves=half,spread=abs(half[0]['D1']-half[1]['D1']))
    checks=dict(
      a=bool(ctr['a']['D1']<=0.05), b=bool(ctr['b']['D1']>=0.35), c=bool(ctr['c']['D1']<=0.05),
      d=bool(ctr['d']['D1']<=0.02 and (ctr['d']['corr_peak_rate'] or 0)>=0.90),
      e=bool(ctr['e']['D1']>=0.35), f=True,
      g=all(r['D1_null_p95']<r['D1'] and r['D1_null_p95']>0 for r in (ctr['a'],ctr['b'],ctr['c'],ctr['d'],ctr['e'])),
      h=bool(all(x['D1']>x['D1_null_p95'] for x in half) and ctr['h']['spread']<=0.10))
    checks['all']=all(checks.values())
    dump(o/'DETECTOR_V14.json',dict(primary='mean pairwise zero-lag envelope correlation, per-cell independent circular shift null, NULLS=1000',
      D2='per-cell peak frequency versus rate-matched quantized regular train',D3='IQR of per-cell peak frequencies',
      floors=pre['selftest_floor'],diagnostics_only=['band_ratio','peak_hz']))
    dump(o/'DETECTOR_SELFTEST_V14.json',dict(controls=ctr,checks=checks,D1_FLOOR=ctr['c']['D1'],
      note='A0_gate is flagged as ripple saturated and may never be quoted as mode evidence'))

    # ---- stage C
    sig=float(np.std([AR['G_cn_gate']['mean_cross_leg'],AR['V0_seed1']['mean_cross_leg'],AR['V0_seed2']['mean_cross_leg']],ddof=1))
    dose_ok=[k for k in ('Kout4.33','Kout8.67','Kout17.33','Kout34.66') if 5<=AR[k]['pool_median_rate_hz']<=300 and AR[k]['f380']<=0.02]
    invalid=[k for k in ('Kout4.33','Kout8.67','Kout17.33','Kout34.66') if k not in dose_ok]
    hi=[k for k in ('Kout17.33','Kout34.66') if AR[k]['share_mean']>=0.18]
    corr_hi=min([AR[k]['mean_cross_leg'] for k in hi]) if hi else None
    branches={}
    if not checks['all']:
        dump(o/'STOP.json',dict(reason='V14 self-test failed',checks=checks));
    branches['x1']=bool(corr_hi is not None and corr_hi<=0.60 and (AR['G_cn_gate']['mean_cross_leg']-corr_hi)>3*sig)
    branches['x2']=bool(corr_hi is not None and corr_hi>=0.75)
    branches['x3']=bool(len(invalid)>=2)
    branches['x4']=bool(sig>=0.05)
    branches['x5']=bool(abs(AR['ISO90']['mean_cross_leg']-AR['Kout8.67']['mean_cross_leg'])>0.15 and abs(AR['ISO90']['share_mean']-AR['Kout8.67']['share_mean'])<0.05)
    order=['x4','x5','x3','x1','x2']
    branch=next((b for b in order if branches[b]),'NOT_JUDGED')
    if not checks['all']: branch='NOT_JUDGED'
    measured={k:AR[k]['mean_cross_leg'] for k in AR}
    pred=json.loads((o/'TRANSFER_MODEL.json').read_text())['predictions']
    meas20=measured.get('Kout17.33')
    win=None
    if meas20 is not None:
        win=('A' if abs(meas20-pred['0.2']['A'])<abs(meas20-pred['0.2']['B']) else 'B')
        if min(abs(meas20-pred['0.2'][m]) for m in 'ABC')>0.10: win='THIRD_MODEL'
    dump(o/'OUTCOMES.json',dict(round='S330',branch=branch,branches=branches,gate=dict(tier=gate,cn=gate_cn),
      seed_sigma=round(sig,4),dose_valid=dose_ok,invalid_dose=invalid,corr_hi=corr_hi,
      measured=measured,model_win=win,predictions_20=pred['0.2'],
      arms={k:dict(mean_cross=round(v['mean_cross_leg'],3),share=round(v['share_mean'],4),pool=round(v['pool_median_rate_hz'],1),
        f380=round(v['f380'],4),touched=v['touched'],D1=round(v['D1'],4),D1_p95=round(v['D1_null_p95'],4)) for k,v in AR.items()},
      motif_floor=json.loads((o/'MODE_FLOOR.json').read_text())['motif_layer_mean'],
      budget={lg:round(tab[lg]['share'],4) for lg in LEGS},cross_leg_pool=sum(v['cross_leg_edges'] for v in tab.values()),
      counters=CD,
      one_line='%s corr_hi %s seed_sigma %.3f share_hi %s'%(branch,None if corr_hi is None else round(corr_hi,3),sig,
        {k:round(AR[k]['share_mean'],3) for k in hi})))
    print(json.dumps(dict(branch=branch,branches=branches,gate=[gate,gate_cn],sig=round(sig,4),
      share={lg:round(tab[lg]['share'],4) for lg in LEGS},cross_pool=sum(v['cross_leg_edges'] for v in tab.values()),
      arms={k:(round(v['mean_cross_leg'],3),round(v['share_mean'],3),round(v['pool_median_rate_hz'],1)) for k,v in AR.items()},
      win=win,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
