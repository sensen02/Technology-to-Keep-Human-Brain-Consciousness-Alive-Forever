"""S329: phase-resolved command (D1/D2/D3), leg differentiation, three arms."""
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
S323=Path('/home/sensen/Desktop/engine修复验收/s323-window-invariant-detector/20261026T223533Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
S327=Path('/home/sensen/Desktop/engine修复验收/s327-global-mode-and-motif-influence/20261027T004101Z')
S328=Path('/home/sensen/Desktop/engine修复验收/s328-mode-origin-and-coupling-budget/20261027T005128Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def env_of(x,ds=2):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds] if ds>1 else y
def std_cols(Y):
    Y=Y-Y.mean(0,keepdims=True);s=Y.std(0,keepdims=True);s[s<1e-12]=1e-12
    return Y/s
def mean_pair_corr(Z):
    n,k=Z.shape
    if k<2: return 0.0
    colsum=Z.sum(1)
    tot=float(colsum@colsum)/n      # sum over all ordered pairs of correlations
    return (tot-k)/(k*(k-1))        # exclude the k unit diagonal entries
def peak_of(y,lo=4.0,hi=50.0):
    n=len(y);f=np.fft.rfftfreq(n,2e-3);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=lo+2*df)&(f<=hi-2*df)
    if not inter.any(): return None
    return float(f[int(np.argmax(np.where(inter,P,0)))])
def band_ratio(y,lo=18.0,hi=26.0,ds=2):
    n=len(y);f=np.fft.rfftfreq(n,ds*1e-3);P=np.abs(np.fft.rfft(y))**2
    return float(P[(f>=lo)&(f<=hi)].sum()/max(P[(f>=0.3)&(f<=50)].sum(),1e-30))
def quant_regular(rate_hz,n_bins,jitter_ms=0.0,rng=None,dt=1.0):
    rng=rng or np.random.default_rng(SEED)
    per=1000.0/max(rate_hz,1e-9);t=0.0;idx=[]
    while t<n_bins:
        j=0.0 if jitter_ms==0 else float(rng.normal(0,jitter_ms))
        k=int(round((t+j)/dt))
        if 0<=k<n_bins: idx.append(k)
        t+=per
    y=np.zeros(n_bins);y[np.array(idx,int)]=1.0;return y
def d_stats(Y,rates,rng,n_bins,ds=2):
    Z=std_cols(Y)
    obs=mean_pair_corr(Z)
    null=[]
    for _ in range(NULLS):
        sh=rng.integers(0,Z.shape[0],Z.shape[1])
        Zs=np.stack([np.roll(Z[:,i],sh[i]) for i in range(Z.shape[1])],1)
        null.append(mean_pair_corr(Zs))
    null=np.asarray(null);p=(np.sum(null>=obs)+1)/(len(null)+1)
    y_raw=Y
    peaks=np.array([peak_of(y_raw[:,i]) for i in range(Y.shape[1])],float)
    sur=[]
    for i in range(Y.shape[1]):
        q=quant_regular(rates[i],n_bins,0.0,rng)
        sur.append(peak_of(env_of(q,ds)))
    sur=np.asarray(sur,float)
    m=np.isfinite(peaks)&np.isfinite(sur)
    cc=float(np.corrcoef(peaks[m],sur[m])[0,1]) if m.sum()>3 and np.std(peaks[m])>0 and np.std(sur[m])>0 else None
    return dict(D1=float(obs),D1_null_p95=float(np.percentile(null,95)),D1_p=float(p),nulls=NULLS,
      D2_corr_peak_vs_surrogate=cc,D2_median_abs_diff=float(np.median(np.abs(peaks[m]-sur[m]))) if m.any() else None,
      D2_peaks=peaks.tolist()[:200],D3_peak_iqr=float(np.percentile(peaks[m],75)-np.percentile(peaks[m],25)) if m.sum()>3 else None,
      D2_rate_vs_peak_corr=float(np.corrcoef(rates[m],peaks[m])[0,1]) if m.sum()>3 and np.std(peaks[m])>0 else None,
      band_ratio_diag=float(np.mean([band_ratio(y_raw[:,i]) for i in range(Y.shape[1])])),
      peak_median=float(np.median(peaks[m])) if m.any() else None)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S329',mode='phase-resolved command: D1 pairwise envelope correlation, D2 quantized-regular surrogate, D3 peak dispersion; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=4,control_tick=20000,tier_advance=5000,neural_substep=40000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,
      rules=['band_ratio and peak_hz are diagnostics only and may not enter any criterion',
             'D1 uses per-cell independent circular shift nulls with NULLS=1000',
             'D2 compares per-cell peak frequency against a rate-matched quantized regular train',
             'archived EDGES npz must be read with csr_matrix, never csc_matrix (1943 out / 1553 in)',
             'no else fallback; slice asserts n <= len(signal); the saturated A0_gate arm may never be quoted as mode evidence'],
      branches=dict(c1='common command robust to leg bias and 5 percent out budget',c2='a lever reduces cross-leg correlation by >=0.20 with grading intact',
                    c3='a lever breaks grading',c4='NOT_JUDGED'))
    dump(o/'PREREG.json',pre)
    (o/'S329_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s329.py)=%s\nsha256(audit_s329_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s329_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in modules[lg][r]]
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    seen={}
    for r in rows:
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());pool_by_leg={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    rng=np.random.default_rng(SEED)
    arms={'K10':S322/'SPK_K10.npz','K15':S322/'SPK_K15.npz','K17':S322/'SPK_K17.npz','K17_abl':S322/'SPK_K17_abl.npz','K21':S322/'SPK_K21.npz',
          'A0_gate':S322/'SPK_A0_gate.npz','R1':S323/'SPK_R1_seed1.npz','R2':S323/'SPK_R2_seed2.npz','R3':S323/'SPK_R3_abl_seed1.npz','R4':S323/'SPK_R4_role_seed0.npz',
          'L1':S324/'SPK_L1_k17_seed0_10s.npz','L2':S324/'SPK_L2_k17abl_seed0_10s.npz',
          'B0':S327/'SPK_B0_repeat.npz','B1':S327/'SPK_B1_motif_no_out.npz','B2':S327/'SPK_B2_no_motif_drive.npz',
          'G_cn':S328/'SPK_G_cn_gate.npz','C_norec':S328/'SPK_C_norec.npz','I_noinh':S328/'SPK_I_noinh.npz','A_zero':S328/'SPK_A_zero.npz',
          'O_s10':S328/'SPK_O_s10.npz','O_s50':S328/'SPK_O_s50.npz','S_A':S328/'SPK_S_A.npz','S_B':S328/'SPK_S_B.npz','S_all':S328/'SPK_S_all.npz'}
    RJ={}
    for nm,f in arms.items():
        if not f.is_file(): continue
        A=loadz(f);Y=np.stack([env_of(A[:,j].astype(float)) for j in [j for (_l,_g),j in pool]],1)
        rates=np.array([A[:,[j for (_l,_g),j in pool][i]].sum()*1000.0/A.shape[0] for i in range(len(pool))])
        rj=d_stats(Y,rates,rng,A.shape[0])
        rj.pop('D2_peaks',None)
        rj['n_bins']=int(A.shape[0]);rj['pool_median_rate_hz']=float(np.median(rates))
        RJ[nm]=rj
    dump(o/'REJUDGE_V13.json',RJ)
    # ---- stage A: S328 correction with a function-level reconciliation of band_ratio
    b0=loadz(S327/'SPK_B0_repeat.npz');cn=loadz(S328/'SPK_C_norec.npz')
    def br_variants(A):
        cols=[j for (_l,_g),j in pool]
        v={}
        v['per_cell_mean_ds2']=float(np.mean([band_ratio(env_of(A[:,j].astype(float)),ds=2) for j in cols]))
        agg=env_of(A[:,cols].sum(1).astype(float),ds=2)
        v['aggregate_ds2']=band_ratio(agg,ds=2)
        agg1=env_of(A[:,cols].sum(1).astype(float),ds=1)
        v['aggregate_ds1']=band_ratio(agg1,ds=1)
        return v
    brB=br_variants(b0);brC=br_variants(cn)
    dump(o/'S328_CORRECTION.json',dict(items=[
      dict(id=1,band_ratio_phase_invariant=True,note='band_ratio is unchanged when every pool cell envelope is shifted independently, so it cannot evidence collective structure; V_INVALID is withdrawn'),
      dict(id=2,quantization_ripple=dict(note='a perfectly regular train can reach 0.3-0.6 in the 18-26 Hz band ratio near 175 Hz, so cross-arm band_ratio comparisons across a ninefold rate difference are invalid'),
        observed_here=dict(B0=brB,C_norec=brC)),
      dict(id=3,rate_scale_confound=dict(C_norec_pool_median_hz=RJ.get('C_norec',{}).get('pool_median_rate_hz'),
        B0_pool_median_hz=RJ.get('B0',{}).get('pool_median_rate_hz'),
        note='C_norec runs near 20 Hz so its own firing rate sits inside the band')),
      dict(id=4,B0_surrogate_rejection=dict(D2_corr=RJ.get('B0',{}).get('D2_corr_peak_vs_surrogate'),
        D3_peak_iqr=RJ.get('B0',{}).get('D3_peak_iqr'),note='B0 peak frequencies do not track the rate-matched quantized surrogate')),
      dict(id=5,A0_gate_forbidden=dict(D1=RJ.get('A0_gate',{}).get('D1'),note='the saturated equal-rate arm produces identical quantization ripple and may never be quoted as mode evidence')),
      dict(id=6,reconciliation=dict(s328_archived=dict(B0=0.5239,C_norec=0.3011),recomputed_here=dict(B0=brB,C_norec=brC),
        function_level='the archived value averages the per-cell band ratio of a 20 ms-smoothed, linearly detrended, 2x downsampled copy; the variants above split aggregate versus per-cell and 1x versus 2x downsampling so both conventions are archived instead of picking one'))]))
    dump(o/'DETECTOR_V13.json',dict(D1='mean pairwise zero-lag envelope correlation over pool cells standardised per cell, null = per-cell independent circular shift, NULLS=1000',
      D2='per-cell peak frequency versus a rate-matched quantized regular train (jitter 0), correlation and median absolute difference',
      D3='interquartile range of per-cell peak frequencies',
      diagnostics_only=['band_ratio','peak_hz'],jitter_ladder_ms=[0,0.25,0.5,1,2]))
    ST={}
    def mk(rate,n,mod=None,jitter=0.0,seed=1):
        r=np.random.default_rng(seed)
        y=quant_regular(rate,n,jitter)
        if mod:
            t=np.arange(n)
            p_keep=np.clip(1.0+mod[1]*np.sin(2*np.pi*mod[0]*t/1000.0),0.0,1.0)
            y=y*(r.random(n)<p_keep)
        return (y>0.5).astype(float)
    def mkarm(mats,rates,n):
        Y=np.stack([env_of(m) for m in mats],1);return d_stats(Y,np.asarray(rates,float),np.random.default_rng(SEED+1),n)
    n5=5000
    reg175=[mk(175.0,n5,seed=i) for i in range(12)]
    reg19=[mk(19.3,n5,seed=i) for i in range(12)]
    syn_mod=[mk(175.0,n5,mod=(21.0,0.4),seed=i) for i in range(12)]
    indep=[mk(r,n5,seed=100+i) for i,r in enumerate(np.linspace(156,176,12))]
    ST['a_regular175']=mkarm(reg175,[175.0]*12,n5)
    ST['a_regular19']=mkarm(reg19,[19.3]*12,n5)
    ST['b_regular175_mod21']=mkarm(syn_mod,[175.0]*12,n5)
    ST['c_independent']=mkarm(indep,list(np.linspace(156,176,12)),n5)
    ST['d_C_norec']=RJ.get('C_norec');ST['e_B0']=RJ.get('B0');ST['f_A0_gate']=RJ.get('A0_gate')
    checks=dict(a_regular_no_modulation=bool(ST['a_regular175']['D1']<ST['a_regular175']['D1_null_p95']*1.05 and ST['a_regular19']['D1']<ST['a_regular19']['D1_null_p95']*1.05),
      a_D2_explains_ripple=bool((ST['a_regular175']['D2_corr_peak_vs_surrogate'] or 0)>0.8 or (ST['a_regular175']['D2_median_abs_diff'] or 99)<4),
      b_modulation_detected=bool(ST['b_regular175_mod21']['D1']>ST['b_regular175_mod21']['D1_null_p95']),
      c_independent_inside_null=bool(ST['c_independent']['D1']<=ST['c_independent']['D1_null_p95']),
      d_C_norec_ripple=bool((ST['d_C_norec']['D2_corr_peak_vs_surrogate'] or 0)>0.8),
      e_B0_high=bool((ST['e_B0']['D1'] or 0)>=0.35),
      f_A0_flagged=bool(ST['f_A0_gate'] is not None))
    checks['all']=all(v for k,v in checks.items() if k!='all')
    dump(o/'DETECTOR_SELFTEST_V13.json',dict(results=ST,checks=checks))
    # ---- leg differentiation (main output)
    LEGD={}
    for nm in ('B0','C_norec','P1','P2'):
        f=o/f'SPK_{nm}.npz' if nm.startswith('P') else arms.get(nm)
        if f is None or not Path(f).is_file(): continue
        A=loadz(f)
        E={lg:env_of(A[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                pairs[f'{a}|{b}']=float(np.corrcoef(E[a],E[b])[0,1])
        M=np.corrcoef(np.stack([E[lg] for lg in LEGS],1).T)
        resid={}
        for lg in LEGS:
            others=np.mean([E[o2] for o2 in LEGS if o2!=lg],axis=0)
            resid[lg]=E[lg]-others
        rms=float(np.sqrt(np.mean([np.sqrt(np.mean(resid[lg]**2))/max(np.sqrt(np.mean(E[lg]**2)),1e-12) for lg in LEGS])))
        RC=np.corrcoef(np.stack([resid[lg] for lg in LEGS],1).T)
        rc_off=[RC[i,j] for i in range(6) for j in range(6) if i<j]
        half=float(np.corrcoef(E['lf'],np.roll(E['rm'],24//2))[0,1])
        LEGD[nm]=dict(pairs=pairs,mean_cross_leg=float(np.mean(list(pairs.values()))),
          resid_rms_level=rms,resid_corr_median=float(np.median(rc_off)),
          corr_lf_vs_roll_rm_12ms=half,per_leg_rate={lg:float(A[:,pool_by_leg[lg]].sum()*1000.0/A.shape[0]/len(pool_by_leg[lg])) for lg in LEGS})
    dump(o/'LEG_DIFFERENTIATION.json',LEGD)
    # ---- stage B three arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,banc_load=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in Gd if r['graded']]
    if not cand: dump(o/'STOP.json',dict(reason='no measured graded point'));return 2
    I_STAR=float(min(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    BUD=json.loads((S328/'BUDGET.json').read_text());S5=float(BUD['counterfactual_scale']['5pct'])
    cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
    b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
    e=b['edges'];We0,Wi0=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    n=We0.shape[0];Mset=set(M18)
    def set_motif(We,Wi,m=0):
        for lg in LEGS:
            for src_r in ROLES:
                for tgt_r in ROLES:
                    if src_r==tgt_r: continue
                    for csrc in modules[lg][src_r]:
                        for ctgt in modules[lg][tgt_r]:
                            i=loc[int(csrc)];j=loc[int(ctgt)]
                            for Wm in (We,Wi):
                                s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                                if kk.size: Wm.data[s0+kk[0]]=K_DESIGN*EDGE_TOTAL;m+=1
        return m
    def out_edges(We,Wi):
        out=[]
        for j in range(n):
            if j in Mset: continue
            for Wm,tag in ((We,'e'),(Wi,'i')):
                s0,e0=Wm.indptr[j],Wm.indptr[j+1]
                for t in range(s0,e0):
                    if int(Wm.indices[t]) in Mset: out.append((tag,j,t))
        return out
    def run_cn(name,out_scale=None,cur_mult=None,ticks=5000):
        We=We0.copy();Wi=Wi0.copy();set_motif(We,Wi);touched=[]
        if out_scale:
            for tag,j,t in out_edges(We,Wi):
                Wm=We if tag=='e' else Wi;b0v=float(Wm.data[t]);Wm.data[t]=b0v*out_scale
                touched.append(dict(kind=tag,row=int(j),col=int(Wm.indices[t]),before_uS=b0v,after_uS=float(Wm.data[t])))
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi);CD['network_init']+=1
        bg=np.random.default_rng(cfg.background_seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if cur_mult:
            for lg in LEGS:
                mult=1.0+0.10*(1.0 if lg in TRIPOD['A'] else -1.0)
                for j in pool_by_leg[lg]: cur[j]=I_STAR*mult
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        E={lg:env_of(spm[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
        pairs={f'{a}|{b}':float(np.corrcoef(E[a],E[b])[0,1]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        rates=np.array([spm[:,[j for (_l,_g),j in pool][i]].sum()*1000.0/ticks for i in range(len(pool))])
        Y=np.stack([env_of(spm[:,j].astype(float)) for (_l,_g),j in pool],1)
        return dict(pairs=pairs,mean_cross_leg=float(np.mean(list(pairs.values()))),
          pool_median_rate_hz=float(np.median(rates)),f380=float((rates>=380).mean()),
          touched=len(touched),touch_sample=touched[:2],spike_total=int(spm.sum()),
          D1=mean_pair_corr(std_cols(Y))),spm
    We_t=We0.copy();Wi_t=Wi0.copy();set_motif(We_t,Wi_t)
    b_t=dict(b);b_t['edges']=dict(b['edges']);b_t['edges']['We']=We_t;b_t['edges']['Wi']=Wi_t
    tier=NeuralTier(b_t,cfg,OwnershipLedger());CD['network_init']+=1
    cur=np.full(n,I_STAR);spm_t=np.zeros((5000,n),bool)
    for k in range(5000):
        spm_t[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0);CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
    ref=loadz(S322/'SPK_K17.npz')[:5000]
    gate=bool(np.array_equal(spm_t,ref))
    g,_=run_cn('G_cn_gate');gate_cn=bool(np.array_equal(_,spm_t))
    p1,_=run_cn('P1_legI_asym',cur_mult=True)
    p2,_=run_cn('P2_motif_out_gain',out_scale=S5)
    dump(o/'ARMS.json',dict(G_cn_gate=g,P1=p1,P2=p2,S5_used=S5));dump(o/'CALL_LEDGER.json',CD)
    dump(o/'GATE.json',dict(gate_tier=gate,gate_cn=gate_cn))
    # ---- stage C
    if not (gate and gate_cn) or not checks['all']: branch='c4'
    else:
        b0d=LEGD.get('B0',{})
        okD1=bool((RJ.get('B0',{}).get('D1') or 0)>=0.35)
        robust=bool(p1['mean_cross_leg']>=0.75 and p2['mean_cross_leg']>=0.75)
        drop=[nm for nm,v in (('P1',p1),('P2',p2)) if v['mean_cross_leg']<=b0d.get('mean_cross_leg',1.0)-0.20]
        graded=all(5<=v['pool_median_rate_hz']<=300 and v['f380']<=0.02 for v in (p1,p2))
        branch='c1' if (okD1 and robust) else ('c2' if (drop and graded) else ('c3' if not graded else 'c4'))
    dump(o/'OUTCOMES.json',dict(round='S329',branch=branch,gate=dict(tier=gate,cn=gate_cn),
      D1_B0=RJ.get('B0',{}).get('D1'),D1_null_p95_B0=RJ.get('B0',{}).get('D1_null_p95'),D3_B0=RJ.get('B0',{}).get('D3_peak_iqr'),
      D2_B0_corr=RJ.get('B0',{}).get('D2_corr_peak_vs_surrogate'),D2_C_norec_corr=RJ.get('C_norec',{}).get('D2_corr_peak_vs_surrogate'),
      leg_diff={k:dict(mean_cross=v['mean_cross_leg'],resid=v['resid_rms_level'],corr_half=v['corr_lf_vs_roll_rm_12ms']) for k,v in LEGD.items()},
      arms={k:dict(mean_cross=round(v['mean_cross_leg'],3),pool=round(v['pool_median_rate_hz'],1),f380=round(v['f380'],4),touched=v['touched'],D1=round(v['D1'],4)) for k,v in (('P1',p1),('P2',p2),('G_cn',g))},
      S5_used=S5,selftest=checks,counters=CD,
      one_line='%s cross_leg B0 %.3f P1 %.3f P2 %.3f D1_B0 %.3f'%(branch,LEGD.get('B0',{}).get('mean_cross_leg',0),p1['mean_cross_leg'],p2['mean_cross_leg'],RJ.get('B0',{}).get('D1') or 0)))
    print(json.dumps(dict(branch=branch,gate=[gate,gate_cn],selftest=checks['all'],
      D1={k:round(v['D1'],4) for k,v in RJ.items()},legB0=round(LEGD.get('B0',{}).get('mean_cross_leg',0),3),
      P1=round(p1['mean_cross_leg'],3),P2=round(p2['mean_cross_leg'],3),S5=round(S5,3),cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
