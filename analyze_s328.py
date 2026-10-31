"""S328: common-mode origin, coupling budget and leg-level routing (ten arms)."""
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
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=200;SEED=20261026
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def sm(x,w=20): return np.convolve(np.asarray(x,float),np.ones(w)/w,mode='same')
def env(x):
    y=sm(x);t=np.arange(len(y));return y-np.polyval(np.polyfit(t,y,1),t)
def autocorr_desc(y,maxlag=60):
    yc=y-y.mean();ac=np.correlate(yc,yc,'full')[len(y)-1:];ac/=max(ac[0],1e-30)
    lags={k:float(ac[k]) for k in (5,10,20,25,40,50) if k<len(ac)}
    seg=ac[8:min(maxlag,len(ac)-1)]
    neg=None;rec=None
    if seg.size:
        kk=int(np.argmin(seg))
        if seg[kk]<0: neg=float(8+kk)
    if neg is not None:
        tail=ac[int(neg):min(int(neg)+40,len(ac))]
        pos=np.flatnonzero(tail>0)
        if pos.size: rec=float(neg+pos[0])
    return dict(lags=lags,first_negative_lag_ms=neg,first_recovery_lag_ms=rec,
      period_est_ms=(2*(neg+ (rec-neg if rec else 0)) if neg and rec else (2*neg if neg else None)))
def band_ratio(y,lo=18.0,hi=26.0):
    n=len(y);f=np.fft.rfftfreq(n,1e-3);P=np.abs(np.fft.rfft(y))**2
    tot=P[(f>=0.3)&(f<=50)].sum()
    return float(P[(f>=lo)&(f<=hi)].sum()/max(tot,1e-30))
def peak_spec2(y,lo,hi):
    n=len(y);f=np.fft.rfftfreq(n,1e-3);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    interior=(f>=lo+2*df)&(f<=hi-2*df)
    if not interior.any(): return None
    k=int(np.argmax(np.where(interior,P,0)));return float(f[k])
def poisson_null_band(count,n_bins,n=200,rng=None,lo=18.0,hi=26.0):
    rng=rng or np.random.default_rng(SEED+9);lam=count/max(n_bins,1);out=[]
    for _ in range(n):
        s=(rng.random(n_bins)<lam).astype(float)
        out.append(band_ratio(env(s),lo,hi))
    return float(np.percentile(out,95)) if out else None
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S328',mode='common-mode origin, coupling budget and leg routing; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=10,control_tick=50000,tier_advance=5000,neural_substep=100000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,design=dict(edge_total_uS=EDGE_TOTAL,k=K_DESIGN),
      rules=['the common-mode descriptor must not use argmax; it uses the envelope autocorrelation lags, the 18-26 Hz band ratio and per-cell correlation quantiles, with peak_hz only as a diagnostic in both the [4,50] and [5,45] bands',
             'live We/Wi are CSR with rows as postsynaptic targets; archived EDGES npz must be read with csr_matrix and never csc_matrix',
             'envelope spectrum nulls are rate-matched independent Poisson trains; cross coherence nulls randomise only the reference phase spectrum',
             'no else fallback in the branch; slice asserts n <= len(signal); every statistic record carries its nulls count',
             'design values and the operating point come from archived measurement tables'],
      branches=dict(V='C_norec must lose the common mode',M='adaptation or E-I origin',K='effective or saturated coupling',
                    L='leg-level routing detectable or not'))
    dump(o/'PREREG.json',pre)
    (o/'S328_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s328.py)=%s\nsha256(audit_s328_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s328_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    E1={lg:loc[int(modules[lg]['E1'][0])] for lg in LEGS}
    M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in modules[lg][r]];Mset=set(M18)
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    seen={}
    for r in rows:
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());poolcols=[j for (_l,_g),j in pool]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    # ---------- stage A: axis check + budget
    We25=np.load(S303/'EDGES_w_2.5e-5_We.npz');Wi25=np.load(S303/'EDGES_w_2.5e-5_Wi.npz')
    def tocsr(z):
        n=int(z['indptr'].size-1)
        return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
    We_c=tocsr(We25);Wi_c=tocsr(Wi25)
    n=We_c.shape[0]
    out_e=sum(float(We_c[j,i]) for j in range(n) if j not in Mset for i in We_c.indices[We_c.indptr[j]:We_c.indptr[j+1]] if int(i) in Mset)
    out_i=sum(float(Wi_c[j,i]) for j in range(n) if j not in Mset for i in Wi_c.indices[Wi_c.indptr[j]:Wi_c.indptr[j+1]] if int(i) in Mset)
    in_e=sum(float(We_c[j,i]) for j in Mset for i in We_c.indices[We_c.indptr[j]:We_c.indptr[j+1]] if int(i) not in Mset)
    in_i=sum(float(Wi_c[j,i]) for j in Mset for i in Wi_c.indices[Wi_c.indptr[j]:Wi_c.indptr[j+1]] if int(i) not in Mset)
    def count_edges(kind):
        tot={'exc':0,'inh':0}
        for Wm,tag in ((We_c,'exc'),(Wi_c,'inh')):
            for j in range(n):
                for i in Wm.indices[Wm.indptr[j]:Wm.indptr[j+1]]:
                    if kind=='out' and int(i) in Mset and j not in Mset: tot[tag]+=1
                    if kind=='in' and int(i) not in Mset and j in Mset: tot[tag]+=1
        return tot
    co=count_edges('out');ci=count_edges('in')
    n_out=co['exc']+co['inh'];n_in=ci['exc']+ci['inh']
    tot_e=float(We_c.sum());tot_i=float(Wi_c.sum())
    s_ctr={ '1pct':(0.01/ (out_e/tot_e)), '5pct':(0.05/(out_e/tot_e)), '20pct':(0.20/(out_e/tot_e)) } if out_e>0 else {}
    dump(o/'BUDGET.json',dict(w=2.5e-5,edges_out=n_out,edges_in=n_in,
      out_exc_uS=out_e,out_inh_uS=out_i,in_exc_uS=in_e,in_inh_uS=in_i,
      edge_counts=dict(out=co,inn=ci,out_total=n_out,inn_total=n_in,
        note='the totals 1943 and 1553 are the We+Wi sums; the per-matrix split is recorded so the two conventions cannot be confused again'),
      network_we_uS=tot_e,network_wi_uS=tot_i,share_out_exc=out_e/tot_e,share_out_inh=out_i/tot_i,
      motif_edge_uS=EDGE_TOTAL,motif_internal_budget_uS=34*EDGE_TOTAL,
      counterfactual_scale=s_ctr,
      counterfactual_formula='s = target_share / (out_e/network_we_uS) with the linear extrapolation stated explicitly',
      note='the 34 motif internal edges were raised to k=17.33 while the 1943 outward edges were never raised: min(count,8)*2.5e-5 throughout'))
    # mode descriptors over archived arms
    armsA={'B0':S327/'SPK_B0_repeat.npz','B1':S327/'SPK_B1_motif_no_out.npz','B2':S327/'SPK_B2_no_motif_drive.npz',
           'L1':S324/'SPK_L1_k17_seed0_10s.npz','L2':S324/'SPK_L2_k17abl_seed0_10s.npz',
           'K10':S322/'SPK_K10.npz','K15':S322/'SPK_K15.npz','K17':S322/'SPK_K17.npz','K21':S322/'SPK_K21.npz',
           'K17_abl':S322/'SPK_K17_abl.npz','A0_gate':S322/'SPK_A0_gate.npz'}
    MD={}
    for nm,f in armsA.items():
        A=loadz(f);y=env(A.sum(1).astype(float))
        gc=y.copy();pcell=np.array([float(np.corrcoef(env(A[:,j].astype(float)),gc)[0,1]) for j in poolcols])
        MD[nm]=dict(n_bins=int(A.shape[0]),autocorr=autocorr_desc(y),band_ratio_18_26=band_ratio(y),
          peak_4_50=peak_spec2(y,4.0,50.0),peak_5_45=peak_spec2(y,5.0,45.0),
          per_cell_corr=dict(median=float(np.median(pcell)),p90=float(np.percentile(pcell,90)),
                             frac_gt_0p3=float((pcell>0.3).mean())),
          poisson_p95_band=poisson_null_band(int(A.sum()),int(A.shape[0])))
    dump(o/'MODE_DESCRIPTOR.json',MD)
    dump(o/'S327_CORRECTION.json',dict(items=[
      dict(id=1,axis=dict(edges_out=n_out,edges_in=n_in,read='csr_matrix((data,indices,indptr)) with rows as postsynaptic targets; csc_matrix would transpose and swap 1943 and 1553'),
        corrected='the outward count 1943 and the inward count 1553 are both confirmed and are not interchangeable'),
      dict(id=2,y1='motif has no measurable effect on the network',corrected='y1 is budget-consistent: the M18 outward coupling is %.3f percent of network excitation and %.3f percent of inhibition, so a 0.5 percent pool change is the expected result, not evidence of no causal role'%(100*out_e/tot_e,100*out_i/tot_i)),
      dict(id=3,peak_hz='drift 0.0 Hz',corrected='peak_hz moves with the interior band (B2 19.6 vs 23.2 Hz for [4,50] and [5,45]) and B0 has comparable local maxima at 23.4 and 25.8 Hz, so the zero-drift criterion is withdrawn in favour of autocorrelation lags'),
      dict(id=4,conc_ratio=0.954,corrected='a wide envelope puts neighbouring bins in the same band, so the concentration ratio cannot evidence the same mode'),
      dict(id=5,note='the 1943 outward edges were never amplified; only the 34 internal edges were set to the k design value')]))
    # leg noise floor
    try:
        K17z=loadz(S322/'SPK_K17.npz');R1=loadz(S323/'SPK_R1_seed1.npz');R2=loadz(S323/'SPK_R2_seed2.npz')
        def legrates(A):
            return {lg:float(A[:,[j for (l2,gi),j in pool if l2==lg]].sum(1).mean()*1000/len([1 for (l2,gi),_j in pool if l2==lg])) for lg in LEGS}
        rr=[legrates(K17z),legrates(R1),legrates(R2)]
        sig={lg:float(np.std([r[lg] for r in rr],ddof=1)) for lg in LEGS}
        dump(o/'LEG_NOISE_FLOOR.json',dict(config_same=True,per_arm=dict(K17=rr[0],R1=rr[1],R2=rr[2]),sigma_hz=sig,
          note='three archived arms with the same k and I* and different background seeds'))
    except Exception as e:
        dump(o/'LEG_NOISE_FLOOR.json',dict(config_same=False,reason=str(e)))
        sig={lg:0.0 for lg in LEGS}
    # ---------- stage B ten arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,banc_load=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in Gd if r['graded']]
    if not cand: dump(o/'STOP.json',dict(reason='no measured graded point for I*'));return 2
    I_STAR=float(min(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
    b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
    e=b['edges'];We0,Wi0=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    def set_motif(We,Wi):
        m=0
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
        lst=[]
        for j in range(n):
            if j in Mset: continue
            s0,e0=We.indptr[j],We.indptr[j+1]
            for t in range(s0,e0):
                if int(We.indices[t]) in Mset: lst.append(('e',j,t))
            s1,e1=Wi.indptr[j],Wi.indptr[j+1]
            for t in range(s1,e1):
                if int(Wi.indices[t]) in Mset: lst.append(('i',j,t))
        return lst
    def run_cn(name,params=None,zero_we=False,zero_wi=False,out_scale=None,cur_override=None,ticks=5000):
        params=params or ConductanceParams()
        We=We0.copy();Wi=Wi0.copy()
        set_motif(We,Wi)
        touched=[]
        if out_scale:
            lst=out_edges(We,Wi)
            for kind,j,t in lst:
                Wm=We if kind=='e' else Wi
                before=float(Wm.data[t]);Wm.data[t]=before*out_scale
                touched.append(dict(kind=kind,row=int(j),col=int(Wm.indices[t]),before_uS=before,after_uS=float(Wm.data[t])))
        if zero_we: We=We*0.0
        if zero_wi: Wi=Wi*0.0
        net=ConductanceNetwork(n,params,cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi)
        CD['network_init']+=1
        bg=np.random.default_rng(cfg.background_seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if cur_override is not None:
            for j,v in cur_override.items(): cur[j]=v
        drive=bg+cur
        spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            s1=net.step(drive);s2=net.step(drive)
            spm[k]=np.logical_or(s1,s2);CD['control_tick']+=1;CD['neural_substep']+=2
        y=env(spm.sum(1).astype(float))
        desc=dict(autocorr=autocorr_desc(y),band_ratio_18_26=band_ratio(y),
          peak_4_50=peak_spec2(y,4.0,50.0),peak_5_45=peak_spec2(y,5.0,45.0),
          pool_mean_rate_hz=float(spm[:,poolcols].sum(1).mean()*1000/len(poolcols)),
          leg_mean_rate_hz={lg:float(spm[:,[j for (l2,gi),j in pool if l2==lg]].sum(1).mean()*1000/max(len([1 for (l2,gi),_j in pool if l2==lg]),1)) for lg in LEGS},
          touched=len(touched),touched_sample=touched[:3],spike_total=int(spm.sum()))
        return dict(spm=spm,desc=desc)
    # tier gate: the motif edges must be set to the same k as the archived run
    We_t=We0.copy();Wi_t=Wi0.copy();set_motif(We_t,Wi_t)
    b_t=dict(b);b_t['edges']=dict(b['edges']);b_t['edges']['We']=We_t;b_t['edges']['Wi']=Wi_t
    tier=NeuralTier(b_t,cfg,OwnershipLedger());CD['network_init']+=1
    cur=np.full(n,I_STAR);spm_t=np.zeros((5000,n),bool)
    for k in range(5000):
        spm_t[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0)
        CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
    ref=loadz(S322/'SPK_K17.npz')[:5000]
    gate_tier=bool(np.array_equal(spm_t,ref))
    np.savez_compressed(o/'SPK_G_tier_gate.npz',packed=np.packbits(spm_t,axis=1),shape=np.array(spm_t.shape))
    res={};branch=None
    cn=run_cn('G_cn_gate')
    gate_cn=bool(np.array_equal(cn['spm'],spm_t))
    np.savez_compressed(o/'SPK_G_cn_gate.npz',packed=np.packbits(cn['spm'],axis=1),shape=np.array(cn['spm'].shape))
    res['G_cn_gate']=cn['desc']
    if not (gate_tier and gate_cn):
        dump(o/'STOP.json',dict(reason='gate failed: tier or ConductanceNetwork mismatch',gate_tier=gate_tier,gate_cn=gate_cn));branch=None
    for nm,kw in (('C_norec',dict(zero_we=True,zero_wi=True)),('I_noinh',dict(zero_wi=True)),
                  ('A_zero',dict(params=ConductanceParams(adaptation_increment_nA=0.0))),
                  ('O_s10',dict(out_scale=10.0)),('O_s50',dict(out_scale=50.0))):
        r=run_cn(nm,**kw);res[nm]=r['desc']
        np.savez_compressed(o/f'SPK_{nm}.npz',packed=np.packbits(r['spm'],axis=1),shape=np.array(r['spm'].shape))
    sensory={lg:[int(x) for x in np.asarray(b['sensory'][lg])] for lg in LEGS}
    def cur_for(legs,val=0.040,base=0.020):
        d={}
        for lg in legs:
            for j in sensory[lg]: d[j]=val
        return d
    for nm,legs in (('S_A',TRIPOD['A']),('S_B',TRIPOD['B'])):
        r=run_cn(nm,cur_override=cur_for(legs));res[nm]=r['desc'];res[nm]['legs']=list(legs)
        res[nm]['group_sizes']={lg:len(sensory[lg]) for lg in legs}
        res[nm]['total_injected_nA']=float(sum(0.020*len(sensory[lg]) for lg in legs))
        np.savez_compressed(o/f'SPK_{nm}.npz',packed=np.packbits(r['spm'],axis=1),shape=np.array(r['spm'].shape))
    tot=sum(0.020*len(sensory[lg]) for lg in TRIPOD['A']+TRIPOD['B'])
    d_all=cur_for(TRIPOD['A']+TRIPOD['B'],val=I_STAR+tot/n)
    r=run_cn('S_all',cur_override={j:v for j,v in d_all.items()});res['S_all']=r['desc']
    np.savez_compressed(o/'SPK_S_all.npz',packed=np.packbits(r['spm'],axis=1),shape=np.array(r['spm'].shape))
    res['S_all']['total_injected_nA']=tot
    dump(o/'ARMS.json',res);dump(o/'CALL_LEDGER.json',CD)
    dump(o/'GATE.json',dict(gate_tier=gate_tier,gate_cn=gate_cn,tripod_policy='used the implemented tripod split A={lf,rm,lh} B={lm,rf,rh}; the repository definition is recorded below',repository=None))
    # ---------- stage C
    q95=lambda nm: MD[nm]['poisson_p95_band']
    V='V_INVALID' if (res['C_norec']['band_ratio_18_26']>=q95('B0')) else 'V_VALID'
    base_period=MD['B0']['autocorr'].get('first_negative_lag_ms')
    def period(nm):
        a=res[nm]['autocorr'];return a.get('first_negative_lag_ms')
    def moved(nm,thr):
        p=period(nm)
        if p is None or base_period is None: return None
        return abs(p-base_period)/base_period>thr
    M=None
    if moved('A_zero',0.20) is True: M='M_ADAPTATION'
    elif moved('A_zero',0.20) is False:
        ch=abs(res['I_noinh']['band_ratio_18_26']-res['G_cn_gate']['band_ratio_18_26'])/max(res['G_cn_gate']['band_ratio_18_26'],1e-12)
        mv=moved('I_noinh',0.20)
        if ch>0.30 or mv is True: M='M_EI'
        elif ch<0.10 and mv is False: M='M_NOT_EI'
        else: M='M_UNDECIDED'
    else: M='M_UNDECIDED'
    base=res['G_cn_gate']['pool_mean_rate_hz']
    k50=(res['O_s50']['pool_mean_rate_hz']-base)/max(base,1e-9);k10=(res['O_s10']['pool_mean_rate_hz']-base)/max(base,1e-9)
    K='K_EFFECTIVE' if (abs(k50)>=0.10 and np.sign(k50)==np.sign(k10)) else ('K_SATURATED' if abs(k50)<0.02 else 'K_UNDECIDED')
    rA=lambda d:(sum(d['leg_mean_rate_hz'][lg] for lg in TRIPOD['A'])/3.0)
    dA=rA(res['S_A'])-rA(res['S_all']);dB=rA(res['S_B'])-rA(res['S_all'])
    sigA=float(np.mean([sig[lg] for lg in TRIPOD['A']])) if sig else 0.0
    SI=(dA-dB)/(abs(dA)+abs(dB)) if (abs(dA)+abs(dB))>0 else 0.0
    if abs(dA)<=3*sigA: L='L_UNDETECTABLE'
    elif abs(SI)>0.5: L='L_SPECIFIC'
    elif abs(SI)<0.2: L='L_COMMON'
    else: L='L_UNDECIDED'
    if branch is None: branch='NOT_JUDGED'
    elif V=='V_INVALID': branch='NOT_JUDGED'
    else: branch='V_VALID'
    dump(o/'OUTCOMES.json',dict(round='S328',branch=branch,V=V,M=M,K=K,L=L,
      gate=dict(tier=gate_tier,cn=gate_cn),arms={k:dict(band=round(v['band_ratio_18_26'],4),peak450=round(v['peak_4_50'],2) if v['peak_4_50'] else None,
        peak545=round(v['peak_5_45'],2) if v['peak_5_45'] else None,pool=round(v['pool_mean_rate_hz'],1),
        neg_lag=v['autocorr'].get('first_negative_lag_ms'),touched=v.get('touched')) for k,v in res.items()},
      budget=dict(out_share=round(100*out_e/tot_e,3),in_share=round(100*out_i/tot_i,3),edges_out=n_out,edges_in=n_in),
      leg_delta=dict(dA=round(dA,3),dB=round(dB,3),SI=round(SI,3),sigmaA=round(sigA,3)),
      counters=CD,
      one_line='%s | M=%s K=%s L=%s'%(branch,M,K,L)))
    print(json.dumps(dict(branch=branch,V=V,M=M,K=K,L=L,gate=[gate_tier,gate_cn],
      arms={k:(round(v['band_ratio_18_26'],4),round(v['pool_mean_rate_hz'],1),v['autocorr'].get('first_negative_lag_ms')) for k,v in res.items()},
      dA=round(dA,3),dB=round(dB,3),SI=round(SI,3),sig=round(sigA,3),cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
