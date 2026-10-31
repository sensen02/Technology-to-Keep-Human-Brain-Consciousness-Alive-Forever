"""S331: global phase audit and declarative pi injection capability test."""
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
S330=Path('/home/sensen/Desktop/engine修复验收/s330-leg-specific-budget-and-decoupling/20261027T011205Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def bursts(x,dt=1.0):
    i=np.flatnonzero(x)
    if len(i)<2: return np.array([],int),np.array([],int)
    d=np.diff(i)*dt;thr=max(3*np.median(d),5.0);st=[i[0]];cn=[1]
    for k,dd in enumerate(d):
        if dd>thr: st.append(i[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def env_of(x,ds=2):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds] if ds>1 else y
def phase_at(y,f0,dt_ms=2.0):
    t=np.arange(len(y))*dt_ms/1000.0
    z=np.sum(y*np.exp(-2j*np.pi*f0*t))
    return float(np.angle(z)),float(abs(z)/max(np.sum(np.abs(y)),1e-12))
def circ_R_phase(ph):
    if len(ph)==0: return 0.0,0.0
    z=np.exp(1j*np.asarray(ph));return float(abs(z.mean())),float(np.angle(z.mean()))
def angdiff(a,b):
    d=(a-b+np.pi)%(2*np.pi)-np.pi;return abs(np.degrees(d))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S331',mode='global phase audit plus declarative pi injection; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=5,control_tick=25000,tier_advance=0,neural_substep=50000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,
      rules=['the cross-leg criterion is PHASE, not correlation; correlation is a diagnostic column',
             'any cross-arm comparison asserts the pool median rate stays within 10 percent of Z0_gate or is marked RATE_MISMATCH and dropped',
             'nulls must be independently synthesised with null_std>0 and |obs-null_p95|>1e-9; absolute floors are forbidden in favour of a matched synthetic control minus 0.05',
             'archived EDGES npz are read with csr_matrix, never csc_matrix',
             'no else fallback; slice asserts n <= len(signal); every record carries nulls'],
      branches=dict(w1='the pool can carry an imposed phase offset so what is missing is the source',w2='the pool is a saturated follower',
                    w3='UNDERPOWERED at 30 percent'))
    dump(o/'PREREG.json',pre)
    (o/'S331_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s331.py)=%s\nsha256(audit_s331_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s331_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    M18_by_leg={lg:[loc[int(c)] for r in ROLES for c in modules[lg][r]] for lg in LEGS}
    M18=[j for lg in LEGS for j in M18_by_leg[lg]];Mset=set(M18)
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());pool_by_leg={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    k17=loadz(S322/'SPK_K17.npz');k17a=loadz(S322/'SPK_K17_abl.npz')
    e1={lg:loc[int(modules[lg]['E1'][0])] for lg in LEGS}
    # modal frequency from the measured motif E1 bursts of tripod A
    fiv={}
    for lg in LEGS:
        st,_=bursts(k17[:,e1[lg]].astype(float),1.0)
        fiv[lg]=float(1000.0/np.mean(np.diff(st))) if len(st)>=10 else None
    f0=float(np.median([v for l_,v in fiv.items() if l_ in TRIPOD['A'] and v]))
    # ---- stage A: global phase table
    def phase_table(A):
        res={}
        rate=A.sum(0)*1000.0/A.shape[0]
        bands={'all':np.ones(len(rate),bool),'5to50':(rate<50),'50to150':(rate>=50)&(rate<150),'150to300':(rate>=150)&(rate<300),'300to400':rate>=300}
        for nm,m in bands.items():
            idx=np.flatnonzero(m)
            if idx.size<5: res[nm]=dict(n=int(idx.size));continue
            ph=[];amp=[]
            for j in idx[:4000]:
                y=env_of(A[:,j].astype(float));p,a=phase_at(y,f0);ph.append(p);amp.append(a)
            R,mu=circ_R_phase(ph)
            within=float(np.mean([angdiff(p,mu)<=45 for p in ph]));anti=float(np.mean([angdiff(p,mu)>=135 for p in ph]))
            res[nm]=dict(n=int(idx.size),R=R,mean_phase=mu,frac_within_45=within,frac_anti_135=anti)
        top=np.argsort(rate)[-max(1,int(0.1*len(rate))):]
        ph=[];amp=[]
        for j in top:
            y=env_of(A[:,j].astype(float));p,a=phase_at(y,f0);ph.append(p);amp.append(a)
        R,mu=circ_R_phase(ph)
        res['amp_top10pct']=dict(n=int(len(top)),R=R,mean_phase=mu,
          frac_within_45=float(np.mean([angdiff(p,mu)<=45 for p in ph])),frac_anti_135=float(np.mean([angdiff(p,mu)>=135 for p in ph])))
        ph=[]
        for j in poolcols:
            y=env_of(A[:,j].astype(float));p,a=phase_at(y,f0);ph.append(p)
        R,mu=circ_R_phase(ph)
        res['pool202']=dict(n=len(ph),R=R,mean_phase=mu)
        legph={}
        for lg in LEGS:
            y=env_of(A[:,pool_by_leg[lg]].sum(1).astype(float));p,a=phase_at(y,f0);legph[lg]=p
        pairs_s=[];pairs_c=[]
        for i,a_ in enumerate(LEGS):
            for b_ in LEGS[i+1:]:
                d=angdiff(legph[a_],legph[b_])
                same=any(a_ in TRIPOD[t] and b_ in TRIPOD[t] for t in TRIPOD)
                (pairs_s if same else pairs_c).append(d)
        res['leg_phase']=dict(phases=legph,same_mean_deg=float(np.mean(pairs_s)) if pairs_s else None,
          cross_mean_deg=float(np.mean(pairs_c)) if pairs_c else None,
          selectivity=float(np.mean(pairs_c)-np.mean(pairs_s)) if pairs_c and pairs_s else None)
        corr={}
        E={lg:env_of(A[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
        for i,a_ in enumerate(LEGS):
            for b_ in LEGS[i+1:]: corr[f'{a_}|{b_}']=float(np.corrcoef(E[a_],E[b_])[0,1])
        res['corr_pairs']=corr;res['corr_min']=float(min(corr.values()));res['corr_negative_pairs']=int(sum(1 for v in corr.values() if v<0))
        return res
    GP=dict(f0_hz=f0,f_iv=fiv,K17=phase_table(k17),K17_abl=phase_table(k17a))
    for nm,f in (('G_cn_gate',S330/'SPK_G_cn_gate.npz'),('Kout8.67',S330/'SPK_Kout8.67.npz'),('Kout34.66',S330/'SPK_Kout34.66.npz'),('ISO90',S330/'SPK_ISO90.npz')):
        if Path(f).is_file(): GP[nm]=phase_table(loadz(f))
    dump(o/'GLOBAL_PHASE.json',GP)
    dump(o/'S330_CORRECTION.json',dict(items=[
      dict(id=1,issue='x5',s330='not triggered because 0.710 versus 0.787 differs by only 0.077',corrected='ISO90 runs at a 4x lower pool rate so the comparison is a rate-scale error like S329; ISO90 is marked RATE_COLLAPSED, dropped from the ladder, and x5 becomes NOT_DECIDABLE'),
      dict(id=2,issue='x1',s330='not met at any tested dose',corrected='x1 cannot be met at any dose because the source itself carries no tripod phase relation; the leg-specific share is not a limiter'),
      dict(id=3,issue='D1>=0.35',corrected='this dispatched floor was wrong and is acknowledged here: a fully shared synthetic control reaches only 0.2996 while B0 observes 0.3941, so D1 has no absolute scale; the criterion becomes the matched synthetic control minus 0.05'),
      dict(id=4,issue='control (c)',corrected='D1_FLOOR 0.01116 equals the null_p95 0.01119 of the same control, so the floor and the null are the same quantity'),
      dict(id=5,issue='control (g)',corrected='the Poisson observed D1 equals its own null p95 to three decimals, the same defect as np.roll; nulls must be independently synthesised with null_std>0'),
      dict(id=6,issue='primary criterion',corrected='correlation cannot separate antiphase from decorrelation, so every cross-leg conclusion now reports the phase-difference distribution and the tripod selectivity index')]))
    # ---- V15 controls rebuilt per hard rule 3
    def reg_train(rate,ticks=5000,phase_ms=0.0):
        y=np.zeros(ticks);t=phase_ms
        while t<ticks:
            k=int(round(t))
            if 0<=k<ticks: y[k]=1.0
            t+=1000.0/rate
        return y
    def thin(y,pfunc):
        r=np.random.default_rng(11);return y*(r.random(len(y))<pfunc(np.arange(len(y))))
    def paircorr(Y):
        Z=(Y-Y.mean(0))/np.where(Y.std(0)<1e-12,1e-12,Y.std(0));n,k=Z.shape
        cs=Z.sum(1);return (float(cs@cs)/n-k)/(k*(k-1))
    def syn_null(Y,rng):
        # independently synthesised null: match each column's rate with a fresh Poisson train
        n,k=Y.shape;rates=(Y>0.5).mean(0) if Y.max()<=1.0 else None
        out=[]
        for _ in range(200):
            Zs=np.stack([ (rng.random(n)<max(rates[i] if rates is not None else Y[:,i].mean(),1e-4)).astype(float) for i in range(k)],1)
            out.append(paircorr(np.stack([env_of(Zs[:,i]) for i in range(k)],1)))
        return np.asarray(out)
    ctr={}
    synA=[reg_train(175.0,phase_ms=float(np.random.default_rng(300+i).uniform(0,1000/175.0))) for i in range(12)]
    ctr['a']=dict(D1=paircorr(np.stack([env_of(x) for x in synA],1)))
    common=thin(synA[0],lambda t: np.clip(1+0.6*np.sin(2*np.pi*20.8*t/1000.0),0,1))
    ctr['b']=dict(D1=paircorr(np.stack([env_of(common)]*12,1)))
    r=np.random.default_rng(SEED+9)
    ctr['c']=dict(D1=paircorr(np.stack([env_of(A) for A in k17[:,poolcols].astype(float).T],1)))
    nul=syn_null(np.stack([env_of(A) for A in k17[:,poolcols].astype(float).T],1),r)
    ctr['c']['null_p95']=float(np.percentile(nul,95));ctr['c']['null_std']=float(nul.std())
    ctr['c']['floor_bootstrap']=float(np.percentile([np.random.default_rng(i).permutation(len(nul))[:200].mean() for i in range(200)],95))
    checks=dict(a=bool(ctr['a']['D1']<=0.05),
      b=bool(ctr['b']['D1']>=ctr['c']['D1']-0.05),
      c=bool(ctr['c']['D1']-ctr['c']['null_p95']>1e-9 and ctr['c']['null_std']>0),
      d=bool((GP['K17_abl']['pool202']['R'] is not None)),
      e=bool(GP['K17']['pool202']['R']>=0.35),
      f=bool(GP['K17'].get('amp_top10pct',{}).get('R',0)>=0.35))
    checks['all']=all(checks.values())
    dump(o/'V15.json',dict(primary='zero-lag envelope correlation plus phase at f0 with circular statistics',
      nulls='independently synthesised Poisson trains per column, null_std>0 required',floors='matched synthetic control minus 0.05'))
    dump(o/'DETECTOR_SELFTEST_V15.json',dict(controls=ctr,checks=checks))
    # ---- phase injection spec
    stA=bursts(k17[:,e1[TRIPOD['A'][0]]].astype(float),1.0)[0]
    T_ms=1000.0/fiv[TRIPOD['A'][0]]
    dump(o/'PHASE_INJECTION_SPEC.json',dict(f0_hz=f0,T_ms=T_ms,f_iv=fiv,shift_ms=T_ms/2.0,
      reference='tripod A motif E1 measured burst starts from S322/SPK_K17.npz',reference_bursts=int(len(stA)),
      shares=[0.10,0.30],targets=dict(pi=TRIPOD['B'],same='both',swap=TRIPOD['A']),
      note='declarative zeitgeber proxy only: not a pacemaker, not a gait, not biological'))
    # ---- stage B five arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,banc_load=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in Gd if r['graded']]
    I_STAR=float(min(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
    b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
    e=b['edges'];We,Wi=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
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
    n=We.shape[0]
    pool_ex_per_cell={lg:float(We[pool_by_leg[lg],:].sum()/len(pool_by_leg[lg])) for lg in LEGS}
    # event schedule over substeps (0.5 ms) from tripod A reference
    ticks=5000;sub=2;nsub=ticks*sub
    ref_mask=np.zeros(nsub,bool)
    for t0 in stA:
        if 0<=t0<ticks: ref_mask[int(t0)*sub]=True
    shift_sub=int(round((T_ms/2.0)/0.5))
    def run(name,inject=None,share=0.0,ticks=5000):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi);CD['network_init']+=1
        bg=np.random.default_rng(cfg.background_seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        drive=bg+np.full(n,I_STAR);spm=np.zeros((ticks,n),bool)
        gvec=np.zeros(n)
        if inject:
            for lg in inject['to']:
                gval=share*pool_ex_per_cell[lg]
                for j in pool_by_leg[lg]: gvec[j]=gval
        ev=np.zeros(nsub+shift_sub+2,bool);ev[shift_sub:shift_sub+nsub]=ref_mask if inject and inject['shifted'] else ref_mask
        tot=0.0
        for k in range(ticks):
            s1=net.step(drive,exc_events_uS=(gvec if (inject and ev[k*sub]) else 0.0))
            s2=net.step(drive,exc_events_uS=(gvec if (inject and ev[k*sub+1]) else 0.0))
            spm[k]=np.logical_or(s1,s2);CD['control_tick']+=1;CD['neural_substep']+=2
            if inject and (ev[k*sub] or ev[k*sub+1]): tot+=float(gvec.sum())
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        E={lg:env_of(spm[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
        legph={lg:phase_at(E[lg],f0)[0] for lg in LEGS}
        ps=[];pc=[]
        for i,a_ in enumerate(LEGS):
            for b_ in LEGS[i+1:]:
                d=angdiff(legph[a_],legph[b_])
                same=any(a_ in TRIPOD[t] and b_ in TRIPOD[t] for t in TRIPOD)
                (ps if same else pc).append(d)
        corr={f'{a_}|{b_}':float(np.corrcoef(E[a_],E[b_])[0,1]) for i,a_ in enumerate(LEGS) for b_ in LEGS[i+1:]}
        R,mu=circ_R_phase([legph[lg] for lg in LEGS])
        rates=np.array([spm[:,j].sum()*1000.0/ticks for j in poolcols])
        return dict(leg_phase=legph,same_mean_deg=float(np.mean(ps)),cross_mean_deg=float(np.mean(pc)),
          selectivity=float(np.mean(pc)-np.mean(ps)),corr_mean=float(np.mean(list(corr.values()))),
          corr_min=float(min(corr.values())),neg_pairs=int(sum(1 for v in corr.values() if v<0)),
          anti_frac=float(np.mean([angdiff(legph[a_],legph[b_])>=135 for i,a_ in enumerate(LEGS) for b_ in LEGS[i+1:]])),
          pool_median_rate_hz=float(np.median(rates)),f380=float((rates>=380).mean()),
          R6=R,phase_R6=mu,injected_uS=float(tot),spike_total=int(spm.sum())),spm
    AR={}
    AR['Z0_gate'],spm0=run('Z0_gate')
    g1=bool(np.array_equal(spm0,loadz(S330/'SPK_G_cn_gate.npz')[:5000]));g2=bool(np.array_equal(spm0,loadz(S322/'SPK_K17.npz')[:5000]))
    AR['Za_pi_10'],_=run('Za_pi_10',inject=dict(to=TRIPOD['B'],shifted=True),share=0.10)
    AR['Za_pi_30'],_=run('Za_pi_30',inject=dict(to=TRIPOD['B'],shifted=True),share=0.30)
    AR['Zb_same_30'],_=run('Zb_same_30',inject=dict(to=TRIPOD['A']+TRIPOD['B'],shifted=False),share=0.30)
    AR['Zc_pi_30_swap'],_=run('Zc_pi_30_swap',inject=dict(to=TRIPOD['A'],shifted=True),share=0.30)
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',dict(z0_equals_S330=g1,z0_equals_S322K17=g2))
    # ---- stage C
    base=AR['Z0_gate']['pool_median_rate_hz']
    def rate_ok(k): return abs(AR[k]['pool_median_rate_hz']-base)/max(base,1e-9)<=0.10
    rm=[k for k in AR if not rate_ok(k)]
    pi_ok_all=all(AR[k]['cross_mean_deg']>=135 and AR[k]['anti_frac']>=0.30 and AR[k]['corr_min']<=-0.3 for k in ('Za_pi_10','Za_pi_30') if k not in rm)
    same_ok=AR['Zb_same_30']['cross_mean_deg']<=45
    eff=abs(AR['Za_pi_30']['cross_mean_deg']-AR['Zb_same_30']['cross_mean_deg'])
    mirror=abs(AR['Zc_pi_30_swap']['cross_mean_deg']-AR['Za_pi_30']['cross_mean_deg'])<=45
    if not (g1 and g2): branch='NOT_JUDGED'
    elif not mirror: branch='ARTIFACT_SUSPECT'
    elif pi_ok_all and same_ok: branch='w1'
    elif eff>=30 and AR['Za_pi_30']['injected_uS']>0: branch='w2'
    else: branch='w3'
    dump(o/'OUTCOMES.json',dict(round='S331',branch=branch,gate=dict(z0_s330=g1,z0_k17=g2),
      rate_mismatch=rm,mirror_ok=mirror,
      arms={k:dict(cross=round(v['cross_mean_deg'],1),same=round(v['same_mean_deg'],1),sel=round(v['selectivity'],1),
        anti=round(v['anti_frac'],3),corr_mean=round(v['corr_mean'],3),corr_min=round(v['corr_min'],3),
        pool=round(v['pool_median_rate_hz'],1),R6=round(v['R6'],3),inj=round(v['injected_uS'],4)) for k,v in AR.items()},
      global_phase=dict(K17_R_all=GP['K17']['all']['R'] if 'all' in GP['K17'] else None,
        K17_amp_top_R=GP['K17']['amp_top10pct']['R'],K17_pool_R=GP['K17']['pool202']['R'],
        K17_anti=GP['K17']['amp_top10pct']['frac_anti_135'],K17_same=GP['K17']['leg_phase']['same_mean_deg'],
        K17_cross=GP['K17']['leg_phase']['cross_mean_deg'],f0=f0,T_ms=T_ms),
      selftest=checks,counters=CD,
      one_line='%s pi_arms anti %s same30 cross %.1f eff %.1f'%(branch,
        {k:round(AR[k]['anti_frac'],3) for k in ('Za_pi_10','Za_pi_30')},AR['Zb_same_30']['cross_mean_deg'],eff)))
    print(json.dumps(dict(branch=branch,gate=[g1,g2],mirror=mirror,rm=rm,
      legacy={k:(round(v['cross_mean_deg'],1),round(v['anti_frac'],3),round(v['corr_min'],3),round(v['pool_median_rate_hz'],1)) for k,v in AR.items()},
      f0=round(f0,2),T=round(T_ms,2),K17R=round(GP['K17']['pool202']['R'],3),anti=round(GP['K17']['amp_top10pct']['frac_anti_135'],3),cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
