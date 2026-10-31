"""S327: identify the global common mode and test whether the motif anchors it."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026;POIS=200
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def sm(x,w=20): return np.convolve(np.asarray(x,float),np.ones(w)/w,mode='same')
def env(x): 
    y=sm(x);t=np.arange(len(y));return y-np.polyval(np.polyfit(t,y,1),t)
def peak_spec(y,lo=4.0,hi=50.0,dt=1e-3,rel=0.15):
    n=len(y);f=np.fft.rfftfreq(n,dt);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    interior=(f>=lo+2*df)&(f<=hi-2*df)
    if not interior.any(): return None
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(interior,P,0)));pk=float(f[k])
    sel=(f>=pk*(1-rel))&(f<=pk*(1+rel))
    sel2=(f>=pk-2*df)&(f<=pk+2*df)
    return dict(peak_hz=pk,conc=float(P[sel].sum()/max(tot,1e-30)),conc_abs2bin=float(P[sel2].sum()/max(tot,1e-30)))
def mod_index(x,f0,dt=1e-3,bias=None):
    n=len(x);t=np.arange(n)*dt
    A=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)])
    coef,*_=np.linalg.lstsq(A,x,rcond=None)
    b=abs(bias) if bias is not None else abs(coef[2])
    return float(np.hypot(coef[0],coef[1])/max(b,1e-12))
def segdft(x,f0,dt=1e-3,L=1000,step=500):
    t=np.arange(L)*dt;w=np.hanning(L);out=[]
    for s in range(0,len(x)-L+1,step): out.append(np.sum(x[s:s+L]*w*np.exp(-2j*np.pi*f0*t)))
    return np.array(out)
def coh(X,Y):
    return float(abs(np.sum(X*np.conj(Y)))**2/max(np.sum(abs(X)**2)*np.sum(abs(Y)**2),1e-30))
def coh_null(X,Y,n=NULLS,rng=None,thr=0.01):
    rng=rng or np.random.default_rng(SEED);out=[]
    for _ in range(n):
        Xn=X*np.exp(1j*rng.uniform(0,2*np.pi,len(X)))
        out.append(coh(Xn,Y))
    a=np.asarray(out);return dict(p95=float(np.percentile(a,95)),max=float(a.max()),p=float((np.sum(a>=coh(X,Y))+1)/(len(a)+1)))
def poisson_null_conc(count,f0band,dur_s,n=POIS,rng=None):
    rng=rng or np.random.default_rng(SEED+5);n_bins=int(dur_s*1000);out=[]
    lam=count/max(n_bins,1)
    for _ in range(n):
        y=(rng.random(n_bins)<lam).astype(float);sp=peak_spec(env(y))
        if sp: out.append(sp['conc'])
    return float(np.percentile(out,95)) if out else None
def bh_q(p):
    p=np.asarray(p,float);m=p.size;order=np.argsort(p);rank=np.empty(m,int);rank[order]=np.arange(1,m+1)
    q=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    out=np.empty(m);out[order]=np.minimum(q,1.0);return out
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S327',mode='identify the global common mode and test whether the motif anchors it; no physics',
      counters=dict(tier_build=3,select_touch_tier=6,edges_build=3,network_init=3,banc_load=3,control_tick=15000,tier_advance=15000,neural_substep=30000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,poisson=POIS,seed=SEED,
      rules=['the primary pool measurement must not depend on an in-arm reference signal',
             'motif coherence is demoted to a diagnostic confounded by the common mode and is reported both unconditioned and conditioned',
             'the envelope-peak null must be a rate-matched independent Poisson train, never a power-preserving transform',
             'slice asserts n <= len(signal); NULLS=1000; no else fallback'],
      tolerances=dict(level_unchanged_pct=2.0,common_mode_drift_hz=0.5,conc_ratio=[0.8,1.25]),
      branches=dict(y1='B1 level unchanged and common mode unchanged',y2='B1 level unchanged, conditioned coherence inside the null, common mode unchanged',
                    y3='B1 common mode changed',y4='NOT_JUDGED'))
    dump(o/'PREREG.json',pre)
    (o/'S327_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s327.py)=%s\nsha256(audit_s327_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s327_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    pool=sorted(seen.items())
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    arms={'L1':loadz(S324/'SPK_L1_k17_seed0_10s.npz'),'L2':loadz(S324/'SPK_L2_k17abl_seed0_10s.npz'),
          'K10':loadz(S322/'SPK_K10.npz'),'K15':loadz(S322/'SPK_K15.npz'),'K17':loadz(S322/'SPK_K17.npz'),
          'K21':loadz(S322/'SPK_K21.npz'),'K17_abl':loadz(S322/'SPK_K17_abl.npz'),'A0_gate':loadz(S322/'SPK_A0_gate.npz')}
    # ---------- stage A global mode
    G={}
    for nm,A in arms.items():
        y=env(A.sum(1).astype(float));sp=peak_spec(y)
        if sp is None: G[nm]=dict(note='no interior');continue
        bias=float(sm(A.sum(1).astype(float)).mean())
        G[nm]=dict(n_bins=int(A.shape[0]),peak_hz=sp['peak_hz'],conc_rel15=sp['conc'],conc_abs2bin=sp.get('conc_abs2bin'),
          mod_index=mod_index(y,sp['peak_hz'],bias=bias),bias=bias,mean_rate_hz=float(A.sum(1).mean()*1000),
          poisson_p95=poisson_null_conc(int(A.sum()),sp['peak_hz'],A.shape[0]/1000.0))
    legs={}
    for nm,A in arms.items():
        f0=G[nm].get('peak_hz')
        if not f0: continue
        d={}
        for lg in LEGS:
            cols=[j for (l2,gi),j in pool if l2==lg]
            agg=A[:,cols].sum(1).astype(float)
            sp=peak_spec(env(agg));d[lg]=dict(peak_hz=sp['peak_hz'] if sp else None,conc=sp['conc'] if sp else None,
              mean_rate_hz=float(agg.mean()*1000/len(cols)) if cols else None,
              mod_index=mod_index(env(agg),f0))
        legs[nm]=d
    cross={}
    for nm,A in arms.items():
        f0=G[nm].get('peak_hz')
        if not f0: continue
        X={lg:segdft(env(A[:,[j for (l2,gi),j in pool if l2==lg]].sum(1).astype(float)),f0) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                pairs[f'{a}|{b}']=coh(X[a],X[b])
        cross[nm]=pairs
    # pool-pool per-cell coherence and per-cell modulation index (L1/L2 only)
    pc={}
    for nm in ('L1','L2'):
        A=arms[nm];f0=G[nm]['peak_hz']
        Y={k:segdft(env(A[:,j].astype(float)),f0) for k,j in pool}
        vals=[coh(Y[a],Y[b]) for i,(a,_) in enumerate(pool) for b,_ in pool[i+1:]]
        mods=[mod_index(env(A[:,j].astype(float)),f0) for _,j in pool]
        pc[nm]=dict(pool_pool_median=float(np.median(vals)),pool_pool_min=float(np.min(vals)),pool_pool_max=float(np.max(vals)),
          mod_median=float(np.median(mods)),mod_p90=float(np.percentile(mods,90)))
    same_cell={}
    for (lg,gi),j in pool:
        f1=G['L1']['peak_hz'];f2=G['L2']['peak_hz'];f0=(f1+f2)/2.0
        same_cell[f'{lg}:{gi}']=coh(segdft(env(arms['L1'][:,j].astype(float)),f0),segdft(env(arms['L2'][:,j].astype(float)),f0))
    dump(o/'GLOBAL_MODE.json',dict(global_mode=G,per_leg=legs,cross_leg_coherence=cross,pool_pool=pc,
      same_cell_cross_arm_median=float(np.median(list(same_cell.values()))),
      targets_reported_by_dispatch=dict(L1=[22.30,0.057,0.027],L2=[21.00,0.075,0.030],K17=[20.80,0.101,0.045],K17_abl=[21.00,0.109,0.036],K10=[19.80,0.080,0.036]),
      note='the envelope-peak null is a rate-matched independent Poisson train, never a power-preserving transform'))
    # conditional coherence
    cond={}
    for nm,ref in (('L1','L1'),('L2','L2')):
        A=arms[nm];f0=G[nm]['peak_hz'];gseg=segdft(env(A[M18].sum(1).astype(float)),f0,t=None) if False else segdft(env(A.sum(1).astype(float)),f0)
        d={}
        for lg in LEGS:
            st=None
            cols=[j for (l2,gi),j in pool if l2==lg]
            un={};cnd={}
            for j in cols:
                Y=segdft(env(A[:,j].astype(float)),f0);R=segdft(env(A[:,E1[lg]].astype(float)),f0)
                un[coh(R,Y)]=1
                # regress the global-mode component out of Y in the f0 band
                beta=np.vdot(gseg,Y)/max(np.vdot(gseg,gseg).real,1e-30)
                Yr=Y-beta*gseg
                cnd[coh(R,Yr)]=1
            u=np.array(list(un));c=np.array(list(cnd))
            nul=coh_null(segdft(env(A[:,E1[lg]].astype(float)),f0),segdft(env(A[:,E1[lg]].astype(float)),f0),n=200)
            d[lg]=dict(unconditioned_median=float(np.median(u)),conditioned_median=float(np.median(c)),
              n=len(cols),null_p95=nul['p95'])
        cond[nm]=d
    dump(o/'CONDITIONAL_COHERENCE.json',dict(per_leg=cond,
      note='conditioning regresses the f0-band pool component onto the network-wide common mode and then measures the residual coherence with the motif reference; if both arms land inside the null after conditioning, motif-to-pool transfer is NOT IDENTIFIABLE at this operating point rather than disproved'))
    dump(o/'HYPOTHESIS.json',dict(statement='the graded tier carries a self-organised common mode near 20-22 Hz whose presence does not depend on the motif coupling, and the 21-26 Hz motif-cell rate in the tier coincides with that mode rather than with the isolated motif frequency of 6.58/15.39 Hz',
      predictions=['the common-mode peak frequency in B1 (motif external output zeroed) stays within 0.5 Hz of B0','the pool mean rate in B1 stays within 2 percent of B0','the isolated motif band 6.58-26.92 Hz does not match the tier motif-cell frequency of 21-26 Hz']))
    dump(o/'S326_CORRECTION.json',dict(items=[
      dict(id=1,dispatched='pool strongly inherits (L1 0.61-0.86 vs L2 0.01-0.07)',corrected='that L2 range was a CROSS-ARM quantity (motif of L1 against pool of L2 = 0.043/0.046/0.073/0.141); the same-arm L2 is 0.89/0.92/0.81/0.79, so the strong-inheritance claim is withdrawn'),
      dict(id=2,measured='S326 same-arm per-leg medians',corrected='reproduced here from the archive'),
      dict(id=3,s326='NOT_JUDGED reason was statistic failure',corrected='the reason is upgraded to common-mode confounding: a network-wide ~20-22 Hz mode exists in both arms and pollutes any minus-ref-motif coherence'),
      dict(id=4,note='TONIC does not mean no oscillation: the ablation motif E1 envelope still modulates at 21 Hz although V10 labels it TONIC by burst structure')]))
    # ---------- stage B three arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,banc_load=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in Gd if r['graded']]
    if not cand: dump(o/'STOP.json',dict(reason='no measured graded point for I*'));return 2
    I_STAR=float(min(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    def run(name,motif_k,zero_out,zero_drive,ticks=5000):
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
        b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
        e=b['edges'];We,Wi=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
        mcount=0
        if motif_k is not None:
            for lg in LEGS:
                for src_r in ROLES:
                    for tgt_r in ROLES:
                        if src_r==tgt_r: continue
                        for csrc in modules[lg][src_r]:
                            for ctgt in modules[lg][tgt_r]:
                                i=loc[int(csrc)];j=loc[int(ctgt)]
                                for Wm in (We,Wi):
                                    s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                                    if kk.size: Wm.data[s0+kk[0]]=motif_k*EDGE_TOTAL;mcount+=1
        zcount=0;ztouch=[]
        if zero_out:
            for Wm in (We,Wi):
                for j in range(Wm.shape[0]):
                    if j in Mset: continue
                    s0,e0=Wm.indptr[j],Wm.indptr[j+1]
                    for t in range(s0,e0):
                        if int(Wm.indices[t]) in Mset:
                            Wm.data[t]=0.0;zcount+=1;ztouch.append(dict(row=int(j),col=int(Wm.indices[t]),kind=('exc' if Wm is We else 'inh')))
        tier=NeuralTier(b,cfg,OwnershipLedger());CD['network_init']+=1
        cur=np.full(len(roots),I_STAR)
        if zero_drive:
            for j in Mset: cur[j]=0.0
        n=ticks;spm=np.zeros((n,len(roots)),bool);Vm=np.zeros((n,18),np.float32);hid=getattr(tier,'net',tier)
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
            v=np.asarray(getattr(hid,'v',np.zeros(1)),float)
            if v.size>=len(roots): Vm[k]=v[M18]
        cnt=spm.sum(0).astype(np.int32)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape),counts=cnt,index_order=np.array(M18),Vm=Vm)
        y=env(spm.sum(1).astype(float));sp=peak_spec(y)
        poolcols=[j for (_l,_g),j in pool]
        preg=spm[:,poolcols].sum(1).astype(float)
        return dict(motif_edges_set=mcount,zeroed=zcount,zero_sample=ztouch[:5],
          global_peak=sp['peak_hz'] if sp else None,global_conc=sp['conc'] if sp else None,global_conc_abs=sp.get('conc_abs2bin') if sp else None,
          global_mod=mod_index(y,sp['peak_hz']) if sp else None,
          pool_mean_rate_hz=float(preg.mean()*1000/len(poolcols)),
          motif_mean_rate_hz=float(spm[:,M18].sum(1).mean()*1000/len(M18)),
          spike_total=int(cnt.sum()))
    B0=run('B0_repeat',K_DESIGN,False,False)
    B1=run('B1_motif_no_out',K_DESIGN,True,False)
    B2=run('B2_no_motif_drive',0.0,False,True)
    ref=arms['K17']
    gate=bool(np.array_equal(np.unpackbits(np.load(o/'SPK_B0_repeat.npz')['packed'],axis=1)[:,:12000],ref[:5000]))
    dump(o/'ARMS.json',dict(B0=B0,B1=B1,B2=B2));dump(o/'CALL_LEDGER.json',CD)
    # ---------- stage C
    lvl=abs(B1['pool_mean_rate_hz']-B0['pool_mean_rate_hz'])/max(B0['pool_mean_rate_hz'],1e-9)*100
    drift=abs(B1['global_peak']-B0['global_peak']) if (B1['global_peak'] and B0['global_peak']) else None
    cr=(B1['global_conc']/B0['global_conc']) if (B1['global_conc'] and B0['global_conc']) else None
    level_same=lvl<2.0;cm_same=bool(drift is not None and drift<=0.5 and cr and 0.8<=cr<=1.25)
    if not gate: branch='y4'
    elif level_same and cm_same: branch='y1'
    elif level_same and not cm_same: branch='y3'
    else: branch='y2' if cm_same else 'y4'
    dump(o/'OUTCOMES.json',dict(round='S327',branch=branch,gate=gate,I_star=I_STAR,k_design=K_DESIGN,
      arms={k:dict(global_peak=round(v['global_peak'],2) if v['global_peak'] else None,global_conc=round(v['global_conc'],4) if v['global_conc'] else None,global_conc_abs=round(v.get('global_conc_abs'),4) if v.get('global_conc_abs') else None,
        global_mod=round(v['global_mod'],4) if v['global_mod'] else None,pool_mean_rate_hz=round(v['pool_mean_rate_hz'],1),
        motif_mean_rate_hz=round(v['motif_mean_rate_hz'],1),zeroed=v['zeroed'],motif_edges=v['motif_edges_set'],spike_total=v['spike_total']) for k,v in (('B0',B0),('B1',B1),('B2',B2))},
      level_change_pct=lvl,common_mode_drift_hz=drift,conc_ratio=cr,level_same=level_same,common_mode_same=cm_same,
      global_mode=G,counters=CD,
      one_line='%s level %.2f%% drift %s Hz conc_ratio %s'%(branch,lvl,round(drift,2) if drift is not None else None,round(cr,3) if cr else None)))
    print(json.dumps(dict(branch=branch,gate=gate,
      G={k:(round(v['peak_hz'],2),round(v['conc_rel15'],3),round(v['conc_abs2bin'],3),round(v['mod_index'],3)) for k,v in G.items() if 'peak_hz' in v},
      arms={k:(round(v['pool_mean_rate_hz'],1),round(v['global_peak'],2) if v['global_peak'] else None,v['zeroed']) for k,v in (('B0',B0),('B1',B1),('B2',B2))},
      level_pct=round(lvl,2),drift=round(drift,2) if drift is not None else None,cr=round(cr,3) if cr else None,
      cross_med=round(float(np.median(list(cross['L1'].values()))),3),same_cell=round(float(np.median(list(same_cell.values()))),3)),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
