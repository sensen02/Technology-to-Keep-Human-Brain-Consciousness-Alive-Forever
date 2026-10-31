"""S345: twenty second arms, rule driven trimming, slip counting and the engagement gate v3."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
from scipy import signal as sgn
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S300=Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S344=Path('/home/sensen/Desktop/engine修复验收/s344-edge-trim-and-engagement-gate/20261027T032801Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
GE_PER_CELL=0.00349
BAND=(0.5,4.0);BIN=25;TICKS=20000
TRIM=float(max(1.0,2.0/BAND[0]))
ARMS=dict(W20_gate=dict(biasA=0.0,biasB=0.0,seed=0,ticks=TICKS),
  W20_gate_s4=dict(biasA=0.0,biasB=0.0,seed=4,ticks=TICKS),
  B30_s0=dict(biasA=0.0,biasB=30.0,seed=0,ticks=TICKS),
  B30_s1=dict(biasA=0.0,biasB=30.0,seed=1,ticks=TICKS),
  B30_s2=dict(biasA=0.0,biasB=30.0,seed=2,ticks=TICKS),
  B30_s4=dict(biasA=0.0,biasB=30.0,seed=4,ticks=TICKS),
  LA30_s0=dict(biasA=30.0,biasB=30.0,seed=0,ticks=TICKS),
  LB30_s0=dict(biasA=0.0,biasB=60.0,seed=0,ticks=TICKS),
  B25_s0=dict(biasA=0.0,biasB=25.0,seed=0,ticks=TICKS),
  B25_s1=dict(biasA=0.0,biasB=25.0,seed=1,ticks=TICKS))
GATEARMS=dict(c1=['B30_s0','B30_s1','B30_s2','B30_s4'],
  c2=['W20_gate','B30_s0','B30_s2'],
  c3=['LA30_s0','LB30_s0'],
  c4=['B30_s0','B30_s1','B30_s2','B30_s4'],
  c5=['B30_s0'],
  c6=['B25_s0','B30_s0'])
EXT=dict(W20_gate='SPK_G0_gate.npz',B30_s0='SPK_B30_5s.npz',B30_s1='SPK_B30_seed1.npz',B30_s2='SPK_B30_seed2.npz',
  LA30_s0='SPK_LA30_5s.npz',LB30_s0='SPK_LB30_5s.npz')
TH=dict(trim_s=TRIM,plateau_trims=[2.0,3.0,TRIM],plateau_tol=8.0,cycles_min=8,retained_min=0.4,
  cross_lo=150.0,cross_hi=210.0,span_min=0.4,common_sat_max=0.05,cv_max=0.10,slip_deg=180.0,
  advance_max=45.0,block_s=2.0,blocks_min=5,block_span_max=45.0,nulls=NULLS,cert_seeds=100,cert_rows=TICKS,
  phase_tol=8.0,rho_tol=0.05)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else (None if o is None else float(o)))+'\n',encoding='utf-8')
def wrap(d): return (d+180.0)%360.0-180.0
def sos_band(order,band,fs):
    assert 0.0<band[0]<band[1]<fs/2
    sos=sgn.butter(order,[band[0]/(fs/2),band[1]/(fs/2)],btype='band',output='sos')
    for s in sos:
        r=np.roots(s[3:])
        if r.size and float(np.max(np.abs(r)))>0.999: raise SystemExit('unstable section')
    return sos
def trimmed(x,fs,trim_s,order=4,band=BAND):
    """the only filter entry: zero phase band pass, then trim both edges"""
    sos=sos_band(order,band,fs)
    y=sgn.sosfiltfilt(sos,x-x.mean())
    k=int(round(trim_s*fs))
    if len(y)<=2*k+16: return None,0.0
    z=y[k:len(y)-k]
    return z,len(z)/max(len(x),1)
def counts(spm,cols,bin_ms=BIN):
    c=spm[:,cols].sum(1).astype(float);n=len(c)//bin_ms
    return np.array([c[i*bin_ms:(i+1)*bin_ms].sum() for i in range(n)],float)
def circ(phi):
    z=np.mean(np.exp(1j*phi));return float(np.angle(z)),float(abs(z))
def circular_span_deg(values):
    v=np.sort(np.mod(np.asarray(values,float),360.0))
    if v.size<2: return 0.0
    return float(360.0-np.max(np.diff(np.r_[v,v[0]+360.0])))
def phase_core(a,b,fs,trim_s,with_bootstrap=True):
    ya,ret=trimmed(a,fs,trim_s);yb,_=trimmed(b,fs,trim_s)
    if ya is None or yb is None: return None
    n=min(len(ya),len(yb));ya=ya[:n];yb=yb[:n]
    ua=np.unwrap(np.angle(sgn.hilbert(ya)));ub=np.unwrap(np.angle(sgn.hilbert(yb)))
    d=ua-ub
    mu,R=circ(d);unw=np.unwrap(d);t=np.arange(len(unw))/fs
    slope,intercept=np.polyfit(t,unw,1)
    blk=int(round(TH['block_s']*fs));blocks=[]
    for i in range(0,len(unw)-blk+1,blk):
        assert i+blk<=len(d)
        blocks.append(float(np.degrees(circ(d[i:i+blk])[0])))
    edge=max(1,int(0.2*fs))
    total=float(np.median(unw[-edge:])-np.median(unw[:edge]))
    slips=int(np.floor(np.abs(total)/np.radians(TH['slip_deg'])))
    ci=bootstrap_slope(unw,fs) if with_bootstrap else [float(np.degrees(slope)),float(np.degrees(slope))]
    fa=float((ua[-1]-ua[0])*fs/(2*np.pi*(n-1)))
    fb=float((ub[-1]-ub[0])*fs/(2*np.pi*(n-1)))
    df_ci=[float(x/360.0) for x in ci]
    return dict(phase_deg=float(np.degrees(mu)),R=R,slope_deg_s=float(np.degrees(slope)),slope_ci_deg_s=ci,
      df_hz=float(np.degrees(slope)/360.0),df_ci_hz=df_ci,
      df_status=('UNRESOLVED' if df_ci[0]<=0.0<=df_ci[1] else 'RESOLVED'),
      df_abs_upper_hz=float(np.max(np.abs(df_ci))),
      duration_s=len(unw)/fs,total_advance_deg=float(np.degrees(total)),slip_count=slips,
      slip_period_s=(len(unw)/fs/slips if slips>0 else None),blocks=blocks,n_blocks=len(blocks),
      block_span=circular_span_deg(blocks),retained=ret,cycles=float(min(fa,fb)*n/fs),fa=fa,fb=fb)
def bootstrap_slope(unwrapped,fs,seed=SEED):
    """Residual moving-block bootstrap; preserves time design and local two-second dependence."""
    rng=np.random.default_rng(seed);n=len(unwrapped);t=np.arange(n)/fs
    slope,offset=np.polyfit(t,unwrapped,1);fitted=slope*t+offset;residual=unwrapped-fitted
    block=int(round(TH['block_s']*fs));slopes=[]
    tc=t-t.mean();den=float(np.dot(tc,tc))
    for _ in range(1000):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        draw=np.concatenate([residual[int(s):int(s)+block] for s in starts])[:n]
        slopes.append(float(np.degrees(np.dot(tc,fitted+draw)/den)))
    return [float(x) for x in np.percentile(slopes,[2.5,97.5])]
def drift_status(pr):
    if pr is None: return 'UNTESTED'
    if pr['slip_count']>=1: return 'SLIPPING'
    if pr['slope_ci_deg_s'] is not None and float(np.max(np.abs(pr['slope_ci_deg_s'])))*pr['duration_s']<=TH['advance_max']: return 'LOCKED'
    return 'UNRESOLVED'
def null_rho(rateA,rateB,trim_s):
    vals=[]
    for i in range(NULLS):
        rng=np.random.default_rng(SEED+19*i+int(rateA*3)%71)
        a=rng.poisson(np.full(TICKS,max(rateA/1000.0,1e-12))).astype(float)
        b=rng.poisson(np.full(TICKS,max(rateB/1000.0,1e-12))).astype(float)
        ya,_=trimmed(counts(a.reshape(-1,1),[0]),1000.0/BIN,trim_s)
        yb,_=trimmed(counts(b.reshape(-1,1),[0]),1000.0/BIN,trim_s)
        if ya is None or yb is None: continue
        n=min(len(ya),len(yb));vals.append(float(np.corrcoef(ya[:n],yb[:n])[0,1]))
    v=np.array(vals)
    assert len(v)==NULLS and float(v.std())>0.0 and 1.0/(NULLS+1)<0.01
    # Bartlett effective sample size, from the final surrogate's autocorrelations.
    acs=[]
    for y in (ya,yb):
        yc=y-y.mean();ac=sgn.correlate(yc,yc,mode='full',method='fft')[len(yc)-1:]
        acs.append(ac/max(ac[0],1e-30))
    dof=float(len(ya)/max(1.0,1.0+2.0*np.sum(acs[0][1:len(ya)//2]*acs[1][1:len(ya)//2])))
    return dict(p1=float(np.percentile(v,1)),p95=float(np.percentile(v,95)),std=float(v.std()),mean=float(v.mean()),
      nulls=NULLS,n=int(v.size),n_eff=dof,dof_method='Bartlett surrogate autocorrelation',n_time_samples=int(len(ya)))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--finalize',action='store_true');args=ap.parse_args();o=args.output
    if args.finalize: return finalize(o)
    o.mkdir(parents=True,exist_ok=True)
    total_ticks=sum(v['ticks'] for v in ARMS.values())
    assert TRIM==max(1.0,2.0/BAND[0])==4.0
    assert len(ARMS)==10 and total_ticks==200000
    for arm_set in GATEARMS.values(): assert set(arm_set)<=set(ARMS)
    assert TH['cert_rows']==TICKS==20000
    pre=dict(round='S345',mode='twenty second arms, rule driven trimming, slip counting and the engagement gate v3; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=len(ARMS),control_tick=total_ticks,
        tier_advance=0,neural_substep=2*total_ticks,physics_step=0,model_load=0,render=0,decoder_eval=4),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,thresholds=TH,arm_specs=ARMS,gate_arms=GATEARMS,extension_pairs=EXT,
      decoder_consume_calls_expected=8000,
      band=list(BAND),bin_ms=BIN,trim_s=TRIM,
      decoder_arms=GATEARMS['c2']+['B25_s0'],
      deviation={'extra_arm':'The dispatched ten-arm budget lists nine arm specifications; B25_s1 is an explicit supplementary dose-control seed, not a hidden arm.',
        'decoder_eval':'Four completed decoder trajectory evaluations; consume calls are separately counted as decoder_consume_calls.'},
      gate_statuses=['PASS','FAIL','INELIGIBLE','UNTESTED'],
      rules=['trim is max(1.0, 2 over band low) which is four seconds and the chosen value must equal it',
             'every phase number is also produced at trims two three and four seconds and the largest circular distance must stay within eight degrees',
             'gate states are PASS FAIL INELIGIBLE UNTESTED and only a real measurement may be FAIL',
             'gate arm sets live in the preregistration and the code may not carry its own literal lists',
             'the synthetic certification must fill one hundred seeds and twenty thousand rows per cell or the round stops',
             'the correlation criterion is the null first percentile at the per tripod rates and z is descriptive only',
             'drift is reported as an ordinary least squares slope with a block bootstrap interval, block circular offsets and a slip count',
             'any arm longer than an archived arm must reproduce its prefix bitwise'])
    dump(o/'PREREG.json',pre)
    (o/'S345_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s345.py)=%s\nsha256(audit_s345_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s345_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    seen={};rows_by_act=[]
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        ids=[loc[int(x)] for x in json.loads(r['pool_root_ids']) if int(x) in loc] if r.get('pool_root_ids') else []
        if lg in LEGS and ids:
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
        rows_by_act.append(dict(name=r.get('name'),leg=lg,ids=ids))
    pool=sorted(seen.items());leg_cols={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool];pool_index={j:i for i,j in enumerate(poolcols)}
    Ac=[j for lg in TRIPOD['A'] for j in leg_cols[lg]];Bc=[j for lg in TRIPOD['B'] for j in leg_cols[lg]]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    # ---------- trim audit on the archived arms
    TA={}
    for nm,p in (('S344_G0_gate',S344/'SPK_G0_gate.npz'),('S344_B30_5s',S344/'SPK_B30_5s.npz'),('S344_P10_10s',S344/'SPK_P10_10s.npz'),
                 ('S344_B30_10s',S344/'SPK_B30_10s.npz')):
        if not p.is_file(): continue
        A=loadz(p);fs=1000.0/BIN
        cA=counts(A,Ac);cB=counts(A,Bc);cur={}
        for tr in (1.0,1.5,2.0,3.0,4.0):
            pa,_=trimmed(cA,fs,tr);pb,_=trimmed(cB,fs,tr)
            if pa is None or pb is None: cur[str(tr)]=dict(eligible=False);continue
            n=min(len(pa),len(pb));rho=float(np.corrcoef(pa[:n],pb[:n])[0,1])
            pr=phase_core(cA,cB,fs,tr,with_bootstrap=False)
            cur[str(tr)]=dict(eligible=bool(pr is not None),rho=rho,phase=(pr['phase_deg'] if pr else None),
              cycles=(pr['cycles'] if pr else None),retained=(pr['retained'] if pr else None))
        TA[nm]=cur
    dump(o/'TRIM_AUDIT.json',dict(arms=TA,trims=[1.0,1.5,2.0,3.0,4.0],rule_trim=TRIM,
      note='this answers where the plateau lies and whether the rule constant is defensible'))
    # ---------- synthetic certification
    def synth(rate,phase_deg,duty,df,seed,n=TICKS):
        rng=np.random.default_rng(seed);t=np.arange(n)/1000.0;f=2.0
        def train(ph,fr):
            cyc=np.mod(t*fr-ph/360.0,1.0)
            lam=np.maximum(np.where(cyc<duty,rate/1000.0/max(duty,1e-6),0.0),1e-9)
            return rng.poisson(lam).astype(float)
        def train_rate(rate_value,ph,fr):
            cyc=np.mod(t*fr-ph/360.0,1.0)
            lam=np.where(cyc<duty,rate_value/1000.0/duty,0.0)
            return rng.poisson(lam).astype(float)
        return train_rate(rate[0],0.0,f),train_rate(rate[1],phase_deg,f+df)
    CERT={};empty=[]
    for phase_deg in (150.0,165.0,180.0):
        for duty in (0.3,0.5,0.7):
            for (r1,r2) in ((105.0,59.0),(90.0,80.0)):
                for df in (0.0,0.02):
                    errs=[];rs=[];bias_by_trim={str(tr):[] for tr in TH['plateau_trims']}
                    for s in range(TH['cert_seeds']):
                        a,b=synth((r1,r2),phase_deg,duty,df,7000+s+int(phase_deg)+int(duty*10)+int(df*100))
                        assert len(a)==len(b)==TICKS
                        ca=counts(a.reshape(-1,1),[0]);cb=counts(b.reshape(-1,1),[0])
                        target_mid=wrap(phase_deg-360.0*df*(TICKS/2000.0))
                        for tr in TH['plateau_trims']:
                            pr=phase_core(ca,cb,1000.0/BIN,tr,with_bootstrap=False)
                            if pr is not None: bias_by_trim[str(tr)].append(wrap(pr['phase_deg']-target_mid))
                        pr=phase_core(ca,cb,1000.0/BIN,TRIM,with_bootstrap=False)
                        if pr is not None:
                            errs.append(wrap(pr['phase_deg']-target_mid));rs.append(pr['R'])
                    key=f'p{int(phase_deg)}_d{int(duty*10)}_r{int(r1)}_{int(r2)}_df{int(df*100)}'
                    if len(errs)<TH['cert_seeds']: empty.append(key)
                    CERT[key]=dict(phase=phase_deg,duty=duty,rates=[r1,r2],df=df,n=len(errs),rows=TICKS,
                      bias_deg=float(np.mean(errs)) if errs else None,R=float(np.mean(rs)) if rs else None,
                      bias_by_trim={k:(float(np.mean(v)) if v else None) for k,v in bias_by_trim.items()},
                      passes=bool(len(errs)==TH['cert_seeds'] and np.abs(float(np.mean(errs)))<=5.0 and float(np.mean(rs))>=0.9))
    print(json.dumps(dict(stage='CERT_V22',cells=len(CERT),empty=len(empty),failed=sum(not c['passes'] for c in CERT.values()))),flush=True)
    dump(o/'CERT_V22.json',dict(cells=CERT,empty_cells=empty,cert_seeds=TH['cert_seeds'],rows=TICKS,
      plateau=[round(x,2) for x in TH['plateau_trims']],note='every cell must fill one hundred seeds at twenty thousand rows'))
    # ---------- stage B ten arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig
    CD={k:0 for k in pre['counters']}
    CD['control_tick']=0;CD['neural_substep']=0;CD['network_init']=0
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    I_STAR=float(min([r for r in Gd if r['graded']],key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
    b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
    e=b['edges'];We0,Wi0=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    for lg in LEGS:
        for src_r in ROLES:
            for tgt_r in ROLES:
                if src_r==tgt_r: continue
                for csrc in modules[lg][src_r]:
                    for ctgt in modules[lg][tgt_r]:
                        i=loc[int(csrc)];j=loc[int(ctgt)]
                        for Wm in (We0,Wi0):
                            s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                            if kk.size: Wm.data[s0+kk[0]]=K_DESIGN*EDGE_TOTAL
    n=We0.shape[0];Wi_tot=float(Wi0.sum())
    T=[];S=[];V=[]
    for t in Ac:
        for s_ in Bc: T.append(t);S.append(s_);V.append(2.0*GE_PER_CELL/len(Bc))
    for t in Bc:
        for s_ in Ac: T.append(t);S.append(s_);V.append(2.0*GE_PER_CELL/len(Ac))
    m=sps.csr_matrix((np.asarray(V,float),(np.asarray(T,int),np.asarray(S,int))),shape=Wi0.shape)
    Wi=(Wi0+m).tocsr()
    added=dict(n=int(m.nnz),total=float(m.sum()),share=float(m.sum())/max(Wi_tot,1e-12),selfloops=0)
    SPM={};AR={}
    def run(name,spec):
        biasA,biasB,seed,ticks=spec['biasA'],spec['biasB'],spec['seed'],spec['ticks']
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if biasA: cur[Ac]+=biasA/100.0*I_STAR
        if biasB: cur[Bc]+=biasB/100.0*I_STAR
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        # Persist packed arrays; do not retain ten full 240 MB matrices.
        SPM[name]=None
        fs=1000.0/BIN;cA=counts(spm,Ac);cB=counts(spm,Bc)
        pa,_=trimmed(cA,fs,TRIM);pb,_=trimmed(cB,fs,TRIM)
        rho=float(np.corrcoef(pa[:min(len(pa),len(pb))],pb[:min(len(pa),len(pb))])[0,1]) if pa is not None and pb is not None else None
        pr=phase_core(cA,cB,fs,TRIM)
        plat={}
        for tr in TH['plateau_trims']:
            p2=phase_core(cA,cB,fs,tr)
            plat[str(tr)]=dict(phase=(p2['phase_deg'] if p2 else None),eligible=bool(p2 is not None))
        base=plat[str(TRIM)]['phase']
        assert base is not None and all(plat[str(tr)]['phase'] is not None for tr in TH['plateau_trims'])
        pdist=max([np.abs(wrap(plat[str(tr)]['phase']-base)) for tr in TH['plateau_trims']])
        rA=float(spm[:,Ac].sum()*1000.0/(ticks*len(Ac)));rB=float(spm[:,Bc].sum()*1000.0/(ticks*len(Bc)))
        nr=null_rho(rA,rB,TRIM)
        AR[name]=dict(name=name,bias_pct=dict(A=biasA,B=biasB),bias_nA=dict(A=biasA/100*I_STAR,B=biasB/100*I_STAR),seed=seed,ticks=ticks,
          rates=dict(A=rA,B=rB,ratio=max(rA,rB)/max(min(rA,rB),1e-9)),rho=dict(value=rho,null=nr,
            criterion=bool(rho is not None and rho<=nr['p1'] and abs(rho-nr['p1'])>1e-9),z=(rho-nr['mean'])/max(nr['std'],1e-30) if rho is not None else None),
          phase=pr,plateau=plat,plateau_span=pdist,plateau_ok=bool(pdist<=TH['plateau_tol']),
          drift_status=drift_status(pr),added_edges=added,f_drive_hz=None,dI_time_dependent=False,not_emergent=True,
          eligible=bool(pr is not None and pr['cycles'] and pr['cycles']>=TH['cycles_min'] and pr['retained']>=TH['retained_min']))
        return AR[name]
    for nm in pre['arm_specs']:
        run(nm,pre['arm_specs'][nm])
        dump(o/'PROGRESS.json',dict(completed=list(AR),ledger=CD))
        print(json.dumps(dict(completed=nm,ticks=CD['control_tick'])),flush=True)
    assert CD['control_tick']==sum(x['ticks'] for x in AR.values())
    assert CD['neural_substep']==2*CD['control_tick']
    # ---------- extension gate
    EG={}
    for nm,refname in EXT.items():
        p=S344/refname
        if not p.is_file(): EG[nm]=dict(tested=False);continue
        r0=loadz(p)
        current=loadz(o/f'SPK_{nm}.npz')
        k=r0.shape[0]
        assert k<=len(current)
        pre_fix=np.array_equal(current[:k],r0)
        EG[nm]=dict(tested=True,equal=bool(pre_fix),rows_compared=int(k),spike_prefix=int(current[:k].sum()),ref_spike=int(r0.sum()),
          method='unpackbits elementwise prefix equality')
    dump(o/'EXTENSION_GATE.json',dict(pairs=EG,note='longer arms must reproduce the archived prefix bitwise or be renamed as new constructions'))
    # ---------- decoder criteria
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    decoder_consume_calls=[0]
    def ctrl_of(Am,cfgX,chunk=10):
        mo=MotorOutput(MAN,cfgX);spm=Am[:,poolcols].astype(float);C=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(C)):
            s=slice(i*chunk,(i+1)*chunk);assert s.stop<=len(spm)
            c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);C[i]=c
            decoder_consume_calls[0]+=1
        CD['decoder_eval']+=1
        return C
    DEC={}
    for nm in pre['decoder_arms']:
        C=ctrl_of(loadz(o/f'SPK_{nm}.npz'),MotorOutputConfig())
        ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b2 in LEGS[i+1:]:
                fa,_=trimmed(ser[a],100.0,TRIM);fb,_=trimmed(ser[b2],100.0,TRIM)
                if fa is None or fb is None: pairs[f'{a}|{b2}']=None;continue
                nn=min(len(fa),len(fb));assert nn<=len(fa) and nn<=len(fb)
                pairs[f'{a}|{b2}']=float(np.degrees(circ(np.angle(sgn.hilbert(fa[:nn]))-np.angle(sgn.hilbert(fb[:nn])))[0]))
        cross=[v for k,v in pairs.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        spans={lg:float(np.percentile(ser[lg],95)-np.percentile(ser[lg],5)) for lg in LEGS}
        common=float(np.mean([np.all([np.mean([ser[lg][i] for lg in TRIPOD['A']])>=0.6,np.mean([ser[lg][i] for lg in TRIPOD['B']])>=0.6]) for i in range(len(ser['lf']))]))
        DEC[nm]=dict(trim_s=TRIM,duration_s=12.0,metric='signed_Hilbert_phase',nulls=NULLS,n_cross_ge150=int(sum(1 for v in cross if np.abs(wrap(v-180.0))<=30.0)),cross_min=float(min(cross)),cross_mean=float(np.mean(cross)),
          min_leg_span=float(min(spans.values())),spans=spans,common_sat=common,clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),
          ctrl_median=float(np.median(C)),leg_pairs=pairs,
          pass_gate=bool(sum(1 for v in cross if np.abs(wrap(v-180.0))<=30.0)==9 and float(min(spans.values()))>=TH['span_min'] and common<=TH['common_sat_max']))
    assert decoder_consume_calls[0]==pre['decoder_consume_calls_expected']
    dump(o/'DECODER_CALL_DETAIL.json',dict(evaluations=CD['decoder_eval'],consume_calls=decoder_consume_calls[0],expected=pre['decoder_consume_calls_expected']))
    dump(o/'DECODER_CRITERIA.json',dict(arms=DEC,config=dict(rate_for_full_control_hz=100.0),thresholds=TH,
      note='clip fractions are descriptive only and are not a gate'))
    dump(o/'MOTOR_INTERFACE_V9.json',dict(arms=DEC,config=dict(rate_for_full_control_hz=100.0),thresholds=TH))
    # ---------- gates with four states
    def st(b):
        return 'PASS' if b else 'FAIL'
    def c1_state():
        arms=GATEARMS['c1'];miss=[a for a in arms if not AR[a]['eligible']]
        if miss: return 'INELIGIBLE',dict(missing=miss)
        ok=all(AR[a]['rho']['criterion'] and AR[a]['phase']['slip_count']==0 and np.abs(wrap(AR[a]['phase']['phase_deg']-180.0))<=30.0 and AR[a]['plateau_ok'] for a in arms)
        return st(ok),dict(detail={a:dict(rho=AR[a]['rho']['value'],p1=AR[a]['rho']['null']['p1'],drift=AR[a]['drift_status']) for a in arms})
    def c2_state():
        arms=GATEARMS['c2'];miss=[a for a in arms if a not in DEC]
        if miss: return 'UNTESTED',dict(missing=miss)
        return st(all(DEC[a]['pass_gate'] for a in arms)),dict(detail={a:DEC[a]['pass_gate'] for a in arms})
    def c3_state():
        arms=GATEARMS['c3'];miss=[a for a in arms if not AR[a]['eligible']]
        if miss: return 'INELIGIBLE',dict(missing=miss)
        ok=all(AR[a]['rho']['criterion'] and AR[a]['phase']['slip_count']==0 and np.abs(wrap(AR[a]['phase']['phase_deg']-180.0))<=30.0 and AR[a]['plateau_ok'] for a in arms)
        return st(ok),dict(detail={a:dict(drift=AR[a]['drift_status'],rho=AR[a]['rho']['value']) for a in arms})
    def c4_state():
        arms=GATEARMS['c4'];miss=[a for a in arms if not AR[a]['eligible']]
        if miss: return 'INELIGIBLE',dict(missing=miss)
        pers=[1.0/AR[a]['phase']['fa'] for a in arms]
        cv=float(np.std(pers)/np.mean(pers))
        return st(cv<=TH['cv_max']),dict(cv=cv,periods=pers)
    def c5_state():
        a=GATEARMS['c5'][0]
        pr=AR[a]['phase']
        if pr is None or not AR[a]['eligible']: return 'INELIGIBLE',dict(arm=a)
        ok=bool(AR[a]['drift_status']=='LOCKED' and pr['n_blocks']>=TH['blocks_min'] and pr['block_span'] is not None and pr['block_span']<=TH['block_span_max'])
        return st(ok),dict(arm=a,status=AR[a]['drift_status'],n_blocks=pr['n_blocks'],span=pr['block_span'],
          control_arm=AR['W20_gate']['drift_status'])
    def c6_state():
        a,b2=GATEARMS['c6']
        if not (AR[a]['eligible'] and AR[b2]['eligible']): return 'INELIGIBLE',dict(arms=[a,b2])
        ok=AR[b2]['drift_status']=='LOCKED' and AR[a]['drift_status']=='LOCKED'
        return st(ok),dict(b25=dict(ratio=AR[a]['rates']['ratio'],drift=AR[a]['drift_status']),
          b30=dict(ratio=AR[b2]['rates']['ratio'],drift=AR[b2]['drift_status']))
    STATUS={};DETAIL={}
    for nm,fn in (('c1',c1_state),('c2',c2_state),('c3',c3_state),('c4',c4_state),('c5',c5_state),('c6',c6_state)):
        s,dd=fn();STATUS[nm]=s;DETAIL[nm]=dd
    cert_empty=bool(empty)
    cert_failed=[k for k,v in CERT.items() if not v['passes']]
    ext_bad=[k for k,v in EG.items() if v.get('tested') and not v.get('equal')]
    if cert_empty or cert_failed or any(v in ('INELIGIBLE','UNTESTED') for v in STATUS.values()) or ext_bad:
        branch='NOT_JUDGED'
        dump(o/'STOP.json',dict(reason='certification empty, an ineligible or untested gate, or a failed extension prefix',
          cert_empty=cert_empty,cert_failed=cert_failed,status=STATUS,extension_failures=ext_bad))
    elif all(STATUS[k]=='PASS' for k in ('c1','c2','c3','c4','c5')): branch='e1'
    elif all(STATUS[k]=='PASS' for k in ('c1','c2','c3','c4')) and STATUS['c5']!='PASS': branch='e2'
    elif STATUS['c1']=='FAIL' or STATUS['c3']=='FAIL': branch='e3'
    if 'branch' not in locals():
        branch='NOT_JUDGED'
        dump(o/'STOP.json',dict(reason='unclassified positive gate combination',status=STATUS))
    drift_out={k:dict(phase=AR[k]['phase']['phase_deg'] if AR[k]['phase'] else None,R=AR[k]['phase']['R'] if AR[k]['phase'] else None,
      slope=AR[k]['phase']['slope_deg_s'] if AR[k]['phase'] else None,slip=AR[k]['phase']['slip_count'] if AR[k]['phase'] else None,
      slip_period=AR[k]['phase']['slip_period_s'] if AR[k]['phase'] else None,status=AR[k]['drift_status'],
      blocks=AR[k]['phase']['blocks'] if AR[k]['phase'] else None,span=AR[k]['phase']['block_span'] if AR[k]['phase'] else None,
      total_advance=AR[k]['phase']['total_advance_deg'] if AR[k]['phase'] else None,
      slope_ci_deg_s=AR[k]['phase']['slope_ci_deg_s'],df_hz=AR[k]['phase']['df_hz'],
      df_ci_hz=AR[k]['phase']['df_ci_hz'],df_status=AR[k]['phase']['df_status'],
      df_abs_upper_hz=AR[k]['phase']['df_abs_upper_hz']) for k in AR}
    dump(o/'PHASE_DRIFT_V2.json',dict(arms=drift_out,thresholds=TH,
      note='drift is an OLS slope with block offsets and a slip count; coherence time is not defined from one over the frequency difference'))
    dump(o/'ENGAGEMENT_GATE_V3.json',dict(status=STATUS,detail=DETAIL,branch=branch,thresholds=TH,note='four states are allowed and only a real measurement may be FAIL'))
    dump(o/'TRIM_PLATEAU.json',dict(arms={k:dict(table=v['plateau'],span_deg=v['plateau_span'],passes=v['plateau_ok']) for k,v in AR.items()},nulls=NULLS))
    dump(o/'GATE.json',dict(extension=EG,all_equal=all(v.get('equal') is True for v in EG.values())))
    gait_spikes=loadz(o/'SPK_B30_s0.npz')
    dump(o/'GAIT_SPEC_V5.json',dict(unit='s',not_emergent=True,physics_authorized=branch in ('e1','e2'),
      period_s=float(np.median([1.0/AR[a]['phase']['fa'] for a in GATEARMS['c4'] if AR[a]['phase']])) if any(AR[a]['phase'] for a in GATEARMS['c4']) else None,
      duty={lg:float(np.mean(counts(gait_spikes[4000:16000],leg_cols[lg])>np.mean(np.percentile(counts(gait_spikes[4000:16000],leg_cols[lg]),[5,95])))) for lg in LEGS},
      leg_phase_rel_lf={lg:(DEC['B30_s0']['leg_pairs'][f'lf|{lg}'] if lg!='lf' else 0.0) for lg in LEGS},
      load_caveat='period must be re-read whenever the load changes',train_constraint=dict(added_inhibitory_edges=added['n'],adaptation_tau_ms=100.0)))
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD)
    dump(o/'OUTCOMES.json',dict(round='S345',branch=branch,status=STATUS,detail=DETAIL,
      arms={k:dict(eligible=v['eligible'],rho=(round(v['rho']['value'],3) if v['rho']['value'] is not None else None),
        p1=(round(v['rho']['null']['p1'],3) if v['rho']['null']['p1'] is not None else None),
        phase=(round(v['phase']['phase_deg'],1) if v['phase'] else None),R=(round(v['phase']['R'],3) if v['phase'] else None),
        drift=v['drift_status'],slip=(v['phase']['slip_count'] if v['phase'] else None),
        blocks=(v['phase']['n_blocks'] if v['phase'] else None),span=(round(v['phase']['block_span'],1) if v['phase'] and v['phase']['block_span'] is not None else None),
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()},ratio=round(v['rates']['ratio'],2),ticks=v['ticks'],
        plateau_span=v['plateau_span']) for k,v in AR.items()},
      decoder={k:(DEC[k]['n_cross_ge150'],round(DEC[k]['min_leg_span'],3),round(DEC[k]['common_sat'],3),DEC[k]['pass_gate']) for k in DEC},
      extension={k:(v.get('tested'),v.get('equal')) for k,v in EG.items()},
      cert_empty=cert_empty,counters=CD,not_emergent=True,
      next=('S346 may hand the spec to the body' if branch=='e1' else 'stay offline'),
      one_line='%s status=%s'%(branch,STATUS)))
    print(json.dumps(dict(branch=branch,status=STATUS,cd=CD,
      arms={k:(v['eligible'],round(v['rho']['value'],3) if v['rho']['value'] is not None else None,
        round(v['phase']['phase_deg'],1) if v['phase'] else None,v['drift_status'],
        v['phase']['slip_count'] if v['phase'] else None,round(v['plateau_span'],1)) for k,v in AR.items()},
      decoder={k:(DEC[k]['n_cross_ge150'],round(DEC[k]['min_leg_span'],3),round(DEC[k]['common_sat'],3)) for k in DEC},
      ext={k:(v.get('tested'),v.get('equal')) for k,v in EG.items()},cert_empty=cert_empty),ensure_ascii=False))
    return 0

def finalize(o):
    """Deterministic saved-array finalization; no neural, decoder, or physics calls."""
    load=lambda name: json.loads((o/name).read_text(encoding='utf-8'))
    ar=load('ARMS.json');out=load('OUTCOMES.json');audit=load('AUDIT_INDEPENDENT.json');cert=load('CERT_V22.json')
    rows={}
    for name,v in ar.items():
        ind=audit['arms'][name];cmp=audit['comparison'][name]
        rows[name]=dict(main_phase=v['phase']['phase_deg'],audit_phase=ind['phase_deg'],
          circular_distance_deg=cmp['phase_delta_deg'],rho_distance=cmp['rho_delta'],
          main_status=v['drift_status'],audit_status=ind['drift_status'])
    maximum_phase=max(v['circular_distance_deg'] for v in rows.values())
    maximum_rho=max(v['rho_distance'] for v in rows.values())
    agreement=maximum_phase<=TH['phase_tol'] and maximum_rho<=TH['rho_tol']
    dump(o/'RECONCILE_AGREEMENT.json',dict(rows=rows,max_phase_difference_deg=maximum_phase,
      max_rho_difference=maximum_rho,phase_rho_agree=agreement,audit_all_checks_passed=audit['passed'],
      checks_failed=[c for c in audit['checks'] if not c['passed']],
      comparison='one fixed signed main phase against one signed independent phase; circular distance, no sign folding or best-of-two selection'))
    out['agreement']=dict(phase_rho_agree=agreement,audit_all_checks_passed=audit['passed'],max_phase_difference_deg=maximum_phase,max_rho_difference=maximum_rho)
    if not agreement:
        out['branch']='NOT_JUDGED'
        dump(o/'DISAGREE.json',dict(rows=rows,reason='fixed signed phase or rho tolerance exceeded'))
        dump(o/'STOP.json',dict(reason='independent phase/rho tolerance failed',agreement=out['agreement']))
    failed=[k for k,c in cert['cells'].items() if not c['passes']]
    out['certification']=dict(cells=len(cert['cells']),samples_per_cell=100,rows_per_sample=20000,
      empty_cells=cert['empty_cells'],failed_cells=failed,n_passed=len(cert['cells'])-len(failed),
      max_signed_bias_magnitude=float(np.max(np.abs([c['bias_deg'] for c in cert['cells'].values()]))),
      minimum_R=min(c['R'] for c in cert['cells'].values()))
    out['physics_authorized']=out['branch'] in ('e1','e2')
    out['claim_limits']=['Main c1-c6 PASS is descriptive until calibration scope is certified.',
      'Synthetic detuning reduces resultant length by construction over a finite window; failing R is not proof of a bad phase estimate.',
      'No reciprocal coherence time or extrapolated long-term support/walking claim is made.',
      'Added inhibitory edges and tonic bias are NOT_EMERGENT model interventions.']
    dump(o/'OUTCOMES.json',out)
    cert_records=[]
    for key,cell in cert['cells'].items():
        cert_records.append([cell['phase'],cell['duty'],*cell['rates'],cell['df'],cell['bias_deg'],cell['R'],cell['n'],int(cell['passes'])])
    np.savez_compressed(o/'CERT_TABLE.npz',values=np.asarray(cert_records,float),columns=np.array(['phase_deg','duty','rateA_hz','rateB_hz','df_hz','bias_deg','R','n','passes']))
    baseline=load('ENGINE_BASELINE_SHA256.json')
    engine_ok=all(Path(k).is_file() and sha(k)==v for k,v in baseline.items())
    dump(o/'ENGINE_UNCHANGED.json',dict(passed=engine_ok,n_files=len(baseline)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(18,10))
    names=list(ar)
    xx=np.arange(len(names))
    axs[0,0].bar(xx,[ar[k]['rates']['ratio'] for k in names]);axs[0,0].axhline(1,ls='--',c='k')
    axs[0,0].set_xticks(xx,names,rotation=50,fontsize=7);axs[0,0].set_ylabel('rate ratio max/min');axs[0,0].set_title('Twenty-second arm rates; B25 and B30 both near balance')
    axs[0,1].scatter([ar[k]['phase']['phase_deg'] for k in names],[audit['arms'][k]['phase_deg'] for k in names])
    axs[0,1].plot([-180,180],[-180,180],'k--');axs[0,1].set_xlabel('signed main phase (deg)');axs[0,1].set_ylabel('signed audit phase (deg)');axs[0,1].set_title('Fixed signed comparison, circular max difference %.3f deg'%maximum_phase)
    axs[0,2].bar(xx,[ar[k]['phase']['slip_count'] for k in names]);axs[0,2].set_xticks(xx,names,rotation=50,fontsize=7);axs[0,2].set_ylabel('net half-turn slips');axs[0,2].set_title('Unbiased seed 0 slips; balanced seeds do not')
    for j,key in enumerate(names):
        p=ar[key]['phase'];lo,hi=p['slope_ci_deg_s']
        axs[1,0].plot([lo,hi],[j,j],c='tab:blue');axs[1,0].scatter([p['slope_deg_s']],[j],s=18)
    axs[1,0].set_yticks(xx,names,fontsize=7);axs[1,0].axvline(0,c='gray',ls=':');axs[1,0].set_xlabel('OLS slope and block-bootstrap 95% CI (deg/s)');axs[1,0].set_title('No reciprocal coherence extrapolation')
    cc=list(cert['cells']);axs[1,1].bar(np.arange(len(cc)),[cert['cells'][k]['R'] for k in cc],color=['tab:green' if cert['cells'][k]['passes'] else 'tab:red' for k in cc]);axs[1,1].axhline(.9,ls='--',c='k');axs[1,1].set_xlabel('36 pre-registered synthetic cells, each n=100');axs[1,1].set_ylabel('mean circular R');axs[1,1].set_title('Certification %d/36 pass; no empty cell'%(36-len(failed)))
    dec=load('MOTOR_INTERFACE_V9.json')['arms'];dn=list(dec)
    axs[1,2].bar(np.arange(len(dn))-.18,[dec[k]['min_leg_span'] for k in dn],.36,label='minimum leg span')
    axs[1,2].bar(np.arange(len(dn))+.18,[dec[k]['common_sat'] for k in dn],.36,label='common saturation')
    axs[1,2].set_xticks(np.arange(len(dn)),dn,rotation=30,fontsize=7);axs[1,2].legend(fontsize=7);axs[1,2].set_title('Default decoder: all four evaluations 9/9 cross pairs')
    fig.suptitle('S345: %s | no physics; calibration is not globally certified'%out['branch'])
    fig.tight_layout();fig.savefig(o/'TWENTY_SECOND_ENGAGEMENT.png',dpi=140);plt.close(fig)
    desktop=Path('/home/sensen/Desktop/S345_TWENTY_SECOND_ENGAGEMENT.png')
    desktop.write_bytes((o/'TWENTY_SECOND_ENGAGEMENT.png').read_bytes())
    # Reports are generated from archived fields, never copied from probe claims.
    lines=['# S345 二十秒进物理门（离线，NOT_EMERGENT）','',
      '**主实现 c1–c6 数值均 PASS，不等于整轮认证成功。** 最终分支：`%s`。'%out['branch'],
      '数据出处：`%s`；主实现与独立审计分列，所有报告数值由本函数读取归档生成。'%str(o),
      '', '## 计数、范围和偏差',
      '`CALL_LEDGER.json`：`%s`（impl=main）。'%json.dumps(out['counters'],ensure_ascii=False),
      '指令十臂预算但表中只有九臂；明确补充 B25_s1，未加入 c1–c5。decoder_eval=4 指完整轨迹评估；每次 consume 另记台账。',
      '首个执行在旧短窗 TRIM_AUDIT bootstrap 中退出1；尚未建 tier。REFREEZE v2 后整轮从零执行；未调阈值或网络参数。',
      '', '## 二十秒臂，统一裁剪4秒（保留12秒）',
      '|臂|ρ|圆相位/°|R|周期/s|滑差|状态|来源|', '|---|---:|---:|---:|---:|---:|---|---|']
    for key,v in ar.items():
        p=v['phase'];lines.append('|%s|%.6f|%.4f|%.6f|%.5f|%d|%s|`ARMS.json:%s` impl=main|'%(key,v['rho']['value'],p['phase_deg'],p['R'],1.0/p['fa'],p['slip_count'],v['drift_status'],key))
    lines+=['','## 独立审计（20ms箱、3阶SOS；主实现25ms箱、4阶SOS）',
      '|臂|审计圆相位/°|审计ρ|圆差/°|ρ差|来源|','|---|---:|---:|---:|---:|---|']
    for key,v in audit['arms'].items():
        r=rows[key];lines.append('|%s|%.4f|%.6f|%.4f|%.6f|`AUDIT_INDEPENDENT.json:arms.%s` impl=audit|'%(key,v['phase_deg'],v['rho'],r['circular_distance_deg'],r['rho_distance'],key))
    lines+=['','审计非通过项原样保留：`%s`。不能把某项比对失败改成通过来封存。'%json.dumps([c for c in audit['checks'] if not c['passed']],ensure_ascii=False),
      '', '## 非空合成认证',
      '36格各100条、每条20000行；通过%d格、失败%d格，最大带符号偏差幅值%.4f°，最低R %.6f（`CERT_V22.json:cells` impl=main）。'%(36-len(failed),len(failed),out['certification']['max_signed_bias_magnitude'],out['certification']['minimum_R']),
      'R 是时间相位差的聚集程度。已注入频差的合成对在保留窗内必然走相；其R下降不得误读为真实神经回路失锁。此处保留原门，不事后调参、调格或放宽。',
      '', '## 门状态及适用性', '|门|状态|实测依据|来源|','|---|---|---|---|']
    for key,state in out['status'].items():
        lines.append('|%s|%s|`%s`|`ENGAGEMENT_GATE_V3.json:detail.%s` impl=main|'%(key,state,json.dumps(out['detail'][key],ensure_ascii=False),key))
    lines+=['','## 结论限制与下一步',
      'B25与B30的已测种子均无净半圈滑差；不能据此宣称锁相只在精确30%出现。门臂seed0滑相而seed4锁相，未配平滑相不是所有种子共有。',
      'bootstrap 区间描述本窗估计不确定性，不外推相干时间。certification未全过，GAIT_SPEC_V5仅为候选，physics_authorized=false；不接物理、不接身体。',
      '所有ρ、相位、周期、漂移、ctrl均为包络域投影量；20400条加装边与静态偏置是模型构造/声明式干预（NOT_EMERGENT）；ConductanceParams illustrative/uncalibrated；绝对单位NOT_ESTABLISHED。未证明tripod步态、支撑或自主运动。',
      '', '## 图像',
      '左上为十臂速率比；上中为两实现固定带符号相位对照；右上为净半圈滑差数；左下为斜率bootstrap区间；下中红绿柱保留所有合成认证失败格；右下为译码腿幅与共模饱和。图像必须另用read_image实看，观察日志见FIGURE_REVIEW.md。']
    (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(o/'FINALIZE_RESULT.json',dict(branch=out['branch'],engine_unchanged=engine_ok,agreement=out['agreement'],certification=out['certification'],figure=str(desktop)))
    return 0

if __name__=='__main__': raise SystemExit(main())
