"""S344: one trimmed filter entry, f0 free signed phase, drift on ten second arms and the engagement gate v2."""
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
S341=Path('/home/sensen/Desktop/engine修复验收/s341-antialias-rescore-and-load-tolerance/20261026T024501Z')
S342=Path('/home/sensen/Desktop/engine修复验收/s342-instrument-repair-and-load-limit/20261027T025504Z')
S343=Path('/home/sensen/Desktop/engine修复验收/s343-balanced-alternation-and-cadence/20261027T031810Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
GE_PER_CELL=0.00349
CHOSEN=dict(bin_ms=25,trim_s=1.5,band=(0.5,4.0))
BINS=(10,20,25,100);TRIMS=(0.0,1.0,1.5);BANDS=((0.5,4.0),(0.8,4.5))
TH=dict(cycles_min=8,retained_min=0.4,trim_floor=1.0,z_gate=-3.0,rho_sigma=1.0,cross_lo=150.0,
  span_min=0.4,common_sat_max=0.05,cv_max=0.10,drift_max=10.0,R_min=0.9,quant_max=10.0,
  phase_tol=8.0,R_tol=0.10,nulls=NULLS)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else (None if o is None else float(o)))+'\n',encoding='utf-8')
def wrap(d): return (d+180.0)%360.0-180.0
def _sos(kind,order,cuts,fs):
    sos=(sgn.butter(order,cuts/(fs/2),btype='low',output='sos') if kind=='low'
         else sgn.butter(order,[cuts[0]/(fs/2),min(cuts[1]/(fs/2),0.98)],btype='band',output='sos'))
    for s in sos:
        r=np.roots(s[3:])
        if r.size and float(np.max(np.abs(r)))>0.999: raise SystemExit('unstable filter section')
    return sos
def trimmed_band(x,fs,band,trim_s,order=4):
    """the single filter entry: zero phase band pass then trim both edges"""
    sos=_sos('band',order,band,fs)
    y=sgn.sosfiltfilt(sos,x-x.mean())
    k=int(round(trim_s*fs))
    if k>0:
        if len(y)<=2*k+8: return None,0.0
        y=y[k:len(y)-k]
    retained=len(y)/max(len(x),1)
    return y,retained
def env_counts(spm,cols,bin_ms):
    c=spm[:,cols].sum(1).astype(float)
    n=len(c)//bin_ms
    return np.array([c[i*bin_ms:(i+1)*bin_ms].sum() for i in range(n)],float)
def circ_stats(phi):
    z=np.mean(np.exp(1j*phi))
    return float(np.angle(z)),float(abs(z))
def phase_report(a,b,fs,task_name=''):
    za=sgn.hilbert(a);zb=sgn.hilbert(b)
    d=np.angle(za)-np.angle(zb)
    mu,R=circ_stats(d)
    unw=np.unwrap(d);t=np.arange(len(unw))/fs
    slope=float(np.polyfit(t,unw,1)[0]) if len(unw)>4 else None
    fa=np.diff(np.unwrap(np.angle(za)))*fs/(2*np.pi);fb=np.diff(np.unwrap(np.angle(zb)))*fs/(2*np.pi)
    df=float(np.mean(fa)-np.mean(fb))
    blocks=[]
    n2=int(2*fs)
    for i in range(0,max(1,len(unw)-n2+1),n2):
        seg=d[i:i+n2]
        if len(seg)>=8: blocks.append(circ_stats(seg)[1])
    return dict(phase_deg=float(np.degrees(mu)),R=R,drift_deg_s=float(np.degrees(slope)) if slope is not None else None,
      df_hz=df,coherence_s=(1.0/abs(df) if df not in (0.0,) else None),R_2s_blocks=blocks,n_blocks=len(blocks),
      fA=float(np.mean(fa)),fB=float(np.mean(fb)))
def lag_check(a,b,fs):
    """the second estimator: 50 ms boxcar, no band pass, detrended, zero lag and half period"""
    k=max(1,int(round(0.05*fs)))
    sm=lambda v: np.convolve(v-v.mean(),np.ones(k)/k,mode='same')
    xa=sm(a);xb=sm(b);t=np.arange(len(xa));xa=xa-np.polyval(np.polyfit(t,xa,1),t);xb=xb-np.polyval(np.polyfit(t,xb,1),t)
    c0=float(np.corrcoef(xa,xb)[0,1])
    ac=sgn.correlate(xa,xa,mode='full',method='fft')[len(xa)-1:];ac/=max(ac[0],1e-30)
    lo,hi=max(2,int(0.2*fs)),min(len(ac)-1,int(1.2*fs))
    per=None
    if hi>lo:
        seg=ac[lo:hi];kk=int(np.argmin(seg))
        if seg[kk]<0: per=2.0*(lo+kk)/fs
    chalf=None
    if per:
        lag=int(round(0.5*per*fs))
        if 0<lag<len(xa)-4: chalf=float(np.corrcoef(xa[lag:],xb[:-lag])[0,1])
    return dict(corr0=c0,corr_half=chalf,period_s=per,sign_ok=bool(c0<0 and (chalf is None or chalf>0)))
def null_rho(rate,bin_ms,band,trim_s):
    vals=[]
    for i in range(NULLS):
        rng=np.random.default_rng(SEED+17*i+int(rate*3)%71)
        a=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        b=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        ya,_=trimmed_band(env_counts(a.reshape(-1,1),[0],bin_ms),1000.0/bin_ms,band,trim_s)
        yb,_=trimmed_band(env_counts(b.reshape(-1,1),[0],bin_ms),1000.0/bin_ms,band,trim_s)
        if ya is None or yb is None: continue
        n=min(len(ya),len(yb));vals.append(float(np.corrcoef(ya[:n],yb[:n])[0,1]))
    v=np.array(vals);return dict(p1=float(np.percentile(v,1)),p95=float(np.percentile(v,95)),std=float(v.std()),
      mean=float(v.mean()),nulls=NULLS,n=int(v.size),n_eff=int(max(1,v.size//2)))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S344',mode='one trimmed filter entry, f0 free signed phase, drift on ten second arms and the engagement gate v2; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=8,control_tick=50000,tier_advance=0,neural_substep=100000,physics_step=0,model_load=0,render=0,decoder_eval=3),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,thresholds=TH,chosen_pipeline=dict(bin_ms=CHOSEN['bin_ms'],trim_s=CHOSEN['trim_s'],band=list(CHOSEN['band'])),
      rules=['one filter entry does zero phase filtering and then trims at least max(1.0, 2 over band low) seconds from each end, asserting eight retained cycles and a retained fraction of at least 0.4',
             'the trim zero setting is archived as edge contaminated and never enters a criterion',
             'phase is the circular statistics of the analytic phase difference and never an absolute mean',
             'any frequency dependent estimator carries its quantisation and may not enter a criterion above ten degrees',
             'the decoder criterion is nine of nine cross pairs at 150 degrees, a minimum leg span of 0.4 and a common saturation at or below 0.05 while clip fractions stay descriptive',
             'nulls are independently synthesised Poisson series at the matched rate',
             'ten second arms report drift blocks coherence time and the tripod frequency difference'])
    dump(o/'PREREG.json',pre)
    (o/'S344_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s344.py)=%s\nsha256(audit_s344_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s344_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    def score(spm,bin_ms,trim_s,band,colsA=None,colsB=None):
        ca=Ac if colsA is None else colsA
        cb=Bc if colsB is None else colsB
        cA=env_counts(spm,ca,bin_ms);cB=env_counts(spm,cb,bin_ms);fs=1000.0/bin_ms
        yA,retA=trimmed_band(cA,fs,band,trim_s);yB,_=trimmed_band(cB,fs,band,trim_s)
        if yA is None or yB is None: return dict(eligible=False,retained=0.0)
        n=min(len(yA),len(yB));yA=yA[:n];yB=yB[:n]
        rho=float(np.corrcoef(yA,yB)[0,1])
        f0=(score.f0A or 2.0) if hasattr(score,'f0A') else 2.0
        pr=phase_report(yA,yB,fs)
        cycles=pr['fA']*n/fs
        elig=bool(cycles>=TH['cycles_min'] and retA>=TH['retained_min'])
        return dict(eligible=elig,retained=retA,cycles=float(cycles),rho=rho,**pr)
    # ---------- synth certification of the chosen pipeline
    def synth(rate,phase_deg,duty,df,seed):
        rng=np.random.default_rng(seed);n=5000;t=np.arange(n)/1000.0;f=2.0
        def train(ph,fx):
            cyc=np.mod(t*fx-ph/360.0,1.0)
            lam=np.maximum(np.where(cyc<duty,rate/1000.0/duty,0.0),1e-9)
            return rng.poisson(lam).astype(float)
        return train(0.0,f),train(phase_deg,f+df)
    CERT={}
    for phase_deg in (150.0,165.0,180.0):
        for duty in (0.3,0.5,0.7):
            for (r1,r2) in ((105.0,59.0),(90.0,80.0)):
                for df in (0.0,0.02):
                    errs=[];rs=[];drifts=[]
                    for s in range(200):
                        a,b=synth(r1,phase_deg,duty,df,9000+s+int(phase_deg)+int(duty*10))
                        spm=np.zeros((5000,2),bool);spm[:,0]=a>0;spm[:,1]=b>0
                        sc=score(spm,CHOSEN['bin_ms'],CHOSEN['trim_s'],CHOSEN['band'],colsA=[0],colsB=[1])
                        if not sc.get('eligible'): continue
                        errs.append(abs(abs(sc['phase_deg'])-phase_deg));rs.append(sc['R'])
                        if sc['drift_deg_s'] is not None: drifts.append(sc['drift_deg_s'])
                    CERT[f'p{int(phase_deg)}_d{int(duty*10)}_r{int(r1)}_{int(r2)}_df{int(df*100)}']=dict(
                      phase=phase_deg,duty=duty,rates=[r1,r2],df=df,n=len(errs),
                      bias_deg=float(np.mean(errs)) if errs else None,R=float(np.mean(rs)) if rs else None,
                      drift_mean=float(np.mean(drifts)) if drifts else None,
                      passes=bool(errs and float(np.mean(errs))<=5.0 and float(np.mean(rs))>=0.9))
    dump(o/'SYNTH_CERT_V21.json',dict(cells=CERT,thresholds=TH,chosen=pre['chosen_pipeline'],
      note='the synthetic grid is archived in full; failing cells narrow the applicability of the chosen pipeline and are not tuned away'))
    # ---------- instrument table on archived arms
    ARCH={}
    for nm,p in (('W0_gate',S342/'SPK_W0_gate.npz'),('L30_A',S342/'SPK_L30_A.npz'),('L30_B',S342/'SPK_L30_B.npz'),
                 ('S343_P0_gate',S343/'SPK_P0_gate.npz'),('S343_B30',S343/'SPK_B30.npz'),('S343_B30_LA30',S343/'SPK_B30_LA30.npz')):
        if p.is_file(): ARCH[nm]=loadz(p)
    TAB={}
    for nm,A in ARCH.items():
        per={}
        for bm in BINS:
            for band in BANDS:
                for tr in TRIMS:
                    sc=score(A,bm,tr,band)
                    key=f'b{bm}_tr{tr}_band{band[0]}-{band[1]}'
                    per[key]=dict(eligible=sc.get('eligible'),retained=sc.get('retained'),cycles=sc.get('cycles'),
                      rho=sc.get('rho'),phase=sc.get('phase_deg'),R=sc.get('R'),
                      contaminated=bool(tr==0.0),quant_deg=float(360.0*(sc.get('fA') or 2.0)/(1000.0/bm)))
        TAB[nm]=per
    dump(o/'INSTRUMENT_V21.json',dict(table=TAB,chosen=pre['chosen_pipeline'],bins=list(BINS),trims=list(TRIMS),bands=[list(b) for b in BANDS],
      note='the trim zero rows are edge contaminated and never enter a criterion'))
    # ---------- stage B arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig
    CD={k:0 for k in pre['counters']}
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
    SPM={}
    def run(name,biasA=0.0,biasB=0.0,seed=0,ticks=5000):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if biasA: cur[Ac]+=biasA/100.0*I_STAR
        if biasB: cur[Bc]+=biasB/100.0*I_STAR
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        SPM[name]=spm
        sc=score(spm,CHOSEN['bin_ms'],CHOSEN['trim_s'],CHOSEN['band'])
        sc1=score(spm,CHOSEN['bin_ms'],1.0,CHOSEN['band'])
        sc0=score(spm,CHOSEN['bin_ms'],0.0,CHOSEN['band'])
        lc=lag_check(env_counts(spm,Ac,CHOSEN['bin_ms']),env_counts(spm,Bc,CHOSEN['bin_ms']),1000.0/CHOSEN['bin_ms'])
        rA=float(spm[:,Ac].sum()*1000.0/(ticks*len(Ac)));rB=float(spm[:,Bc].sum()*1000.0/(ticks*len(Bc)))
        nr=null_rho((rA+rB)/2.0,CHOSEN['bin_ms'],CHOSEN['band'],CHOSEN['trim_s'])
        z=(sc['rho']-nr['mean'])/max(nr['std'],1e-30)
        return dict(name=name,seed=seed,ticks=ticks,bias_pct=dict(A=biasA,B=biasB),bias_nA=dict(A=biasA/100*I_STAR,B=biasB/100*I_STAR),
          rates=dict(A=rA,B=rB,ratio=max(rA,rB)/max(min(rA,rB),1e-9)),main=sc,trim_1p0=sc1,trim_0=dict(sc0,contaminated=True),
          lag=lc,null=nr,z=z,added_edges=added,f_drive_hz=None,dI_time_dependent=False,not_emergent=True)
    AR={}
    AR['G0_gate']=run('G0_gate')
    AR['B30_5s']=run('B30_5s',biasB=30.0)
    AR['B30_seed1']=run('B30_seed1',biasB=30.0,seed=1)
    AR['B30_seed2']=run('B30_seed2',biasB=30.0,seed=2)
    AR['LA30_5s']=run('LA30_5s',biasA=30.0,biasB=30.0)
    AR['LB30_5s']=run('LB30_5s',biasB=60.0)
    AR['P10_10s']=run('P10_10s',ticks=10000)
    AR['B30_10s']=run('B30_10s',biasB=30.0,ticks=10000)
    gate={}
    refs=[S343/'SPK_P0_gate.npz',S342/'SPK_W0_gate.npz',S341/'SPK_A0_gate.npz']
    lst=[]
    for p in refs:
        if p.is_file():
            r0=loadz(p)[:5000];lst.append(dict(ref=str(p.parent.name)+'/'+p.name,equal=bool(np.array_equal(SPM['G0_gate'],r0)),spike_total=int(r0.sum())))
    gate['G0_gate']=dict(all_equal=bool(lst) and all(x['equal'] for x in lst),refs=lst,spike_total=int(SPM['G0_gate'].sum()),shape=list(SPM['G0_gate'].shape),
      method='unpackbits elementwise equality')
    dump(o/'GATE.json',gate)
    # ---------- decoder criteria
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    def ctrl_of(Am,cfgX,chunk=10):
        mo=MotorOutput(MAN,cfgX);spm=Am[:,poolcols].astype(float);C=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(C)):
            s=slice(i*chunk,(i+1)*chunk);c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);C[i]=c
        CD['decoder_eval']+=1
        return C
    DEC={}
    for nm in ('G0_gate','B30_5s','B30_seed2'):
        C=ctrl_of(SPM[nm],MotorOutputConfig())
        ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
        fs=100.0
        tA=np.mean([ser[lg] for lg in TRIPOD['A']],axis=0);tB=np.mean([ser[lg] for lg in TRIPOD['B']],axis=0)
        pairs={}
        for i,a in enumerate(LEGS):
            for b2 in LEGS[i+1:]:
                ya,_=trimmed_band(ser[a],fs,CHOSEN['band'],1.0);yb,_=trimmed_band(ser[b2],fs,CHOSEN['band'],1.0)
                pr=phase_report(ya,yb,fs);pairs[f'{a}|{b2}']=abs(pr['phase_deg'])
        cross=[v for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        same=[v for k,v in pairs.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        satA=(ser[[lg for lg in TRIPOD['A']][0]]>=0.999);satB=(ser[[lg for lg in TRIPOD['B']][0]]>=0.999)
        common=float(np.mean([np.all([np.mean([ser[lg][i] for lg in TRIPOD['A']])>=0.6,np.mean([ser[lg][i] for lg in TRIPOD['B']])>=0.6]) for i in range(len(tA))]))
        spans={lg:float(np.percentile(ser[lg],95)-np.percentile(ser[lg],5)) for lg in LEGS}
        DEC[nm]=dict(n_cross_ge150=int(sum(1 for v in cross if v>=TH['cross_lo'])),cross_min=float(min(cross)),cross_mean=float(np.mean(cross)),
          same_max=float(max(same)),min_leg_span=float(min(spans.values())),spans=spans,common_sat=common,
          clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),ctrl_median=float(np.median(C)),ctrl_p99=float(np.percentile(C,99)),
          pass_gate=bool(sum(1 for v in cross if v>=TH['cross_lo'])==9 and float(min(spans.values()))>=TH['span_min'] and common<=TH['common_sat_max']),
          leg_pairs=pairs)
    dump(o/'DECODER_CRITERIA.json',dict(arms=DEC,criterion='nine of nine cross pairs at 150 degrees, minimum leg span 0.4, common saturation at or below 0.05',
      note='clip fractions are descriptive only and are not a gate',thresholds=TH))
    dump(o/'MOTOR_INTERFACE_V8.json',dict(arms=DEC,config=dict(rate_for_full_control_hz=100.0),thresholds=TH))
    # ---------- drift
    DR={}
    for nm in ('P10_10s','B30_10s'):
        r=AR[nm]
        DR[nm]=dict(phase_deg=r['main']['phase_deg'],R=r['main']['R'],drift_deg_s=r['main']['drift_deg_s'],
          df_hz=r['main']['df_hz'],coherence_s=r['main']['coherence_s'],R_2s=r['main']['R_2s_blocks'],
          n_blocks=r['main']['n_blocks'],eligible=r['main']['eligible'],ticks=r['ticks'])
    dump(o/'PHASE_DRIFT.json',dict(arms=DR,thresholds=dict(drift=TH['drift_max'],R=TH['R_min'],blocks=5),
      note='drift is the slope of the unwrapped circular phase difference and the coherence time is one over the tripod frequency difference'))
    # ---------- gates
    bal=['B30_5s','B30_seed1','B30_seed2']
    def c1_ok(k):
        r=AR[k]
        return bool(r['main']['eligible'] and r['z']<=TH['z_gate'] and r['main']['rho']<=r['null']['p1']-TH['rho_sigma']*r['null']['std'])
    c1=all(c1_ok(k) for k in bal)
    c1_elig=[k for k in bal if AR[k]['main']['eligible']]
    c2=bool(DEC['G0_gate']['pass_gate'] and DEC['B30_5s']['pass_gate'] and DEC['B30_seed2']['pass_gate'])
    c3=bool(c1_ok('LA30_5s') and c1_ok('LB30_5s'))
    pers_s=[1.0/AR[k]['main']['fA'] for k in bal]
    cv=float(np.std(pers_s)/np.mean(pers_s)) if len(pers_s)>=3 else None
    c4=bool(cv is not None and cv<=TH['cv_max'])
    d=DR['B30_10s']
    c5=bool(d['drift_deg_s'] is not None and abs(d['drift_deg_s'])<=TH['drift_max'] and d['n_blocks']>=5 and all(x>=TH['R_min'] for x in d['R_2s']))
    flags=dict(c1=c1,c2=c2,c3=c3,c4=c4,c5=c5)
    if c1 and c2 and c3 and c4 and c5: branch='d1'
    elif c1 and c2 and c3 and c4 and not c5: branch='d2'
    elif not c1: branch='d3'
    else: branch='d3'
    if branch in ('d1','d2'):
        hand=dict(period_s=float(np.median(pers_s)),period_route='analytic instantaneous frequency',duty={lg:None for lg in LEGS},
          swing=DEC['B30_5s']['spans'],leg_phase_rel_lf={lg:abs(DEC['B30_5s']['leg_pairs'].get(f'lf|{lg}',0.0)) if lg!='lf' else 0.0 for lg in LEGS},
          R=AR['B30_5s']['main']['R'],drift_deg_s=d['drift_deg_s'],reference_current_nA=I_STAR,
          decoder=dict(rate_for_full_control_hz=100.0),not_emergent=True,not_before='S345')
        dump(o/'PHYSICS_HANDOFF.json',hand)
    dump(o/'ENGAGEMENT_GATE_V2.json',dict(flags=flags,branch=branch,eligible_balance_arms=c1_elig,
      c1_detail={k:dict(z=round(AR[k]['z'],2),rho=round(AR[k]['main']['rho'],3),eligible=AR[k]['main']['eligible']) for k in bal},
      c2_detail={k:DEC[k]['pass_gate'] for k in DEC},c4_cv=cv,c5_detail=dict(drift=d['drift_deg_s'],blocks=d['R_2s']),thresholds=TH))
    # ---------- independent audit coverage handled by the audit script
    dump(o/'ARMS.json',{k:{kk:vv for kk,vv in v.items()} for k,v in AR.items()});dump(o/'CALL_LEDGER.json',CD)
    dump(o/'OUTCOMES.json',dict(round='S344',branch=branch,flags=flags,gate=gate,
      arms={k:dict(eligible=v['main']['eligible'],cycles=round(v['main'].get('cycles') or 0,1),rho=round(v['main']['rho'],3),
        phase=round(v['main']['phase_deg'],1),R=round(v['main']['R'],3),z=round(v['z'],2),
        drift=(round(v['main']['drift_deg_s'],2) if v['main']['drift_deg_s'] is not None else None),
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()},ticks=v['ticks']) for k,v in AR.items()},
      decoder={k:(DEC[k]['n_cross_ge150'],round(DEC[k]['cross_min'],1),round(DEC[k]['min_leg_span'],3),round(DEC[k]['common_sat'],3),DEC[k]['pass_gate']) for k in DEC},
      counters=CD,not_emergent=True,
      next=('S345 hands PHYSICS_HANDOFF to the body' if branch in ('d1','d2') else 'stay offline and fix only the failing predicate'),
      one_line='%s flags=%s cv=%s drift=%s'%(branch,flags,cv,(round(d['drift_deg_s'],2) if d['drift_deg_s'] is not None else None))))
    print(json.dumps(dict(branch=branch,flags=flags,gate=gate['G0_gate']['all_equal'],
      arms={k:(v['main']['eligible'],round(v['main']['rho'],3),round(v['main']['phase_deg'],1),round(v['main']['R'],3),round(v['z'],2)) for k,v in AR.items()},
      drift={k:(round(v['drift_deg_s'],2) if v['drift_deg_s'] is not None else None,round(v['df_hz'],4),round(v['R'],3)) for k,v in DR.items()},
      decoder={k:(DEC[k]['n_cross_ge150'],round(DEC[k]['min_leg_span'],3),round(DEC[k]['common_sat'],3),DEC[k]['pass_gate']) for k in DEC},
      cv=cv,cert_fail=[k for k,v in CERT.items() if not v['passes']],cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
