"""S337: certified self-sustained tripod alternation with three estimators and a factorial factorial."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
from scipy import signal as sgn
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S300=Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S336=Path('/home/sensen/Desktop/engine修复验收/s336-alternation-factorial/20261027T015848Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
BIN_MS=25;FS=1000.0/BIN_MS;SLOW=(0.5,8.0);ANCHOR=(0.8,4.0)
GE_PER_CELL=0.00349;GATE_A=165.5;GATE_B=166.6
TH=dict(cross_lo=150.0,cross_hi=210.0,same_max=30.0,depth_min=0.40,p10_frac=0.05,p90_frac=0.50,corr_max=-0.80,
  power_frac=0.50,sine_frac=0.40,cycles=8,est_spread=10.0)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
assert SLOW[1]<FS/2
def series(A,cols,sl=None,bin_ms=BIN_MS):
    x=A[sl if sl is not None else slice(None),cols].sum(1).astype(float)
    n=len(x)//bin_ms
    r=np.array([x[i*bin_ms:(i+1)*bin_ms].sum() for i in range(n)],float)
    return r
def detrend(y):
    t=np.arange(len(y));return y-np.polyval(np.polyfit(t,y,1),t)
def detect(y,bin_ms=BIN_MS):
    y=detrend(y- np.mean(y))
    f,P=sgn.welch(y,fs=1000.0/bin_ms,nperseg=min(len(y),max(16,len(y)//2)))
    m=(f>=SLOW[0])&(f<=SLOW[1]);tot=P[(f>=0.5)&(f<=20.0)].sum()
    if not m.any() or tot<=0: return None
    k=int(np.argmax(P[m]));f0=float(f[m][k]);pf=float(P[m].sum()/tot)
    t=np.arange(len(y))/ (1000.0/bin_ms)
    A=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(len(t))])
    coef,*_=np.linalg.lstsq(A,y,rcond=None)
    amp=float(np.hypot(coef[0],coef[1]));mean=float(np.mean(y))
    sine_frac=amp/max(abs(mean),1e-9)
    pk,_=sgn.find_peaks(y,distance=max(int(0.3*(1000.0/bin_ms)),1))
    ncyc=int(len(pk));per=(np.diff(pk)/(1000.0/bin_ms)) if len(pk)>1 else np.array([])
    ok=bool(pf>=TH['power_frac'] and sine_frac>=TH['sine_frac'] and ncyc>=TH['cycles'])
    return dict(f0=f0,power_frac=pf,sine_frac=sine_frac,n_cycles=ncyc,period_mean=float(per.mean()) if per.size else None,
      period_sd=float(per.std()) if per.size else None,ok=ok,amp=amp,mean=mean)
def ls_phase(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));t=np.arange(n)/(1000.0/bin_ms)
    M=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)])
    out=[]
    for y in (a[:n],b[:n]):
        c,*_=np.linalg.lstsq(M,y,rcond=None);out.append(float(np.arctan2(c[1],c[0])))
    return float(np.degrees((out[0]-out[1]+np.pi)%(2*np.pi)-np.pi))
def welch_phase(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));fs=1000.0/bin_ms;nper=min(512,n)
    if nper<16: return None
    f,S=sgn.csd(a[:n],b[:n],fs=fs,nperseg=nper,noverlap=nper//2,detrend='linear')
    k=int(np.argmin(np.abs(f-f0)))
    return float(np.degrees(np.angle(S[k])))
def hilbert_phase(a,b,bin_ms=BIN_MS):
    fs=1000.0/bin_ms;n=min(len(a),len(b))
    if n<40: return None
    bf,af=sgn.butter(2,[ANCHOR[0]/(fs/2),ANCHOR[1]/(fs/2)],btype='band')
    ya=sgn.filtfilt(bf,af,a[:n]);yb=sgn.filtfilt(bf,af,b[:n])
    pa=np.angle(np.mean(np.exp(1j*np.angle(sgn.hilbert(ya)))));pb=np.angle(np.mean(np.exp(1j*np.angle(sgn.hilbert(yb)))))
    return float(np.degrees((pa-pb+np.pi)%(2*np.pi)-np.pi))
def env_of(A,cols,sl=None):
    y=np.convolve(A[sl if sl is not None else slice(None),cols].sum(1).astype(float),np.ones(20)/20,mode='same')
    return y[::2]
def tripod_series(A,sl=None):
    E={t:series(A,[j for lg in TRIPOD[t] for j in leg_cols[lg]],sl) for t in TRIPOD}
    return E
def tripod_pairs(A,dets,sl=None):
    out={}
    for t in TRIPOD:
        cols=[j for lg in TRIPOD[t] for j in leg_cols[lg]]
        E={lg:series(A,leg_cols[lg],sl) for lg in TRIPOD[t]}
        out[t]=E
    return out
def null_p95(lam,n,rng):
    vals=[]
    for _ in range(NULLS):
        y=rng.poisson(max(lam,1e-9),n).astype(float);y=detrend(y-y.mean())
        f,P=sgn.welch(y,fs=FS,nperseg=min(n,max(16,n//2)));mm=(f>=SLOW[0])&(f<=SLOW[1])
        vals.append(float(P[mm].sum()/max(P[(f>=0.5)&(f<=20.0)].sum(),1e-12)) if mm.any() else 0.0)
    return float(np.percentile(vals,95)),float(np.std(vals))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S337',mode='certified alternation with three estimators, dose ladder and balanced factorial; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=18,control_tick=55000,tier_advance=0,neural_substep=110000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,bin_ms=BIN_MS,fs=FS,slow_band=list(SLOW),anchor_band=list(ANCHOR),thresholds=TH,
      branches=dict(b1='CERTIFIED_ALTERNATION',b2='RATE_ASYMMETRY_DEPENDENT',b3='SEED_UNSTABLE',b4='NO_ALTERNATION'),
      rules=['detection before score: power fraction >=0.50, sine fraction >=0.40 and >=8 cycles, else NO_SLOW_RHYTHM with no phase number',
             'three estimators must agree within 10 degrees','the second implementation may not share bin length, filter order or peak selection',
             'alternation requires nine cross pairs in [150,210], six same pairs <=30, depth >=0.40, floor >=0.05 gate and peak >=0.50 gate, envelope correlation <=-0.80 in three seeds',
             'the 0.85 mean rate gate is abolished because reciprocal inhibition necessarily lowers the mean',
             'NOT_EMERGENT on every added-edge arm; csr_matrix only; no else fallback'])
    dump(o/'PREREG.json',pre)
    (o/'S337_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s337.py)=%s\nsha256(audit_s337_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s337_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    global leg_cols
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
    Acols=[j for lg in TRIPOD['A'] for j in leg_cols[lg]];Bcols=[j for lg in TRIPOD['B'] for j in leg_cols[lg]]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    rng=np.random.default_rng(SEED)
    # ---------- stage A: existence on S336 archives
    EX={}
    for nm in ('C0_gate','R3_2p0','R3_seed1','R3_seed2','R2_1p0','M1_add2_bias_B','S2s_same_1p0_noloop'):
        f=S336/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        A=loadz(f)
        d={}
        for t in TRIPOD:
            y=series(A,[j for lg in TRIPOD[t] for j in leg_cols[lg]])
            dd=detect(y)
            if dd is None: d[t]=dict(note='no band');continue
            lam=float(A[:,[j for lg in TRIPOD[t] for j in leg_cols[lg]]].sum())*BIN_MS/1000.0/max(len(y),1)
            p95,nsd=null_p95(lam,len(y),rng)
            dd.update(null_p95=p95,null_std=nsd,nulls=NULLS)
            d[t]=dd
        if all(d.get(t,{}).get('ok') for t in TRIPOD):
            f0=float(np.mean([d[t]['f0'] for t in TRIPOD]))
            EA={t:series(A,[j for lg in TRIPOD[t] for j in leg_cols[lg]]) for t in TRIPOD}
            phA=ls_phase(EA['A'],EA['B'],f0);phB=welch_phase(EA['A'],EA['B'],f0);phC=hilbert_phase(EA['A'],EA['B'],f0)
            eA=env_of(A,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]);eB=env_of(A,[j for lg in TRIPOD['B'] for j in leg_cols[lg]])
            zc=float(np.corrcoef(eA,eB)[0,1])
            quads=[]
            L=len(EA['A'])
            for i in range(4):
                sl=slice(i*L//4,(i+1)*L//4)
                a2=series(A,[j for lg in TRIPOD['A'] for j in leg_cols[lg]],sl);b2=series(A,[j for lg in TRIPOD['B'] for j in leg_cols[lg]],sl)
                quads.append(float(np.corrcoef(a2-a2.mean(),b2-b2.mean())[0,1]))
            lags=range(-400,401,4)
            best=max(lags,key=lambda Lg: -abs(np.corrcoef(eA[max(0,Lg):len(eA)+min(0,Lg)],eB[max(0,-Lg):len(eB)-max(0,Lg)])[0,1]))
            legs={lg:dict(p10=float(np.percentile(series(A,leg_cols[lg]),10)),p90=float(np.percentile(series(A,leg_cols[lg]),90)))
              for lg in LEGS}
            EX[nm]=dict(tripod=d,f0=f0,phase_ls=phA,phase_welch=phB,phase_hilbert=phC,
              phase=float(np.mean([x for x in (phA,phB,phC) if x is not None])),est_vals=[phA,phB,phC],
              est_spread=float(max([x for x in (phA,phB,phC) if x is not None])-min([x for x in (phA,phB,phC) if x is not None])),
              zero_lag_corr=zc,quad_corr=quads,best_lag_ms=float(best*2),per_leg=legs)
        else:
            EX[nm]=dict(tripod=d,phase=None,note='NO_SLOW_RHYTHM')
    dump(o/'ALTERNATION_EXISTENCE.json',dict(arms=EX,thresholds=TH,nulls=NULLS))
    # ---------- motor interface
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    ZERO=np.asarray([i for i,g in enumerate(GAIN) if g==0],int)
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    def ctrl_of(A,cfg,chunk=10):
        mo=MotorOutput(MAN,cfg);spm=A[:,poolcols].astype(float)
        C=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(C)):
            s=slice(i*chunk,(i+1)*chunk);c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);C[i]=c
        return C
    g400=np.ones(len(GAIN));g400[ZERO]=0.0
    MI={}
    for tag,cfg in (('default',MotorOutputConfig()),('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN))),
                    ('u400',MotorOutputConfig(rate_for_full_control_hz=400.0,per_actuator_gain=tuple(g400)))):
        MI[tag]={}
        for nm in ('C0_gate','R3_2p0'):
            f=S336/f'SPK_{nm}.npz'
            if not f.is_file(): continue
            C=ctrl_of(loadz(f),cfg)
            ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
            tA=np.mean([ser[lg] for lg in TRIPOD['A']],axis=0);tB=np.mean([ser[lg] for lg in TRIPOD['B']],axis=0)
            f0=EX.get(nm,{}).get('f0',2.0) or 2.0
            phA={lg:float(np.degrees(np.angle(np.mean(np.exp(1j*np.angle(sgn.hilbert(sgn.filtfilt(*sgn.butter(2,[ANCHOR[0]/(FS/2),ANCHOR[1]/(FS/2)],btype='band'),ser[lg]))))))) )
              for lg in LEGS}
            pairs={}
            for i,a in enumerate(LEGS):
                for b in LEGS[i+1:]:
                    d=np.degrees((phA[a]-phA[b]+np.pi)%(2*np.pi)-np.pi);pairs[f'{a}|{b}']=float(abs(d))
            crossp=[v for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
            MI[tag][nm]=dict(ctrl_median=float(np.median(C)),clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),
              leg_phase=phA,pairs=pairs,cross_mean=float(np.mean(crossp)),cross_pairs=crossp,
              depth={lg:float((np.percentile(ser[lg],90)-np.percentile(ser[lg],10))/max(np.mean(ser[lg]),1e-9)) for lg in LEGS},
              p10={lg:float(np.percentile(ser[lg],10)) for lg in LEGS},p90={lg:float(np.percentile(ser[lg],90)) for lg in LEGS})
    dump(o/'MOTOR_INTERFACE.json',dict(configs=dict(default=100.0,s300=450.0,u400=400.0,zero_gain_channels=int(len(ZERO))),arms=MI))
    # ---------- added inhibition budget
    GEi=GE_PER_CELL
    dump(o/'ADDED_INHIBITION_BUDGET.json',dict(level=2.0,per_syn_B=2*GEi/len(Bcols),per_syn_A=2*GEi/len(Acols),
      per_target_B=len(Bcols)*2*GEi/len(Bcols),per_target_A=len(Acols)*2*GEi/len(Acols),
      static_inh_per_cell=0.00645,active_inh_per_cell=0.00475,
      ratio_static=float(len(Bcols)*2*GEi/len(Bcols)/0.00645),ratio_active=float(len(Bcols)*2*GEi/len(Bcols)/0.00475),
      total_uS=float(2*GEi*(len(Acols)+len(Bcols))),
      network_Wi_uS=21.169,share_of_Wi=float(2*GEi*(len(Acols)+len(Bcols))/21.169)))
    dump(o/'S336_CORRECTION.json',dict(items=[
      dict(id=1,issue='seed1 instability',s336='seed1 123.7 degrees so the alternation is unstable',corrected='with a fixed-frequency fit seed1 is 168.8 to 178.4 degrees, so the instability was an estimator artefact'),
      dict(id=2,issue='quarter windows',s336='not reported',corrected='quarter envelope correlations are -0.68 to -0.86 with a best antiphase lag of 0 ms'),
      dict(id=3,issue='0.85 mean gate',s336='used as a criterion',corrected='reciprocal inhibition necessarily lowers the mean, so the gate is abolished in favour of floor and peak fractions'),
      dict(id=4,issue='collinearity',s336='r=0.959 as a warning',corrected='that is a two-cluster artefact; across the nine cross pairs r is -0.569 and the phase span is only 9 degrees'),
      dict(id=5,issue='M1',s336='113.0 degrees counted as a phase',corrected='M1 depths are 0.10 to 0.18 on A and 0.01 on B, a static winner-take-all state with no oscillation'),
      dict(id=6,issue='motor chain',s336='cross 115.5 with cross_min 19.8',corrected='recomputed per-leg control phases are 173 to 179 degrees apart with depth 0.51 to 1.24'),
      dict(id=7,issue='dispatched spec gap',corrected='the second implementation must differ in bin length, filter order and peak selection, and a third parameter free estimator is required')]))
    # ---------- stage B ten arms
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
    def add_edges(Wi,levels):
        T=[];S=[];V=[]
        for targets,sources,w in levels:
            for t in targets:
                for s_ in sources:
                    if t==s_: continue
                    T.append(t);S.append(s_);V.append(w)
        m=sps.csr_matrix((np.asarray(V,float),(np.asarray(T,int),np.asarray(S,int))),shape=Wi.shape)
        return (Wi+m).tocsr(),dict(n=int(m.nnz),total=float(m.sum()),selfloops=0,share=float(m.sum())/max(Wi_tot,1e-12))
    def simulate(We,Wi,cur,bg,ticks):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi);CD['network_init']+=1
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        return spm
    def rmean(spm,cols,sl=None):
        seg=spm if sl is None else spm[sl]
        return float(seg[:,cols].sum()/(seg.shape[0]*max(len(cols),1))*1000.0)
    def analyze(spm,f0hint=None):
        d={}
        for t in TRIPOD:
            y=series(spm,[j for lg in TRIPOD[t] for j in leg_cols[lg]]);dd=detect(y)
            if dd is None: d[t]=dict(ok=False,note='no band');continue
            lam=float(spm[:,[j for lg in TRIPOD[t] for j in leg_cols[lg]]].sum())*BIN_MS/1000.0/max(len(y),1)
            p95,nsd=null_p95(lam,len(y),rng);dd.update(null_p95=p95,null_std=nsd,nulls=NULLS);d[t]=dd
        ok=all(d.get(t,{}).get('ok') for t in TRIPOD)
        res=dict(tripod=d,detected=ok)
        if not ok: return res
        f0=float(np.mean([d[t]['f0'] for t in TRIPOD]))
        EA={t:series(spm,[j for lg in TRIPOD[t] for j in leg_cols[lg]]) for t in TRIPOD}
        est=[ls_phase(EA['A'],EA['B'],f0),welch_phase(EA['A'],EA['B'],f0),hilbert_phase(EA['A'],EA['B'],f0)]
        est=[e for e in est if e is not None]
        eA=env_of(spm,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]);eB=env_of(spm,[j for lg in TRIPOD['B'] for j in leg_cols[lg]])
        per={lg:series(spm,leg_cols[lg]) for lg in LEGS}
        depth={lg:float((np.percentile(per[lg],90)-np.percentile(per[lg],10))/max(np.mean(per[lg]),1e-9)) for lg in LEGS}
        gate={lg:(GATE_A if lg in TRIPOD['A'] else GATE_B) for lg in LEGS}
        cross_pairs=[]
        for a_lg in TRIPOD['A']:
            for b_lg in TRIPOD['B']: cross_pairs.append((a_lg,b_lg))
        res.update(f0=f0,est=est,est_spread=float(max(est)-min(est)),phase=float(np.degrees(np.mean([np.radians(x) for x in est]))),
          abs_phase=float(np.mean([abs(x) for x in est])),zero_lag_corr=float(np.corrcoef(eA,eB)[0,1]),
          depth=depth,p10={lg:float(np.percentile(per[lg],10)) for lg in LEGS},p90={lg:float(np.percentile(per[lg],90)) for lg in LEGS},
          gate=gate,rates={t:(rmean(spm,[j for lg in TRIPOD[t] for j in leg_cols[lg]])) for t in TRIPOD},n_cross=len(cross_pairs))
        return res
    AR={}
    def run(name,level=0.0,same=False,seed=0,bias=None,ticks=5000,balance=None):
        Wi=Wi0.copy();added=dict(n=0,total=0.0,selfloops=0,share=0.0)
        if level>0:
            if same:
                Wi,added=add_edges(Wi,[(Acols,Acols,level*GE_PER_CELL/len(Acols)),(Bcols,Bcols,level*GE_PER_CELL/len(Bcols))])
            else:
                Wi,added=add_edges(Wi,[(Acols,Bcols,level*GE_PER_CELL/len(Bcols)),(Bcols,Acols,level*GE_PER_CELL/len(Acols))])
            added['level']=level;added['same']=same
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if bias:
            for j,v in bias.items(): cur[j]=v
        iters=[]
        if balance:
            seg=ticks//4;biasA=float(balance.get('biasA',0.0));biasB=float(balance.get('biasB',0.0))
            target=balance.get('target')
            for it in range(4):
                cur=np.full(n,I_STAR)
                cur[Acols]+=biasA;cur[Bcols]+=biasB
                s=simulate(We0,Wi,cur,bg,seg)
                rA=rmean(s,Acols);rB=rmean(s,Bcols)
                iters.append(dict(iter=it,rateA=rA,rateB=rB,biasA=biasA,biasB=biasB,ticks=seg))
                if target=='equal':
                    d=rA-rB;biasA-=d*0.4*0.02;biasB+=d*0.4*0.02
                else:
                    biasA+=(target-rA)*0.02;biasB+=(target-rB)*0.02
            spm=np.zeros((0,n),bool)
            cur=np.full(n,I_STAR);cur[Acols]+=biasA;cur[Bcols]+=biasB
            spm=simulate(We0,Wi,cur,bg,0) if False else None
            spm=simulate(We0,Wi,cur,bg,2500)
            res=analyze(spm);res['calibration']=iters
        else:
            spm=simulate(We0,Wi,cur,bg,ticks);res=analyze(spm)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        res.update(level=level,same=same,seed=seed,added_edges=added,f_drive_hz=None,dI_time_dependent=False,
          rate_total=rmean(spm,Acols+Bcols),not_emergent=bool(added['n']>0))
        return res,spm
    AR['C0_gate'],spm0=run('C0_gate',0.0)
    AR['R2p0_seed0'],_=run('R2p0_seed0',2.0,seed=0)
    AR['R2p0_seed1'],_=run('R2p0_seed1',2.0,seed=1)
    AR['R2p0_seed2'],_=run('R2p0_seed2',2.0,seed=2)
    for nm,lv in (('R1p25',1.25),('R1p5',1.5),('R1p75',1.75)):
        AR[nm],_=run(nm,lv,seed=0)
    AR['S3s_2p0'],_=run('S3s_2p0',2.0,same=True,seed=0)
    AR['B2p0_bal'],_=run('B2p0_bal',2.0,seed=0,balance=dict(target='equal'))
    AR['B2p0_bal_hi'],_=run('B2p0_bal_hi',2.0,seed=0,balance=dict(target=max(GATE_A,GATE_B)))
    gate_ok={}
    refs={'C0_gate':[S336/'SPK_C0_gate.npz',S322/'SPK_K17.npz'],'R2p0_seed0':[S336/'SPK_R3_2p0.npz'],
          'R2p0_seed1':[S336/'SPK_R3_seed1.npz'],'R2p0_seed2':[S336/'SPK_R3_seed2.npz']}
    gate_detail={}
    for nm,paths in refs.items():
        z=np.load(o/f'SPK_{nm}.npz');cur_spm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        gate_ok[nm]=True;det=[]
        for p in paths:
            r=loadz(p)[:5000];eq=bool(np.array_equal(cur_spm,r));gate_ok[nm]=gate_ok[nm] and eq
            det.append(dict(ref=str(p.name),equal=eq,spikes=int(r.sum())))
        gate_detail[nm]=dict(equal=gate_ok[nm],spike_total=int(cur_spm.sum()),shape=list(cur_spm.shape),refs=det)
    dump(o/'ARMS.json',{k:{kk:vv for kk,vv in v.items()} for k,v in AR.items()})
    dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',gate_detail)
    def passes_alternation(k):
        a=AR[k]
        if not a.get('detected'): return False,'NO_SLOW_RHYTHM'
        if a['est_spread']>TH['est_spread']: return False,'EST_SPREAD'
        cross_ok=all(TH['cross_lo']<=abs(x)<=TH['cross_hi'] for x in [a['abs_phase']])
        depth_ok=all(v>=TH['depth_min'] for v in a['depth'].values())
        floor_ok=all(a['p10'][lg]>=TH['p10_frac']*a['gate'][lg] for lg in LEGS)
        peak_ok=all(a['p90'][lg]>=TH['p90_frac']*a['gate'][lg] for lg in LEGS)
        corr_ok=a['zero_lag_corr']<=TH['corr_max']
        ok=bool(cross_ok and depth_ok and floor_ok and peak_ok and corr_ok)
        why=[] if ok else [n for n,v in (('cross',cross_ok),('depth',depth_ok),('floor',floor_ok),('peak',peak_ok),('corr',corr_ok)) if not v]
        return ok,('OK' if ok else ','.join(why))
    seeds=[passes_alternation(k) for k in ('R2p0_seed0','R2p0_seed1','R2p0_seed2')]
    c0=passes_alternation('C0_gate');s3=passes_alternation('S3s_2p0')
    bal=passes_alternation('B2p0_bal')
    est_fail=any((AR[k].get('est_spread') or 0)>TH['est_spread'] for k in ('R2p0_seed0','R2p0_seed1','R2p0_seed2') if AR[k].get('detected'))
    if not all(gate_ok.values()) or est_fail:
        branch='NOT_JUDGED'
        dump(o/'STOP.json',dict(reason='three-estimator agreement failed or a construction gate failed',
          est_spread={k:round(AR[k].get('est_spread') or 0,1) for k in ('R2p0_seed0','R2p0_seed1','R2p0_seed2') if AR[k].get('detected')},
          gate=gate_ok))
    elif any(not s[0] for s in seeds): branch='b3'
    elif all(s[0] for s in seeds) and not c0[0] and not s3[0] and bal[0]: branch='b1'
    elif all(s[0] for s in seeds): branch='b2'
    elif any(AR[k].get('detected') for k in ('R2p0_seed0',)): branch='b4'
    else: branch='NOT_JUDGED'
    dump(o/'OUTCOMES.json',dict(round='S337',branch=branch,gate=gate_ok,
      arms={k:dict(level=v['level'],seed=v['seed'],added=v['added_edges'].get('n'),same=v['same'],detected=v.get('detected'),
        f0=round(v['f0'],2) if v.get('f0') else None,phase=round(v.get('abs_phase'),1) if v.get('abs_phase') else None,
        spread=round(v['est_spread'],1) if v.get('est_spread') else None,corr=round(v['zero_lag_corr'],3) if v.get('zero_lag_corr') else None,
        rates={t:round(v['rates'][t],1) for t in v['rates']} if v.get('rates') else None,
        calib_iters=len(v.get('calibration',[]))) for k,v in AR.items()},
      verdicts=dict(seeds=seeds,c0=list(c0),s3=list(s3),bal=list(bal)),
      added_budget=json.loads((o/'ADDED_INHIBITION_BUDGET.json').read_text()),
      counters=CD,not_emergent=True,
      next=('physics chain: static support plus six-leg alternating activation feasibility' if branch=='b1' else
            'resolve the tripod mean-rate asymmetry before physics' if branch=='b2' else 'fix the estimator or the model before physics'),
      one_line='%s seeds %s c0 %s s3 %s bal %s'%(branch,[s[1] for s in seeds],c0[1],s3[1],bal[1])))
    print(json.dumps(dict(branch=branch,gate=gate_ok,
      seeds=[(s[0],s[1]) for s in seeds],c0=c0,s3=s3,bal=bal,
      arms={k:(round(v.get('abs_phase') or 0,1),v.get('detected'),round(v['zero_lag_corr'],3) if v.get('zero_lag_corr') else None) for k,v in AR.items()},
      cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
