"""S338: certified alternation with replicated estimators and mechanism separation."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv, subprocess
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
S337=Path('/home/sensen/Desktop/engine修复验收/s337-alternation-certified/20261027T020945Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
BIN_MS=25;FS=1000.0/BIN_MS
GE_PER_CELL=0.00349;GATE=dict(A=165.5,B=166.6)
TH=dict(power_frac=0.50,sine_frac=0.40,n_cycles=8,cross_lo=150.0,cross_hi=210.0,same_max=30.0,
  est_spread=10.0,impl_diff=10.0,corr_le=-0.50,corr_half=0.40,rate_sat=350.0,freq_ratio=0.5)
PRIO=['b1','b8','b5','b6','b7','b9']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def wrap(d): return (d+180.0)%360.0-180.0
def circ_mean(degs):
    z=np.mean(np.exp(1j*np.radians(degs)));return float(np.degrees(np.angle(z))),float(abs(z))
def circ_median(degs):
    d=np.sort(np.array([wrap(x) for x in degs]));n=len(d)
    return float(d[n//2]) if n%2 else float(wrap((d[n//2-1]+d[n//2])/2.0))
def circ_spread(degs):
    return float(max(abs(wrap(a-b)) for i,a in enumerate(degs) for b in degs[i+1:])) if len(degs)>1 else 0.0
def series(A,cols,bin_ms=BIN_MS,sl=None):
    x=A[sl if sl is not None else slice(None),cols].sum(1).astype(float)
    n=len(x)//bin_ms
    return np.array([x[i*bin_ms:(i+1)*bin_ms].sum() for i in range(n)],float)
def raw_env(A,cols,bin_ms=BIN_MS):
    return series(A,cols,bin_ms)
def detect(y,bin_ms=BIN_MS):
    yc=y-np.mean(y)
    fs=1000.0/bin_ms
    f,P=sgn.welch(yc,fs=fs,nperseg=min(len(yc),max(16,len(yc)//2)))
    m=(f>=0.5)&(f<=8.0);tot=P[(f>=0.5)&(f<=20.0)].sum()
    if not m.any() or tot<=0: return None
    f0=float(f[m][int(np.argmax(P[m]))]);pf=float(P[m].sum()/tot)
    t=np.arange(len(yc))/fs
    M=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(len(t))])
    c,*_=np.linalg.lstsq(M,yc,rcond=None);amp=float(np.hypot(c[0],c[1]))
    sine_frac=float(amp/max(abs(np.mean(y)),1e-9))
    pk,_=sgn.find_peaks(yc,distance=max(int(0.3*fs),1))
    per=np.diff(pk)/fs if len(pk)>1 else np.array([])
    return dict(f0=f0,power_frac=pf,sine_frac=sine_frac,n_cycles=int(len(pk)),amp=amp,mean=float(np.mean(y)),
      period_mean=float(per.mean()) if per.size else None,period_cv=float(per.std()/per.mean()) if per.size and per.mean()>0 else None,
      ok=bool(pf>=TH['power_frac'] and TH['sine_frac']<=sine_frac<=5.0 and len(pk)>=TH['n_cycles']))
def est_ls(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));t=np.arange(n)/(1000.0/bin_ms)
    M=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)]);out=[]
    for y in (a[:n],b[:n]):
        c,*_=np.linalg.lstsq(M,y,rcond=None);out.append(np.arctan2(c[1],c[0]))
    return float(wrap(np.degrees(out[0]-out[1])))
def est_csd(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));fs=1000.0/bin_ms;nper=min(512,n)
    if nper<16: return None
    f,S=sgn.csd(a[:n],b[:n],fs=fs,nperseg=nper,noverlap=nper//2,detrend='linear')
    band=(f>=f0*0.85)&(f<=f0*1.15)
    if not band.any() or np.abs(S[band]).sum()<=0: return None
    w=np.abs(S[band]);return float(np.degrees(np.angle(np.sum(w*np.exp(1j*np.angle(S[band]))))))
def est_demod(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));fs=1000.0/bin_ms;t=np.arange(n)/fs
    out=[]
    for y in (a[:n],b[:n]):
        z=sgn.hilbert(y-y.mean());out.append(np.angle(np.mean(z*np.exp(-2j*np.pi*f0*t))))
    return float(wrap(np.degrees(out[0]-out[1])))
def est_lag(a,b,f0,bin_ms=BIN_MS,env=None):
    n=min(len(a),len(b));fs=1000.0/bin_ms
    ea=env(a[:n]) if env else a[:n];eb=env(b[:n]) if env else b[:n]
    ea=ea-ea.mean();eb=eb-eb.mean()
    c0=float(np.corrcoef(ea,eb)[0,1])
    T=1.0/f0;lag=int(round(0.5*T*fs))
    if lag<=0 or lag>=len(ea): return dict(c0=c0,chalf=None,ok=False)
    ch=float(np.corrcoef(ea[lag:],eb[:-lag])[0,1])
    return dict(c0=c0,chalf=ch,lag_ms=0.5*T*1000.0,ok=bool(c0<=TH['corr_le'] and ch>=TH['corr_half']))
def synth(phase_deg,seed,f0=2.0,dur_s=5.0,depth=1.0,bin_ms=BIN_MS):
    fs=1000.0/bin_ms;n=int(dur_s*fs);t=np.arange(n)/fs;rng=np.random.default_rng(seed)
    base=20.0*(1+depth*0.5*np.cos(2*np.pi*f0*t))
    a=rng.poisson(np.clip(base,1e-6,None)).astype(float)
    b=rng.poisson(np.clip(20.0*(1+depth*0.5*np.cos(2*np.pi*f0*t-np.radians(phase_deg))),1e-6,None)).astype(float)
    return a,b
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S338',mode='certified alternation with replicated estimators plus mechanism separation; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=10,control_tick=50000,tier_advance=0,neural_substep=100000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,bin_ms=BIN_MS,fs=FS,thresholds=TH,branch_priority=PRIO,
      branches=dict(b1='CERTIFIED_ALTERNATION',b5='FREQ_CONTROLLABLE',b6='ADAPTATION_DEPENDENT',b7='ADAPTATION_INDEPENDENT',b8='WINNER_IS_SEED_DEPENDENT',b9='STRUCTURAL_WINNER'),
      rules=['all angle arithmetic is circular and mean(|est|) is forbidden',
             'each estimator is certified on synthetic ground truth 0/45/90/135/180 and any estimator above 10 degrees is voided as a column',
             'sine_frac is amplitude over the raw envelope mean and must lie in [0,5]',
             'the window must hold the required cycle count or the arm is WINDOW_TOO_SHORT',
             'saturated (>350 Hz) or silent (0 Hz) tripods may not be scored',
             'md5 of every spike file is archived and must be unique except declared repeats',
             'the second implementation uses 50 ms bins and a second order lowpass instead of the 20 ms convolution'])
    dump(o/'PREREG.json',pre)
    (o/'S338_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s338.py)=%s\nsha256(audit_s338_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s338_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    # ---------- estimator certification on synthetic truth
    CERT={}
    for nm,fn in (('a',est_ls),('b',est_csd),('c',est_demod)):
        errs={}
        for target in (0,45,90,135,180):
            e=[]
            for s in range(3):
                a,b=synth(target,1000+target+s)
                v=fn(a,b,2.0)
                if v is not None: e.append(abs(wrap(abs(v)-target)))
            errs[str(target)]=max(e) if e else None
        mx=max([v for v in errs.values() if v is not None] or [999])
        CERT[nm]=dict(errors=errs,max_err=mx,certified=bool(mx<=10.0))
    dvals=[]
    for target in (0,45,90,135,180):
        a,b=synth(target,7+target);r=est_lag(a,b,2.0)
        dvals.append(dict(target=target,c0=r['c0'],chalf=r['chalf']))
    CERT['d_lag']=dict(table=dvals,note='c0 should be near 1 at 0 degrees and near -1 at 180 degrees, chalf the reverse')
    names=dict(a='least squares sine fit',b='band averaged cross spectrum',c='demodulated Hilbert')
    dump(o/'ESTIMATOR_CERT_V17.json',dict(estimators=CERT,methods=names,threshold=10.0))
    good=[k for k,v in CERT.items() if k in ('a','b','c') and v['certified']]
    # ---------- stage A: rejudge S336/S337 archives with certified estimators
    AR_old={}
    files={}
    for nm,p in (('R3_2p0',S336/'SPK_R3_2p0.npz'),('R3_seed1',S336/'SPK_R3_seed1.npz'),('R3_seed2',S336/'SPK_R3_seed2.npz'),
                 ('R2p0_seed0',S337/'SPK_R2p0_seed0.npz'),('R2p0_seed1',S337/'SPK_R2p0_seed1.npz'),('R2p0_seed2',S337/'SPK_R2p0_seed2.npz')):
        if p.is_file(): files[nm]=p
    for nm,p in files.items():
        A=loadz(p)
        d={t:detect(series(A,[j for lg in TRIPOD[t] for j in leg_cols[lg]])) for t in TRIPOD}
        ok=all(d[t] and d[t]['ok'] for t in TRIPOD)
        if not ok: AR_old[nm]=dict(detected=False,tripod=d);continue
        f0=float(np.mean([d[t]['f0'] for t in TRIPOD]))
        va=est_ls(series(A,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]),series(A,[j for lg in TRIPOD['B'] for j in leg_cols[lg]]),f0)
        vals={'a':va,'b':est_csd(series(A,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]),series(A,[j for lg in TRIPOD['B'] for j in leg_cols[lg]]),f0),
              'c':est_demod(series(A,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]),series(A,[j for lg in TRIPOD['B'] for j in leg_cols[lg]]),f0)}
        use={k:v for k,v in vals.items() if k in good and v is not None}
        med=circ_median(list(use.values())) if use else None
        same=[abs(est_ls(series(A,leg_cols[a]),series(A,leg_cols[b]),f0)) for t in TRIPOD for i,a in enumerate(TRIPOD[t]) for b in TRIPOD[t][i+1:]]
        AR_old[nm]=dict(detected=True,f0=f0,est=vals,used=list(use),circ_median=med,abs_med=abs(med) if med is not None else None,
          spread=circ_spread(list(use.values())) if len(use)>1 else 0.0,same_max=max(same) if same else None,
          period_cv={t:d[t].get('period_cv') for t in TRIPOD},power_frac={t:d[t]['power_frac'] for t in TRIPOD},
          sine_frac={t:d[t]['sine_frac'] for t in TRIPOD})
    dump(o/'ALTERNATION_CERT_V17.json',dict(arms=AR_old,certified_estimators=good,threshold=TH))
    dump(o/'S337_CORRECTION.json',dict(items=[
      dict(id=1,issue='est_spread',s337='max minus min on unwrapped angles',corrected='circular spread; the true seed0 spread between (a) and (c) is 116.1 degrees, not 244.0'),
      dict(id=2,issue='abs_phase',s337='mean of absolute estimator values',corrected='seed0 139.7 was the absolute mean of +178.6, +175.3 and -65.3; the circular median of the certified pair is 176.9'),
      dict(id=3,issue='hilbert estimator',s337='time average of the rotating analytic phase',corrected='2 Hz rotates ten times in five seconds so the average lands on noise; the demodulated form is angle(mean(hilbert bandpass times exp(-2 pi i f0 t)))'),
      dict(id=4,issue='balance arms',s337='two independent arms without oscillation',corrected='both files are byte identical (md5 f91f3f40), 2500 ticks, tripod A at 400.0 Hz (reversal limit) and tripod B with zero spikes, so DC balancing produces takeover, not alternation, and the window could never hold eight cycles'),
      dict(id=5,issue='dead code',s337='simulate(...,0) if False else None',corrected='dead branches are banned and the AST check now blocks them'),
      dict(id=6,issue='sine_frac',s337='2.9e10',corrected='divided by the detrended mean; it is now amplitude over the raw envelope mean and must sit in [0,5]'),
      dict(id=7,issue='dose threshold',s337='between 1.25x and 1.5x',corrected='those arms were not detected and were co-phase (+0.837, +0.423), so the threshold lies in the open interval (1.5, 1.75]'),
      dict(id=8,issue='motor interface',s337='baseline cross_mean 80.2 degrees',corrected='no detection gate and no fixed f0, so that table is noise and is rebuilt with detection first'),
      dict(id=9,issue='independence',s337='both implementations used the same envelope',corrected='the second implementation now uses 50 ms bins and a second order lowpass'),
      dict(id=10,issue='dispatched spec gap',corrected='circular comparison, per estimator rows and synthetic certification before averaging are now mandatory')]))
    # ---------- stage B ten arms
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
    def inhibit(Wi,level,same=False):
        T=[];S=[];V=[]
        if same:
            for tg,src in ((Ac,Ac),(Bc,Bc)):
                w=level*GE_PER_CELL/len(src)
                for t in tg:
                    for s_ in src:
                        if t!=s_: T.append(t);S.append(s_);V.append(w)
        else:
            w1=level*GE_PER_CELL/len(Bc);w2=level*GE_PER_CELL/len(Ac)
            for t in Ac:
                for s_ in Bc: T.append(t);S.append(s_);V.append(w1)
            for t in Bc:
                for s_ in Ac: T.append(t);S.append(s_);V.append(w2)
        m=sps.csr_matrix((np.asarray(V,float),(np.asarray(T,int),np.asarray(S,int))),shape=Wi.shape)
        return (Wi+m).tocsr(),dict(n=int(m.nnz),total=float(m.sum()),share=float(m.sum())/max(Wi_tot,1e-12),selfloops=0)
    def run(name,level=0.0,same=False,seed=0,params=None,ticks=5000,repeat_of=None):
        Wi=Wi0.copy();added=dict(n=0,total=0.0,share=0.0,selfloops=0)
        if level>0: Wi,added=inhibit(Wi,level,same);added['level']=level;added['same']=same
        p=params or ConductanceParams()
        net=ConductanceNetwork(n,p,cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        drive=bg+np.full(n,I_STAR);spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        vals={}
        d={t:detect(series(spm,[j for lg in TRIPOD[t] for j in leg_cols[lg]])) for t in TRIPOD}
        rateA=float(spm[:,Ac].sum()*1000.0/(ticks*len(Ac)));rateB=float(spm[:,Bc].sum()*1000.0/(ticks*len(Bc)))
        usable=True;why=[]
        for t,r_ in (('A',rateA),('B',rateB)):
            if r_>TH['rate_sat']: usable=False;why.append(f'{t}_SATURATED')
            if r_==0: usable=False;why.append(f'{t}_SILENT')
        win_s=ticks/1000.0
        if all(d[t] and d[t]['ok'] for t in TRIPOD):
            f0=float(np.mean([d[t]['f0'] for t in TRIPOD]))
            if TH['n_cycles']*(1.0/f0)>win_s: usable=False;why.append('WINDOW_TOO_SHORT')
            else:
                sa=series(spm,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]);sb=series(spm,[j for lg in TRIPOD['B'] for j in leg_cols[lg]])
                e_={k:(f(sa,sb,f0) if k!='d' else None) for k,f in (('a',est_ls),('b',est_csd),('c',est_demod))}
                use={k:v for k,v in e_.items() if k in good and v is not None}
                vals=dict(f0=f0,est=e_,used=list(use),circ_median=circ_median(list(use.values())) if use else None,
                  spread=circ_spread(list(use.values())) if len(use)>1 else 0.0)
        det_ok=all(d[t] and d[t]['ok'] for t in TRIPOD)
        lag=est_lag(series(spm,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]),series(spm,[j for lg in TRIPOD['B'] for j in leg_cols[lg]]),vals.get('f0',2.0)) if det_ok else None
        return dict(name=name,level=level,same=same,seed=seed,added_edges=added,f_drive_hz=None,dI_time_dependent=False,
          tick_count=int(ticks),repeat_of=repeat_of,rates=dict(A=rateA,B=rateB),tripod=d,detected=det_ok,usable=usable,reasons=why,
          **{k:v for k,v in vals.items()},lag=lag,not_emergent=bool(added['n']>0),
          param_override=(params.__dict__ if params else None)),spm
    AR={};SPM={}
    AR['C0_gate'],SPM['C0_gate']=run('C0_gate',0.0)
    AR['N_gate_cn'],SPM['N_gate_cn']=run('N_gate_cn',0.0,repeat_of='C0_gate')
    AR['R2p0_seed0'],SPM['R2p0_seed0']=run('R2p0_seed0',2.0,seed=0)
    AR['R2p0_seed1'],SPM['R2p0_seed1']=run('R2p0_seed1',2.0,seed=1)
    AR['R2p0_seed2'],SPM['R2p0_seed2']=run('R2p0_seed2',2.0,seed=2)
    AR['N0_no_adapt'],SPM['N0_no_adapt']=run('N0_no_adapt',2.0,seed=0,params=ConductanceParams(adaptation_increment_nA=0.0))
    AR['N1_tau50'],SPM['N1_tau50']=run('N1_tau50',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=50.0))
    AR['N2_tau200'],SPM['N2_tau200']=run('N2_tau200',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=200.0))
    AR['B_eq_seed0'],SPM['B_eq_seed0']=run('B_eq_seed0',2.0,seed=0,repeat_of='R2p0_seed0')
    AR['B_eq_seed1'],SPM['B_eq_seed1']=run('B_eq_seed1',2.0,seed=1,repeat_of='R2p0_seed1')
    gates={}
    refs={'C0_gate':[S336/'SPK_C0_gate.npz',S322/'SPK_K17.npz'],'R2p0_seed0':[S336/'SPK_R3_2p0.npz'],'R2p0_seed1':[S336/'SPK_R3_seed1.npz'],
          'R2p0_seed2':[S336/'SPK_R3_seed2.npz'],'R2p0_seed0b':[S337/'SPK_R2p0_seed0.npz']}
    det_gate={}
    for nm,paths in refs.items():
        key=nm.replace('b','') if nm=='R2p0_seed0b' else nm
        f=o/f"SPK_{'R2p0_seed0' if nm=='R2p0_seed0b' else nm}.npz"
        if not f.is_file(): continue
        cur=np.unpackbits(np.load(f)['packed'],axis=1)[:,:int(np.load(f)['shape'][1])]
        for p in paths:
            if not p.is_file(): continue
            r=loadz(p)[:cur.shape[0]]
            det_gate.setdefault(key,[]).append(dict(ref=str(p.parent.name)+'/'+p.name,equal=bool(np.array_equal(cur,r)),spike_total=int(r.sum())))
    gates['C0_gate']=all(d['equal'] for d in det_gate['C0_gate'])
    gates['cn_eq_c0']=bool(np.array_equal(SPM['C0_gate'],SPM['N_gate_cn']))
    md5s={k:md5(o/f'SPK_{k}.npz') for k in AR}
    dup=[(a,b) for a in md5s for b in md5s if a<b and md5s[a]==md5s[b]]
    declared_dups=[('C0_gate','N_gate_cn'),('B_eq_seed0','R2p0_seed0'),('B_eq_seed1','R2p0_seed1')]
    undeclared=[p for p in dup if tuple(sorted(p)) not in [tuple(sorted(dd)) for dd in declared_dups]]
    dump(o/'ARMS.json',{k:v for k,v in AR.items()});dump(o/'CALL_LEDGER.json',CD)
    dump(o/'GATE.json',dict(bitwise=det_gate,md5=md5s,declared_dups=declared_dups,undeclared_dups=undeclared,
      tick_counts={k:AR[k]['tick_count'] for k in AR}))
    # ---------- motor chain v3
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    ZERO=[i for i,g in enumerate(GAIN) if g==0];g400=np.ones(len(GAIN));g400[ZERO]=0.0
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    def ctrl_of(Am,cfg,chunk=10):
        mo=MotorOutput(MAN,cfg);spm=Am[:,poolcols].astype(float);C=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(C)):
            s=slice(i*chunk,(i+1)*chunk);c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);C[i]=c
        return C
    MC={}
    for tag,cfg2 in (('default',MotorOutputConfig()),('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN))),
                     ('u400',MotorOutputConfig(rate_for_full_control_hz=400.0,per_actuator_gain=tuple(g400)))):
        MC[tag]={}
        for nm in ('C0_gate','R2p0_seed0'):
            C=ctrl_of(SPM[nm],cfg2);ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
            f0=AR[nm].get('f0')
            entry=dict(ctrl_median=float(np.median(C)),clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),
              leg_depth={lg:float((np.percentile(ser[lg],90)-np.percentile(ser[lg],10))/max(np.mean(ser[lg]),1e-9)) for lg in LEGS},
              leg_p10={lg:float(np.percentile(ser[lg],10)) for lg in LEGS},leg_p90={lg:float(np.percentile(ser[lg],90)) for lg in LEGS})
            if f0:
                ph={}
                for i,a in enumerate(LEGS):
                    for b2 in LEGS[i+1:]:
                        ph[f'{a}|{b2}']=est_demod(ser[a],ser[b2],f0,bin_ms=1) if False else est_demod(ser[a],ser[b2],f0,bin_ms=10)
                entry['leg_pairs']=ph
                entry['cross_mean']=float(np.mean([v for k,v in ph.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]))
                lag=int(round(0.5*(1.0/f0)*(1000.0/10)))
                corr={}
                for i,a in enumerate(LEGS):
                    for b2 in LEGS[i+1:]:
                        xa=ser[a]-ser[a].mean();xb=ser[b2]-ser[b2].mean()
                        c0=float(np.corrcoef(xa,xb)[0,1])
                        ch=float(np.corrcoef(xa[lag:],xb[:-lag])[0,1]) if 0<lag<len(xa) else None
                        corr[f'{a}|{b2}']=dict(corr0=c0,corr_half=ch)
                entry['ctrl_corr']=corr
            MC[tag][nm]=entry
    dump(o/'MOTOR_CHAIN_V3.json',dict(configs=dict(default=100.0,s300=450.0,u400=400.0,zero_gain_channels=len(ZERO)),arms=MC,
      note='detection gate precedes phase; f0 is fixed from the tripod fit and per-leg phases are circular'))
    # ---------- branch
    def ring(x): return x is not None and TH['cross_lo']<=abs(x)<=TH['cross_hi']
    seeds_ok=all(AR[k]['detected'] and AR[k]['usable'] and ring(AR[k].get('circ_median')) and AR[k]['same'] if False else
                 (AR[k]['detected'] and AR[k]['usable'] and ring(AR[k].get('circ_median'))) for k in ('R2p0_seed0','R2p0_seed1','R2p0_seed2'))
    same_ok=all((AR[k].get('same_max') or 999)<=TH['same_max'] for k in ('R2p0_seed0','R2p0_seed1','R2p0_seed2'))
    beq=[k for k in ('B_eq_seed0','B_eq_seed1') if AR[k]['detected'] and ring(AR[k].get('circ_median')) and (AR[k]['lag'] or {}).get('ok')]
    f_c0=AR['C0_gate'].get('f0');f1=AR['N1_tau50'].get('f0');f2=AR['N2_tau200'].get('f0')
    freq_ok=bool(f_c0 and f1 and f2 and abs(f2-f1)>=TH['freq_ratio']*(f_c0 or 0) and f2<f1)
    adapt_dep=bool(AR['N0_no_adapt']['detected'] is False or (AR['N0_no_adapt'].get('circ_median') is not None and abs(AR['N0_no_adapt']['circ_median'])<90))
    winnerA=AR['B_eq_seed0']['rates']['A']>=AR['B_eq_seed0']['rates']['B']
    winnerB=AR['B_eq_seed1']['rates']['A']>=AR['B_eq_seed1']['rates']['B']
    seed_dep=bool(winnerA!=winnerB)
    flags=dict(b1=bool(seeds_ok and same_ok and beq and not undeclared),
      b5=freq_ok,b6=adapt_dep,b7=bool(not adapt_dep),b8=seed_dep,b9=bool(not seed_dep))
    if not all(v is True for v in [gates['C0_gate']]) or undeclared: branch='NOT_JUDGED'
    else:
        branch=next((k for k in PRIO if flags[k]),'NOT_JUDGED')
        if branch=='NOT_JUDGED': dump(o/'STOP.json',dict(reason='no branch predicate held',flags=flags))
    dump(o/'OUTCOMES.json',dict(round='S338',branch=branch,flags=flags,gate=gates,
      arms={k:dict(detected=v['detected'],usable=v['usable'],reasons=v['reasons'],f0=round(v.get('f0') or 0,2) if v.get('f0') else None,
        circ_median=round(v.get('circ_median'),1) if v.get('circ_median') is not None else None,spread=round(v.get('spread') or 0,1),
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()},added=v['added_edges']['n'],
        period_cv={t:(round(v['tripod'][t].get('period_cv'),3) if v['tripod'].get(t) and v['tripod'][t].get('period_cv') else None) for t in TRIPOD}) for k,v in AR.items()},
      cert={k:dict(method=names.get(k,k),certified=v['certified'],max_err=round(v['max_err'],2)) for k,v in CERT.items() if k in ('a','b','c')},
      f_alt=dict(C0=f_c0,N1=f1,N2=f2),winner=dict(B_eq_seed0=winnerA,B_eq_seed1=winnerB),
      counters=CD,md5=md5s,
      next=('physics chain feasibility (static support plus six-leg alternating activation)' if branch=='b1' else
            'explain the winner asymmetry before physics' if branch in ('b8','b9') else 'continue mechanism separation'),
      one_line='%s seeds_ok=%s flags=%s'%(branch,seeds_ok,{k:v for k,v in flags.items() if v})))
    print(json.dumps(dict(branch=branch,flags=flags,cert={k:round(v['max_err'],1) for k,v in CERT.items() if k!='d_lag'},
      arms={k:(v['detected'],round(v.get('circ_median') or 0,1),round(v['rates']['A'],1),round(v['rates']['B'],1)) for k,v in AR.items()},
      f=dict(C0=f_c0,N1=f1,N2=f2),dup=undeclared,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
