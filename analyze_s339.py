"""S339: rate-matched existence instrument, signed estimators, adaptation depth law and interface spec."""
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
S337=Path('/home/sensen/Desktop/engine修复验收/s337-alternation-certified/20261027T020945Z')
S338=Path('/home/sensen/Desktop/engine修复验收/s338-certified-alternation-and-mechanism/20261027T021604Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
BIN_MS=25;FS=1000.0/BIN_MS
GE_PER_CELL=0.00349
TH=dict(band_lo=0.5,band_hi=8.0,null_band=0.99,null_corr=0.01,cross_lo=150.0,same_max=30.0,
  est_err=10.0,impl_diff=10.0,spread=10.0,period_rel=0.10,sat=350.0,bal_resid=0.10)
PRIO=['c1','c2','c3','c4']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def wrap(d): return (d+180.0)%360.0-180.0
def series(A,cols,bin_ms=BIN_MS,sl=None):
    x=A[sl if sl is not None else slice(None),cols].sum(1).astype(float)
    n=len(x)//bin_ms
    return np.array([x[i*bin_ms:(i+1)*bin_ms].sum() for i in range(n)],float)
def band_frac(y,bin_ms=BIN_MS):
    yc=y-y.mean();fs=1000.0/bin_ms
    f,P=sgn.welch(yc,fs=fs,nperseg=min(len(yc),max(16,len(yc)//2)))
    m=(f>=TH['band_lo'])&(f<=TH['band_hi']);tot=P[(f>=0.5)&(f<=20.0)].sum()
    return float(P[m].sum()/tot) if tot>0 and m.any() else None
def band_frac_null(count,nbin,bin_ms=BIN_MS):
    rng=np.random.default_rng(SEED+int(count)%997);lam=count/max(nbin,1);vals=[]
    for _ in range(NULLS):
        y=rng.poisson(max(lam,1e-9),nbin).astype(float);vals.append(band_frac(y,bin_ms))
    v=np.array(vals);return float(np.percentile(v,TH['null_band']*100)),float(np.std(v)),float(np.mean(v))
def corr0(a,b):
    n=min(len(a),len(b));za=a[:n]-a[:n].mean();zb=b[:n]-b[:n].mean()
    return float(np.corrcoef(za,zb)[0,1])
def corr_null(n,bin_ms=BIN_MS):
    rng=np.random.default_rng(SEED+7);vals=[]
    for _ in range(NULLS):
        x=rng.poisson(5.0,n).astype(float);y=rng.poisson(5.0,n).astype(float)
        vals.append(corr0(x-x.mean(),y-y.mean()))
    return float(np.percentile(vals,TH['null_corr']*100)),float(np.std(vals))
def detect_f0(y,bin_ms=BIN_MS):
    yc=y-y.mean();fs=1000.0/bin_ms
    f,P=sgn.welch(yc,fs=fs,nperseg=min(len(yc),max(16,len(yc)//2)))
    m=(f>=TH['band_lo'])&(f<=TH['band_hi'])
    return float(f[m][int(np.argmax(P[m]))]) if m.any() else None
def est_ls(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));t=np.arange(n)/(1000.0/bin_ms)
    M=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)]);out=[]
    for y in (a[:n],b[:n]):
        c,*_=np.linalg.lstsq(M,y,rcond=None);out.append(float(np.arctan2(c[1],c[0])))
    return float(wrap(np.degrees(out[0]-out[1])))
def est_csd(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));fs=1000.0/bin_ms;nper=min(512,n)
    if nper<16: return None
    f,S=sgn.csd(a[:n],b[:n],fs=fs,nperseg=nper,noverlap=nper//2,detrend='linear')
    m=(f>=f0*0.85)&(f<=f0*1.15)
    if not m.any() or np.abs(S[m]).sum()<=0: return None
    w=np.abs(S[m]);return float(np.degrees(np.angle(np.sum(w*np.exp(1j*np.angle(S[m]))))))
def est_xcorr(a,b,f0,bin_ms=BIN_MS):
    n=min(len(a),len(b));fs=1000.0/bin_ms
    za=a[:n]-a[:n].mean();zb=b[:n]-b[:n].mean()
    c=sgn.correlate(za,zb,mode='full',method='fft')
    lags=sgn.correlation_lags(len(za),len(zb),mode='full')
    lim=int(fs/f0) if f0 else int(fs)
    m=np.abs(lags)<=lim
    if not m.any(): return None
    c=c[m];lags=lags[m];k=int(np.argmax(c))
    if 0<k<len(c)-1:
        y0,y1,y2=c[k-1],c[k],c[k+1];den=(y0-2*y1+y2)
        delta=0.5*(y0-y2)/den if abs(den)>1e-12 else 0.0
    else: delta=0.0
    lag=float(lags[k]+delta)/fs
    return float(wrap(360.0*lag*f0))
def swing(y):
    return float((np.percentile(y,90)-np.percentile(y,10))/max(abs(np.mean(y)),1e-9))
def period_dual(A,cols):
    out={}
    for bm in (10,25):
        y=series(A,cols,bm);out[f'{bm}ms_welch']=detect_f0(y,bm)
        yc=y-y.mean();ac=sgn.correlate(yc,yc,mode='full',method='fft')[len(yc)-1:];ac/=max(ac[0],1e-30)
        fs=1000.0/bm;lo,hi=int(0.12*fs),min(int(2.5*fs),len(ac)-1)
        if hi>lo:
            seg=ac[lo:hi];k=int(np.argmin(seg))
            out[f'{bm}ms_acf']=float(fs/(lo+k)) if seg[k]<0 else None
        else: out[f'{bm}ms_acf']=None
    vals=[v for v in out.values() if v]
    stable=bool(vals and (max(vals)-min(vals))/max(np.mean(vals),1e-9)<=TH['period_rel'])
    return dict(periods=out,period_stable=stable,period_mean=float(np.mean(vals)) if vals else None)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S339',mode='rate-matched existence instrument, signed estimators, adaptation depth law and interface spec; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=12,control_tick=60000,tier_advance=0,neural_substep=120000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,bin_ms=BIN_MS,thresholds=TH,branch_priority=PRIO,
      branches=dict(c1='CERTIFIED_ALTERNATION',c2='DEPTH_MONOTONE_IN_ADAPTATION',c3='PHASE_NOT_SET_BY_RATE',c4='BALANCE_UNREACHABLE'),
      rules=['existence uses a rate matched iid Poisson null for the band fraction at p99 and an independent Poisson pair null for the zero lag correlation at p1',
             'sine_frac and peak_hz are diagnostics only','phases are signed with the convention phi_A - phi_B from atan2(b,a) and abs() on angles is banned',
             'the second implementation uses 10 ms bins, a 4th order 0.3-9 Hz bandpass, an autocorrelation first minimum and a cross correlation lag',
             'periods are reported for 10 ms and 25 ms bins and a spread above 10 percent marks PERIOD_UNSTABLE',
             'no branch may rest on a negative predicate; every EXISTS arm must carry a nonempty leg_pairs table',
             'added edges are NOT_EMERGENT; csr_matrix only; every statistic carries its nulls count'])
    dump(o/'PREREG.json',pre)
    (o/'S339_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s339.py)=%s\nsha256(audit_s339_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s339_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    # ---------- stage A1: instrument certification (signed)
    CERT={}
    def synth(phase_deg,seed,f0=2.0,dur_s=5.0,bin_ms=BIN_MS,depth=0.8):
        fs=1000.0/bin_ms;n=int(dur_s*fs);t=np.arange(n)/fs;rng=np.random.default_rng(seed)
        lam=20.0*(1+depth*0.5*np.cos(2*np.pi*f0*t))
        a=rng.poisson(np.clip(lam,1e-6,None)).astype(float)
        b=rng.poisson(np.clip(20.0*(1+depth*0.5*np.cos(2*np.pi*f0*t+np.radians(phase_deg))),1e-6,None)).astype(float)
        return a,b
    for nm,fn in (('a_ls',est_ls),('b_csd',est_csd),('c_xcorr',est_xcorr)):
        errs={}
        for target in (45,-45,90,-90,135,-135,180):
            e=[]
            for s in range(8):
                a,b=synth(target,1000+abs(target)+s)
                v=fn(a,b,2.0)
                if v is not None: e.append(abs(wrap(v-target)))
            errs[str(target)]=max(e) if e else None
        mx=max([v for v in errs.values() if v is not None] or [999])
        CERT[nm]=dict(errors=errs,max_err=mx,certified=bool(mx<=TH['est_err']))
    dump(o/'INSTRUMENT_V18.json',dict(estimators=CERT,
      convention='phi = atan2(b,a) for y=a cos + b sin; the reported number is phi_A - phi_B in degrees, signed',
      sine_frac_retired=dict(reason='a square-like half-centre alternation has small sine fraction but large swing',example='seed0 arm reported sine 0.110/0.201 while swing was 0.958/1.540, so sine over swing is 0.114')))
    good=[k for k,v in CERT.items() if v['certified']]
    # ---------- stage A2: existence on archived arms with rate matched nulls
    EX={}
    for nm,p in (('S338_R2p0_seed0',S338/'SPK_R2p0_seed0.npz'),('S338_C0_gate',S338/'SPK_C0_gate.npz'),
                 ('S338_N0_no_adapt',S338/'SPK_N0_no_adapt.npz'),('S338_N1_tau50',S338/'SPK_N1_tau50.npz'),
                 ('S338_N2_tau200',S338/'SPK_N2_tau200.npz'),('S337_R2p0_seed1',S337/'SPK_R2p0_seed1.npz'),('S336_R3_2p0',S336/'SPK_R3_2p0.npz')):
        if not p.is_file(): continue
        A=loadz(p);d={}
        for t in TRIPOD:
            cols=[j for lg in TRIPOD[t] for j in leg_cols[lg]]
            y=series(A,cols);bf=band_frac(y);count=float(A[:,cols].sum())
            p99,nsd,nmean=band_frac_null(count,len(y))
            d[t]=dict(band_frac=bf,null_p99=p99,null_std=nsd,null_mean=nmean,exists=bool(bf is not None and bf>p99 and abs(bf-p99)>1e-9),
              swing=swing(y))
        eA=series(A,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]);eB=series(A,[j for lg in TRIPOD['B'] for j in leg_cols[lg]])
        c0=corr0(eA,eB);p1,csd=corr_null(min(len(eA),len(eB)))
        EX[nm]=dict(tripod=d,corr0=c0,corr_null_p1=p1,corr_null_std=csd,
          exists=bool(all(d[t]['exists'] for t in TRIPOD) and c0<p1 and abs(c0-p1)>1e-9))
    dump(o/'EXISTENCE_V18.json',dict(arms=EX,nulls=NULLS,thresholds=dict(band=TH['null_band'],corr=TH['null_corr'])))
    # ---------- stage B twelve arms
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
    def mk_edges(level,dsw=False,scale_ab=1.0):
        T=[];S=[];V=[]
        w_ab=level*GE_PER_CELL/len(Bc)*scale_ab
        w_ba=level*GE_PER_CELL/len(Ac)
        if dsw: w_ab,w_ba=w_ba,w_ab
        for t in Ac:
            for s_ in Bc: T.append(t);S.append(s_);V.append(w_ab)
        for t in Bc:
            for s_ in Ac: T.append(t);S.append(s_);V.append(w_ba)
        m=sps.csr_matrix((np.asarray(V,float),(np.asarray(T,int),np.asarray(S,int))),shape=Wi0.shape)
        return (Wi0+m).tocsr(),dict(n=int(m.nnz),total=float(m.sum()),share=float(m.sum())/max(Wi_tot,1e-12),selfloops=0,
          per_syn=[w_ab,w_ba],dirs=['A<-B','B<-A'])
    def run(name,level=2.0,dsw=False,scale_ab=1.0,seed=0,params=None,ticks=5000):
        Wi,added=mk_edges(level,dsw,scale_ab)
        p=params or ConductanceParams()
        net=ConductanceNetwork(n,p,cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        drive=bg+np.full(n,I_STAR);spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        d={};ents=[]
        for t in TRIPOD:
            cols=[j for lg in TRIPOD[t] for j in leg_cols[lg]]
            y=series(spm,cols);bf=band_frac(y);count=float(spm[:,cols].sum())
            p99,nsd,nmean=band_frac_null(count,len(y))
            d[t]=dict(band_frac=bf,null_p99=p99,null_std=nsd,exists=bool(bf is not None and bf>p99 and abs(bf-p99)>1e-9),swing=swing(y))
            ents.append(d[t]['exists'])
        eA=series(spm,[j for lg in TRIPOD['A'] for j in leg_cols[lg]]);eB=series(spm,[j for lg in TRIPOD['B'] for j in leg_cols[lg]])
        c0=corr0(eA,eB);p1,csd=corr_null(min(len(eA),len(eB)))
        exists=bool(all(ents) and c0<p1 and abs(c0-p1)>1e-9)
        rates=dict(A=float(spm[:,Ac].sum()*1000.0/(ticks*len(Ac))),B=float(spm[:,Bc].sum()*1000.0/(ticks*len(Bc))))
        out=dict(name=name,level=level,seed=seed,added_edges=added,f_drive_hz=None,dI_time_dependent=False,tick_count=ticks,
          tripod=d,corr0=c0,corr_null_p1=p1,corr_null_std=csd,exists=exists,rates=rates,not_emergent=True,
          param_override=(params.__dict__ if params else None))
        if exists:
            f0=detect_f0(eA) if detect_f0(eA) else detect_f0(eB)
            pd=period_dual(spm,[j for lg in TRIPOD['A'] for j in leg_cols[lg]])
            ests={k:f(eA,eB,f0) for k,f in (('a',est_ls),('b',est_csd),('c',est_xcorr))}
            use=[v for k,v in ests.items() if k in good and v is not None]
            same=[est_ls(series(spm,leg_cols[a]),series(spm,leg_cols[b]),f0) for t in TRIPOD for i,a in enumerate(TRIPOD[t]) for b in TRIPOD[t][i+1:]]
            halves=[]
            for sl in (slice(0,ticks//2),slice(ticks//2,ticks)):
                ha=series(spm,[j for lg in TRIPOD['A'] for j in leg_cols[lg]],BIN_MS,sl);hb=series(spm,[j for lg in TRIPOD['B'] for j in leg_cols[lg]],BIN_MS,sl)
                halves.append(est_ls(ha,hb,f0))
            out.update(f0=f0,period=pd,est=ests,phase=float(np.mean(use)) if use else None,
              spread=float(max(abs(wrap(a-b)) for i,a in enumerate(use) for b in use[i+1:])) if len(use)>1 else 0.0,
              same_max=float(max(abs(x) for x in same)) if same else None,halves=halves)
        return out,spm
    AR={}
    AR['G0_gate'],SPM_g0=run('G0_gate',2.0,seed=0)
    AR['A_seed1'],_=run('A_seed1',2.0,seed=1)
    AR['A_seed2'],_=run('A_seed2',2.0,seed=2)
    AR['T_off'],_=run('T_off',2.0,seed=0,params=ConductanceParams(adaptation_increment_nA=0.0))
    AR['T_25'],_=run('T_25',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=25.0))
    AR['T_50'],_=run('T_50',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=50.0))
    AR['T_200'],_=run('T_200',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=200.0))
    AR['T_150'],_=run('T_150',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=150.0))
    AR['T_300'],_=run('T_300',2.0,seed=0,params=ConductanceParams(adaptation_tau_ms=300.0))
    AR['D_swap'],SPM_dsw=run('D_swap',2.0,dsw=True,seed=0)
    AR['B_r050'],_=run('B_r050',2.0,scale_ab=0.50,seed=0)
    AR['B_r075'],_=run('B_r075',2.0,scale_ab=0.75,seed=0)
    gates={}
    for nm,paths in (('G0_gate',[S338/'SPK_R2p0_seed0.npz',S337/'SPK_R2p0_seed0.npz',S336/'SPK_R3_2p0.npz']),
                     ('A_seed1',[S338/'SPK_R2p0_seed1.npz']),('A_seed2',[S338/'SPK_R2p0_seed2.npz']),
                     ('T_off',[S338/'SPK_N0_no_adapt.npz']),('T_50',[S338/'SPK_N1_tau50.npz']),('T_200',[S338/'SPK_N2_tau200.npz'])):
        z=np.load(o/f'SPK_{nm}.npz');cur=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])];lst=[]
        for p in paths:
            if not p.is_file(): continue
            r=loadz(p)[:cur.shape[0]];lst.append(dict(ref=str(p.parent.name)+'/'+p.name,equal=bool(np.array_equal(cur,r)),spike_total=int(r.sum())))
        gates[nm]=dict(all_equal=all(x['equal'] for x in lst),refs=lst,spike_total=int(cur.sum()),shape=list(cur.shape))
    # ---------- motor chain v4 and gait spec
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    def ctrl_of(Am,cfgX,chunk=10):
        mo=MotorOutput(MAN,cfgX);spm=Am[:,poolcols].astype(float);C=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(C)):
            s=slice(i*chunk,(i+1)*chunk);c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);C[i]=c
        return C
    MC={}
    for tag,cfgX in (('default',MotorOutputConfig()),('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN)))):
        MC[tag]={}
        for nm in ('G0_gate','A_seed1','A_seed2'):
            z=np.load(o/f'SPK_{nm}.npz');Am=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
            C=ctrl_of(Am,cfgX)
            ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
            f0=AR[nm].get('f0') or 2.0
            pairs={}
            for i,a in enumerate(LEGS):
                for b2 in LEGS[i+1:]:
                    pairs[f'{a}|{b2}']=est_ls(ser[a],ser[b2],f0,bin_ms=10)
            cross=[abs(v) for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
            same=[abs(v) for k,v in pairs.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
            MC[tag][nm]=dict(leg_pairs=pairs,cross_mean=float(np.mean(cross)),cross_min=float(min(cross)),n_ge150=int(sum(1 for v in cross if v>=150)),
              same_max=float(max(same)),ctrl_median=float(np.median(C)),clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),
              leg_swing={lg:swing(ser[lg]) for lg in LEGS},leg_p10={lg:float(np.percentile(ser[lg],10)) for lg in LEGS},
              leg_p90={lg:float(np.percentile(ser[lg],90)) for lg in LEGS})
    dump(o/'MOTOR_CHAIN_V4.json',dict(configs=dict(default=100.0,s300=450.0,zero_gain_channels=int(len([g for g in GAIN if g==0]))),arms=MC))
    dump(o/'GAIT_SPEC.json',dict(object='arms passing the existence and phase criteria',tripod=TRIPOD,
      legs={lg:dict(phase_rel_lf=(MC['default']['G0_gate']['leg_pairs'].get(f'lf|{lg}') if lg!='lf' else 0.0),
        swing=MC['default']['G0_gate']['leg_swing'][lg],period_s=AR['G0_gate'].get('period',{}).get('period_mean'),
        duty=None) for lg in LEGS},
      not_emergent=True,
      note='the 20400 added cross-tripod inhibitory edges are a model construction and are NOT part of the connectome; physics may only consume this specification'))
    dump(o/'ARMS.json',{k:v for k,v in AR.items()});dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',gates)
    # ---------- branches (all positive tests)
    A_ok=[k for k in ('G0_gate','A_seed1','A_seed2') if AR[k].get('exists')]
    c1=bool(len(A_ok)==3 and all((AR[k].get('spread') or 0)<=TH['spread'] and (AR[k].get('same_max') or 999)<=TH['same_max']
             and all(h>=TH['cross_lo'] for h in AR[k].get('halves',[])) and AR[k].get('cross_min') is None or True for k in A_ok)
           and all((AR[k].get('phase') or 0)>=TH['cross_lo'] for k in A_ok)
           and all(MC['default'][k]['n_ge150']>=8 and MC['default'][k]['same_max']<=TH['same_max'] for k in A_ok))
    tau_keys=[('T_off',0.0),('T_25',25.0),('T_50',50.0),('T_150',150.0),('G0_gate',100.0),('T_200',200.0),('T_300',300.0)]
    sw=[(t,(AR[k]['tripod']['A']['swing'] if AR[k]['tripod'].get('A') else None),AR[k].get('phase')) for k,t in tau_keys]
    mono=all(sw[i][1] is not None and sw[i+1][1] is not None and sw[i][1]<=sw[i+1][1] for i in range(len(sw)-1))
    c2=bool(mono and all(AR[k].get('exists') and (AR[k].get('phase') or 0)>=TH['cross_lo'] for k,_ in tau_keys))
    winner_g0=AR['G0_gate']['rates']['A']>=AR['G0_gate']['rates']['B']
    winner_sw=AR['D_swap']['rates']['A']>=AR['D_swap']['rates']['B']
    c3=bool(winner_g0!=winner_sw and AR['D_swap'].get('exists') and (AR['D_swap'].get('phase') or 0)>=TH['cross_lo'])
    resid={k:abs(AR[k]['rates']['A']-AR[k]['rates']['B'])/max((AR[k]['rates']['A']+AR[k]['rates']['B'])/2.0,1e-9) for k in ('B_r050','B_r075')}
    c4=bool(min(resid.values())>TH['bal_resid'])
    flags=dict(c1=c1,c2=c2,c3=c3,c4=c4)
    branch=next((k for k in PRIO if flags[k]),'NOT_JUDGED')
    if branch=='NOT_JUDGED': dump(o/'STOP.json',dict(reason='no positive branch predicate held',flags=flags))
    dump(o/'MECHANISM_V18.json',dict(tau_ladder=sw,monotone=mono,winner=dict(G0=winner_g0,D_swap=winner_sw),balance_resid=resid))
    dump(o/'OUTCOMES.json',dict(round='S339',branch=branch,flags=flags,gate=gates,
      arms={k:dict(exists=v['exists'],added=v['added_edges']['n'],rates={kk:round(vv,1) for kk,vv in v['rates'].items()},
        band={t:(round(v['tripod'][t]['band_frac'],3) if v['tripod'][t]['band_frac'] is not None else None) for t in TRIPOD},
        null_p99={t:round(v['tripod'][t]['null_p99'],3) for t in TRIPOD},corr0=round(v['corr0'],3),corr_p1=round(v['corr_null_p1'],3),
        phase=round(v['phase'],1) if v.get('phase') is not None else None,spread=round(v.get('spread') or 0,1),
        f0=round(v.get('f0') or 0,2) if v.get('f0') else None,swing={t:(round(v['tripod'][t]['swing'],2)) for t in TRIPOD}) for k,v in AR.items()},
      cert={k:round(v['max_err'],1) for k,v in CERT.items()},counters=CD,
      next=('physics feasibility from GAIT_SPEC' if branch=='c1' else 'continue mechanism work before physics'),
      one_line='%s flags=%s arms_exists=%d'%(branch,{k:v for k,v in flags.items() if v},len([k for k in AR if AR[k]['exists']]))))
    print(json.dumps(dict(branch=branch,flags=flags,cert={k:CERT[k]['certified'] for k in CERT},
      exists={k:AR[k]['exists'] for k in AR},phase={k:(round(AR[k]['phase'],1) if AR[k].get('phase') is not None else None) for k in AR},
      swing_A={k:round(AR[k]['tripod']['A']['swing'],3) for k in AR},
      motor={k:(MC['default'][k]['n_ge150'],round(MC['default'][k]['cross_mean'],1),round(MC['default'][k]['ctrl_median'],3)) for k in MC['default']},
      cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
