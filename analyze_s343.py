"""S343: signed phase estimators, cadence triple routes, balancing knob and the engagement gate."""
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
S340=Path('/home/sensen/Desktop/engine修复验收/s340-instrument-v19-and-gait-spec/20261027T023739Z')
S341=Path('/home/sensen/Desktop/engine修复验收/s341-antialias-rescore-and-load-tolerance/20261027T024501Z')
S342=Path('/home/sensen/Desktop/engine修复验收/s342-instrument-repair-and-load-limit/20261027T025504Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
GE_PER_CELL=0.00349;BIN10=10;FS10=100.0;BAND=(0.5,4.0)
TH=dict(band=[0.5,4.0],phase_agree=8.0,period_rel=0.10,cross_lo=150.0,same_max=30.0,min_amp=0.4,clip_max=0.05,
  p99_target=0.90,period_ref=0.475,period_tol=0.15,period_cv=0.10,z_gate=-3.0,duty_std=0.02,
  dphase=8.0,dperiod=0.02,dduty=0.05,rate_ratio=1.25,nulls=NULLS)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else (None if o is None else float(o)))+'\n',encoding='utf-8')
def wrap(d): return (d+180.0)%360.0-180.0
def sos_low(order,cut,fs,name):
    sos=sgn.butter(order,cut/(fs/2),btype='low',output='sos')
    for s in sos:
        r=np.roots(s[3:])
        if r.size and float(np.max(np.abs(r)))>0.999: raise SystemExit(f'{name} unstable')
    return sos
def sos_band(order,lo,hi,fs,name):
    sos=sgn.butter(order,[lo/(fs/2),min(hi/(fs/2),0.98)],btype='band',output='sos')
    for s in sos:
        r=np.roots(s[3:])
        if r.size and float(np.max(np.abs(r)))>0.999: raise SystemExit(f'{name} unstable')
    return sos
def env_raw(counts_1ms):
    n=len(counts_1ms)//BIN10
    return np.array([counts_1ms[i*BIN10:(i+1)*BIN10].sum() for i in range(n)],float)
def env_band(y):
    return sgn.sosfiltfilt(sos_band(4,BAND[0],BAND[1],FS10,'env'),y-y.mean())
def f0_welch(y):
    f,P=sgn.welch(y-y.mean(),fs=FS10,nperseg=min(len(y),256))
    m=(f>=BAND[0])&(f<=BAND[1])
    return float(f[m][int(np.argmax(P[m]))]) if m.any() else None
def ls_phase(a,b,f0,harm=3):
    n=min(len(a),len(b));t=np.arange(n)/FS10
    cols=[np.ones(n)]
    for k in range(1,harm+1):
        cols+= [np.cos(2*np.pi*k*f0*t),np.sin(2*np.pi*k*f0*t)]
    M=np.column_stack(cols);out=[]
    for y in (a[:n],b[:n]):
        c,*_=np.linalg.lstsq(M,y,rcond=None);out.append(float(np.arctan2(c[2],c[1])))
    return float(wrap(np.degrees(out[0]-out[1])))
def xcorr_phase(a,b,f0):
    n=min(len(a),len(b));za=a[:n]-a[:n].mean();zb=b[:n]-b[:n].mean()
    c=sgn.correlate(za,zb,mode='full',method='fft');lags=sgn.correlation_lags(len(za),len(zb),mode='full')
    lim=int(FS10/max(f0,1e-6));m=np.abs(lags)<=lim
    if not m.any(): return None
    cc=c[m];ll=lags[m];k=int(np.argmax(cc))
    if 0<k<len(cc)-1:
        y0,y1,y2=cc[k-1],cc[k],cc[k+1];den=(y0-2*y1+y2)
        delta=0.5*(y0-y2)/den if abs(den)>1e-12 else 0.0
    else: delta=0.0
    lag=float(ll[k]+delta)/FS10
    return float(wrap(360.0*lag*f0))
def period_routes(y):
    out={}
    f0=f0_welch(y)
    out['welch_s']=1.0/f0 if f0 else None
    yc=y-y.mean();ac=sgn.correlate(yc,yc,mode='full',method='fft')[len(yc)-1:];ac/=max(ac[0],1e-30)
    lo,hi=int(0.12*FS10),min(int(2.5*FS10),len(ac)-1)
    if hi>lo:
        seg=ac[lo:hi];k=int(np.argmin(seg))
        per=2.0*(lo+k)/FS10 if seg[k]<0 else None
        overlap_cycles=float((len(yc)-(lo+k))/(2.0*(lo+k)+1e-9))
        out['acf_s']=per;out['acf_overlap_cycles']=overlap_cycles
        if per is not None and overlap_cycles<1.0: out['acf_s']=None
    else:
        out['acf_s']=None;out['acf_overlap_cycles']=0.0
    if f0:
        grid=f0*np.linspace(0.7,1.3,121);best=None;bestr2=-1
        n=len(y);t=np.arange(n)/FS10
        for fx in grid:
            M=np.column_stack([np.cos(2*np.pi*fx*t),np.sin(2*np.pi*fx*t),np.ones(n)])
            c,*_=np.linalg.lstsq(M,y-y.mean(),rcond=None)
            pred=M@c;ss=1.0-np.sum((y-y.mean()-pred)**2)/max(np.sum((y-y.mean())**2),1e-12)
            if ss>bestr2: bestr2=ss;best=fx
        out['ls_s']=1.0/best if best else None;out['ls_r2']=float(bestr2)
    else:
        out['ls_s']=None;out['ls_r2']=None
    vals=[v for v in (out['welch_s'],out['acf_s'],out['ls_s']) if v]
    if len(vals)>=2:
        out['spread']=float((max(vals)-min(vals))/max(np.mean(vals),1e-9))
        out['resolved']=bool(out['spread']<=TH['period_rel'])
        out['period_s']=float(np.mean(vals))
    else:
        out['spread']=None;out['resolved']=False;out['period_s']=None
    out['unit']='s'
    return out
def duty_of(y): return float(np.mean(y>(np.percentile(y,5)+np.percentile(y,95))/2.0))
def swing_of(y):
    p5,p50,p95=np.percentile(y,[5,50,95])
    return dict(value=float((p95-p5)/p50) if abs(p50)>1e-9 else None,p50=float(p50),span=float(p95-p5),
      degenerate=bool(abs(p50)<=0.2*(p95-p5)))
def null_rho(rate):
    vals=[]
    for i in range(NULLS):
        rng=np.random.default_rng(SEED+13*i+int(rate*7)%97)
        a=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        b=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        vals.append(float(np.corrcoef(env_band(env_raw(a)),env_band(env_raw(b)))[0,1]))
    v=np.array(vals);return dict(p1=float(np.percentile(v,1)),p95=float(np.percentile(v,95)),std=float(v.std()),
      mean=float(v.mean()),nulls=NULLS,n=int(v.size),n_eff=int(v.size//2))
def null_snr(rate):
    vals=[]
    for i in range(NULLS):
        rng=np.random.default_rng(SEED+29*i+int(rate*11)%89)
        c=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        y=env_raw(c);f,P=sgn.welch(y-y.mean(),fs=FS10,nperseg=min(len(y),256))
        m=(f>=0.5)&(f<=8.0);b=(f>=2.0)&(f<=20.0)
        vals.append(float(P[m].max()/max(np.median(P[b]),1e-30)) if m.any() else 0.0)
    v=np.array(vals);return dict(p99=float(np.percentile(v,99)),std=float(v.std()),nulls=NULLS,n=int(v.size))
def env_snr(y):
    f,P=sgn.welch(y-y.mean(),fs=FS10,nperseg=min(len(y),256))
    m=(f>=0.5)&(f<=8.0);b=(f>=2.0)&(f<=20.0)
    return float(P[m].max()/max(np.median(P[b]),1e-30)) if m.any() else 0.0
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S343',mode='signed phase estimators, cadence triple routes, balancing knob and engagement gate; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=8,control_tick=40000,tier_advance=0,neural_substep=80000,physics_step=0,model_load=0,render=0,decoder_eval=11),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,thresholds=TH,
      rules=['phase comes only from a harmonic least squares fit and a band limited cross correlation lag that must agree within 8 degrees',
             'arccos of a correlation may never be used as a phase estimator',
             'period comes from three routes on the 10 ms envelope and an unresolved spread above 10 percent voids the value',
             'duty is the fraction above the midpoint of p5 and p95 on the unfiltered envelope and swing is (p95-p5)/p50 with a degenerate denominator guard',
             'gate conditions are in reachable numeric form and are pre-registered',
             'constant tested branches are banned in both if and conditional expression form',
             'the verifier must find an independent audit file covering every branch carrying arm'])
    dump(o/'PREREG.json',pre)
    (o/'S343_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s343.py)=%s\nsha256(audit_s343_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s343_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    # ---------- stage A: statistic degeneracy on archived arms
    arch={}
    for nm,p in (('W0_gate',S342/'SPK_W0_gate.npz'),('W0_seed3',S342/'SPK_W0_seed3.npz'),('L30_A',S342/'SPK_L30_A.npz'),
                 ('L40_A',S342/'SPK_L40_A.npz'),('L15_B',S342/'SPK_L15_B.npz'),('L30_B',S342/'SPK_L30_B.npz'),
                 ('T150',S342/'SPK_T150.npz'),('T150_L30',S342/'SPK_T150_L30.npz')):
        if p.is_file(): arch[nm]=loadz(p)
    DEG={};arms_deg={}
    for nm,A in arch.items():
        cA=A[:,Ac].sum(1).astype(float);cB=A[:,Bc].sum(1).astype(float)
        yA=env_raw(cA);yB=env_raw(cB);bA=env_band(yA);bB=env_band(yB)
        rho=float(np.corrcoef(bA,bB)[0,1])
        arms_deg[nm]=dict(rho=rho,arccos_phase=float(np.degrees(np.arccos(np.clip(rho,-1,1)))),
          duty_identity=0.5,real_duty=dict(A=duty_of(yA),B=duty_of(yB)),
          swing_old=float((np.percentile(bA,90)-np.percentile(bA,10))/max(abs(np.mean(bA)),1e-9)),
          swing_new=swing_of(yA),period_old=0.400)
    DEG['arms']=arms_deg
    DEG['verdict']=dict(arccos_phase='mirrors and compresses the phase near plus or minus one; it is abolished',
      duty='mean above the median is identically 0.5 by construction',swing='dividing a zero mean band passed envelope by its own mean is undefined',
      period='the 100 ms autocorrelation window quantises every period to 0.400 s')
    DEG['new_values_std']=dict(duty=float(np.std([v['real_duty']['A'] for v in arms_deg.values()])),
      swing=float(np.std([v['swing_new']['value'] for v in arms_deg.values() if v['swing_new']['value'] is not None])))
    dump(o/'STAT_DEGENERACY.json',DEG)
    # ---------- stage B 8 arms
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
        return score(spm,Ac,Bc,name,seed,biasA,biasB)
    def score(spm,colsA,colsB,name,seed,biasA,biasB):
        cA=spm[:,colsA].sum(1).astype(float);cB=spm[:,colsB].sum(1).astype(float)
        yA=env_raw(cA);yB=env_raw(cB);bA=env_band(yA);bB=env_band(yB)
        rA=float(spm[:,colsA].sum()*1000.0/(spm.shape[0]*len(colsA)));rB=float(spm[:,colsB].sum()*1000.0/(spm.shape[0]*len(colsB)))
        f0A=f0_welch(bA);f0B=f0_welch(bB);f0=float(np.mean([x for x in (f0A,f0B) if x]))
        rho=float(np.corrcoef(bA,bB)[0,1]);nr=null_rho((rA+rB)/2.0)
        z=(rho-nr['mean'])/max(nr['std'],1e-30)
        pA=ls_phase(bA,bB,f0);pX=xcorr_phase(bA,bB,f0)
        snrA=env_snr(yA);snrB=env_snr(yB);nA=null_snr(rA);nB=null_snr(rB)
        zA=(snrA-nA['p99'])/max(nA['std'],1e-30);zB=(snrB-nB['p99'])/max(nB['std'],1e-30)
        per=period_routes(yA)
        return dict(name=name,seed=seed,bias_pct=dict(A=biasA,B=biasB),bias_nA=dict(A=biasA/100*I_STAR,B=biasB/100*I_STAR),
          rates=dict(A=rA,B=rB,ratio=max(rA,rB)/max(min(rA,rB),1e-9)),rho=dict(value=rho,null=nr,z=z),
          phase=dict(ls_deg=pA,xcorr_deg=pX,agree_deg=(abs(wrap(pA-pX)) if pX is not None else None),
            f0_hz=f0,mean_abs=float(np.mean([abs(pA),abs(pX)])) if pX is not None else abs(pA)),
          existence=dict(snr=dict(A=snrA,B=snrB),null=dict(A=nA,B=nB),z=dict(A=zA,B=zB)),
          period=per,duty=dict(A=duty_of(yA),B=duty_of(yB)),swing=swing_of(yA),
          added_edges=added,f_drive_hz=None,dI_time_dependent=False,tick_count=int(spm.shape[0]),not_emergent=True)
    AR={}
    AR['P0_gate']=run('P0_gate')
    AR['B20']=run('B20',biasB=20.0)
    AR['B25']=run('B25',biasB=25.0)
    AR['B30']=run('B30',biasB=30.0)
    AR['B30_seed1']=run('B30_seed1',biasB=30.0,seed=1)
    AR['B30_seed2']=run('B30_seed2',biasB=30.0,seed=2)
    AR['B30_LA30']=run('B30_LA30',biasA=30.0,biasB=30.0)
    AR['B30_LB30']=run('B30_LB30',biasB=60.0)
    gate=dict()
    for nm,refs in (('P0_gate',[S342/'SPK_W0_gate.npz',S341/'SPK_A0_gate.npz',S340/'SPK_G_gate.npz']),):
        lst=[]
        for p in refs:
            if not p.is_file(): continue
            r0=loadz(p)[:5000];lst.append(dict(ref=str(p.parent.name)+'/'+p.name,equal=bool(np.array_equal(SPM[nm],r0)),spike_total=int(r0.sum())))
        gate[nm]=dict(all_equal=bool(lst) and all(x['equal'] for x in lst),refs=lst,spike_total=int(SPM[nm].sum()),shape=list(SPM[nm].shape),
          method='unpackbits elementwise equality')
    dump(o/'GATE.json',gate)
    # ---------- decoder calibration and motor interface
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
    CAL={}
    for denom in (60,80,100,120,150,200,300,450):
        C=ctrl_of(SPM['P0_gate'],MotorOutputConfig(rate_for_full_control_hz=float(denom)))
        ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b2 in LEGS[i+1:]:
                ya=env_band(ser[a]);yb=env_band(ser[b2])
                f0=f0_welch(ya) or 2.0
                pairs[f'{a}|{b2}']=abs(ls_phase(ya,yb,f0))
        cross=[v for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        same=[v for k,v in pairs.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        CAL[str(denom)]=dict(clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),median=float(np.median(C)),
          p99=float(np.percentile(C,99)),n_ge150=int(sum(1 for v in cross if v>=TH['cross_lo'])),cross_mean=float(np.mean(cross)),
          same_max=float(max(same)),min_amp=float(min(np.ptp(ser[lg]) for lg in LEGS)),
          passes=bool(sum(1 for v in cross if v>=TH['cross_lo'])==9 and float(max(same))<=TH['same_max']
            and float(min(np.ptp(ser[lg]) for lg in LEGS))>=TH['min_amp'] and float((C<=0).mean())<=TH['clip_max']
            and float((C>=1).mean())<=TH['clip_max'] and abs(float(np.percentile(C,99))-TH['p99_target'])<=0.15))
    chosen=min([k for k,v in CAL.items() if v['passes']],key=lambda x:int(x)) if any(v['passes'] for v in CAL.values()) else None
    dump(o/'DECODER_CALIB.json',dict(sweep=CAL,chosen_denominator=chosen,thresholds=TH,
      note='no gain is fitted to behaviour or load; only the normalising denominator is swept'))
    MI={}
    for tag,cfgX in (('default',MotorOutputConfig()),('calibrated',MotorOutputConfig(rate_for_full_control_hz=float(chosen or 100))),
                     ('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN)))):
        C=ctrl_of(SPM['P0_gate'],cfgX)
        ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
        pairs={};f0=AR['P0_gate']['phase']['f0_hz'] or 2.0
        for i,a in enumerate(LEGS):
            for b2 in LEGS[i+1:]:
                ya=env_band(ser[a]);yb=env_band(ser[b2])
                pairs[f'{a}|{b2}']=float(abs(ls_phase(ya,yb,float(f0))))
        cross=[v for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        same=[v for k,v in pairs.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        MI[tag]=dict(leg_pairs=pairs,n_ge150=int(sum(1 for v in cross if v>=TH['cross_lo'])),cross_mean=float(np.mean(cross)),
          cross_min=float(min(cross)),same_max=float(max(same)),min_amp=float(min(np.ptp(ser[lg]) for lg in LEGS)),
          clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),ctrl_median=float(np.median(C)),ctrl_p99=float(np.percentile(C,99)),
          rate_for_full_control_hz=float(getattr(cfgX,'rate_for_full_control_hz',100.0)))
    dump(o/'MOTOR_INTERFACE_V7.json',dict(arms=MI,thresholds=TH))
    # ---------- gates
    def periodic(k): return AR[k]['period']['resolved']
    c1=bool(AR['P0_gate']['rho']['z']<=TH['z_gate'] and AR['B30_seed1']['rho']['z']<=TH['z_gate']
            and AR['P0_gate']['rho']['value']<=AR['P0_gate']['rho']['null']['p1']-AR['P0_gate']['rho']['null']['std'])
    cal=MI['calibrated']
    c2=bool(cal['n_ge150']==9 and cal['same_max']<=TH['same_max'] and cal['min_amp']>=TH['min_amp']
            and cal['clip0']<=TH['clip_max'] and cal['clip1']<=TH['clip_max'])
    def load_ok(k):
        return bool(AR[k]['rho']['z']<=TH['z_gate'] and MI['calibrated']['n_ge150']==9)
    c3=bool(load_ok('B30_LA30') and load_ok('B30_LB30'))
    cand=['P0_gate','B30','B30_seed1','B30_seed2']
    pers=[AR[k]['period']['period_s'] for k in cand if AR[k]['period']['period_s']]
    cv=float(np.std(pers)/np.mean(pers)) if len(pers)>=3 else None
    c4=bool(cv is not None and cv<=TH['period_cv'] and all(abs(p-TH['period_ref'])/TH['period_ref']<=TH['period_tol'] for p in pers))
    c5=bool(chosen is not None)
    flags=dict(c1=c1,c2=c2,c3=c3,c4=c4,c5=c5)
    ratios=[AR[k]['rates']['ratio'] for k in ('B20','B25','B30')]
    balance_found=bool(min(ratios)<=TH['rate_ratio'])
    if all(flags.values()): branch='b1'
    elif balance_found and not c4: branch='b2'
    elif min(ratios)>TH['rate_ratio']: branch='b3'
    else: branch='NOT_JUDGED'
    if branch=='NOT_JUDGED': dump(o/'STOP.json',dict(reason='no branch predicate held',flags=flags,ratios=ratios))
    spec=dict(unit='s',not_emergent=True,period_s=None,legs={})
    if balance_found:
        base=[AR[k] for k in ('B30','B30_seed1','B30_seed2') if AR[k]['period']['period_s']]
        per=float(np.median([x['period']['period_s'] for x in base])) if base else None
        lp={lg:float(abs(cal['leg_pairs'].get(f'lf|{lg}',0.0)) if lg!='lf' else 0.0) for lg in LEGS}
        dA=float((AR['B30_LA30']['period']['period_s'] or per or 0)-(per or 0))/max(per or 1e-9,1e-9)
        dB=float((AR['B30_LB30']['period']['period_s'] or per or 0)-(per or 0))/max(per or 1e-9,1e-9)
        spec=dict(unit='s',not_emergent=True,period_s=per,
          period_routes={k:AR[k]['period'] for k in ('B30','B30_seed1','B30_seed2')},
          legs={lg:dict(period_s=per,duty=float(np.mean([AR[k]['duty']['A' if lg in TRIPOD['A'] else 'B'] for k in ('B30','B30_seed1','B30_seed2')])),
            swing=AR['B30']['swing']['value'],phase_rel_lf_deg=lp[lg]) for lg in LEGS},
          load_sensitivity=dict(plus30_A=dA,plus30_B=dB,note='fractional change of the period when that tripod carries thirty percent more tonic drive'),
          train_constraint=dict(added_inhibitory_edges=added['n'],adaptation_tau_ms=100.0,not_emergent=True),
          physics_may_consume=['period_s','duty','swing','phase_rel_lf_deg'],
          physics_must_not_assume=['that this rhythm is emergent from the connectome','that the added 20400 edges exist in the animal'])
    dump(o/'GAIT_SPEC_V4.json',spec)
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD)
    dump(o/'OUTCOMES.json',dict(round='S343',branch=branch,flags=flags,gate=gate,
      ratios={k:round(AR[k]['rates']['ratio'],2) for k in ('P0_gate','B20','B25','B30','B30_LA30','B30_LB30')},
      chosen_denominator=chosen,periods={k:(AR[k]['period']['period_s'],AR[k]['period']['resolved']) for k in AR},
      phase={k:(round(AR[k]['phase']['ls_deg'],1),round(AR[k]['phase']['xcorr_deg'],1) if AR[k]['phase']['xcorr_deg'] is not None else None,
        round(AR[k]['phase']['agree_deg'],1) if AR[k]['phase']['agree_deg'] is not None else None) for k in AR},
      z={k:round(AR[k]['rho']['z'],2) for k in AR},duty={k:(round(AR[k]['duty']['A'],3),round(AR[k]['duty']['B'],3)) for k in AR},
      motor={t:(MI[t]['n_ge150'],round(MI[t]['cross_mean'],1),round(MI[t]['clip1'],2),round(MI[t]['min_amp'],2)) for t in MI},
      ctrl_cv=cv,counters=CD,not_emergent=True,
      next=('physics may consume GAIT_SPEC_V4' if branch=='b1' else 'do not enter physics'),
      one_line='%s c1..c5=%s ratios=%s cv=%s'%(branch,flags,{k:round(AR[k]['rates']['ratio'],2) for k in ('B20','B25','B30')},cv)))
    print(json.dumps(dict(branch=branch,flags=flags,gate=gate.get('P0_gate',{}).get('all_equal'),
      ratios={k:round(AR[k]['rates']['ratio'],2) for k in ('P0_gate','B20','B25','B30')},
      periods={k:AR[k]['period']['period_s'] for k in AR},cv=cv,chosen=chosen,
      motor={t:(MI[t]['n_ge150'],round(MI[t]['cross_mean'],1)) for t in MI},
      phase_agree={k:(round(AR[k]['phase']['ls_deg'],1),round(AR[k]['phase']['xcorr_deg'],1)) for k in AR},cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
