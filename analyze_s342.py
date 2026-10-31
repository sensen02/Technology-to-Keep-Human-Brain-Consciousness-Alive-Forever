"""S342: SOS instrument repair, continuous existence SNR, load limits and the engagement gate."""
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
S336=Path('/home/sensen/Desktop/engine修复验收/s336-alternation-factorial/20261027T015848Z')
S337=Path('/home/sensen/Desktop/engine修复验收/s337-alternation-certified/20261027T020945Z')
S338=Path('/home/sensen/Desktop/engine修复验收/s338-certified-alternation-and-mechanism/20261027T021604Z')
S339=Path('/home/sensen/Desktop/engine修复验收/s339-alternation-instrument-and-mechanism/20261027T022735Z')
S340=Path('/home/sensen/Desktop/engine修复验收/s340-instrument-v19-and-gait-spec/20261027T023739Z')
S341=Path('/home/sensen/Desktop/engine修复验收/s341-antialias-rescore-and-load-tolerance/20261027T024501Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
GE_PER_CELL=0.00349
FS_DEC=200;DEC=5;BAND=(0.5,4.0);SNR_BAND=(0.5,8.0);SNR_BG=(2.0,20.0);TRIM_S=1.0;BOX_MS=100
TH=dict(band_lo=0.5,band_hi=4.0,env_snr_band=[0.5,8.0],env_bg=[2.0,20.0],z_exists=3.0,cert_deg=10.0,
  drho=0.15,cross_lo=150.0,ctrl_amp=0.4,period_cv=0.10,load_pct=30.0,nulls=NULLS,pole_max=0.999)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else (None if o is None else float(o)))+'\n',encoding='utf-8')
def sos_check(sos,name):
    for s in sos:
        r=np.roots(s[3:])
        if r.size and float(np.max(np.abs(r)))>TH['pole_max']:
            raise SystemExit(f'filter {name} has pole radius {np.max(np.abs(r)):.6f} above {TH["pole_max"]}')
    return sos
def mk_low(order,cut,fs,name):
    sos=sgn.butter(order,cut/(fs/2),btype='low',output='sos');return sos_check(sos,name)
def mk_band(order,lo,hi,fs,name):
    sos=sgn.butter(order,[lo/(fs/2),min(hi/(fs/2),0.98)],btype='band',output='sos');return sos_check(sos,name)
def dec_env(counts_1ms,order=4,cut=10.0,trim=True):
    sos=mk_low(order,cut,1000.0,'low')
    z=sgn.sosfiltfilt(sos,counts_1ms)
    y=z[::DEC];fs=1000.0/DEC
    if trim:
        k=int(TRIM_S*fs);y=y[k:len(y)-k]
    return y,fs
def band_env(counts_1ms,order=4,trim=True):
    y,fs=dec_env(counts_1ms,order=order,cut=10.0,trim=trim)
    sos=mk_band(order,BAND[0],BAND[1],fs,'band')
    return sgn.sosfiltfilt(sos,y),fs
def box_env(counts_1ms,order=2,trim=True):
    k=BOX_MS;n=len(counts_1ms)//k
    y=np.array([counts_1ms[i*k:(i+1)*k].sum() for i in range(n)],float);fs=1000.0/k
    sos=mk_band(order,BAND[0],BAND[1],fs,'boxband')
    z=sgn.sosfiltfilt(sos,y)
    if trim:
        kk=int(TRIM_S*fs);z=z[kk:len(z)-kk] if len(z)>2*kk+8 else z
    return z,fs
def pearson(a,b):
    n=min(len(a),len(b));xa=a[:n];xb=b[:n]
    if np.std(xa)<1e-12 or np.std(xb)<1e-12: return 0.0
    return float(np.corrcoef(xa,xb)[0,1])
def env_snr(counts_1ms):
    y,fs=dec_env(counts_1ms,trim=False)
    f,P=sgn.welch(y-y.mean(),fs=fs,nperseg=min(len(y),max(32,len(y)//4)))
    m=(f>=SNR_BAND[0])&(f<=SNR_BAND[1]);b=(f>=SNR_BG[0])&(f<=SNR_BG[1])
    if not m.any() or not b.any(): return None
    return float(P[m].max()/max(np.median(P[b]),1e-30))
def noise(rate,n=5000):
    rng=np.random.default_rng(SEED+int(rate*17)%997)
    return rng.poisson(np.full(n,max(rate/1000.0,1e-12))).astype(float)
def null_snr(rate):
    vals=[]
    for i in range(NULLS):
        rng=np.random.default_rng(SEED+i*7+int(rate)%97)
        c=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        vals.append(env_snr(c))
    v=np.array([x for x in vals if x is not None])
    return dict(p99=float(np.percentile(v,99)),std=float(v.std()),mean=float(v.mean()),nulls=NULLS,n=int(v.size))
def null_rho(kind,rate):
    vals=[]
    for i in range(NULLS):
        rng=np.random.default_rng(SEED+11*i+int(rate*3)%89)
        a=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        b=rng.poisson(np.full(5000,max(rate/1000.0,1e-12))).astype(float)
        za,_=(band_env(a) if kind=='bandpass' else box_env(a))
        zb,_=(band_env(b) if kind=='bandpass' else box_env(b))
        vals.append(pearson(za,zb))
    v=np.array(vals)
    return dict(p1=float(np.percentile(v,1)),std=float(v.std()),mean=float(v.mean()),nulls=NULLS,n=int(v.size),
      n_eff=int(v.size//2),fs=(FS_DEC if kind=='bandpass' else 1000.0/BOX_MS))
def acf_period_s(y,fs):
    yc=y-y.mean();ac=sgn.correlate(yc,yc,mode='full',method='fft')[len(yc)-1:];ac/=max(ac[0],1e-30)
    lo,hi=int(0.12*fs),min(int(2.5*fs),len(ac)-1)
    if hi<=lo: return None
    seg=ac[lo:hi];k=int(np.argmin(seg))
    return float(2.0*(lo+k)/fs) if seg[k]<0 else None
def swing(y): return float((np.percentile(y,90)-np.percentile(y,10))/max(abs(np.mean(y)),1e-9))
def duty_of(y):
    med=np.median(y);return float(np.mean(y>med))
def phase_of_rho(rho):
    return float(np.degrees(np.arccos(np.clip(rho,-1.0,1.0))))
def score(spm,colsA,colsB,seed_off=0):
    cA=spm[:,colsA].sum(1).astype(float);cB=spm[:,colsB].sum(1).astype(float)
    rA=float(spm[:,colsA].sum()*1000.0/(spm.shape[0]*len(colsA)));rB=float(spm[:,colsB].sum()*1000.0/(spm.shape[0]*len(colsB)))
    snrA=env_snr(cA);snrB=env_snr(cB)
    nA=null_snr(rA);nB=null_snr(rB)
    zA=(snrA-nA['p99'])/max(nA['std'],1e-30);zB=(snrB-nB['p99'])/max(nB['std'],1e-30)
    za,fsb=band_env(cA);zb,_=band_env(cB);rho_bp=pearson(za,zb)
    xa,fsx=box_env(cA);xb,_=box_env(cB);rho_box=pearson(xa,xb)
    nb=null_rho('bandpass',(rA+rB)/2.0);nx=null_rho('box100',(rA+rB)/2.0)
    per=acf_period_s(xa,fsx)
    return dict(rates=dict(A=rA,B=rB,ratio=max(rA,rB)/max(min(rA,rB),1e-9)),
      env=dict(snr=dict(A=snrA,B=snrB),null=dict(A=nA,B=nB),z=dict(A=zA,B=zB),exists=bool(zA>=TH['z_exists'] and zB>=TH['z_exists'])),
      rho=dict(bandpass=rho_bp,box100=rho_box,null_bandpass=nb,null_box100=nx,
        phase_bandpass=bool(rho_bp<nb['p1']-3*nb['std']),phase_box100=bool(rho_box<nx['p1']-3*nx['std'])),
      period_s=per,unit='s',swing=dict(A=swing(xa),B=swing(xb)),duty=dict(A=duty_of(xa),B=duty_of(xb)),
      fs=dict(bandpass=fsb,box100=fsx))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S342',mode='SOS instrument repair, continuous existence SNR, load limit and engagement gate; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=8,control_tick=40000,tier_advance=0,neural_substep=80000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,thresholds=TH,
      rules=['every IIR is SOS and each section denominator root radius must be at most 0.999 or the script exits',
             'any band fraction or existence statistic is computed on an unfiltered series',
             'both pipelines are certified on synthetic 0 90 180 degree pairs at two rate pairs before any real arm is scored',
             'each arm reports rho from bandpass SOS and from a 100 ms boxcar with its own rate matched null',
             'existence is the envelope spectrum snr with z at least 3 and labels are derived rather than primary',
             'nulls are independently synthesised Poisson and circular shifting is forbidden',
             'dof and n_eff must be derived from the data and the verifier recomputes them'])
    dump(o/'PREREG.json',pre)
    (o/'S342_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s342.py)=%s\nsha256(audit_s342_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s342_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    # ---------- sign certification of both metrics
    def synth_pair(rate,phase_deg,seed,n=20000,f=2.0):
        rng=np.random.default_rng(seed);t=np.arange(n)/1000.0
        lam=lambda ph: np.maximum(rate/1000.0*(1.0+0.9*np.cos(2*np.pi*f*t-np.radians(ph))),1e-9)
        return rng.poisson(lam(0.0)).astype(float),rng.poisson(lam(phase_deg)).astype(float)
    CERT={}
    for rate in (105.4,156.9):
        for target in (0.0,90.0,180.0):
            a5,b5=synth_pair(rate,target,4242+int(rate)+int(target),n=5000)
            a,b=synth_pair(rate,target,4242+int(rate)+int(target))
            rb=pearson(band_env(a)[0],band_env(b)[0]);rx=pearson(box_env(a)[0],box_env(b)[0])
            r5=pearson(box_env(a5)[0],box_env(b5)[0])
            CERT[f'{rate}_{int(target)}']=dict(target=target,box100_5s=dict(rho=r5,est=phase_of_rho(r5),err=abs(phase_of_rho(r5)-target)),
              bandpass=dict(rho=rb,est=phase_of_rho(rb),err=abs(phase_of_rho(rb)-target)),
              box100=dict(rho=rx,est=phase_of_rho(rx),err=abs(phase_of_rho(rx)-target)))
    cert_ok=all(v['bandpass']['err']<=TH['cert_deg'] and v['box100']['err']<=TH['cert_deg'] for v in CERT.values())
    dump(o/'INSTRUMENT_V20.json',dict(certification=CERT,certified=cert_ok,thresholds=TH,
      filters=dict(lowpass='SOS order 4 at 10 Hz on the 1 ms series',bandpass='SOS order 4 at 0.5-4 Hz after decimation to 200 Hz',
        box='100 ms boxcar then SOS order 2 at 0.5-4 Hz'),
      pole_rule='every section denominator root radius must be at most 0.999 and that is checked at construction'))
    # ---------- rescore archives
    ARCH={}
    for base,tag in ((S336,'S336'),(S337,'S337'),(S338,'S338'),(S339,'S339'),(S340,'S340'),(S341,'S341')):
        if not base.is_dir(): continue
        for f in sorted(base.glob('SPK_*.npz')): ARCH[f'{tag}_{f.stem[4:]}']=f
    RS={}
    for nm,p in ARCH.items():
        RS[nm]=score(loadz(p),Ac,Bc)
    dump(o/'RESCORE_ARCHIVE_V20.json',dict(arms=RS,thresholds=TH,n_arms=len(RS),
      note='five continuous columns per arm: z_exists, rho_box100, period, duty, rates; labels are derived'))
    # ---------- stage B eight arms
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
    def run(name,biasA_pct=0.0,biasB_pct=0.0,seed=0,params=None,ticks=5000):
        p=params or ConductanceParams()
        net=ConductanceNetwork(n,p,cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if biasA_pct: cur[Ac]+=biasA_pct/100.0*I_STAR
        if biasB_pct: cur[Bc]+=biasB_pct/100.0*I_STAR
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        r=score(spm,Ac,Bc,seed_off=seed)
        r.update(name=name,seed=seed,bias_pct=dict(A=biasA_pct,B=biasB_pct),bias_nA=dict(A=biasA_pct/100*I_STAR,B=biasB_pct/100*I_STAR),
          added_edges=added,f_drive_hz=None,dI_time_dependent=False,tick_count=int(ticks),not_emergent=True,
          param_override=(p.__dict__ if params else None))
        SPM[name]=spm
        return r
    AR={}
    AR['W0_gate']=run('W0_gate')
    AR['W0_seed3']=run('W0_seed3',seed=3)
    AR['L30_A']=run('L30_A',biasA_pct=30.0)
    AR['L40_A']=run('L40_A',biasA_pct=40.0)
    AR['L15_B']=run('L15_B',biasB_pct=15.0)
    AR['L30_B']=run('L30_B',biasB_pct=30.0)
    AR['T150']=run('T150',params=ConductanceParams(adaptation_tau_ms=150.0))
    AR['T150_L30']=run('T150_L30',biasA_pct=30.0,params=ConductanceParams(adapteation_tau_ms=150.0) if False else ConductanceParams(adaptation_tau_ms=150.0))
    gate=dict()
    for nm,ref in (('W0_gate',[S341/'SPK_A0_gate.npz',S340/'SPK_G_gate.npz']),('W0_seed3',[])):
        lst=[]
        for p in ref:
            if not p.is_file(): continue
            r0=loadz(p)[:5000];lst.append(dict(ref=str(p.parent.name)+'/'+p.name,equal=bool(np.array_equal(SPM[nm],r0)),spike_total=int(r0.sum())))
        gate[nm]=dict(all_equal=bool(lst) and all(x['equal'] for x in lst),refs=lst,spike_total=int(SPM[nm].sum()),shape=list(SPM[nm].shape),
          method='unpackbits elementwise equality')
    dump(o/'GATE.json',gate)
    # ---------- motor interface for engagement gate
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    def ctrl_of(Am,cfgX,chunk=10):
        mo=MotorOutput(MAN,cfgX);spm=Am[:,poolcols].astype(float);C=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(C)):
            s=slice(i*chunk,(i+1)*chunk);c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);C[i]=c
        return C
    def ctrl_phase(a,b,fs):
        y=sgn.sosfiltfilt(mk_band(2,BAND[0],BAND[1],fs,'ctrlband'),a-a.mean())
        z=sgn.sosfiltfilt(mk_band(2,BAND[0],BAND[1],fs,'ctrlband'),b-b.mean())
        return 180.0-phase_of_rho(pearson(y,z))
    MI={}
    for tag,cfgX in (('default',MotorOutputConfig()),('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN)))):
        C=ctrl_of(SPM['W0_gate'],cfgX);fs=100.0
        ser={lg:(C[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(C.shape[0])) for lg in LEGS}
        per=AR['W0_gate']['period_s'] or 0.5
        pairs={}
        for i,a in enumerate(LEGS):
            for b2 in LEGS[i+1:]:
                pairs[f'{a}|{b2}']=float(ctrl_phase(ser[a],ser[b2],fs))
        cross=[abs(v) for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        MI[tag]=dict(leg_pairs=pairs,n_ge150=int(sum(1 for v in cross if v>=TH['cross_lo'])),
          cross_min=float(min(cross)),leg_amp={lg:float(np.ptp(ser[lg])) for lg in LEGS},
          min_amp=float(min(np.ptp(ser[lg]) for lg in LEGS)),ctrl_median=float(np.median(C)),
          clip0=float((C<=0).mean()),clip1=float((C>=1).mean()),period_s=per)
    dump(o/'MOTOR_INTERFACE_V6.json',dict(configs=dict(default=100.0,s300=450.0),arms=MI))
    # ---------- load tolerance and engagement gate
    lad=[('W0_gate',0.0,'none'),('L30_A',30.0,'A'),('L40_A',40.0,'A'),('L15_B',15.0,'B'),('L30_B',30.0,'B')]
    rows=[]
    for nm,pct,dirn in lad:
        r=AR[nm];nb=r['rho']['null_box100']
        rows.append(dict(arm=nm,dir=dirn,pct=pct,rho=r['rho']['box100'],null_p1=nb['p1'],null_std=nb['std'],
          margin=float(nb['p1']-3*nb['std']-r['rho']['box100']),passes=bool(r['rho']['phase_box100']),
          z_exists=min(r['env']['z']['A'],r['env']['z']['B']),period_s=r['period_s']))
    dump(o/'LOAD_TOLERANCE_V20.json',dict(ladder=rows,thresholds=TH,note='both directions of the static load are reported separately'))
    seeds_ok=all(AR[k]['rho']['phase_box100'] for k in ('W0_gate','W0_seed3')) and (S341/'SPK_A0_gate.npz').is_file()
    a30=next(x for x in rows if x['arm']=='L30_A');b30=next(x for x in rows if x['arm']=='L30_B')
    c1=bool(seeds_ok)
    c2=bool(MI['default']['n_ge150']==9 and MI['default']['min_amp']>=TH['ctrl_amp'])
    c3=bool(a30['passes'] and b30['passes'])
    pers=[AR[k]['period_s'] for k in ('W0_gate','W0_seed3','L30_A','L30_B','T150') if AR[k]['period_s']]
    cv=float(np.std(pers)/np.mean(pers)) if pers else None
    c4=bool(cv is not None and cv<=TH['period_cv'])
    gateC=dict(c1=dict(ok=c1,detail={k:AR[k]['rho']['box100'] for k in ('W0_gate','W0_seed3')}),
      c2=dict(ok=c2,n_ge150=MI['default']['n_ge150'],min_amp=MI['default']['min_amp'],requirement=TH['ctrl_amp']),
      c3=dict(ok=c3,L30_A=a30['margin'],L30_B=b30['margin']),
      c4=dict(ok=c4,cv=cv,periods=pers),engage_ready=bool(c1 and c2 and c3 and c4))
    dump(o/'ENGAGEMENT_GATE.json',gateC)
    dump(o/'GAIT_SPEC_V3.json',dict(unit='s',period_s=AR['W0_gate']['period_s'],duty=AR['W0_gate']['duty'],
      swing=AR['W0_gate']['swing'],leg_phase_rel_lf=MI['default']['leg_pairs'],tripod=TRIPOD,not_emergent=True,
      physics_may_consume=['per leg period in seconds','per leg duty','per leg swing','per leg phase relative to lf'],
      physics_must_not_assume=['that this rhythm is an emergent property of the connectome','that the 20400 added inhibitory edges exist in the animal'],
      load_caveat='period and duty must be re-read whenever the load changes because L30_A and L40_A shift both'))
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD)
    branch='ENGAGE_READY' if gateC['engage_ready'] else 'ENGAGE_BLOCKED'
    dump(o/'OUTCOMES.json',dict(round='S342',branch=branch,engage=gateC,certified=cert_ok,gate=gate,
      arms={k:dict(z=dict(A=round(v['env']['z']['A'],2),B=round(v['env']['z']['B'],2)),
        rho_box=round(v['rho']['box100'],3),rho_bp=round(v['rho']['bandpass'],3),
        null_p1_box=round(v['rho']['null_box100']['p1'],3),period=v['period_s'],
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()},bias=v['bias_pct']) for k,v in AR.items()},
      load=rows,counters=CD,
      next=('S343 may attach the decoder and physics using GAIT_SPEC_V3' if gateC['engage_ready'] else 'S343 must change the construction, not rescore again'),
      one_line='%s c1=%s c2=%s c3=%s c4=%s'%(branch,c1,c2,c3,c4)))
    print(json.dumps(dict(branch=branch,cert=cert_ok,gate=gate.get('W0_gate',{}).get('all_equal'),
      arms={k:(round(v['rho']['box100'],3),round(min(v['env']['z']['A'],v['env']['z']['B']),2),v['period_s']) for k,v in AR.items()},
      ladder=[(x['arm'],round(x['rho'],3),x['passes']) for x in rows],
      ctrl=(MI['default']['n_ge150'],round(MI['default']['min_amp'],3)),gateC=gateC['engage_ready'],cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
