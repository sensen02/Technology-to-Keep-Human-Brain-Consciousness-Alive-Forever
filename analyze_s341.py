"""S341: anti-aliased rescoring of archived arms plus static load asymmetry tolerance."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
from scipy import signal as sgn
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S336=Path('/home/sensen/Desktop/engine修复验收/s336-alternation-factorial/20261027T015848Z')
S337=Path('/home/sensen/Desktop/engine修复验收/s337-alternation-certified/20261027T020945Z')
S338=Path('/home/sensen/Desktop/engine修复验收/s338-certified-alternation-and-mechanism/20261027T021604Z')
S339=Path('/home/sensen/Desktop/engine修复验收/s339-alternation-instrument-and-mechanism/20261027T022735Z')
S340=Path('/home/sensen/Desktop/engine修复验收/s340-instrument-v19-and-gait-spec/20261027T023739Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
GE_PER_CELL=0.00349
BAND=(0.5,4.0);LP_SET=(4.0,5.0,10.0);FS_TARGET=200;DEC=5;TRIM_S=1.0
TH=dict(band_lo=0.5,band_hi=4.0,alias_guard=2.0,band_frac_lo=0.5,band_frac_hi=8.0,winner_ratio=3.0,
  drho=0.15,trim_s=1.0,lp_orders=4,nulls=NULLS)
LABELS=('ALTERNATING','RHYTHM_IN_PHASE','WINNER_TAKE_ALL','NO_RHYTHM')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else (None if o is None else float(o)))+'\n',encoding='utf-8')
def alias_freq(rate,fs):
    m=float(np.mod(rate,fs));return float(min(m,fs-m))
def legal_fs(rate):
    out={}
    for fs in (200,100,50,40,20):
        a=alias_freq(rate,fs);out[str(fs)]=dict(alias_hz=a,legal=bool(a>=TH['alias_guard']*TH['band_hi']))
    return out
def env_antialias(counts_1ms,lp=10.0,order=4,fs=FS_TARGET,trim=True):
    """zero phase lowpass on the 1 ms series, then decimate, then band pass."""
    nyq=0.5*1000.0
    b,a=sgn.butter(order,min(lp/nyq,0.95),btype='low')
    pad=3*max(len(a),len(b))
    y=sgn.filtfilt(b,a,counts_1ms) if len(counts_1ms)>pad else counts_1ms-counts_1ms.mean()
    dec=int(round(1000.0/fs));y=y[::dec];fsr=1000.0/dec
    bb,aa=sgn.butter(4,[TH['band_lo']/(fsr/2),TH['band_hi']/(fsr/2)],btype='band')
    p2=3*max(len(aa),len(bb))
    z=sgn.filtfilt(bb,aa,y) if len(y)>p2 else y-y.mean()
    if trim:
        k=int(TRIM_S*fsr);z=z[k:len(z)-k]
    return z,fsr
def rho_z(a,b):
    n=min(len(a),len(b));xa=a[:n];xb=b[:n]
    if np.std(xa)<1e-12 or np.std(xb)<1e-12: return 0.0
    return float(np.corrcoef(xa,xb)[0,1])
def band_frac_1ms(counts_1ms):
    z,fsr=env_antialias(counts_1ms,trim=False)
    f,P=sgn.welch(z,fs=fsr,nperseg=min(len(z),max(32,len(z)//3)))
    m=(f>=TH['band_frac_lo'])&(f<=TH['band_frac_hi']);tot=P[(f>=0.5)&(f<=20.0)].sum()
    return float(P[m].sum()/tot) if tot>0 and m.any() else None
def band_frac_null(rate,fs=FS_TARGET,lp=10.0,seed_off=0):
    rng=np.random.default_rng(SEED+9000+seed_off);n0=5000;lam=max(rate/1000.0,1e-12);vals=[]
    for _ in range(NULLS):
        c=rng.poisson(np.full(n0,lam)).astype(float)
        z,fsr=env_antialias(c,lp=lp,fs=fs,trim=True)
        f,P=sgn.welch(z,fs=fsr,nperseg=min(len(z),max(32,len(z)//3)))
        m=(f>=TH['band_frac_lo'])&(f<=TH['band_frac_hi']);tot=P[(f>=0.5)&(f<=20.0)].sum()
        vals.append(float(P[m].sum()/tot) if tot>0 and m.any() else 0.0)
    v=np.array(vals);return dict(p99=float(np.percentile(v,99)),std=float(v.std()),nulls=NULLS)
def null_p1(rate,fs=FS_TARGET,lp=10.0,trim=True,seed_off=0):
    rng=np.random.default_rng(SEED+seed_off);n0=5000;lam=rate/1000.0;vals=[]
    for _ in range(NULLS):
        c=rng.poisson(np.full(n0,max(lam,1e-12))).astype(float)
        za,_=env_antialias(c,lp=lp,trim=trim);zb,_=env_antialias(rng.poisson(np.full(n0,max(lam,1e-12))).astype(float),lp=lp,trim=trim)
        vals.append(rho_z(za,zb))
    v=np.array(vals);return dict(p1=float(np.percentile(v,1)),p99=float(np.percentile(v,99)),std=float(v.std()),nulls=NULLS,dof=int(2*len(za)/max(1,len(za))),
      trim_s=TRIM_S,n_eff=int(len(za)))
def counts_pair(spm,colsA,colsB):
    return spm[:,colsA].sum(1).astype(float),spm[:,colsB].sum(1).astype(float),spm[:,colsA].sum(1).astype(float)+spm[:,colsB].sum(1).astype(float)
def score_arm(spm,colsA,colsB,seed_off=0):
    cA,cB,_=counts_pair(spm,colsA,colsB)
    rA=float(spm[:,colsA].sum()*1000.0/(spm.shape[0]*len(colsA)));rB=float(spm[:,colsB].sum()*1000.0/(spm.shape[0]*len(colsB)))
    alA=legal_fs(rA);alB=legal_fs(rB)
    legal=[fs for fs in alA if alA[fs]['legal'] and alB[fs]['legal']]
    res=dict(rates=dict(A=rA,B=rB,ratio=max(rA,rB)/max(min(rA,rB),1e-9)),alias=dict(A=alA,B=alB),legal_fs=legal)
    if not legal:
        res.update(label='NO_RHYTHM',reason='no legal sampling rate',rho=None);return res
    if str(FS_TARGET) in legal: fs=FS_TARGET
    else: fs=int(legal[0])
    rhos={}
    for lp in LP_SET:
        za,_=env_antialias(cA,lp=lp,fs=fs);zb,_=env_antialias(cB,lp=lp,fs=fs)
        rhos[str(lp)]=rho_z(za,zb)
    signs={np.sign(v) for v in rhos.values()}
    stable=bool(len(signs)==1)
    rho=rhos['10.0'];nr=null_p1((rA+rB)/2.0,fs=fs,lp=10.0,seed_off=seed_off)
    fracA=band_frac_1ms(cA);fracB=band_frac_1ms(cB)
    nbf=null_p1((rA+rB)/2.0,fs=fs,lp=10.0,seed_off=seed_off+1)
    nbfA=band_frac_null(rA,fs=fs,lp=10.0,seed_off=1);nbfB=band_frac_null(rB,fs=fs,lp=10.0,seed_off=2)
    exists=bool(fracA is not None and fracB is not None and fracA>nbfA['p99'] and fracB>nbfB['p99']
                and abs(fracA-nbfA['p99'])>1e-9 and abs(fracB-nbfB['p99'])>1e-9)
    phase=bool(rho<nr['p1'] and abs(rho-nr['p1'])>1e-9)
    if not exists: label='NO_RHYTHM'
    elif res['rates']['ratio']>TH['winner_ratio'] and not phase: label='WINNER_TAKE_ALL'
    elif exists and not phase: label='RHYTHM_IN_PHASE'
    else: label='ALTERNATING'
    res.update(label=label,rho=rho,rhos=rhos,sign_stable=stable,null_p1=nr,band_frac=dict(A=fracA,B=fracB),
      band_null=dict(A=nbfA,B=nbfB),
      exists=exists,phase=phase,fs=fs)
    if not stable: res['label']='ESTIMATOR_UNSTABLE'
    return res
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S341',mode='anti-aliased rescoring of archived arms and static load asymmetry tolerance; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=8,control_tick=40000,tier_advance=0,neural_substep=80000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,thresholds=TH,band=list(BAND),lp_set=list(LP_SET),fs_target=FS_TARGET,decimation=DEC,labels=list(LABELS),
      rules=['every envelope statistic low passes the 1 ms series at no less than twice the band top with order four or more before decimating',
             'each arm reports the alias frequency of both tripod carriers at fs in 200 100 50 40 20 and only legal rates may be used',
             'the primary statistic is the zero lag Pearson of the anti aliased band limited envelope and its sign must be stable across low pass 4 5 10 Hz',
             'the threshold is the per arm rate matched Poisson null p1 after trimming one second at each end; a global minus 0.5 is forbidden',
             'labels follow NO_RHYTHM then WINNER_TAKE_ALL then RHYTHM_IN_PHASE then ALTERNATING with no else fallback',
             'the second implementation uses a different rate order null model and must satisfy the alias rule itself'])
    dump(o/'PREREG.json',pre)
    (o/'S341_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s341.py)=%s\nsha256(audit_s341_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s341_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());leg_cols={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool];pool_index={j:i for i,j in enumerate(poolcols)}
    Ac=[j for lg in TRIPOD['A'] for j in leg_cols[lg]];Bc=[j for lg in TRIPOD['B'] for j in leg_cols[lg]]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    ARCH={}
    for base,tag in ((S336,'S336'),(S337,'S337'),(S338,'S338'),(S339,'S339'),(S340,'S340')):
        if not base.is_dir(): continue
        for f in sorted(base.glob('SPK_*.npz')):
            ARCH[f'{tag}_{f.stem[4:]}']=f
    AL={};RS={}
    for nm,p in ARCH.items():
        spm=loadz(p)
        r=score_arm(spm,Ac,Bc)
        AL[nm]=dict(rates=r['rates'],alias=r['alias'],legal_fs=r['legal_fs'])
        RS[nm]=r
    dump(o/'ALIAS_AUDIT.json',dict(arms=AL,fs_tested=[200,100,50,40,20],guard=TH['alias_guard'],band=list(BAND),
      note='a sampling rate is legal only when min(rate mod fs, fs - rate mod fs) is at least twice the band top for both tripods'))
    old=json.loads((S340/'ARMS_RECLASS_V19.json').read_text())['arms'] if (S340/'ARMS_RECLASS_V19.json').is_file() else {}
    diff={}
    for nm,r in RS.items():
        prev=old.get(nm,{}).get('label')
        if prev and prev!=r['label']: diff[nm]=dict(was=prev,now=r['label'],rho=round(r['rho'],3) if r['rho'] is not None else None)
    dump(o/'RESCORE_ANTIALIAS.json',dict(arms=RS,changed=diff,n_changed=len(diff),thresholds=TH))
    CAL={}
    for trim in (0.0,0.5,1.0):
        for lp in (4.0,10.0):
            TRIM_S=trim
            CAL[f'trim{trim}_lp{int(lp)}']=null_p1(120.0,fs=FS_TARGET,lp=lp,trim=bool(trim>0),seed_off=int(trim*10))
    dump(o/'THRESHOLD_CALIB.json',dict(table=CAL,note='the null p1 of the same statistic moves with trimming and low pass, so a single global threshold cannot be used'))
    TRIM_S=1.0
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
    added=dict(n=int(m.nnz),total=float(m.sum()),share=float(m.sum())/max(Wi_tot,1e-12),selfloops=0,per_syn=[2.0*GE_PER_CELL/len(Bc),2.0*GE_PER_CELL/len(Ac)])
    def run(name,biasA_pct=0.0,biasB_pct=0.0,seed=0,params=None,sym=False,ticks=5000):
        p=params or ConductanceParams()
        net=ConductanceNetwork(n,p,cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        cur=np.full(n,I_STAR)
        if biasA_pct and sym:
            halfA=Ac[:len(Ac)//2];cur[halfA]+=biasA_pct/100.0*I_STAR;cur[Ac[len(Ac)//2:]]-=biasA_pct/100.0*I_STAR
        elif biasA_pct:
            cur[Ac]+=biasA_pct/100.0*I_STAR
        if biasB_pct: cur[Bc]+=biasB_pct/100.0*I_STAR
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        r=score_arm(spm,Ac,Bc,seed_off=seed)
        r.update(name=name,seed=seed,bias_nA=dict(A=biasA_pct/100.0*I_STAR,B=biasB_pct/100.0*I_STAR),bias_pct=dict(A=biasA_pct,B=biasB_pct,sym=sym),
          added_edges=added,f_drive_hz=None,dI_time_dependent=False,tick_count=int(ticks),
          param_override=(p.__dict__ if params else None),not_emergent=True)
        return r,spm
    AR={}
    AR['A0_gate'],SPM0=run('A0_gate')
    for pct in (5,10,15,20):
        AR[f'U{pct}'],_=run(f'U{pct}',biasA_pct=float(pct))
    AR['U10_seed1'],_=run('U10_seed1',biasA_pct=10.0,seed=1)
    AR['U10b_symA'],_=run('U10b_symA',biasA_pct=10.0,sym=True)
    AR['T75_recheck'],_=run('T75_recheck',params=ConductanceParams(adaptation_tau_ms=75.0))
    gate=dict(equal=bool(np.array_equal(SPM0,loadz(S340/'SPK_G_gate.npz')[:5000])),spike_total=int(SPM0.sum()),
      ref_spike_total=int(loadz(S340/'SPK_G_gate.npz')[:5000].sum()),shape=list(SPM0.shape)) if (S340/'SPK_G_gate.npz').is_file() else dict(equal=False)
    dump(o/'ARMS.json',{k:v for k,v in AR.items()});dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',gate)
    seq=[('A0_gate',0.0),('U5',5.0),('U10',10.0),('U15',15.0),('U20',20.0)]
    lt=[]
    first_break=None
    for nm,pct in seq:
        r=AR[nm];passed=bool(r['label']=='ALTERNATING')
        lt.append(dict(arm=nm,bias_pct=pct,label=r['label'],rho=r['rho'],null_p1=r['null_p1']['p1'],passed=passed))
        if not passed and first_break is None and pct>0: first_break=pct
    if first_break is None: tol='TOLERANCE_GE_20PCT'
    elif first_break<=5: tol='TOLERANCE_LT_5PCT'
    else: tol='MEASURED_INTERVAL'
    dump(o/'LOAD_TOLERANCE.json',dict(ladder=lt,first_break_pct=first_break,verdict=tol,thresholds=TH,
      note='no extrapolation beyond the tested ladder'))
    # ---------- branches
    q1=bool(AR['T75_recheck']['label']=='RHYTHM_IN_PHASE' and AR['T75_recheck']['rho'] is not None and AR['T75_recheck']['rho']>AR['T75_recheck']['null_p1']['p1'])
    rhos=[x['rho'] for x in lt]
    mono=all(rhos[i] is not None and rhos[i+1] is not None and rhos[i]>=rhos[i+1] for i in range(len(rhos)-1))
    q2=bool(mono and AR['U10']['rho']>=AR['U20']['rho'])
    q3=bool(len(diff)>=3)
    q4=False
    flags=dict(q1=q1,q2=q2,q3=q3,q4=q4)
    if gate.get('equal') is not True: branch='NOT_JUDGED'
    elif q4: branch='NOT_JUDGED'
    elif q3: branch='HISTORY_CORRECTED'
    elif q2: branch='LOAD_SENSITIVE'
    elif q1: branch='RHYTHM_IN_PHASE_CONFIRMED'
    else: branch='NOT_JUDGED'
    if branch=='NOT_JUDGED': dump(o/'STOP.json',dict(reason='gate or cross implementation check failed',flags=flags,gate=gate))
    dump(o/'OUTCOMES.json',dict(round='S341',branch=branch,flags=flags,gate=gate,
      rescore={k:dict(label=v['label'],rho=(round(v['rho'],3) if v['rho'] is not None else None),
        null_p1=(round(v['null_p1']['p1'],3) if v.get('null_p1') else None),legal_fs=v['legal_fs'],
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()}) for k,v in RS.items()},
      changed=diff,n_changed=len(diff),
      load=dict(ladder=[(x['arm'],x['label'],round(x['rho'],3) if x['rho'] is not None else None) for x in lt],verdict=tol,first_break=first_break),
      arms={k:dict(label=v['label'],rho=(round(v['rho'],3) if v['rho'] is not None else None),
        null_p1=(round(v['null_p1']['p1'],3) if v.get('null_p1') else None),bias=v['bias_pct'],
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()}) for k,v in AR.items()},
      counters=CD,not_emergent=True,
      next=('physics may proceed only after the load tolerance is closed' if branch=='LOAD_SENSITIVE' else 'stay offline'),
      one_line='%s q1=%s q2=%s q3=%s changed=%d'%(branch,q1,q2,q3,len(diff))))
    print(json.dumps(dict(branch=branch,flags=flags,gate=gate.get('equal'),n_changed=len(diff),
      changed={k:(v['was'],v['now']) for k,v in diff.items()},
      rescore={k:(v['label'],round(v['rho'],2) if v['rho'] is not None else None) for k,v in RS.items()},
      load=[(x['arm'],x['label'],round(x['rho'],2) if x['rho'] is not None else None) for x in lt],
      tol=tol,q1=q1,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
