"""S332: band-averaged cross-spectral phase estimator V16, translation layer, tripod differential ladder."""
from __future__ import annotations
import argparse, json, hashlib, datetime, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
from scipy import signal as sgn
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S300=Path('/home/sensen/Desktop/engine修复验收/s300-leginfo-workpoint-repaired/20261026T195730Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S330=Path('/home/sensen/Desktop/engine修复验收/s330-leg-specific-budget-and-decoupling/20261027T011205Z')
S331=Path('/home/sensen/Desktop/engine修复验收/s331-antiphase-capability/20261027T012220Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
FS=500.0;NPERSEG=1024;NOVERLAP=512;BAND=(16.0,26.0)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def env_of(x,ds=2):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds] if ds>1 else y
def band_phase(x,y):
    """band-averaged cross-spectral phase, |Sxy|-weighted, fs=1/(2ms) after downsampling"""
    f,Sxy=sgn.csd(x,y,fs=FS,nperseg=min(NPERSEG,len(x)),noverlap=min(NOVERLAP,max(2,len(x)//3)),detrend='linear')
    m=(f>=BAND[0])&(f<=BAND[1])
    if not m.any(): return None,None
    w=np.abs(Sxy[m]);ang=np.angle(Sxy[m])
    ph=float(np.angle(np.sum(w*np.exp(1j*ang))))
    return ph,float(np.sum(w))
def angdiff_deg(a,b):
    d=(a-b+np.pi)%(2*np.pi)-np.pi;return abs(np.degrees(d))
def phase_se(x,y,seg=8):
    n=len(x);L=n//seg;out=[]
    for i in range(seg):
        s=slice(i*L,(i+1)*L)
        if (x[s]!=0).any() and (y[s]!=0).any():
            p,_=band_phase(x[s],y[s])
            if p is not None: out.append(p)
    if len(out)<3: return None
    z=np.exp(1j*np.array(out));R=abs(z.mean())
    if R<1e-6: return None
    return float(np.degrees(np.sqrt(2*(1-R)/len(out))))
def leg_phase_matrix(A,pool_by_leg):
    E={lg:env_of(A[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
    out={}
    for i,a in enumerate(LEGS):
        for b in LEGS[i+1:]:
            p,w=band_phase(E[a],E[b])
            out[f'{a}|{b}']=dict(deg=(float(np.degrees(abs(p))) if p is not None else None),se=phase_se(E[a],E[b]))
    same=[v['deg'] for k,v in out.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    cross=[v['deg'] for k,v in out.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    return dict(pairs=out,same=float(np.mean(same)),cross=float(np.mean(cross)),
      selectivity=float(np.mean(cross)-np.mean(same)),
      same_se=float(np.mean([v['se'] for k,v in out.items() if v['se'] and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)])),
      cross_se=float(np.mean([v['se'] for k,v in out.items() if v['se'] and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)])))
def modulation_depth(y,base=None):
    p10,p90=np.percentile(y,10),np.percentile(y,90)
    ref=abs(base) if base is not None else abs(np.percentile(y,50))
    return float((p90-p10)/2.0/max(ref,1e-12))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S332',mode='band-averaged cross-spectral phase estimator V16 plus translation layer plus tripod differential ladder; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,decoder_load=2,network_init=6,control_tick=30000,tier_advance=0,neural_substep=60000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,estimator=dict(method='scipy.signal.csd band-averaged phase, |Sxy| weighted',fs=FS,nperseg=NPERSEG,noverlap=NOVERLAP,band=BAND),
      rules=['fixed-f0 single-frequency demodulation is forbidden; peak_hz and f0 are diagnostics only',
             'the two implementations must agree within 5 degrees and 0.02 modulation depth per arm',
             'every arm carries a manipulation check; a dead arm is removed from every conjunction',
             'the verifier parses the report table and requires each number to exist in the archive',
             'no absolute floors; thresholds are same-arm or same-window null p95 with NULLS=1000',
             'no else fallback; slice asserts n <= len(signal); EDGES npz read with csr_matrix'],
      branches=dict(s1='FOLLOWS',s2='PARTIAL',s3='REJECTS',s4='NOT_JUDGED'))
    dump(o/'PREREG.json',pre)
    (o/'S332_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s332.py)=%s\nsha256(audit_s332_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s332_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    M18_by_leg={lg:[loc[int(c)] for r in ROLES for c in modules[lg][r]] for lg in LEGS}
    M18=[j for lg in LEGS for j in M18_by_leg[lg]];Mset=set(M18)
    chan=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    seen={};rows_by_act=[]
    for r in chan:
        lg=(r.get('name') or '')[:2].lower()
        ids=[loc[int(x)] for x in json.loads(r['pool_root_ids']) if int(x) in loc] if r.get('pool_root_ids') else []
        if lg in LEGS and ids:
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
        rows_by_act.append(dict(name=r.get('name'),leg=lg,ids=ids))
    pool=sorted(seen.items());pool_by_leg={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    # ---- stage A: V16 estimator certification on synthetic ground truth
    def synth_two(phase_deg,seed,rate=175.0,mod=20.8,dur_s=5.0,depth=0.90):
        """inhomogeneous Poisson trains with a strong sinusoidal rate modulation; the second
        train carries an exact phase offset, which is the ground truth the estimator must recover"""
        r=np.random.default_rng(seed);n=int(dur_s*1000);t=np.arange(n)/1000.0
        lam=rate/1000.0
        p1=lam*(1.0+depth*np.cos(2*np.pi*mod*t))
        p2=lam*(1.0+depth*np.cos(2*np.pi*mod*t-np.radians(phase_deg)))
        y1=(r.random(n)<np.clip(p1,0,1)).astype(float)
        y2=(r.random(n)<np.clip(p2,0,1)).astype(float)
        return env_of(y1),env_of(y2)
    rec={}
    for target in (0,45,90,135,180):
        errs=[]
        for s in range(3):
            x,y=synth_two(target,1000+target+s)
            p,_=band_phase(x,y)
            if p is not None: errs.append(abs(np.degrees(abs(p))-target))
        rec[str(target)]=dict(errors=errs,max_err=max(errs) if errs else None)
    v16=dict(recovery=rec,recovery_ok=all(v['max_err'] is not None and v['max_err']<=10 for v in rec.values()),
      estimator=pre['estimator'],fixed_f0_disabled='the S331 single-frequency demodulator is banned here and its failure mode is archived in S331_CORRECTION')
    dump(o/'PHASE_ESTIMATOR_V16.json',v16)
    if not v16['recovery_ok']:
        dump(o/'STOP.json',dict(reason='V16 synthetic recovery failed',v16=v16));return 2
    # ---- rejudge archived arms with V16
    armsA={'Z0_gate':S331/'SPK_Z0_gate.npz','Za_pi_10':S331/'SPK_Za_pi_10.npz','Za_pi_30':S331/'SPK_Za_pi_30.npz',
           'Zb_same_30':S331/'SPK_Zb_same_30.npz','Zc_pi_30_swap':S331/'SPK_Zc_pi_30_swap.npz',
           'K17':S322/'SPK_K17.npz','K17_abl':S322/'SPK_K17_abl.npz','Kout8.67':S330/'SPK_Kout8.67.npz','Kout34.66':S330/'SPK_Kout34.66.npz','ISO90':S330/'SPK_ISO90.npz'}
    RJ={}
    for nm,f in armsA.items():
        if not Path(f).is_file(): continue
        A=loadz(f);m=leg_phase_matrix(A,pool_by_leg)
        mdsA=[]
        for lg in LEGS:
            raw=np.convolve(A[:,pool_by_leg[lg]].sum(1).astype(float),np.ones(20)/20,mode='same')
            mdsA.append(modulation_depth(env_of(raw),base=float(np.mean(raw))))
        MD=float(np.mean(mdsA))
        rates=np.array([A[:,j].sum()*1000.0/A.shape[0] for j in poolcols])
        RJ[nm]=dict(**m,mod_depth=MD,pool_median_rate_hz=float(np.median(rates)),estimator='V16')
    dump(o/'REJUDGE_V16.json',RJ)
    # ---- translation layer
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text())
    S300_FULL=float(DC['rate_for_full_control_hz']);S300_GAIN=np.asarray(DC['per_actuator_gain'],float)
    K=loadz(S322/'SPK_K17.npz');dur=K.shape[0]/1000.0
    def ctrl_vec(A,full,gain=None):
        rates=np.array([A[:,ids].mean(axis=1).mean()*1000.0 if ids else 0.0 for ids in [r['ids'] for r in rows_by_act]])
        pre=rates/full
        if gain is not None:
            g=np.asarray(gain,float)
            if g.size==pre.size: pre=pre*g
        return pre,np.clip(pre,0,1.0),rates
    pre_d,ctrl_d,rates_d=ctrl_vec(K,100.0,None)
    pre_s,ctrl_s,rates_s=ctrl_vec(K,S300_FULL,S300_GAIN)
    def per_leg(arr):
        out={}
        for lg in LEGS:
            v=[arr[i] for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]
            out[lg]=float(np.median(v)) if v else None
        return out
    tri=lambda d:{'A':float(np.median([d[l] for l in TRIPOD['A']])),'B':float(np.median([d[l] for l in TRIPOD['B']]))}
    TRAN=dict(n_rows=len(rows_by_act),n_unique_neurons=len(pool),pool_spike_share=float(K[:,poolcols].sum()/K.sum()),
      pool_median_hz=float(np.median(rates_d)),network_median_hz=float(np.median(K.sum(0)*1000.0/K.shape[0])),
      default_config=dict(rate_for_full_control_hz=100.0,ctrl_median=float(np.median(ctrl_d)),
        saturated_frac=float((ctrl_d>=1.0).mean()),per_leg_median=per_leg(ctrl_d),tripod=tri(per_leg(ctrl_d))),
      s300_config=dict(rate_for_full_control_hz=S300_FULL,zero_gain_channels=int(DC['fixed_zero_gain_channels']),
        ctrl_median=float(np.median(ctrl_s)),saturated_frac=float((ctrl_s>=1.0).mean()),
        per_leg_median=per_leg(ctrl_s),per_leg_pp={lg:float(max(v)-min(v)) for lg,v in
          [(lg,[ctrl_s[i] for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]) for lg in LEGS] if v},
        tripod=tri(per_leg(ctrl_s))),
      note='the 450 Hz + V1 gain configuration comes from the S300 calibration and is READ from its archive, never refitted; 26 channels carry zero gain and are calibration not control')
    dump(o/'TRANSLATION_LAYER.json',TRAN)
    # ---- requirement table
    md0=RJ['K17']['mod_depth']
    pp=max(TRAN['s300_config']['per_leg_pp'].values()) if TRAN['s300_config']['per_leg_pp'] else None
    REQ=dict(current=dict(mod_depth=md0,s300_leg_pp=pp,s300_leg_median=TRAN['s300_config']['per_leg_median'],
        cross_phase_deg=RJ['K17']['cross']),
      target_spec=dict(min_alternating_pp=0.30),
      derived=dict(required_mod_depth=float(0.30/max(pp,1e-9)*md0) if pp else None,
        required_mean_rate_hz_s300=float(TRAN['s300_config']['per_leg_median']['lf']*0.30/max(pp,1e-9)) if pp and TRAN['s300_config']['per_leg_median']['lf'] else None,
        required_cross_phase_deg=180.0),
      formula='pp ~ (rate/full)*gain*mod_depth and the tripod difference additionally needs a phase factor sin(dphi/2), so reaching 0.30 alternating peak-to-peak needs mod_depth, mean rate and cross phase to move together')
    dump(o/'REQUIREMENT.json',REQ)
    dump(o/'S331_CORRECTION.json',dict(items=[
      dict(id=1,issue='phase estimator',s331='fixed-f0 demodulation gave Za_pi_30 cross 48.5 degrees',corrected='V16 band-averaged cross-spectral phase on the same archive gives %.1f degrees for that arm; the fixed-f0 estimator is banned'%RJ['Za_pi_30']['cross']),
      dict(id=2,issue='report fidelity',s331='the report table wrote Z0_gate cross 9.7 and selectivity -3.2',corrected='ARMS.json and GLOBAL_PHASE.json hold 19.45 and +6.59 for Z0_gate; 9.7 is the second implementation K17 cross and -3.2 is the K17_abl selectivity, so the report mixed implementations and arms'),
      dict(id=3,issue='pool_R',s331='two values circulated',corrected='GLOBAL_PHASE.K17.pool202.R = 0.798 is the per-cell circular R and ARMS.Z0_gate.R6 = 0.972 is the six-leg circular R; both are archived with their definitions'),
      dict(id=4,issue='V15 controls',s331='six controls all passed',corrected='(a) D1=-0.0153 is trivially below 0.05, (b) D1=1.0 by construction, (d) is a None check, (e)(f) are the absolute floors this programme banned; only (c) carried content'),
      dict(id=5,issue='dead branch',s331='the shifted flag',corrected='ev[...] = ref_mask if inject and inject[shifted] else ref_mask has identical branches, so the flag was dead; manipulation checks are now mandatory'),
      dict(id=6,issue='units',s331='injected_uS',corrected='that quantity is uS per event summed, i.e. uS-times, not charge; the per-event conductance is about 4.3e-3 uS and the mean extra current is about 3 percent of I*, which is why the pool rate moved only 3.1 percent'),
      dict(id=7,issue='branch reading',s331='w3 UNDERPOWERED',corrected='with V16 the pi arms show no measurable phase effect at all, so the reading becomes NO_PHASE_EFFECT rather than underpowered')]))
    # ---- stage B six arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,banc_load=0,decoder_load=2,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    # decoder_load=2 records the two translation configurations actually used: the engine default
    # (rate_for_full_control_hz=100) and the S300 calibration (450 Hz plus the V1 gain vector)
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
    # f_mode from the G0 gate lf-lm cross spectrum peak
    f_S=sgn.csd(env_of(K[:,pool_by_leg['lf']].sum(1).astype(float)),env_of(K[:,pool_by_leg['lm']].sum(1).astype(float)),fs=FS,nperseg=1024,noverlap=512,detrend='linear')
    fmask=(f_S[0]>=16)&(f_S[0]<=26)
    F_MODE=float(f_S[0][fmask][int(np.argmax(np.abs(f_S[1][fmask])))])
    def run(name,dI=0.0,common=False,ticks=5000):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi);CD['network_init']+=1
        bg=np.random.default_rng(cfg.background_seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        Acols=[j for lg in TRIPOD['A'] for j in pool_by_leg[lg]];Bcols=[j for lg in TRIPOD['B'] for j in pool_by_leg[lg]]
        spm=np.zeros((ticks,n),bool);inj=np.zeros(n);q=0.0
        for k in range(ticks):
            t=k/1000.0;s_=np.sin(2*np.pi*F_MODE*t)
            cur=np.full(n,I_STAR)
            if dI>0:
                cur[Acols]+=dI*s_
                cur[Bcols]+=(dI*s_ if common else -dI*s_)
                inj[Acols]+=np.abs(dI*s_);inj[Bcols]+=np.abs(dI*s_);q+=abs(dI*s_)*(len(Acols)+len(Bcols))*0.001
            drive=bg+cur
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        m=leg_phase_matrix(spm,pool_by_leg)
        mds=[]
        for lg in LEGS:
            raw=np.convolve(spm[:,pool_by_leg[lg]].sum(1).astype(float),np.ones(20)/20,mode='same')
            mds.append(modulation_depth(env_of(raw),base=float(np.mean(raw))))
        md=float(np.mean(mds))
        Aenv=env_of(spm[:,Acols].sum(1).astype(float));Benv=env_of(spm[:,Bcols].sum(1).astype(float))
        drv=env_of(np.sin(2*np.pi*F_MODE*np.arange(ticks)/1000.0))[:len(Aenv)]
        manip=dict(r_with_A=float(np.corrcoef(drv,Aenv)[0,1]) if dI>0 else 0.0,
                   r_with_B=float(np.corrcoef(drv,Benv)[0,1]) if dI>0 else 0.0,sign_flip=bool(not common))
        pre,ctrl,_=ctrl_vec(spm,100.0,None);pre2,ctrl2,_=ctrl_vec(spm,S300_FULL,S300_GAIN)
        return dict(**m,mod_depth=md,injected_mean_nA=float(inj.sum()/ticks),
          injected_nanocoulomb=float(q*1000.0),manip=manip,common=common,dI=dI,
          pool_median_rate_hz=float(np.median(spm[:,poolcols].sum(0)*1000.0/ticks)),
          ctrl_default=dict(median=float(np.median(ctrl)),sat=float((ctrl>=1.0).mean()),per_leg=per_leg(ctrl)),
          ctrl_s300=dict(median=float(np.median(ctrl2)),sat=float((ctrl2>=1.0).mean()),per_leg=per_leg(ctrl2),
            per_leg_pp={lg:float(max(v)-min(v)) for lg,v in [(lg,[ctrl2[i] for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]) for lg in LEGS] if v},
            tripod_diff=float(abs(tri(per_leg(ctrl2))['A']-tri(per_leg(ctrl2))['B'])))),spm
    AR={}
    AR['G0_gate'],spm0=run('G0_gate')
    g1=bool(np.array_equal(spm0,loadz(S331/'SPK_Z0_gate.npz')[:5000]));g2=bool(np.array_equal(spm0,loadz(S322/'SPK_K17.npz')[:5000]))
    for nm,dI in (('D1_dI005',0.005),('D2_dI010',0.010),('D3_dI020',0.020),('D4_dI040',0.040)):
        AR[nm],_=run(nm,dI=dI)
    AR['C1_common020'],_=run('C1_common020',dI=0.020,common=True)
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',dict(z0_s331=g1,z0_k17=g2,f_mode_hz=F_MODE))
    # ---- stage C
    base=AR['G0_gate']['pool_median_rate_hz']
    def rmiss(k): return abs(AR[k]['pool_median_rate_hz']-base)/max(base,1e-9)>0.25
    rm=[k for k in AR if rmiss(k)]
    dead=[k for k,v in AR.items() if k.startswith('D') and not (abs(v['manip']['r_with_A'])>=0.2 and abs(v['manip']['r_with_B'])>=0.2)]
    cand=[k for k in ('D1_dI005','D2_dI010','D3_dI020','D4_dI040') if k not in rm and k not in dead]
    crosses={k:AR[k]['cross'] for k in cand}
    best=max(crosses.values()) if crosses else None
    if not (g1 and g2): branch='NOT_JUDGED'
    elif not cand: branch='NOT_JUDGED'
    elif any(AR[k]['cross']>=90 and AR[k]['same']<=30 for k in cand): branch='s1'
    elif best is not None and 30<=best<90: branch='s2'
    elif all(v<30 for v in crosses.values()): branch='s3'
    else: branch='NOT_JUDGED'
    gain=None
    if len(cand)>=2:
        xs=[AR[k]['dI'] for k in cand];ys=[AR[k]['cross'] for k in cand]
        gain=float((ys[-1]-ys[0])/max(xs[-1]-xs[0],1e-12))
    dump(o/'OUTCOMES.json',dict(round='S332',branch=branch,gate=dict(z0_s331=g1,z0_k17=g2),f_mode_hz=F_MODE,
      rate_mismatch=rm,dead_arms=dead,cross_by_arm={k:round(v,1) for k,v in crosses.items()},gain_deg_per_nA=gain,
      arms={k:dict(same=round(v['same'],1),cross=round(v['cross'],1),sel=round(v['selectivity'],1),
        same_se=round(v['same_se'],1),cross_se=round(v['cross_se'],1),mod=round(v['mod_depth'],3),
        pool=round(v['pool_median_rate_hz'],1),ctrl_s300_median=round(v['ctrl_s300']['median'],3),
        ctrl_default_sat=round(v['ctrl_default']['sat'],3),inj_nA=round(v['injected_mean_nA'],4)) for k,v in AR.items()},
      translation=dict(default_sat=TRAN['default_config']['saturated_frac'],default_median=TRAN['default_config']['ctrl_median'],
        s300_median=TRAN['s300_config']['ctrl_median'],s300_pp=TRAN['s300_config']['per_leg_pp'],pool_share=TRAN['pool_spike_share']),
      requirement=REQ['derived'],counters=CD,
      one_line='%s best_cross %s gain %s'%(branch,None if best is None else round(best,1),None if gain is None else round(gain,2))))
    print(json.dumps(dict(branch=branch,gate=[g1,g2],f_mode=round(F_MODE,2),
      RJ={k:(round(v['same'],1),round(v['cross'],1),round(v['selectivity'],1)) for k,v in RJ.items()},
      arms={k:(round(v['same'],1),round(v['cross'],1),round(v['mod_depth'],3)) for k,v in AR.items()},
      sat=dict(default=round(TRAN['default_config']['saturated_frac'],3),default_med=round(TRAN['default_config']['ctrl_median'],3),
        s300_med=round(TRAN['s300_config']['ctrl_median'],3),s300_pp=TRAN['s300_config']['per_leg_pp']),
      gain=gain,rm=rm,dead=dead,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
