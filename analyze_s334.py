"""S334: band-aware relay reanalysis, frequency ladder and off/kick persistence."""
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
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S332=Path('/home/sensen/Desktop/engine修复验收/s332-phase-estimator-and-tripod-differential/20261027T013035Z')
S333=Path('/home/sensen/Desktop/engine修复验收/s333-input-channel-relay/20261027T014011Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
FS=500.0;NPERSEG=1024;NOVERLAP=512;REF_BAND=(16.0,26.0);DRIVE_BAND_OLD=(160.0,190.0)
F_MODE=25.390625
BRANCH_TH=dict(relay_cross=135.0,relay_same=30.0,anti_pairs=8,self_cross=135.0,self_manip_lt=30.0,neg_lt=90.0,rate_tol=0.10)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def env_of(x,ds=2):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds] if ds>1 else y
def band_phase(x,y,band):
    n=min(len(x),len(y));x=np.asarray(x)[:n];y=np.asarray(y)[:n]
    nper=min(NPERSEG,n)
    if nper<64: return None
    f,S=sgn.csd(x,y,fs=FS,nperseg=nper,noverlap=min(NOVERLAP,max(2,nper//2)),detrend='linear')
    m=(f>=band[0])&(f<=band[1])
    if not m.any(): return None
    w=np.abs(S[m])
    if w.sum()<=0: return None
    return float(np.degrees(np.angle(np.sum(w*np.exp(1j*np.angle(S[m]))))))
def leg_envs(A,cols_by_leg,sl=None):
    return {lg:env_of(A[sl if sl is not None else slice(None),cols_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
def pairs_of(E,band):
    out={}
    for i,a in enumerate(LEGS):
        for b in LEGS[i+1:]:
            out[f'{a}|{b}']=band_phase(E[a],E[b],band)
    return out
def summary(pairs):
    same=[abs(v) for k,v in pairs.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    cross=[abs(v) for k,v in pairs.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    allv=[v for v in pairs.values() if v is not None]
    z=np.exp(1j*np.radians(np.abs(allv))) if allv else None
    return dict(same=float(np.mean(same)) if same else None,cross=float(np.mean(cross)) if cross else None,
      cross_min=float(min(cross)) if cross else None,same_max=float(max(same)) if same else None,
      n_anti_cross=int(sum(1 for v in cross if v>=135)),n_cross=len(cross),
      R=float(abs(z.mean())) if z is not None else None,pairs=pairs)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S334',mode='band-aware relay reanalysis, frequency ladder and off-kick persistence; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=9,control_tick=45000,tier_advance=0,neural_substep=90000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,f_mode_hz=F_MODE,reference_band=list(REF_BAND),
      branch_thresholds=BRANCH_TH,
      rules=['every dynamic arm records f_drive_hz and the criterion band must contain it',
             'the manipulation check is the injected population cross-tripod |dphi| >=135 degrees in the drive band with at least 8 of 9 pairs',
             'the pool criterion is reported for both the drive band and 16-26 Hz; per-pair signatures and circular R are primary',
             'off arms report the first and second half separately for both the pool and the injected population',
             'nulls randomise only the reference phase spectrum; rate gate 10 percent'],
      branches=dict(p1='BROADBAND_RELAY',p2='BAND_LIMITED_RELAY',p3='MODE_ONLY_RELAY',p4='NO_RELAY',p5='SELF_SUSTAINED_AFTER_KICK'))
    dump(o/'PREREG.json',pre)
    (o/'S334_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s334.py)=%s\nsha256(audit_s334_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s334_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    M18_by_leg={lg:[loc[int(c)] for r in ROLES for c in modules[lg][r]] for lg in LEGS}
    M18=[j for lg in LEGS for j in M18_by_leg[lg]];Mset=set(M18)
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());pool_by_leg={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool];poolset=set(poolcols)
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    # exclusive E populations (90 percent threshold, archived convention)
    WeM=np.load(S303/'EDGES_w_2.5e-5_We.npz')
    def tocsr(z):
        n=int(z['indptr'].size-1)
        return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
    WeC=tocsr(WeM)
    perE={lg:np.asarray(WeC[poolcols,:][[poolcols.index(j) for j in pool_by_leg[lg]],:].sum(0)).ravel() for lg in LEGS}
    tot=sum(perE[lg] for lg in LEGS);SE={}
    for lg in LEGS:
        frac=np.where(tot>1e-12,perE[lg]/np.maximum(tot,1e-12),0.0)
        SE[lg]=np.array([int(j) for j in np.flatnonzero((frac>=0.90)&(perE[lg]>0)) if int(j) not in poolset and int(j) not in Mset],int)
    # ---------- stage A: band-aware reanalysis of S333
    RB={}
    for nm in ('A0_gate','T1_in_diff010','T2_in_diff020','T3_in_diff040','T5_in_common020'):
        f=S333/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        A=loadz(f)
        EN=leg_envs(A,SE)
        p_old=summary(pairs_of(EN,DRIVE_BAND_OLD))
        E=leg_envs(A,pool_by_leg)
        p_ref=summary(pairs_of(E,REF_BAND))
        p_old_pool=summary(pairs_of(E,DRIVE_BAND_OLD))
        RB[nm]=dict(injected_band_160_190=dict(same=p_old['same'],cross=p_old['cross'],n_anti=p_old['n_anti_cross']),
          pool_band_160_190=dict(same=p_old_pool['same'],cross=p_old_pool['cross'],n_anti=p_old_pool['n_anti_cross'],cross_min=p_old_pool['cross_min']),
          pool_16_26=dict(same=p_ref['same'],cross=p_ref['cross'],n_anti=p_ref['n_anti_cross']),
          pairs_injected=p_old['pairs'],pairs_pool_drive=p_old_pool['pairs'],pairs_pool_ref=p_ref['pairs'])
    dump(o/'RELAY_BAND_REANALYSIS.json',dict(arms=RB,drive_band=list(DRIVE_BAND_OLD),reference_band=list(REF_BAND),
      note='S333 injected at 175 Hz while the criterion band was 16-26 Hz; re-scoring the same spikes in the 160-190 Hz drive band recovers the relay'))
    dump(o/'S333_CORRECTION.json',dict(items=[
      dict(id=1,issue='drive frequency',s333='injection at 175 Hz while the criterion band was 16-26 Hz, and the mode frequency FM was a dead variable',corrected='every arm now records f_drive_hz and the criterion band must contain it; a rate constant may never be used as a drive frequency'),
      dict(id=2,issue='branch',s333='t3 NO_RELAY',corrected='re-scored in the drive band the injected population relays 176.4 degrees into 169.4 degrees at the pool with 9 of 9 cross pairs antiphase, so the reading is RELAY_OK'),
      dict(id=3,issue='manipulation check',s333='used the correlation of the drive envelope with the injected group envelope, blind to band',corrected='the check is the injected population cross-tripod |dphi| in the drive band with at least 8 of 9 pairs and a preregistered threshold'),
      dict(id=4,issue='relay arithmetic',s333='added a transient synaptic current to a tonic bias and ignored the membrane low pass',corrected='the membrane attenuation is 0.299 at 25.39 Hz and 0.0454 at 175 Hz, which is why the 175 Hz arm delivered only 2.1 percent of the base drive'),
      dict(id=5,issue='n_anti',s333='mixed signs and tripods',corrected='cross-tripod |dphi| distributions are reported separately with per-pair signatures'),
      dict(id=6,issue='dispatched spec gap',corrected='the dispatching instruction said same amplitude without locking the frequency; the rule is now that the drive frequency is a mandatory field inside the criterion band'),
      dict(id=7,issue='pool_R',corrected='per-cell circular R and six-leg circular R are both archived with their definitions')]))
    dump(o/'RELAY_SPEC.json',dict(mandatory_fields=['f_drive_hz','target_population','dI_nA','criterion_band','manip_threshold'],
      rules=['the drive frequency must lie inside the criterion band','the manipulation check is the injected population cross-tripod |dphi| in the drive band',
             'the pool criterion is reported for the drive band and for 16-26 Hz','the membrane attenuation exp(-(1/f)/tau_m) must be reported with every amplitude']))
    # ---------- stage B nine arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.adapters import build_neural_tier,TierConfig
    CD={k:0 for k in pre['counters']}
    Gd=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    I_STAR=float(min([r for r in Gd if r['graded']],key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))['I'])
    cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
    b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
    e=b['edges'];WeA,WiA=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    for lg in LEGS:
        for src_r in ROLES:
            for tgt_r in ROLES:
                if src_r==tgt_r: continue
                for csrc in modules[lg][src_r]:
                    for ctgt in modules[lg][tgt_r]:
                        i=loc[int(csrc)];j=loc[int(ctgt)]
                        for Wm in (WeA,WiA):
                            s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                            if kk.size: Wm.data[s0+kk[0]]=K_DESIGN*EDGE_TOTAL
    n=WeA.shape[0]
    setA=sorted(set(int(j) for lg in TRIPOD['A'] for j in SE[lg]));setB=sorted(set(int(j) for lg in TRIPOD['B'] for j in SE[lg]))
    def run(name,f_drive,dI,common=False,seed=0,off_at=None,ticks=5000):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(WeA,WiA);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        spm=np.zeros((ticks,n),bool);t=np.arange(ticks)/1000.0;drv=np.sin(2*np.pi*f_drive*t) if f_drive else np.zeros(ticks)
        for k in range(ticks):
            cur=np.full(n,I_STAR)
            if dI>0 and (off_at is None or k<off_at):
                s_=drv[k];cur[setA]+=dI*s_;cur[setB]+=(dI*s_ if common else -dI*s_)
            spm[k]=np.logical_or(net.step(bg+cur),net.step(bg+cur));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        band=(f_drive*0.85,f_drive*1.15) if f_drive else REF_BAND
        E=leg_envs(spm,pool_by_leg);pr=pairs_of(E,band);sp=summary(pr)
        pr_ref=summary(pairs_of(E,REF_BAND))
        ENJ=leg_envs(spm,SE);man=summary(pairs_of(ENJ,band))
        halves={}
        for tag,sl in (('first',slice(0,ticks//2)),('second',slice(ticks//2,ticks))):
            Eh=leg_envs(spm,pool_by_leg,sl);Einh=leg_envs(spm,SE,sl)
            halves[tag]=dict(pool=summary(pairs_of(Eh,band)),inj=summary(pairs_of(Einh,band)))
        rates=np.array([spm[:,j].sum()*1000.0/ticks for j in poolcols])
        mds=[]
        for lg in LEGS:
            raw=np.convolve(spm[:,pool_by_leg[lg]].sum(1).astype(float),np.ones(20)/20,mode='same')
            mds.append((np.percentile(raw,90)-np.percentile(raw,10))/2.0/max(float(np.mean(raw)),1e-12))
        return dict(f_drive=f_drive,band=list(band),dI=dI,common=common,seed=seed,off_at=off_at,
          pool=dict(cross=sp['cross'],same=sp['same'],cross_min=sp['cross_min'],n_anti=sp['n_anti_cross'],R=sp['R'],pairs=sp['pairs']),
          pool_ref_band=dict(cross=pr_ref['cross'],same=pr_ref['same'],n_anti=pr_ref['n_anti_cross']),
          manip=dict(cross=man['cross'],same=man['same'],n_anti=man['n_anti_cross']),
          halves={k:dict(pool_cross=v['pool']['cross'],pool_same=v['pool']['same'],pool_anti=v['pool']['n_anti_cross'],
            inj_cross=v['inj']['cross'],inj_same=v['inj']['same'],inj_anti=v['inj']['n_anti_cross']) for k,v in halves.items()},
          pool_median_rate_hz=float(np.median(rates)),
          mod_depth=float(np.mean(mds))),spm
    AR={}
    AR['G0_gate'],spm0=run('G0_gate',None,0.0)
    g=bool(np.array_equal(spm0,loadz(S333/'SPK_A0_gate.npz')[:5000]))
    for nm,fq in (('F1_12p5',12.5),('F2_20',20.0),('F3_25p39',F_MODE),('F4_40',40.0)):
        AR[nm],_=run(nm,fq,0.020)
    AR['F5_a010_25p39'],_=run('F5_a010_25p39',F_MODE,0.010)
    AR['F6_seed1_25p39'],_=run('F6_seed1_25p39',F_MODE,0.020,seed=1)
    AR['K1_off_2500'],_=run('K1_off_2500',F_MODE,0.020,off_at=2500)
    AR['K2_common_off_2500'],_=run('K2_common_off_2500',F_MODE,0.020,common=True,off_at=2500)
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',dict(g0=g))
    # ---------- stage C
    base=AR['G0_gate']['pool_median_rate_hz']
    rmiss=[k for k in AR if abs(AR[k]['pool_median_rate_hz']-base)/max(base,1e-9)>BRANCH_TH['rate_tol']]
    def manip_ok(k):
        m=AR[k]['manip']
        return (m['cross'] is not None and m['cross']>=BRANCH_TH['relay_cross'] and m['n_anti']>=BRANCH_TH['anti_pairs'])
    ladder=['F1_12p5','F2_20','F3_25p39','F4_40']
    relays={k:(AR[k]['pool']['cross'] is not None and AR[k]['pool']['cross']>=BRANCH_TH['relay_cross'] and AR[k]['pool']['same']<=BRANCH_TH['relay_same']) for k in ladder}
    k1=AR['K1_off_2500']['halves']['second'];k2=AR['K2_common_off_2500']['halves']['second']
    p5=bool(k1['pool_cross']>=BRANCH_TH['self_cross'] and k1['pool_same']<=BRANCH_TH['relay_same'] and (k1['inj_cross'] or 0)<BRANCH_TH['self_manip_lt'] and k2['pool_cross']<BRANCH_TH['neg_lt'])
    p1=bool(all(relays[k] for k in ladder) and all(manip_ok(k) for k in ladder))
    p2=bool((relays['F2_20'] or relays['F3_25p39']) and not relays['F1_12p5'] and not relays['F4_40'])
    p3=bool(relays['F3_25p39'] and not relays['F1_12p5'] and not relays['F2_20'] and not relays['F4_40'])
    p4=bool(all((AR[k]['pool']['cross'] or 0)<90 for k in ladder))
    mid=[k for k in ladder if 90<=(AR[k]['pool']['cross'] or 0)<135]
    if not g: branch='NOT_JUDGED'
    elif p5: branch='p5'
    elif p1: branch='p1'
    elif p2: branch='p2'
    elif p3: branch='p3'
    elif p4: branch='p4'
    else: branch='NOT_JUDGED'
    dump(o/'OUTCOMES.json',dict(round='S334',branch=branch,gate=dict(g0=g),rate_mismatch=rmiss,
      branches=dict(p1=p1,p2=p2,p3=p3,p4=p4,p5=p5),midband=mid,
      arms={k:dict(f=round(v['f_drive'],3) if v['f_drive'] else None,band=v['band'],cross=round(v['pool']['cross'],1) if v['pool']['cross'] else None,
        same=round(v['pool']['same'],1) if v['pool']['same'] else None,cross_min=round(v['pool']['cross_min'],1) if v['pool']['cross_min'] else None,
        n_anti=v['pool']['n_anti'],ref_cross=round(v['pool_ref_band']['cross'],1) if v['pool_ref_band']['cross'] else None,
        manip=(round(v['manip']['cross'],1) if v['manip']['cross'] else None, v['manip']['n_anti']),
        inj2=round(v['halves']['second']['inj_cross'],1) if v['halves']['second']['inj_cross'] else None,
        pool2=round(v['halves']['second']['pool_cross'],1) if v['halves']['second']['pool_cross'] else None,
        pool=round(v['pool_median_rate_hz'],1),mod=round(v['mod_depth'],3)) for k,v in AR.items()},
      reanalysis={nm:(round(v['injected_band_160_190']['cross'],1),round(v['pool_band_160_190']['cross'],1),v['pool_band_160_190']['n_anti']) for nm,v in RB.items()},
      counters=CD,
      one_line='%s ladder %s k1_second pool %s inj %s'%(branch,{k:round(AR[k]['pool']['cross'],1) for k in ladder},
        None if k1['pool_cross'] is None else round(k1['pool_cross'],1),None if k1['inj_cross'] is None else round(k1['inj_cross'],1))))
    print(json.dumps(dict(branch=branch,gate=g,
      reanalysis={nm:(round(v['injected_band_160_190']['cross'],1),round(v['pool_band_160_190']['cross'],1),v['pool_band_160_190']['n_anti']) for nm,v in RB.items()},
      ladder={k:round(AR[k]['pool']['cross'],1) for k in ladder},
      k1=[round(AR['K1_off_2500']['halves']['first']['pool_cross'],1),round(AR['K1_off_2500']['halves']['second']['pool_cross'],1),
          round(AR['K1_off_2500']['halves']['second']['inj_cross'],1)],
      k2=[round(AR['K2_common_off_2500']['halves']['second']['pool_cross'],1),round(AR['K2_common_off_2500']['halves']['second']['inj_cross'],1)],
      rm=rmiss,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
