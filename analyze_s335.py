"""S335: motor chain closure, tier phase split and self-sustained alternation by added inhibition."""
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
S332=Path('/home/sensen/Desktop/engine修复验收/s332-phase-estimator-and-tripod-differential/20261027T013035Z')
S334=Path('/home/sensen/Desktop/engine修复验收/s334-relay-and-persistence/20261027T014702Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
SLOW_BAND=(0.5,8.0);FAST_BAND=(20.0,26.0);BIN_MS=100.0;FS_SLOW=1000.0/BIN_MS
GE_PER_CELL=0.00349;GI_PER_CELL=0.00475
BRANCH_TH=dict(anti_cross=135.0,anti_pairs=8,rate_min=40.0,desync_corr=0.5,same_less=0.5)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def slow_env(x):
    n=len(x);nb=max(n//int(BIN_MS),8)
    r=np.array([x[i*int(BIN_MS):(i+1)*int(BIN_MS)].sum() for i in range(nb)],float)
    r=r-r.mean()
    b,a=sgn.butter(3,[SLOW_BAND[0]/(FS_SLOW/2),min(SLOW_BAND[1]/(FS_SLOW/2),0.99)],btype='band')
    return sgn.filtfilt(b,a,r) if len(r)>12 else r
def phase_slow(y):
    z=sgn.hilbert(y);return float(np.angle(np.mean(np.exp(1j*np.angle(z)))))
def env_fast(x):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::2]
def band_phase(x,y,band):
    n=min(len(x),len(y));x=np.asarray(x)[:n];y=np.asarray(y)[:n];nper=min(1024,n)
    if nper<64: return None
    f,S=sgn.csd(x,y,fs=1/(2e-3),nperseg=nper,noverlap=nper//2,detrend='linear')
    m=(f>=band[0])&(f<=band[1])
    if not m.any() or np.abs(S[m]).sum()<=0: return None
    w=np.abs(S[m]);return float(np.degrees(np.angle(np.sum(w*np.exp(1j*np.angle(S[m]))))))
def pair_stats(ph):
    same=[abs(ph[f'{a}|{b}']) for i,a in enumerate(LEGS) for b in LEGS[i+1:] if any(a in TRIPOD[t] and b in TRIPOD[t] for t in TRIPOD) and ph.get(f'{a}|{b}') is not None]
    cross=[abs(ph[f'{a}|{b}']) for i,a in enumerate(LEGS) for b in LEGS[i+1:] if not any(a in TRIPOD[t] and b in TRIPOD[t] for t in TRIPOD) and ph.get(f'{a}|{b}') is not None]
    return dict(same=float(np.mean(same)) if same else None,cross=float(np.mean(cross)) if cross else None,
      cross_min=float(min(cross)) if cross else None,n_anti=int(sum(1 for v in cross if v>=BRANCH_TH['anti_cross'])),n_cross=len(cross))
def leg_pool_envs(A,pool_by_leg,sl=None):
    return {lg:slow_env(A[sl if sl is not None else slice(None),pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
def slow_phases(A,pool_by_leg,sl=None):
    E=leg_pool_envs(A,pool_by_leg,sl)
    ph={f'{a}|{b}':phase_slow(E[a])-phase_slow(E[b]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
    ph={k:float(np.degrees((v+np.pi)%(2*np.pi)-np.pi)) for k,v in ph.items()}
    return {k:abs(v) for k,v in ph.items()}
def build_motor(rows_by_act,pool_index):
    rows=[]
    for ai,r in enumerate(rows_by_act):
        rows.append(dict(actuator=r['name'],actuator_index=ai,local_indices=[int(pool_index[j]) for j in r['ids']],
          pool_root_ids=[],tier_root_ids=[],status='complete'))
    return dict(n_neurons=len(pool_index),n_actuators=len(rows),rows=rows)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S335',mode='motor chain closure, tier phase split and self-sustained alternation by added inhibition; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=7,control_tick=35000,tier_advance=0,neural_substep=70000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,criterion_band_slow=list(SLOW_BAND),diagnostic_band=list(FAST_BAND),bin_ms=BIN_MS,
      branch_thresholds=BRANCH_TH,added_edge_levels=[0.5,1.0,2.0,4.0],
      predictions=dict(a='if reciprocal inhibition produces alternation the period is set by adaptation_tau_ms=100 so the slow envelope peak falls at 2.5-5 Hz',
        b='the same-tripod control S2_same_1p0 must not produce antiphase',c='doses below 1x leave the slow phase difference under 45 degrees'),
      rules=['every external current is time independent and f_drive_hz is null',
             'every added edge is declared with count, direction, per-synapse conductance and share of the original Wi',
             'the primary criterion is slow envelope 0.5-8 Hz; the 20-26 Hz carrier is diagnostic',
             'both halves must pass; RATE_COLLAPSED below 40 Hz is removed from the primary criterion',
             'nulls are independently synthesised; no absolute floors; no else fallback'])
    dump(o/'PREREG.json',pre)
    (o/'S335_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s335.py)=%s\nsha256(audit_s335_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s335_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    M18_by_leg={lg:[loc[int(c)] for r in ROLES for c in modules[lg][r]] for lg in LEGS}
    seen={};rows_by_act=[]
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        ids=[loc[int(x)] for x in json.loads(r['pool_root_ids']) if int(x) in loc] if r.get('pool_root_ids') else []
        if lg in LEGS and ids:
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
        rows_by_act.append(dict(name=r.get('name'),leg=lg,ids=ids))
    pool=sorted(seen.items());pool_by_leg={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    poolcols=[j for (_l,_g),j in pool];pool_index={j:i for i,j in enumerate(poolcols)}
    Acols=[j for lg in TRIPOD['A'] for j in pool_by_leg[lg]];Bcols=[j for lg in TRIPOD['B'] for j in pool_by_leg[lg]]
    def loadz(p):
        z=np.load(p);return np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    G0=loadz(S334/'SPK_G0_gate.npz');F3=loadz(S334/'SPK_F3_25p39.npz');K1=loadz(S334/'SPK_K1_off_2500.npz')
    # ---------- stage A1: tier phase split
    def tier_split(A,sl=None):
        B=A[sl if sl is not None else slice(None)]
        rate=B.sum(0)*1000.0/B.shape[0]
        rows={}
        bands={'all':np.ones(B.shape[0],bool),'lt50':rate<50,'50to150':(rate>=50)&(rate<150),'150to300':(rate>=150)&(rate<300),'gt300':rate>=300}
        for nm,msk in bands.items():
            idx=np.flatnonzero(msk)
            if idx.size<8: rows[nm]=dict(n=int(idx.size));continue
            sub=idx[np.argsort(-rate[idx])][:600]
            phs=[];amps=[]
            for j in sub:
                y=env_fast(B[:,j].astype(float))
                f,S=sgn.csd(y,y,fs=1/(2e-3),nperseg=min(1024,len(y)),noverlap=512)
                m=(f>=FAST_BAND[0])&(f<=FAST_BAND[1])
                if not m.any(): continue
                phs.append(float(np.angle(np.sum(np.abs(S[m])*np.exp(1j*np.angle(S[m]))))));amps.append(float(np.abs(S[m]).sum()))
            if len(phs)<8: rows[nm]=dict(n=int(idx.size));continue
            z=np.sum(np.asarray(amps)*np.exp(1j*np.asarray(phs)))/max(np.sum(amps),1e-30)
            mu=float(np.angle(z));R=float(abs(z))
            rows[nm]=dict(n=int(idx.size),n_used=len(phs),R=R,mu_deg=float(np.degrees(mu)),
              frac_within_45=float(np.mean([abs(np.degrees((p-mu+np.pi)%(2*np.pi)-np.pi))<=45 for p in phs])),
              frac_anti_135=float(np.mean([abs(np.degrees((p-mu+np.pi)%(2*np.pi)-np.pi))>=135 for p in phs])))
        return rows
    TPS=dict(G0=tier_split(G0),F3=tier_split(F3),K1_first=tier_split(K1,slice(0,2500)),K1_second=tier_split(K1,slice(2500,5000)),
      pool_only=dict(G0=pair_stats(slow_phases(G0,pool_by_leg)),F3=pair_stats(slow_phases(F3,pool_by_leg)),
        K1_first=pair_stats(slow_phases(K1,pool_by_leg,slice(0,2500))),K1_second=pair_stats(slow_phases(K1,pool_by_leg,slice(2500,5000)))))
    dump(o/'TIER_PHASE_SPLIT.json',TPS)
    # ---------- stage A2: motor chain
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    MAN=build_motor(rows_by_act,pool_index)
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text())
    GAIN=np.asarray(DC['per_actuator_gain'],float)
    def motor_series(A,cfg,chunk=10):
        mo=MotorOutput(MAN,cfg)
        spm=A[:,poolcols].astype(float)
        ctrl=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(ctrl)):
            s=slice(i*chunk,(i+1)*chunk)
            c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001)
            ctrl[i]=c
        return ctrl
    def leg_ctrl(ctrl):
        out={}
        for lg in LEGS:
            cols=[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]
            out[lg]=ctrl[:,cols] if cols else np.zeros((ctrl.shape[0],0))
        return out
    def leg_tripod_phase(ctrl):
        E={lg:slow_env(ctrl[:,i].astype(float)) for lg,i in [(lg,k) for lg in LEGS for k in [np.argmax([sum(ld[:,i]) if ld.shape[1]>0 else 0 for ld in [leg_ctrl(ctrl)]])]]}
        return None
    def slow_pair_phases(legser):
        ph={f'{a}|{b}':float(np.degrees((phase_slow(legser[a])-phase_slow(legser[b])+np.pi)%(2*np.pi)-np.pi)) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        return {k:abs(v) for k,v in ph.items()}
    MC={}
    for tag,cfg in (('default',MotorOutputConfig()),('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN)))):
        for nm,A in (('G0',G0),('F3',F3),('K1_first',K1[:2500]),('K1_second',K1[2500:])):
            ctrl=motor_series(A,cfg)
            lc=leg_ctrl(ctrl)
            ser={lg:lc[lg].mean(1) if lc[lg].shape[1] else np.zeros(ctrl.shape[0]) for lg in LEGS}
            ph=slow_pair_phases(ser)
            st=pair_stats(ph)
            tri={t:float(np.median([LC for lg in TRIPOD[t] for LC in [lc[lg].mean() if lc[lg].shape[1] else 0]])) for t in TRIPOD}
            MC.setdefault(tag,{})[nm]=dict(ctrl_median=float(np.median(ctrl)),sat_frac=float((ctrl>=1.0).mean()),
              leg_pp={lg:float(np.ptp(lc[lg].mean(1))) if lc[lg].shape[1] else None for lg in LEGS},
              leg_mod_frac={lg:float(np.ptp(lc[lg].mean(1))) if lc[lg].shape[1] else None for lg in LEGS},
              within_leg_spread={lg:float(np.median(lc[lg].std(1))) if lc[lg].shape[1] else None for lg in LEGS},
              cross=st['cross'],same=st['same'],n_anti=st['n_anti'],tripod_median=tri)
    MC['configs']=dict(default=dict(rate_for_full_control_hz=100.0),s300=dict(rate_for_full_control_hz=450.0,zero_gain_channels=int(DC['fixed_zero_gain_channels'])))
    MC['criterion']=dict(calibrated_cross=135.0,calibrated_pairs=8,min_leg_mod_frac=0.05)
    dump(o/'MOTOR_CHAIN.json',MC)
    # ---------- stage A3: mod depth single definition
    def mod_depth_slow(A):
        mds=[]
        for lg in LEGS:
            y=slow_env(A[:,pool_by_leg[lg]].sum(1).astype(float))
            mds.append(float((np.percentile(y,90)-np.percentile(y,10))/2.0/max(abs(np.mean(y))+np.std(y),1e-12)))
        return float(np.median(mds))
    dump(o/'MOD_DEPTH_DEFINITION.json',dict(definition='mod depth = (p90-p10)/2 of the 100 ms binned, 0.5-8 Hz band passed leg-aggregate envelope, divided by the envelope standard deviation plus absolute mean',
      current=dict(G0=mod_depth_slow(G0),F3=mod_depth_slow(F3),K1_first=mod_depth_slow(K1[:2500]),K1_second=mod_depth_slow(K1[2500:])),
      superseded='the S325/S326/S327/S334 numbers used a 20 ms envelope divided by the detrended mean and are NOT comparable; they are marked superseded here'))
    dump(o/'S334_CORRECTION.json',dict(items=[
      dict(id=1,issue='p1 wording',s334='broadband relay',corrected='decompose it: low threshold 0.010 nA, broadband 12.5-40 Hz, and no self-sustainment; with 74.7 percent of pool excitation injected a sign flip being forwarded is the expected linear result'),
      dict(id=2,issue='gate evidence',s334='GATE.json held a bare boolean',corrected='the gate now records spike totals, unpacked shapes and the comparison method'),
      dict(id=3,issue='mod depth',s334='0.172 for G0 while S325 reported 0.137 and 0.026',corrected='a single definition is fixed in MOD_DEPTH_DEFINITION and earlier numbers are marked superseded'),
      dict(id=4,issue='leakage',s334='the 16-26 Hz band was called undisturbed',corrected='F1 shows 28.9 and F4 11.6 degrees in the reference band versus 9.5 at baseline, so the reference band absorbs drive leakage and is not a baseline'),
      dict(id=5,issue='rate gate',s334='rate gate barely passed',corrected='pool median fell 6.4 to 8.6 percent; every arm now reports the pool median and both tripod rates')]))
    # ---------- stage B seven arms
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
    n=We0.shape[0]
    Wi_orig_total=float(Wi0.sum())
    def add_inhibition(Wi,targets,sources,w_per):
        T=np.repeat(np.asarray(targets),len(sources));S=np.tile(np.asarray(sources),len(targets))
        add=sps.csr_matrix((np.full(T.size,w_per),(T,S)),shape=Wi.shape)
        return (Wi+add).tocsr(),int(T.size),float(add.sum())
    def run(name,level=0.0,same=False,seed=0,ticks=5000):
        Wi=Wi0.copy();added=dict(n=0,dirs=[],per_syn=0.0,total=0.0,share_of_Wi=0.0)
        if level>0:
            w=level*GE_PER_CELL/max(len(Bcols),1)
            if same:
                Wi,n1,t1=add_inhibition(Wi,Acols,Acols,w)
                Wi,n2,t2=add_inhibition(Wi,Bcols,Bcols,w)
                added=dict(n=n1+n2,dirs=['A<-A','B<-B'],per_syn=w,total=t1+t2,share_of_Wi=(t1+t2)/max(Wi_orig_total,1e-12))
            else:
                Wi,n1,t1=add_inhibition(Wi,Acols,Bcols,level*GE_PER_CELL/max(len(Bcols),1))
                Wi,n2,t2=add_inhibition(Wi,Bcols,Acols,level*GE_PER_CELL/max(len(Acols),1))
                added=dict(n=n1+n2,dirs=['A<-B','B<-A'],per_syn=[level*GE_PER_CELL/len(Bcols),level*GE_PER_CELL/len(Acols)],total=t1+t2,share_of_Wi=(t1+t2)/max(Wi_orig_total,1e-12))
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We0,Wi);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        drive=bg+np.full(n,I_STAR)
        spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        ph_all=slow_phases(spm,pool_by_leg)
        st=pair_stats(ph_all)
        halves={}
        for tag,sl in (('first',slice(0,ticks//2)),('second',slice(ticks//2,ticks))):
            halves[tag]=pair_stats(slow_phases(spm,pool_by_leg,sl))
        fast={f'{a}|{b}':band_phase(env_fast(spm[:,pool_by_leg[a]].sum(1).astype(float)),env_fast(spm[:,pool_by_leg[b]].sum(1).astype(float)),FAST_BAND) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        fast={k:(abs(v) if v is not None else None) for k,v in fast.items()}
        rates=np.array([spm[:,j].sum()*1000.0/ticks for j in poolcols])
        E=leg_pool_envs(spm,pool_by_leg)
        corr=float(np.mean([np.corrcoef(E[a],E[b])[0,1] for i,a in enumerate(LEGS) for b in LEGS[i+1:]]))
        return dict(level=level,same=same,seed=seed,f_drive_hz=None,dI_time_dependent=False,added_edges=added,
          slow=dict(cross=st['cross'],same=st['same'],cross_min=st['cross_min'],n_anti=st['n_anti'],n_cross=st['n_cross'],pairs=ph_all),
          halves={k:dict(cross=v['cross'],same=v['same'],n_anti=v['n_anti']) for k,v in halves.items()},
          fast_diag=pair_stats(fast),pool_median_rate_hz=float(np.median(rates)),
          tripod_rates=dict(A=float(np.median([spm[:,j].sum()*1000.0/ticks for j in Acols])),
            B=float(np.median([spm[:,j].sum()*1000.0/ticks for j in Bcols]))),
          env_corr=corr,mod_depth=mod_depth_slow(spm)),spm
    AR={}
    AR['C0_gate'],spm0=run('C0_gate',0.0)
    gate=dict(equal=bool(np.array_equal(spm0,loadz(S334/'SPK_G0_gate.npz')[:5000])),
      spike_total=int(spm0.sum()),ref_spike_total=int(loadz(S334/'SPK_G0_gate.npz')[:5000].sum()),
      shape=list(spm0.shape),method='unpackbits(packed)[:, :shape[1]] elementwise equality')
    for nm,lv in (('R1_0p5',0.5),('R2_1p0',1.0),('R3_2p0',2.0),('R4_4p0',4.0)):
        AR[nm],_=run(nm,lv)
    AR['S2_same_1p0'],_=run('S2_same_1p0',1.0,same=True)
    AR['R2_seed1'],_=run('R2_seed1',1.0,seed=1)
    dump(o/'ARMS.json',{k:{kk:vv for kk,vv in v.items() if kk!='spm'} for k,v in AR.items()})
    dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',gate)
    # ---------- stage C labels
    doses=['R1_0p5','R2_1p0','R3_2p0','R4_4p0']
    def passes(k):
        a=AR[k]
        if a['pool_median_rate_hz']<BRANCH_TH['rate_min']: return False
        h=a['halves']
        return all(h[t]['cross'] is not None and h[t]['cross']>=BRANCH_TH['anti_cross'] and h[t]['n_anti']>=BRANCH_TH['anti_pairs'] for t in ('first','second'))
    same_fails=not passes('S2_same_1p0')
    a1=any(passes(k) for k in doses) and same_fails
    desync=[k for k in doses if (AR[k]['env_corr'] is not None and AR[k]['env_corr']<BRANCH_TH['desync_corr'])]
    collapsed=[k for k in doses if AR[k]['pool_median_rate_hz']<BRANCH_TH['rate_min']]
    vals=[AR[k]['slow']['cross'] for k in doses]
    mono=all(vals[i]<=vals[i+1]+5 for i in range(len(vals)-1)) if all(v is not None for v in vals) else False
    a4=bool(mono and max(v or 0 for v in vals)<BRANCH_TH['anti_cross'])
    a2=bool(desync and not a1);a3=bool(len(collapsed)==len(doses))
    if not gate['equal']: branch='NOT_JUDGED'
    elif a1: branch='a1'
    elif a2: branch='a2'
    elif a3: branch='a3'
    elif a4: branch='a4'
    else: branch='NOT_JUDGED'
    dump(o/'OUTCOMES.json',dict(round='S335',branch=branch,gate=gate,
      arms={k:dict(level=v['level'],same=v['same'],added=v['added_edges'],slow_cross=round(v['slow']['cross'],1) if v['slow']['cross'] else None,
        slow_same=round(v['slow']['same'],1) if v['slow']['same'] else None,n_anti=v['slow']['n_anti'],
        h1=round(v['halves']['first']['cross'],1) if v['halves']['first']['cross'] else None,
        h2=round(v['halves']['second']['cross'],1) if v['halves']['second']['cross'] else None,
        fast=round(v['fast_diag']['cross'],1) if v['fast_diag']['cross'] else None,
        pool=round(v['pool_median_rate_hz'],1),tripod=v['tripod_rates'],corr=round(v['env_corr'],3),mod=round(v['mod_depth'],3)) for k,v in AR.items()},
      dose_table={k:round(AR[k]['slow']['cross'],1) for k in doses},monotone=mono,same_control_fails=same_fails,
      branches=dict(a1=a1,a2=a2,a3=a3,a4=a4),counters=CD,
      one_line='%s dose %s same_control %s'%(branch,{k:round((AR[k]['slow']['cross'] or 0),1) for k in doses},same_fails)))
    print(json.dumps(dict(branch=branch,gate=gate['equal'],
      dose={k:(round(AR[k]['slow']['cross'],1) if AR[k]['slow']['cross'] else None,
              round(AR[k]['halves']['first']['cross'],1) if AR[k]['halves']['first']['cross'] else None,
              round(AR[k]['halves']['second']['cross'],1) if AR[k]['halves']['second']['cross'] else None,
              AR[k]['slow']['n_anti'],round(AR[k]['pool_median_rate_hz'],1)) for k in doses},
      same=round(AR['S2_same_1p0']['slow']['cross'],1) if AR['S2_same_1p0']['slow']['cross'] else None,
      added=AR['R2_1p0']['added_edges']['n'],cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
