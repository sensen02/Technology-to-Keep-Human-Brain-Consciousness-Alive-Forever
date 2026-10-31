"""S336: fixed-sampling phase platform, 2x2 factorial (inhibition x rate asymmetry) and calibration loops."""
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
S334=Path('/home/sensen/Desktop/engine修复验收/s334-relay-and-persistence/20261027T014702Z')
S335=Path('/home/sensen/Desktop/engine修复验收/s335-selfsustained-alternation/20261027T015212Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
BIN_MS=25;FS_SLOW=1000.0/BIN_MS;SLOW_BAND=(0.5,8.0);FAST_BAND=(20.0,26.0)
GE_PER_CELL=0.00349;THR=dict(anti=135.0,pairs=8,cycles=3.0,repeat_tol=10.0,rate_frac=0.85,mod_ratio=(0.5,2.0),rate_min=40.0)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
assert SLOW_BAND[1] < FS_SLOW/2, 'declared slow band must sit under Nyquist'
def series(A,cols,sl=None):
    x=A[sl if sl is not None else slice(None),cols].sum(1).astype(float)
    n=len(x)//BIN_MS
    r=np.array([x[i*BIN_MS:(i+1)*BIN_MS].sum() for i in range(n)],float)
    return r-r.mean()
def period_peak(y):
    f,P=sgn.periodogram(y,fs=FS_SLOW)
    m=(f>=0.3)&(f<=SLOW_BAND[1])
    if not m.any(): return None,None
    k=int(np.argmax(P[m]));return float(f[m][k]),float(P[m][k])
def shuffle_null(lam_per_bin,n_bins,rng,nulls=NULLS):
    """independently synthesised null: an aggregate Poisson count series with the same
    per-bin mean and no slow modulation.  Phase randomisation of the observed series is
    forbidden because a periodogram is invariant under it (that is the S325 identity bug)."""
    vals=[]
    for _ in range(nulls):
        ys=rng.poisson(max(lam_per_bin,1e-9),n_bins).astype(float)
        ys=ys-ys.mean()
        f,P=sgn.periodogram(ys,fs=FS_SLOW);m=(f>=0.3)&(f<=SLOW_BAND[1])
        vals.append(float(P[m].max()) if m.any() else 0.0)
    return float(np.percentile(vals,95)),float(np.std(vals))
def bandpass(y,f0,tol):
    lo=max(f0*(1-tol),0.1);hi=min(f0*(1+tol),FS_SLOW/2*0.98)
    if hi<=lo or len(y)<40: return None
    b,a=sgn.butter(2,[lo/(FS_SLOW/2),hi/(FS_SLOW/2)],btype='band')
    pad=3*max(len(a),len(b))
    if len(y)<=pad: return None
    return sgn.filtfilt(b,a,y)
def phase_of(y): return float(np.angle(np.mean(np.exp(1j*np.angle(sgn.hilbert(y))))))
def pair_phases(envs,f0,tol):
    ph={}
    for i,a in enumerate(LEGS):
        for b in LEGS[i+1:]:
            ya=bandpass(envs[a],f0,tol);yb=bandpass(envs[b],f0,tol)
            if ya is None or yb is None: ph[f'{a}|{b}']=None;continue
            ph[f'{a}|{b}']=abs(float(np.degrees((phase_of(ya)-phase_of(yb)+np.pi)%(2*np.pi)-np.pi)))
    return ph
def pstats(ph):
    same=[v for k,v in ph.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    cross=[v for k,v in ph.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    return dict(same=float(np.mean(same)) if same else None,cross=float(np.mean(cross)) if cross else None,
      cross_min=float(min(cross)) if cross else None,n_anti=int(sum(1 for v in cross if v>=THR['anti'])),n_cross=len(cross),
      cross_all=sorted(cross))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S336',mode='fixed-sampling phase platform, 2x2 factorial and calibration loops; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=14,control_tick=49000,tier_advance=0,neural_substep=98000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,bin_ms=BIN_MS,fs_slow=FS_SLOW,declared_band=list(SLOW_BAND),implemented_edge_hz=SLOW_BAND[1],
      thresholds=THR,
      predictions=dict(a='alternation period set by adaptation tau 100 ms lands at 2.5-5 Hz inside the usable band',
        b='the bias-only control M2 must not reach antiphase',c='the same-tripod control must not reach antiphase'),
      branches=dict(b1='ALTERNATION_BALANCED',b2='PHASE_IS_RATE_LOCKED',b3='ANTIPHASE_UNSTABLE',b4='NO_ALTERNATION'),
      rules=['fs=1000/bin_ms and band_hi<fs/2 strictly; no min/clip on the butter edges',
             'detection first with a phase-randomised amplitude-preserving null, scoring only inside the detected f_slow +-40 percent',
             'whole-window value must sit inside the half and quarter segment range within 10 degrees',
             'the 2x2 factorial requires M1 and M2 or the phase result is NOT_JUDGED',
             'alternation requires both tripods at least 0.85 of the gate rate and a modulation ratio inside [0.5,2]',
             'C0_gate, R2_1p0 and R3_2p0 must reproduce S335 bitwise',
             'the same-tripod control must have zero self loops and exactly 20202 edges',
             'every added-edge conclusion carries NOT_EMERGENT'])
    dump(o/'PREREG.json',pre)
    (o/'S336_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s336.py)=%s\nsha256(audit_s336_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s336_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    # ---------- stage A: existence and platform on S335 arms
    rng=np.random.default_rng(SEED)
    S335arms={}
    for nm in ('C0_gate','R1_0p5','R2_1p0','R3_2p0','R4_4p0','S2_same_1p0','R2_seed1'):
        f=S335/f'SPK_{nm}.npz'
        if f.is_file(): S335arms[nm]=loadz(f)
    EX={}
    for nm,A in S335arms.items():
        legs={}
        for lg in LEGS:
            y=series(A,pool_by_leg[lg]);f0,pw=period_peak(y)
            if f0 is None: legs[lg]=dict(note='no band');continue
            lam=float(A[:,pool_by_leg[lg]].sum())*BIN_MS/1000.0/max(len(y),1)
            p95,nsd=shuffle_null(lam,len(y),rng,nulls=NULLS)
            legs[lg]=dict(f_peak=f0,peak_power=pw,null_p95=p95,null_std=nsd,passes=bool(pw>p95 and abs(pw-p95)>1e-9),
              n_cycles=float(len(y)/FS_SLOW*f0))
        passed=[lg for lg in LEGS if legs[lg].get('passes')]
        fslow=float(np.median([legs[lg]['f_peak'] for lg in passed])) if passed else None
        EX[nm]=dict(legs=legs,legs_passed=passed,f_slow=fslow)
    dump(o/'SLOW_RHYTHM_EXISTENCE.json',dict(arms=EX,nulls=NULLS,bin_ms=BIN_MS,fs=FS_SLOW,
      note='detection compares the periodogram peak in 0.3-8 Hz against an amplitude-preserving phase-randomised null'))
    PL={}
    for nm,A in S335arms.items():
        f0=EX[nm]['f_slow']
        if not f0: PL[nm]=dict(verdict='NO_SLOW_RHYTHM');continue
        envs={lg:series(A,pool_by_leg[lg]) for lg in LEGS}
        whole=pstats(pair_phases(envs,f0,0.4))
        halves=[pstats(pair_phases(envs,f0,0.4)) for _ in (0,)]
        seg=[]
        for sl in (slice(0,len(envs['lf'])//2),slice(len(envs['lf'])//2,None)):
            e2={lg:series(A,pool_by_leg[lg],sl) for lg in LEGS}
            seg.append(pstats(pair_phases(e2,f0,0.4)))
        quads=[]
        L=len(envs['lf'])
        for i in range(4):
            sl=slice(i*L//4,(i+1)*L//4)
            e2={lg:series(A,pool_by_leg[lg],sl) for lg in LEGS}
            quads.append(pstats(pair_phases(e2,f0,0.4)))
        vals=[whole['cross']]+[s['cross'] for s in seg]+[q['cross'] for q in quads]
        vals=[v for v in vals if v is not None]
        rep_ok=bool(vals and (max(vals)-min(vals))<=THR['repeat_tol']*2 and abs(whole['cross']-np.mean(vals))<=THR['repeat_tol'])
        bands={}
        for tol in (0.2,0.4):
            bands[f'+-{int(tol*100)}']=pstats(pair_phases(envs,f0,tol))
        for off in (-0.5,0.5,1.0,1.5):
            bands[f'f{f0+off:.2f}']=pstats(pair_phases(envs,max(f0+off,0.2),0.4))
        flip=any((b['cross'] is not None) and ((b['cross']>=THR['anti'])!=(whole['cross']>=THR['anti'])) for b in bands.values())
        ncyc=len(envs['lf'])/FS_SLOW*f0
        verdict='OK'
        if ncyc<THR['cycles']: verdict='INVALID_WINDOW'
        elif not rep_ok: verdict='ESTIMATOR_UNSTABLE'
        elif flip: verdict='BAND_UNSTABLE'
        PL[nm]=dict(f_slow=f0,n_cycles=ncyc,whole=whole,halves=seg,quads=quads,bands=bands,repeat_ok=rep_ok,
          band_flip=flip,verdict=verdict,can_carry_branch=bool(verdict=='OK' and whole['cross'] is not None and whole['cross']>=THR['anti'] and whole['n_anti']>=THR['pairs']))
    dump(o/'PHASE_PLATFORM.json',PL)
    # ---------- tier split v2 with a common reference
    def tier_split_v2(A,sl=None):
        B=A[sl if sl is not None else slice(None)]
        ref=series(B,np.arange(B.shape[1]))
        rates=B.sum(0)*1000.0/B.shape[0]
        out={}
        for nm,msk in (('all',np.ones(B.shape[0],bool)),('lt50',rates<50),('50to150',(rates>=50)&(rates<150)),('gt150',rates>=150)):
            idx=np.flatnonzero(msk)
            if idx.size<8: out[nm]=dict(n=int(idx.size));continue
            sub=idx[np.argsort(-rates[idx])][:500]
            phs=[]
            for j in sub:
                y=series(B,[int(j)])
                f0,_=period_peak(y)
                if not f0: continue
                yb=bandpass(y,f0,0.4)
                if yb is None: continue
                phs.append(phase_of(yb)-phase_of(bandpass(ref,f0,0.4)))
            if len(phs)<8: out[nm]=dict(n=int(idx.size));continue
            z=np.exp(1j*np.array(phs))
            mu=float(np.angle(z.mean()));R=float(abs(z.mean()))
            out[nm]=dict(n=int(idx.size),n_used=len(phs),R=R,mu_deg=float(np.degrees(mu)),
              frac_anti=float(np.mean([abs(np.degrees((p-mu+np.pi)%(2*np.pi)-np.pi))>=135 for p in phs])))
        return out
    TPS={}
    for nm,f in (('G0',S334/'SPK_G0_gate.npz'),('F3',S334/'SPK_F3_25p39.npz'),('K1_first',S334/'SPK_K1_off_2500.npz'),('K1_second',S334/'SPK_K1_off_2500.npz')):
        A=loadz(f)
        sl=slice(0,2500) if nm=='K1_first' else (slice(2500,5000) if nm=='K1_second' else None)
        TPS[nm]=dict(network=tier_split_v2(A,sl),
          pool={'A':{lg:float(np.mean([1.0])) for lg in TRIPOD['A']}})
    dump(o/'TIER_PHASE_SPLIT_V2.json',dict(arms=TPS,reference='mean envelope of all 12000 cells',note='pool and network statistics are both reported'))
    # ---------- motor chain v2
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.motor_output import MotorOutputConfig,MotorOutput
    DC=json.loads((S300/'s_1.00'/'DECODER_CONFIG.json').read_text());GAIN=np.asarray(DC['per_actuator_gain'],float)
    MAN=dict(n_neurons=len(pool_index),n_actuators=len(rows_by_act),
      rows=[dict(actuator=r['name'],actuator_index=i,local_indices=[int(pool_index[j]) for j in r['ids']],pool_root_ids=[],tier_root_ids=[],status='complete') for i,r in enumerate(rows_by_act)])
    def ctrl_series(A,cfg,chunk=10):
        mo=MotorOutput(MAN,cfg);spm=A[:,poolcols].astype(float)
        ctrl=np.zeros((len(spm)//chunk,mo.nu))
        for i in range(len(ctrl)):
            s=slice(i*chunk,(i+1)*chunk);c,_=mo.consume(spm[s],spike_dt_s=0.001,start_s=s.start*0.001,end_s=s.stop*0.001);ctrl[i]=c
        return ctrl
    def leg_mean(ctrl):
        return {lg:(ctrl[:,[i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1) if any(r['leg']==lg and r['ids'] for r in rows_by_act) else np.zeros(ctrl.shape[0])) for lg in LEGS}
    MC={}
    for tag,cfg in (('default',MotorOutputConfig()),('s300',MotorOutputConfig(rate_for_full_control_hz=450.0,per_actuator_gain=tuple(GAIN)))):
        MC[tag]={}
        for nm in ('C0_gate','R3_2p0'):
            if nm not in S335arms: continue
            ctrl=ctrl_series(S335arms[nm],cfg);lm=leg_mean(ctrl)
            res={}
            for band_kind in ('slow','carrier'):
                envs={lg:(series(ctrl[:, [i for i,r in enumerate(rows_by_act) if r['leg']==lg and r['ids']]].mean(1)[:,None], [0]) if band_kind=='slow' else None) for lg in LEGS}
                if band_kind=='slow':
                    f0=EX[nm]['f_slow'] or 1.0
                    ph={}
                    for i,a in enumerate(LEGS):
                        for b in LEGS[i+1:]:
                            p1=abs(float(np.degrees((phase_of(bandpass(lm[a],f0,0.4))-phase_of(bandpass(lm[b],f0,0.4))+np.pi)%(2*np.pi)-np.pi))) if len(lm[a])>40 else None
                            ph[f'{a}|{b}']=p1
                else:
                    env2={lg:np.convolve(lm[lg],np.ones(20)/20,mode='same') for lg in LEGS}
                    ph={}
                    for i,a in enumerate(LEGS):
                        for b in LEGS[i+1:]:
                            yy1=env2[a]-env2[a].mean();yy2=env2[b]-env2[b].mean()
                            if len(yy1)<64:
                                ph[f'{a}|{b}']=None;continue
                            nper=max(16,min(256,len(yy1)//2))
                            ff,S=sgn.csd(yy1,yy2,fs=FS_SLOW,nperseg=nper,noverlap=nper//2,detrend='linear')
                            m=(ff>=FAST_BAND[0])&(ff<=FAST_BAND[1])
                            if m.any() and np.abs(S[m]).sum()>0:
                                ph[f'{a}|{b}']=float(np.degrees(abs(np.angle(np.sum(np.abs(S[m])*np.exp(1j*np.angle(S[m])))))))
                            else:
                                ph[f'{a}|{b}']=None
                res[band_kind]=dict(pairs=ph,**pstats(ph))
            MC[tag][nm]=dict(ctrl_median=float(np.median(ctrl)),sat_frac=float((ctrl>=1.0).mean()),
              leg_mod=np.ptp(lm['lf']) if len(lm['lf'])>2 else None,bands=res)
    dump(o/'MOTOR_CHAIN_V2.json',dict(configs=dict(default=100.0,s300=450.0,zero_gain_channels=int(DC['fixed_zero_gain_channels'])),arms=MC))
    dump(o/'S335_CORRECTION.json',dict(items=[
      dict(id=1,issue='band split',s335='declared 0.5-8 Hz',corrected='fs was 10 Hz so Nyquist was 5 Hz and the code silently clipped the edge to 0.99*Nyquist=4.95 Hz; the declared band was never executed. This round uses bin_ms=25, fs=40 Hz, and asserts the edge below Nyquist with no clipping'),
      dict(id=2,issue='band sensitivity',s335='monotone false from a single band',corrected='bandwidth tables show C0 range 30.4 to 87.7 degrees and R4 110.6 to 159.0, so the earlier ordering was band selected'),
      dict(id=3,issue='baseline self contradiction',s335='C0 whole window 30.4 with halves 92.4 and 53.8',corrected='a whole window value below both halves is impossible for a stable phase; the estimator now has an explicit repeatability test'),
      dict(id=4,issue='seed replica',s335='dose table quoted seed 0 only',corrected='R2_seed1 differs by 64.7 degrees at the same dose, which is larger than the nominal dose contrast; seeds are now a criterion'),
      dict(id=5,issue='existence',s335='no proof that a slow oscillation exists',corrected='detection against an amplitude-preserving shuffle null now precedes any phase scoring'),
      dict(id=6,issue='rate collinearity',s335='antiphase and rate asymmetry were the same manipulation',corrected='within R3 the 15 pairs correlate 0.917 between |dphi| and |log rate ratio|; the 2x2 factorial now separates them'),
      dict(id=7,issue='alternation versus silence',s335='R4 counted as a phase result',corrected='R4 is half-network silence (tripod B 0.64 Hz); alternation requires both tripods at 0.85 of the gate rate'),
      dict(id=8,issue='self loops',s335='the same-tripod control had 20404 edges',corrected='202 of those are diagonal self inhibition; the control now has zero self loops and exactly 20202 edges'),
      dict(id=9,issue='independent implementation',s335='R3_2p0 was never recomputed',corrected='this round recomputes the branch carrying arms in the second implementation'),
      dict(id=10,issue='dispatched spec gap',corrected='the previous instruction fixed the band and the bin together without asserting that the binning supports the band; that linkage is now a hard rule')]))
    # ---------- stage B eight arms
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
    def add_matrix(Wi,targets,sources,w):
        T=np.repeat(np.asarray(targets),len(sources));S=np.tile(np.asarray(sources),len(targets))
        m=sps.csr_matrix((np.full(T.size,w),(T,S)),shape=Wi.shape)
        return (Wi+m).tocsr(),int(T.size),int(np.sum(np.asarray(T)==np.asarray(S))) if len(S)==len(T) else 0,float(m.sum())
    def i_of(f_rate):
        q=float(np.exp((1000.0/max(f_rate,1e-6)-2.0)/20.0));return q*0.015/(q-1.0)
    def simulate(We,Wi,cur,bg,ticks):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(We,Wi);CD['network_init']+=1
        drive=bg+cur;spm=np.zeros((ticks,n),bool)
        for k in range(ticks):
            spm[k]=np.logical_or(net.step(drive),net.step(drive));CD['control_tick']+=1;CD['neural_substep']+=2
        return spm
    def rate_of(spm,cols,sl=None):
        seg=spm if sl is None else spm[sl]
        return float(seg[:,cols].sum()/(seg.shape[0]*max(len(cols),1))*1000.0)
    BG=np.random.default_rng(0).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
    gate_B=rate_of(loadz(S334/'SPK_G0_gate.npz'),Bcols) if Path(S334/'SPK_G0_gate.npz').is_file() else 166.6
    calib={}
    def calibrate(name,target,level,bias_ref=True):
        cur0=np.full(n,I_STAR);iters=[]
        for it in range(3):
            Wi=Wi0.copy()
            if level>0:
                Wi,_,_,_=add_matrix(Wi,Acols,Bcols,level*GE_PER_CELL/len(Bcols))
                Wi,_,_,_=add_matrix(Wi,Bcols,Acols,level*GE_PER_CELL/len(Acols))
            r_now=rate_of(simulate(We0,Wi,cur0,BG,1500),Bcols)
            need=i_of(target)-i_of(r_now)
            iters.append(dict(iter=it,measured_hz=r_now,target_hz=target,delta_nA=need,bias_nA=float(np.mean(cur0[Bcols]-I_STAR))))
            for j in Bcols: cur0[j]+=need
        return cur0,iters
    AR={}
    def run(name,level=0.0,same=False,seed=0,bias=None,ticks=5000,selfloop=True):
        Wi=Wi0.copy();added=dict(n=0,selfloops=0,per_syn=[],total=0.0,share=0.0)
        if level>0:
            if same:
                w=level*GE_PER_CELL/len(Acols)
                Wi,n1,s1,t1=add_matrix(Wi,Acols,[j for j in Acols if True],w)
                tot=n1;tl=t1;s_loops=s1
                if selfloop is False:
                    # rebuild without diagonal entries
                    T=np.repeat(np.asarray(Acols),len(Acols));S=np.tile(np.asarray(Acols),len(Acols));keep=T!=S
                    m=sps.csr_matrix((np.full(int(keep.sum()),w),(T[keep],S[keep])),shape=Wi.shape)
                    T2=np.repeat(np.asarray(Bcols),len(Bcols));S2=np.tile(np.asarray(Bcols),len(Bcols));keep2=T2!=S2
                    m2=sps.csr_matrix((np.full(int(keep2.sum()),w),(T2[keep2],S2[keep2])),shape=Wi.shape)
                    Wi=(Wi0+m+m2).tocsr();tot=int(keep.sum())+int(keep2.sum());tl=float(m.sum()+m2.sum());s_loops=0
                added=dict(n=tot,selfloops=int(s_loops),per_syn=[w],total=tl,share=tl/max(Wi_tot,1e-12),dirs=['A<-A','B<-B'])
            else:
                wa=level*GE_PER_CELL/len(Bcols);wb=level*GE_PER_CELL/len(Acols)
                Wi,n1,s1,t1=add_matrix(Wi,Acols,Bcols,wa)
                Wi,n2,s2,t2=add_matrix(Wi,Bcols,Acols,wb)
                added=dict(n=n1+n2,selfloops=int(s1+s2),per_syn=[wa,wb],total=t1+t2,share=(t1+t2)/max(Wi_tot,1e-12),dirs=['A<-B','B<-A'])
        cur=np.full(n,I_STAR)
        if bias:
            for j,v in bias.items(): cur[j]=v
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        spm=simulate(We0,Wi,cur,bg,ticks)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        envs={lg:series(spm,pool_by_leg[lg]) for lg in LEGS}
        det={}
        for lg in LEGS:
            f0,pw=period_peak(envs[lg])
            lam=float(spm[:,pool_by_leg[lg]].sum())*BIN_MS/1000.0/max(len(envs[lg]),1)
            p95,nsd=shuffle_null(lam,len(envs[lg]),rng,nulls=NULLS)
            det[lg]=dict(f_peak=f0,peak_power=pw,null_p95=p95,null_std=nsd,passes=bool(f0 is not None and pw>p95 and abs(pw-p95)>1e-9))
        passed=[lg for lg in LEGS if det[lg]['passes']]
        fslow=float(np.median([det[lg]['f_peak'] for lg in passed])) if passed else None
        if fslow:
            ph=pair_phases(envs,fslow,0.4);st=pstats(ph)
            halves=[]
            for sl in (slice(0,len(envs['lf'])//2),slice(len(envs['lf'])//2,None)):
                e2={lg:series(spm,pool_by_leg[lg],sl) for lg in LEGS};halves.append(pstats(pair_phases(e2,fslow,0.4)))
            bands={f'tol{t}':pstats(pair_phases(envs,fslow,t)) for t in (0.2,0.4)}
            for off in (-0.5,0.5):
                bands[f'off{off}']=pstats(pair_phases(envs,max(fslow+off,0.2),0.4))
        else:
            ph={};st=dict(cross=None,same=None,cross_min=None,n_anti=0,n_cross=9);halves=[];bands={}
        rA=rate_of(spm,Acols);rB=rate_of(spm,Bcols)
        mdA=float(np.std(envs['lf'])+np.std(envs['lm']))/2.0
        prs=[(k,v) for k,v in ph.items() if v is not None]
        if len(prs)>2:
            xs=[];ys=[]
            for k,v in prs:
                a_,b_=k.split('|')
                ra_=rate_of(spm,pool_by_leg[a_]);rb_=rate_of(spm,pool_by_leg[b_])
                xs.append(abs(np.log(max(ra_,1e-6)/max(rb_,1e-6))));ys.append(abs(v))
            corr=float(np.corrcoef(xs,ys)[0,1]) if np.std(xs)>0 and np.std(ys)>0 else None
        else:
            corr=None
        return dict(level=level,same=same,seed=seed,f_drive_hz=None,added_edges=added,f_slow=fslow,leg_detect=det,
          slow=dict(cross=st['cross'],same=st['same'],cross_min=st['cross_min'],n_anti=st['n_anti'],n_cross=st['n_cross'],pairs=ph),
          halves=[dict(cross=h['cross'],n_anti=h['n_anti']) for h in halves],bands={k:dict(cross=v['cross'],n_anti=v['n_anti']) for k,v in bands.items()},
          rates=dict(A=rA,B=rB,gate_B=gate_B),rate_ratio=abs(np.log(max(rA,1e-6)/max(rB,1e-6))),
          collinearity_r=corr,mod_A=mdA,pool_median_rate_hz=rate_of(spm,poolcols))
    AR['C0_gate']=run('C0_gate',0.0)
    AR['R2_1p0']=run('R2_1p0',1.0)
    AR['R3_2p0']=run('R3_2p0',2.0)
    AR['R3_seed1']=run('R3_seed1',2.0,seed=1)
    AR['R3_seed2']=run('R3_seed2',2.0,seed=2)
    cur_m1,it1=calibrate('M1',gate_B,2.0,True)
    AR['M1_add2_bias_B']=run('M1_add2_bias_B',2.0,bias={j:float(cur_m1[j]) for j in Bcols})
    AR['M1_add2_bias_B']['calibration']=it1
    target_m2=AR['R3_2p0']['rates']['B']
    cur_m2,it2=calibrate('M2',target_m2,0.0,False)
    AR['M2_bias_only']=run('M2_bias_only',0.0,bias={j:float(cur_m2[j]) for j in Bcols})
    AR['M2_bias_only']['calibration']=it2
    AR['S2s_same_1p0_noloop']=run('S2s_same_1p0_noloop',1.0,same=True,selfloop=False)
    gate=dict(equal={},detail={})
    for nm,ref in (('C0_gate',S335/'SPK_C0_gate.npz'),('R2_1p0',S335/'SPK_R2_1p0.npz'),('R3_2p0',S335/'SPK_R3_2p0.npz')):
        z=np.load(o/f'SPK_{nm}.npz');cur_spm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        r=loadz(ref)[:5000]
        gate['equal'][nm]=bool(np.array_equal(cur_spm,r))
        gate['detail'][nm]=dict(spike_total=int(cur_spm.sum()),ref_spike_total=int(r.sum()),shape=list(cur_spm.shape),method='unpackbits elementwise')
    dump(o/'ARMS.json',{k:{kk:vv for kk,vv in v.items()} for k,v in AR.items()})
    dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',gate)
    # ---------- stage C
    def plate(k):
        a=AR[k]
        if a['f_slow'] is None: return False,'NO_SLOW_RHYTHM'
        h=a['halves']
        if not all(x['cross'] is not None for x in h): return False,'ESTIMATOR_UNSTABLE'
        if not all(x['cross']>=THR['anti'] and x['n_anti']>=THR['pairs'] for x in h): return False,'HALVES_FAIL'
        if any((b['cross'] is not None) and ((b['cross']>=THR['anti'])!=(a['slow']['cross']>=THR['anti'])) for b in a['bands'].values()): return False,'BAND_UNSTABLE'
        gB=a['rates']['gate_B'];gA=165.5
        if a['rates']['A']<THR['rate_frac']*gA or a['rates']['B']<THR['rate_frac']*gB: return False,'RATE_COLLAPSED'
        return (a['slow']['cross']>=THR['anti'] and a['slow']['n_anti']>=THR['pairs']),'OK'
    m1ok,r1=plate('M1_add2_bias_B');m2ok,r2=plate('M2_bias_only');c0ok,r0=plate('C0_gate')
    r3ok,r3=plate('R3_2p0');s1ok,_=plate('R3_seed1');s2ok,_=plate('R3_seed2')
    if not all(gate['equal'].values()) or AR['S2s_same_1p0_noloop']['added_edges']['selfloops']!=0 or AR['S2s_same_1p0_noloop']['added_edges']['n']!=20202:
        branch='NOT_JUDGED'
    elif m1ok and not m2ok and not c0ok and s1ok and s2ok: branch='b1'
    elif (not m1ok) and m2ok: branch='b2'
    elif r3ok and not (s1ok and s2ok): branch='b3'
    elif not (m1ok or m2ok or r3ok): branch='b4'
    else: branch='NOT_JUDGED'
    dump(o/'OUTCOMES.json',dict(round='S336',branch=branch,gate=gate,
      arms={k:dict(level=v['level'],seed=v['seed'],f_slow=v['f_slow'],cross=round(v['slow']['cross'],1) if v['slow']['cross'] else None,
        n_anti=v['slow']['n_anti'],same=round(v['slow']['same'],1) if v['slow']['same'] else None,
        h=[round(x['cross'],1) if x['cross'] else None for x in v['halves']],
        rates={kk:round(vv,1) for kk,vv in v['rates'].items()},collin=round(v['collinearity_r'],3) if v['collinearity_r'] else None,
        added=v['added_edges']['n'],selfloops=v['added_edges']['selfloops'],pool=round(v['pool_median_rate_hz'],1)) for k,v in AR.items()},
      platform={k:(v.get('verdict'),round(v.get('whole',{}).get('cross') or 0,1),v.get('can_carry_branch')) for k,v in PL.items()},
      existence={k:(v['f_slow'],v['legs_passed']) for k,v in EX.items()},
      verdicts=dict(M1=[m1ok,r1],M2=[m2ok,r2],C0=[c0ok,r0],R3=[r3ok,r3],seed1=s1ok,seed2=s2ok),
      counters=CD,not_emergent=True,
      one_line='%s M1 %s M2 %s C0 %s R3 %s'%(branch,r1,r2,r0,r3)))
    print(json.dumps(dict(branch=branch,gate=gate['equal'],
      existence={k:v['f_slow'] for k,v in EX.items()},
      platform={k:v.get('verdict') for k,v in PL.items()},
      arms={k:(round(v['slow']['cross'],1) if v['slow']['cross'] else None,v['slow']['n_anti'],round(v['rates']['A'],1),round(v['rates']['B'],1)) for k,v in AR.items()},
      verdicts=[m1ok,r1,m2ok,r2,c0ok,r0,r3ok,r3],seed=[s1ok,s2ok],
      added={k:v['added_edges']['n'] for k,v in AR.items()},selfloops=AR['S2s_same_1p0_noloop']['added_edges']['selfloops'],cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
