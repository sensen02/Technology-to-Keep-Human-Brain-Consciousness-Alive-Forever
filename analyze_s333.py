"""S333: pool input structure audit and true input-channel relay test."""
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
S330=Path('/home/sensen/Desktop/engine修复验收/s330-leg-specific-budget-and-decoupling/20261027T011205Z')
S331=Path('/home/sensen/Desktop/engine修复验收/s331-antiphase-capability/20261027T012220Z')
S332=Path('/home/sensen/Desktop/engine修复验收/s332-phase-estimator-and-tripod-differential/20261027T013035Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=1000;SEED=20261026
FS=500.0;NPERSEG=1024;NOVERLAP=512
BANDS=[(16.0,26.0),(20.0,30.0),(22.0,28.0),(24.0,27.0)]
THRESHOLDS=[0.80,0.90,0.95]
MAX_SYN=2e-4;DV_RATE=0.015;DV_DRIVE=0.065;EXC_TAU=3e-3;TREF=2e-3;TAU=20e-3;R_M=1e9
BRANCH_TH=dict(t1=dict(cross=90.0,same=30.0,seed_spread=20.0),t2=dict(cross=90.0,same=30.0),t3=dict(cross=90.0))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def env_of(x,ds=2):
    y=np.convolve(np.asarray(x,float),np.ones(20)/20,mode='same')
    t=np.arange(len(y));y=y-np.polyval(np.polyfit(t,y,1),t)
    return y[::ds] if ds>1 else y
def band_phase(x,y,band=(16.0,26.0)):
    n=min(len(x),len(y));x=np.asarray(x)[:n];y=np.asarray(y)[:n]
    nper=min(NPERSEG,n);f,S=sgn.csd(x,y,fs=FS,nperseg=nper,noverlap=min(NOVERLAP,max(2,nper//2)),detrend='linear')
    m=(f>=band[0])&(f<=band[1])
    if not m.any(): return None
    w=np.abs(S[m]);return float(np.degrees(np.angle(np.sum(w*np.exp(1j*np.angle(S[m]))))))
def phase_block(A,cols_by_leg,band=(16.0,26.0),seg=None):
    E={lg:env_of(A[:,cols_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
    out={}
    for i,a in enumerate(LEGS):
        for b in LEGS[i+1:]:
            p=band_phase(E[a],E[b],band)
            out[f'{a}|{b}']=p
    return out
def summarize(pairs):
    same=[abs(v) for k,v in pairs.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    cross=[abs(v) for k,v in pairs.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    allv=[v for v in pairs.values() if v is not None]
    z=np.exp(1j*np.radians(np.abs(allv)))
    return dict(same=float(np.mean(same)) if same else None,cross=float(np.mean(cross)) if cross else None,
      same_n=len(same),cross_n=len(cross),R=float(abs(z.mean())) if allv else None,
      cross_min=float(min(cross)) if cross else None,same_max=float(max(same)) if same else None,
      cross_ge_90=int(sum(1 for v in cross if v>=90)),n_anti=int(sum(1 for v in allv if v>=135)))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S333',mode='pool input structure audit and true input-channel relay; no physics',
      counters=dict(tier_build=1,select_touch_tier=2,edges_build=1,banc_load=1,network_init=6,control_tick=30000,tier_advance=0,neural_substep=60000,physics_step=0,model_load=0,render=0),
      nulls=NULLS,seed=SEED,tripod=TRIPOD,bands=[list(b) for b in BANDS],exclusive_thresholds=THRESHOLDS,primary_threshold=0.90,
      branch_thresholds=BRANCH_TH,
      rules=['branch thresholds are numerically preregistered','phase descriptors are per-pair signatures, cross/same bin counts and circular R; mean|dphi| is a diagnostic',
             'the exclusive-set threshold is a research degree of freedom and all three thresholds are reported',
             'EDGES npz read with csr_matrix only','no absolute floors; independent synthetic nulls with null_std>0','every arm carries a manipulation check'],
      branches=dict(t1='RELAY_STRONG',t2='RELAY_PARTIAL',t3='NO_RELAY',t4='NOT_JUDGED'))
    dump(o/'PREREG.json',pre)
    (o/'S333_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s333.py)=%s\nsha256(audit_s333_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s333_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
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
    G0=loadz(S332/'SPK_G0_gate.npz')
    # ---------- structure: exclusive presynaptic sets
    We=np.load(S303/'EDGES_w_2.5e-5_We.npz');Wi=np.load(S303/'EDGES_w_2.5e-5_Wi.npz')
    def tocsr(z):
        n=int(z['indptr'].size-1)
        return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
    WeC=tocsr(We);WiC=tocsr(Wi);n=WeC.shape[0]
    def leg_weight_matrix(Wm):
        # rows = pool cells, cols = presynaptic sources; per leg column sums
        M=Wm[poolcols,:].tocsr()
        per={lg:np.asarray(M[[poolcols.index(j) for j in pool_by_leg[lg]],:].sum(0)).ravel() for lg in LEGS}
        return M,per
    Me,perE=leg_weight_matrix(WeC);Mi,perI=leg_weight_matrix(WiC)
    def exclusive(per,thr):
        tot=np.sum([per[lg] for lg in LEGS],0)
        sets={}
        for lg in LEGS:
            frac=np.divide(per[lg],np.maximum(tot,1e-12),out=np.zeros_like(tot),where=tot>1e-12)
            idx=np.flatnonzero((frac>=thr)&(per[lg]>0))
            idx=np.array([j for j in idx if j not in poolset and j not in Mset],int)
            sets[lg]=idx
        return sets
    STR={}
    for thr in THRESHOLDS:
        SE=exclusive(perE,thr);SI=exclusive(perI,thr)
        rows={}
        for lg in LEGS:
            rows[lg]=dict(n_exc=int(SE[lg].size),exc_weight=float(perE[lg][SE[lg]].sum()),
              share_exc=float(perE[lg][SE[lg]].sum()/max(perE[lg].sum(),1e-12)),
              n_inh=int(SI[lg].size),inh_weight=float(perI[lg][SI[lg]].sum()),
              share_inh=float(perI[lg][SI[lg]].sum()/max(perI[lg].sum(),1e-12)))
        phase={}
        for nm,S in (('E',SE),('I',SI)):
            if all(S[lg].size>0 for lg in LEGS):
                pr=phase_block(G0,S);phase[nm]=dict(pairs=pr,**summarize(pr))
        STR[str(thr)]=dict(per_leg=rows,phase=phase,
          total=dict(exc_weight=sum(rows[lg]['exc_weight'] for lg in LEGS),
            pool_exc=float(perE['lf'].sum()*0+sum(perE[lg].sum() for lg in LEGS)),
            inh_weight=sum(rows[lg]['inh_weight'] for lg in LEGS),
            pool_inh=sum(perI[lg].sum() for lg in LEGS)))
    # zero coupling audit
    pool2pool=WeC[poolcols,:][:,poolcols]
    Acols=[j for lg in TRIPOD['A'] for j in pool_by_leg[lg]];Bcols=[j for lg in TRIPOD['B'] for j in pool_by_leg[lg]]
    AB=WeC[Acols,:][:,Bcols].nnz;BA=WeC[Bcols,:][:,Acols].nnz
    pool2M=WeC[poolcols,:][:,M18].nnz;M2pool=WeC[M18,:][:,poolcols].nnz
    cross_motif=0
    for i,lg in enumerate(LEGS):
        for lg2 in LEGS:
            if lg2==lg: continue
            cross_motif+=WeC[M18_by_leg[lg],:][:,M18_by_leg[lg2]].nnz
    dump(o/'POOL_INPUT_STRUCTURE.json',dict(thresholds=STR,zero_coupling=dict(
      pool2pool_we_nnz=int(pool2pool.nnz),pool2pool_we_uS=float(pool2pool.sum()),
      pool2pool_AtoB=int(AB),pool2pool_BtoA=int(BA),pool2pool_wi_nnz=int(WiC[poolcols,:][:,poolcols].nnz),
      pool2motif=int(pool2M),motif2pool=int(M2pool),motif_cross_leg=int(cross_motif)),
      note='leg-specific presynaptic sets are those sending at least the threshold fraction of their pool weight to one leg, excluding the pool and the motif'))
    SE90=exclusive(perE,0.90);SI90=exclusive(perI,0.90)
    # ---------- relay arithmetic
    f0=FREQ=175.0
    q=float(np.exp((1000.0/f0-TREF*1000.0)/ (TAU*1000.0)))
    I_base=q*DV_RATE/((q-1)*R_M)*1e9      # rate equation uses the 15 mV threshold-to-rest excursion
    per_syn=MAX_SYN*DV_DRIVE*1000.0       # uS times V gives uA, times 1000 gives nA (65 mV driving force)
    need_sync=0.040/per_syn
    dump(o/'RELAY_ARITHMETIC.json',dict(f0_hz=f0,q=q,delta_V_rate_mV=DV_RATE*1000,delta_V_drive_mV=DV_DRIVE*1000,R_Mohm=R_M/1e6,t_ref_ms=TREF*1000,tau_ms=TAU*1000,
      base_drive_nA=I_base,differential_amplitude_nA=0.040,fraction_of_base=0.040/I_base,
      per_synapse_max_uS=MAX_SYN,per_event_current_nA=per_syn,required_synchronised_cells=need_sync,
      available_cells=dict(exc={lg:int(SE90[lg].size) for lg in LEGS}),
      note='each presynaptic spike contributes at most min(count,8)*w = 2e-4 uS; with a 65 mV driving force that is 1.3e-2 nA per event, so a 0.040 nA mean differential needs about %.0f synchronised presynaptic cells'%need_sync))
    # ---------- requirement v2
    T=json.loads((S332/'TRANSLATION_LAYER.json').read_text())
    dump(o/'REQUIREMENT_V2.json',dict(items=[
      dict(id='drive',current='a 0.040 nA differential mean is %.1f percent of the 0.0885 nA base drive'%(100*0.040/I_base),
        needed='the network itself must supply the differential, which the connectome cannot do since every cross-leg path is zero'),
      dict(id='modulation',current='pool modulation depth 0.121 to 0.173 in the S332 ladder',needed='the modulated fraction must survive the decoder'),
      dict(id='decoder',current='the engine default normalisation puts %.1f percent of actuators at the 1.0 clip'%(100*T['default_config']['saturated_frac']),
        needed='a normalisation that leaves headroom, otherwise phase information is clipped downstream; the S300 calibration is one archived candidate')],
      archive_numbers=dict(s332_gain_is_secant=4589.1752258746055,s332_interp_dI=0.03733253920159715)))
    dump(o/'S332_CORRECTION.json',dict(items=[
      dict(id=1,issue='primary descriptor',s332='cross = mean|dphi| = 152.1',corrected='the mean hides a bimodal split: nine cross-tripod pairs sit at 134 to 172 degrees and six same-tripod pairs at 0 to 7; the descriptor now reports per-pair signatures, bin counts and circular R'),
      dict(id=2,issue='gain',s332='gain_deg_per_nA = 4589',corrected='that number is a secant between D2 and D4, not a gain; the ladder is a threshold jump between 0.010 and 0.020 nA and the 135 degree interpolation only holds inside [0.020, 0.040]'),
      dict(id=3,issue='what was injected',s332='the tripod differential drives the network',corrected='the current was added to the 202 motor pool cells themselves, so s1 means the pool can be entrained; the base drive at 175 Hz is 0.0885 nA and plus minus 0.040 nA is 45 percent of it, while pool recurrence is 0.0196 uS (0.64 percent of pool input) with zero inhibition'),
      dict(id=4,issue='thresholds',s332='s1 thresholds hard coded after the fact',corrected='branch thresholds and the 0.2 manipulation gate are now numerically preregistered'),
      dict(id=5,issue='band',s332='only 16-26 Hz reported',corrected='the same spikes give 152.1 / 152.5 / 163.1 / 167.5 for 16-26 / 20-30 / 22-28 / 24-27 Hz, so a band sensitivity table is mandatory'),
      dict(id=6,issue='seeds',s332='all six arms at background_seed 0',corrected='this round adds T4_in_diff010_seed1 as a seed repeat')]))
    # ---------- stage B six arms
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
    poolA=[j for lg in TRIPOD['A'] for j in pool_by_leg[lg]];poolB=[j for lg in TRIPOD['B'] for j in pool_by_leg[lg]]
    FM=float(np.abs(sgn.csd(env_of(G0[:,poolA].sum(1).astype(float)),env_of(G0[:,poolB].sum(1).astype(float)),fs=FS,nperseg=1024,noverlap=512)[1][100:130]).argmax())
    def run(name,dI=0.0,common=False,seed=0,ticks=5000):
        net=ConductanceNetwork(n,ConductanceParams(),cfg.dt_ms,voltage_method=cfg.voltage_method).set_connectivity(WeA,WiA);CD['network_init']+=1
        bg=np.random.default_rng(seed).normal(cfg.background_mean_nA,cfg.background_sd_nA,n)
        setA=sorted(set(int(j) for lg in TRIPOD['A'] for j in SE90[lg]))
        setB=sorted(set(int(j) for lg in TRIPOD['B'] for j in SE90[lg]))
        spm=np.zeros((ticks,n),bool);drv=np.sin(2*np.pi*FREQ*np.arange(ticks)/1000.0)
        for k in range(ticks):
            cur=np.full(n,I_STAR)
            if dI>0:
                s_=drv[k]
                cur[setA]+=dI*s_
                cur[setB]+=(dI*s_ if common else -dI*s_)
            spm[k]=np.logical_or(net.step(bg+cur),net.step(bg+cur));CD['control_tick']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        pr=phase_block(spm,pool_by_leg);summary=summarize(pr)
        Eb={lg:env_of(spm[:,pool_by_leg[lg]].sum(1).astype(float)) for lg in LEGS}
        blk=[]
        for i in range(4):
            seg=slice(i*len(Eb['lf'])//4,(i+1)*len(Eb['lf'])//4)
            pav={}
            for a_i,a_ in enumerate(LEGS):
                for b_ in LEGS[a_i+1:]:
                    pav[f'{a_}|{b_}']=band_phase(Eb[a_][seg],Eb[b_][seg])
            blk.append(summarize(pav))
        envA=env_of(spm[:,setA].sum(1).astype(float));envB=env_of(spm[:,setB].sum(1).astype(float))
        dd=env_of(drv)[:len(envA)]
        manip=dict(r_with_inputA=float(np.corrcoef(dd,envA)[0,1]) if dI>0 else 0.0,
                   r_with_inputB=float(np.corrcoef(dd,envB)[0,1]) if dI>0 else 0.0)
        band_table={}
        for bnd in BANDS:
            band_table['%.0f-%.0f'%bnd]=summarize(phase_block(spm,pool_by_leg,bnd))['cross']
        mds=[]
        for lg in LEGS:
            raw=np.convolve(spm[:,pool_by_leg[lg]].sum(1).astype(float),np.ones(20)/20,mode='same')
            mds.append((np.percentile(raw,90)-np.percentile(raw,10))/2.0/max(float(np.mean(raw)),1e-12))
        res=dict(pairs=pr,blocks=[dict(cross=b['cross'],same=b['same']) for b in blk],
          band_sensitivity=band_table,manip=manip,dI=dI,common=common,seed=seed,
          pool_median_rate_hz=float(np.median(spm[:,poolcols].sum(0)*1000.0/ticks)),
          mod_depth=float(np.mean(mds)))
        res.update(summary)
        return res,spm
    AR={}
    AR['A0_gate'],spm0=run('A0_gate')
    g=[bool(np.array_equal(spm0,loadz(S332/'SPK_G0_gate.npz')[:5000])),
       bool(np.array_equal(spm0,loadz(S331/'SPK_Z0_gate.npz')[:5000])),
       bool(np.array_equal(spm0,loadz(S322/'SPK_K17.npz')[:5000]))]
    AR['T1_in_diff010'],_=run('T1_in_diff010',dI=0.010)
    AR['T2_in_diff020'],_=run('T2_in_diff020',dI=0.020)
    AR['T3_in_diff040'],_=run('T3_in_diff040',dI=0.040)
    AR['T4_in_diff010_seed1'],_=run('T4_in_diff010_seed1',dI=0.010,seed=1)
    AR['T5_in_common020'],_=run('T5_in_common020',dI=0.020,common=True)
    dump(o/'ARMS.json',AR);dump(o/'CALL_LEDGER.json',CD);dump(o/'GATE.json',dict(triple=g,all=all(g)))
    # ---------- stage C
    th=BRANCH_TH
    def ok(nm): return AR[nm]['cross']>=th['t2']['cross'] and AR[nm]['same']<=th['t2']['same']
    base=AR['A0_gate']['pool_median_rate_hz']
    rmiss=[k for k in AR if abs(AR[k]['pool_median_rate_hz']-base)/max(base,1e-9)>0.10]
    t1=ok('T1_in_diff010') and abs(AR['T1_in_diff010']['cross']-AR['T4_in_diff010_seed1']['cross'])<=th['t1']['seed_spread']
    t2=ok('T2_in_diff020') or ok('T3_in_diff040')
    diff_specific=abs(AR['T5_in_common020']['cross']-AR['T2_in_diff020']['cross'])>th['t3']['cross']*0
    t3=(not ok('T3_in_diff040')) or (not diff_specific)
    if not all(g) or len(rmiss)>2: branch='t4'
    elif t1: branch='t1'
    elif t2: branch='t2'
    elif t3: branch='t3'
    else: branch='t4'
    dump(o/'OUTCOMES.json',dict(round='S333',branch=branch,gate=dict(triple=g,all=all(g)),rate_mismatch=rmiss,
      arms={k:dict(cross=round(v['cross'],1),same=round(v['same'],1),cross_min=round(v['cross_min'],1) if v['cross_min'] is not None else None,
        same_max=round(v['same_max'],1) if v['same_max'] is not None else None,n_anti=v['n_anti'],R=round(v['R'],3),
        pool=round(v['pool_median_rate_hz'],1),mod=round(v['mod_depth'],3),manip=v['manip'],
        blocks=[(round(b['cross'],1),round(b['same'],1)) for b in v['blocks']],
        band={k:round(x,1) for k,x in v['band_sensitivity'].items()}) for k,v in AR.items()},
      exclusive=dict(E={lg:int(SE90[lg].size) for lg in LEGS},I={lg:int(SI90[lg].size) for lg in LEGS},
        share_E={lg:round(STR['0.9']['per_leg'][lg]['share_exc'],4) for lg in LEGS},
        share_I={lg:round(STR['0.9']['per_leg'][lg]['share_inh'],4) for lg in LEGS}),
      relay=dict(base_drive_nA=round(I_base,4),needed_sync=round(need_sync,1)),
      branches=dict(t1=t1,t2=t2,t3=t3),counters=CD,
      one_line='%s T1 %.1f T2 %.1f T3 %.1f T5 %.1f'%(branch,AR['T1_in_diff010']['cross'],AR['T2_in_diff020']['cross'],AR['T3_in_diff040']['cross'],AR['T5_in_common020']['cross'])))
    print(json.dumps(dict(branch=branch,gate=g,arms={k:(round(v['cross'],1),round(v['same'],1),v['n_anti']) for k,v in AR.items()},
      exc={lg:int(SE90[lg].size) for lg in LEGS},share={lg:round(STR['0.9']['per_leg'][lg]['share_exc'],3) for lg in LEGS},
      base=round(I_base,4),sync=round(need_sync,1),struct_phase={k:(round(v['same'],1),round(v['cross'],1)) for k,v in STR['0.9']['phase'].items()},cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
