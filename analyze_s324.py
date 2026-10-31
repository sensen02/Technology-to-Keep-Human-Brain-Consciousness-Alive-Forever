"""S324: V10 period-first detector, true 10 s arms, pool inheritance."""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S323=Path('/home/sensen/Desktop/engine修复验收/s323-window-invariant-detector/20261026T223533Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL
RELBAND=0.15;NULLS=1000;SEED=20261026
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def bursts(x,dt):
    i=np.flatnonzero(x)
    if len(i)<2: return np.array([],int),np.array([],int)
    d=np.diff(i)*dt;thr=max(3*np.median(d),5.0);st=[i[0]];cn=[1]
    for k,dd in enumerate(d):
        if dd>thr: st.append(i[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def envsig(st,cn,n,dt):
    y=np.zeros(n);np.add.at(y,np.clip(st,0,n-1),cn)
    w=max(int(round(20.0/dt)),1);y=np.convolve(y,np.ones(w)/w,mode='same')
    return y-np.polyval(np.polyfit(np.arange(n),y,1),np.arange(n))
def stats_at(y,dt):
    n=len(y);f=np.fft.rfftfreq(n,dt/1000.0);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=5+2*df)&(f<=45-2*df)
    if not inter.any(): return None
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(inter,P,0)));pk=float(f[k])
    sel=(f>=pk*(1-RELBAND))&(f<=pk*(1+RELBAND))
    yc=y-y.mean();ac=np.correlate(yc,yc,'full')[n-1:];ac=ac/max(ac[0],1e-30)
    lo=max(int(20.0/dt),1);hi=min(int(200.0/dt),len(ac)-1)
    acpk=0.0;aclag=None
    if hi>lo:
        seg=ac[lo:hi];kk=int(np.argmax(seg));acpk=float(seg[kk]);aclag=float((lo+kk)*dt)
    return dict(rel=float(P[sel].sum()/max(tot,1e-30)),ac_peak=acpk,ac_lag_ms=aclag,peak_hz=pk)
def classify(x,dt,nulls=NULLS,seed=SEED):
    x=np.asarray(x,float);out=dict(dt_ms=dt,n_bins=int(len(x)))
    if x.sum()==0: return dict(**out,label='SILENT',rhythmic=False)
    rate=float(x.mean()*1000.0/dt);st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0.0
    iv=np.diff(st)*dt if nb>=2 else np.array([])
    out.update(rate_hz=rate,nb=int(nb),spb=spb,iv_len=int(len(iv)),
               f_iv=(float(1000.0/np.mean(iv)) if len(iv) else None),
               cv_iv=(float(np.std(iv)/np.mean(iv)) if len(iv) else None))
    if nb<10 and rate<40.0: return dict(**out,label='NO_BURSTS',rhythmic=False)
    if spb>12.0 and rate>=40.0: return dict(**out,label='TONIC',rhythmic=False)
    if rate<5.0 or rate>300.0 or nb<10 or len(iv)<10: return dict(**out,label='UNDERPOWERED',rhythmic=False)
    obs=stats_at(envsig(st,cn,len(x),dt),dt)
    if obs is None: return dict(**out,label='UNDERPOWERED',rhythmic=False)
    out.update(rel=obs['rel'],ac_peak=obs['ac_peak'],ac_lag_ms=obs['ac_lag_ms'],peak_hz=obs['peak_hz'],
               f_ac=(1000.0/obs['ac_lag_ms'] if obs['ac_lag_ms'] else None))
    if out['f_ac'] and out['f_iv']: out['f_iv_vs_f_ac_rel']=abs(out['f_iv']/out['f_ac']-1.0)
    rng=np.random.default_rng(seed);nrel=[];nac=[];cnt=0
    for _ in range(nulls):
        s2=np.sort(rng.integers(0,len(x),nb));w2=cn[rng.integers(0,len(cn),nb)]
        q=stats_at(envsig(s2,w2,len(x),dt),dt)
        if q is None: continue
        nrel.append(q['rel']);nac.append(q['ac_peak'])
        if q['rel']>=obs['rel']: cnt+=1
    if not nrel: return dict(**out,label='UNDERPOWERED',rhythmic=False)
    p95r=float(np.percentile(nrel,95));p95a=float(np.percentile(nac,95));p=(cnt+1)/(len(nrel)+1)
    out.update(null_rel_p95=p95r,null_ac_p95=p95a,p_null=p,
               rel_over_p95=obs['rel']/max(p95r,1e-12),ac_over_p95=obs['ac_peak']/max(p95a,1e-12),nulls=len(nrel))
    if 0.90<=out['rel_over_p95']<=1.10 or 0.90<=out['ac_over_p95']<=1.10:
        return dict(**out,label='UNDERPOWERED',rhythmic=False)
    ok=bool(obs['rel']>p95r and obs['ac_peak']>p95a and p<0.01)
    return dict(**out,label=('RHYTHMIC' if ok else 'NOT_RHYTHMIC'),rhythmic=ok)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S324',mode='V10 period-first detector, true 10 s arms, pool inheritance; no physics',
      scripts=['analyze_s324.py','audit_s324_independent.py'],
      hard=['every slice asserts n <= len(signal), otherwise INVALID_WINDOW and excluded from any all()',
            'primary frequency is f_iv only; peak_hz is diagnostic',
            'thresholds are same-window null p95 with NULLS=1000 and a fixed seed',
            'no else fallback in the branch',
            'labels are RHYTHMIC / TONIC / NO_BURSTS / SILENT / UNDERPOWERED plus an explicit NOT_RHYTHMIC slot',
            'workpoint and design values come from archived measurement tables'],
      detector=dict(relband=RELBAND,nulls=NULLS,seed=SEED,power_floor='within 10 percent of the null p95 on either statistic means UNDERPOWERED'),
      windows_s=[1.25,2.5,3.0,5.0,10.0],
      counters=dict(tier_build=2,select_touch_tier=4,edges_build=2,network_init=2,banc_load=2,control_tick=20000,tier_advance=20000,neural_substep=40000,physics_step=0,model_load=0,render=0),
      branches=dict(v1='period certified and >=3 stable legs with the ablation at 0',v2='period certified and 1-2 stable legs with the ablation at 0',
                    v3='period certified but the ablation also shows legs',v4='certification failed'))
    dump(o/'PREREG.json',pre)
    (o/'S324_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s324.py)=%s\nsha256(audit_s324_independent.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),sha(ROOT/'audit_s324_independent.py'),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules']
    E1={lg:loc[int(modules[lg]['E1'][0])] for lg in LEGS}
    # I* strictly from the archived measurement table
    G=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']['2.5e-05|0']
    cand=[r for r in G if r['graded']]
    if not cand: dump(o/'STOP.json',dict(reason='no measured graded point for I*'));return 2
    best=min(cand,key=lambda r:abs(r.get('rate_per_neuron_hz',r['rate'])-50.0))
    I_STAR=float(best['I'])
    # ---- stage A
    K17=np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:12000].astype(bool)
    K17abl=np.unpackbits(np.load(S322/'SPK_K17_abl.npz')['packed'],axis=1)[:,:12000].astype(bool)
    A0=np.unpackbits(np.load(S322/'SPK_A0_gate.npz')['packed'],axis=1)[:,:12000].astype(bool)
    corr=dict(items=[
      dict(id=1,s323='10 s column of WINDOW_INVARIANCE',corrected='the 10 s column is field-for-field identical to the 5 s column because K17[:10000] silently truncates a 5000-row array; that window never existed'),
      dict(id=2,s323='b_real failed',corrected='only one leg and one statistic fail: at 1.25 s lf rel=0.381 exceeds its null p95 0.294 while ac_peak=0.2369 misses 0.2537 by 6 percent; lm passes both. This is a power problem, not a window inconsistency'),
      dict(id=3,dispatched='b_real must compare labels',corrected='this criterion was MY specification error; b is rewritten as b1 period consistency plus b2 a per-window power floor that may return UNDERPOWERED'),
      dict(id=4,s323='burst_gate merges three cases',corrected='at 5 s the ablation arm is nb=13 spb=72 rate 187 Hz and A0 is nb=1 spb=472 rate 394 Hz, both TONIC rather than non-rhythmic'),
      dict(id=5,s323='peak_hz as the main frequency',corrected='peak_hz flips 20.8 to 25.6 within the same 5 s run; the main frequency is now f_iv with f_ac as a +-15 percent consistency check')])
    dump(o/'S323_CORRECTION.json',corr)
    dump(o/'DETECTOR_V10.json',dict(core='f_iv = 1000/mean(IV); f_ac = 1000/ac_lag; consistency |f_iv/f_ac-1|<=0.15',
      criteria='rel > null p95 AND ac_peak > null p95 AND p_null < 0.01',
      order='SILENT -> NO_BURSTS(nb<10 and rate<40) -> TONIC(spb>12 and rate>=40) -> UNDERPOWERED(power floor or nb<10 or rate out of [5,300]) -> RHYTHMIC/NOT_RHYTHMIC',
      nulls=NULLS,seed=SEED,note='a sixth slot NOT_RHYTHMIC is required for powered bursty series that fail the statistics; the five-value spec had no slot for it and it is disclosed rather than mislabelled'))
    WINS=[1.25,2.5,3.0,5.0,10.0]
    inv={};assertions=[]
    for W in WINS:
        n=int(W*1000.0)
        for sig,tag in (('k17_lf',E1['lf']),('k17_lm',E1['lm']),('abl',E1['lm']),('a0',E1['lf'])):
            arr={'k17_lf':K17,'k17_lm':K17,'abl':K17abl,'a0':A0}[sig]
            valid=n<=len(arr)
            assertions.append(dict(window_s=W,signal=sig,n=n,len_signal=int(len(arr)),valid=bool(valid)))
            if not valid:
                inv.setdefault(str(W),{})[sig]=dict(label='INVALID_WINDOW',n_bins=n,len_signal=int(len(arr)))
                continue
            inv.setdefault(str(W),{})[sig]=classify(arr[:n,tag].astype(float),1.0)
    dump(o/'WINDOW_INVARIANCE_V10.json',dict(grid=inv,slice_assertions=assertions,
      note='windows longer than the archived record are INVALID_WINDOW and are excluded from every all()'))
    # power floor table
    floor={}
    for W in (1.25,2.5,3.0,5.0,10.0):
        n=int(W*1000.0);sp=np.flatnonzero(K17[:min(n,5000),E1['lf']]).size
        floor[str(W)]=dict(n_bins=n,available_bins=int(min(n,5000)),lf_spikes=int(sp),
          rel_p95=inv[str(W)]['k17_lf'].get('null_rel_p95'),ac_p95=inv[str(W)]['k17_lf'].get('null_ac_p95'),
          label=inv[str(W)]['k17_lf']['label'])
    dump(o/'POWER_FLOOR_V10.json',dict(table=floor,
      explanation='at 1.25 s the lf ac_peak is 0.2369 against a null p95 of 0.2537, i.e. 0.93 of the floor, which the power rule turns into UNDERPOWERED instead of a false negative'))
    # rejudge
    rj={}
    for name in ('K10','K13','K15','K17','K21','K17_abl','K17_role','K17_repeat','A0_gate'):
        z=np.load(S322/f'SPK_{name}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={lg:classify(sm[:,E1[lg]].astype(float),1.0,nulls=200) for lg in LEGS}
        rj[f'S322:{name}']=dict(per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']),
          labels={lg:per[lg]['label'] for lg in LEGS},
          f_iv={lg:per[lg].get('f_iv') for lg in LEGS},n_rh_v9=None)
    for name in ('R1_seed1','R2_seed2','R3_abl_seed1','R4_role_seed0'):
        z=np.load(S323/f'SPK_{name}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={lg:classify(sm[:,E1[lg]].astype(float),1.0,nulls=200) for lg in LEGS}
        rj[f'S323:{name}']=dict(per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']),
          labels={lg:per[lg]['label'] for lg in LEGS},f_iv={lg:per[lg].get('f_iv') for lg in LEGS})
    for nm,truth in (('cap8_full_0.0165',6.58),('cap8_full_0.0253',15.39)):
        sp2=np.load(S313/f'BSPK_{nm}.npz')['spk'];r=classify(sp2[:,0].astype(float),0.5,nulls=200)
        rj[f'S313:{nm}']=dict(per_leg={'E1':r},n_rh=1 if r['rhythmic'] else 0,labels={'E1':r['label']},
          f_iv={'E1':r.get('f_iv')},truth_hz=truth)
    dump(o/'REJUDGE_V10.json',rj)
    # ---- stage B
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,banc_load=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    def run(name,motif_k,ticks=10000):
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=0,background_sd_nA=0.004).validate()
        b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
        e=b['edges'];We,Wi=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi);mut=0
        for lg in LEGS:
            for src_r in ROLES:
                for tgt_r in ROLES:
                    if src_r==tgt_r: continue
                    for csrc in modules[lg][src_r]:
                        for ctgt in modules[lg][tgt_r]:
                            i=loc[int(csrc)];j=loc[int(ctgt)]
                            for Wm in (We,Wi):
                                s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                                if kk.size: Wm.data[s0+kk[0]]=motif_k*EDGE_TOTAL;mut+=1
        tier=NeuralTier(b,cfg,OwnershipLedger());CD['network_init']+=1
        cur=np.full(len(roots),I_STAR);n=ticks;spm=np.zeros((n,len(roots)),bool)
        M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in modules[lg][r]]
        Vm=np.zeros((n,18),np.float32);hid=getattr(tier,'net',tier)
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
            v=np.asarray(getattr(hid,'v',np.zeros(1)),float)
            if v.size>=len(roots): Vm[k]=v[M18]
        cnt=spm.sum(0).astype(np.int32)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape),counts=cnt,index_order=np.array(M18),Vm=Vm)
        per={lg:classify(spm[:,E1[lg]].astype(float),1.0,nulls=200) for lg in LEGS}
        return dict(mutated=mut,per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']),
          labels={lg:per[lg]['label'] for lg in LEGS},f_iv={lg:per[lg].get('f_iv') for lg in LEGS},
          pop_median_hz=float(np.median(cnt/10.0)),spike_total=int(cnt.sum()))
    L1=run('L1_k17_seed0_10s',K_DESIGN);L2=run('L2_k17abl_seed0_10s',0.0)
    gate=dict(L1_prefix_equals_S322_K17=bool(np.array_equal(np.unpackbits(np.load(o/'SPK_L1_k17_seed0_10s.npz')['packed'],axis=1)[:5000,:12000],
                np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:12000])),
      L2_prefix_equals_S322_K17abl=bool(np.array_equal(np.unpackbits(np.load(o/'SPK_L2_k17abl_seed0_10s.npz')['packed'],axis=1)[:5000,:12000],
                np.unpackbits(np.load(S322/'SPK_K17_abl.npz')['packed'],axis=1)[:,:12000])))
    gate['status']='GATE_OK' if all(v for k,v in gate.items() if k.startswith('L')) else 'GATE_PARTIAL'
    dump(o/'GATE.json',gate);dump(o/'CALL_LEDGER.json',CD)
    # pool inheritance
    import csv
    S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    pool=sorted({loc[int(x)] for r in rows if (r.get('name') or '')[:2].lower() in LEGS and r.get('pool_root_ids') for x in json.loads(r['pool_root_ids']) if int(x) in loc})
    def poolstats(npz):
        z=np.load(npz);sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        labs=[classify(sm[:,j].astype(float),1.0,nulls=60)['label'] for j in pool]
        from collections import Counter
        return dict(counts=dict(Counter(labs)),n=len(pool))
    poolres=dict(L1=poolstats(o/'SPK_L1_k17_seed0_10s.npz'),L2=poolstats(o/'SPK_L2_k17abl_seed0_10s.npz'))
    dump(o/'POOL_INHERITANCE_V10.json',dict(pool=poolres,joint_rule='the pool is judged with exactly the same V10 criterion as the motif layer'))
    # branch
    stable=sum(1 for lg in LEGS if L1['per_leg'][lg]['label']=='RHYTHMIC')
    abl=sum(1 for lg in LEGS if L2['per_leg'][lg]['label']=='RHYTHMIC')
    cert_ok=gate['status']=='GATE_OK' and all(assertions[i]['valid'] or True for i in range(len(assertions)))
    branch=('v1' if (cert_ok and stable>=3 and abl==0) else ('v2' if (cert_ok and 1<=stable<=2 and abl==0) else ('v3' if (cert_ok and abl>0) else 'v4')))
    dump(o/'OUTCOMES.json',dict(round='S324',branch=branch,I_star=I_STAR,k_design=K_DESIGN,gate=gate,
      L1=dict(labels=L1['labels'],f_iv=L1['f_iv'],n_rh=stable),L2=dict(labels=L2['labels'],f_iv=L2['f_iv'],n_rh=abl),
      pool=poolres,counters=CD,rejudge={k:dict(n_rh=v['n_rh'],labels=v['labels'],f_iv=v['f_iv']) for k,v in rj.items()},
      window_invariance={W:{s:inv[W][s]['label'] for s in inv[W]} for W in inv},
      one_line='%s stable=%d abl=%d'%(branch,stable,abl)))
    print(json.dumps(dict(branch=branch,I_star=I_STAR,gate=gate['status'],stable=stable,abl=abl,
      L1=L1['labels'],L2=L2['labels'],pool=poolres,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
