"""S319: motor-interface audit, saturation audit, graded-family ten arms."""
from __future__ import annotations
import argparse, json, hashlib, csv, datetime
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S316=Path('/home/sensen/Desktop/engine修复验收/s316-percell-mapping-repair/20261026T213550Z')
S318=Path('/home/sensen/Desktop/engine修复验收/s318-detector-certification/20261026T214801Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
NULLS=200;SEED=20261026;CONC_MIN=0.15;P_MAX=0.01
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
def envp(st,cn,n,dt):
    y=np.zeros(n);np.add.at(y,np.clip(st,0,n-1),cn)
    w=max(int(round(20.0/dt)),1);y=np.convolve(y,np.ones(w)/w,mode='same')
    return y-np.polyval(np.polyfit(np.arange(n),y,1),np.arange(n))
def spec_env(y,dt,lo=5.0,hi=45.0):
    n=len(y);f=np.fft.rfftfreq(n,dt/1000.0);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=lo+2*df)&(f<=hi-2*df)
    if not inter.any(): return None
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(inter,P,0)));pk=float(f[k])
    sel=(f>=pk-2*df)&(f<=pk+2*df)
    return dict(peak_hz=pk,conc=float(P[sel].sum()/max(tot,1e-30)),band_20_40=float(P[(f>=20)&(f<=40)].sum()/max(tot,1e-30)))
def v8(x,dt,rng,nulls=NULLS):
    x=np.asarray(x,float)
    if x.sum()==0: return dict(rhythmic=False,reason='silent')
    rate=x.mean()*1000.0/dt;st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0.0
    out=dict(rate_hz=rate,nb=int(nb),spb=spb)
    if nb<10 or spb>12 or rate<5: return dict(**out,rhythmic=False,reason='burst_gate')
    o=spec_env(envp(st,cn,len(x),dt),dt)
    if o is None: return dict(**out,rhythmic=False,reason='no_interior')
    cv=float(np.std(np.diff(st)*dt)/max(np.mean(np.diff(st)*dt),1e-9))
    c=0
    for _ in range(nulls):
        s2=np.sort(rng.integers(0,len(x),nb));w2=cn[rng.integers(0,len(cn),nb)]
        q=spec_env(envp(s2,w2,len(x),dt),dt)
        if q and q['conc']>=o['conc']: c+=1
    p=(c+1)/(nulls+1);ratio=o['peak_hz']/max(rate,1e-9)
    selfd=bool(spb<=1.5 and 0.8<=ratio<=1.2)
    return dict(**out,rhythmic=bool(p<P_MAX and o['conc']>=CONC_MIN and not selfd),peak_hz=o['peak_hz'],
                conc=o['conc'],p_null=p,cv=cv,band_20_40=o['band_20_40'],self_discharge=selfd)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S319',mode='motor-interface audit and graded-family arms; no physics, no rendering',
      counters=dict(normal=dict(tier_build=10,select_touch_tier=20,edges_build=10,network_init=10,control_tick=50000,tier_advance=50000,neural_substep=100000,physics_step=0,model_load=0,render=0)),
      stage_a=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,connectome_load=0,render=0),
      arms=['G1_gate_w5e-4','W1_w1p5e-5','W2_w2p5e-5','W3_w5e-5','W4_w1p5e-5_zero','W5_w2p5e-5_zero','W6_w5e-5_zero','D1_pool_ext_scaled_003','D2_pool_ext_scaled_010','D3_scaled003_motif_zero'],
      predictions={'w1p5e-5':'equivalent drive ~0.0156 nA, below the 0.0165 nA band floor so this point may not burst','w5p5e-5':'equivalent drive ~0.052 nA at the band upper edge'},
      branches=dict(m1='a w exists with motif-level rhythm and pool grading and pool-level rhythm and its ablation is 0',
        m2='the motif-rhythmic w and the pool-graded w are disjoint',m3='D1/D2 show pool rhythm while D3 does not',m4='gate failure or none holds'),
      freeze='S319_PREREG_FROZEN must carry sha256(PREREG.json), sha256(analyze_s319.py) and UTC, written before any read; no exemptions')
    dump(o/'PREREG.json',pre)
    (o/'S319_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s319.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(__file__),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    rng=np.random.default_rng(SEED)
    import sys;sys.path.insert(0,str(ROOT))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18g=sorted({int(c) for lg in LEGS for r in ROLES for c in mod[lg][r]});M18=[loc[g] for g in M18g]
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    poolper={}
    for r in rows:
        nm=r.get('name') or ''
        lg=nm[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            ids=[loc[int(x)] for x in json.loads(r['pool_root_ids']) if int(x) in loc]
            if ids: poolper.setdefault(lg,[]).extend(ids)
    poolall=sorted({c for v in poolper.values() for c in v})
    # ---------- stage A
    sp0=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    poolrate0=sp0[:,poolall].sum(0)/5.0
    We=np.load(S303/'EDGES_w_5e-4_We.npz');Wi=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    Mset=set(M18)
    def col_sums(W_,cells,inside):
        tot=0.0;n=0
        for j in cells:
            s,e=W_['indptr'][j],W_['indptr'][j+1];src=W_['indices'][s:e];dat=np.asarray(W_['data'][s:e],float)
            m=np.array([int(x) in Mset for x in src])
            sel=(~m) if not inside else m
            if sel.any(): tot+=float(dat[sel].sum());n+=int(sel.sum())
        return tot,n
    motif_to_pool_e=0.0
    for j in poolall:
        s,e=We['indptr'][j],We['indptr'][j+1];src=We['indices'][s:e];dat=np.asarray(We['data'][s:e],float)
        m=np.array([int(x) in Mset for x in src])
        if m.any(): motif_to_pool_e+=float(dat[m].sum())
    motif_to_pool_i=0.0
    for j in poolall:
        s,e=Wi['indptr'][j],Wi['indptr'][j+1];src=Wi['indices'][s:e];dat=np.asarray(Wi['data'][s:e],float)
        m=np.array([int(x) in Mset for x in src])
        if m.any(): motif_to_pool_i+=float(dat[m].sum())
    pool_in_e=float(np.concatenate([np.asarray(We['data'][We['indptr'][j]:We['indptr'][j+1]],float) for j in poolall]).sum())
    pool_in_i=float(np.concatenate([np.asarray(Wi['data'][Wi['indptr'][j]:Wi['indptr'][j+1]],float) for j in poolall]).sum())
    with_motif=0
    for j in poolall:
        s,e=We['indptr'][j],We['indptr'][j+1];src=We['indices'][s:e]
        if any(int(x) in Mset for x in src): with_motif+=1
    armfiles={'G0_gate':S316/'SPK_G0_gate.npz','P3_source_faithful':S316/'SPK_P3_source_faithful.npz','P5_role_coupled':S316/'SPK_P5_role_coupled.npz','P7_ablation_zero_motif':S316/'SPK_P7_ablation_zero_motif.npz'}
    poolnb={}
    for nm,f in armfiles.items():
        z=np.load(f);sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        nb=[len(bursts(sm[:,j].astype(float),1.0)[0]) for j in poolall]
        poolnb[nm]=dict(n_with_bursts=int(sum(1 for v in nb if v>=10)),median_rate=float(np.median(sm[:,poolall].sum(0)/5.0)))
    dump(o/'MOTOR_INTERFACE.json',dict(n_pool=len(poolall),per_leg={lg:len(v) for lg,v in poolper.items()},
      motif_to_pool_exc_uS=motif_to_pool_e,pool_total_exc_uS=pool_in_e,share_exc=motif_to_pool_e/max(pool_in_e,1e-12),
      motif_to_pool_inh_uS=motif_to_pool_i,pool_total_inh_uS=pool_in_i,share_inh=motif_to_pool_i/max(pool_in_i,1e-12),
      pool_cells_with_any_motif_input=with_motif,
      pool_rate_by_arm=poolnb,
      unresolved_channels=16,unresolved_note='the 16 unresolved channels are listed, not redistributed'))
    rates=sp0.sum(0)/5.0
    sat=dict(silent=int((rates<1).sum()),mid=int(((rates>=1)&(rates<=300)).sum()),top=int((rates>300).sum()),
      at_refractory=int((rates>=390).sum()),duty=float(sp0.mean()),total=len(rates))
    try: s303m=json.loads((S303/'ARM_METRICS.json').read_text())
    except Exception: s303m={}
    dump(o/'SATURATION_AUDIT.json',dict(distribution=sat,s303_arms=s303m,
      statement='S307-S318 all ran in the w=5e-4 family that S303 already registered as non-graded, so none of those results may be projected onto the motor output or onto physics'))
    scale=[]
    for w in (1.5e-5,2.5e-5,5e-5,5e-4):
        e1=0.0
        for lg in LEGS:
            src=set(M18g)&{int(c) for c in mod[lg]['E1']}
            for tgt in ('E2','I1'):
                for cs in mod[lg][tgt]:
                    for c in mod[lg]['E1']:
                        pass
        # equivalent drive into E1 = sum of motif edges * w * 65 mV, over sources inside M18
        acc=0.0
        for j in [loc[int(c)] for lg in LEGS for c in mod[lg]['E1']]:
            s,e=We['indptr'][j],We['indptr'][j+1];src=We['indices'][s:e];dat=np.asarray(We['data'][s:e],float)
            m=np.array([int(x) in Mset for x in src])
            if m.any(): acc+=float(dat[m].sum())
        scale.append(dict(w=w,motif_edge_uS=float(8*w),drive_into_E1_uS=acc,syn_drive_uS=acc,equiv_drive_nA=float(acc*65.0*w/5e-4)))
    dump(o/'MOTIF_SCALE_TABLE.json',dict(table=scale,note='equivalent drive scales linearly with w; the prediction column is checked after stage B'))
    # ---------- stage B
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    Cnt=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    Cur={'w':5e-4}
    def build():
        b=build_neural_tier(TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=Cur['w']).validate())
        Cnt['tier_build']+=1;Cnt['select_touch_tier']+=2;Cnt['edges_build']+=1;return b
    def edge(obj):
        e=obj['edges'];return (e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    refspk=sp0
    def run_arm(name,w,override,zero_motif,pool_scale_target):
        Cur['w']=w
        b=build();We_,Wi_=edge(b);audit=dict(requested=0,applied=0)
        if override and zero_motif:
            pairs=[('E1','E2'),('E2','E1'),('E2','I1'),('E1','I1'),('I1','E1'),('I1','E2')]
            for (src_r,tgt_r) in pairs:
                for lg in LEGS:
                    for csrc in mod[lg][src_r]:
                        for ctgt in mod[lg][tgt_r]:
                            i=loc[int(csrc)];j=loc[int(ctgt)]
                            for Wm in (We_,Wi_):
                                s0,e0=Wm.indptr[j],Wm.indptr[j+1]
                                k=np.flatnonzero(Wm.indices[s0:e0]==i)
                                if k.size:
                                    audit['requested']+=1
                                    Wm.data[s0+k[0]]=0.0;audit['applied']+=1
        if pool_scale_target is not None:
            # scale pool external in-edges per cell: s_j = target/(65*G_j), G_j = non-motif incoming weight sum
            sd={}
            for j in poolall:
                g=0.0
                for Wm in (We_,Wi_):
                    s,e=Wm.indptr[j],Wm.indptr[j+1];src=Wm.indices[s:e];dat=Wm.data[s:e]
                    m=np.array([int(x) in Mset for x in src])
                    if (~m).any(): g+=float(np.asarray(dat,float)[~m].sum())
                sd[int(roots[j])]=(g,(pool_scale_target/(65.0*g)) if g>0 else None)
            for j in poolall:
                gi=int(roots[j]);g,sj=sd[gi]
                if sj is None: continue
                for Wm in (We_,Wi_):
                    s,e=Wm.indptr[j],Wm.indptr[j+1];src=Wm.indices[s:e]
                    for t in range(s,e):
                        if int(Wm.indices[t]) not in Mset:
                            Wm.data[t]*=sj;audit['applied']+=1
            np.savez_compressed(o/'POOL_SCALE_MAP.npz',keys=np.array(sorted(sd)),G=np.array([sd[k][0] for k in sorted(sd)]),
              s=np.array([-1 if sd[k][1] is None else sd[k][1] for k in sorted(sd)]))
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());Cnt['network_init']+=1
        n=5000;spm=np.zeros((n,len(roots)),bool)
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,np.zeros(len(roots))),dtype=bool).any(axis=0)
            Cnt['control_tick']+=1;Cnt['tier_advance']+=1;Cnt['neural_substep']+=2
        r=spm[:,poolall].sum(0)/5.0
        pool_metrics=dict(median=float(np.median(r)),p10=float(np.percentile(r,10)),p90=float(np.percentile(r,90)),
          f380=float((r>=380).mean()),n_silent=int((r<1).sum()),n=len(r))
        legv8={}
        for lg in LEGS:
            cols_=[loc[int(c)] for c in mod[lg]['E1']]
            e1=spm[:,cols_].any(axis=1).astype(float)
            v=v8(e1,1.0,rng)
            i1=spm[:,[loc[int(c)] for c in mod[lg]['I1']]].any(axis=1).astype(float)
            vi=v8(i1,1.0,rng)
            ok=bool(v['rhythmic'] and vi['rhythmic'] and abs((v.get('peak_hz') or 0)-(vi.get('peak_hz') or 0))/max(v.get('peak_hz') or 1e-9,1e-9)<=0.2)
            # pool-level: aggregate this leg's pool neurons
            agg=spm[:,poolper.get(lg,[])].mean(axis=1)
            pv=v8(agg,1.0,rng)
            legv8[lg]=dict(motif_rhythmic=ok,motif_peak=v.get('peak_hz'),motif_nb=v.get('nb'),
              pool_rhythmic=bool(pv['rhythmic']),pool_peak=pv.get('peak_hz'),pool_conc=pv.get('conc'),pool_nb=pv.get('nb'))
        E={lg:np.convolve(spm[:,[loc[int(c)] for c in mod[lg]['E1']]].any(axis=1).astype(float),np.ones(20)/20,mode='same') for lg in LEGS}
        pairs={}
        for i,a in enumerate(LEGS):
            for b in LEGS[i+1:]:
                obs=float(np.corrcoef(E[a],E[b])[0,1])
                null=[float(np.corrcoef(np.roll(E[a],k*50),E[b])[0,1]) for k in (1,2,3,5,7,11,13,17,19,23)]
                pairs[f'{a}|{b}']=dict(r=obs,null_max=float(np.max(np.abs(null))),significant=bool(abs(obs)>np.max(np.abs(null))))
        com=float(np.mean([abs(v['r']) for v in pairs.values()]))
        return dict(pool=pool_metrics,legs=legv8,n_motif_legs=sum(1 for lg in LEGS if legv8[lg]['motif_rhythmic']),
          n_pool_legs=sum(1 for lg in LEGS if legv8[lg]['pool_rhythmic']),common_mode=com,cross=pairs,
          audit=audit,share_after=float(motif_to_pool_e/ max(pool_in_e,1e-12)),gate_bitwise=bool(np.array_equal(spm,refspk)) if name.startswith('G1') else None)
    res={}
    plan=[('G1_gate_w5e-4',5e-4,False,False,None),
          ('W1_w1p5e-5',1.5e-5,True,False,None),('W2_w2p5e-5',2.5e-5,True,False,None),('W3_w5e-5',5e-5,True,False,None),
          ('W4_w1p5e-5_zero',1.5e-5,True,True,None),('W5_w2p5e-5_zero',2.5e-5,True,True,None),('W6_w5e-5_zero',5e-5,True,True,None),
          ('D1_pool_ext_scaled_003',5e-4,False,False,0.03),('D2_pool_ext_scaled_010',5e-4,False,False,0.10),
          ('D3_scaled003_motif_zero',5e-4,False,True,0.03)]
    for name,w,ov,zm,tg in plan:
        res[name]=run_arm(name,w,ov,zm,tg)
        print('arm done',name,flush=True)
    dump(o/'ARMS.json',res);dump(o/'CALL_LEDGER.json',Cnt)
    # ---------- branch
    motif_ok=[n for n,r in res.items() if r['n_motif_legs']>=2]
    graded=[n for n,r in res.items() if r['pool']['median']<300 and r['pool']['f380']<0.05]
    poolok=[n for n,r in res.items() if r['n_pool_legs']>=2]
    m1=bool(set(motif_ok)&set(graded)&set(poolok))
    m2=bool(motif_ok and graded and not m1)
    m3=bool(res['D1_pool_ext_scaled_003']['n_pool_legs']>=2 and res['D3_scaled003_motif_zero']['n_pool_legs']==0)
    g1ok=res['G1_gate_w5e-4']['gate_bitwise']
    branch='m1' if (g1ok and m1) else ('m3' if (g1ok and m3) else ('m2' if (g1ok and m2) else 'm4'))
    dump(o/'OUTCOMES.json',dict(round='S319',branch=branch,gate=g1ok,
      motif_rhythmic_arms=motif_ok,graded_arms=graded,pool_rhythmic_arms=poolok,
      arms={k:dict(n_motif=r['n_motif_legs'],n_pool=r['n_pool_legs'],pool_median=round(r['pool']['median'],1),
        pool_f380=round(r['pool']['f380'],3),common_mode=round(r['common_mode'],3)) for k,r in res.items()},
      prediction_check=[dict(w=t['w'],predicted_equiv_nA=t['equiv_drive_nA'],
         measured_motif_legs=res.get({1.5e-5:'W1_w1p5e-5',2.5e-5:'W2_w2p5e-5',5e-5:'W3_w5e-5',5e-4:'G1_gate_w5e-4'}[t['w']],{}).get('n_motif_legs')) for t in scale],
      unresolved=['rf_E2 external net drive is negative','rm_E2 requires an out-of-family scale','ConductanceParams illustrative/uncalibrated','16 unresolved channels listed not redistributed'],
      one_line=branch))
    print(json.dumps(dict(branch=branch,gate=g1ok,motif_ok=motif_ok,graded=graded,poolok=poolok,
      counts=Cnt),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
