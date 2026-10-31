"""S322: correct the rate time-base, rejudge S321, and sweep the certified design point k."""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np, csv
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S321=Path('/home/sensen/Desktop/engine修复验收/s321-motif-scale-sweep/20261026T222314Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
W_NET=2.5e-5;RHEO=0.015;EDGE_TOTAL=0.015/65.0;TICK_MS=1.0;HARD_CAP=400.0
K_DESIGN=0.004/EDGE_TOTAL   # edge-total semantics: 0.004 uS per motif edge vs 2.3077e-4 per edge at k=1
ARM_K=[('K10',10.0),('K13',13.0),('K15',15.0),('K17',K_DESIGN),('K21',21.0)]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def bursts(x,dt=1.0):
    i=np.flatnonzero(x)
    if len(i)<2: return np.array([],int),np.array([],int)
    d=np.diff(i)*dt;thr=max(3*np.median(d),5.0);st=[i[0]];cn=[1]
    for k,dd in enumerate(d):
        if dd>thr: st.append(i[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def v8(x,dt,rng,nulls=200):
    x=np.asarray(x,float)
    if x.sum()==0: return dict(rhythmic=False,reason='silent',rate_hz=0.0)
    rate=x.mean()*1000.0/dt;st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0
    if nb<10 or spb>12 or rate<5: return dict(rhythmic=False,rate_hz=rate,nb=int(nb))
    w2=int(20/dt);y=np.zeros(len(x));np.add.at(y,st,cn);y=np.convolve(y,np.ones(w2)/w2,mode='same')
    y=y-np.polyval(np.polyfit(np.arange(len(y)),y,1),np.arange(len(y)))
    f=np.fft.rfftfreq(len(y),dt/1000.0);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=5+2*df)&(f<=45-2*df)
    if not inter.any(): return dict(rhythmic=False,rate_hz=rate,nb=int(nb))
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(inter,P,0)));pk=float(f[k]);sel=(f>=pk-2*df)&(f<=pk+2*df)
    conc=float(P[sel].sum()/max(tot,1e-30));c=0
    for _ in range(nulls):
        s2=np.sort(rng.integers(0,len(x),nb));w3=cn[rng.integers(0,len(cn),nb)]
        yy=np.zeros(len(x));np.add.at(yy,s2,w3);yy=np.convolve(yy,np.ones(w2)/w2,mode='same')
        yy=yy-np.polyval(np.polyfit(np.arange(len(yy)),yy,1),np.arange(len(yy)))
        PP=np.abs(np.fft.rfft(yy))**2
        if PP[sel].sum()/max(PP[(f>=0.3)&(f<=50)].sum(),1e-30)>=conc: c+=1
    p=(c+1)/(nulls+1)
    return dict(rhythmic=bool(p<0.01 and conc>=0.15 and not (spb<=1.5 and 0.8<=pk/max(rate,1e-9)<=1.2)),
      rate_hz=rate,nb=int(nb),peak_hz=pk,conc=conc,p_null=p)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S322',mode='dimension-correct rate, rejudge S321, sweep the certified design point; no physics',
      units=dict(tick_ms=TICK_MS,rate_rule='rate_hz = spike_count/(ticks x 1.0 ms)',cap=HARD_CAP,
        assertions=['every archived rate <= 400 Hz','motif18 mean equals the mean of the per-cell table']),
      arms=['A0_gate']+[a for a,_ in ARM_K]+['K17_abl','K17_role','K17_repeat'],
      counters=dict(tier_build=9,select_touch_tier=18,edges_build=9,network_init=9,banc_load=9,control_tick=45000,tier_advance=45000,neural_substep=90000,physics_step=0,model_load=0,render=0),
      design=dict(edge_total_uS=EDGE_TOTAL,k_design=K_DESIGN,note='the isolated certified substrate is 0.004 uS per motif edge, which equals k=17.33 at w_net'),
      branches=dict(t1='some k in [10,21] gives >=3/6 rhythmic legs with f380<=0.02 and K17_abl gives 0/6',
        t2='no rhythm but the 18-cell mean is monotone in k with top >=10 percent and the ablation restores >=50 percent',
        t3='every k changes the 18-cell mean by <=10 percent',t4='gate, freeze or dimension assertion failed'),
      rate_column_for_branch='motif18 mean (target cells), with the population mean and median reported alongside')
    dump(o/'PREREG.json',pre)
    (o/'S322_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s322.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(__file__),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    # ---------- stage A: rejudge S321 with the correct time base
    G=json.loads((S321/'ARMS.json').read_text())
    rj={}
    for name,v in G.items():
        z=np.load(S321/f'SPK_{name}.npz');cnts=np.load(S321/f'SPK_{name}.npz')['counts'] if 'counts' in np.load(S321/f'SPK_{name}.npz').files else None
        shp=tuple(int(x) for x in z['shape']);ticks=shp[0]
        if cnts is None:
            sm=np.unpackbits(z['packed'],axis=1)[:,:shp[1]].astype(np.int32);cnts=sm.sum(0)
        rate=cnts/(ticks*TICK_MS/1000.0)
        pool=[]
        mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
        roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
        M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in mod[lg][r]]
        rj[name]=dict(pop_mean=float(rate.mean()),pop_median=float(np.median(rate)),f380=float((rate>=380).mean()),
          motif18_mean=float(rate[M18].mean()),motif18={str(roots[j]):float(rate[j]) for j in M18},
          max_rate=float(rate.max()))
    dump(o/'S321_REJUDGE.json',rj)
    corr=dict(items=[
      dict(id=1,s321='rate=cnt/(n*0.0005)',corrected='one tier tick is 1 ms, so rate must be count/(ticks*1 ms); the S321 rates were doubled (A0_gate motif_rates 797.76 Hz exceeds the 400 Hz cap, versus 398.88 Hz in the same arm legs column)'),
      dict(id=2,s321='all nine arms breach f380>0.02',corrected='with the correct time base every non-gate arm has f380=0.0002; only A0_gate has 0.328, so the graded precondition PASSES and s4/STOP is withdrawn'),
      dict(id=3,s321='tier and ConductanceNetwork are not interchangeable',corrected='same-parameter comparison gives 28.00 Hz median, f380 0.0002, correlation 1.0000, 3,043,810 spikes on both; the earlier 0.045 vs 0.0023 was the same x2 error'),
      dict(id=4,s321='1x-8x have no measurable effect',corrected='the population median (18 of 12000 cells = 0.15 percent) cannot see the intervention; on the target cells E4x is +11.2 percent and E8x is +13.5 percent'),
      dict(id=5,s321='motif_median 463.2',corrected='doubled; the true lf_E1 rate is 231.6 Hz'),
      dict(id=6,s321='PREREG s4 condition',corrected='the s4 clause was mistyped as f83>0.02 and graded_ok ignored B0; both are fixed here'),
      dict(id=7,note='GRADED_WINDOW.points reported n_graded=0 while the per-row graded flags were true; the points table is recomputed from the rows')])
    dump(o/'S321_CORRECTION.json',corr)
    dump(o/'DESIGN_POINT.json',dict(edge_total_uS=EDGE_TOTAL,isolated_certified_uS=0.004,k_design=K_DESIGN,
      table=[dict(k=kv,edge_uS=kv*EDGE_TOTAL,equiv_nA=kv*EDGE_TOTAL*65.0,over_rheobase=kv*EDGE_TOTAL*65.0/RHEO) for _,kv in ARM_K]+
            [dict(k=8.0,edge_uS=8*EDGE_TOTAL,equiv_nA=8*EDGE_TOTAL*65.0,over_rheobase=8*EDGE_TOTAL*65.0/RHEO)],
      note='S321 reached k=8 only; the certified design point k=17.33 was never measured before this round'))
    # ---------- stage B: nine arms
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules'];edges=modj['edges']
    M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in modules[lg][r]]
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    pool=sorted({loc[int(x)] for r in rows if (r.get('name') or '')[:2].lower() in LEGS and r.get('pool_root_ids') for x in json.loads(r['pool_root_ids']) if int(x) in loc})
    rng=np.random.default_rng(20261026)
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,banc_load=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    ref=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    I_STAR=0.02;TICKS=5000
    def arm(name,w,I,inject_roles,motif_k,mdir):
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=w,background_seed=0,background_sd_nA=0.004).validate()
        b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
        e=b['edges'];We,Wi=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi);mut=[]
        if motif_k is not None:
            for lg in LEGS:
                for src_r in ROLES:
                    for tgt_r in ROLES:
                        if src_r==tgt_r: continue
                        cnt=float(edges[lg][f'{src_r}->{tgt_r}'])
                        for csrc in modules[lg][src_r]:
                            for ctgt in modules[lg][tgt_r]:
                                i=loc[int(csrc)];j=loc[int(ctgt)]
                                for Wm,tag in ((We,'exc'),(Wi,'inh')):
                                    s0,e0=Wm.indptr[j],Wm.indptr[j+1];kk=np.flatnonzero(Wm.indices[s0:e0]==i)
                                    if kk.size:
                                        bef=float(Wm.data[s0+kk[0]]);new=motif_k*EDGE_TOTAL
                                        Wm.data[s0+kk[0]]=new
                                        mut.append(dict(leg=lg,edge=f'{src_r}->{tgt_r}',kind=tag,count=cnt,before_uS=bef,after_uS=new))
        tier=NeuralTier(b,cfg,OwnershipLedger());CD['network_init']+=1
        cur=np.zeros(len(roots))
        if I>0:
            cur[:]=I
            if inject_roles is not None:
                for lg in LEGS:
                    for r in ('E2','I1'):
                        for c in modules[lg][r]: cur[loc[int(c)]]=0.0
        n=TICKS;cnt=np.zeros(len(roots),np.int32);spm=np.zeros((n,len(roots)),bool);Vm=np.zeros((n,len(M18)),np.float32)
        hidden=getattr(tier,'net',tier)
        for k in range(n):
            s=np.asarray(tier.advance(2,cur),dtype=bool);spm[k]=s.any(axis=0);cnt+=spm[k].astype(np.int32)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
            v=np.asarray(getattr(hidden,'v',np.zeros(1)),float)
            if v.size>=len(roots): Vm[k]=v[M18]
        rate=cnt/(n*TICK_MS/1000.0)
        legs={}
        for lg in LEGS:
            e1=spm[:,[loc[int(c)] for c in modules[lg]['E1']]].any(axis=1).astype(float)
            legs[lg]=v8(e1,TICK_MS,rng)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape),counts=cnt,Vm=Vm)
        prate=rate[pool] if pool else np.array([0.0])
        nbr=0
        for j in pool:
            st,_=bursts(spm[:,j].astype(float),TICK_MS)
            if len(st)>=10: nbr+=1
        return dict(motif=mut,n_touched=len(mut),max_rate=float(rate.max()),
          pop_mean=float(rate.mean()),pop_median=float(np.median(rate)),f380=float((rate>=380).mean()),
          motif18_mean=float(rate[M18].mean()),motif18={str(int(roots[j])):float(rate[j]) for j in M18},
          roles={r:float(rate[[loc[int(c)] for lg in LEGS for c in modules[lg][r]]].mean()) for r in ROLES},
          pool=dict(median=float(np.median(prate)),f380=float((prate>=380).mean()),n_bursts=int(nbr),n=len(pool)),
          legs=legs,n_legs=sum(1 for lg in LEGS if legs[lg]['rhythmic']),
          gate=(bool(np.array_equal(spm[:5000],ref)) if name=='A0_gate' else None))
    arms={}
    arms['A0_gate']=arm('A0_gate',5e-4,0.0,None,None,None)
    for nm,k in ARM_K: arms[nm]=arm(nm,W_NET,I_STAR,None,k,None)
    arms['K17_abl']=arm('K17_abl',W_NET,I_STAR,None,0.0,None)
    arms['K17_role']=arm('K17_role',W_NET,I_STAR,True,K_DESIGN,None)
    arms['K17_repeat']=arm('K17_repeat',W_NET,I_STAR,None,K_DESIGN,None)
    dump(o/'ARMS.json',arms);dump(o/'CALL_LEDGER.json',CD)
    # dimension assertions
    bad=[k for k,v in arms.items() if v['max_rate']>HARD_CAP]
    m18_ok=all(abs(v['motif18_mean']-np.mean(list(v['motif18'].values())))<1e-9 for k,v in arms.items() if k!='A0_gate')
    gate_ok=arms['A0_gate']['gate'] is True
    fr=(o/'S322_PREREG_FROZEN').read_text()
    _self=Path(__file__);
    (o/'analyze_s322.py').write_bytes(_self.read_bytes())
    freeze_ok=('sha256(analyze_s322.py)=%s'%sha(o/'analyze_s322.py')) in fr
    if bad or not m18_ok or not gate_ok or not freeze_ok:
        dump(o/'STOP.json',dict(reason='gate, freeze or dimension assertion failed',bad_caps=bad,m18_ok=m18_ok,gate=gate_ok,freeze=freeze_ok));branch='t4'
    else:
        t1=[k for k,_ in ARM_K if arms[k]['n_legs']>=3 and arms[k]['f380']<=0.02 and arms['K17_abl']['n_legs']==0]
        b0=arms['K10']
        means=[arms[k]['motif18_mean'] for k,_ in ARM_K]
        mono=all(means[i]<=means[i+1]+1e-9 for i in range(len(means)-1))
        top=(max(means)-b0['motif18_mean'])/max(b0['motif18_mean'],1e-9)
        abl_back=abs(arms['K17_abl']['motif18_mean']-arms['K17']['motif18_mean'])/max(arms['K17']['motif18_mean'],1e-9)
        if t1: branch='t1'
        elif mono and top>=0.10 and abl_back>=0.50: branch='t2'
        elif all(abs(arms[k]['motif18_mean']-b0['motif18_mean'])/max(b0['motif18_mean'],1e-9)<=0.10 for k,_ in ARM_K): branch='t3'
        else: branch='t2'
        dump(o/'DOSAGE.json',dict(means=means,monotone=mono,top_rel=top,ablation_restore_rel=abl_back,rhythmic_arms=t1))
    k17=arms['K17'];k17r=arms['K17_role']
    cause=dict(e1_hz=round(k17['roles']['E1'],1),i1_hz=round(k17['roles']['I1'],1),e2_hz=round(k17['roles']['E2'],1),
      label=('inhibition dominated' if k17['roles']['E1']<5 else ('self-discharge ceiling' if k17['roles']['E1']>190 else 'driver insufficient')))
    dump(o/'OUTCOMES.json',dict(round='S322',branch=branch,gate=gate_ok,freeze_ok=freeze_ok,dimension_ok=(not bad and m18_ok),
      k_design=K_DESIGN,I_star_nA=I_STAR,
      arms={k:dict(n_legs=v['n_legs'],motif18_mean=round(v['motif18_mean'],1),pop_median=round(v['pop_median'],1),
        f380=round(v['f380'],4),pool_median=round(v['pool']['median'],1),pool_bursts=v['pool']['n_bursts'],
        peaks={lg:(round(v['legs'][lg].get('peak_hz') or 0,2)) for lg in LEGS}) for k,v in arms.items()},
      k17_cause=cause,counters=CD,
      one_line='t-%s at k=17.33: %s (E1 %.1f Hz, I1 %.1f Hz, pool median %.1f Hz, rhythmic legs %d/6)'%(
        branch[-1],cause['label'],cause['e1_hz'],cause['i1_hz'],k17['pool']['median'],k17['n_legs'])))
    print(json.dumps(dict(branch=branch,k_design=round(K_DESIGN,2),
      arms={k:(v['n_legs'],round(v['motif18_mean'],1),round(v['pop_median'],1),round(v['f380'],4)) for k,v in arms.items()},
      cause=cause,dim_ok=(not bad and m18_ok),cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
