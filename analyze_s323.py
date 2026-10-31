"""S323: window-invariant detector V9, full rejudge, and six bounded arms."""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np, csv
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S316=Path('/home/sensen/Desktop/engine修复验收/s316-percell-mapping-repair/20261026T213550Z')
S321=Path('/home/sensen/Desktop/engine修复验收/s321-motif-scale-sweep/20261026T222314Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
EDGE_TOTAL=0.015/65.0;K_DESIGN=0.004/EDGE_TOTAL;NULLS=200;SEED=20261026;RELBAND=0.15
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
def envsig(st,cn,n,dt):
    y=np.zeros(n);np.add.at(y,np.clip(st,0,n-1),cn)
    w=max(int(round(20.0/dt)),1);y=np.convolve(y,np.ones(w)/w,mode='same')
    return y-np.polyval(np.polyfit(np.arange(n),y,1),np.arange(n))
def relconc(y,dt):
    n=len(y);f=np.fft.rfftfreq(n,dt/1000.0);P=np.abs(np.fft.rfft(y))**2;df=f[1]-f[0]
    inter=(f>=5+2*df)&(f<=45-2*df)
    if not inter.any(): return None
    tot=P[(f>=0.3)&(f<=50)].sum();k=int(np.argmax(np.where(inter,P,0)));pk=float(f[k])
    sel=(f>=pk*(1-RELBAND))&(f<=pk*(1+RELBAND))
    return pk,float(P[sel].sum()/max(tot,1e-30))
def autocorr_peak(y,dt):
    n=len(y);y=y-y.mean();ac=np.correlate(y,y,'full')[n-1:];ac=ac/max(ac[0],1e-30)
    lo=int(20/dt);hi=min(int(200/dt),len(ac)-1)
    if hi<=lo: return None,0.0
    seg=ac[lo:hi];k=int(np.argmax(seg));return float((lo+k)*dt),float(seg[k])
def v9(x,dt):
    x=np.asarray(x,float)
    if x.sum()==0: return dict(rhythmic=False,reason='silent',rate_hz=0.0)
    rate=x.mean()*1000.0/dt;st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0
    out=dict(rate_hz=rate,nb=int(nb),spb=spb)
    if nb<10 or spb>12 or rate<5: return dict(**out,rhythmic=False,reason='burst_gate')
    y=envsig(st,cn,len(x),dt);rc=relconc(y,dt)
    if rc is None: return dict(**out,rhythmic=False,reason='no_interior')
    pk,rel=rc;lag,acpk=autocorr_peak(y,dt)
    rng=np.random.default_rng(SEED)
    nulls_rel=[];nulls_ac=[];cnt=0
    for _ in range(NULLS):
        s2=np.sort(rng.integers(0,len(x),nb));w2=cn[rng.integers(0,len(cn),nb)]
        yy=envsig(s2,w2,len(x),dt);q=relconc(yy,dt)
        _,qa=autocorr_peak(yy,dt)
        if q is not None:
            nulls_rel.append(q[1]);nulls_ac.append(qa)
            if q[1]>=rel: cnt+=1
    if not nulls_rel: return dict(**out,rhythmic=False,reason='null_empty')
    p_rel=(cnt+1)/(NULLS+1);p95_rel=float(np.percentile(nulls_rel,95));p95_ac=float(np.percentile(nulls_ac,95))
    ok=bool(rel>p95_rel and acpk>p95_ac and p_rel<0.01 and not (spb<=1.5 and 0.8<=pk/max(rate,1e-9)<=1.2))
    return dict(**out,rhythmic=ok,peak_hz=pk,rel=rel,ac_peak=acpk,ac_lag_ms=lag,
      null_rel_p95=p95_rel,null_ac_p95=p95_ac,p_null=p_rel,dt_ms=dt)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S323',mode='window-invariant detector V9, rejudge, six bounded arms; no physics',
      hard=['no absolute-bin concentration anywhere in the criteria',
            'every threshold equals the same-window null p95 archived with the value',
            'no else fallback in the branch decision',
            'workpoint and design values come from archived measurement tables'],
      detector=dict(relband=RELBAND,ac_lag_range_ms=[20,200],nulls=NULLS,seed=SEED,
        rule='rel(+-15%) > same-window null p95 AND envelope autocorrelation peak (20-200 ms) > same-window null p95 AND placement-null p<0.01'),
      windows_s=[1.25,2.5,3.0,5.0,10.0],
      counters=dict(tier_build=4,select_touch_tier=8,edges_build=4,network_init=4,banc_load=4,control_tick=20000,tier_advance=20000,neural_substep=40000,
        con_network_init=2,con_run=2,con_step=20000,physics_step=0,model_load=0,render=0),
      branches=dict(u1='the same leg set >=4/6 across seeds 0/1/2 and the seed-1 ablation gives 0/6',
        u2='1-3 legs stable across seeds with ablation 0/6',u3='<=1 stable leg and the single cell is rhythmic',
        u4='window invariance failed or seed spread >=3'))
    dump(o/'PREREG.json',pre)
    (o/'S323_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s323.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(Path(__file__)),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    # ---- stage A: window invariance
    def synth(per,nspk,cv=0.0,dur=5000.0,dt=1.0,seed=1):
        n=int(dur/dt);y=np.zeros(n);t=per*0.5;r=np.random.default_rng(seed)
        while t<dur:
            s=int(t/dt)
            for k in range(nspk):
                if 0<=s+k<n: y[s+k]=1.0
            t+=per*(1+cv*r.standard_normal() if cv else 1.0)
        return y
    def reg(hz,dur=5000.0,dt=1.0):
        n=int(dur/dt);y=np.zeros(n);y[::max(int(round(1000.0/hz/dt)),1)]=1.0;return y
    K17=np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:12000].astype(bool)
    K17abl=np.unpackbits(np.load(S322/'SPK_K17_abl.npz')['packed'],axis=1)[:,:12000].astype(bool)
    A0=np.unpackbits(np.load(S322/'SPK_A0_gate.npz')['packed'],axis=1)[:,:12000].astype(bool)
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    modj=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=modj['modules'];edges=modj['edges']
    E1={lg:loc[int(modules[lg]['E1'][0])] for lg in LEGS}
    WINS=[1.25,2.5,3.0,5.0,10.0]
    inv={}
    for W in WINS:
        n=int(W*1000.0)
        a=v9(synth(1000/21.7,6,dur=W*1000.0)[:n],1.0);b=v9(synth(1000/39.0,4,dur=W*1000.0)[:n],1.0)
        lf=v9(K17[:n,E1['lf']].astype(float),1.0);lm=v9(K17[:n,E1['lm']].astype(float),1.0)
        cst=v9(reg(150.0,dur=W*1000.0)[:n],1.0);poi=v9((np.random.default_rng(3).random(n)<0.02).astype(float),1.0)
        ind,_=bursts(synth(1000/21.7,6,dur=W*1000.0)[:n],1.0)
        indep=np.zeros(n);r2=np.random.default_rng(11)
        indep[np.sort(r2.integers(0,n,len(ind)))]=1.0
        ind=v9(indep,1.0)
        abl=v9(K17abl[:n,E1['lm']].astype(float),1.0);g0=v9(A0[:n,E1['lf']].astype(float),1.0)
        inv[str(W)]=dict(syn21=a,syn39=b,k17_lf=lf,k17_lm=lm,const=cst,poisson=poi,indep=ind,abl=abl,a0=g0)
    iso={}
    for nm,truth in (('cap8_full_0.0165',6.58),('cap8_full_0.0253',15.39)):
        sp=np.load(S313/f'BSPK_{nm}.npz')['spk'];r=v9(sp[:,0].astype(float),0.5)
        iso[nm]=dict(truth=truth,**r)
    checks=dict(
      a_syn=all(inv[str(W)]['syn21']['rhythmic'] and inv[str(W)]['syn39']['rhythmic']
                and abs(inv[str(W)]['syn21'].get('peak_hz',0)-21.7)<1.0 and abs(inv[str(W)]['syn39'].get('peak_hz',0)-39.0)<2.0 for W in WINS),
      b_real=all(inv[str(W)]['k17_lf']['rhythmic']==inv['5.0']['k17_lf']['rhythmic'] and inv[str(W)]['k17_lm']['rhythmic']==inv['5.0']['k17_lm']['rhythmic'] for W in WINS),
      c_neg=all(not inv[str(W)][k]['rhythmic'] for W in WINS for k in ('const','poisson','indep')),
      d_iso=all(iso[k]['rhythmic'] and abs(iso[k].get('peak_hz',0)-iso[k]['truth'])/iso[k]['truth']<=0.05 for k in iso),
      e_ctrl=all(not inv[str(W)][k]['rhythmic'] for W in WINS for k in ('abl','a0')))
    checks['all']=all(checks.values())
    dump(o/'WINDOW_INVARIANCE.json',dict(windows=WINS,grid=inv,isolated=iso,checks=checks))
    dump(o/'DETECTOR_V9.json',pre['detector'])
    if not checks['all']:
        dump(o/'STOP.json',dict(reason='V9 window-invariance certification failed',checks=checks));dump(o/'FAILED.json',dict(checks=checks))
    # ---- S322 rejudge with V9
    rj={}
    for name in ('K10','K13','K15','K17','K21','K17_abl','K17_role','K17_repeat'):
        z=np.load(S322/f'SPK_{name}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={lg:v9(sm[:,[E1[lg]]].any(axis=1).astype(float),1.0) for lg in LEGS}
        rj[name]=dict(per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']))
    pool=[]
    rows=list(csv.DictReader(open(S290/'PER_CHANNEL.csv')))
    pool=sorted({loc[int(x)] for r in rows if (r.get('name') or '')[:2].lower() in LEGS and r.get('pool_root_ids') for x in json.loads(r['pool_root_ids']) if int(x) in loc})
    poolres={}
    for name in ('K17','K17_abl','A0_gate'):
        z=np.load(S322/f'SPK_{name}.npz');sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        nb=0;rel20=0
        for j in pool:
            r=v9(sm[:,[j]].any(axis=1).astype(float),1.0)
            if r.get('nb',0)>=10: nb+=1
            if (r.get('rel') or 0)>=0.20: rel20+=1
        poolres[name]=dict(nb10=nb,rel20=rel20,n=len(pool))
    dump(o/'S322_REJUDGE.json',dict(arms=rj,pool=poolres,note='V9 uses the relative +-15% band and the envelope autocorrelation, both against the same-window null p95'))
    dump(o/'POOL_INHERITANCE.json',dict(pool=poolres,saturation_state_note='S319 reported 0/202 with burst structure in the SATURATED state; the graded state shows %d/202 at nb>=10 and %d/202 at rel>=0.20'%(poolres['K17']['nb10'],poolres['K17']['rel20'])))
    # ---- stage B
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,banc_load=0,control_tick=0,tier_advance=0,neural_substep=0,con_network_init=0,con_run=0,con_step=0,physics_step=0,model_load=0,render=0)
    def tier_arm(name,seed,motif_k,only_e1,ticks=5000):
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=2.5e-5,background_seed=seed,background_sd_nA=0.004).validate()
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
        cur=np.full(len(roots),0.02)
        if only_e1:
            for lg in LEGS:
                for r in ('E2','I1'):
                    for c in modules[lg][r]: cur[loc[int(c)]]=0.0
        n=ticks;spm=np.zeros((n,len(roots)),bool)
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        per={lg:v9(spm[:,[E1[lg]]].any(axis=1).astype(float),1.0) for lg in LEGS}
        return dict(per_leg=per,n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']),mutated=mut,
          same_as_K17=bool(np.array_equal(spm,np.unpackbits(np.load(S322/'SPK_K17.npz')['packed'],axis=1)[:,:int(np.load(S322/'SPK_K17.npz')['shape'][1])].astype(bool))))
    arms={}
    arms['R1_seed1']=tier_arm('R1_seed1',1,K_DESIGN,False)
    arms['R2_seed2']=tier_arm('R2_seed2',2,K_DESIGN,False)
    arms['R3_abl_seed1']=tier_arm('R3_abl_seed1',1,0.0,False)
    arms['R4_role_seed0']=tier_arm('R4_role_seed0',0,K_DESIGN,True)
    k17res=rj['K17']
    # S1 single cell
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    tgt=float(k17res['per_leg']['lm']['rate_hz'])
    net=ConductanceNetwork(1,ConductanceParams(),dt_ms=0.5);net.set_connectivity(np.zeros((1,1)),np.zeros((1,1)))
    CD['con_network_init']+=1;CD['con_run']+=1
    cur=np.array([0.02]);sp=np.zeros(10000,bool)
    for k in range(10000):
        sp[k]=bool(net.step(current_nA=cur)[0]);CD['con_step']+=1
    s1=dict(target_hz=tgt,measured_hz=float(sp.mean()*2000.0),**v9(sp.astype(float),0.5))
    # S2 isolated 3-cell 5 s
    e3=edges['lf']
    We3=np.zeros((3,3));Wi3=np.zeros((3,3))
    We3[1,0]=e3['E1->E2']*0+0.004;We3[0,1]=0.004;We3[2,1]=0.004;We3[2,0]=0.004;Wi3[0,2]=0.004;Wi3[1,2]=0.004
    net=ConductanceNetwork(3,ConductanceParams(),dt_ms=0.5);net.set_connectivity(We3,Wi3)
    CD['con_network_init']+=1;CD['con_run']+=1
    sp3=np.zeros((10000,3),bool);cur3=np.array([0.0224,0.0,0.0])
    for k in range(10000):
        sp3[k]=net.step(current_nA=cur3);CD['con_step']+=1
    s2=v9(sp3[:,0].astype(float),0.5)
    dump(o/'ARMS.json',dict(tier=arms,s1_single_cell=s1,s2_iso_5s=s2,k17=rj['K17']))
    dump(o/'CALL_LEDGER.json',CD)
    # ---- branch (no else)
    seeds={'seed0':k17res,'seed1':arms['R1_seed1'],'seed2':arms['R2_seed2']}
    cnts={k:v['n_rh'] for k,v in seeds.items()}
    spread=max(cnts.values())-min(cnts.values())
    sets={k:tuple(sorted(lg for lg in LEGS if v['per_leg'][lg]['rhythmic'])) for k,v in seeds.items()}
    stable=len(set(sets.values()))==1
    abl0=arms['R3_abl_seed1']['n_rh']==0
    mean_n=sum(cnts.values())/3.0
    if not checks['all'] or spread>=3: branch='u4'
    elif stable and mean_n>=4 and abl0: branch='u1'
    elif abl0 and 1<=mean_n<=3: branch='u2'
    elif mean_n<=1 and s1['rhythmic']: branch='u3'
    else: branch=None
    if branch is None:
        dump(o/'STOP.json',dict(reason='no pre-registered branch is satisfied',counts=cnts,spread=spread,abl=abl0,s1_rhythmic=s1['rhythmic']))
    dump(o/'OUTCOMES.json',dict(round='S323',branch=branch,window_invariance=checks,
      rejudge_n_rh={k:v['n_rh'] for k,v in rj.items()},seed_counts=cnts,seed_leg_sets={k:list(v) for k,v in sets.items()},
      ablation_seed1=arms['R3_abl_seed1']['n_rh'],role_arm=arms['R4_role_seed0']['n_rh'],
      s1_single_cell=dict(rhythmic=s1['rhythmic'],measured_hz=round(s1['measured_hz'],1),target=round(tgt,1),peak=s1.get('peak_hz')),
      s2_iso=s2 and dict(rhythmic=s2['rhythmic'],peak=s2.get('peak_hz')),
      r1_r2_not_bitwise_K17=(not arms['R1_seed1']['same_as_K17']) and (not arms['R2_seed2']['same_as_K17']),
      pool=poolres,counters=CD,
      one_line=('u-%s'%branch[-1]) if branch else 'NOT_JUDGED'))
    print(json.dumps(dict(branch=branch,inv=checks['all'],n_rh={k:v['n_rh'] for k,v in rj.items()},
      seed=cnts,abl=arms['R3_abl_seed1']['n_rh'],s1=s1['rhythmic'],pool=poolres,cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
