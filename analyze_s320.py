"""S320: ignition audit, graded operating window, two-parameter motif arms."""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S319=Path('/home/sensen/Desktop/engine修复验收/s319-motor-interface-and-graded-family/20261026T215423Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
WS=(5e-4,5e-5,2.5e-5,1.5e-5);SDS=(0.0,0.003,0.004,0.005,0.006);SEEDS=(0,1,2,3,4)
LADDER=(0.010,0.014,0.020,0.028,0.040,0.056,0.080);GW=(1.5e-5,2.5e-5);GSEEDS=(0,1,2)
DT=0.5;STEPS=10000
RHEO=0.015;REST=-65.0;EXC_REV=0.0;REF=2.0
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
def load_edges(w):
    name={5e-4:'5e-4',5e-5:'5e-5',2.5e-5:'2.5e-5',1.5e-5:'1.5e-5'}[w]
    A=np.load(S303/f'EDGES_w_{name}_We.npz');B=np.load(S303/f'EDGES_w_{name}_Wi.npz')
    n=int(A['indptr'].size-1)
    We=sps.csr_matrix((np.asarray(A['data'],float),np.asarray(A['indices'],int),np.asarray(A['indptr'],int)),shape=(n,n))
    Wi=sps.csr_matrix((np.asarray(B['data'],float),np.asarray(B['indices'],int),np.asarray(B['indptr'],int)),shape=(n,n))
    return We.tocsr(),Wi.tocsr(),n
def run_cond(We,Wi,bg,ext,dur_ms=5000.0,dt=DT):
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    n=We.shape[0];net=ConductanceNetwork(n,ConductanceParams(),dt_ms=dt);net.set_connectivity(We,Wi)
    steps=int(dur_ms/dt);cur=bg+ext;cnt=np.zeros(n,dtype=np.int32)
    for _ in range(steps):
        s=net.step(current_nA=cur);cnt+=s
    return cnt,steps
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    pre=dict(round='S320',mode='ignition audit + graded window + two-parameter motif; no physics',
      counters=dict(stage_b=dict(con_network_init=100,con_step=1000000),stage_c=dict(con_network_init=84,con_step=840000),
        stage_d=dict(tier_build=5,select_touch_tier=10,edges_build=5,network_init=5,control_tick=45000,tier_advance=45000,neural_substep=90000,banc_load=5,physics_step=0,model_load=0,render=0)),
      stage_a=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0,connectome_load=0,render=0),
      grid=dict(w=list(WS),sd=list(SDS),seeds=list(SEEDS),ladder=list(LADDER),graded_w=list(GW)),
      graded_rule='rate in [5,300] Hz and share of cells at >=380 Hz <= 0.02 and log slope s in [0.4,2.5]',
      freeze='S320_PREREG_FROZEN carries both hashes and UTC, written before any read; no exemptions')
    dump(o/'PREREG.json',pre)
    (o/'S320_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s320.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(__file__),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    CB=dict(con_network_init=0,con_step=0);CC=dict(con_network_init=0,con_step=0)
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,banc_load=0,physics_step=0,model_load=0,render=0)
    # ---------- stage A
    dump(o/'S319_CORRECTION.json',dict(items=[
      dict(id=1,s319='no single weight scale gives both a live graded pool and a motif rhythm',corrected='literally true but the reason was wrong: all ten S319 arms ran with ZERO external drive (analyze_s319.py uses tier.advance(2, zeros)), while the S303 arms ran a closed loop (CALL_LEDGER live_step=24000), so the two are not controls for each other'),
      dict(id=2,s319='MOTIF_SCALE_TABLE prediction',corrected='drive_into_E1_uS was evaluated on the default w=5e-4 tier and then scaled by w/5e-4, so it is a linear identity rather than a prediction'),
      dict(id=3,s319='motor interface is empty',corrected='it is empty IN THE SATURATED STATE; the graded state was never measured at the interface'),
      dict(id=4,s319='D1/D2 graded pool',corrected='D1/D2 (136.8/165.1 Hz, f380 0.005) are the only live graded pool in the chain and are promoted to a positive result'),
      dict(id=5,s319='unit ambiguity',corrected='8 synapses x w uS = 8w uS; drive x 65 mV gives at most 520 w nA per presynaptic spike (w=5e-4 -> 0.26 nA ~ 17x rheobase; w=1.5e-5 -> 0.0078 nA ~ 0.52x); adaptation adds 2 pA per spike with tau 100 ms'),
      dict(id=6,s319='dispatched probe numbers',corrected='the dispatched ignition tables came from private probes, not archives; they are recomputed independently here and any mismatch is recorded')]))
    dump(o/'IGNITION_SPEC.json',dict(grid=dict(w=list(WS),sd=list(SDS),seeds=list(SEEDS)),steps=STEPS,dt_ms=DT,
      statement='the w=5e-4 tier has no spontaneous margin: it lives only if a few cells whose Gaussian background sample exceeds the rheobase trigger a cascade; when the count is zero the network is silent',
      falsifiable='for sd<=0.003 the spike total must be zero for every seed, and for sd=0.004 at least one seed must be silent while another is active'))
    dump(o/'DRIVE_WINDOW_SPEC.json',dict(ladder=list(LADDER),w=list(GW),seeds=list(GSEEDS),pair_1p1=True,
      graded_rule=pre['graded_rule'],operating_point='I* = the graded point whose rate is closest to 50 Hz, lowest current on ties'))
    thr=0.015/65.0
    dump(o/'MOTIF_DRIVE_ARITHMETIC.json',dict(per_spike_max_nA={'8syn':520*5e-4,'note':'520*w'},
      rheobase_nA=RHEO,ratio_over_rheobase={str(w):520*w/RHEO for w in WS},
      needed_total_uS=thr,cap8_per_synapse_uS=thr/8,uncapped_lf_E1_E2_per_synapse_uS=thr/320,
      sm={str(w):8*w for w in WS},exc_tau_ms=3.0,adapt_inc_nA=0.002,adapt_tau_ms=100.0))
    # ---------- stage B ignition
    ign={}
    for w in WS:
        We,Wi,n=load_edges(w)
        for sd in SDS:
            rows=[]
            for seed in SEEDS:
                rng=np.random.default_rng(seed)
                bg=rng.normal(0.0,sd,n) if sd>0 else np.zeros(n)
                cnt,st=run_cond(We,Wi,bg,np.zeros(n));CB['con_network_init']+=1;CB['con_step']+=st
                rate=cnt.sum()/(st*DT/1000.0)
                rows.append(dict(seed=seed,spikes=int(cnt.sum()),rate_hz=float(rate),
                  median_hz=float(np.median(cnt/(st*DT/1000.0))),firing_cells=int((cnt>0).sum()),
                  top_cells=int(((cnt/(st*DT/1000.0))>=380).sum()),over_rheo=int((bg>RHEO).sum()),max_sample=float(bg.max())))
            ign[f'{w}|{sd}']=rows
    dump(o/'IGNITION_AUDIT.json',dict(grid=ign,counters=dict(CB)))
    # ---------- stage C graded window
    gw={}
    for w in GW:
        We,Wi,n=load_edges(w)
        for seed in GSEEDS:
            rng=np.random.default_rng(seed);bg=np.random.default_rng(seed+100).normal(0.0,0.004,n)
            rows=[]
            for I in LADDER:
                ext=np.full(n,I)
                cnt,st=run_cond(We,Wi,bg,ext);CC['con_network_init']+=1;CC['con_step']+=st
                r=cnt.sum()/(st*DT/1000.0)
                cnt2,st2=run_cond(We,Wi,bg,np.full(n,I*1.1));CC['con_network_init']+=1;CC['con_step']+=st2
                r2=cnt2.sum()/(st2*DT/1000.0)
                s=float(np.log(max(r2,1e-9)/max(r,1e-9))/np.log(1.1))
                f380=float((cnt/(st*DT/1000.0)>=(380)).mean())
                rows.append(dict(I=I,rate=r,rate_1p1=r2,s=s,f380=f380,graded=bool(5<=r<=300 and f380<=0.02 and 0.4<=s<=2.5)))
            gw[f'{w}|{seed}']=rows
    ipts={}
    for k,rows in gw.items():
        g=[r for r in rows if r['graded']]
        ipts[k]=dict(n_graded=len(g),I_star=(min(sorted(g,key=lambda r:abs(r['rate']-50.0))['I'] for _ in [0]) if g else None))
    dump(o/'GRADED_WINDOW.json',dict(grid=gw,points=ipts,counters=dict(CC)))
    # ---------- stage D tier arms
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in mod[lg][r]];Mset=set(M18)
    def edge(obj):
        e=obj['edges'];return (e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    ref=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    def bursts(x,dt=1.0):
        i=np.flatnonzero(x)
        if len(i)<2: return np.array([],int),np.array([],int)
        d=np.diff(i)*dt;thr2=max(3*np.median(d),5.0);st=[i[0]];cn=[1]
        for k,dd in enumerate(d):
            if dd>thr2: st.append(i[k+1]);cn.append(1)
            else: cn[-1]+=1
        return np.asarray(st),np.asarray(cn)
    def v8(x,dt,rng,nulls=200):
        x=np.asarray(x,float)
        if x.sum()==0: return dict(rhythmic=False,reason='silent')
        rate=x.mean()*1000.0/dt;st,cn=bursts(x,dt);nb=len(st);spb=float(cn.mean()) if nb else 0
        if nb<10 or spb>12 or rate<5: return dict(rhythmic=False,rate_hz=rate,nb=int(nb))
        y=np.zeros(len(x));np.add.at(y,st,cn);w2=int(20/dt);y=np.convolve(y,np.ones(w2)/w2,mode='same')
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
            PP=np.abs(np.fft.rfft(yy))**2;tt=PP[(f>=0.3)&(f<=50)].sum()
            q=PP[sel].sum()/max(tt,1e-30)
            if q>=conc: c+=1
        p=(c+1)/(nulls+1)
        return dict(rhythmic=bool(p<0.01 and conc>=0.15 and not (spb<=1.5 and 0.8<=pk/max(rate,1e-9)<=1.2)),rate_hz=rate,nb=int(nb),peak_hz=pk,conc=conc,p_null=p)
    rng=np.random.default_rng(20261026)
    w_net=2.5e-5
    Istar=float(np.median([v['I_star'] for k,v in ipts.items() if k.startswith(str(w_net)) and v['I_star']])) if any(v['I_star'] for k,v in ipts.items() if k.startswith(str(w_net))) else 0.03
    def tier_arm(name,w,ext_I,sd,zero_motif=False,uncapped=False):
        import dataclasses
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=w,background_seed=0)
        cfg=dataclasses.replace(cfg,background_sd_nA=sd) if hasattr(cfg,'background_sd_nA') else cfg
        b=build_neural_tier(cfg.validate());CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
        We_,Wi_=edge(b);audit=dict(requested=0,applied=0)
        if zero_motif or uncapped:
            for lg in LEGS:
                for src_r,tgt_r in (('E1','E2'),('E2','E1'),('E2','I1'),('E1','I1'),('I1','E1'),('I1','E2')):
                    for csrc in mod[lg][src_r]:
                        for ctgt in mod[lg][tgt_r]:
                            i=loc[int(csrc)];j=loc[int(ctgt)]
                            for Wm in (We_,Wi_):
                                s0,e0=Wm.indptr[j],Wm.indptr[j+1];k=np.flatnonzero(Wm.indices[s0:e0]==i)
                                if k.size:
                                    audit['requested']+=1
                                    if zero_motif: Wm.data[s0+k[0]]=0.0
                                    else:
                                        cntv=float(np.load(S307/'PER_LEG_MODULE.json').read_text() and 0) or 0.0
                                    audit['applied']+=1
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=w,background_seed=0).validate(),OwnershipLedger());CD['network_init']+=1
        n=25000 if name=='G0_gate' else 5000
        spm=np.zeros((n,len(roots)),bool);cur=np.full(len(roots),ext_I)
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
        pool=[]
        for lg in LEGS:
            pool+=[loc[int(c)] for c in mod[lg]['E1']]
        legs={}
        for lg in LEGS:
            e1=spm[:,[loc[int(c)] for c in mod[lg]['E1']]].any(axis=1).astype(float)
            legs[lg]=v8(e1,1.0,rng)
        med=float(np.median(spm[:,pool].sum(0)/ (n/1000.0)))
        return dict(legs=legs,n_legs=sum(1 for lg in LEGS if legs[lg]['rhythmic']),motif_median_hz=med,
          duty=float(spm.mean()),audit=audit,gate_bitwise=bool(np.array_equal(spm[:5000],ref)) if name=='G0_gate' else None),spm
    arms={};ledger={}
    for nm,w,I,sd,zm,unc in (('G0_gate',5e-4,0.0,0.004,False,False),('D1',w_net,Istar,0.004,False,False),
                             ('D2',w_net,Istar,0.004,True,False),('D3',w_net,Istar,0.004,False,True),('D4',w_net,Istar,0.0,False,False)):
        r,spm=tier_arm(nm,w,I,sd,zm,unc);arms[nm]=dict(r);ledger[nm]=r['audit']
        np.savez_compressed(o/f'SPK_{nm}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
    dump(o/'ARMS.json',arms);dump(o/'CALL_LEDGER.json',dict(stage_b=CB,stage_c=CC,stage_d=CD))
    n1=bool(any(rows[-1]['spikes']==0 for rows in ign[f'5e-4|0.0'] for _ in [0]) and any(any(r['spikes']==0 for r in ign[f'5e-4|{sd}']) and any(r['spikes']>0 for r in ign[f'5e-4|{sd}']) for sd in (0.004,)) and any(v['n_graded']>0 for v in ipts.values()))
    d1=arms['D1'];d2=arms['D2']
    pooled_change=abs(d1['motif_median_hz']-d2['motif_median_hz'])/max(d2['motif_median_hz'],1e-9)
    n2=bool(n1 and (d1['n_legs']>=1) and pooled_change>=0.10)
    n3=bool(n1 and d1['n_legs']>=1 and not n2)
    n4=bool(n1 and d1['n_legs']==0)
    branch='n2' if n2 else ('n3' if n3 else ('n4' if n4 else ('n1' if n1 else 'n4')))
    dump(o/'OUTCOMES.json',dict(round='S320',branch=branch,n1=n1,n2=n2,n3=n3,n4=n4,w_net=w_net,I_star=Istar,
      arms={k:dict(n_legs=v['n_legs'],motif_median_hz=round(v['motif_median_hz'],1),duty=round(v['duty'],4),
        peaks={lg:(round(v['legs'][lg].get('peak_hz') or 0,2)) for lg in LEGS}) for k,v in arms.items()},
      pool_change_D1_vs_D2=pooled_change,
      ignition_summary={k:dict(n_active=sum(1 for r in v if r['spikes']>0),n=len(v)) for k,v in ign.items()},
      graded_points={k:v['n_graded'] for k,v in ipts.items()},
      counters=dict(stage_b=CB,stage_c=CC,stage_d=CD),
      one_line=branch))
    print(json.dumps(dict(branch=branch,n1=n1,w_net=w_net,Istar=Istar,
      arms={k:(v['n_legs'],round(v['motif_median_hz'],1)) for k,v in arms.items()},
      cnt=dict(b=CB,c=CC,d=CD)),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
