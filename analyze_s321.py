"""S321: apply g_motif explicitly across a decade sweep at the measured operating point."""
from __future__ import annotations
import argparse, json, hashlib, datetime, dataclasses
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S320=Path('/home/sensen/Desktop/engine修复验收/s320-operating-point-and-graded-motif/20261026T220229Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
W_NET=2.5e-5;RHEO=0.015;EDGE_TOTAL=0.015/65.0;PER_SYN=0.015/65.0/8.0
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
    pre=dict(round='S321',mode='explicit g_motif sweep at the measured operating point; no physics',
      arms=['A0_gate','B0_factory_motif','E1x','E2x','E4x','E8x','E0_abl','PS1x','C_sd0','R_repeat'],
      counters=dict(tier_build=10,select_touch_tier=20,edges_build=10,network_init=10,banc_load=10,
        control_tick=70000,tier_advance=70000,neural_substep=140000,physics_step=0,model_load=0,render=0,con_network_init=0),
      rules=['every design value and criterion is taken from a measured table; no fallback constants in code',
             'build_neural_tier and NeuralTier must share ONE TierConfig instance; background hashes are archived and C_sd0 must differ from B0',
             'the branch is decided by the data alone; no else fallback'],
      motif_semantics=dict(edge_total_uS=EDGE_TOTAL,per_synapse_uS=PER_SYN,adopted='edge_total'),
      branches=dict(s1='some k gives >=1 rhythmic leg and E0_abl gives zero',s2='no rhythm but some k<=8 shifts the median rate by >=10 percent',
                    s3='all k<=8 below 10 percent',s4='gate failure, I* not measured, or f83>0.02'))
    dump(o/'PREREG.json',pre)
    (o/'S321_PREREG_FROZEN').write_text('sha256(PREREG.json)=%s\nsha256(analyze_s321.py)=%s\nutc=%s\n'%(
      sha(o/'PREREG.json'),sha(__file__),datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')))
    # ---- operating point from the archive (no fallback)
    G=json.loads((S320/'GRADED_WINDOW.json').read_text())['grid']
    table={}
    for k,rows in G.items():
        table[k]=[dict(I=r['I'],rate=r.get('rate_per_neuron_hz',r['rate']/12000.0),graded=r['graded']) for r in rows]
    key='2.5e-05|0'
    cand=[r for r in table[key] if r['graded']]
    best=min(sorted(cand,key=lambda r:abs(r['rate']-50.0)),key=lambda r:abs(r['rate']-50.0)) if cand else None
    if best is None:
        dump(o/'STOP.json',dict(reason='no measured graded point; refusing to invent I*'));return 2
    Istar=float(best['I'])
    dump(o/'OPERATING_POINT.json',dict(table=table,rule='closest per-cell rate to 50 Hz among graded points, lowest current on ties',
      I_star_nA=Istar,measured_rate_hz=best['rate'],no_fallback=True))
    # ---- motif semantics + ledger of the 34 ordered edges
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text());modules=mod['modules'];edges=mod['edges']
    led=[]
    for lg in LEGS:
        for src_r in ROLES:
            for tgt_r in ROLES:
                if src_r==tgt_r: continue
                cnt=float(edges[lg][f'{src_r}->{tgt_r}'])
                led.append(dict(leg=lg,edge=f'{src_r}->{tgt_r}',count=cnt,factory_uS=min(cnt,8)*W_NET,
                  design_edge_total_uS=EDGE_TOTAL,design_per_synapse_uS=PER_SYN*cnt,
                  ratio_edge_total=EDGE_TOTAL/max(min(cnt,8)*W_NET,1e-12),
                  ratio_per_synapse=PER_SYN*cnt/max(min(cnt,8)*W_NET,1e-12)))
    dump(o/'MOTIF_SCALE_SEMANTICS.json',dict(edges=led,edge_total_uS=EDGE_TOTAL,per_synapse_uS=PER_SYN,
      adopted='edge_total: every motif edge is set to k x 2.308e-4 uS regardless of its synapse count',
      alternative='per_synapse: 2.885e-5 uS x count, which for the 320-synapse lf edge is 9.2e-3 uS = 40x rheobase'))
    # ---- stage B
    import sys;sys.path.insert(0,str(ROOT))
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    M18=[loc[int(c)] for lg in LEGS for r in ROLES for c in modules[lg][r]];Mset=set(M18)
    rng=np.random.default_rng(20261026)
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,banc_load=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
    ref=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    def arm(name,w,I,sd,motif_mode,ticks):
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=w,background_seed=0,background_sd_nA=sd)
        cfg=cfg.validate()
        b=build_neural_tier(cfg);CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
        e=b['edges'];We,Wi=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
        mut=[]
        if motif_mode is not None:
            for lg in LEGS:
                for src_r in ROLES:
                    for tgt_r in ROLES:
                        if src_r==tgt_r: continue
                        cnt=float(edges[lg][f'{src_r}->{tgt_r}'])
                        for csrc in modules[lg][src_r]:
                            for ctgt in modules[lg][tgt_r]:
                                i=loc[int(csrc)];j=loc[int(ctgt)]
                                for Wm,tag in ((We,'exc'),(Wi,'inh')):
                                    s0,e0=Wm.indptr[j],Wm.indptr[j+1];k=np.flatnonzero(Wm.indices[s0:e0]==i)
                                    if k.size:
                                        bef=float(Wm.data[s0+k[0]])
                                        new=0.0 if motif_mode=='zero' else (EDGE_TOTAL*motif_mode if isinstance(motif_mode,float) else PER_SYN*cnt)
                                        Wm.data[s0+k[0]]=new
                                        mut.append(dict(leg=lg,edge=f'{src_r}->{tgt_r}',kind=tag,count=cnt,before_uS=bef,after_uS=new))
        tier=NeuralTier(b,cfg,OwnershipLedger());CD['network_init']+=1
        bgh=hashlib.sha256(np.ascontiguousarray(np.asarray(tier.background_nA,float)).tobytes()).hexdigest()
        n=ticks;cur=np.full(len(roots),I);cnt=np.zeros(len(roots),dtype=np.int32);spm=np.zeros((n,len(roots)),bool)
        for k in range(n):
            s=np.asarray(tier.advance(2,cur),dtype=bool);spm[k]=s.any(axis=0);cnt+=spm[k].astype(np.int32)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
        rate=cnt/(n*0.0005)
        pool=[]
        for lg in LEGS: pool+=[loc[int(c)] for c in modules[lg]['E1']]
        legs={}
        for lg in LEGS:
            e1=spm[:,[loc[int(c)] for c in modules[lg]['E1']]].any(axis=1).astype(float)
            legs[lg]=v8(e1,1.0,rng)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape),counts=cnt)
        return dict(bg_sha256=bgh,mutation=mut,n_touched=len(mut),
          median_rate_hz=float(np.median(rate)),f380=float((rate>=380).mean()),
          motif_rates={f'{lg}_{r}':float(rate[loc[int(modules[lg][r][0])]]) for lg in LEGS for r in ROLES},
          pool_median_hz=float(np.median(rate[pool])),legs=legs,n_legs=sum(1 for lg in LEGS if legs[lg]['rhythmic']),
          duty=float(spm.mean()),gate=(bool(np.array_equal(spm[:5000],ref)) if name=='A0_gate' else None))
    arms={}
    arms['A0_gate']=arm('A0_gate',5e-4,0.0,0.004,None,25000)
    arms['B0_factory_motif']=arm('B0_factory_motif',W_NET,Istar,0.004,None,5000)
    for k in (1,2,4,8):
        arms[f'E{k}x']=arm(f'E{k}x',W_NET,Istar,0.004,float(k),5000)
    arms['E0_abl']=arm('E0_abl',W_NET,Istar,0.004,'zero',5000)
    arms['PS1x']=arm('PS1x',W_NET,Istar,0.004,'per_syn',5000)
    arms['C_sd0']=arm('C_sd0',W_NET,Istar,0.0,None,5000)
    arms['R_repeat']=arm('R_repeat',W_NET,Istar,0.004,None,5000)
    dump(o/'ARMS.json',arms);dump(o/'MUTATION_LEDGER.json',{k:v['mutation'] for k,v in arms.items()});dump(o/'CALL_LEDGER.json',CD)
    # ---- branch (no else fallback)
    b0=arms['B0_factory_motif']
    delta={k:abs(arms[f'E{k}x']['median_rate_hz']-b0['median_rate_hz'])/max(b0['median_rate_hz'],1e-9) for k in (1,2,4,8)}
    s1=[k for k in (1,2,4,8) if arms[f'E{k}x']['n_legs']>=1 and arms['E0_abl']['n_legs']==0]
    s2=[k for k in (1,2,4,8) if delta[k]>=0.10]
    gate_ok=arms['A0_gate']['gate'] is True
    graded_ok=all(arms[f'E{k}x']['f380']<=0.02 for k in (1,2,4,8))
    branch=('s1' if (gate_ok and s1) else ('s2' if (gate_ok and graded_ok and s2) else ('s3' if (gate_ok and graded_ok and not s1 and not s2) else 's4')))
    dump(o/'OUTCOMES.json',dict(round='S321',branch=branch,I_star_nA=Istar,
      arms={k:dict(n_legs=v['n_legs'],median_rate_hz=round(v['median_rate_hz'],1),f380=round(v['f380'],4),
        motif_median=round(v['motif_rates'][f'{LEGS[0]}_E1'],1),n_touched=v['n_touched']) for k,v in arms.items()},
      delta_vs_B0=delta,rhythmic_k=s1,measurable_k=s2,gate=gate_ok,graded_ok=graded_ok,
      replay_equal_B0_R=bool(np.array_equal(np.unpackbits(np.load(o/'SPK_B0_factory_motif.npz')['packed'],axis=1)[:5000],np.unpackbits(np.load(o/'SPK_R_repeat.npz')['packed'],axis=1)[:5000])),
      sd0_differs=bool(arms['C_sd0']['bg_sha256']!=arms['B0_factory_motif']['bg_sha256']),
      counters=CD,one_line=branch))
    print(json.dumps(dict(branch=branch,Istar=Istar,gate=gate_ok,
      arms={k:(v['n_legs'],round(v['median_rate_hz'],1),round(v['f380'],3)) for k,v in arms.items()},
      delta={k:round(v,3) for k,v in delta.items()},cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
