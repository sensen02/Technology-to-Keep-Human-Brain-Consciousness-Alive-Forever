"""S320 completion: D3/D4 arms plus assembly (reuses the completed stage B/C archives)."""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1')
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
        PP=np.abs(np.fft.rfft(yy))**2;q=PP[sel].sum()/max(PP[(f>=0.3)&(f<=50)].sum(),1e-30)
        if q>=conc: c+=1
    p=(c+1)/(nulls+1)
    return dict(rhythmic=bool(p<0.01 and conc>=0.15 and not (spb<=1.5 and 0.8<=pk/max(rate,1e-9)<=1.2)),rate_hz=rate,nb=int(nb),peak_hz=pk,conc=conc,p_null=p)
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules'];edges=json.loads((S307/'PER_LEG_MODULE.json').read_text())['edges']
    gw=json.loads((o/'GRADED_WINDOW.json').read_text());ign=json.loads((o/'IGNITION_AUDIT.json').read_text())
    ipts=gw['points'];w_net=2.5e-5
    cand=[v['I_star'] for k,v in ipts.items() if k.startswith(str(w_net)) and v['I_star']]
    Istar=float(np.median(cand)) if cand else 0.03
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    import dataclasses
    CD=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,banc_load=0,physics_step=0,model_load=0,render=0)
    rng=np.random.default_rng(20261026)
    ref=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    def run(name,w,I,sd,mode):
        cfg=TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=w,background_seed=0)
        if hasattr(cfg,'background_sd_nA'): cfg=dataclasses.replace(cfg,background_sd_nA=sd)
        b=build_neural_tier(cfg.validate());CD['tier_build']+=1;CD['select_touch_tier']+=2;CD['edges_build']+=1;CD['banc_load']+=1
        e=b['edges'];We,Wi=(e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi);audit=dict(requested=0,applied=0)
        if mode in ('zero','uncapped'):
            for lg in LEGS:
                for src_r,tgt_r in (('E1','E2'),('E2','E1'),('E2','I1'),('E1','I1'),('I1','E1'),('I1','E2')):
                    cnt=float(edges[lg][f'{src_r}->{tgt_r}'])
                    for csrc in mod[lg][src_r]:
                        for ctgt in mod[lg][tgt_r]:
                            i=loc[int(csrc)];j=loc[int(ctgt)]
                            for Wm in (We,Wi):
                                s0,e0=Wm.indptr[j],Wm.indptr[j+1];k=np.flatnonzero(Wm.indices[s0:e0]==i)
                                if k.size:
                                    audit['requested']+=1
                                    Wm.data[s0+k[0]]=0.0 if mode=='zero' else cnt*w
                                    audit['applied']+=1
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',weight_scale_uS_per_synapse=w,background_seed=0).validate(),OwnershipLedger());CD['network_init']+=1
        n=25000 if name=='G0_gate' else 5000
        spm=np.zeros((n,len(roots)),bool);cur=np.full(len(roots),I)
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,cur),dtype=bool).any(axis=0)
            CD['control_tick']+=1;CD['tier_advance']+=1;CD['neural_substep']+=2
        legs={};rates={}
        for lg in LEGS:
            cols=[loc[int(c)] for c in mod[lg]['E1']]
            e1=spm[:,cols].any(axis=1).astype(float)
            legs[lg]=v8(e1,1.0,rng);rates[lg]=float(spm[:,cols].sum()/(n/1000.0))
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape))
        return dict(legs=legs,n_legs=sum(1 for lg in LEGS if legs[lg]['rhythmic']),
          motif_median_hz=float(np.median(list(rates.values()))),duty=float(spm.mean()),audit=audit,
          gate_bitwise=(bool(np.array_equal(spm[:5000],ref)) if name=='G0_gate' else None))
    arms={}
    arms['G0_gate']=run('G0_gate',5e-4,0.0,0.004,None)
    arms['D1']=run('D1',w_net,Istar,0.004,None)
    arms['D2']=run('D2',w_net,Istar,0.004,'zero')
    arms['D3']=run('D3',w_net,Istar,0.004,'uncapped')
    arms['D4']=run('D4',w_net,Istar,0.0,None)
    dump(o/'ARMS.json',arms)
    dump(o/'CALL_LEDGER.json',dict(stage_b=ign['counters'],stage_c=gw['counters'],stage_d=CD))
    d1=arms['D1'];d2=arms['D2'];ch=abs(d1['motif_median_hz']-d2['motif_median_hz'])/max(d2['motif_median_hz'],1e-9)
    n1=bool(any(r['spikes']==0 for r in ign['grid']['0.0005|0.003']) and any(r['spikes']==0 for r in ign['grid']['0.0005|0.004'])
            and any(r['spikes']>0 for r in ign['grid']['0.0005|0.004']) and any(v['n_graded']>0 for v in ipts.values()))
    n2=bool(n1 and d1['n_legs']>=1 and ch>=0.10);n3=bool(n1 and d1['n_legs']>=1 and not n2);n4=bool(n1 and d1['n_legs']==0)
    branch='n2' if n2 else ('n3' if n3 else ('n4' if n4 else ('n1' if n1 else 'n4')))
    dump(o/'OUTCOMES.json',dict(round='S320',branch=branch,n1=n1,n2=n2,n3=n3,n4=n4,w_net=w_net,I_star=Istar,
      arms={k:dict(n_legs=v['n_legs'],motif_median_hz=round(v['motif_median_hz'],1),duty=round(v['duty'],4),
        peaks={lg:(round(v['legs'][lg].get('peak_hz') or 0,2)) for lg in LEGS}) for k,v in arms.items()},
      pool_change_D1_vs_D2=ch,
      ignition={k:dict(n_active=sum(1 for r in v if r['spikes']>0),n=len(v),
        medians=[round(r['rate_hz'],1) for r in v]) for k,v in ign['grid'].items() if k.startswith('0.0005') or k.startswith('1.5e-05')},
      graded_points={k:v['n_graded'] for k,v in ipts.items()},
      counters=dict(stage_b=ign['counters'],stage_c=gw['counters'],stage_d=CD),one_line=branch))
    print(json.dumps(dict(branch=branch,n1=n1,Istar=Istar,ch=round(ch,3),
      arms={k:(v['n_legs'],round(v['motif_median_hz'],1)) for k,v in arms.items()},cd=CD),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
