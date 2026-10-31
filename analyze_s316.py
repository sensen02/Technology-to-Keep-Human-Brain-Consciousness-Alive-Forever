"""S316: per-cell mapping repair (dict-keyed), eight role-conditioned arms."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S314=Path('/home/sensen/Desktop/engine修复验收/s314-tier-leg-burst-scale/20261026T212103Z')
S315=Path('/home/sensen/Desktop/engine修复验收/s315-percell-drive-derivation/20261026T212824Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1');FP='56e0322a33c51b7f03aaae9f7df1ef3f';BAND=(6.58,26.92)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
C=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
def bursts(x,dt=1.0):
    i=np.flatnonzero(x)
    if len(i)<3: return np.array([],int),np.array([],int)
    isi=np.diff(i)*dt;thr=3*np.median(isi);st=[i[0]];cn=[1]
    for k,d in enumerate(isi):
        if d>thr: st.append(i[k+1]);cn.append(1)
        else: cn[-1]+=1
    return np.asarray(st),np.asarray(cn)
def v5(e1,i1,dt=1.0):
    st,cn=bursts(e1,dt);n=len(st);o=dict(n_bursts=int(n),mean_hz=float(e1.mean()*1000.0/dt))
    if n<11: o.update(rhythmic=False,reason='few_bursts');return o
    per=np.diff(st)*dt;cv=float(np.std(per)/max(float(np.mean(per)),1e-9));spc=float(cn[1:-1].mean())
    si,_=bursts(i1,dt)
    if len(si)<3:
        t=np.flatnonzero(i1);si=[]
        for s0 in st:
            k=np.searchsorted(t,s0)
            if k<len(t): si.append(t[k])
        si=np.asarray(si,int)
    lag=None;p=1.0
    if len(si)>=3:
        L=[]
        for s0 in st:
            k=np.searchsorted(si,s0)
            if k<len(si): L.append(((si[k]-s0)*dt/np.mean(per))%1.0)
        if L:
            lag=float(np.median(L));z=np.exp(2j*np.pi*np.asarray(L));R=float(abs(z.mean()));nn=len(L)
            p=float(np.exp(np.sqrt(1+4*nn+4*(nn**2-(nn*R)**2))-(1+2*nn))) if nn>1 and R<1 else 0.0
    f=float(1000.0/np.mean(per))
    o.update(rhythmic=bool(len(per)>=10 and cv<0.25 and lag is not None and 0.0<lag<0.25 and p<0.01 and 1.0<=spc<=12.0 and 5.0<=o['mean_hz']<=300.0),
             freq_hz=f,cv=cv,spikes_per_cycle=spc,lag=lag,rayleigh_p=p,band_ok=bool(BAND[0]<=f<=BAND[1]))
    return o
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    import sys;sys.path.insert(0,str(ROOT))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    ident={}
    for lg in LEGS:
        for a in ORDER:
            for c in mod[lg][a]: ident[int(c)]=dict(leg=lg,role=a,global_index=loc[int(c)])
    GI2={v['global_index']:k for k,v in ident.items()}
    g2local={v['global_index']:v['global_index'] if False else loc[k] for k,v in ident.items()}
    def L(gi): return g2local[gi]
    assert len(ident)==18 and len(GI2)==18
    # ---- stage A: G_j and s_j keyed by global_index, from spikes
    sp=np.load(S309/'SPIKES_k1_control.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    W=np.load(S303/'EDGES_w_5e-4_We.npz');I=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    mset=set(GI2)
    net={}
    for gi in GI2:
        row=np.zeros(spk.shape[0])
        for arr,sgn in ((W,1.0),(I,-1.0)):
            j=gi;s,e=arr['indptr'][j],arr['indptr'][j+1];src=arr['indices'][s:e];dat=np.asarray(arr['data'][s:e],float)
            mask=np.array([int(x) in mset for x in src])
            if (~mask).any(): row+=sgn*(spk[:,src[~mask]].astype(float)@dat[~mask])
        net[gi]=row
    ker=np.ones(20)/20
    G={gi:float(np.median(np.convolve(net[gi],ker,mode='same'))) for gi in GI2}
    s={gi:(0.03/(65.0*G[gi])) if G[gi]>0 else None for gi in GI2}
    rule=[]
    for gi in sorted(GI2):
        g=G[gi];sj=s[gi]
        rule.append(dict(global_index=int(gi),leg=ident[GI2[gi]]['leg'],role=ident[GI2[gi]]['role'],G_uS=g,s_j=sj,
                         predicted_drive_nA=(g*sj*65.0 if sj else None),
                         disposition=('zero_external' if g<=0 else 'scale')))
    dump(o/'DRIVE_RULE_V3.json',dict(rows=rule,rule='s_j = 0.03 nA / (65 mV x G_j) keyed by global_index',
      rf_E2_two_options=[dict(option='keep_unchanged',s_j=1.0),dict(option='zero_external',s_j=0.0)]))
    # S315 correction table
    old=json.loads((S315/'DRIVE_RULE_V2.json').read_text())
    entries_order=[]  # S315 iterated entries in (leg,role) order
    for lg in LEGS:
        for a in ORDER:
            for c in mod[lg][a]: entries_order.append(loc[int(c)])
    hetS315=json.loads((S315/'HETEROGENEITY.json').read_text())['per_cell']
    table=[]
    for idx,gi in enumerate(entries_order):
        s_own=s[gi];s_app=(hetS315[idx]['s_j'] if hetS315[idx]['s_j'] is not None else 1.0)
        table.append(dict(leg=ident[GI2[gi]]['leg'],role=ident[GI2[gi]]['role'],global_index=int(gi),G_own_uS=G[gi],
          s_own=s_own,s_applied_s315=s_app,actual_drive_nA=(G[gi]*s_app*65.0),ratio_to_target=((G[gi]*s_app*65.0)/0.03 if G[gi]>0 else None)))
    bad=sum(1 for r in table if r['ratio_to_target'] is not None and abs(r['ratio_to_target']-1)>0.05)
    dump(o/'S315_CORRECTION.json',dict(
      table=table,n_cells_off_by_5pct=int(bad),drive_range=[min(r['actual_drive_nA'] for r in table),max(r['actual_drive_nA'] for r in table)],
      defect='analyze_s315.py built s_map with zip(entries, het) where het follows M18=sorted(global_index) and entries follows (leg,role); 17 of 18 cells received another cell\'s scale factor',
      counts_note='audit scaled recounted: 1587 = 1553 external in-edges + 34 motif internal edges; percell arms were recorded as 1587 rather than the 1553 actually scaled',
      heterogeneity_corrected=dict(min=min(G.values()),max=max(G.values()),ratio=float(max(G.values())/max(min(G.values()),1e-9))),
      indptr_axis='ARCHIVED indptr is CSR over POSTSYNAPTIC rows and indices are presynaptic sources; verified against the live CSR (row slices 144/69, 47/35, 44/16 and nnz 293473/148811). The earlier note that the archive is CSC is WRONG and is corrected here.',
      branches_y=['y1','y2','y3'],branches_status='all NOT_JUDGED because the per-cell mapping was positional'))
    dump(o/'TARGET_JUSTIFICATION.json',dict(target_nA=0.03,band=[0.0165,0.05],
      band_provenance='the isolated band was measured with constant current into E1 ONLY (S312 run injects only the six E1 cells), so equalising all three roles to 0.03 nA is a declarative intervention that conflicts with the source (only E1 receives descending input)',
      source_faithful_arm='P3: scale the six E1 external inputs; zero the external input of the twelve E2/I1 cells'))
    dump(o/'DETECTOR_V5_RESTATED.json',dict(criteria='V5 unchanged: per-leg bursts>=11, cv<0.25, spikes/cycle 1-12, mean 5-300 Hz, I1 burst-start lag in (0,0.25) with Rayleigh p<0.01',band_hz=list(BAND)))
    pre=dict(round='S316',arms=['G0_gate','P1_percell_all','P2_percell_all_zero_nonpos','P3_source_faithful','P4_e1_only_scaled','P5_role_coupled','P6_ablation_no_I1E1','P7_ablation_zero_motif'],
      hard_rule='ALL per-cell parameters must be keyed by global_index in a dict and archived as (global_index, value) pairs; positional zip is FORBIDDEN; assert len(set(keys))==18',
      counters=dict(normal=dict(tier_build=8,select_touch_tier=16,edges_build=8,network_init=8,control_tick=40000,tier_advance=40000,neural_substep=80000,physics_step=0,model_load=0,render=0),
                    gate_fail=dict(tier_build=1,select_touch_tier=2,edges_build=1,network_init=1,control_tick=5000,tier_advance=5000,neural_substep=10000)),
      s_dict={str(k):v for k,v in s.items()},
      branches=dict(z1='P3 >=5/6 legs rhythmic in band and P6,P7 both destroy the rhythm',
                    z2='P3 1-4/6',z3='P3 none but P1/P2 some',z4='all eight arms none'))
    dump(o/'PREREG.json',pre);(o/'S316_PREREG_FROZEN').write_text('frozen before any dynamic call\n')
    # ---- stage B
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    def edges(b):
        e=b['edges'];return (e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    def scale_cells(We,Wi,sd,z,zero=False):
        touches=[]
        for gi,sj in sd.items():
            j=gi
            for Wm,kd in ((We,'exc'),(Wi,'inh')):
                s,e=Wm.indptr[j],Wm.indptr[j+1]
                bef=float(np.sum([Wm.data[t] for t in range(s,e) if int(Wm.indices[t]) not in mset]))
                for t in range(s,e):
                    if int(Wm.indices[t]) in mset: continue
                    if zero: Wm.data[t]=0.0
                    else: Wm.data[t]*=sj
                aft=float(np.sum([Wm.data[t] for t in range(s,e) if int(Wm.indices[t]) not in mset]))
                touches.append(dict(global_index=int(gi),kind=kd,before_uS=bef,after_uS=aft,s_applied=sj))
        return touches
    def zero_edges(We,Wi,edges_list):
        n=0
        for (post,pre) in edges_list:
            j=post;i=pre
            for Wm in (We,Wi):
                s,e=Wm.indptr[j],Wm.indptr[j+1]
                for t in range(s,e):
                    if int(Wm.indices[t])==i: Wm.data[t]=0.0;n+=1
        return n
    E1g=[gi for gi in GI2 if ident[GI2[gi]]['role']=='E1'];OTH=[gi for gi in GI2 if ident[GI2[gi]]['role']!='E1']
    legE1={lg:next(gi for gi in E1g if ident[GI2[gi]]['leg']==lg) for lg in LEGS}
    refspk=np.unpackbits(np.load(S309/'SPIKES_k1_control.npz')['packed'],axis=1)[:,:12000].astype(bool)
    arm_defs=[
      ('G0_gate',None),('P1_percell_all',('all',False)),('P2_percell_all_zero_nonpos',('all',True)),
      ('P3_source_faithful',('src',False)),('P4_e1_only_scaled',('e1',False)),('P5_role_coupled',('coupled',False)),
      ('P6_ablation_no_I1E1',('src_noI1E1',False)),('P7_ablation_zero_motif',('src_zeromotif',False))]
    res={};ledger={}
    for name,spec in arm_defs:
        b=build_neural_tier(TierConfig(sign_fill='all_empty').validate());C['tier_build']+=1;C['select_touch_tier']+=2;C['edges_build']+=1
        We,Wi=edges(b);touch=[]
        if spec:
            mode,zeroNP=spec
            if mode=='all':
                sd={gi:(1.0 if s[gi] is None else s[gi]) for gi in GI2}
                touch=scale_cells(We,Wi,sd,None,zero=False)
                if zeroNP:
                    touch+=scale_cells(We,Wi,{gi:1.0 for gi in GI2 if s[gi] is None},None,zero=True)
            elif mode=='src':
                sd={gi:s[gi] for gi in E1g}
                touch=scale_cells(We,Wi,sd,None,zero=False)
                touch+=scale_cells(We,Wi,{gi:1.0 for gi in OTH},None,zero=True)
            elif mode=='e1':
                touch=scale_cells(We,Wi,{gi:s[gi] for gi in E1g},None,zero=False)
            elif mode=='coupled':
                touch=scale_cells(We,Wi,{gi:s[gi] for gi in E1g},None,zero=False)
                touch+=scale_cells(We,Wi,{gi:s[legE1[ident[GI2[gi]]['leg']]] for gi in OTH},None,zero=False)
            elif mode=='src_noI1E1':
                touch=scale_cells(We,Wi,{gi:s[gi] for gi in E1g},None,zero=False)
                touch+=scale_cells(We,Wi,{gi:1.0 for gi in OTH},None,zero=True)
                n=0
                for lg in LEGS:
                    e1=legE1[lg]
                    i1=[gi for gi in OTH if ident[GI2[gi]]['leg']==lg and ident[GI2[gi]]['role']=='I1'][0]
                    n+=zero_edges(We,Wi,[(e1,i1)])
                touch.append(dict(kind='ablation_noI1E1',removed=n))
            elif mode=='src_zeromotif':
                sd={gi:s[gi] for gi in E1g}
                touch=scale_cells(We,Wi,sd,None,zero=False)
                touch+=scale_cells(We,Wi,{gi:1.0 for gi in OTH},None,zero=True)
                n=0
                for gi in GI2:
                    for Wm in (We,Wi):
                        s,e=Wm.indptr[gi],Wm.indptr[gi+1]
                        for t in range(s,e):
                            if int(Wm.indices[t]) in mset: Wm.data[t]=0.0;n+=1
                touch.append(dict(kind='ablation_zero_motif',removed=n))
        nonneg=bool(np.all(We.data>=0) and np.all(Wi.data>=0))
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());C['network_init']+=1
        hidden=getattr(tier,'net',tier)
        n=5000;spm=np.zeros((n,len(roots)),bool);Vm=np.zeros((n,18),np.float32)
        idxmap=[g2local[g] for g in sorted(GI2)]
        for k in range(n):
            spm[k]=np.asarray(tier.advance(2,np.zeros(len(roots))),dtype=bool).any(axis=0)
            C['control_tick']+=1;C['tier_advance']+=1;C['neural_substep']+=2
            v=np.asarray(getattr(hidden,'v',np.zeros(1)),float)
            if v.size>=len(roots): Vm[k]=v[[g2local[g] for g in sorted(GI2)]]
        per={}
        for lg in LEGS:
            e1=spm[:,g2local[legE1[lg]]];i1=spm[:,[g2local[gi] for gi in OTH if ident[GI2[gi]]['leg']==lg and ident[GI2[gi]]['role']=='I1'][0]]
            r=v5(e1,i1);r['e1_hz']=float(e1.sum()/5.0);per[lg]=r
        res[name]=dict(n_rh=sum(1 for lg in LEGS if per[lg]['rhythmic']),n_band=sum(1 for lg in LEGS if per[lg].get('band_ok')),
          per_leg={lg:{k:v for k,v in per[lg].items() if k in ('rhythmic','freq_hz','n_bursts','cv','e1_hz','band_ok')} for lg in LEGS},
          duty=float(spm.mean()),nonneg=nonneg,gate_bitwise=bool(np.array_equal(spm,refspk)) if name=='G0_gate' else None)
        ledger[name]=touch
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape),Vm=Vm,index_order=np.array(sorted(GI2)))
    dump(o/'ARMS.json',res);dump(o/'MUTATION_LEDGER.json',ledger);dump(o/'CALL_LEDGER.json',C)
    p3=res['P3_source_faithful'];p6=res['P6_ablation_no_I1E1'];p7=res['P7_ablation_zero_motif']
    def destroys(a): return all((not a['per_leg'][lg]['rhythmic']) or (a['per_leg'][lg]['n_bursts']<11) for lg in LEGS)
    z1=bool(p3['n_rh']>=5 and destroys(p6) and destroys(p7))
    z2=1<=p3['n_rh']<=4;z3=(p3['n_rh']==0 and (res['P1_percell_all']['n_rh']>0 or res['P2_percell_all_zero_nonpos']['n_rh']>0))
    z4=all(res[k]['n_rh']==0 for k in res)
    branch='z1' if z1 else ('z2' if z2 else ('z3' if z3 else ('z4' if z4 else 'z4')))
    dump(o/'OUTCOMES.json',dict(round='S316',branch=branch,arms={k:dict(n_rh=v['n_rh'],n_band=v['n_band'],duty=round(v['duty'],4),
      freqs={lg:(round(v['per_leg'][lg].get('freq_hz') or 0,2)) for lg in LEGS},e1_hz={lg:round(v['per_leg'][lg]['e1_hz'],1) for lg in LEGS}) for k,v in res.items()},
      gate_bitwise=res['G0_gate']['gate_bitwise'],counters=C,
      one_line=('z%d'%int(branch[1]))))
    print(json.dumps(dict(branch=branch,n_rh={k:v['n_rh'] for k,v in res.items()},gate=res['G0_gate']['gate_bitwise'],counters=C),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
