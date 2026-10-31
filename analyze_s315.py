"""S315: corrected input split, per-cell drive derivation, five arms with ablation causality."""
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
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1');FP='56e0322a33c51b7f03aaae9f7df1ef3f'
BAND=(6.58,26.92);DRIVE_TARGET=0.03
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
C=dict(tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
def bursts(x,dt=1.0):
    idx=np.flatnonzero(x)
    if len(idx)<3: return np.array([],int),np.array([],int)
    isi=np.diff(idx)*dt;thr=3*np.median(isi);st=[idx[0]];cnt=[1]
    for i,d in enumerate(isi):
        if d>thr: st.append(idx[i+1]);cnt.append(1)
        else: cnt[-1]+=1
    return np.asarray(st),np.asarray(cnt)
def v5(e1,i1,dt=1.0):
    st,cnt=bursts(e1,dt);n=len(st);out=dict(n_bursts=int(n),mean_hz=float(e1.mean()*1000.0/dt))
    if n<11: out.update(rhythmic=False,reason='few_bursts');return out
    per=np.diff(st)*dt;cv=float(np.std(per)/max(float(np.mean(per)),1e-9));spc=float(cnt[1:-1].mean())
    si,_=bursts(i1,dt);fb=False
    if len(si)<3:
        i1t=np.flatnonzero(i1);si=[]
        for s0 in st:
            k=np.searchsorted(i1t,s0)
            if k<len(i1t): si.append(i1t[k])
        si=np.asarray(si,int);fb=True
    lag=None;p=1.0
    if len(si)>=3:
        lags=[]
        for s0 in st:
            k=np.searchsorted(si,s0)
            if k<len(si): lags.append(((si[k]-s0)*dt/np.mean(per))%1.0)
        if lags:
            lag=float(np.median(lags));z=np.exp(2j*np.pi*np.asarray(lags));R=float(abs(z.mean()));nn=len(lags)
            p=float(np.exp(np.sqrt(1+4*nn+4*(nn**2-(nn*R)**2))-(1+2*nn))) if nn>1 and R<1 else 0.0
    f=float(1000.0/np.mean(per))
    rh=bool(len(per)>=10 and cv<0.25 and lag is not None and 0.0<lag<0.25 and p<0.01 and 1.0<=spc<=12.0 and 5.0<=out['mean_hz']<=300.0)
    out.update(rhythmic=rh,freq_hz=f,cv=cv,spikes_per_cycle=spc,lag=lag,rayleigh_p=p,i1_fallback=fb,
               band_ok=bool(BAND[0]<=f<=BAND[1]))
    return out
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    import sys;sys.path.insert(0,str(ROOT))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    entries=[]
    for lg in LEGS:
        for a in ORDER:
            for c in mod[lg][a]:
                entries.append(dict(leg=lg,role=a,root=int(c),global_index=int(loc[int(c)])))
    M18=sorted(e['global_index'] for e in entries);mset=set(M18)
    dump(o/'M18_GLOBAL_INDEX.json',dict(M18=M18,entries=entries))
    # ---- stage A: corrected split
    sp=np.load(S309/'SPIKES_k1_control.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    W=np.load(S303/'EDGES_w_5e-4_We.npz');I=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    n=spk.shape[0];ne=len(M18)
    exc=np.zeros((n,ne));inh=np.zeros((n,ne));inn=np.zeros((n,ne))
    rng_ok=dict(internal_min_ok=True,internal_max_ok=True)
    internal_bounds={}
    for jj,j in enumerate(M18):
        lo=hi=0.0
        for arr,sgn,key in ((W,1.0,'exc'),(I,1.0,'inh')):
            s,e=arr['indptr'][j],arr['indptr'][j+1];src=arr['indices'][s:e];dat=np.asarray(arr['data'][s:e],float)
            mask=np.array([int(x) in mset for x in src])
            if (~mask).any():
                v=spk[:,src[~mask]].astype(float)@dat[~mask]
                (exc if key=='exc' else inh)[:,jj]+=v
            if mask.any():
                v=spk[:,src[mask]].astype(float)@dat[mask]
                inn[:,jj]+= (v if key=='exc' else -v)   # internal is SIGNED: exc minus inh
                if key=='exc': hi+=float(dat[mask].sum())
                else: lo-=float(dat[mask].sum())
        internal_bounds[jj]=dict(lo=lo,hi=hi,min=float(inn[:,jj].min()),max=float(inn[:,jj].max()))
        rng_ok['internal_min_ok']&=bool(inn[:,jj].min()>=lo-1e-12)
        rng_ok['internal_max_ok']&=bool(inn[:,jj].max()<=hi+1e-12)
    net=exc-inh
    ti=np.load(S313/'TSPK_1.npz')['net_in']
    split=dict(exc_q={k:float(np.percentile(exc,p)) for k,p in (('p10',10),('p50',50),('p90',90))},
      inh_q={k:float(np.percentile(inh,p)) for k,p in (('p10',10),('p50',50),('p90',90))},
      net_q={k:float(np.percentile(net,p)) for k,p in (('p10',10),('p50',50),('p90',90))},
      internal_q={k:float(np.percentile(inn,p)) for k,p in (('p10',10),('p50',50),('p90',90))},
      old_s314=dict(exc_q=None,internal_q=json.loads((S314/'EXTERNAL_INPUT_SPLIT.json').read_text())['int_q']),
      identity_ok=bool(np.allclose(net+inn,ti,atol=1e-9)),max_abs_diff=float(np.abs(net+inn-ti).max()),
      range_assertions=rng_ok,internal_bounds=internal_bounds,
      note='S314 wrote the external INHIBITORY component into the internal term because both loops shared the target array')
    dump(o/'SPLIT_V2.json',split)
    # heterogeneity: 20 ms moving average of net external per cell
    win=20;ker=np.ones(win)/win
    smooth=np.stack([np.convolve(net[:,jj],ker,mode='same') for jj in range(ne)],1)
    G=[float(np.median(smooth[:,jj])) for jj in range(ne)]
    by_gi={e['global_index']:e for e in entries}
    het=[]
    for jj in range(ne):
        e=by_gi[M18[jj]]
        g=G[jj];s=(DRIVE_TARGET/(65.0*g)) if g>0 else None
        prof=[(0.03/(65.0*max(g2,1e-12))) for g2 in (g,)]
        het.append(dict(leg=e['leg'],role=e['role'],global_index=e['global_index'],g_net_uS=g,
          mean_uS=float(smooth[:,jj].mean()),sd_uS=float(smooth[:,jj].std()),
          cv=float(smooth[:,jj].std()/max(abs(smooth[:,jj].mean()),1e-12)),
          s_j=(s if s and np.isfinite(s) and s>0 else None),
          in_band_after_scale=bool(s is not None and 0.0165<=g*s*65.0<=0.05)))
    combined_vs=sorted(h['s_j'] for h in het if h['s_j'])
    dump(o/'HETEROGENEITY.json',dict(per_cell=het,G_spread=dict(min=min(G),max=max(G),ratio=float(max(G)/max(min(G),1e-12))),
      spread_note='per-cell 20 ms mean external drive spans more than an order of magnitude; a global or per-leg scale cannot keep all cells in band',
      unrecoverable=[h for h in het if not h['in_band_after_scale']]))
    dump(o/'DRIVE_RULE_V2.json',dict(formula='s_j = 0.03 nA / (65 mV x G_j)',
      target_nA=DRIVE_TARGET,note='0.03 nA is the geometric midpoint of the isolated band [0.0165,0.05] nA, a proxy from a 3-cell constant-current experiment, NOT a biological threshold for these cells',
      s_j=[h['s_j'] for h in het],s_j_used=[(h['s_j'] if h['s_j'] is not None else 1.0) for h in het],per_cell=het,
      unrecoverable_note='cells with non-positive net external drive keep s_j=1.0 (unchanged); they cannot be brought into band by any positive scale'))
    dump(o/'DETECTOR_V6.json',dict(a='V5 per-leg rhythm',b='leg burst frequency in [6.58,26.92] Hz from S313',
      c='ablation causality: removing I1->E1 must destroy the rhythm (bursts<11 or cv>=0.25)'))
    pre=dict(round='S315',arms=['G1_A1_gate','B1_A4_repeat','B2_percell_v2','B3_percell_v2_no_I1E1','B4_percell_v2_zero_motif'],
      counters=dict(normal=dict(tier_build=5,select_touch_tier=10,edges_build=5,network_init=5,control_tick=25000,tier_advance=25000,neural_substep=50000,physics_step=0,model_load=0,render=0),
                    gate_fail=dict(tier_build=1,select_touch_tier=2,edges_build=1,network_init=1,control_tick=5000,tier_advance=5000,neural_substep=10000)),
      drive_rule='s_j = 0.03 nA / (65 mV x G_j), G_j = per-cell median of the 20 ms moving average of its net external input',
      branches=dict(y1='B2 >=5/6 legs pass all three V6 criteria',y2='B2 1-4/6',y3='B2 0/6 but B4 silent and B3 loses the rhythm',y4='gate or B1 bitwise failure'),
      hard=['verifier writes only verify/','FROZEN before dynamics','npz arrays','no physics/render'])
    dump(o/'PREREG.json',pre);(o/'S315_PREREG_FROZEN').write_text('frozen before any dynamic call\n')
    # ---- stage B
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    half_f=float(json.loads((S314/'SCALE_RULE.json').read_text())['half_f'])
    def edge(obj):
        e=obj['edges'];return (e['We'],e['Wi']) if isinstance(e,dict) else (e.we,e.wi)
    def scale_cols(We,Wi,f,cells,inside_only=False,outside_only=False,zero_motif=False):
        cnt=0
        for Wm in (We,Wi):
            for j in cells:
                s,e=Wm.indptr[j],Wm.indptr[j+1]
                for t in range(s,e):
                    ins=int(Wm.indices[t]) in mset
                    if outside_only and ins: continue
                    if inside_only and not ins: continue
                    if zero_motif and ins: Wm.data[t]=0.0;cnt+=1;continue
                    Wm.data[t]*=f;cnt+=1
        return cnt
    M_local=[loc[e['root']] for e in entries]
    s_map={loc[e['root']]:(h['s_j'] if h['s_j'] is not None else 1.0) for e,h in zip(entries,het)}
    s_used_note='cells whose net external drive is not positive keep s_j = 1.0 (unchanged) and are listed as unrecoverable'
    arms=[('G1_A1_gate',None),('B1_A4_repeat',('a4',half_f)),('B2_percell_v2',('percell',None)),
          ('B3_percell_v2_no_I1E1',('percell_noI1E1',None)),('B4_percell_v2_zero_motif',('percell_zeromotif',None))]
    res={}
    ref=np.load(S309/'SPIKES_k1_control.npz');ref_spk=np.unpackbits(ref['packed'],axis=1)[:,:int(ref['shape'][1])].astype(bool)
    a4=np.load(S314/'SPK_A4_ext_scale_half_f.npz');a4_spk=np.unpackbits(a4['packed'],axis=1)[:,:int(a4['shape'][1])].astype(bool)
    for name,spec in arms:
        b=build_neural_tier(TierConfig(sign_fill='all_empty').validate());C['tier_build']+=1;C['select_touch_tier']+=2;C['edges_build']+=1
        We,Wi=edge(b)
        audit=dict(scaled=0)
        if spec:
            mode,f=spec
            if mode=='a4': audit['scaled']=scale_cols(We,Wi,f,M_local,outside_only=True)
            elif mode=='percell':
                for c in M_local:
                    s,e=We.indptr[c],We.indptr[c+1]
                    sj=s_map[c]
                    for t in range(s,e):
                        if int(We.indices[t]) not in mset: We.data[t]*=sj
                    s2,e2=Wi.indptr[c],Wi.indptr[c+1]
                    for t in range(s2,e2):
                        if int(Wi.indices[t]) not in mset: Wi.data[t]*=sj
                    audit['scaled']+= (e-s)+(e2-s2)
            elif mode=='percell_noI1E1':
                for c in M_local:
                    s,e=We.indptr[c],We.indptr[c+1]
                    for t in range(s,e):
                        if int(We.indices[t]) not in mset: We.data[t]*=s_map[c]
                    s2,e2=Wi.indptr[c],Wi.indptr[c+1]
                    for t in range(s2,e2):
                        src=int(Wi.indices[t])
                        if src not in mset: Wi.data[t]*=s_map[c]
                        elif entries[[x['global_index'] for x in entries].index(src)]['role']=='I1': Wi.data[t]=0.0
                    audit['scaled']+= (e-s)+(e2-s2)
            elif mode=='percell_zeromotif':
                for c in M_local:
                    for Wm in (We,Wi):
                        s,e=Wm.indptr[c],Wm.indptr[c+1]
                        for t in range(s,e):
                            src=int(Wm.indices[t])
                            if src in mset: Wm.data[t]=0.0;audit['scaled']+=1
                            else: Wm.data[t]*=s_map[c];audit['scaled']+=1
        audit['nonneg']=bool(np.all(We.data>=0) and np.all(Wi.data>=0))
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());C['network_init']+=1
        hidden=getattr(tier,'net',tier)
        nn=5000;spm=np.zeros((nn,len(roots)),bool);Vm=np.zeros((nn,ne),np.float32)
        for k in range(nn):
            spm[k]=np.asarray(tier.advance(2,np.zeros(len(roots))),dtype=bool).any(axis=0)
            C['control_tick']+=1;C['tier_advance']+=1;C['neural_substep']+=2
            v=np.asarray(getattr(hidden,'v',np.zeros(1)),float)
            if v.size>=len(roots): Vm[k]=v[M_local]
        per={}
        for lg in LEGS:
            e1=spm[:,loc[int(mod[lg]['E1'][0])]];i1=spm[:,[loc[int(c)] for c in mod[lg]['I1']]].any(axis=1)
            r=v5(e1,i1);r['e1_hz']=float(e1.sum()/5.0);per[lg]=r
        res[name]=dict(audit=audit,per_leg=per,duty=float(spm.mean()),Vm_mean=float(Vm.mean()),
          n_v5=sum(1 for lg in LEGS if per[lg]['rhythmic']),
          n_band=sum(1 for lg in LEGS if per[lg].get('band_ok')),
          bit_vs_s309=bool(np.array_equal(spm,ref_spk)) if name=='G1_A1_gate' else None,
          bit_vs_s314a4=bool(np.array_equal(spm,a4_spk)) if name=='B1_A4_repeat' else None)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(spm,axis=1),shape=np.array(spm.shape),V=Vm)
    dump(o/'ARMS.json',res)
    # V6 per-leg: ablation causality needs B3 vs B2
    for lg in LEGS:
        b2=res['B2_percell_v2']['per_leg'][lg];b3=res['B3_percell_v2_no_I1E1']['per_leg'][lg]
        abl=dict(b2_rhythmic=b2['rhythmic'],b3_rhythmic=b3['rhythmic'],
                 b3_destroyed=bool((not b3['rhythmic']) or (b3['n_bursts']<11) or (b3.get('cv') or 1)>=0.25))
        res['B2_percell_v2']['per_leg'][lg]['v6_c']=abl
    dump(o/'ARMS.json',res)
    # stage C reset audit
    V=np.load(o/'SPK_B2_percell_v2.npz')['V'];dV=np.abs(np.diff(V,axis=0))
    th=(np.abs(V).max(axis=0))
    reset=dict(dV_q={k:float(np.percentile(dV,p)) for k,p in (('p50',50),('p90',90),('p99',99),('max',100))},
      span_mV={entries[j]['leg']+'_'+entries[j]['role']:float(V[:,j].max()-V[:,j].min()) for j in range(ne)},
      frac_bins_crossing_half_span=float(np.mean(dV>(th[None,:]*0.5))),
      note='the binarization audit requested in S314 x3, now with archived membrane potentials')
    dump(o/'RESET_AUDIT.json',reset)
    dump(o/'CALL_LEDGER.json',C)
    # predictor check
    meas={ (e['leg'],e['role']):float(res['B2_percell_v2']['per_leg'][e['leg']]['e1_hz']) for e in entries if e['role']=='E1'}
    pred=dict(note='per-cell scaling was derived from the FACTORY trajectory, so the post-scaling network differs; this is an approximation and a >2x deviation is declared a prediction failure',
      predicted_isolated_rate_hz=dict(lo=30.0,hi=70.0,source='S313 isolated band rates at 0.0253-0.05 nA'),
      measured_e1_hz={f'{k[0]}_{k[1]}':v for k,v in meas.items()})
    pred['deviation_gt_2x']={f'{k[0]}_{k[1]}':bool(v>140.0 or v<15.0) for k,v in meas.items()}
    dump(o/'PREDICTOR_CHECK.json',pred)
    print(json.dumps(dict(n_v5={k:v['n_v5'] for k,v in res.items()},n_band={k:v['n_band'] for k,v in res.items()},
      gate=res['G1_A1_gate']['bit_vs_s309'],b1=res['B1_A4_repeat']['bit_vs_s314a4'],counters=C,
      G_spread=[round(min(G),5),round(max(G),5)]),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
