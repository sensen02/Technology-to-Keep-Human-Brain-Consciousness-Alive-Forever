"""S314: per-leg rejudge (V5), external-input split, closed-form f*, five tier arms."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S309=Path('/home/sensen/Desktop/engine修复验收/s309-native-cpg/20261026T205541Z')
S312=Path('/home/sensen/Desktop/engine修复验收/s312-tier-motif-input-isolation/20261026T211120Z')
S313=Path('/home/sensen/Desktop/engine修复验收/s313-burst-band-and-tier-drive/20261026T211545Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');ORDER=('E1','E2','I1');FP='56e0322a33c51b7f03aaae9f7df1ef3f'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else float(o))+'\n',encoding='utf-8')
C=dict(banc_load=0,tier_build=0,select_touch_tier=0,edges_build=0,network_init=0,control_tick=0,tier_advance=0,neural_substep=0,physics_step=0,model_load=0,render=0)
def bursts(x,dt=1.0):
    idx=np.flatnonzero(x)
    if len(idx)<3: return np.array([],int),np.array([],int)
    isi=np.diff(idx)*dt;thr=3*np.median(isi);st=[idx[0]];cnt=[1]
    for i,d in enumerate(isi):
        if d>thr: st.append(idx[i+1]);cnt.append(1)
        else: cnt[-1]+=1
    return np.asarray(st),np.asarray(cnt)
def v5_leg(e1,i1,dt=1.0):
    st,cnt=bursts(e1,dt);n=len(st)
    out=dict(n_bursts=int(n),mean_hz=float(e1.mean()*1000.0/dt))
    if n<11: out.update(rhythmic=False,reason='few_bursts');return out
    per=np.diff(st)*dt;cv=float(np.std(per)/max(float(np.mean(per)),1e-9));spc=float(cnt[1:-1].mean())
    si,sc=bursts(i1,dt)  # I1 BURST STARTS under the same ISI rule
    fallback=False
    if len(si)<3:   # a single spike per cycle yields one I1 group; use one start per E1 cycle
        i1t=np.flatnonzero(i1);si=[]
        for s0 in st:
            k=np.searchsorted(i1t,s0)
            if k<len(i1t): si.append(i1t[k])
        si=np.asarray(si,int);fallback=True
    lag=None;p=1.0
    if len(si)>=3:
        lags=[]
        for s0 in st:
            k=np.searchsorted(si,s0)
            if k<len(si): lags.append(((si[k]-s0)*dt/np.mean(per))%1.0)
        if lags:
            lag=float(np.median(lags));z=np.exp(2j*np.pi*np.asarray(lags));R=float(abs(z.mean()));nn=len(lags)
            p=float(np.exp(np.sqrt(1+4*nn+4*(nn**2-(nn*R)**2))-(1+2*nn))) if nn>1 and R<1 else 0.0
    rh=bool(len(per)>=10 and cv<0.25 and lag is not None and 0.0<lag<0.25 and p<0.01 and 1.0<=spc<=12.0 and 5.0<=out['mean_hz']<=300.0)
    out.update(rhythmic=rh,freq_hz=float(1000.0/np.mean(per)),cv=cv,spikes_per_cycle=spc,lag=lag,rayleigh_p=p,i1_bursts=int(len(si)),i1_start_fallback=fallback)
    return out
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    o.mkdir(parents=True,exist_ok=True);(o/'verify').mkdir(exist_ok=True)
    import sys;sys.path.insert(0,str(ROOT))
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    mapping=[]
    for lg in LEGS:
        for a in ORDER:
            for c in mod[lg][a]:
                gi=loc[int(c)];mapping.append(dict(leg=lg,role=a,root=int(c),global_index=int(gi)))
    M18=sorted(d['global_index'] for d in mapping)
    dump(o/'M18_GLOBAL_INDEX.json',dict(M18=M18,entries=mapping,order_note='S312 archived motif columns as sorted(global_index); E1 columns are NOT 0,3,...'))
    # per-leg rejudge of S312 arms, correct columns
    col={ (d['leg'],d['role']):M18.index(d['global_index']) for d in mapping }
    rej={}
    for arm in ('T2_iso_cap8_d050','T5_iso_raw_d050','T6_iso_raw_d160','T1_gate','T4_iso_cap8_d000'):
        f=S312/f'SPK_{arm}.npz'
        if not f.is_file(): continue
        m=np.load(f)['motif'];per={}
        for lg in LEGS:
            e1=m[:,col[(lg,'E1')]];i1=m[:,col[(lg,'I1')]]
            r=v5_leg(e1,i1,1.0);r['e1_hz']=float(e1.sum()/5.0);r['i1_hz']=float(i1.sum()/5.0);r['e2_hz']=float(m[:,col[(lg,'E2')]].sum()/5.0)
            per[lg]=r
        merged=v5_leg(m[:,[col[(lg,'E1')] for lg in LEGS]].any(axis=1),m[:,[col[(lg,'I1')] for lg in LEGS]].any(axis=1),1.0)
        rej[arm]=dict(per_leg=per,merged=merged,n_rhythmic=sum(1 for lg in LEGS if per[lg]['rhythmic']))
    dump(o/'LEG_REJUDGE.json',rej)
    # external input split from S309 k1_control full-net spikes
    sp=np.load(S309/'SPIKES_k1_control.npz');spk=np.unpackbits(sp['packed'],axis=1)[:,:int(sp['shape'][1])].astype(bool)
    e0w=np.load(S303/'EDGES_w_5e-4_We.npz');e0i=np.load(S303/'EDGES_w_5e-4_Wi.npz')
    mset=set(M18);ext=np.zeros((spk.shape[0],len(M18)));inn=np.zeros_like(ext)
    for jj,j in enumerate(M18):
        for W,sgn,tgt in ((e0w,1.0,ext),(e0i,-1.0,inn)):
            s,e=W['indptr'][j],W['indptr'][j+1];src=W['indices'][s:e];dat=np.asarray(W['data'][s:e],float)
            mask=np.array([int(x) in mset for x in src])
            if (~mask).any(): tgt[:,jj]+=sgn*(spk[:,src[~mask]].astype(float)@dat[~mask])
            if mask.any(): inn[:,jj]+=sgn*(spk[:,src[mask]].astype(float)@dat[mask])
    ti=np.load(S313/'TSPK_1.npz')['net_in']
    split=dict(ext_q=dict(p10=float(np.percentile(ext,10)),p50=float(np.percentile(ext,50)),p90=float(np.percentile(ext,90))),
      int_q=dict(p10=float(np.percentile(inn,10)),p50=float(np.percentile(inn,50)),p90=float(np.percentile(inn,90))),
      sum_equals_TSPK=bool(np.allclose(ext+inn,ti,atol=1e-9)),max_abs_diff=float(np.abs(ext+inn-ti).max()),
      sign_note='external net input is the signed sum over sources NOT in M18; internal is over M18')
    dump(o/'EXTERNAL_INPUT_SPLIT.json',split)
    G_ext=float(np.median([np.median(ext[:,jj]) for jj in range(len(M18))]))
    f_star=0.05/(65.0*max(G_ext,1e-12))
    dump(o/'SCALE_RULE.json',dict(G_ext_uS=G_ext,formula='f* = 0.05 nA / (65 mV x G_ext)',f_star=f_star,two_f=2*f_star,half_f=f_star/2,
      note='f is a closed-form constant, not a grid'))
    pre=dict(round='S314',stage_b_arms=['A1_control','A2_ext_scale_f','A3_ext_scale_2f','A4_ext_scale_half_f','A5_row_uniform_f'],
      scale_rule=dict(G_ext_uS=G_ext,f_star=f_star,two_f=2*f_star,half_f=f_star/2),
      counters=dict(normal=dict(banc_load=5,tier_build=5,select_touch_tier=10,edges_build=5,network_init=5,control_tick=25000,tier_advance=25000,neural_substep=50000,physics_step=0,model_load=0,render=0),
                    gate_fail=dict(banc_load=1,tier_build=1,select_touch_tier=2,edges_build=1,network_init=1,control_tick=5000,tier_advance=5000,neural_substep=10000)),
      v5='per-leg: >=10 bursts, cv<0.25, spikes/cycle in [1,12], per-cell mean in [5,300] Hz, I1 BURST-start lag in (0,0.25) with Rayleigh p<0.01',
      branches=dict(x1='A2/A3/A4 gives >=5/6 legs rhythmic with duty>=0.05 and no fully silent motif cell',
                    x2='rhythm only in the row-uniform arm',x3='no arm rhythmic',x4='gate or selftest failure'),
      hard=['verifier writes only verify/','PREREG then FROZEN before dynamics','npz for arrays','no physics/rendering'])
    dump(o/'PREREG.json',pre);(o/'S314_PREREG_FROZEN').write_text('frozen before any dynamic call\n')
    # detector v5 selftest (f)(g)
    def synth(per,nspk,lag,i1per=1,dt=1.0,dur=3000.0):
        n=int(dur/dt);e=np.zeros(n,bool);i=np.zeros(n,bool);t=per*0.5
        while t<dur:
            s=int(t/dt)
            for k in range(nspk):
                if s+k<n: e[s+k]=True
            for k in range(i1per):
                j=int((t+lag*per)/dt)+k
                if j<n: i[j]=True
            t+=per
        return e,i
    e,i=synth(1000/14.0,2,0.15);a=v5_leg(e,i)
    e,i=synth(1000/17.0,2,0.15);b=v5_leg(e,i)
    e,_=synth(1000/14.0,2,0.15);_,j=synth(1000/14.0,1,0.0);c=v5_leg(e,j)
    e,i=synth(1000/14.0,2,0.15,i1per=3);g=v5_leg(e,i)
    # (f) six different-frequency legs merged
    legs=[synth(1000/(10+2*k),2,0.15)[0] for k in range(6)];i1s=[synth(1000/(10+2*k),1,0.15)[1] for k in range(6)]
    merged=v5_leg(np.any(legs,axis=0),np.any(i1s,axis=0))
    perleg=[v5_leg(legs[k],i1s[k]) for k in range(6)]
    st=dict(a=a,b=b,c=c,g=g,merged=merged,perleg_rhythmic=sum(1 for x in perleg if x['rhythmic']),
      pass_a=bool(a['rhythmic'] and abs(a['freq_hz']-14)<1),pass_b=bool(b['rhythmic'] and abs(b['freq_hz']-17)<1.5),
      pass_c=bool(not c['rhythmic']),
      pass_f=bool(merged['rhythmic'] is False and sum(1 for x in perleg if x['rhythmic'])==6),
      pass_g=bool(g['rhythmic']))
    dump(o/'DETECTOR_V5.json',dict(criteria=pre['v5']));dump(o/'DETECTOR_SELFTEST_V5.json',st)
    if not all(st[k] for k in ('pass_a','pass_b','pass_c','pass_f','pass_g')):
        dump(o/'STOP.json',dict(reason='detector v5 selftest failed',st=st));dump(o/'CALL_LEDGER.json',C);return 3
    # ---- stage B
    from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
    armspec=[('A1_control',None),('A2_ext_scale_f',('ext',f_star)),('A3_ext_scale_2f',('ext',2*f_star)),
             ('A4_ext_scale_half_f',('ext',f_star/2)),('A5_row_uniform_f',('both',f_star))]
    res={}
    ref_spk=spk.copy()
    for k,(name,spec) in enumerate(armspec):
        b=build_neural_tier(TierConfig(sign_fill='all_empty').validate());C['tier_build']+=1;C['banc_load']+=1;C['select_touch_tier']+=2;C['edges_build']+=1
        e=b['edges'];We=e['We'] if isinstance(e,dict) else e.we;Wi=e['Wi'] if isinstance(e,dict) else e.wi
        audit=dict(scaled=0,removed=0)
        if spec:
            mode,f=spec
            for W in ((We,Wi) if mode=='both' else (We,Wi)):
                for j in M18:
                    s,en=W.indptr[j],W.indptr[j+1]
                    for t in range(s,en):
                        src=int(W.indices[t]);inside=src in mset
                        if mode=='ext' and inside: continue
                        W.data[t]*=f;audit['scaled']+=1
                        if W.data[t]==0.0: audit['removed']+=1
        audit['nonneg']=bool(np.all(We.data>=0) and np.all(Wi.data>=0))
        tier=NeuralTier(b,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());C['network_init']+=1
        n=5000;sp=np.zeros((n,len(roots)),bool)
        for kk in range(n):
            sp[kk]=np.asarray(tier.advance(2,np.zeros(len(roots))),dtype=bool).any(axis=0);C['control_tick']+=1;C['tier_advance']+=1;C['neural_substep']+=2
        per={}
        for lg in LEGS:
            e1=sp[:,loc[int(mod[lg]['E1'][0])]] if len(mod[lg]['E1'])==1 else sp[:,[loc[int(c)] for c in mod[lg]['E1']]].any(axis=1)
            i1=sp[:,[loc[int(c)] for c in mod[lg]['I1']]].any(axis=1)
            r=v5_leg(e1,i1,1.0);r['e1_hz']=float(e1.sum()/5.0)
            per[lg]=r
        res[name]=dict(audit=audit,per_leg=per,n_rhythmic=sum(1 for lg in LEGS if per[lg]['rhythmic']),
          duty=float(sp.mean()),e1_hz={lg:round(per[lg]['e1_hz'],1) for lg in LEGS},
          a1_bitwise=bool(np.array_equal(sp,ref_spk)) if name=='A1_control' else None)
        np.savez_compressed(o/f'SPK_{name}.npz',packed=np.packbits(sp,axis=1),shape=np.array(sp.shape))
    dump(o/'ARMS.json',res);dump(o/'CALL_LEDGER.json',C)
    print(json.dumps(dict(n_rh={k:v['n_rhythmic'] for k,v in res.items()},a1=res['A1_control']['a1_bitwise'],
      f=f_star,G=G_ext,counters=C,leg_rej={k:v['n_rhythmic'] for k,v in rej.items()}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
