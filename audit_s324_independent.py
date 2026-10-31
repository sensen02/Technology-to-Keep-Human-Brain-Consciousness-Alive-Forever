"""Independent second implementation for S324: different code, different null seed, 120 draws."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
LEGS=('lf','lm','lh','rf','rm','rh');LEGS4=('lf','lm','rh','rm');ROLES=('E1','E2','I1')
NULLS=120;SEED=777123;REL=0.15
def bursts2(x,dt):
    idx=np.where(x>0)[0]
    if idx.size<2: return None,None
    gaps=np.diff(idx)*dt;cut=max(3.0*np.median(gaps),5.0)
    starts=[idx[0]];sizes=[1]
    for g,cur in zip(gaps,idx[1:]):
        if g>cut: starts.append(cur);sizes.append(1)
        else: sizes[-1]+=1
    return np.array(starts),np.array(sizes,float)
def env2(st,cn,n,dt):
    v=np.zeros(n)
    for s,c in zip(st,cn):
        if 0<=s<n: v[s]+=c
    k=max(int(round(20/dt)),1);ker=np.ones(k)/k
    v=np.convolve(v,ker,mode='same')
    t=np.arange(n);v=v-np.polyval(np.polyfit(t,v,1),t)
    return v
def feats2(v,dt):
    n=v.size;fr=np.fft.rfftfreq(n,dt/1000.0);pw=np.abs(np.fft.rfft(v))**2;step=fr[1]-fr[0]
    ok=(fr>=5+2*step)&(fr<=45-2*step)
    if not ok.any(): return None
    tot=pw[(fr>=0.3)&(fr<=50)].sum();pk=fr[int(np.argmax(np.where(ok,pw,0)))]
    band=(fr>=pk*(1-REL))&(fr<=pk*(1+REL))
    r=float(pw[band].sum()/max(tot,1e-30))
    c=v-v.mean();ac=np.correlate(c,c,'full')[n-1:];ac/=max(ac[0],1e-30)
    a=int(20/dt);b=min(int(200/dt),ac.size-1);ap=0.0
    if b>a: ap=float(ac[a:b].max())
    return r,ap
def label2(x,dt):
    x=np.asarray(x,float)
    if x.sum()==0: return 'SILENT',None
    rate=x.mean()*1000.0/dt;st,cn=bursts2(x,dt)
    nb=0 if st is None else st.size;spb=0.0 if nb==0 else float(cn.mean())
    if nb<10 and rate<40: return 'NO_BURSTS',None
    if spb>12 and rate>=40: return 'TONIC',None
    if not (5<=rate<=300) or nb<10 or nb<2: return 'UNDERPOWERED',None
    iv=np.diff(st)*dt
    if iv.size<10: return 'UNDERPOWERED',None
    fiv=1000.0/float(iv.mean())
    obs=feats2(env2(st,cn,x.size,dt),dt)
    if obs is None: return 'UNDERPOWERED',fiv
    rng=np.random.default_rng(SEED);nr=[];na=[];c=0
    for _ in range(NULLS):
        s2=np.sort(rng.integers(0,x.size,nb));w2=cn[rng.integers(0,cn.size,nb)]
        q=feats2(env2(s2,w2,x.size,dt),dt)
        if q is None: continue
        nr.append(q[0]);na.append(q[1])
        if q[0]>=obs[0]: c+=1
    if not nr: return 'UNDERPOWERED',fiv
    pr=np.percentile(nr,95);pa=np.percentile(na,95);p=(c+1)/(len(nr)+1)
    if 0.90<=obs[0]/max(pr,1e-12)<=1.10 or 0.90<=obs[1]/max(pa,1e-12)<=1.10:
        return 'UNDERPOWERED',fiv
    ok=bool(obs[0]>pr and obs[1]>pa and p<0.01)
    return ('RHYTHMIC' if ok else 'NOT_RHYTHMIC'),fiv
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    E1={lg:loc[int(mod[lg]['E1'][0])] for lg in LEGS}
    out={}
    srcs={'K17':S322/'SPK_K17.npz','K17_abl':S322/'SPK_K17_abl.npz','L1':o/'SPK_L1_k17_seed0_10s.npz','L2':o/'SPK_L2_k17abl_seed0_10s.npz'}
    for nm,f in srcs.items():
        if not f.is_file(): continue
        z=np.load(f);sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
        per={}
        for W in (2.5,3.0,5.0,10.0):
            n=int(W*1000)
            if n>sm.shape[0]:
                per[str(W)]={lg:dict(label='INVALID_WINDOW') for lg in LEGS4};continue
            per[str(W)]={lg:(lambda t:(dict(label=t[0],f_iv=t[1])))(label2(sm[:n,E1[lg]].astype(float),1.0)) for lg in LEGS4}
        out[nm]=per
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(nulls=NULLS,seed=SEED,legs=list(LEGS4),result=out,
      note='second implementation: list-based burst detection, FFT autocorrelation, 120 draws, seed 777123'),indent=2)+'\n')
    print(json.dumps({k:{w:{lg:v[lg]['label'] for lg in v} for w,v in d.items()} for k,d in out.items()},ensure_ascii=False)[:800])
    return 0
if __name__=='__main__': raise SystemExit(main())
