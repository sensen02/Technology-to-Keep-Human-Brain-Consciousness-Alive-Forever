"""Independent second implementation for S325: matched-window labels and f_iv only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S324=Path('/home/sensen/Desktop/engine修复验收/s324-period-certified-detector/20261027T001625Z')
LEGS4=('lf','lm','rh','rm');NULLS=1000;SEED=991117;REL=0.15
def cuts(x):
    idx=np.where(x>0)[0]
    if idx.size<2: return [],[]
    g=np.diff(idx);cut=max(3.0*np.median(g),5.0);st=[idx[0]];sz=[1]
    for d,c in zip(g,idx[1:]):
        if d>cut: st.append(c);sz.append(1)
        else: sz[-1]+=1
    return st,sz
def env(st,sz,n):
    v=np.zeros(n)
    for a,b in zip(st,sz): v[a]+=b
    k=20;v=np.convolve(v,np.ones(k)/k,'same');t=np.arange(n)
    return v-np.polyval(np.polyfit(t,v,1),t)
def fts(v):
    n=v.size;fr=np.fft.rfftfreq(n,1e-3);pw=np.abs(np.fft.rfft(v))**2;st_=fr[1]-fr[0]
    ok=(fr>=5+2*st_)&(fr<=45-2*st_)
    if not ok.any(): return None
    tot=pw[(fr>=0.3)&(fr<=50)].sum();pk=fr[int(np.argmax(np.where(ok,pw,0)))]
    band=(fr>=pk*(1-REL))&(fr<=pk*(1+REL))
    c=v-v.mean();ac=np.correlate(c,c,'full')[n-1:];ac/=max(ac[0],1e-30)
    a,b=20,min(200,ac.size-1);ap=float(ac[a:b].max()) if b>a else 0.0
    return float(pw[band].sum()/max(tot,1e-30)),ap
def lab(x):
    if x.sum()==0: return 'SILENT',None
    rate=x.mean()*1000.0;st,sz=cuts(x)
    if not st: return ('NO_BURSTS' if rate<40 else 'TONIC'),None
    nb=len(st);spb=float(np.mean(sz))
    if nb<10 and rate<40: return 'NO_BURSTS',None
    if spb>12 and rate>=40: return 'TONIC',None
    if not(5<=rate<=300) or nb<10 or nb<2: return 'UNDERPOWERED',None
    iv=np.diff(st);fiv=1000.0/float(iv.mean())
    if iv.size<10: return 'UNDERPOWERED',fiv
    obs=fts(env(st,sz,x.size))
    if obs is None: return 'UNDERPOWERED',fiv
    rng=np.random.default_rng(SEED);nr=[];na=[];c=0
    for _ in range(NULLS):
        s2=np.sort(rng.integers(0,x.size,nb));z2=np.array([sz[k] for k in rng.integers(0,len(sz),nb)],float)
        q=fts(env(s2,z2,x.size))
        if q is None: continue
        nr.append(q[0]);na.append(q[1])
        if q[0]>=obs[0]: c+=1
    if not nr or 1.0/(len(nr)+1)>=0.01: return 'NO_DECISION',fiv
    pr=np.percentile(nr,95);pa=np.percentile(na,95);p=(c+1)/(len(nr)+1)
    if 0.90<=obs[0]/max(pr,1e-12)<=1.10 or 0.90<=obs[1]/max(pa,1e-12)<=1.10: return 'UNDERPOWERED',fiv
    return ('RHYTHMIC' if (obs[0]>pr and obs[1]>pa and p<0.01) else 'NOT_RHYTHMIC'),fiv
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    E1={lg:loc[int(mod[lg]['E1'][0])] for lg in LEGS4}
    z=np.load(S324/'SPK_K17.npz') if (S324/'SPK_K17.npz').is_file() else np.load(S322/'SPK_K17.npz')
    sm=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])].astype(bool)
    out={}
    for W in (2.5,3.0,5.0):
        n=int(W*1000)
        out[str(W)]={lg:(lambda t:dict(label=t[0],f_iv=t[1]))(lab(sm[:n,E1[lg]].astype(float))) for lg in LEGS4}
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(nulls=NULLS,seed=SEED,result={'K17':out}),indent=2)+'\n')
    print(json.dumps({w:{lg:out[w][lg]['label'] for lg in out[w]} for w in out},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
