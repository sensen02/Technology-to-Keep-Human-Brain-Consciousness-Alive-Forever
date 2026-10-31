"""Independent second implementation for S333: exclusive sets and pooled phase signatures."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S332=Path('/home/sensen/Desktop/engine修复验收/s332-phase-estimator-and-tripod-differential/20261027T013035Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
def csr(z):
    n=int(z['indptr'].size-1)
    return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
def env(x):
    y=np.convolve(x.astype(float),np.ones(20)/20,'same');t=np.arange(y.size)
    return (y-np.polyval(np.polyfit(t,y,1),t))[::2]
def ph(x,y,fs=500.0,nper=1024,band=(16.0,26.0)):
    nper=min(nper,len(x));step=nper//2;f=np.fft.rfftfreq(nper,1/fs);acc=np.zeros(nper//2+1,complex);c=0
    for s in range(0,len(x)-nper+1,step):
        X=np.fft.rfft(x[s:s+nper]*np.hanning(nper));Y=np.fft.rfft(y[s:s+nper]*np.hanning(nper))
        acc+=X*np.conj(Y);c+=1
    if c==0: return None
    S=acc/c;m=(f>=band[0])&(f<=band[1])
    w=np.abs(S[m]);return float(np.degrees(np.angle(np.sum(w*np.exp(1j*np.angle(S[m]))))))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18={lg:[loc[int(c)] for r in ROLES for c in mod[lg][r]] for lg in LEGS}
    Mset=set(j for lg in LEGS for j in M18[lg])
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    cols=[j for (_l,_g),j in pool];poolset=set(cols)
    We=csr(np.load(S303/'EDGES_w_2.5e-5_We.npz'));Wi=csr(np.load(S303/'EDGES_w_2.5e-5_Wi.npz'))
    def sets(W,thr):
        out={}
        per={lg:np.asarray(W[cols,:][[cols.index(j) for j in by[lg]],:].sum(0)).ravel() for lg in LEGS}
        tot=sum(per[lg] for lg in LEGS)
        for lg in LEGS:
            frac=np.where(tot>1e-12,per[lg]/np.maximum(tot,1e-12),0.0)
            idx=[j for j in np.flatnonzero((frac>=thr)&(per[lg]>0)) if int(j) not in poolset and int(j) not in Mset]
            out[lg]=(np.array(idx,int),float(per[lg][idx].sum()),float(per[lg].sum()))
        return out
    SE={t:sets(We,t) for t in (0.8,0.9,0.95)};SI={t:sets(Wi,t) for t in (0.8,0.9,0.95)}
    A=np.unpackbits(np.load(S332/'SPK_G0_gate.npz')['packed'],axis=1)[:,:12000]
    E={lg:env(A[:,by[lg]].sum(1)) for lg in LEGS}
    pairs={f'{a}|{b}':ph(E[a],E[b]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
    same=[abs(v) for k,v in pairs.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    cross=[abs(v) for k,v in pairs.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
    res_arms={}
    for nm,f in (('A0_gate',o/'SPK_A0_gate.npz'),('T1_in_diff010',o/'SPK_T1_in_diff010.npz'),('T3_in_diff040',o/'SPK_T3_in_diff040.npz'),('T5_in_common020',o/'SPK_T5_in_common020.npz')):
        if not Path(f).is_file(): continue
        z=np.load(f);B=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        EE={lg:env(B[:,by[lg]].sum(1)) for lg in LEGS}
        pp={f'{a}|{b}':ph(EE[a],EE[b]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        s_=[abs(v) for k,v in pp.items() if v is not None and any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        c_=[abs(v) for k,v in pp.items() if v is not None and not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
        res_arms[nm]=dict(same=float(np.mean(s_)),cross=float(np.mean(c_)),cross_min=float(min(c_)),n_anti=int(sum(1 for v in c_ if v>=135)))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(
      exclusive={str(t):{lg:dict(n=len(SE[t][lg][0]),weight=SE[t][lg][1],total=SE[t][lg][2]) for lg in LEGS} for t in SE},
      exclusive_I={str(t):{lg:dict(n=len(SI[t][lg][0]),weight=SI[t][lg][1],total=SI[t][lg][2]) for lg in LEGS} for t in SI},
      g0=dict(same=float(np.mean(same)),cross=float(np.mean(cross)),pairs=pairs),arms=res_arms),indent=2)+'\n')
    print(json.dumps(dict(exc={lg:len(SE[0.9][lg][0]) for lg in LEGS},
      share={lg:round(SE[0.9][lg][1]/SE[0.9][lg][2],4) for lg in LEGS},
      g0=[round(float(np.mean(same)),1),round(float(np.mean(cross)),1)],
      arms={k:(round(v['same'],1),round(v['cross'],1),v['n_anti']) for k,v in res_arms.items()}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
