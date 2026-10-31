"""Independent second implementation for S336: slow-series detection and pair phases."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
from scipy import signal as sgn
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
LEGS=('lf','lm','lh','rf','rm','rh');TRIPOD={'A':('lf','rm','lh'),'B':('lm','rf','rh')}
BIN=25;FS=1000.0/BIN;NULLS=1000
def ser(A,cols):
    x=A[:,cols].sum(1).astype(float);n=len(x)//BIN
    r=np.array([x[i*BIN:(i+1)*BIN].sum() for i in range(n)],float);return r-r.mean()
def peak(y):
    f,P=sgn.periodogram(y,fs=FS);m=(f>=0.3)&(f<=8.0)
    if not m.any(): return None,None
    k=int(np.argmax(P[m]));return float(f[m][k]),float(P[m][k])
def null(lam,n,rng):
    """independently synthesised aggregate Poisson count series with the same per-bin mean"""
    out=[]
    for _ in range(NULLS):
        ys=rng.poisson(max(lam,1e-9),n).astype(float);ys=ys-ys.mean()
        f,P=sgn.periodogram(ys,fs=FS);m=(f>=0.3)&(f<=8.0)
        out.append(float(P[m].max()) if m.any() else 0.0)
    return float(np.percentile(out,95)),float(np.std(out))
def bp(y,f0):
    lo,hi=max(f0*0.6,0.1),min(f0*1.4,FS/2*0.98)
    if hi<=lo: return None
    b,a=sgn.butter(3,[lo/(FS/2),hi/(FS/2)],btype='band');return sgn.filtfilt(b,a,y)
def ph(y): return float(np.angle(np.mean(np.exp(1j*np.angle(sgn.hilbert(y))))))
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    rng=np.random.default_rng(777);out={}
    for nm in ('C0_gate','R2_1p0','R3_2p0','R3_seed1','R3_seed2','M1_add2_bias_B','M2_bias_only','S2s_same_1p0_noloop'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        d={}
        for lg in LEGS:
            y=ser(A,by[lg]);f0,pw=peak(y)
            lam=float(A[:,by[lg]].sum())*BIN/1000.0/max(len(y),1)
            p95,nsd=null(lam,len(y),rng)
            d[lg]=dict(f_peak=f0,peak=pw,p95=p95,passes=bool(f0 is not None and pw>p95))
        passed=[lg for lg in LEGS if d[lg]['passes']]
        fslow=float(np.median([d[lg]['f_peak'] for lg in passed])) if passed else None
        if fslow:
            E={lg:ser(A,by[lg]) for lg in LEGS};pairs={}
            for i,a in enumerate(LEGS):
                for b in LEGS[i+1:]:
                    ya,yb=bp(E[a],fslow),bp(E[b],fslow)
                    pairs[f'{a}|{b}']=abs(float(np.degrees((ph(ya)-ph(yb)+np.pi)%(2*np.pi)-np.pi)))
            same=[v for k,v in pairs.items() if any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
            cross=[v for k,v in pairs.items() if not any(k.split('|')[0] in TRIPOD[t] and k.split('|')[1] in TRIPOD[t] for t in TRIPOD)]
            out[nm]=dict(f_slow=fslow,cross=float(np.mean(cross)),same=float(np.mean(same)),n_anti=int(sum(1 for v in cross if v>=135)),
              rate_A=float(A[:,[j for lg in TRIPOD['A'] for j in by[lg]]].sum(1).mean()*1000.0/A.shape[0]),
              rate_B=float(A[:,[j for lg in TRIPOD['B'] for j in by[lg]]].sum(1).mean()*1000.0/A.shape[0]),pairs=pairs)
        else:
            out[nm]=dict(f_slow=None,cross=None,n_anti=0,legs=d)
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['cross'],1) if v.get('cross') else None,v.get('n_anti'),v.get('f_slow')) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
