"""Independent second implementation for S340: 10 ms bins, 3rd order 0.4-4.5 Hz, circular shift null."""
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
BIN=10;FS=1000.0/BIN;NULLS=1000
def ser(A,cols):
    x=A[:,cols].sum(1).astype(float);n=len(x)//BIN
    return np.array([x[i*BIN:(i+1)*BIN].sum() for i in range(n)],float)
def bpf(y):
    b,a=sgn.butter(3,[0.4/(FS/2),4.5/(FS/2)],btype='band');pad=3*max(len(a),len(b))
    return sgn.filtfilt(b,a,y) if len(y)>pad else y-y.mean()
def rho(a,b):
    n=min(len(a),len(b));xa=bpf(a[:n]);xb=bpf(b[:n])
    if np.std(xa)<1e-12 or np.std(xb)<1e-12: return 0.0
    return float(np.corrcoef(xa,xb)[0,1])
def shift_null(n):
    rng=np.random.default_rng(9091);v=[]
    for _ in range(NULLS):
        x=rng.poisson(5.0,n).astype(float);y=rng.poisson(5.0,n).astype(float)
        k=int(rng.integers(int(FS),max(int(FS)+1,n-1)))
        v.append(rho(x,np.roll(y,k)))
    v=np.array(v);return dict(p1=float(np.percentile(v,1)),std=float(v.std()),nulls=NULLS)
def band_frac(y):
    yc=y-y.mean();f,P=sgn.welch(yc,fs=FS,nperseg=min(len(yc),max(16,len(yc)//2)))
    m=(f>=0.5)&(f<=8.0);tot=P[(f>=0.5)&(f<=20.0)].sum()
    return float(P[m].sum()/tot) if tot>0 and m.any() else None
def band_null(count,n):
    rng=np.random.default_rng(555);lam=count/max(n,1);v=[]
    for _ in range(NULLS):
        v.append(band_frac(rng.poisson(max(lam,1e-9),n).astype(float)))
    v=np.array(v);return dict(p99=float(np.percentile(v,99)),std=float(v.std()),nulls=NULLS)
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
    Ac=[j for lg in TRIPOD['A'] for j in by[lg]];Bc=[j for lg in TRIPOD['B'] for j in by[lg]]
    out={}
    names=['G_gate','T_60','T_75','T_90','T_250','U1_asymA10','U2_asymB10']
    for nm in names:
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        eA=ser(A,Ac);eB=ser(A,Bc)
        bfA=band_frac(eA);bfB=band_frac(eB);nA=band_null(float(A[:,Ac].sum()),len(eA));nB=band_null(float(A[:,Bc].sum()),len(eB))
        r=rho(eA,eB);nr=shift_null(min(len(eA),len(eB)))
        ex=bool(bfA>nA['p99'] and bfB>nB['p99'])
        anti=bool(r<nr['p1'])
        rA=float(A[:,Ac].sum()*1000.0/(A.shape[0]*len(Ac)));rB=float(A[:,Bc].sum()*1000.0/(A.shape[0]*len(Bc)))
        ratio=max(rA,rB)/max(min(rA,rB),1e-9)
        lab='ALTERNATING' if (ex and anti) else ('RHYTHM_NOT_ANTIPHASE' if ex else ('WINNER_TAKE_ALL' if ratio>3 else 'NO_RHYTHM'))
        out[nm]=dict(label=lab,rho_bp=r,rho_null=nr,band_frac=dict(A=bfA,B=bfB),band_null=dict(A=nA,B=nB),
          rates=dict(A=rA,B=rB,ratio=ratio))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(v['label'],round(v['rho_bp'],3),round(v['rho_null']['p1'],3),round(v['rates']['ratio'],2)) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
