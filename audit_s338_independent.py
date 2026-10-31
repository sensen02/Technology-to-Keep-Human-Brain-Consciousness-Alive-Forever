"""Independent second implementation for S338: 50 ms bins, 2nd order lowpass envelope, autocorrelation peak."""
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
BIN=50;FS=1000.0/BIN
def wrap(d): return (d+180.0)%360.0-180.0
def env(A,cols):
    x=A[:,cols].sum(1).astype(float)
    n=len(x)//BIN;r=np.array([x[i*BIN:(i+1)*BIN].sum() for i in range(n)],float)
    b,a=sgn.butter(2,8.0/(FS/2),btype='low');return sgn.filtfilt(b,a,r)
def acf_f0(y):
    c=y-y.mean();a=np.correlate(c,c,'full')[len(c)-1:];a/=max(a[0],1e-30)
    lo,hi=int(0.15*FS),min(int(2.0*FS),len(a)-1)
    if hi<=lo: return None
    seg=a[lo:hi];k=int(np.argmax(seg))
    return float(FS/(lo+k)) if seg[k]>0 else None
def ph_ls(a,b,f0):
    n=min(len(a),len(b));t=np.arange(n)/FS
    M=np.column_stack([np.cos(2*np.pi*f0*t),np.sin(2*np.pi*f0*t),np.ones(n)]);out=[]
    for y in (a[:n],b[:n]):
        c,*_=np.linalg.lstsq(M,y,rcond=None);out.append(np.arctan2(c[1],c[0]))
    return float(wrap(np.degrees(out[0]-out[1])))
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
    out={}
    for nm in ('C0_gate','R2p0_seed0','R2p0_seed1','R2p0_seed2','N0_no_adapt','N1_tau50','N2_tau200','B_eq_seed0','B_eq_seed1','N_gate_cn'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        eA=env(A,[j for lg in TRIPOD['A'] for j in by[lg]]);eB=env(A,[j for lg in TRIPOD['B'] for j in by[lg]])
        f0a=acf_f0(eA);f0b=acf_f0(eB)
        if f0a is None or f0b is None:
            out[nm]=dict(detected=False,rates=dict(A=float(A[:,[j for lg in TRIPOD['A'] for j in by[lg]]].sum()*1000.0/(A.shape[0]*100)),
              B=float(A[:,[j for lg in TRIPOD['B'] for j in by[lg]]].sum()*1000.0/(A.shape[0]*102))));continue
        f0=float((f0a+f0b)/2)
        out[nm]=dict(detected=True,f0=f0,phase=abs(ph_ls(eA,eB,f0)),
          corr=float(np.corrcoef(eA,eB)[0,1]),
          rates=dict(A=float(A[:,[j for lg in TRIPOD['A'] for j in by[lg]]].sum()*1000.0/(A.shape[0]*100)),
            B=float(A[:,[j for lg in TRIPOD['B'] for j in by[lg]]].sum()*1000.0/(A.shape[0]*102))))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v.get('phase') or 0,1),round(v.get('corr') or 0,3),round(v.get('f0') or 0,2)) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
