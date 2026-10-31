"""Independent second implementation for S339: 10 ms bins, 4th order 0.3-9 Hz, autocorrelation period, cross-correlation lag phase."""
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
def wrap(d): return (d+180.0)%360.0-180.0
def ser(A,cols):
    x=A[:,cols].sum(1).astype(float)
    n=len(x)//BIN
    r=np.array([x[i*BIN:(i+1)*BIN].sum() for i in range(n)],float)
    b,a=sgn.butter(4,[0.3/(FS/2),9.0/(FS/2)],btype='band');return sgn.filtfilt(b,a,r)
def acf_period(y):
    c=y-y.mean();a=np.correlate(c,c,'full')[len(c)-1:];a/=max(a[0],1e-30)
    lo,hi=int(0.12*FS),min(int(2.5*FS),len(a)-1)
    if hi<=lo: return None
    seg=a[lo:hi];k=int(np.argmin(seg))
    return float(FS/(lo+k)) if seg[k]<0 else None
def xcorr_phase(a,b,f0):
    za=a-a.mean();zb=b-b.mean()
    c=sgn.correlate(za,zb,mode='full',method='fft');lags=sgn.correlation_lags(len(za),len(zb),mode='full')
    lim=int(FS/f0) if f0 else int(FS);m=np.abs(lags)<=lim
    if not m.any(): return None
    cc=c[m];ll=lags[m];k=int(np.argmax(cc));lag=float(ll[k])/FS
    return float(wrap(360.0*lag*f0))
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
    for nm in ('G0_gate','A_seed1','A_seed2','T_off','T_25','T_50','T_150','T_200','T_300','D_swap','B_r050','B_r075'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        eA=ser(A,[j for lg in TRIPOD['A'] for j in by[lg]]);eB=ser(A,[j for lg in TRIPOD['B'] for j in by[lg]])
        T=acf_period(eA) or acf_period(eB);f0=1.0/T if T else None
        c0=float(np.corrcoef(eA,eB)[0,1])
        out[nm]=dict(period_s=T,f0=f0,phase=(xcorr_phase(eA,eB,f0) if f0 else None),corr0=c0,
          rateA=float(A[:,[j for lg in TRIPOD['A'] for j in by[lg]]].sum()*1000.0/(A.shape[0]*100)),
          rateB=float(A[:,[j for lg in TRIPOD['B'] for j in by[lg]]].sum()*1000.0/(A.shape[0]*102)))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['phase'],1) if v['phase'] is not None else None,round(v['corr0'],3),round(v['period_s'],3) if v['period_s'] else None) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
