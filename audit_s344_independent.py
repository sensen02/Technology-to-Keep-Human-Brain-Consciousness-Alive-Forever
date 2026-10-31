"""Independent second implementation for S344: 50 ms boxcar, no band pass, detrended, lag half period check."""
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
BIN=50
def env(c):
    n=len(c)//BIN
    return np.array([c[i*BIN:(i+1)*BIN].sum() for i in range(n)],float)
def detrend(y):
    t=np.arange(len(y));return y-np.polyval(np.polyfit(t,y,1),t)
def stats(a,b):
    fs=1000.0/BIN;k=int(0.05*fs)
    xa=np.convolve(a-a.mean(),np.ones(k)/k,'same');xb=np.convolve(b-b.mean(),np.ones(k)/k,'same')
    xa=detrend(xa);xb=detrend(xb)
    c0=float(np.corrcoef(xa,xb)[0,1])
    ac=sgn.correlate(xa,xa,'full',method='fft')[len(xa)-1:];ac/=max(ac[0],1e-30)
    lo,hi=max(2,int(0.2*fs)),min(len(ac)-1,int(1.2*fs));per=None
    if hi>lo:
        seg=ac[lo:hi];k2=int(np.argmin(seg))
        if seg[k2]<0: per=2.0*(lo+k2)/fs
    chalf=None
    if per:
        lag=int(round(0.5*per*fs))
        if 0<lag<len(xa)-4: chalf=float(np.corrcoef(xa[lag:],xb[:-lag])[0,1])
    ph=float(np.degrees(np.abs(np.angle(np.mean(np.exp(1j*(np.angle(sgn.hilbert(xa))-np.angle(sgn.hilbert(xb)))))))))
    return dict(corr0=c0,corr_half=chalf,period_s=per,phase_deg=ph)
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
    for nm in ('G0_gate','B30_5s','B30_seed1','B30_seed2','LA30_5s','LB30_5s','P10_10s','B30_10s'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        a=env(A[:,Ac].sum(1).astype(float));b=env(A[:,Bc].sum(1).astype(float))
        st=stats(a,b)
        st['rateA']=float(A[:,Ac].sum()*1000.0/(A.shape[0]*len(Ac)));st['rateB']=float(A[:,Bc].sum()*1000.0/(A.shape[0]*len(Bc)))
        st['ticks']=int(A.shape[0])
        out[nm]=st
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['phase_deg'],1),round(v['corr0'],3),v['period_s']) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
