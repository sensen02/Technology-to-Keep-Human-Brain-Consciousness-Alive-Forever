"""Independent second implementation for S343: 20 ms bins, order 3 lowpass, hilbert envelope, LS harmonic phase."""
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
BIN=20;FS=1000.0/BIN
def wrap(d): return (d+180.0)%360.0-180.0
def env(c):
    n=len(c)//BIN
    y=np.array([c[i*BIN:(i+1)*BIN].sum() for i in range(n)],float)
    sos=sgn.butter(3,[0.5/(FS/2),min(4.0/(FS/2),0.98)],btype='band',output='sos')
    return sgn.sosfiltfilt(sos,y-y.mean()),y
def f0_of(y):
    f,P=sgn.welch(y-y.mean(),fs=FS,nperseg=min(len(y),128))
    m=(f>=0.5)&(f<=4.0)
    return float(f[m][int(np.argmax(P[m]))]) if m.any() else 2.0
def phase(a,b,f0,harm=2):
    n=min(len(a),len(b));t=np.arange(n)/FS
    cols=[np.ones(n)]
    for k in range(1,harm+1): cols+=[np.cos(2*np.pi*k*f0*t),np.sin(2*np.pi*k*f0*t)]
    M=np.column_stack(cols);o=[]
    for y in (a[:n],b[:n]):
        cc,*_=np.linalg.lstsq(M,y,rcond=None);o.append(float(np.arctan2(cc[2],cc[1])))
    return abs(wrap(np.degrees(o[0]-o[1])))
def period(y):
    f0=f0_of(y)
    yc=y-y.mean();ac=sgn.correlate(yc,yc,mode='full',method='fft')[len(yc)-1:];ac/=max(ac[0],1e-30)
    lo,hi=int(0.2*FS),min(int(2.0*FS),len(ac)-1)
    p_acf=None
    if hi>lo:
        seg=ac[lo:hi];k=int(np.argmin(seg))
        if seg[k]<0: p_acf=2.0*(lo+k)/FS
    p_w=1.0/f0 if f0 else None
    return dict(welch=p_w,acf=p_acf,period_s=(0.5*(p_w+p_acf) if p_w and p_acf else (p_w or p_acf)))
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
    for nm in ('P0_gate','B20','B25','B30','B30_seed1','B30_seed2','B30_LA30','B30_LB30'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        cA=A[:,Ac].sum(1).astype(float);cB=A[:,Bc].sum(1).astype(float)
        bA,yA=env(cA);bB,yB=env(cB)
        f0=f0_of(bA)
        ph=phase(bA,bB,f0)
        pr=period(yA)
        duty=float(np.mean(yA>(np.percentile(yA,5)+np.percentile(yA,95))/2.0))
        out[nm]=dict(phase_deg=ph,period_s=pr,period_routes=pr,duty_A=duty,
          rates=dict(A=float(A[:,Ac].sum()*1000.0/(A.shape[0]*len(Ac))),B=float(A[:,Bc].sum()*1000.0/(A.shape[0]*len(Bc)))),
          f0=f0)
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['phase_deg'],1),round(v['period_s']['period_s'],3) if v['period_s']['period_s'] else None,round(v['duty_A'],3)) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
