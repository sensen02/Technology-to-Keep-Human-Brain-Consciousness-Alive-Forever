"""Independent second implementation for S342: 100 ms boxcar as the primary statistic, SOS order 6 lowpass at fs=1000, Poisson nulls."""
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
NULLS=1000;BOX=100
def sos_ok(sos):
    for s in sos:
        r=np.roots(s[3:])
        if r.size and float(np.max(np.abs(r)))>0.999: raise SystemExit('unstable section')
    return sos
def box_env(c,order=3):
    n=len(c)//BOX
    y=np.array([c[i*BOX:(i+1)*BOX].sum() for i in range(n)],float);fs=1000.0/BOX
    sos=sos_ok(sgn.butter(order,[0.5/(fs/2),min(4.0/(fs/2),0.98)],btype='band',output='sos'))
    pad=3*max(len(sos[0]),len(sos))
    z=sgn.sosfiltfilt(sos,y) if len(y)>pad else y-y.mean()
    k=int(fs);return (z[k:len(z)-k] if len(z)>2*k+8 else z),fs
def pearson(a,b):
    n=min(len(a),len(b))
    if np.std(a[:n])<1e-12 or np.std(b[:n])<1e-12: return 0.0
    return float(np.corrcoef(a[:n],b[:n])[0,1])
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
    for nm in ('W0_gate','W0_seed3','L30_A','L40_A','L15_B','L30_B','T150','T150_L30'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        cA=A[:,Ac].sum(1).astype(float);cB=A[:,Bc].sum(1).astype(float)
        za,_=box_env(cA);zb,_=box_env(cB)
        r=pearson(za,zb)
        rA=float(A[:,Ac].sum()*1000.0/(A.shape[0]*len(Ac)));rB=float(A[:,Bc].sum()*1000.0/(A.shape[0]*len(Bc)))
        vals=[]
        for i in range(NULLS):
            rng=np.random.default_rng(7777+i)
            a=rng.poisson(np.full(5000,max(rA/1000.0,1e-12))).astype(float);b=rng.poisson(np.full(5000,max(rB/1000.0,1e-12))).astype(float)
            vals.append(pearson(box_env(a)[0],box_env(b)[0]))
        v=np.array(vals)
        out[nm]=dict(rho=r,null_p1=float(np.percentile(v,1)),null_std=float(v.std()),nulls=NULLS,n_eff=int(v.size//2),
          phase=bool(r<np.percentile(v,1)-3*v.std()),rates=dict(A=rA,B=rB))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['rho'],3),round(v['null_p1'],3),v['phase']) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
