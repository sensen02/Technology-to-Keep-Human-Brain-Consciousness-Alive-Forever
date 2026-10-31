"""Independent second implementation for S341: order 6 lowpass at 12 Hz, no decimation, circular shift null."""
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
NULLS=1000;FSKEEP=1000.0
def env(y,order=6,cut=12.0):
    b,a=sgn.butter(order,cut/(FSKEEP/2),btype='low');pad=3*max(len(a),len(b))
    z=sgn.filtfilt(b,a,y) if len(y)>pad else y-y.mean()
    bb,aa=sgn.butter(4,[0.6/(FSKEEP/2),3.5/(FSKEEP/2)],btype='band')
    return sgn.filtfilt(bb,aa,z) if len(z)>pad else z-z.mean()
def rho(a,b):
    n=min(len(a),len(b));xa=a[:n];xb=b[:n]
    if np.std(xa)<1e-12 or np.std(xb)<1e-12: return 0.0
    return float(np.corrcoef(xa,xb)[0,1])
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
    rng=np.random.default_rng(31337);out={}
    for nm in ('A0_gate','U5','U10','U15','U20','U10_seed1','U10b_symA','T75_recheck'):
        f=o/f'SPK_{nm}.npz'
        if not f.is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        cA=A[:,Ac].sum(1).astype(float);cB=A[:,Bc].sum(1).astype(float)
        za=env(cA);zb=env(cB)
        r=rho(za,zb)
        vals=[]
        n=min(len(za),len(zb))
        for _ in range(NULLS):
            k=int(rng.integers(1000,n-1))
            vals.append(rho(za[:n-1000],np.roll(zb[:n-1000],k)))
        p1=float(np.percentile(vals,1))
        rA=float(A[:,Ac].sum()*1000.0/(A.shape[0]*len(Ac)));rB=float(A[:,Bc].sum()*1000.0/(A.shape[0]*len(Bc)))
        ratio=max(rA,rB)/max(min(rA,rB),1e-9)
        phase=bool(r<p1)
        if not phase and ratio>3: lab='WINNER_TAKE_ALL'
        elif phase: lab='ALTERNATING'
        else: lab='RHYTHM_IN_PHASE'
        out[nm]=dict(rho=r,circ_null_p1=p1,label=lab,rates=dict(A=rA,B=rB,ratio=ratio))
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(round(v['rho'],3),round(v['circ_null_p1'],3),v['label']) for k,v in out.items()},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
