"""Independent second implementation for S329: pairwise envelope correlation with a direct loop."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
S327=Path('/home/sensen/Desktop/engine修复验收/s327-global-mode-and-motif-influence/20261027T004101Z')
S328=Path('/home/sensen/Desktop/engine修复验收/s328-mode-origin-and-coupling-budget/20261027T005128Z')
LEGS=('lf','lm','lh','rf','rm','rh');NULLS=1000;SEED=707070
def env(x,ds=2):
    y=np.convolve(x.astype(float),np.ones(20)/20,'same');t=np.arange(y.size)
    y=y-np.polyval(np.polyfit(t,y,1),t);return y[::ds]
def direct_mean_pair(Y):
    n,k=Y.shape;Z=(Y-Y.mean(0))/np.where(Y.std(0)<1e-12,1e-12,Y.std(0))
    C=np.corrcoef(Z.T);iu=np.triu_indices(k,1)
    return float(C[iu].mean())
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());cols=[j for (_l,_g),j in pool]
    by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    out={}
    for nm,f in (('B0',S327/'SPK_B0_repeat.npz'),('C_norec',S328/'SPK_C_norec.npz'),('P1',o/'SPK_P1_legI_asym.npz'),('P2',o/'SPK_P2_motif_out_gain.npz')):
        if not Path(f).is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        Y=np.stack([env(A[:,j]) for j in cols],1)
        obs=direct_mean_pair(Y)
        rng=np.random.default_rng(SEED);null=[]
        for _ in range(NULLS):
            Ys=np.stack([np.roll(Y[:,i],int(rng.integers(0,Y.shape[0]))) for i in range(Y.shape[1])],1)
            null.append(direct_mean_pair(Ys))
        null=np.array(null)
        E={lg:env(A[:,by[lg]].sum(1)) for lg in LEGS}
        pairs={f'{a}|{b}':float(np.corrcoef(E[a],E[b])[0,1]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        out[nm]=dict(D1_direct=obs,D1_null_p95=float(np.percentile(null,95)),D1_p=float((null>=obs).sum()+1)/(len(null)+1),
          mean_cross_leg=float(np.mean(list(pairs.values()))),pairs=pairs,nulls=NULLS)
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(seed=SEED,result=out),indent=2)+'\n')
    print(json.dumps({k:(round(v['D1_direct'],4),round(v['D1_null_p95'],4),round(v['mean_cross_leg'],3)) for k,v in out.items()}))
    return 0
if __name__=='__main__': raise SystemExit(main())
