"""Independent second implementation for S330: budget table and mean cross-leg correlation."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np, scipy.sparse as sps
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
S290=Path('/home/sensen/Desktop/engine修复验收/s290-motor-mapping-coverage/20261026T174500Z')
S303=Path('/home/sensen/Desktop/engine修复验收/s303-readout-graded-window/20261026T201133Z')
S306=Path('/home/sensen/Desktop/engine修复验收/s306-cpg-source-audit/20261026T203904Z')
S307=Path('/home/sensen/Desktop/engine修复验收/s307-cpg-per-leg-audit/20261026T204344Z')
S322=Path('/home/sensen/Desktop/engine修复验收/s322-design-point-motif/20261026T222837Z')
LEGS=('lf','lm','lh','rf','rm','rh');ROLES=('E1','E2','I1');NULLS=1000;SEED=808080
def csr(z):
    n=int(z['indptr'].size-1)
    return sps.csr_matrix((np.asarray(z['data'],float),np.asarray(z['indices'],int),np.asarray(z['indptr'],int)),shape=(n,n))
def env(x):
    y=np.convolve(x.astype(float),np.ones(20)/20,'same');t=np.arange(y.size)
    y=y-np.polyval(np.polyfit(t,y,1),t);return y[::2]
def cross_mean(Y):
    n,k=Y.shape;Z=(Y-Y.mean(0))/np.where(Y.std(0)<1e-12,1e-12,Y.std(0))
    C=np.corrcoef(Z.T);iu=np.triu_indices(k,1);return float(C[iu].mean())
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output
    roots=np.load(S306/'TIER_ROOT_IDS.npz')['root_ids'];loc={int(r):i for i,r in enumerate(roots)}
    mod=json.loads((S307/'PER_LEG_MODULE.json').read_text())['modules']
    M18={lg:[loc[int(c)] for r in ROLES for c in mod[lg][r]] for lg in LEGS}
    seen={}
    for r in csv.DictReader(open(S290/'PER_CHANNEL.csv')):
        lg=(r.get('name') or '')[:2].lower()
        if lg in LEGS and r.get('pool_root_ids'):
            for x in json.loads(r['pool_root_ids']):
                if int(x) in loc: seen.setdefault((lg,int(x)),loc[int(x)])
    pool=sorted(seen.items());by={lg:[j for (l2,gi),j in pool if l2==lg] for lg in LEGS}
    We=csr(np.load(S303/'EDGES_w_2.5e-5_We.npz'));Wi=csr(np.load(S303/'EDGES_w_2.5e-5_Wi.npz'))
    budget={}
    for lg in LEGS:
        pc=by[lg];own=sorted(M18[lg])
        e_tot=float(We[pc,:].sum());e_mine=float(We[pc,:][:,own].sum())
        cross=0
        for lg2 in LEGS:
            if lg2==lg: continue
            cross+=int(We[pc,:][:,sorted(M18[lg2])].nnz)
        budget[lg]=dict(share=e_mine/max(e_tot,1e-12),poolexc_uS=e_tot,motifE_uS=e_mine,we_edges=int(We[pc,:][:,own].nnz),cross=cross)
    out={}
    for nm,f in (('G_cn_gate',o/'SPK_G_cn_gate.npz'),('Kout8.67',o/'SPK_Kout8.67.npz'),('Kout17.33',o/'SPK_Kout17.33.npz'),('Kout34.66',o/'SPK_Kout34.66.npz'),('ISO90',o/'SPK_ISO90.npz'),('V0_seed1',o/'SPK_V0_seed1.npz')):
        if not Path(f).is_file(): continue
        z=np.load(f);A=np.unpackbits(z['packed'],axis=1)[:,:int(z['shape'][1])]
        E={lg:env(A[:,by[lg]].sum(1)) for lg in LEGS}
        pairs={f'{a}|{b}':float(np.corrcoef(E[a],E[b])[0,1]) for i,a in enumerate(LEGS) for b in LEGS[i+1:]}
        Y=np.stack([env(A[:,j]) for j in [j for (_l,_g),j in pool]],1)
        obs=cross_mean(Y);rng=np.random.default_rng(SEED);null=[]
        for _ in range(NULLS):
            null.append(cross_mean(np.stack([np.roll(Y[:,i],int(rng.integers(0,Y.shape[0]))) for i in range(Y.shape[1])],1)))
        out[nm]=dict(mean_cross_leg=float(np.mean(list(pairs.values()))),pairs=pairs,
          D1=obs,D1_null_p95=float(np.percentile(null,95)),nulls=NULLS)
    (o/'AUDIT_INDEPENDENT.json').write_text(json.dumps(dict(seed=SEED,budget=budget,result=out),indent=2)+'\n')
    print(json.dumps(dict(budget={k:round(v['share'],4) for k,v in budget.items()},
      cross={k:round(v['mean_cross_leg'],3) for k,v in out.items()}),ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
