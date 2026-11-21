"""Two finite sweeps over motor-unit gains and previously untested active synapse subsets."""
import argparse
from pathlib import Path
import numpy as np
import refine as r
from bench import HERE, Bench


def main(start,out,sweep):
    r.B=Bench()
    r.OUT=out
    out.mkdir()
    with np.load(start) as f:
        r.P={k:f[k] for k in f.files}
    r.OBJECTIVE='handoff'
    r.HORIZON=250.
    b=r.B
    arms=b.motor_rows
    base_seed=24680+100*sweep
    rng=np.random.default_rng(base_seed)
    for j in range(3):
        # Scale motor pairs by active root, while freezing all pairs outside these two legs.
        idx=np.flatnonzero((arms//15)%3==j)
        coords=rng.permutation(364)[b.motor_cols[idx]]%22
        r.batch('motor_gain',idx,coords,'units'+str(j),base_seed+j,8,.55/(1+.25*sweep))
    for j,field in enumerate(('inh_fraction','exc_fraction')):
        prep=b.prepare(r.P)
        w=b.io.network[1-j]
        pre=w.indices
        post=np.repeat(np.arange(w.shape[0]),np.diff(w.indptr))
        score=prep[3][pre]*w.data
        eligible=np.flatnonzero(score>0)
        # Sample reproducible disjoint-within-block subsets biased towards frequently used edges.
        ids=rng.choice(eligible,size=min(len(eligible),6144),replace=False,p=np.sqrt(score[eligible])/np.sqrt(score[eligible]).sum())
        groups=pre[ids]%18
        r.batch(field,ids,groups,field,base_seed+3+j,5,.85)
    for j in range(3):
        idx=np.flatnonzero(arms//30==j)
        r.batch('motor_gain',idx,arms[idx]%30,'arms'+str(j),base_seed+5+j,7,.32)
    np.savez_compressed(out/'trained.npz',**r.P)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('start',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('sweep',type=int)
    a=parser.parse_args()
    main(a.start,a.out,a.sweep)
