"""Export the trained deltas with an explicit source root, target root or actuator name."""
import json
import numpy as np
from bench import BASE, HERE, Interfaces, body, sha


def main():
    io=Interfaces(body(1.))
    selected=HERE/'CANDIDATE.npz'
    archive=dict(source_checkpoint=sha(BASE),trained_checkpoint=sha(selected))
    with np.load(BASE) as parent,np.load(selected) as current:
        for sign,matrix in zip(('exc','inh'),io.network):
            key=sign+'_fraction'
            frac=current[key]
            indices=np.flatnonzero(frac!=parent[key])
            rows=np.searchsorted(matrix.indptr,indices,side='right')-1
            cols=matrix.indices[indices]
            archive.update({sign+'_pre':io.roots[cols],sign+'_post':io.roots[rows],
                            sign+'_base_uS':matrix.data[indices],sign+'_fraction':frac[indices],
                            sign+'_trained_uS':matrix.data[indices]*(1+frac[indices])})
        for key,labels in [('motor_gain',['muscle','motor_root','exact_candidate']),
                           ('feco_gain',['feco_root','feco_uses_angle']),('vnc_gain_uS',['vnc_pre','vnc_post'])]:
            mask=parent[key]!=current[key]
            archive[key+'_old']=parent[key][mask]
            archive[key+'_new']=current[key][mask]
            for label in labels:
                archive[label]=current[label][mask]
    np.savez_compressed(HERE/'CONNECTION_CHANGES.npz',**archive)


if __name__=='__main__':
    main()
