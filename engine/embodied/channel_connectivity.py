"""Structural runtime coverage, never a substitute for functional/biological validation."""
from collections import deque
import numpy as np
from scipy import sparse


def channel_connectivity(we, wi, groups, motor_indices):
    if we.shape != wi.shape or len(we.shape)!=2 or we.shape[0]!=we.shape[1]:
        raise ValueError('matching square connectivity required')
    n=we.shape[0]
    # Logical union, never algebraic cancellation. Row=post, column=pre.
    graph=(sparse.csr_matrix(we).astype(bool)+sparse.csr_matrix(wi).astype(bool)).tocsc()
    graph.eliminate_zeros()
    motors=np.asarray(sorted(set(motor_indices)),dtype=int)
    if motors.size and (motors.min()<0 or motors.max()>=n):raise ValueError('motor index out of range')
    result={}
    for leg,channels in groups.items():
        result[leg]={}
        for name,indices in channels.items():
            seeds=np.asarray(sorted(set(int(i) for i in indices)),dtype=int)
            if seeds.size and (seeds.min()<0 or seeds.max()>=n):raise ValueError('sensory index out of range')
            hops=np.full(n,-1,dtype=int);hops[seeds]=0;queue=deque(seeds.tolist())
            while queue:
                source=queue.popleft()
                for target in graph.indices[graph.indptr[source]:graph.indptr[source+1]]:
                    if hops[target]<0:
                        hops[target]=hops[source]+1;queue.append(int(target))
            reaches=hops[motors];reachable=reaches[reaches>=0]
            result[leg][name]=dict(cells=len(seeds),outgoing_nonzero=int(graph[:,seeds].count_nonzero()),
                reachable_cells=int((hops>=0).sum()),reachable_motor_cells=len(reachable),
                minimum_motor_hops=int(reachable.min()) if reachable.size else None,
                status='NO_TIER_CELLS' if not seeds.size else 'NO_MODELED_MOTOR_PATH' if not reachable.size else 'STRUCTURAL_PATH_ONLY')
    return dict(orientation='postsynaptic rows, presynaptic columns',channels=result,
                limitation='Reachability is structural only. Missing signed edges may reflect unknown transmitter annotations. No weights or labels changed.')
