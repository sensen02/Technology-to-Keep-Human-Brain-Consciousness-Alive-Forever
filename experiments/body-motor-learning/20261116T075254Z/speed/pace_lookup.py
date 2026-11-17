"""Optional memoryless engineering pacing controller fitted as a geometry lookup."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

import pace_bench
from pace_extended import ContinuousLoop
from speed_teacher import NOMINAL_V, WARM, DT, CYCLE_START, REFERENCE_END, stand


class ObservedPhase(ContinuousLoop):
    def __init__(self,core=None):
        super().__init__(core)
        original = self.path[WARM:].copy()
        self.search_weights = 1./self.half
        self.paths = [original,self.periodic]
        parts,edges = [],[]
        for i,path in enumerate(self.paths):
            parts.append(path)
            edges.extend((i,j) for j in range(len(path)))
        self.tree = cKDTree(np.vstack(parts)*self.search_weights)
        self.lookup = tuple(edges)

    def phase_in(self,target):
        _, candidates = self.tree.query(target*self.search_weights,k=5)
        best = (float("inf"),0,0.)
        for k in candidates:
            part,index = self.lookup[k]
            path=self.paths[part]
            for start in (max(0,index-1),min(index,len(path)-2)):
                a,b=path[start:start+2]
                tangent=(b-a)*self.search_weights
                error=(target-a)*self.search_weights
                lengths=float(tangent@tangent)
                p= float(np.clip(tangent@error/lengths,0.,1.)) if lengths>1e-20 else 0.
                delta=error-p*tangent
                distance=float(delta@delta)
                if distance<best[0]:
                    best=(distance,part,start+p)
        return best[1:]

    def __call__(self,x,command):
        requested,yaw = np.asarray(command,dtype=float)
        if abs(yaw)>1e-10:
            raise ValueError("Straight pace decoder only")
        if requested==0.:
            return self.stock(x,0)
        target=x[126:168]*self.half + self.reference_zero
        part,phase=self.phase_in(target)
        future=phase+requested/NOMINAL_V*1000.*DT
        if part==0 and future<=REFERENCE_END-WARM:
            wanted=self.interpolate(self.paths[0],future)
        else:
            if part==0:
                future-=REFERENCE_END-WARM
            future%=(REFERENCE_END-CYCLE_START)
            wanted=self.interpolate(self.paths[1],future)
        return np.clip((wanted-target)/(DT*stand.physics.MAX_RATE),-1.,1.)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("plan",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    pace_bench.PhaseFollower=ObservedPhase
    pace_bench.roll(a.output.resolve(),a.plan.resolve())
