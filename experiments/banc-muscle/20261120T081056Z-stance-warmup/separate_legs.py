"""Add explicit convex distances for close inter-leg mesh pairs during static fitting."""
import itertools
import sys
from pathlib import Path
from refine import read_seed, refine
from solve import LEG_PREFIXES, SEGMENTS
from solve import Equilibrium, HERE, body, mj, np, load_pose


def separate(seed,out):
    e=Equilibrium(body(1.))
    m=e.m
    geoms=[g for g in range(m.ngeom) if m.geom(g).name[:2] in LEG_PREFIXES
           and any(s in m.geom(g).name for s in SEGMENTS)]
    pairs=[(i,j) for i,j in itertools.combinations(geoms,2) if m.geom(i).name[:2]!=m.geom(j).name[:2]]
    _,initial=read_seed(seed)
    near=set()
    for k in np.linspace(0.,1.,6):
        x=(1-k)*e.reference+k*initial[:e.n]
        e.set_pose(x)
        mj.mj_forward(m,e.d)
        near.update((i,j) for i,j in pairs if mj.mj_geomDistance(m,e.d,i,j,.15,None)<.09)
    near=sorted(near)
    def footprint(self):
        d=self.d
        local=(d.geom_xpos[self.foot_geoms]-d.xpos[self.obs.thorax])@d.xmat[self.obs.thorax].reshape(3,3)
        margins=local[:,1]*self.signs
        gaps=np.array([mj.mj_geomDistance(self.m,d,i,j,.08,None) for i,j in near])
        loads=self.obs.measure(d)['ground_load_w'][:6]
        return np.r_[np.maximum(.11-margins,0),3*np.maximum(.003-gaps,0),
                     np.maximum(np.array([.065,.025,.025,.065,.025,.025])-loads,0),
                     max(0.,.955-d.xmat[self.obs.thorax,8])]
    refine(seed,out,constraints=footprint)


if __name__=='__main__':
    separate(Path(sys.argv[1]),Path(sys.argv[2]))
