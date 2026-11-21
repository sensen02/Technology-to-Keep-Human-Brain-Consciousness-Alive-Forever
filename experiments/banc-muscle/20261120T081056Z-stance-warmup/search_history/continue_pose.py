"""Start from the tight physical balance and progressively separate the hind and front feet."""
from refine import refine
from solve import Equilibrium, HERE, np
import sys
from pathlib import Path


def progress(out):
    out.mkdir(exist_ok=True,parents=True)
    with np.load(HERE/'narrow_candidate/refined.npz') as a:
        start=np.r_[a['x'][:43],0.,0.,a['x'][43:]]
    prior=out/'initial.npz'
    np.savez_compressed(prior,x=start,contact_ms=1.)
    for stage,(side, spacing, forehind, midhind) in enumerate([
            (.025,.12,.10,.04),(.065,.155,.13,.075),(.105,.18,.17,.10),
            (.15,.21,.23,.14),(.20,.24,.30,.18),(.24,.25,.40,.22)]):
        def footprint(self):
            d=self.d
            thor=d.xmat[self.obs.thorax].reshape(3,3)
            local=(d.geom_xpos[self.foot_geoms]-d.xpos[self.obs.thorax])@thor
            xy=d.geom_xpos[self.ids,:2]
            distance=np.array([np.linalg.norm(xy[i]-xy[j]) for i,j in ((0,1),(1,2),(0,2),(3,4),(4,5),(3,5))])
            return np.maximum(np.r_[side-local[:,1]*self.signs,side-d.geom_xpos[self.foot_geoms,1]*self.signs,
                                  forehind-xy[[0,3],0]+xy[[2,5],0],midhind-xy[[1,4],0]+xy[[2,5],0],spacing-distance],0)
        Equilibrium.footprint=footprint
        dest=out/f'stage{stage}.npz'
        refine(prior,dest)
        prior=dest
        if max(abs(np.load(dest)['balance']))>1e-7:
            break


if __name__=='__main__':
    progress(Path(sys.argv[1]))
