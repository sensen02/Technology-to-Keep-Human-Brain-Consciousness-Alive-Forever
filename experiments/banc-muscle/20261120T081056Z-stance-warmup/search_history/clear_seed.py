"""Separate neighbouring legs while keeping the six foot-ground distances near their targets."""
import itertools
import sys
from pathlib import Path
from scipy.optimize import least_squares
from solve import Equilibrium, body, mj, np
from select_pose import LEG_PREFIXES,SEGMENTS


def clear(seed,dest):
    e=Equilibrium(body(1.))
    m,d=e.m,e.d
    z=np.load(seed)
    total=z['x'].copy()
    x=total[:e.n].copy()
    geoms=[g for g in range(m.ngeom) if m.geom(g).name[:2] in LEG_PREFIXES
           and any(s in m.geom(g).name for s in SEGMENTS)]
    pairs=[(i,j) for i,j in itertools.combinations(geoms,2) if m.geom(i).name[:2]!=m.geom(j).name[:2]]
    def update(y):
        e.set_pose(y)
        mj.mj_kinematics(m,d)
    def foot_heights():
        return np.array([d.geom_xpos[g,2]+min(v@d.geom_xmat[g,6:9]) for g,v in zip(e.ids,e.verts)])
    update(x)
    target=foot_heights()
    ref=x.copy()
    for phase in range(3):
        update(x)
        near=[(i,j) for i,j in pairs if mj.mj_geomDistance(m,d,i,j,.1,None)<.06]
        def fun(y):
            update(y)
            gap=np.array([mj.mj_geomDistance(m,d,i,j,.1,None) for i,j in near])
            local=(d.geom_xpos[e.foot_geoms]-d.xpos[e.obs.thorax])@d.xmat[e.obs.thorax].reshape(3,3)
            return np.r_[3*np.minimum(gap-.006,0.),3*(foot_heights()-target),.002*(y-ref),
                         np.minimum(local[:,1]*e.signs-.12,0),min(0,d.xmat[e.obs.thorax,8]-.96)]
        res=least_squares(fun,x,bounds=(e.lo[:e.n],e.hi[:e.n]),max_nfev=100)
        x=res.x
        update(x)
        mingap=min(mj.mj_geomDistance(m,d,i,j,.1,None) for i,j in pairs)
        print('clear',phase,mingap,flush=True)
        if mingap>.002 and max(abs(foot_heights()-target))<.0001: break
    total[:e.n]=x
    np.savez_compressed(dest,x=total,qpos=d.qpos,contact_ms=1.)
if __name__=='__main__':
    clear(Path(sys.argv[1]),Path(sys.argv[2]))
