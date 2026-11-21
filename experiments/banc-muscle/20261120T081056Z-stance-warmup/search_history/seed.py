"""Bounded vertical-force seed, solved offline and then checked with real floor contacts."""
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from solve import Equilibrium, body, HERE


def solve(output, height_delta=0.):
    eq = Equilibrium(body())
    m,d,n = eq.m,eq.d,eq.n
    jp=np.empty((3,m.nv))
    lo=np.r_[eq.lo,np.full(6,.04)]
    hi=np.r_[eq.hi,np.full(6,.40)]
    def residual(x,reg):
        eq.set_pose(x[:n])
        d.act[:]=x[n:n+m.nu]
        mj.mj_fwdPosition(m,d)
        mj.mj_fwdVelocity(m,d)
        mj.mj_fwdActuation(m,d)
        force=(d.qfrc_bias-d.qfrc_passive-d.qfrc_actuator)/eq.obs.weight
        gaps=[]
        for idx,(g,verts) in enumerate(zip(eq.ids,eq.verts,strict=True)):
            mat=d.geom_xmat[g].reshape(3,3)
            z=verts@mat[2]
            # smooth local contact point during optimization
            w=np.exp(-(z-z.min())/.003)
            point=d.geom_xpos[g]+mat@(w@verts / w.sum())
            gaps.append(d.geom_xpos[g,2]+z.min()+.003)
            mj.mj_jac(m,d,jp,None,point,m.geom_bodyid[g])
            force-=jp[2]*x[-6+idx]
        return np.r_[force,4*np.array(gaps),2*eq.footprint(),reg*(x[:n]-eq.reference),
                     reg*.1*(x[n:n+m.nu]-.25),reg*.3*(x[-6:]-1/6)]
    x=np.clip(np.r_[eq.reference,np.full(m.nu,.25),np.full(6,1/6)],lo+1e-9,hi-1e-9)
    x[0]+=height_delta
    for reg in (.02,.002):
        def jac(v):
            orig=residual(v,reg)
            J=np.zeros((len(orig),len(v)))
            for i in [*range(n),*range(len(v)-6,len(v))]:
                w=v.copy();w[i]+=1e-7
                J[:,i]=(residual(w,reg)-orig)/1e-7
            residual(v,reg)
            from run_body import moment_matrix
            moments=moment_matrix(m,d)
            d.act[:]=0
            mj.mj_fwdActuation(m,d); f=d.actuator_force.copy()
            d.act[:]=1
            mj.mj_fwdActuation(m,d)
            J[:m.nv,n:n+m.nu]=-moments.T*(d.actuator_force-f)/eq.obs.weight
            J[-6-m.nu:-6,n:n+m.nu]=np.eye(m.nu)*reg*.1
            return J
        opt=least_squares(lambda v:residual(v,reg), x, jac=jac, bounds=(lo,hi),
                          max_nfev=260, gtol=1e-11)
        x=opt.x
        r=residual(x,reg)
        print(reg,max(abs(r[:m.nv])),x[-6:],flush=True)
    np.savez_compressed(output,x=x[:-6],qpos=d.qpos,act=x[n:n+m.nu],vertical_load_guess=x[-6:])


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'geometry_seed.npz')
    parser.add_argument('--height-delta',type=float,default=0.)
    args=parser.parse_args()
    solve(args.output,args.height_delta)
