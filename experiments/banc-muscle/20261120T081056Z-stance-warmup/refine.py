"""Refine a bounded native inverse-dynamics balance by local active-set least squares."""
import sys
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.optimize import lsq_linear
from solve import Equilibrium, MODEL, body
from assay import sha
from run_body import moment_matrix


def read_seed(seed):
    if seed.suffix == '.json':
        saved = json.loads(seed.read_text())
        return float(saved['contact_ms']), np.array(saved['x'])
    with np.load(seed) as saved:
        return float(saved['contact_ms']), saved['x'].copy()


def refine(seed, out, constraints=None):
    ms,x=read_seed(seed)
    eq=Equilibrium(body(ms))
    n,m,d=eq.n,eq.m,eq.d
    def residual(v):
        balance=eq.residual(v,0)[:m.nv]
        extra=eq.footprint() if constraints is None else constraints(eq)
        return np.r_[balance,2*extra]
    rng=np.random.default_rng(601)
    best=x.copy()
    lowest=np.inf
    for step in range(1200):
        r=residual(x)
        cost=np.linalg.norm(r)
        if cost<lowest:
            lowest=cost
            best=x.copy()
        if step%100==0:
            print(step,max(abs(r)),flush=True)
        if max(abs(r))<1e-12:
            break
        J=np.zeros((len(r),len(x)))
        for i in range(n):
            v=x.copy();v[i]+=1e-8
            J[:,i]=(residual(v)-r)/1e-8
        eq.residual(x,0)
        moments=moment_matrix(m,d)
        d.act[:]=0
        mj.mj_fwdActuation(m,d); f=d.actuator_force.copy()
        d.act[:]=1
        mj.mj_fwdActuation(m,d)
        J[:m.nv,n:]=-moments.T*(d.actuator_force-f)/eq.obs.weight
        limit=np.r_[np.full(n,.014),np.full(m.nu,.25)]
        lo=np.maximum(eq.lo-x,-limit)
        hi=np.minimum(eq.hi-x,limit)
        delta=lsq_linear(np.vstack((J,np.diag(np.r_[np.full(n,.001),np.full(m.nu,.00005)]))),
                         np.r_[-r,np.zeros(len(x))],bounds=(lo,hi),method='bvls',tol=1e-13).x
        for alpha in 2.**-np.arange(22):
            candidate=x+alpha*delta
            new=residual(candidate)
            if np.linalg.norm(new)<np.linalg.norm(r)*(1-1e-13):
                x=candidate
                break
        else:
            alpha=0.
        if alpha<1e-5:
            # A near-tied supporting mesh vertex can change the contact set.
            # Re-seed near the best physical pose and recompute the native contacts.
            x=best.copy()
            x[:n]=np.clip(x[:n]+rng.normal(0,.00015,n),eq.lo[:n]+1e-9,eq.hi[:n]-1e-9)
    x=best
    r=eq.residual(x,0)[:m.nv]
    np.savez_compressed(out,x=x,qpos=d.qpos.copy(),act=x[n:],balance=r,contact_ms=ms)
    stance = dict(body_xml_sha256=sha(MODEL), floor_contact_time_ms=ms,
                  inverse_residual_max_per_weight=float(max(abs(r))),
                  qpos=d.qpos.tolist(), tonic_activation=x[n:].tolist(),
                  joint_order=[m.joint(j).name for j in range(m.njnt)],
                  actuator_order=[m.actuator(j).name for j in range(m.nu)])
    out.with_suffix('.json').write_text(json.dumps(stance, indent=2)+'\n')
    print('final',step,max(abs(r)),flush=True)

if __name__=='__main__':
    refine(Path(sys.argv[1]),Path(sys.argv[2]))
