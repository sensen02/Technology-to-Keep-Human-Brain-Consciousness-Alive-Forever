"""Balance the free torso before allocating joint torques to its native muscles."""
import json
import sys
from pathlib import Path

from scipy.optimize import least_squares, lsq_linear
from solve import Equilibrium, body, np, mj, MODEL
from run_body import moment_matrix
from assay import sha


def solve(seed,out):
    with np.load(seed) as z:
        x=z['x'].copy()
        ms=z['contact_ms'].item()
    e=Equilibrium(body(ms))
    d,m,n=e.d,e.m,e.n
    idx=[0,n-2,n-1]
    def residual(y):
        x[idx]=y
        return e.residual(x,0)[[2,3,4]]
    fit=least_squares(residual,x[idx],xtol=1e-14,gtol=1e-14,ftol=1e-14,max_nfev=100)
    residual(fit.x)
    moments=moment_matrix(m,d)
    d.act[:]=0
    mj.mj_fwdActuation(m,d);base=d.qfrc_actuator.copy();f=d.actuator_force.copy()
    d.act[:]=1
    mj.mj_fwdActuation(m,d)
    muscle=moments.T*(d.actuator_force-f)
    trim=lsq_linear(muscle[6:],d.qfrc_inverse[6:]-base[6:],bounds=(.0001,1.),method='bvls',tol=1e-13)
    x[n:]=trim.x
    raw=e.residual(x,0)[:m.nv]*e.obs.weight
    print(out, 'root',raw[:6], 'joint',max(abs(raw[6:])), 'norm',np.linalg.norm(raw[6:]))
    result=dict(body_xml_sha256=sha(MODEL),floor_contact_time_ms=ms,qpos=d.qpos.tolist(),
                tonic_activation=x[n:].tolist(),warm_teacher_hinge_trim=raw[6:].tolist(),
                inverse_residual_max_per_weight=float(max(abs(raw))/e.obs.weight),
                joint_order=[m.joint(j).name for j in range(m.njnt)],actuator_order=[m.actuator(j).name for j in range(m.nu)])
    out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    solve(Path(sys.argv[1]),Path(sys.argv[2]))
