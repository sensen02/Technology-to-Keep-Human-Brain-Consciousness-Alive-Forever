"""Physical diagnostic and learned ANN entrypoint for the v3 FK reference.
Parent must inspect FK results before execution; this script does not claim FKpass.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T073258Z')
import neural_walk as n
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075536Z')
n.GAIT=ROOT/'GAIT.npz'
n.OUT=ROOT/'neural'
n.FREQ=1.  # Initial slow diagnostic, within unchanged target-rate limits.
OriginalReference=n.Reference


class BoundedReference(OriginalReference):
    def __init__(self,p):
        self.lo=p.m.jnt_range[p.jids,0]
        self.hi=p.m.jnt_range[p.jids,1]
        self.q0=p.q0[p.qa].copy()
        with np.load(n.GAIT) as z:
            assert np.array_equal(z['qpos_addresses'],p.qa)
            phase=z['phase'];q=z['qpos_hinges']
            xp=np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi]
            qp=np.concatenate((q[-3:-1],q,q[1:3]),axis=0)
            self.spline=PchipInterpolator(xp,qp,axis=0,extrapolate=False)
        grid=self.at(np.linspace(0,2*np.pi,4096,endpoint=False))
        self.center=(grid.max(0)+grid.min(0))/2
        self.scale=np.maximum((grid.max(0)-grid.min(0))/2,.025)

    def at(self,phase):
        q=self.spline(np.asarray(phase)%(2*np.pi))
        assert np.isfinite(q).all()
        assert np.all(q>=self.lo-1e-9) and np.all(q<=self.hi+1e-9), 'reference violates native joint bounds'
        return q  # Do not clip invalid interpolation.


n.Reference=BoundedReference

if __name__=='__main__':
    assert n.GAIT.exists(),n.GAIT
    n.prereg()
    protocol=n.OUT/'REFERENCE_PROTOCOL.json'
    if not protocol.exists():
        n.dump(protocol,dict(reference=str(n.GAIT),sha256=n.sha(n.GAIT),
            interpolation='periodic two-knot-extended PCHIP, explicit modulo phase, no clipping, matching denseFK',
            frequency_hz=n.FREQ,
            controller='replacement ANN after1s in ann mode; continuous algorithm teacher only when explicitly requested',
            runtime=str(Path(n.__file__)),runtime_sha256=n.sha(n.__file__),
            driver=str(Path(__file__)),driver_sha256=n.sha(__file__),
            body='unchanged full free-root71mesh, normal original gravity/contact and original torque caps',
            readiness='Parent must review FK and record limitations before physical execution; FK alone is not walking'))
    if len(sys.argv)>1 and sys.argv[1]=='teacher':
        seconds=float(sys.argv[2]) if len(sys.argv)>2 else 4.
        tag=sys.argv[3] if len(sys.argv)>3 else 'teacher_4s'
        old=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
        # Oldmodel supplies inert logger history only, its actions are never used
        # in teacher mode; independent gait audit needs --frequency 1 here.
        n.simulate(old,tag,seconds,'teacher',False)
    else:
        n.main()
