"""Finite-actuator path follower for continuous, physically explicit pace commands."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "20261116T000536Z"
sys.path.insert(0,str(SOURCE))
from walking_bench import stand
from prototype import policy as stock_policy

REFERENCE = SOURCE / "PROTO_nominal" / "TRACE.npz"
NOMINAL_V = .07171300063848401
DT = .005
WARM = 500
REFERENCE_END = 10100
CYCLE_START = 5300
RANGE = (0.,1.24)


class PhaseFollower:
    """Stateful geometric teacher. Inputs use native speed and yaw radians/s."""
    def __init__(self,core=None):
        model = stand.physics.act_mod.load_full_model(stand.physics.PAD_XML) if core is None else core.plant
        ids = model.actuator_trnid[:,0]
        self.reference_zero = model.key_qpos[0][model.jnt_qposadr[ids]].copy()
        self.half = np.maximum(np.ptp(model.jnt_range[ids],axis=1)/2,.1)
        self.core = core
        with np.load(REFERENCE,allow_pickle=False) as archive:
            self.path = archive["target"][:REFERENCE_END+1].copy()
        # Whole-cycle repeats only after initial phase, with a small periodic splice.
        segment = self.path[CYCLE_START:].copy()
        u = np.linspace(0.,1.,len(segment))
        difference=segment[-1]-segment[0]
        segment -= (u*u*(3.-2.*u))[:,None]*difference
        self.periodic=segment
        self.phase = float(WARM)
        self.stock = stock_policy()

    @staticmethod
    def interpolate(path, phase):
        i = min(int(phase),len(path)-2)
        r = phase-i
        return (1.-r)*path[i]+r*path[i+1]

    def __call__(self,x,command):
        requested,yaw = np.asarray(command,dtype=float)
        if abs(yaw)>1.e-10:
            raise ValueError("Straight pace teacher only")
        if requested==0.:
            return self.stock(x,0)
        factor = requested/NOMINAL_V
        if not RANGE[0]<factor<=RANGE[1]:
            raise ValueError("Forward pace command outside configured teacher range")
        self.phase += factor*1000.*DT
        if self.phase<=REFERENCE_END:
            target=self.interpolate(self.path,self.phase)
        else:
            phase=(self.phase-REFERENCE_END)%(REFERENCE_END-CYCLE_START)
            target=self.interpolate(self.periodic,phase)
        own_target=(x[126:168]*self.half + self.reference_zero
                    if self.core is None else self.core.actuator.target_rad)
        return np.clip((target-own_target)/(DT*stand.physics.MAX_RATE),-1.,1.)
