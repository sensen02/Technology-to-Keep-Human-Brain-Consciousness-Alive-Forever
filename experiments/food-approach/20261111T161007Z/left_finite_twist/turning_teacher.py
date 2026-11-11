"""Finite SE2 horizontal-only candidate; no simulator imports/construction at import.

The original conditional teacher is loaded by absolute path and never edited.
This candidate deliberately only supports speed_scale=1. Actual teacher qdot
continues to come from StaticTeacher's IK q_des differences, not velocity_xy.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'attempt02' / 'turning_teacher.py'
_spec = importlib.util.spec_from_file_location('_left_finite_original_teacher', LEGACY)
_legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _legacy
_spec.loader.exec_module(_legacy)
BASELINE = _legacy.BASELINE
pin_baseline = _legacy.pin_baseline


def finite_xy_velocity(rest, target, velocity, k, kdot):
    """Analytic d/dt of R(ks)r + integral_0^s R(ku)ex du + ell ey.

    Translation derivatives: dT/ds=(cos(theta),sin(theta));
    dT/dk=s^2*(sinc'(theta), ((1-cos(theta))/theta)').
    Taylor branch |theta|<1e-3 avoids subtractive cancellation in k derivative.
    Zero displacement assigns the original target exactly.
    """
    rest = np.asarray(rest, dtype=float)
    target = np.asarray(target, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    s, ell = (target-rest).T
    sdot, elldot = velocity.T
    theta = k*s
    ct, st = np.cos(theta), np.sin(theta)
    rr = np.column_stack([ct*rest[:, 0]-st*rest[:, 1],
                          st*rest[:, 0]+ct*rest[:, 1]])
    trans = np.column_stack([s*np.sinc(theta/np.pi),
                            .5*k*s*s*np.sinc(theta/(2*np.pi))**2])
    p = rr+trans
    p[:, 1] += ell
    p[s == 0.] = target[s == 0.]
    ap, bp = np.empty_like(theta), np.empty_like(theta)
    small = np.abs(theta) < 1e-3
    z = theta[small]
    ap[small] = -z/3.+z**3/30.-z**5/840.+z**7/45360.
    bp[small] = .5-z*z/8.+z**4/144.-z**6/5760.
    z = theta[~small]
    ap[~small] = (z*np.cos(z)-np.sin(z))/(z*z)
    bp[~small] = (z*np.sin(z)-2*np.sin(z/2.)**2)/(z*z)
    theta_dot = kdot*s+k*sdot
    pdot = theta_dot[:, None]*np.column_stack([-rr[:, 1], rr[:, 0]])
    pdot += sdot[:, None]*np.column_stack([ct, st])
    pdot += (kdot*s*s)[:, None]*np.column_stack([ap, bp])
    pdot[:, 1] += elldot
    return p, pdot


class FiniteTwistReference(_legacy.ConditionalTwistReference):
    """Only sample horizontal mapping differs; inherited command is unchanged."""
    def __init__(self, base, speed_scale=1., yaw_rate=.06, prime_s=4.8, ramp_s=1.):
        if speed_scale != 1.:
            raise ValueError('Preregistered finite candidate only supports speed_scale=1')
        super().__init__(base, speed_scale, yaw_rate, prime_s, ramp_s)
        self.metadata = dict(self.metadata, label='algorithm teacher finite SE2 horizontal twist',
                             horizontal_mapping='R(k*s)r + [s*sinc(k*s), .5*k*s^2*sinc(k*s/2)^2] + ell*ey',
                             velocity='analytic chain derivative with kdot; Taylor near zero',
                             changed_fields=['target_xy', 'velocity_xy'],
                             hypothesis_only=True)

    def sample(self, elapsed_s):
        phase, scale, k = self.command(elapsed_s)
        sample = self.base.sample(phase)
        # With speed_scale1, k=0 must preserve every field bit-for-bit.
        if k == 0.:
            return sample
        u = np.clip((elapsed_s-self.prime_s)/self.ramp_s, 0., 1.)
        kdot = self.curvature*6.*u*(1.-u)/self.ramp_s
        sample['target_xy'], sample['velocity_xy'] = finite_xy_velocity(
            self.rest, sample['target_xy'], sample['velocity_xy'], k, kdot)
        return sample


def make_teacher(core, *, speed_scale=1., yaw_rate=.06, prime_s=4.8):
    pin_baseline()
    from static_teacher import StaticTeacher
    from student_wave import teacher_config
    teacher = StaticTeacher(core, teacher_config(100))
    teacher.reference = FiniteTwistReference(teacher.reference, speed_scale, yaw_rate, prime_s)
    teacher.metadata['conditional_twist'] = teacher.reference.metadata
    return teacher
