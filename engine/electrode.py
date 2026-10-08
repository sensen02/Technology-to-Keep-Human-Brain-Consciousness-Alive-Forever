"""Finite spherical volume contacts in infinite homogeneous ohmic medium.

A uniformly distributed spherical current source regularizes the point-source
singularity. This is NOT a metal equipotential electrode, FEM anatomy, or an
interface electrochemistry model. Conductivity S/m, coordinates/radius um,
current nA, potential mV. Source and return currents must sum to zero.
Analytic quasi-static Poisson solution; reference is an explicit observation
point. All chosen geometry/conductivity values are illustrative.
"""
from dataclasses import dataclass
import numpy as np


def points(value):
    a=np.asarray(value,dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError('expected finite (n,3) coordinates in um')
    return a


@dataclass(frozen=True)
class Contact:
    center_um: tuple
    radius_um: float
    def __post_init__(self):
        p=points([self.center_um])[0]
        if not np.isfinite(self.radius_um) or self.radius_um<=0:
            raise ValueError('positive finite radius required')
        object.__setattr__(self,'center_um',tuple(p))
        object.__setattr__(self,'radius_um',float(self.radius_um))


def transfer_mV_per_nA(xyz_um, contacts, conductivity_S_m):
    """Green function for uniform spherical VOLUME sources; units mV/nA."""
    xyz=points(xyz_um)
    sigma=float(conductivity_S_m)
    if not np.isfinite(sigma) or sigma<=0:
        raise ValueError('positive finite conductivity required')
    out=np.empty((len(xyz),len(contacts)))
    for k,c in enumerate(contacts):
        r=np.linalg.norm(xyz-np.asarray(c.center_um),axis=1)
        a=c.radius_um
        # nA/um -> 1e-3 V = mV, so no extra factor is needed.
        out[:,k]=np.where(r<a,(3-(r/a)**2)/(2*a),1/np.maximum(r,a))/(4*np.pi*sigma)
    return out


class BipolarField:
    def __init__(self, contacts, conductivity_S_m=0.3, reference_um=(0,1000,0)):
        self.contacts=tuple(contacts)
        if len(self.contacts)<2:
            raise ValueError('source and explicit return contacts required')
        self.sigma=float(conductivity_S_m)
        self.reference=points([reference_um])
        self._reference_transfer=transfer_mV_per_nA(self.reference,self.contacts,self.sigma)

    def potential_mV(self, xyz_um, currents_nA):
        I=np.asarray(currents_nA,dtype=float)
        if I.shape!=(len(self.contacts),) or not np.isfinite(I).all():
            raise ValueError('one finite current per contact required')
        if abs(I.sum())>1e-12*max(1,float(np.abs(I).sum())):
            raise ValueError('source/return current balance violated')
        T=transfer_mV_per_nA(xyz_um,self.contacts,self.sigma)
        return (T-self._reference_transfer)@I

    def inward_axial_drive_nA(self, cable, currents_nA):
        """For Vm=Vi-Ve, add sum_j g_ij*(Ve_j-Ve_i) to passive RHS.

        Constant extracellular reference offsets cancel. This method is a
        frozen extracellular field approximation; no feedback on conductivity.
        """
        m=cable.m
        v=self.potential_mV(np.column_stack((m.x,m.y,m.z)),currents_nA)
        I=np.zeros(m.n)
        for i,p in enumerate(m.parent):
            if p>=0:
                q=cable.g_ax[i]*(v[p]-v[i])
                I[i]+=q
                I[p]-=q
        return I


class VoltageRecorder:
    """Point voltage sampling + first-order bandwidth + noise and artifact.

    Tissue membrane sources are supplied explicitly as finite spherical source
    contacts with signed outward nA currents. No inference from mean Vm. The
    caller must supply a balanced source set in the chosen closed domain.
    Recording is a point probe, not contact-area averaging.
    """
    def __init__(self, position_um, reference_um, conductivity_S_m=0.3,
                 bandwidth_Hz=1000, noise_sd_mV=0, seed=0):
        self.xyz=points([position_um,reference_um])
        self.sigma=float(conductivity_S_m)
        self.bandwidth=float(bandwidth_Hz)
        self.noise=float(noise_sd_mV)
        if not np.isfinite([self.sigma,self.bandwidth,self.noise]).all() or min(self.sigma,self.bandwidth)<=0 or self.noise<0:
            raise ValueError('invalid recorder parameters')
        self.filtered=0.0
        self.rng=np.random.default_rng(seed)

    def step(self, dt_ms, source_contacts=(), outward_currents_nA=(),
             stimulus_field=None, stimulus_currents_nA=()):
        if not np.isfinite(dt_ms) or dt_ms<=0:
            raise ValueError('positive finite dt required')
        I=np.asarray(outward_currents_nA,float)
        if I.shape!=(len(source_contacts),) or not np.isfinite(I).all():
            raise ValueError('source/current shape or finite-value error')
        if abs(I.sum())>1e-12*max(1,float(np.abs(I).sum())):
            raise ValueError('recorded membrane sources must balance')
        T=transfer_mV_per_nA(self.xyz,source_contacts,self.sigma)
        neural=float((T[0]-T[1])@I)
        artifact=0.0
        if stimulus_field is not None:
            v=stimulus_field.potential_mV(self.xyz,stimulus_currents_nA)
            artifact=float(v[0]-v[1])
        alpha=-np.expm1(-2*np.pi*self.bandwidth*dt_ms/1000)
        self.filtered+=alpha*(neural+artifact-self.filtered)
        measured=self.filtered+self.rng.normal(0,self.noise)
        return {'measured_mV':float(measured),'neural_unfiltered_mV':neural,
                'artifact_unfiltered_mV':artifact}
