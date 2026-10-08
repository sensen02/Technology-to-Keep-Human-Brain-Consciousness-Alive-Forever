"""STIMULATION extension of the electrode model: current injection -> extracellular
field -> neural activation, under charge-injection safety limits.

WHAT THIS IS
------------
A hand-built research prototype that adds the three missing pieces to the
project's electrode model:

1. :class:`StimWaveform` -- a current-injection waveform (biphasic
   charge-balanced, monophasic, or a full-cycle sinusoid) with an EXPLICIT
   charge per phase and charge density per phase on the FIXED contact geometry.
2. :class:`ActivatingFunction` -- the second spatial difference of the
   extracellular potential (the standard Rattay-type "activating function"),
   evaluated on a real geometry.  Excitation is driven by the CURVATURE of the
   extracellular potential, not by its absolute value; the polarity of the
   membrane response therefore depends on how the neurite is oriented relative
   to the electrode.
3. :func:`charge_injection_limits` -- a safety/limit model with a provenance
   basis that is either MEASURED_CITED (a source that was actually fetched) or
   ASSUMED with a swept range.  :class:`StimLedger` REFUSES the two dishonest
   combinations outright.
4. :func:`stimulate_and_record` -- inject through some channels, record on
   others, with the stimulus artifact produced EXPLICITLY through the existing
   :class:`engine.electrode.VoltageRecorder` artifact channel.

WHAT THIS IS NOT
----------------
* NOT a validated device model: no electrode impedance, no interface
  electrochemistry, no double layer, no Faraday/water-window check, no
  electrode corrosion model, no tissue displacement, no safety certification.
* The volume conductor is the SANCTIONED one and only one:
  :func:`engine.electrode.transfer_mV_per_nA` (uniform spherical volume
  sources in an infinite homogeneous ohmic medium).  There is no second field
  solver in this module.
* Quasi-static.  No capacitive tissue response, no dispersion, no anisotropy,
  no inhomogeneity, no CSF/saline layer, no encapsulation.
* "Activation" here means a membrane-potential deviation in the project's
  OWN models: the passive :class:`engine.cable.CableNeuron` driven through
  :meth:`engine.electrode.BipolarField.inward_axial_drive_nA`, and the
  illustrative squid-HH cable of :mod:`engine.neural_active`.  Neither is a fly
  neuron.  No claim is made that anything in a fly is excited, let alone that
  the fly sees, notices, attends to or recognises anything.
* The safety limit is a CITED cochlear-implant guideline applied by
  EXTRAPOLATION to a 3.5 um-radius contact, 3 orders of magnitude below the
  contact areas it was derived from.  The extrapolation is flagged in the
  ledger; it is not a measurement of fly tissue tolerance.

FIXED GEOMETRY (user decision -- this module never searches over it)
-------------------------------------------------------------------
``SHAFT_DIAMETER_UM = 7.0`` and ``CHANNEL_PITCH_UM = 20.0``.  The channel
COUNT is free.  The contact is a spherical volume source of radius
``shaft_diameter / 2 = 3.5 um`` (that is what ``Contact`` is), whose surface
area ``4*pi*r^2`` is used for charge density, so the density and the field use
the SAME geometry rather than two different idealisations.

UNITS
-----
length um, area cm^2, time ms (us for pulse parameters), current nA,
charge nC/uC, charge density uC/cm^2, potential mV, activating function
mV/um^2, conductivity S/m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# The ONE sanctioned volume-conduction solver, and the ONE sanctioned recorder.
from .electrode import (
    BipolarField,
    Contact,
    VoltageRecorder,
    transfer_mV_per_nA,
)
# Reuse the project's provenance vocabulary and its ENFORCING ledger; do not
# invent a second provenance scheme.
from .embodied.access_map import (
    BANC_SOMA_ATTRIBUTION,
    DISCLAIMER,
    PROVENANCE,
    Ledger,
    ProvenanceRecord,
)

__all__ = [
    "SHAFT_DIAMETER_UM", "CHANNEL_PITCH_UM", "CONTACT_RADIUS_UM", "SHAPES",
    "PROTOTYPE_NOTICE", "NO_CLAIMS", "DEFAULT_CONDUCTIVITY_S_M",
    "DEFAULT_REFERENCE_UM", "CITED_SAFETY_SOURCES", "CITED_SHANNON_K",
    "CITED_MAX_CHARGE_DENSITY_UC_CM2", "CITED_EXTREME_DENSITIES_UC_CM2",
    "ASSUMED_DENSITY_SWEEP_UC_CM2",
    "StimWaveform", "ActivatingFunction", "StimLedger",
    "ChargeInjectionLimits", "charge_injection_limits", "record_limits",
    "stimulate_and_record", "shank_channel_centres", "shank_contacts",
    "straight_neurite", "chain_morphology", "orientation_sweep",
    "passive_membrane_response", "densest_soma_anchor", "nearest_soma_distances",
    "attenuation_law_check",
]

# ---------------------------------------------------------------------------
# fixed geometry (user decision) and declared defaults
# ---------------------------------------------------------------------------
#: FIXED user decision.  Never searched, never optimised, never re-derived.
SHAFT_DIAMETER_UM = 7.0
#: FIXED user decision.
CHANNEL_PITCH_UM = 20.0
#: The contact is the spherical volume source the sanctioned solver integrates
#: over, so its radius is the shaft radius.
CONTACT_RADIUS_UM = SHAFT_DIAMETER_UM / 2.0

#: Declared isotropic conductivity of the modelled medium.  Not a fly-tissue
#: measurement; swept by the runner.  engine.electrode's own default is 0.3.
DEFAULT_CONDUCTIVITY_S_M = 0.3

#: The explicit observation point the sanctioned field solver references to.
#: 20 mm away on the z axis: a DECLARED closed-domain convention (the zero of
#: the potential), not a model of a reference electrode.
DEFAULT_REFERENCE_UM = (0.0, 0.0, 20000.0)

#: The DISTANT RETURN contact used for monopolar stimulation, on a DIFFERENT
#: far axis from the observation reference.  Keeping the two apart matters:
#: if a contact sits exactly on the potential's zero point, the solver's
#: in-sphere value of that contact's own Green function leaks into every
#: differential reading as a large spurious offset.  :func:`stimulate_and_record`
#: and :class:`ActivatingFunction` both refuse that configuration.
DEFAULT_RETURN_UM = (0.0, 20000.0, 0.0)

PROTOTYPE_NOTICE = (
    "STIMULATION PROTOTYPE.  Current injection -> quasi-static extracellular "
    "field -> membrane deviation in the project's own passive cable and in an "
    "ILLUSTRATIVE squid-HH cable.  No electrode impedance, no double layer, no "
    "interface electrochemistry, no corrosion model, no tissue mechanics, no "
    "fly neuron.  The safety limit is a cited cochlear-implant guideline "
    "EXTRAPOLATED to a 3.5 um-radius contact.  Hand-built research prototype, "
    "NOT a validated device model."
)

NO_CLAIMS = (
    "No claim that any fly neuron is excited, and no claim that the fly sees, "
    "notices, attends to, recognises, remembers or is conscious of anything.  "
    "No behavioural claim of any kind is made or implied by this module."
)


def _finite(value, name, minimum=None, strict=False, integer=False):
    v = float(value)
    if integer:
        if isinstance(value, bool) or not float(value).is_integer():
            raise ValueError("%s must be an integer-valued number" % name)
        v = int(round(v))
    if not math.isfinite(v):
        raise ValueError("%s must be finite" % name)
    if minimum is not None:
        if (v <= minimum) if strict else (v < minimum):
            raise ValueError("%s must be %s %r" % (name, ">" if strict else ">=", minimum))
    return v


def _check_reference_clearance(contacts, reference_um, min_um=50.0):
    """Refuse a contact sitting on the potential's own zero point.

    The sanctioned solver evaluates a FINITE SPHERE contact with the
    volume-averaged Green function, which is large INSIDE the contact.  If a
    contact coincides with the field's reference point, that in-sphere value
    leaks into every differential potential as a large spurious offset (a
    ~300 mV artefact for a 3 uA return through a 3.5 um sphere).  The reference
    must therefore be a genuine observation point away from every contact; the
    50 um floor is ~14 contact radii and still allows a differential recording
    between two sites of the SAME array.
    """
    r = np.asarray(reference_um, dtype=float)
    for c in contacts:
        d = float(np.linalg.norm(r - np.asarray(c.center_um, dtype=float)))
        if d < min_um:
            raise ValueError(
                "a contact sits %.3f um from the field's reference point; the "
                "reference must be a DISTINCT point (>= %.0f um) or the "
                "solver's in-sphere value of that contact leaks a spurious "
                "offset into every differential reading" % (d, min_um))


def _check_clear_of_contacts(xyz_um, contacts, margin_um=0.0):
    """Refuse a trajectory that enters a contact's own sphere.

    Inside the contact the solver returns the regularised sphere solution, not
    a 1/r monopole, so a second difference evaluated there is not the activating
    function of an electrode-tissue interface.
    """
    p = np.asarray(xyz_um, dtype=float)
    for c in contacts:
        d = np.linalg.norm(p - np.asarray(c.center_um, dtype=float), axis=1)
        need = float(c.radius_um) + float(margin_um)
        if float(d.min()) < need:
            raise ValueError(
                "the trajectory comes within %.3f um of a contact of radius "
                "%.3f um; keep the stated neurite outside the contact "
                "(minimum clearance %.3f um required)"
                % (float(d.min()), float(c.radius_um), need))


def _unit(v, name="direction"):
    a = np.asarray(v, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("%s must be a finite 3-vector" % name)
    n = float(np.linalg.norm(a))
    if n <= 0:
        raise ValueError("%s must be nonzero" % name)
    return a / n


def _path(points):
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 5:
        raise ValueError("a trajectory needs (n>=5, 3) finite coordinates")
    if not np.isfinite(a).all():
        raise ValueError("trajectory must be finite")
    steps = np.linalg.norm(np.diff(a, axis=0), axis=1)
    h = float(steps.mean())
    if np.any(np.abs(steps - h) > 1e-6 * max(h, 1e-9)):
        raise ValueError(
            "the second difference is only a spatial derivative on an EQUALLY "
            "SPACED trajectory; spacing varies by %.3g%%"
            % (100.0 * float(np.abs(steps - h).max() / h)))
    return a, h


# ---------------------------------------------------------------------------
# 1. the stimulation waveform
# ---------------------------------------------------------------------------
SHAPES = ("biphasic", "monophasic", "sinusoid")


@dataclass(frozen=True)
class StimWaveform:
    """A current-injection waveform on the FIXED contact geometry.

    Parameters
    ----------
    shape
        ``"biphasic"``  cathodic (or anodic) first rectangular phase, then an
                        interphase gap, then the OPPOSITE phase of equal
                        amplitude and width: charge-balanced per pulse.
        ``"monophasic"`` one rectangular phase only.  NOT charge-balanced; it
                        accumulates charge and is the shape that drives
                        irreversible reactions.  Modelled because the brief
                        asks for a second shape, and reported as UNBALANCED.
        ``"sinusoid"``  one FULL sine cycle per repetition, so the negative and
                        positive half cycles are exact mirror images and the
                        pulse is charge-balanced.  ``phase_width_us`` is the
                        half-period, i.e. the width of one phase.
    amplitude_nA
        peak magnitude of the injected current at the active contact.
    phase_width_us
        width of ONE phase (one polarity).
    interphase_gap_us
        gap between the two phases of a biphasic pulse.  Must be 0 for the
        sinusoid (a gap would break the cycle) and is ignored for the
        monophasic shape (reported as 0).
    repetition_rate_Hz
        pulses per second.  The period must be at least one full pulse.
    contact_radius_um
        defaults to the FIXED shaft radius 3.5 um.  The contact area used for
        charge density is the sphere area ``4*pi*r^2`` -- the same geometry the
        sanctioned field solver integrates over.
    first_phase
        ``"cathodic"`` (default) or ``"anodic"``.  Cathodic-first is the shape
        the cited safety literature and clinical stimulators use.

    ``charge_per_phase_uC`` and ``charge_density_per_phase_uC_cm2`` are
    properties of THIS object, not of a tissue, and carry no safety implication
    by themselves -- :func:`charge_injection_limits` supplies the limit.
    """

    shape: str = "biphasic"
    amplitude_nA: float = 1000.0
    phase_width_us: float = 100.0
    interphase_gap_us: float = 20.0
    repetition_rate_Hz: float = 1.0
    contact_radius_um: float = CONTACT_RADIUS_UM
    first_phase: str = "cathodic"

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError("shape must be one of %r" % (SHAPES,))
        if self.first_phase not in ("cathodic", "anodic"):
            raise ValueError("first_phase must be 'cathodic' or 'anodic'")
        object.__setattr__(self, "amplitude_nA",
                           _finite(self.amplitude_nA, "amplitude_nA", 0, True))
        object.__setattr__(self, "phase_width_us",
                           _finite(self.phase_width_us, "phase_width_us", 0, True))
        object.__setattr__(self, "interphase_gap_us",
                           _finite(self.interphase_gap_us, "interphase_gap_us", 0))
        object.__setattr__(self, "repetition_rate_Hz",
                           _finite(self.repetition_rate_Hz, "repetition_rate_Hz", 0, True))
        object.__setattr__(self, "contact_radius_um",
                           _finite(self.contact_radius_um, "contact_radius_um", 0, True))
        if self.shape == "sinusoid" and self.interphase_gap_us != 0.0:
            raise ValueError("a sinusoid has no interphase gap; set it to 0")
        if self.period_us < self.pulse_width_us - 1e-9:
            raise ValueError(
                "repetition_rate_Hz=%.6g gives a %.3f us period, shorter than the "
                "%.3f us pulse" % (self.repetition_rate_Hz, self.period_us,
                                   self.pulse_width_us))

    # -- geometry ----------------------------------------------------------
    @property
    def contact_area_cm2(self):
        """Sphere surface area of the contact, in cm^2 (FIXED 7 um geometry)."""
        r_cm = self.contact_radius_um * 1e-4
        return 4.0 * math.pi * r_cm * r_cm

    @property
    def phase_sign(self):
        return -1.0 if self.first_phase == "cathodic" else 1.0

    # -- timing ------------------------------------------------------------
    @property
    def pulse_width_us(self):
        if self.shape == "biphasic":
            return 2.0 * self.phase_width_us + self.interphase_gap_us
        if self.shape == "sinusoid":
            return 2.0 * self.phase_width_us
        return self.phase_width_us

    @property
    def period_us(self):
        return 1e6 / self.repetition_rate_Hz

    @property
    def duty_cycle(self):
        return self.pulse_width_us / self.period_us

    @property
    def charge_balanced(self):
        """True only for shapes whose two phases are exact mirrors."""
        return self.shape in ("biphasic", "sinusoid")

    # -- waveform ----------------------------------------------------------
    def sample_outward_current_nA(self, t_us):
        """Outward current at the ACTIVE contact, nA, at time ``t_us``.

        NEGATIVE means current flows from tissue INTO the contact (cathodic).
        One pulse starts at t=0 of each period.
        """
        t = float(t_us) % self.period_us
        w, g = self.phase_width_us, self.interphase_gap_us
        s = self.phase_sign
        if self.shape == "sinusoid":
            if t >= 2.0 * w:
                return 0.0
            # one full cycle of frequency 1/(2w); negative half first
            return s * self.amplitude_nA * math.sin(math.pi * t / w)
        if t < w:
            return s * self.amplitude_nA
        if self.shape == "biphasic":
            if t < w + g:
                return 0.0
            if t < 2.0 * w + g:
                return -s * self.amplitude_nA
        return 0.0

    def samples(self, dt_us=1.0, n_periods=1):
        """(t_us, outward current nA) sampled over ``n_periods`` full periods."""
        dt = _finite(dt_us, "dt_us", 0, True)
        n_periods = _finite(n_periods, "n_periods", 0, True, integer=True)
        n = int(round(n_periods * self.period_us / dt))
        t = np.arange(n + 1) * dt
        return t, np.array([self.sample_outward_current_nA(v) for v in t])

    # -- charge ------------------------------------------------------------
    @property
    def charge_per_phase_uC(self):
        """Charge delivered in ONE phase, uC, by exact integration of the shape."""
        if self.shape == "sinusoid":
            # integral of A*sin(pi t/w) over one half cycle = A*2w/pi
            q_na_us = self.amplitude_nA * 2.0 * self.phase_width_us / math.pi
        else:
            q_na_us = self.amplitude_nA * self.phase_width_us
        return q_na_us * 1e-9  # nA * us = 1e-9 uC

    @property
    def charge_per_phase_nC(self):
        return self.charge_per_phase_uC * 1000.0

    @property
    def charge_density_per_phase_uC_cm2(self):
        """Charge per phase divided by the FIXED 7 um contact's sphere area."""
        return self.charge_per_phase_uC / self.contact_area_cm2

    @property
    def net_charge_per_pulse_uC(self):
        """Net charge after a whole pulse: 0 for a balanced shape, 2x phase
        charge for the monophasic one (the accumulating case)."""
        if self.charge_balanced:
            return 0.0
        return self.charge_per_phase_uC

    def as_dict(self):
        return {
            "shape": self.shape,
            "first_phase": self.first_phase,
            "amplitude_nA": self.amplitude_nA,
            "phase_width_us": self.phase_width_us,
            "interphase_gap_us": (0.0 if self.shape == "monophasic"
                                  else self.interphase_gap_us),
            "repetition_rate_Hz": self.repetition_rate_Hz,
            "pulse_width_us": self.pulse_width_us,
            "period_us": self.period_us,
            "duty_cycle": self.duty_cycle,
            "charge_balanced": self.charge_balanced,
            "net_charge_per_pulse_uC": self.net_charge_per_pulse_uC,
            "contact_radius_um": self.contact_radius_um,
            "contact_area_cm2": self.contact_area_cm2,
            "shaft_diameter_um": SHAFT_DIAMETER_UM,
            "channel_pitch_um": CHANNEL_PITCH_UM,
            "charge_per_phase_uC": self.charge_per_phase_uC,
            "charge_per_phase_nC": self.charge_per_phase_nC,
            "charge_density_per_phase_uC_cm2": self.charge_density_per_phase_uC_cm2,
            "geometry_note": ("contact area is the SPHERE area 4*pi*r^2 of the "
                              "sanctioned solver's contact, radius = shaft/2; the "
                              "geometry is the FIXED user input and is never "
                              "searched here"),
        }

    @classmethod
    def at_charge_density(cls, density_uC_cm2, **kw):
        """The amplitude whose charge density equals ``density_uC_cm2``.

        Charge is linear in amplitude for every shape here, so this is exact.
        """
        probe = cls(amplitude_nA=1.0, **kw)
        per_nA = probe.charge_density_per_phase_uC_cm2  # density at 1 nA
        if per_nA <= 0:
            raise ValueError("zero phase width gives zero charge")
        return cls(amplitude_nA=float(density_uC_cm2) / per_nA, **kw)


# ---------------------------------------------------------------------------
# 2. the activating function
# ---------------------------------------------------------------------------
def straight_neurite(centre_um, direction, length_um, n_points=201):
    """Equally spaced points along a straight, STATED neurite trajectory.

    The trajectory is a modelling choice, not a reconstruction: the BANC soma
    release is single voxel POINTS and carries no morphology.  What is real in
    the runner is the electrode position and the distance -- the direction and
    length of the segment are declared.
    """
    c = np.asarray(centre_um, dtype=float)
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError("centre_um must be a finite 3-vector")
    u = _unit(direction)
    L = _finite(length_um, "length_um", 0, True)
    n = _finite(n_points, "n_points", 4, True, integer=True)
    s = np.linspace(-L / 2.0, L / 2.0, int(n))
    return c[None, :] + s[:, None] * u[None, :]


def chain_morphology(points_um, diameter_um):
    """A single unbranched chain through ``points_um``, for engine.cable.

    Uses the project's own :class:`engine.cable.Morphology`; nothing about the
    cable discretisation is reimplemented here.
    """
    from .cable import Morphology
    p, _h = _path(points_um)
    d = _finite(diameter_um, "diameter_um", 0, True)
    n = p.shape[0]
    parent = np.concatenate(([-1], np.arange(n - 1)))
    return Morphology(p[:, 0], p[:, 1], p[:, 2], np.full(n, d), parent)


class ActivatingFunction:
    """Second spatial derivative of the extracellular potential on a neurite.

    For a passive cable with membrane potential ``Vm = Vi - Ve``::

        Cm dVm/dt = (d/4Ra) * (Vm'' + Ve'') - g_m (Vm - E_leak)

    so the extracellular field enters ONLY through ``Ve''``.  ``Ve''`` is the
    activating function: positive means the field injects depolarising current,
    negative means hyperpolarising current.  A CONSTANT ``Ve`` has zero
    curvature and therefore does nothing to a straight uniform cable -- which
    is why "the absolute value of the field" is not the mechanism.

    The potential comes from the sanctioned solver
    :func:`engine.electrode.transfer_mV_per_nA`; this class performs the
    differencing and nothing else.  The class never builds a second field
    solver.

    ``method``
        ``"second_difference"`` (default): the literal discrete second
        difference on the equally spaced trajectory, second-order one-sided
        stencils at the two ends.
        ``"analytic_monopole"``: the closed form for a pure 1/r monopole at
        perpendicular distance d, ``Ve'' = K*(2x^2 - d^2)/(x^2+d^2)^2.5``
        with ``K = I/(4*pi*sigma)``.  Used ONLY as an independent check of the
        differencing, never as the reported field.
    """

    def __init__(self, contacts, currents_nA, conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M,
                 reference_um=DEFAULT_REFERENCE_UM):
        self.contacts = tuple(contacts)
        if not self.contacts:
            raise ValueError("at least one contact is required")
        self.sigma = _finite(conductivity_S_m, "conductivity_S_m", 0, True)
        self.currents = np.asarray(currents_nA, dtype=float)
        if self.currents.shape != (len(self.contacts),) or not np.isfinite(self.currents).all():
            raise ValueError("one finite current per contact required")
        if abs(self.currents.sum()) > 1e-12 * max(1.0, float(np.abs(self.currents).sum())):
            raise ValueError(
                "the contact currents must balance: an unbalanced set is not a "
                "closed-domain field. Add an explicit distant return contact.")
        # The sanctioned field object (it also validates balance and reference).
        self.field = BipolarField(self.contacts, self.sigma, reference_um=reference_um)
        _check_reference_clearance(self.contacts, reference_um)
        self.reference_um = tuple(float(v) for v in reference_um)

    # -- potential ---------------------------------------------------------
    def potential_mV(self, xyz_um):
        """Extracellular potential at ``xyz_um``, referenced to the declared
        explicit observation point."""
        return self.field.potential_mV(xyz_um, self.currents)

    # -- the activating function ------------------------------------------
    def along(self, xyz_um):
        """Activating function along an equally spaced trajectory.

        Returns a dict with the potential, the second difference in mV/um^2,
        the peak depolarising and hyperpolarising sites, and the locations of
        the sign changes -- the quantity that makes the ORIENTATION dependence
        measurable instead of asserted.
        """
        p, h = _path(xyz_um)
        _check_clear_of_contacts(p, self.contacts)
        v = self.potential_mV(p)
        f = np.empty_like(v)
        # standard stride-1 central difference on the interior (truncation
        # (h^2/12)*Ve''''), second-order one-sided stencils at the two ends
        f[1:-1] = (v[:-2] - 2.0 * v[1:-1] + v[2:]) / (h * h)
        f[0] = (2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3]) / (h * h)
        f[-1] = (2.0 * v[-1] - 5.0 * v[-2] + 4.0 * v[-3] - v[-4]) / (h * h)
        s = np.arange(v.size) * h
        s = s - s.mean()
        sign = np.sign(f)
        idx = np.flatnonzero(np.diff(sign) != 0)
        crossings = []
        for i in idx:
            # linear interpolation of the zero of a piecewise-linear f
            f0, f1 = f[i], f[i + 1]
            t = f0 / (f0 - f1) if f0 != f1 else 0.5
            crossings.append(float(s[i] + t * h))
        return {
            "n": int(v.size),
            "spacing_um": float(h),
            "s_um": s.tolist(),
            "potential_mV": v.tolist(),
            "af_mV_per_um2": f.tolist(),
            "af_peak_depolarising_mV_per_um2": float(f.max()),
            "af_peak_depolarising_s_um": float(s[int(np.argmax(f))]),
            "af_peak_hyperpolarising_mV_per_um2": float(f.min()),
            "af_peak_hyperpolarising_s_um": float(s[int(np.argmin(f))]),
            "af_at_midpoint_mV_per_um2": float(f[v.size // 2]),
            "n_sign_changes": int(len(crossings)),
            "sign_change_s_um": crossings,
            "sign_convention": ("positive activating function = the field injects "
                                "DEPOLARISING current; the field's source term in "
                                "the cable equation is +(d/4Ra)*Ve''"),
        }

    def at_point(self, xyz_um, i=None):
        """Activating function at one index of an equally spaced trajectory."""
        r = self.along(xyz_um)
        k = r["n"] // 2 if i is None else int(i)
        return float(r["af_mV_per_um2"][k])

    # -- independent analytic check ---------------------------------------
    @staticmethod
    def analytic_monopole_mV_per_um2(current_nA, conductivity_S_m, x_um, d_um):
        """Ve'' for an ideal 1/r monopole at perpendicular distance ``d``.

        With ``Ve = K/sqrt(x^2+d^2)`` and ``K = I/(4*pi*sigma)`` (mV*um for I in
        nA and sigma in S/m) the second derivative is
        ``Ve'' = K*(2x^2 - d^2)/(x^2+d^2)^2.5`` in mV/um^2.  At x=0 this is
        ``-K/d^3``; it is zero at ``x = d/sqrt(2)`` and positive beyond it.
        Used ONLY as an independent check of the discrete differencing.
        """
        K = float(current_nA) / (4.0 * math.pi * float(conductivity_S_m))
        x, d = np.asarray(x_um, float), float(d_um)
        if d <= 0:
            raise ValueError("d_um must be positive")
        return K * (2.0 * x * x - d * d) / np.power(x * x + d * d, 2.5)


def orientation_sweep(contacts, currents_nA, electrode_um, distance_um, angles_deg,
                      length_um=400.0, n_points=201,
                      conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M,
                      reference_um=DEFAULT_REFERENCE_UM, perpendicular=None):
    """Measure the activating function at the membrane point nearest the
    electrode as the neurite is ROTATED about that point.

    ``angles_deg`` are measured from the radial direction: 0 deg means the
    neurite points straight at (or away from) the electrode -- an END-ON
    neurite; 90 deg means the neurite passes by TANGENTIALLY.  Everything else
    (distance, current, electrode) is held fixed, so a sign change with angle
    IS the orientation dependence.
    """
    e = np.asarray(electrode_um, dtype=float)
    d = _finite(distance_um, "distance_um", 0, True)
    L = _finite(length_um, "length_um", 0, True)
    if L / 2.0 > d - CONTACT_RADIUS_UM:
        raise ValueError(
            "for the sweep to be comparable across angles the WHOLE trajectory "
            "must stay outside the contact at every angle, which needs "
            "length_um/2 <= distance_um - contact_radius_um = %.3f um; got "
            "length_um=%.3f um. Use a shorter segment, or measure the long "
            "profile separately along a TANGENTIAL direction."
            % (d - CONTACT_RADIUS_UM, L))
        raise ValueError(
            "the trajectory must stay OUTSIDE the contact (distance %.3f um > "
            "contact radius %.3f um); inside it the field is the regularised "
            "sphere solution and is not a 1/r monopole"
            % (d, CONTACT_RADIUS_UM))
    # put the electrode on +y relative to the trajectory midpoint
    centre = e + np.array([0.0, d, 0.0])
    radial = np.array([0.0, 1.0, 0.0])
    if perpendicular is None:
        # the perpendicular that keeps the trajectory in the y-z plane
        perpendicular = np.array([0.0, 0.0, 1.0])
    perp = _unit(perpendicular)
    if abs(float(perp @ radial)) > 1e-9:
        perp = perp - float(perp @ radial) * radial
        perp = _unit(perp)
    af = ActivatingFunction(contacts, currents_nA, conductivity_S_m, reference_um)
    rows = []
    for a in np.asarray(angles_deg, dtype=float):
        th = math.radians(float(a))
        u = math.cos(th) * radial + math.sin(th) * perp
        pts = straight_neurite(centre, u, length_um, n_points)
        r = af.along(pts)
        rows.append({
            "angle_deg": float(a),
            "neurite_direction": [float(v) for v in u],
            "af_midpoint_mV_per_um2": r["af_at_midpoint_mV_per_um2"],
            "af_max_mV_per_um2": r["af_peak_depolarising_mV_per_um2"],
            "af_min_mV_per_um2": r["af_peak_hyperpolarising_mV_per_um2"],
            "n_sign_changes": r["n_sign_changes"],
        })
    return rows


def passive_membrane_response(points_um, diameter_um, currents_nA, contacts,
                              conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M,
                              reference_um=DEFAULT_REFERENCE_UM, dt_ms=0.05,
                              duration_ms=60.0, node=None):
    """MEASURED membrane deviation of the project's PASSIVE cable in a field.

    Volume conduction is reused, not reimplemented: the field is
    :class:`engine.electrode.BipolarField` and the drive handed to the cable is
    exactly :meth:`BipolarField.inward_axial_drive_nA`, which is the discrete
    form of the ``(d/4Ra)*Ve''`` term.  The cable is
    :class:`engine.cable.CableNeuron` with the project's own DN_PASSIVE
    parameters (fitted to DNp01/DNp03, PMC11071487).

    Returns the steady-state deviation from rest at the requested node (default:
    the midpoint, i.e. the membrane point nearest the electrode) plus the whole
    profile, so the AF prediction can be checked against the cable model
    instead of being believed.
    """
    from .cable import CableNeuron
    m = chain_morphology(points_um, diameter_um)
    cell = CableNeuron(m, dt_ms=dt_ms)
    field = BipolarField(tuple(contacts), conductivity_S_m, reference_um=reference_um)
    _check_reference_clearance(tuple(contacts), reference_um)
    _check_clear_of_contacts(np.asarray(points_um, float), tuple(contacts))
    drive = field.inward_axial_drive_nA(cell, currents_nA)
    v = cell.run(duration_ms, i_inject=drive)[-1]
    dev = v - cell.E_leak
    k = m.n // 2 if node is None else int(node)
    return {
        "vm_deviation_mV": float(dev[k]),
        "vm_deviation_profile_mV": dev.tolist(),
        "vm_max_mV": float(dev.max()),
        "vm_min_mV": float(dev.min()),
        "node": int(k),
        "node_index_of_voltage_max": int(np.argmax(dev)),
        "node_index_of_voltage_min": int(np.argmin(dev)),
        "dt_ms": float(dt_ms),
        "duration_ms": float(duration_ms),
        "cable_model": "engine.cable.CableNeuron, DN_PASSIVE (passive, no channels)",
        "drive_model": "engine.electrode.BipolarField.inward_axial_drive_nA",
    }


# ---------------------------------------------------------------------------
# geometry helpers (the FIXED array)
# ---------------------------------------------------------------------------
def shank_channel_centres(origin_um, axis=(0.0, 1.0, 0.0), n_channels=16,
                          pitch_um=CHANNEL_PITCH_UM):
    """Contact CENTRES of a linear shank: pitch 20 um, centred on ``origin_um``.

    ``n_channels`` is the free parameter; the pitch is the FIXED user input and
    is not searched.  Ordering is by increasing coordinate along the axis, so a
    channel index means the same thing across runs.
    """
    n = _finite(n_channels, "n_channels", 1, True, integer=True)
    p = _finite(pitch_um, "pitch_um", 0, True)
    if abs(p - CHANNEL_PITCH_UM) > 1e-9:
        raise ValueError(
            "channel pitch is a FIXED user decision (%.3f um); %.3f um was "
            "requested" % (CHANNEL_PITCH_UM, p))
    o = np.asarray(origin_um, dtype=float)
    if o.shape != (3,) or not np.isfinite(o).all():
        raise ValueError("origin_um must be a finite 3-vector")
    u = _unit(axis, "axis")
    k = np.arange(int(n), dtype=float) - (int(n) - 1) / 2.0
    return o[None, :] + (k * p)[:, None] * u[None, :]


def shank_contacts(centres_um, radius_um=CONTACT_RADIUS_UM):
    """``Contact`` objects for the same fixed geometry the solver integrates."""
    a = np.asarray(centres_um, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("centres_um must be finite (n,3)")
    return [Contact(tuple(float(v) for v in row), radius_um) for row in a]


def densest_soma_anchor(positions_um, axis, radius_um=25.0):
    """A REAL data anchor for placing the shank.

    Returns the index and position of the soma that has the most neighbours
    within ``radius_um`` in the real soma cloud (a KD-tree count).  This is a
    PLACEMENT CHOICE justified as "put the shank where the somas are", not an
    anatomical finding.  Ties break on the smaller row index, so the result is
    deterministic and no RNG is involved.
    """
    from scipy.spatial import cKDTree
    p = np.asarray(positions_um, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("positions_um must be finite (n>=2,3)")
    r = _finite(radius_um, "radius_um", 0, True)
    tree = cKDTree(p)
    counts = np.array(tree.query_ball_point(p, r, return_length=True))
    best = int(np.argmax(counts))  # argmax returns the FIRST maximum => deterministic
    return {
        "row": best,
        "position_um": [float(v) for v in p[best]],
        "n_neighbours_within_radius": int(counts[best]),
        "radius_um": float(r),
        "axis_requested": [float(v) for v in _unit(axis, "axis")],
        "note": ("DATA-ANCHORED PLACEMENT: the shank is centred on the real soma "
                 "with the most neighbours within the stated radius. A choice, "
                 "not an anatomical finding."),
    }


# ---------------------------------------------------------------------------
# 3. charge-injection safety limits
# ---------------------------------------------------------------------------
#: Every external number this module may cite lives here, with the page it came
#: from and the date it was fetched.  If a value here is ever used without this
#: block, the ledger refuses it (MEASURED_CITED requires a source).
CITED_SAFETY_SOURCES = {
    "shepherd2018": {
        "citation": ("Shepherd RK, Carter PM, Enke YL, Wise AK, Fallon JB. "
                     "Chronic intracochlear electrical stimulation at high charge "
                     "densities results in platinum dissolution but not neural loss "
                     "or functional changes in vivo. J Neural Eng 2018;16(2):026009."),
        "doi": "10.1088/1741-2552/aaf66b",
        "pmid": "30523828",
        "pmcid": "PMC8687872",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8687872/",
        "fetched_utc": "2026-10-03",
        "fetched_how": ("web_fetch of the PMC full text, HTTP 200; the quoted "
                        "sentences below are verbatim from that page"),
        "quotes": {
            "shannon_equation": (
                "the so-called \u201cShannon limit\u201d ... defines the maximum "
                "safe relationship between charge density and charge per phase "
                "for neural prostheses by the equation: log(Q/A) = k \u2212 log(Q) "
                "where Q is the charge per phase (\u03bcC per phase), A is the "
                "geometric surface area of the stimulating electrode, Q/A is the "
                "charge density per phase (\u03bcC/cm2/phase) ... with the "
                "variable k within the range 1.5 \u2013 1.85"),
            "aami_standard": (
                "the Association for the Advancement of Medical Instrumentation "
                "(AAMI) standard for commercial cochlear implants (defined as the "
                "Shannon limit (k=1.75) with a maximum charge density of "
                "216 \u03bcC/cm2/phase)"),
            "measured_anchors": (
                "Compared with unstimulated control electrodes and electrodes "
                "stimulated at 100 \u03bcC/cm2/phase, stimulation at 267 or "
                "400 \u03bcC/cm2/phase resulted in significant Pt corrosion ... "
                "no evidence of a reduction in AN function associated with chronic "
                "stimulation at 100, 267 or 400 \u03bcC/cm2/phase"),
            "extrapolation_warning": (
                "it was derived from acute studies using large surface area "
                "electrodes in direct contact with cortical neurons ... it does "
                "not accurately define safety levels for applications outside "
                "that subset"),
        },
        "areas_used_by_source_cm2": [0.005, 0.0075, 0.02],
        "note": ("the source's own contacts were 0.05, 0.075 and 0.2 mm^2, i.e. "
                 "3.2e3 to 1.3e4 times the 7 um contact's sphere area. Applying a "
                 "CHARGE DENSITY from that regime to a 3.5 um-radius contact is an "
                 "EXTRAPOLATION over 3 orders of magnitude in area."),
    },
}

#: AAMI (as reported by the fetched source): Shannon k = 1.75 AND a hard
#: charge-density ceiling of 216 uC/cm^2/phase.  The ceiling is what binds at a
#: small contact, because the Shannon curve rises steeply as Q falls.
CITED_SHANNON_K = 1.75
CITED_MAX_CHARGE_DENSITY_UC_CM2 = 216.0
CITED_EXTREME_DENSITIES_UC_CM2 = (100.0, 267.0, 400.0)

#: Densities swept when the cited limit is treated as merely one point in an
#: assumption family (the honest alternative to pretending it transfers).
ASSUMED_DENSITY_SWEEP_UC_CM2 = (10.0, 50.0, 100.0, 216.0, 400.0)


@dataclass(frozen=True)
class ChargeInjectionLimits:
    """Safety limits for the FIXED 7 um contact, with their basis attached.

    Two independent ceilings are reported because a stimulator must satisfy
    both:

    * a MAXIMUM CHARGE DENSITY per phase (uC/cm^2/phase) -- the cited one;
    * a MAXIMUM CURRENT per phase (nA) -- NO verified source was fetched for a
      current-per-phase ceiling in this geometry/tissue, so it is ASSUMED with
      an explicit sweep range, and the module says so rather than dressing a
      guess as a standard.

    The current ceiling implied by the charge ceiling is a DERIVATION:
    ``I = Q/t``.  It is reported separately and never presented as a citation.
    """

    contact_radius_um: float
    contact_area_cm2: float
    density_limit_uC_cm2: float
    density_provenance: str
    density_source: Optional[str]
    density_sweep_uC_cm2: tuple
    shannon_k: Optional[float]
    shannon_note: str
    assumed_current_limit_nA: float
    assumed_current_limit_source: Optional[str]
    assumed_current_limit_sweep_nA: tuple
    extrapolation_note: str

    # -- charge ------------------------------------------------------------
    def charge_limit_per_phase_uC(self):
        return self.density_limit_uC_cm2 * self.contact_area_cm2

    def charge_limit_per_phase_nC(self):
        return self.charge_limit_per_phase_uC() * 1000.0

    def amplitude_limit_nA(self, phase_width_us, density_uC_cm2=None):
        """Amplitude at which the chosen density ceiling is reached.

        Charge is linear in amplitude for every waveform here, so this is the
        exact breaching amplitude for a given phase width.
        """
        w = _finite(phase_width_us, "phase_width_us", 0, True)
        q_lim = (self.density_limit_uC_cm2 if density_uC_cm2 is None
                 else float(density_uC_cm2)) * self.contact_area_cm2
        return q_lim / (w * 1e-9)

    def current_limit_nA(self, phase_width_us):
        """The smaller of the derived charge-derived current and the ASSUMED
        current ceiling.  Both constituents are named in the result."""
        i_charge = self.amplitude_limit_nA(phase_width_us)
        return {
            "phase_width_us": float(phase_width_us),
            "charge_derived_current_limit_nA": float(i_charge),
            "assumed_current_limit_nA": float(self.assumed_current_limit_nA),
            "binding_limit_nA": float(min(i_charge, self.assumed_current_limit_nA)),
            "binding": ("charge_density" if i_charge <= self.assumed_current_limit_nA
                        else "assumed_current_ceiling"),
        }

    def margin(self, waveform: StimWaveform):
        """Margin of a waveform against BOTH ceilings.

        ``margin > 1`` means inside the limit; ``margin < 1`` means the limit is
        breached by that factor.
        """
        d = waveform.charge_density_per_phase_uC_cm2
        i = waveform.amplitude_nA
        m_density = self.density_limit_uC_cm2 / d
        m_current = self.assumed_current_limit_nA / i
        return {
            "waveform": waveform.as_dict(),
            "density_limit_uC_cm2": self.density_limit_uC_cm2,
            "achieved_density_uC_cm2": d,
            "density_margin": float(m_density),
            "assumed_current_limit_nA": self.assumed_current_limit_nA,
            "achieved_current_nA": i,
            "current_margin": float(m_current),
            "binding_margin": float(min(m_density, m_current)),
            "verdict": ("WITHIN_CITED_LIMIT" if min(m_density, m_current) >= 1.0
                        else "BREACHES_LIMIT"),
            "breaching_amplitude_nA": float(self.amplitude_limit_nA(waveform.phase_width_us)),
            "charge_per_phase_at_limit_nC": self.charge_limit_per_phase_nC(),
            "charge_per_phase_uC": waveform.charge_per_phase_uC,
        }

    def as_dict(self):
        return {
            "contact_radius_um": self.contact_radius_um,
            "contact_area_cm2": self.contact_area_cm2,
            "shaft_diameter_um": SHAFT_DIAMETER_UM,
            "channel_pitch_um": CHANNEL_PITCH_UM,
            "density_limit_uC_cm2": self.density_limit_uC_cm2,
            "density_provenance": self.density_provenance,
            "density_source": self.density_source,
            "density_sweep_uC_cm2": list(self.density_sweep_uC_cm2),
            "shannon_k": self.shannon_k,
            "shannon_note": self.shannon_note,
            "charge_limit_per_phase_uC": self.charge_limit_per_phase_uC(),
            "charge_limit_per_phase_nC": self.charge_limit_per_phase_nC(),
            "assumed_current_limit_nA": self.assumed_current_limit_nA,
            "assumed_current_limit_source": self.assumed_current_limit_source,
            "assumed_current_limit_sweep_nA": list(self.assumed_current_limit_sweep_nA),
            "extrapolation_note": self.extrapolation_note,
        }


def charge_injection_limits(contact_radius_um=CONTACT_RADIUS_UM, *, ledger=None,
                            record=True):
    """The safety/limit model at the FIXED contact geometry.

    The density ceiling is taken from the FETCHED source named in
    :data:`CITED_SAFETY_SOURCES` and is recorded as MEASURED_CITED with that
    source.  The current ceiling has NO fetched source behind it and is
    therefore recorded as ASSUMED with an explicit sweep range, which the
    ledger enforces.
    """
    r = _finite(contact_radius_um, "contact_radius_um", 0, True)
    if abs(r - CONTACT_RADIUS_UM) > 1e-9:
        raise ValueError(
            "the contact radius is fixed by the FIXED 7 um shaft diameter "
            "(radius %.3f um); %.3f um was requested" % (CONTACT_RADIUS_UM, r))
    area = 4.0 * math.pi * (r * 1e-4) ** 2
    src = CITED_SAFETY_SOURCES["shepherd2018"]
    source_string = ("%s doi:%s (PMID %s, PMCID %s), fetched %s from %s"
                     % (src["citation"], src["doi"], src["pmid"], src["pmcid"],
                        src["fetched_utc"], src["url"]))
    lim = ChargeInjectionLimits(
        contact_radius_um=r,
        contact_area_cm2=area,
        density_limit_uC_cm2=CITED_MAX_CHARGE_DENSITY_UC_CM2,
        density_provenance=PROVENANCE.MEASURED_CITED,
        density_source=source_string,
        density_sweep_uC_cm2=ASSUMED_DENSITY_SWEEP_UC_CM2,
        shannon_k=CITED_SHANNON_K,
        shannon_note=(
            "Shannon log10(Q/A) = k - log10(Q), k=1.75, Q in uC/phase, A in cm^2, "
            "as reported by the cited source for the AAMI cochlear-implant "
            "standard. At this contact's area the k=1.75 curve sits far ABOVE the "
            "216 uC/cm^2 ceiling (it would allow ~%.0f uC/cm^2 at the ceiling's "
            "charge), so the flat ceiling is what binds here -- not the Shannon "
            "curve." % _shannon_density(CITED_SHANNON_K,
                                        CITED_MAX_CHARGE_DENSITY_UC_CM2 * area)),
        assumed_current_limit_nA=1.0e4,
        assumed_current_limit_source=None,
        assumed_current_limit_sweep_nA=(1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5),
        extrapolation_note=(
            "The cited ceiling comes from 0.05-0.2 mm^2 platinum cochlear contacts "
            "in guinea pig. This contact is %.1f um across, %.0fx smaller in area. "
            "The transfer of a density ceiling across that gap is NOT established "
            "by the source, which explicitly warns it does not define safety "
            "outside its own parameter subset. Treat the limit as an ORDER OF "
            "MAGNITUDE guard, not a permission."
            % (2 * r, src["areas_used_by_source_cm2"][0] / area)),
    )
    if record and ledger is not None:
        record_limits(ledger, lim, src)
    return lim


def _shannon_density(k, q_uC):
    return 10.0 ** (float(k) - math.log10(float(q_uC)))


def record_limits(ledger, limits: ChargeInjectionLimits, src=None):
    """Write the limit model into a :class:`StimLedger`, provenance enforced."""
    src = src or CITED_SAFETY_SOURCES["shepherd2018"]
    source_string = ("%s doi:%s (PMID %s, PMCID %s), fetched %s from %s"
                     % (src["citation"], src["doi"], src["pmid"], src["pmcid"],
                        src["fetched_utc"], src["url"]))
    ledger.record(
        "stim.limit.max_charge_density_uC_cm2", limits.density_limit_uC_cm2,
        "uC/cm^2/phase", PROVENANCE.MEASURED_CITED, source=source_string,
        note=("AAMI limit as reported by the fetched source: the Shannon relation "
              "(k=1.75) PLUS a flat ceiling of 216 uC/cm^2/phase. Quote: "
              + src["quotes"]["aami_standard"]))
    ledger.record(
        "stim.limit.cited_extreme_densities_uC_cm2",
        list(CITED_EXTREME_DENSITIES_UC_CM2), "uC/cm^2/phase",
        PROVENANCE.MEASURED_CITED, source=source_string,
        note=("the source's own measured anchors: 100 uC/cm^2/phase produced no "
              "significant tissue response or corrosion, 267 and 400 did. Quote: "
              + src["quotes"]["measured_anchors"]))
    ledger.record(
        "stim.limit.shannon_k", limits.shannon_k, "1", PROVENANCE.MEASURED_CITED,
        source=source_string,
        note=("k=1.75 is the AAMI variant named by the source; the source also "
              "states the underlying k range is 1.5-1.85. Quote: "
              + src["quotes"]["shannon_equation"]))
    ledger.record(
        "stim.limit.contact_area_cm2", limits.contact_area_cm2, "cm^2",
        PROVENANCE.MEASURED_LOCAL,
        source="computed: 4*pi*(7 um / 2)^2 in cm^2, the sanctioned solver's "
               "sphere contact",
        note="FIXED geometry input (shaft diameter 7 um), not a measurement")
    ledger.record(
        "stim.limit.charge_per_phase_at_density_limit_uC",
        limits.charge_limit_per_phase_uC(), "uC/phase", PROVENANCE.MEASURED_LOCAL,
        source="computed: density limit x contact area",
        note=("DERIVED from the cited density and the FIXED geometry; the "
              "derivation is this module's, not the source's."))
    ledger.record(
        "stim.limit.max_current_per_phase_nA", limits.assumed_current_limit_nA,
        "nA", PROVENANCE.ASSUMED,
        source=limits.assumed_current_limit_source,
        sweep=limits.assumed_current_limit_sweep_nA,
        note=("NO source was fetched for a current-per-phase ceiling in this "
              "geometry or tissue, so this is an assumption and the sweep names "
              "the range it must be varied over before any claim is made."))
    ledger.record(
        "stim.limit.extrapolation_warning", limits.extrapolation_note, "text",
        PROVENANCE.ASSUMED,
        sweep=src["areas_used_by_source_cm2"],
        note=("sweep = the contact AREAS the cited limit was actually derived "
              "from, in cm^2; the fixed contact is %.3g cm^2. Quote of the "
              "source's own warning: %s"
              % (limits.contact_area_cm2, src["quotes"]["extrapolation_warning"])))
    ledger.record(
        "stim.limit.density_sweep_uC_cm2", list(limits.density_sweep_uC_cm2),
        "uC/cm^2/phase", PROVENANCE.ASSUMED,
        sweep=limits.density_sweep_uC_cm2,
        note=("the density ceiling treated as an ASSUMPTION FAMILY the runner "
              "sweeps (rather than a transferable truth) for the case where fly "
              "tissue tolerance is not established."))
    return ledger


# ---------------------------------------------------------------------------
# provenance ledger (reuses the project's enforcing Ledger)
# ---------------------------------------------------------------------------
class StimLedger(Ledger):
    """The project's ENFORCING provenance ledger, specialised for stimulation.

    The two refusal rules are inherited, unchanged, from
    :meth:`engine.embodied.access_map.Ledger.record`:

    1. ``MEASURED_CITED`` without a ``source`` RAISES -- a cited measurement
       with no citation is fabricated provenance.
    2. ``ASSUMED`` without a declared sweep/range RAISES -- an assumption you
       are not prepared to vary is an unexamined constant.

    :meth:`selftest_refusals` proves both rules fire, so the guarantee is
    measured rather than asserted.
    """

    def record_limit(self, key, value, unit, provenance, **kw):
        return self.record(key, value, unit, provenance, **kw)

    def selftest_refusals(self):
        """Attempt both forbidden records; return what happened."""
        out = {}
        try:
            self.record("__selftest_cited_without_source", 1.0, "1",
                        PROVENANCE.MEASURED_CITED, note="must be refused")
            out["MEASURED_CITED_without_source"] = "ACCEPTED (BUG)"
        except ValueError as exc:
            out["MEASURED_CITED_without_source"] = "REFUSED: %s" % (exc,)
        try:
            self.record("__selftest_assumed_without_sweep", 1.0, "1",
                        PROVENANCE.ASSUMED, note="must be refused")
            out["ASSUMED_without_sweep"] = "ACCEPTED (BUG)"
        except ValueError as exc:
            out["ASSUMED_without_sweep"] = "REFUSED: %s" % (exc,)
        # and the legal forms must still work
        try:
            self.record("__selftest_cited_with_source", 1.0, "1",
                        PROVENANCE.MEASURED_CITED, source="selftest source")
            out["MEASURED_CITED_with_source"] = "ACCEPTED"
        except ValueError as exc:  # pragma: no cover
            out["MEASURED_CITED_with_source"] = "REFUSED: %s" % (exc,)
        try:
            self.record("__selftest_assumed_with_sweep", 1.0, "1",
                        PROVENANCE.ASSUMED, sweep=(1.0, 2.0))
            out["ASSUMED_with_sweep"] = "ACCEPTED"
        except ValueError as exc:  # pragma: no cover
            out["ASSUMED_with_sweep"] = "REFUSED: %s" % (exc,)
        out["all_rules_enforced"] = all(
            "REFUSED" in str(out[k])
            for k in ("MEASURED_CITED_without_source", "ASSUMED_without_sweep"))
        return out


# ---------------------------------------------------------------------------
# 4. stimulate and record
# ---------------------------------------------------------------------------
def stimulate_and_record(stim_contacts, stim_currents_nA, record_sites_um,
                         reference_um=DEFAULT_REFERENCE_UM,
                         conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M,
                         bandwidth_Hz=1000.0, noise_sd_mV=0.0, dt_ms=0.1,
                         n_steps=1, membrane_contacts=(), membrane_currents_nA=(),
                         seed=0):
    """Inject through the stimulating contacts; record on OTHER sites.

    The stimulus ARTIFACT is produced explicitly by the EXISTING front end:
    :class:`engine.electrode.VoltageRecorder` takes ``stimulus_field`` and
    ``stimulus_currents_nA`` and returns ``artifact_unfiltered_mV`` on its own
    artifact channel.  This module does not compute any potential itself.

    The recording site and the recorder's reference are the same two points for
    every channel, and the recorder's reference is the field's own explicit
    reference, so ``artifact = Ve(recording site) - Ve(reference)`` is a real
    differential voltage between the two recorded points, not a difference of
    two differently-referenced numbers.

    ``membrane_contacts`` / ``membrane_currents_nA`` optionally add explicitly
    balanced membrane current sources (the convention of
    :func:`engine.embodied.access_map.access_map_to_recording`), so a
    signal-to-artifact ratio can be formed.  They are static current sources of
    a declared magnitude, NOT spiking cells.
    """
    contacts = tuple(stim_contacts)
    if len(contacts) < 2:
        raise ValueError(
            "stimulating contacts must include a return path (>=2 contacts); an "
            "unbalanced injection is not a closed-domain field")
    cur = np.asarray(stim_currents_nA, dtype=float)
    if cur.shape != (len(contacts),) or not np.isfinite(cur).all():
        raise ValueError("one finite current per stimulating contact required")
    sites = np.asarray(record_sites_um, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 3 or not np.isfinite(sites).all():
        raise ValueError("record_sites_um must be finite (n,3)")
    # A recording site may not sit inside a current-carrying contact: there the
    # solver returns the in-sphere value, which is not a recordable tissue
    # potential (it would report hundreds of mV).
    for c in contacts:
        dmin = float(np.linalg.norm(sites - np.asarray(c.center_um), axis=1).min())
        if dmin < float(c.radius_um) + 1.0:
            raise ValueError(
                "a recording site is %.3f um from a current-carrying contact of "
                "radius %.3f um; a channel cannot record inside a stimulating "
                "contact. Move the site or choose different channels."
                % (dmin, float(c.radius_um)))
    steps = _finite(n_steps, "n_steps", 1, True, integer=True)

    field = BipolarField(contacts, conductivity_S_m, reference_um=reference_um)
    _check_reference_clearance(contacts, reference_um)
    ref = np.asarray(reference_um, dtype=float)

    mem_contacts = tuple(membrane_contacts)
    mem_cur = np.asarray(membrane_currents_nA, dtype=float)
    if mem_contacts and mem_cur.shape != (len(mem_contacts),):
        raise ValueError("one finite current per membrane source contact required")

    out = []
    for i, site in enumerate(sites):
        rec = VoltageRecorder(position_um=tuple(float(v) for v in site),
                              reference_um=tuple(float(v) for v in ref),
                              conductivity_S_m=conductivity_S_m,
                              bandwidth_Hz=bandwidth_Hz, noise_sd_mV=noise_sd_mV,
                              seed=seed + i)
        trace = []
        for _ in range(int(steps)):
            row = rec.step(dt_ms, source_contacts=mem_contacts,
                           outward_currents_nA=mem_cur,
                           stimulus_field=field,
                           stimulus_currents_nA=cur)
            trace.append(row)
        art = [t["artifact_unfiltered_mV"] for t in trace]
        neu = [t["neural_unfiltered_mV"] for t in trace]
        out.append({
            "site_um": [float(v) for v in site],
            "distance_to_nearest_stim_contact_um": float(
                min(np.linalg.norm(site - np.asarray(c.center_um)) for c in contacts)),
            "artifact_unfiltered_mV": float(art[-1]),
            "artifact_peak_abs_mV": float(max(abs(v) for v in art)),
            "neural_unfiltered_mV": float(neu[-1]),
            "measured_mV": float(trace[-1]["measured_mV"]),
            "trace_artifact_mV": [float(v) for v in art],
            "n_steps": int(steps),
            "dt_ms": float(dt_ms),
        })
    return {
        "stim_contacts_um": [list(c.center_um) for c in contacts],
        "stim_contact_radius_um": [float(c.radius_um) for c in contacts],
        "stim_currents_nA": [float(v) for v in cur],
        "reference_um": [float(v) for v in ref],
        "conductivity_S_m": float(conductivity_S_m),
        "bandwidth_Hz": float(bandwidth_Hz),
        "noise_sd_mV": float(noise_sd_mV),
        "n_membrane_sources": int(len(mem_contacts)),
        "channels": out,
        "artifact_model": ("engine.electrode.VoltageRecorder artifact channel "
                           "('artifact_unfiltered_mV'); the potential comes from "
                           "engine.electrode.transfer_mV_per_nA via BipolarField"),
        "artifact_interpretation": (
            "a differential voltage between the recording site and the declared "
            "reference, BEFORE the recorder's first-order bandwidth filter; it is "
            "a quasistatic ohmic field, not an amplifier saturation, not an "
            "impedance effect and not a double-layer transient"),
        "limitations": [
            "no electrode impedance, no double layer, no amplifier model",
            "no common-mode rejection, no blanking/switching model",
            "static membrane sources only; no spiking cells contribute here",
            NO_CLAIMS,
        ],
    }


def nearest_soma_distances(centres_um, soma_positions_um):
    """Distance from each contact to the nearest REAL soma point."""
    from scipy.spatial import cKDTree
    c = np.asarray(centres_um, dtype=float)
    tree = cKDTree(np.asarray(soma_positions_um, dtype=float))
    d, idx = tree.query(c, k=1)
    return {"distance_um": [float(v) for v in d],
            "nearest_row": [int(v) for v in idx],
            "min_um": float(np.min(d)), "median_um": float(np.median(d)),
            "max_um": float(np.max(d))}


def attenuation_law_check(distances_um, potentials_mV, current_nA,
                          conductivity_S_m=DEFAULT_CONDUCTIVITY_S_M):
    """Compare a measured potential profile to the ideal 1/r monopole.

    Returns the ratio measured/ideal at each distance.  This is the check that
    says the field really is the solver's Green function and not something else.
    """
    d = np.asarray(distances_um, dtype=float)
    v = np.asarray(potentials_mV, dtype=float)
    if d.shape != v.shape or np.any(d <= 0):
        raise ValueError("distances and potentials must match and be positive")
    ideal = float(current_nA) / (4.0 * math.pi * float(conductivity_S_m) * d)
    return {"distances_um": d.tolist(), "measured_mV": v.tolist(),
            "ideal_monopole_mV": ideal.tolist(),
            "ratio": (v / ideal).tolist()}
