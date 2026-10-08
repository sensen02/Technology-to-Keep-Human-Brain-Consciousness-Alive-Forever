"""Illustrative dopamine release/clearance model driven by a MEASURED protocol.

WHAT IS MEASURED AND ADOPTED VERBATIM (cited, not fitted)
--------------------------------------------------------
Source (open-access full text, read directly): Dumitrescu, Copeland & Venton,
ACS Chem Neurosci, PMC9897283 (NIHMS1865290), dissected adult Drosophila brain,
fast-scan cyclic voltammetry (FSCV) detection after a pharmacological pulse.
Measured protocol structure used exactly as reported:
  * 6 consecutive stimuli, every stimulus a 0.2 pmol ACh bolus;
  * the bolus is applied 5 s into that stimulus' record;
  * 10 min (600 s) between stimuli; 1 min (60 s) of recording per stimulus;
  * representative traces are 15 s long;
  * about 20 min (1200 s) of rest after dissection before recording.
Measured QUALITATIVE outcome used as the acceptance target (again, not fitting):
  * young control: release approximately constant across the 6 stimuli;
  * aged: release decreases across stimuli;
  * the half-decay time t1/2 of each trace is NOT significantly different
    between groups, i.e. the change is in RELEASE AMOUNT, not clearance speed.

WHAT IS *NOT* MEASURED HERE (illustrative, swept, and labelled as such)
----------------------------------------------------------------------
There is NO quantitative fit to the paper and none is claimed. The paper's
absolute concentration-vs-time values exist only inside figures; the PDF could
not be retrieved (blocked), so no digitised curve is available. Therefore every
kinetic constant, pool size, volume, efficacy and noise level below is an
ILLUSTRATIVE choice of ours, explicitly NOT a measurement:
  pool_capacity_pmol, refill_tau_s, stimulus_efficacy, release_tau_s,
  clearance_tau_s, voxel_volume_um3, release_shape_coupling, noise_std_nM.
The testable structural claim in this module is a HYPOTHESIS, not a mechanism
found in the source: changing the RELEASABLE POOL (size and/or refilling rate)
reproduces "release declines while t1/2 stays put", whereas changing CLEARANCE
breaks the t1/2 invariance. Whether the aged fly brain actually does this is an
open question; the source reports no pool or clearance measurement.

SUBSTANCE IDENTITY (do not conflate)
------------------------------------
The stimulus in the source experiment is acetylcholine (ACh); the transmitter
whose evoked release was detected is dopamine (DA). The 0.2 pmol ACh bolus is a
STIMULUS DOSE and is NEVER treated as a dopamine amount. Released DA comes from
a finite releasable pool through the explicitly illustrative parameter
`stimulus_efficacy` [pmol DA per pmol ACh at full pool]. The causal chain from
ACh receptor activation to DA release (receptor subtype, membrane current,
release probability) is NOT modelled.

UNITS (explicit throughout; see also engine/local_tissue.py conventions)
-----------------------------------------------------------------------
Amount: pmol (1 pmol = 1e-12 mol). Concentration: nM and uM. Volume: um^3.
Time: s. Concentration in a well-mixed volume V [um^3] is
    C[nM] = amount[pmol] * 1e-12 mol/pmol / (V * 1e-15 L/um^3) * 1e9 nM/(mol/L)
so 1 pmol uniformly in 1 um^3 = 1e12 nM, and 1 pmol in 1 uL (1e9 um^3) = 1 uM.
The default `voxel_volume_um3 = 1e8` um^3 (= 0.1 mm^3 = 100 nL, a nominal
tissue/release volume for a small brain) is ILLUSTRATIVE, not measured; the
report states it and shows the conversion. Absolute concentrations inherit that
arbitrariness and are NOT calibrated to the paper (which reports uM-level
peaks in figures only).

MODEL (deterministic, exact, no hidden repairs)
-----------------------------------------------
Finite releasable pool R [pmol], capacity R0, well mixed, refilling to R0 with
first-order time constant `refill_tau_s`:
    R_next = R0 + (R - A_n - R0) * exp(-ISI / refill_tau_s)
Per stimulus n (at the measured bolus time), with the measured ACh dose D:
    A_n = min( stimulus_efficacy * D * (R_n / R0),  R_n )     [pmol of DA]
    R_n -= A_n                                    (pool decrement at the bolus)
Extracellular appearance is a single-exponential flux of total A_n with time
constant `release_tau_s` (release_tau_s = 0 means an instantaneous bolus), and
extracellular dopamine is removed by FIRST-ORDER clearance with rate
k = 1/clearance_tau_s [1/s], giving clearance half-life ln2/k. Because removal
is first order and the release waveform shape does not depend on the pool when
`release_shape_coupling = 0`, the concentration trace is C_A(t) = A * shape(t):
a pure amplitude scaling. Hence the decline of A_n across stimuli changes PEAK
HEIGHT ONLY and leaves the measured t1/2 exactly invariant, whereas changing
clearance changes the shape and therefore moves t1/2. That is the whole
structural argument; `release_shape_coupling > 0` (release shortens as the pool
depletes) is provided to show explicitly where the invariance BREAKS.

Analytic propagation is used between stimuli (no time-stepping error); sampled
"recorded" traces are produced on the measured 60 s record windows with the
bolus 5 s in, plus an optional additive seeded observation noise that never
touches the state (it models detector noise, NOT a measured noise level).

NOT CLAIMED
-----------
No quantitative fit, no validated physiology, no statement about consciousness,
viability, welfare or medical use, no claim that the aged fly brain changes its
pool, and no claim that ACh is the transmitter being modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .params import DERIVED, ILLUSTRATIVE, MEASURED, ParamRegistry

SOURCE = ("Dumitrescu, Copeland & Venton, ACS Chem Neurosci, PMC9897283 "
          "(NIHMS1865290), open-access full text; protocol timing and the "
          "qualitative release/t1/2 pattern are the measured inputs")
SPECIES = "Drosophila melanogaster"
STAGE = "adult (dissected brain)"

PMOL_TO_MOL = 1e-12
UM3_TO_LITRE = 1e-15
MOL_PER_LITRE_TO_NM = 1e9
NM_PER_UM3_PER_PMOL = 1e12          # 1 pmol / 1 um^3 = 1e12 nM
NM_TO_UM = 1e-3
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------
# validation helpers (same style as engine/local_tissue.py: bad input raises)
# ----------------------------------------------------------------------------
def _number(value, name, minimum=None, exclusive=False, integer=False):
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        if exclusive and x <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {x}")
        if not exclusive and x < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {x}")
    if integer and float(x) != float(round(x)):
        raise ValueError(f"{name} must be an integer, got {x}")
    return x


def amount_pmol_to_concentration_nM(amount_pmol, volume_um3):
    """Explicit pmol -> nM conversion in a stated well-mixed volume [um^3].

    C[nM] = amount[pmol] * 1e-12 mol/pmol / (V[um^3] * 1e-15 L/um^3) * 1e9.
    """
    amount = _number(amount_pmol, "amount_pmol", minimum=0.0)
    volume = _number(volume_um3, "volume_um3", minimum=0.0, exclusive=True)
    moles = amount * PMOL_TO_MOL
    litres = volume * UM3_TO_LITRE
    return moles / litres * MOL_PER_LITRE_TO_NM


def concentration_nM_to_amount_pmol(concentration_nM, volume_um3):
    """Inverse of :func:`amount_pmol_to_concentration_nM` (amount in pmol)."""
    conc = _number(concentration_nM, "concentration_nM")
    if conc < 0:
        raise ValueError("concentration_nM must be >= 0")
    volume = _number(volume_um3, "volume_um3", minimum=0.0, exclusive=True)
    return conc * volume / NM_PER_UM3_PER_PMOL


# ----------------------------------------------------------------------------
# measured protocol
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtocolSpec:
    """MEASURED timing structure of the source experiment (PMC9897283).

    Every default here is a measured protocol quantity, adopted verbatim. Do
    not treat these as tunable kinetics; they are the input timeline.
    """

    n_stimulations: int = 6
    ach_dose_pmol: float = 0.2
    inter_stimulus_interval_s: float = 600.0
    recording_duration_s: float = 60.0
    representative_trace_s: float = 15.0
    pre_bolus_baseline_s: float = 5.0
    post_dissection_rest_s: float = 1200.0

    def __post_init__(self):
        _number(self.n_stimulations, "n_stimulations", minimum=1.0, integer=True)
        if int(self.n_stimulations) < 1:
            raise ValueError("n_stimulations must be >= 1")
        _number(self.ach_dose_pmol, "ach_dose_pmol", minimum=0.0)
        _number(self.inter_stimulus_interval_s, "inter_stimulus_interval_s",
                minimum=0.0, exclusive=True)
        _number(self.recording_duration_s, "recording_duration_s",
                minimum=0.0, exclusive=True)
        _number(self.representative_trace_s, "representative_trace_s",
                minimum=0.0, exclusive=True)
        _number(self.pre_bolus_baseline_s, "pre_bolus_baseline_s", minimum=0.0)
        _number(self.post_dissection_rest_s, "post_dissection_rest_s", minimum=0.0)
        if self.pre_bolus_baseline_s > self.recording_duration_s:
            raise ValueError("pre_bolus_baseline_s must not exceed the record")
        if self.representative_trace_s > self.recording_duration_s:
            raise ValueError("representative_trace_s must not exceed the record")

    # -- derived timeline ----------------------------------------------------
    @property
    def bolus_times_s(self):
        """Absolute bolus times [s] counted from the first record's start."""
        return tuple(self.pre_bolus_baseline_s
                     + n * self.inter_stimulus_interval_s
                     for n in range(int(self.n_stimulations)))

    @property
    def record_starts_s(self):
        return tuple(t - self.pre_bolus_baseline_s for t in self.bolus_times_s)

    @property
    def record_ends_s(self):
        return tuple(t + self.recording_duration_s - self.pre_bolus_baseline_s
                     for t in self.bolus_times_s)

    @property
    def session_end_s(self):
        return float(self.record_ends_s[-1])

    @property
    def representative_ends_s(self):
        return tuple(t + self.representative_trace_s - self.pre_bolus_baseline_s
                     for t in self.bolus_times_s)

    def n_record_samples(self, sample_dt_s):
        dt = _number(sample_dt_s, "sample_dt_s", minimum=0.0, exclusive=True)
        return int(round(self.recording_duration_s / dt)) + 1

    def timing_report(self):
        """Machine-checkable statement of the measured timeline."""
        bolus = self.bolus_times_s
        starts, ends = self.record_starts_s, self.record_ends_s
        spacings = tuple(b - a for a, b in zip(bolus, bolus[1:]))
        return {
            "n_stimulations": int(self.n_stimulations),
            "bolus_times_s": list(bolus),
            "bolus_spacing_s": list(spacings),
            "bolus_spacing_all_exactly_600_s": all(s == 600.0 for s in spacings),
            "ach_dose_pmol_per_bolus": self.ach_dose_pmol,
            "bolus_offset_within_record_s": self.pre_bolus_baseline_s,
            "record_windows_s": [[s, e] for s, e in zip(starts, ends)],
            "record_duration_s": self.recording_duration_s,
            "representative_trace_s": self.representative_trace_s,
            "post_dissection_rest_s": self.post_dissection_rest_s,
            "session_span_s": self.session_end_s,
            "session_span_min": self.session_end_s / 60.0,
            "first_record_allows_5_s_baseline_before_bolus":
                starts[0] == 0.0 and bolus[0] == self.pre_bolus_baseline_s,
            "provenance": "measured, " + SOURCE,
        }


# ----------------------------------------------------------------------------
# illustrative kinetics
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DAKinetics:
    """ILLUSTRATIVE kinetics and geometry. None of these is a measurement.

    `pool_capacity_pmol`  finite releasable DA pool, pmol (illustrative).
    `refill_tau_s`        first-order refilling time constant of that pool, s.
    `stimulus_efficacy`   pmol DA released per pmol ACh at full pool, 1
                          (a stimulus->release conversion, illustrative; it is
                          NOT a measured ACh->DA coupling).
    `release_tau_s`       single-exponential extracellular appearance constant
                          of a released bolus, s (0 = instantaneous).
    `clearance_tau_s`     first-order extracellular DA removal time constant, s;
                          half-life = ln2 * clearance_tau_s.
    `voxel_volume_um3`    assumed well-mixed volume for the pmol -> nM step.
    `release_shape_coupling` 0 = release waveform shape independent of pool
                          content (invariance holds); 1 = release time constant
                          scales linearly with pool content (invariance breaks).
    `noise_std_nM`        additive observation noise on stored traces only.
    """

    pool_capacity_pmol: float = 1.0
    refill_tau_s: float = 30.0
    stimulus_efficacy: float = 0.05
    release_tau_s: float = 0.5
    clearance_tau_s: float = 5.0
    voxel_volume_um3: float = 1e8
    release_shape_coupling: float = 0.0
    noise_std_nM: float = 0.0
    label: str = "illustrative"

    def __post_init__(self):
        _number(self.pool_capacity_pmol, "pool_capacity_pmol",
                minimum=0.0, exclusive=True)
        _number(self.refill_tau_s, "refill_tau_s", minimum=0.0, exclusive=True)
        _number(self.stimulus_efficacy, "stimulus_efficacy", minimum=0.0)
        _number(self.release_tau_s, "release_tau_s", minimum=0.0)
        _number(self.clearance_tau_s, "clearance_tau_s", minimum=0.0, exclusive=True)
        _number(self.voxel_volume_um3, "voxel_volume_um3", minimum=0.0, exclusive=True)
        _number(self.release_shape_coupling, "release_shape_coupling", minimum=0.0)
        _number(self.noise_std_nM, "noise_std_nM", minimum=0.0)
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a nonempty string")

    # -- derived -------------------------------------------------------------
    @property
    def clearance_k_s(self):
        """First-order removal rate [1/s]."""
        return 1.0 / self.clearance_tau_s

    @property
    def clearance_half_life_s(self):
        """ln2/k [s], the clearance-only half-decay."""
        return LN2 * self.clearance_tau_s

    def refill_recovery_per_interval(self, protocol):
        """Fraction of a pool deficit erased during one inter-stimulus gap."""
        p = protocol or ProtocolSpec()
        return 1.0 - math.exp(-p.inter_stimulus_interval_s / self.refill_tau_s)

    def refill_tau_over_isi(self, protocol):
        p = protocol or ProtocolSpec()
        return self.refill_tau_s / p.inter_stimulus_interval_s

    def conversion_report(self, protocol=None):
        """Explicit pmol -> concentration arithmetic for the assumed volume."""
        p = protocol or ProtocolSpec()
        v = self.voxel_volume_um3
        per_pmol = amount_pmol_to_concentration_nM(1.0, v)
        dose_release = self.stimulus_efficacy * p.ach_dose_pmol
        return {
            "voxel_volume_um3": v,
            "voxel_volume_mm3": v * 1e-9,
            "voxel_volume_uL": v * 1e-9,
            "volume_is_illustrative": True,
            "formula": ("C[nM] = amount[pmol]*1e-12/(V[um^3]*1e-15)*1e9 "
                        "= amount[pmol]*1e12/V[um^3]"),
            "nM_per_pmol_in_this_volume": per_pmol,
            "uM_per_pmol_in_this_volume": per_pmol * NM_TO_UM,
            "pmol_released_per_stimulus_at_full_pool": dose_release,
            "equivalent_uM_if_one_stimulus_bolus_in_this_volume":
                dose_release * per_pmol * NM_TO_UM,
            "note": ("the ACh dose is a stimulus, not dopamine; released DA is "
                     "dose*stimulus_efficacy, and the volume is illustrative"),
        }


# ----------------------------------------------------------------------------
# exact analytic release/clearance propagation
# ----------------------------------------------------------------------------
def _conc_shape(u, c_start_nM, pre_s, q_nM, release_tau_s, k_s):
    """Extracellular concentration [nM] at offsets u [s] from a bolus.

    Valid for u >= -pre_s. The release contributes a total amount whose uniform
    equivalent concentration is q_nM [nM], appearing as a single exponential of
    time constant release_tau_s; removal is first order with rate k_s.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if not np.all(np.isfinite(u)):
        raise ValueError("query times must be finite")
    if c_start_nM < 0 or q_nM < 0 or k_s < 0:
        raise ValueError("concentration, release and clearance must be nonnegative")
    out = np.empty(u.shape, dtype=float)
    before = u < 0.0
    if np.any(before):
        out[before] = c_start_nM * np.exp(-k_s * (u[before] + pre_s))
    after = ~before
    if np.any(after):
        ua = u[after]
        a = k_s
        c0 = c_start_nM * math.exp(-a * pre_s)
        if release_tau_s <= 0.0:
            out[after] = (c0 + q_nM) * np.exp(-a * ua)
        else:
            b = 1.0 / release_tau_s
            if abs(a - b) <= 1e-12 * max(a, b):
                out[after] = (c0 + q_nM * b * ua) * np.exp(-a * ua)
            else:
                k_amp = q_nM * b / (a - b)
                out[after] = (c0 * np.exp(-a * ua)
                              + k_amp * (np.exp(-b * ua) - np.exp(-a * ua)))
    if np.any(out < -1e-12):
        raise FloatingPointError("negative concentration from analytic shape")
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("nonfinite concentration from analytic shape")
    return np.maximum(out, 0.0)


def _conc_at(u, c_start_nM, pre_s, q_nM, release_tau_s, k_s):
    return float(_conc_shape(np.array([u]), c_start_nM, pre_s, q_nM,
                             release_tau_s, k_s)[0])


def _peak_time_s(c_at_bolus_nM, q_nM, release_tau_s, k_s):
    """Analytic time [s] from the bolus to the trace maximum (>= 0).

    C(u) = c0*exp(-a u) + K*(exp(-b u) - exp(-a u)), K = q*b/(a-b), a=k, b=1/tau.
    C'(u)=0 gives exp((b-a)u) = K*b/(a*(K-c0)); the interior maximum exists iff
    the initial release flux beats clearance of the carried-over concentration,
    q*b > a*c0 (K may be of either sign, so the guard is on q*b, not on K).
    """
    if q_nM <= 0.0 or release_tau_s <= 0.0:
        return 0.0
    a, b = k_s, 1.0 / release_tau_s
    c0 = c_at_bolus_nM
    if q_nM * b <= a * c0:
        return 0.0
    if abs(a - b) <= 1e-12 * max(a, b):
        denom = a * q_nM * b
        if denom <= 0.0:
            return 0.0
        return max((q_nM * b - a * c0) / denom, 0.0)
    k_amp = q_nM * b / (a - b)
    ratio = k_amp * b / (a * (k_amp - c0))
    if ratio <= 0.0 or not math.isfinite(ratio):
        return 0.0
    return max(math.log(ratio) / (b - a), 0.0)


def _half_decay_s(c_at_bolus_nM, q_nM, release_tau_s, k_s, peak_time,
                  peak_nM, pre_s=0.0, c_start_nM=None):
    """Time [s] from the trace maximum to half of that maximum (200 bisections).

    Returns inf if the trace is flat (no release and no clearance).
    """
    if peak_nM <= 0.0:
        return float("inf")
    if k_s <= 0.0:
        return float("inf")
    start = c_start_nM if c_start_nM is not None else c_at_bolus_nM
    target = 0.5 * peak_nM
    lo = peak_time
    if _conc_at(lo, start, pre_s, q_nM, release_tau_s, k_s) <= target:
        return 0.0

    def f(x):
        return _conc_at(x, start, pre_s, q_nM, release_tau_s, k_s) - target

    hi = lo + 60.0 / k_s
    for _ in range(200):
        if f(hi) <= 0.0:
            break
        hi = lo + 2.0 * (hi - lo)
    else:
        return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) - peak_time


def curve_features(c_at_bolus_nM, q_nM, release_tau_s, k_s, pre_s=0.0,
                   c_start_nM=None):
    """Analytic (peak_nM, peak_time_s, half_time_from_bolus_s, t_half_s)."""
    c0 = float(c_at_bolus_nM)
    pre = float(pre_s)
    c_start = c0 * math.exp(k_s * pre) if c_start_nM is None else float(c_start_nM)
    peak_time = _peak_time_s(c0, q_nM, release_tau_s, k_s)
    peak = _conc_at(peak_time, c_start, pre, q_nM, release_tau_s, k_s)
    if q_nM <= 0.0:
        peak = max(c0, c_start)
        peak_time = 0.0
        t_half = LN2 / k_s if k_s > 0 and peak > 0 else float("inf")
        return peak, peak_time, peak_time + t_half, t_half
    t_half = _half_decay_s(c0, q_nM, release_tau_s, k_s, peak_time, peak,
                           pre_s=pre, c_start_nM=c_start)
    return peak, peak_time, peak_time + t_half, t_half


def _conc_integral(u1, u2, c_start_nM, pre_s, q_nM, release_tau_s, k_s):
    """Integral of C du [nM*s] between offsets u1 <= u2 (>= -pre_s)."""
    if u2 < u1:
        raise ValueError("integral bounds must satisfy u2 >= u1")
    total = 0.0
    a = k_s
    lo_pre, hi_pre = max(u1, -pre_s), min(u2, 0.0)
    if hi_pre > lo_pre:
        if a > 0.0:
            total += c_start_nM * math.exp(-a * pre_s) * (
                math.exp(-a * lo_pre) - math.exp(-a * hi_pre)) / a
        else:
            total += c_start_nM * (hi_pre - lo_pre)
    lo, hi = max(u1, 0.0), u2
    if hi > lo:
        c0 = c_start_nM * math.exp(-a * pre_s)

        def decay(rate):
            if rate <= 0.0:
                return hi - lo
            return (math.exp(-rate * lo) - math.exp(-rate * hi)) / rate

        if release_tau_s <= 0.0:
            total += (c0 + q_nM) * decay(a)
        elif q_nM <= 0.0:
            total += c0 * decay(a)
        else:
            b = 1.0 / release_tau_s
            if abs(a - b) <= 1e-12 * max(a, b):
                total += c0 * decay(a) + q_nM * b * (
                    (math.exp(-a * lo) * (1.0 + a * lo)
                     - math.exp(-a * hi) * (1.0 + a * hi)) / a ** 2)
            else:
                k_amp = q_nM * b / (a - b)
                total += c0 * decay(a) + k_amp * ((math.exp(-b * lo)
                    - math.exp(-b * hi)) / b - decay(a))
    return total


def measure_trace_features(time_s, concentration_nM):
    """Measure (peak, peak time, half time, t1/2) from a SAMPLED trace.

    Uses linear interpolation of the first descending crossing of peak/2. This
    is the "as recorded" measurement; :func:`curve_features` is the analytic
    counterpart and the two are compared in the selftest.
    """
    t = np.asarray(time_s, dtype=float)
    c = np.asarray(concentration_nM, dtype=float)
    if t.ndim != 1 or c.ndim != 1 or t.size != c.size or t.size < 3:
        raise ValueError("trace arrays must be 1-D of equal size >= 3")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)):
        raise ValueError("trace arrays must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("trace times must be strictly increasing")
    if np.any(c < 0.0):
        raise ValueError("trace concentrations must be nonnegative")
    i = int(np.argmax(c))
    peak = float(c[i])
    peak_time = float(t[i])
    if peak <= 0.0:
        return {"peak_nM": 0.0, "peak_time_s": peak_time,
                "half_time_s": float("nan"), "t_half_s": float("nan")}
    target = 0.5 * peak
    tail = c[i:]
    idx = np.flatnonzero(tail <= target)
    if idx.size == 0:
        return {"peak_nM": peak, "peak_time_s": peak_time,
                "half_time_s": float("nan"), "t_half_s": float("nan")}
    j = i + int(idx[0])
    if j == i:
        half_time = peak_time
    else:
        c1, c2 = c[j - 1], c[j]
        t1, t2 = t[j - 1], t[j]
        frac = 0.0 if c1 == c2 else (c1 - target) / (c1 - c2)
        half_time = float(t1 + frac * (t2 - t1))
    return {"peak_nM": peak, "peak_time_s": peak_time,
            "half_time_s": half_time, "t_half_s": half_time - peak_time}


# ----------------------------------------------------------------------------
# core scalar solve (exact pool recurrence + exact clearance integral)
# ----------------------------------------------------------------------------
def _run_core(protocol, kinetics):
    p, kin = protocol, kinetics
    n = int(p.n_stimulations)
    isi, pre = p.inter_stimulus_interval_s, p.pre_bolus_baseline_s
    k_s = kin.clearance_k_s
    tau_rel = kin.release_tau_s
    r0 = kin.pool_capacity_pmol
    g = math.exp(-isi / kin.refill_tau_s)

    out = {key: np.zeros(n) for key in (
        "bolus_times_s", "released_pmol", "requested_pmol", "pool_before_pmol",
        "pool_after_pmol", "refill_drawn_pmol", "release_tau_s", "q_nM",
        "c_start_nM", "c_at_bolus_nM", "peak_nM", "peak_time_s", "half_time_s",
        "t_half_s", "cleared_pmol", "interval_integral_nM_s", "capped")}
    pool = r0
    c_start = 0.0
    for i in range(n):
        bolus = p.bolus_times_s[i]
        out["bolus_times_s"][i] = bolus
        out["pool_before_pmol"][i] = pool
        requested = kin.stimulus_efficacy * p.ach_dose_pmol * (pool / r0) if r0 > 0 else 0.0
        released = min(requested, pool)
        out["requested_pmol"][i] = requested
        out["released_pmol"][i] = released
        out["capped"][i] = 1.0 if requested > pool else 0.0
        pool_after = pool - released
        out["pool_after_pmol"][i] = pool_after
        tau_i = tau_rel * (pool / r0) ** kin.release_shape_coupling
        tau_i = max(tau_i, 0.0)
        out["release_tau_s"][i] = tau_i
        q = amount_pmol_to_concentration_nM(released, kin.voxel_volume_um3)
        out["q_nM"][i] = q
        out["c_start_nM"][i] = c_start
        c_at_bolus = c_start * math.exp(-k_s * pre)
        out["c_at_bolus_nM"][i] = c_at_bolus
        peak, tpk, thalf_abs, thalf = curve_features(
            c_at_bolus, q, tau_i, k_s, pre_s=pre, c_start_nM=c_start)
        out["peak_nM"][i] = peak
        out["peak_time_s"][i] = tpk
        out["half_time_s"][i] = thalf_abs
        out["t_half_s"][i] = thalf
        # advance to the start of the next record window (or session end)
        u_next = ((p.bolus_times_s[i + 1] - pre) - bolus if i + 1 < n
                  else p.session_end_s - bolus)
        integral = _conc_integral(-pre, u_next, c_start, pre, q, tau_i, k_s)
        out["interval_integral_nM_s"][i] = integral
        out["cleared_pmol"][i] = (
            k_s * integral * kin.voxel_volume_um3 / NM_PER_UM3_PER_PMOL)
        c_start = _conc_at(u_next, c_start, pre, q, tau_i, k_s)
        # refilling during the whole inter-stimulus interval
        deficit_closed = (r0 - pool_after) * (1.0 - g)
        out["refill_drawn_pmol"][i] = deficit_closed
        pool = pool_after + deficit_closed
    out["pool_final_pmol"] = pool
    out["c_session_end_nM"] = c_start
    out["r0_pmol"] = r0
    out["clearance_k_s"] = k_s
    return out


def release_sequence(protocol=None, kinetics=None):
    """Per-stimulus release amounts [pmol] and pool contents; no sampling.

    Returns a dict of numpy arrays plus a mass ledger for the pool:
    R0 + total refilled = final pool + total released.
    """
    p = protocol or ProtocolSpec()
    kin = kinetics or DAKinetics()
    core = _run_core(p, kin)
    released = core["released_pmol"]
    refilled = core["refill_drawn_pmol"]
    pool_residual = (kin.pool_capacity_pmol + float(refilled.sum())
                     - float(released.sum()) - float(core["pool_final_pmol"]))
    return {
        "protocol": p,
        "kinetics": kin,
        "bolus_times_s": core["bolus_times_s"],
        "pool_before_pmol": core["pool_before_pmol"],
        "pool_after_pmol": core["pool_after_pmol"],
        "released_pmol": released,
        "refill_drawn_pmol": refilled,
        "release_ratio_last_first": (float(released[-1] / released[0])
                                     if released[0] > 0 else float("nan")),
        "declines_monotonically": bool(np.all(np.diff(released) <= 1e-12 * max(
            1e-300, float(np.max(np.abs(released)))))),
        "t_half_s": core["t_half_s"],
        "t_half_relative_spread": _relative_spread(core["t_half_s"]),
        "pool_ledger_residual_pmol": pool_residual,
        "ephemeral": core,
    }


def _relative_spread(values):
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return float("nan")
    mean = float(finite.mean())
    if mean == 0.0:
        return 0.0 if float(finite.max() - finite.min()) == 0.0 else float("inf")
    return float((finite.max() - finite.min()) / abs(mean))


# ----------------------------------------------------------------------------
# result container + full sampled simulation
# ----------------------------------------------------------------------------
@dataclass
class DAProtocolResult:
    protocol: ProtocolSpec
    kinetics: DAKinetics
    seed: int
    bolus_times_s: np.ndarray
    released_pmol: np.ndarray
    requested_pmol: np.ndarray
    capped_by_pool: np.ndarray
    pool_before_pmol: np.ndarray
    pool_after_pmol: np.ndarray
    refill_drawn_pmol: np.ndarray
    release_tau_s: np.ndarray
    q_nM: np.ndarray
    c_start_nM: np.ndarray
    c_at_bolus_nM: np.ndarray
    peak_nM: np.ndarray
    t_half_s: np.ndarray
    measured_peak_nM: np.ndarray
    measured_t_half_s: np.ndarray
    record_time_s: np.ndarray
    record_nM: np.ndarray
    representative_nM: np.ndarray
    timeline_time_s: np.ndarray
    timeline_nM: np.ndarray
    timeline_is_record: np.ndarray
    ledger: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    # -- derived views -------------------------------------------------------
    @property
    def release_ratio_last_first(self):
        a = self.released_pmol
        return float(a[-1] / a[0]) if a[0] > 0 else float("nan")

    @property
    def relative_release_decline(self):
        return 1.0 - self.release_ratio_last_first

    @property
    def declines_monotonically(self):
        d = np.diff(self.released_pmol)
        return bool(np.all(d <= 1e-12 * max(1e-300, float(np.abs(self.released_pmol).max()))))

    @property
    def t_half_relative_spread(self):
        return _relative_spread(self.measured_t_half_s)

    @property
    def t_half_relative_spread_analytic(self):
        return _relative_spread(self.t_half_s)

    def summary(self):
        kin, p = self.kinetics, self.protocol
        return {
            "label": kin.label,
            "measured_protocol": self.timing,
            "illustrative_parameters": {
                "pool_capacity_pmol": kin.pool_capacity_pmol,
                "refill_tau_s": kin.refill_tau_s,
                "refill_tau_over_isi": kin.refill_tau_over_isi(p),
                "refill_recovery_per_interval": kin.refill_recovery_per_interval(p),
                "stimulus_efficacy_pmol_DA_per_pmol_ACh": kin.stimulus_efficacy,
                "release_tau_s": kin.release_tau_s,
                "clearance_tau_s": kin.clearance_tau_s,
                "clearance_k_s": kin.clearance_k_s,
                "clearance_half_life_s": kin.clearance_half_life_s,
                "voxel_volume_um3": kin.voxel_volume_um3,
                "release_shape_coupling": kin.release_shape_coupling,
                "noise_std_nM": kin.noise_std_nM,
                "all_illustrative": True,
            },
            "conversion": kin.conversion_report(p),
            "per_stimulus": {
                "released_pmol": [float(x) for x in self.released_pmol],
                "release_fraction_of_pool": [float(x) for x in
                                             self.released_pmol / kin.pool_capacity_pmol],
                "peak_nM_analytic": [float(x) for x in self.peak_nM],
                "peak_uM_analytic": [float(x) * NM_TO_UM for x in self.peak_nM],
                "peak_nM_measured_from_trace": [float(x) for x in self.measured_peak_nM],
                "t_half_s_analytic": [float(x) for x in self.t_half_s],
                "t_half_s_measured_from_trace": [float(x) for x in self.measured_t_half_s],
                "pool_before_pmol": [float(x) for x in self.pool_before_pmol],
                "release_tau_s": [float(x) for x in self.release_tau_s],
                "capped_by_pool": [bool(x) for x in self.capped_by_pool],
            },
            "pattern": {
                "release_ratio_last_first": self.release_ratio_last_first,
                "relative_release_decline": self.relative_release_decline,
                "declines_monotonically": self.declines_monotonically,
                "t_half_relative_spread_measured": self.t_half_relative_spread,
                "t_half_relative_spread_analytic": self.t_half_relative_spread_analytic,
            },
            "ledger": self.ledger,
        }


def simulate(protocol=None, kinetics=None, sample_dt_s=0.02, coarse_dt_s=0.5,
             seed=None):
    """Run the measured protocol; return per-stimulus scalars and traces.

    `sample_dt_s` sets the intra-record sampling of the measured 1 min record
    windows (the bolus sits 5 s into each). Observation noise
    (`kinetics.noise_std_nM`) is added to the STORED traces only, never to the
    state; `seed=None` means the fixed default seed 0, so runs replay exactly.
    """
    p = protocol or ProtocolSpec()
    kin = kinetics or DAKinetics()
    dt = _number(sample_dt_s, "sample_dt_s", minimum=0.0, exclusive=True)
    coarse = _number(coarse_dt_s, "coarse_dt_s", minimum=0.0, exclusive=True)
    rng_seed = 0 if seed is None else int(_number(seed, "seed", integer=True))
    core = _run_core(p, kin)
    n = int(p.n_stimulations)
    m = p.n_record_samples(dt)

    rec_t = np.empty((n, m))
    rec_c = np.empty((n, m))
    rep = np.empty((n, m))
    rep[:, :] = np.nan
    n_rep = int(round(p.representative_trace_s / dt)) + 1
    if n_rep > m:
        raise ValueError("representative trace exceeds the record window")
    for i in range(n):
        start = p.record_starts_s[i]
        t = np.linspace(start, start + p.recording_duration_s, m)
        rec_t[i] = t
        rec_c[i] = _conc_shape(t - p.bolus_times_s[i], core["c_start_nM"][i],
                               p.pre_bolus_baseline_s, core["q_nM"][i],
                               core["release_tau_s"][i], kin.clearance_k_s)
        rep[i, :n_rep] = rec_c[i, :n_rep]

    measured_peak, measured_half = np.zeros(n), np.zeros(n)
    for i in range(n):
        feats = measure_trace_features(rec_t[i], rec_c[i])
        measured_peak[i] = feats["peak_nM"]
        measured_half[i] = feats["t_half_s"]

    if kin.noise_std_nM > 0.0:
        rng = np.random.default_rng(rng_seed)
        rec_c = np.maximum(rec_c + rng.normal(0.0, kin.noise_std_nM, rec_c.shape), 0.0)
        rep = rep.copy()
        rep[:, :n_rep] = rec_c[:, :n_rep]

    # timeline: rest (no release, C = 0) then every cycle at coarse sampling
    rest_n = int(round(p.post_dissection_rest_s / coarse)) + 1
    t_parts = [np.linspace(-p.post_dissection_rest_s, 0.0, rest_n)]
    c_parts = [np.zeros(rest_n)]
    flag_parts = [np.zeros(rest_n, dtype=bool)]
    for i in range(n):
        bolus = p.bolus_times_s[i]
        end = (p.bolus_times_s[i + 1] - p.pre_bolus_baseline_s if i + 1 < n
               else p.session_end_s)
        cnt = int(round((end - (bolus - p.pre_bolus_baseline_s)) / coarse)) + 1
        t_i = np.linspace(bolus - p.pre_bolus_baseline_s, end, cnt)
        c_i = _conc_shape(t_i - bolus, core["c_start_nM"][i],
                          p.pre_bolus_baseline_s, core["q_nM"][i],
                          core["release_tau_s"][i], kin.clearance_k_s)
        flag_i = np.zeros(cnt, dtype=bool)
        flag_i[t_i <= p.record_ends_s[i] + 1e-9] = True
        t_parts.append(t_i)
        c_parts.append(c_i)
        flag_parts.append(flag_i)
    timeline_t = np.concatenate(t_parts)
    timeline_c = np.concatenate(c_parts)
    timeline_flag = np.concatenate(flag_parts)

    released_total = float(core["released_pmol"].sum())
    cleared_total = float(core["cleared_pmol"].sum())
    final_extracellular = concentration_nM_to_amount_pmol(
        float(core["c_session_end_nM"]), kin.voxel_volume_um3)
    ledger = {
        "released_total_pmol": released_total,
        "cleared_total_pmol": cleared_total,
        "extracellular_at_session_end_pmol": final_extracellular,
        "refill_drawn_total_pmol": float(core["refill_drawn_pmol"].sum()),
        "pool_initial_pmol": kin.pool_capacity_pmol,
        "pool_final_pmol": float(core["pool_final_pmol"]),
        "pool_ledger_residual_pmol": (kin.pool_capacity_pmol
                                      + float(core["refill_drawn_pmol"].sum())
                                      - released_total
                                      - float(core["pool_final_pmol"])),
        "extracellular_ledger_residual_pmol": (released_total - cleared_total
                                               - final_extracellular),
        "units": "pmol (1 pmol = 1e-12 mol)",
        "note": ("released = cleared + still extracellular at session end; "
                 "refilling is drawn from an implicit unlimited precursor, so "
                 "the pool equation is closed only with that refill term"),
    }
    return DAProtocolResult(
        protocol=p, kinetics=kin, seed=rng_seed,
        bolus_times_s=core["bolus_times_s"],
        released_pmol=core["released_pmol"],
        requested_pmol=core["requested_pmol"],
        capped_by_pool=core["capped"].astype(bool),
        pool_before_pmol=core["pool_before_pmol"],
        pool_after_pmol=core["pool_after_pmol"],
        refill_drawn_pmol=core["refill_drawn_pmol"],
        release_tau_s=core["release_tau_s"],
        q_nM=core["q_nM"],
        c_start_nM=core["c_start_nM"],
        c_at_bolus_nM=core["c_at_bolus_nM"],
        peak_nM=core["peak_nM"],
        t_half_s=core["t_half_s"],
        measured_peak_nM=measured_peak,
        measured_t_half_s=measured_half,
        record_time_s=rec_t,
        record_nM=rec_c,
        representative_nM=rep,
        timeline_time_s=timeline_t,
        timeline_nM=timeline_c,
        timeline_is_record=timeline_flag,
        ledger=ledger,
        timing=p.timing_report(),
    )


# ----------------------------------------------------------------------------
# acceptance criteria + sweeps (report the region, do not claim a fit)
# ----------------------------------------------------------------------------
DEFAULT_DECLINE_THRESHOLD = 0.7      # A6/A1 <= 0.7  -> >= 30% decline
DEFAULT_T_HALF_TOLERANCE = 0.05      # spread(A) / mean(A) <= 5%


def pattern_verdict(result, decline_threshold=DEFAULT_DECLINE_THRESHOLD,
                    t_half_tolerance=DEFAULT_T_HALF_TOLERANCE):
    """Does this parameter point show 'release declines, t1/2 approximately fixed'?"""
    ratio = result.release_ratio_last_first
    spread = result.t_half_relative_spread
    declines = bool(ratio <= decline_threshold and result.declines_monotonically)
    invariant = bool(spread <= t_half_tolerance)
    return {
        "release_ratio_last_first": ratio,
        "release_declines_enough": declines,
        "t_half_relative_spread": spread,
        "t_half_approximately_constant": invariant,
        "reproduces_measured_pattern": bool(declines and invariant),
        "criteria": {"decline_threshold_ratio": decline_threshold,
                     "t_half_relative_spread_tolerance": t_half_tolerance},
        "caveat": ("criteria are OUR tolerance choices on a QUALITATIVE pattern; "
                   "this is not a statistical test and not a fit"),
    }


def sweep_pool_refill_efficacy(protocol=None, kinetics=None,
                               pool_sizes_pmol=None, refill_taus_s=None,
                               efficacies=None,
                               release_shape_coupling=None,
                               decline_threshold=DEFAULT_DECLINE_THRESHOLD,
                               t_half_tolerance=DEFAULT_T_HALF_TOLERANCE):
    """Grid over (pool size, refilling time constant, release efficacy).

    Returns per-combination decline ratio, monotonicity, t1/2 spread and the
    boolean acceptance region, plus explicit numeric boundaries.
    """
    p = protocol or ProtocolSpec()
    base = kinetics or DAKinetics()
    pool_sizes = list(pool_sizes_pmol if pool_sizes_pmol is not None
                      else (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0))
    refills = list(refill_taus_s if refill_taus_s is not None
                   else (5.0, 15.0, 60.0, 300.0, 600.0, 1800.0, 5400.0, 1.0e5))
    effs = list(efficacies if efficacies is not None
                else (0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
    coupling = (base.release_shape_coupling if release_shape_coupling is None
                else float(release_shape_coupling))
    for name, values in (("pool_sizes_pmol", pool_sizes),
                         ("refill_taus_s", refills), ("efficacies", effs)):
        if len(values) < 1:
            raise ValueError(f"{name} must have at least one value")
    r0_ref = base.pool_capacity_pmol
    rows, ratio, spread, declined, invariant, accepted = [], [], [], [], [], []
    for eff in effs:
        for tau in refills:
            for r0 in pool_sizes:
                kin = replace(base, pool_capacity_pmol=float(r0),
                              refill_tau_s=float(tau),
                              stimulus_efficacy=float(eff),
                              release_shape_coupling=coupling)
                core = _run_core(p, kin)
                rel = core["released_pmol"]
                ratio_last_first = float(rel[-1] / rel[0]) if rel[0] > 0 else float("nan")
                mono = bool(np.all(np.diff(rel) <= 1e-12 * max(1e-300, float(np.abs(rel).max()))))
                sp = _relative_spread(core["t_half_s"])
                dec = bool(ratio_last_first <= decline_threshold and mono)
                inv = bool(sp <= t_half_tolerance)
                rows.append({"pool_capacity_pmol": float(r0),
                             "refill_tau_s": float(tau),
                             "stimulus_efficacy": float(eff),
                             "release_fraction_of_pool_per_stimulus":
                                 float(eff * p.ach_dose_pmol / r0),
                             "released_pmol": [float(x) for x in rel],
                             "pool_before_pmol": [float(x) for x in core["pool_before_pmol"]],
                             "release_ratio_last_first": ratio_last_first,
                             "monotone_decreasing": mono,
                             "t_half_s": [float(x) for x in core["t_half_s"]],
                             "t_half_relative_spread": sp,
                             "peak_nM_first": float(core["peak_nM"][0]),
                             "peak_uM_last": float(core["peak_nM"][-1]) * NM_TO_UM})
                ratio.append(ratio_last_first)
                spread.append(sp)
                declined.append(dec)
                invariant.append(inv)
                accepted.append(bool(dec and inv))
    shape = (len(effs), len(refills), len(pool_sizes))
    ratio_arr = np.array(ratio).reshape(shape)
    spread_arr = np.array(spread).reshape(shape)
    acc_arr = np.array(accepted).reshape(shape)
    # boundaries: largest pool size still showing a decline, per (eff, refill)
    boundaries = {}
    for ie, eff in enumerate(effs):
        per_refill = {}
        for it, tau in enumerate(refills):
            ok = [pool_sizes[ip] for ip in range(len(pool_sizes)) if accepted[ie * len(refills) * len(pool_sizes) + it * len(pool_sizes) + ip]]
            dec_only = [pool_sizes[ip] for ip in range(len(pool_sizes))
                        if declined[ie * len(refills) * len(pool_sizes) + it * len(pool_sizes) + ip]]
            per_refill[f"refill_tau_{tau:g}s"] = {
                "max_pool_size_with_decline_and_invariance": (max(ok) if ok else None),
                "max_pool_size_with_decline_only": (max(dec_only) if dec_only else None),
                "min_pool_size_in_grid": min(pool_sizes),
            }
        boundaries[f"efficacy_{eff:g}"] = per_refill
    n_accept = int(acc_arr.sum())
    return {
        "axes": {
            "pool_sizes_pmol": [float(x) for x in pool_sizes],
            "refill_taus_s": [float(x) for x in refills],
            "efficacies": [float(x) for x in effs],
            "release_shape_coupling": coupling,
            "reference_pool_capacity_pmol": r0_ref,
            "inter_stimulus_interval_s": p.inter_stimulus_interval_s,
        },
        "criteria": {"decline_threshold_ratio": decline_threshold,
                     "t_half_relative_spread_tolerance": t_half_tolerance},
        "n_combinations": int(acc_arr.size),
        "n_accepted": n_accept,
        "fraction_accepted": n_accept / float(acc_arr.size),
        "region_found": bool(n_accept > 0),
        "decline_ratio_grid": ratio_arr.tolist(),
        "t_half_relative_spread_grid": spread_arr.tolist(),
        "acceptance_grid": acc_arr.tolist(),
        "boundaries_largest_pool_still_declining": boundaries,
        "rows": rows,
        "interpretation": [
            "decline needs the per-stimulus release to be a non-negligible "
            "fraction of the pool AND refilling to be slow relative to the "
            "measured 600 s inter-stimulus interval",
            "a pool far larger than the per-stimulus release cannot decline "
            "(release fraction -> 0), which is the young-control plateau",
            "t1/2 spread is ~0 throughout this grid whenever "
            "release_shape_coupling = 0: the pool changes amplitude, not shape",
        ],
        "warning": ("region of a QUALITATIVE pattern under OUR tolerance choice; "
                    "no quantitative fit to the source study is claimed"),
    }


def sweep_clearance(protocol=None, kinetics=None, clearance_taus_s=None,
                    pool_sizes_pmol=None, efficacies=None):
    """Sweep clearance only, for the fixed pool: shows release unchanged, t1/2 moved."""
    p = protocol or ProtocolSpec()
    base = kinetics or DAKinetics()
    taus = list(clearance_taus_s if clearance_taus_s is not None
                else (1.25, 2.5, 5.0, 10.0, 20.0, 40.0))
    pools = list(pool_sizes_pmol if pool_sizes_pmol is not None
                 else (base.pool_capacity_pmol,))
    effs = list(efficacies if efficacies is not None else (base.stimulus_efficacy,))
    for name, values in (("clearance_taus_s", taus), ("pool_sizes_pmol", pools),
                         ("efficacies", effs)):
        if len(values) < 1:
            raise ValueError(f"{name} must have at least one value")
    rows = []
    for tau in taus:
        for r0 in pools:
            for eff in effs:
                kin = replace(base, clearance_tau_s=float(tau),
                              pool_capacity_pmol=float(r0),
                              stimulus_efficacy=float(eff),
                              noise_std_nM=0.0)
                core = _run_core(p, kin)
                rel = core["released_pmol"]
                rows.append({
                    "clearance_tau_s": float(tau),
                    "clearance_k_s": kin.clearance_k_s,
                    "clearance_half_life_s": kin.clearance_half_life_s,
                    "pool_capacity_pmol": float(r0),
                    "stimulus_efficacy": float(eff),
                    "release_ratio_last_first": float(rel[-1] / rel[0]) if rel[0] > 0 else float("nan"),
                    "released_pmol": [float(x) for x in rel],
                    "t_half_s": [float(x) for x in core["t_half_s"]],
                    "t_half_relative_spread": _relative_spread(core["t_half_s"]),
                    "peak_nM_first": float(core["peak_nM"][0]),
                    "peak_ratio_last_first": (float(core["peak_nM"][-1] / core["peak_nM"][0])
                                              if core["peak_nM"][0] > 0 else float("nan")),
                })
    ref = [r for r in rows if r["clearance_tau_s"] == base.clearance_tau_s]
    ref_t = ref[0]["t_half_s"][0] if ref else float("nan")
    for r in rows:
        r["t_half_first_vs_reference_relative_change"] = (
            (r["t_half_s"][0] - ref_t) / ref_t if ref_t not in (0.0, float("nan")) and math.isfinite(ref_t) else float("nan"))
    return {
        "rows": rows,
        "reference_clearance_tau_s": base.clearance_tau_s,
        "interpretation": [
            "changing clearance scales t1/2 essentially linearly and leaves the "
            "across-stimulus release pattern (and its ratio) unchanged, so a "
            "clearance change cannot produce the aged declining release",
            "evidence: release_ratio_last_first is constant down each pool row "
            "while t_half_s moves with clearance_tau_s",
        ],
        "caveat": ("the source reports t1/2 NOT significantly different between "
                   "groups, which is a non-rejection, not proof of equality; a "
                   "small clearance change could escape detection"),
    }


def sweep_shape_coupling(protocol=None, kinetics=None, couplings=None):
    """Where the invariance BREAKS: release waveform shape tracking pool content."""
    p = protocol or ProtocolSpec()
    base = kinetics or DAKinetics()
    vals = list(couplings if couplings is not None else (0.0, 0.25, 0.5, 1.0, 2.0))
    if len(vals) < 1:
        raise ValueError("couplings must have at least one value")
    rows = []
    for c in vals:
        kin = replace(base, release_shape_coupling=float(c))
        core = _run_core(p, kin)
        rows.append({
            "release_shape_coupling": float(c),
            "t_half_s": [float(x) for x in core["t_half_s"]],
            "t_half_relative_spread": _relative_spread(core["t_half_s"]),
            "release_tau_s": [float(x) for x in core["release_tau_s"]],
            "release_ratio_last_first": (float(core["released_pmol"][-1] / core["released_pmol"][0])
                                         if core["released_pmol"][0] > 0 else float("nan")),
        })
    return {
        "rows": rows,
        "interpretation": ("coupling 0 keeps t1/2 exactly invariant (amplitude-only "
                           "pool effect); coupling > 0 shortens the release waveform "
                           "as the pool depletes and therefore moves t1/2, which is "
                           "where the 't1/2 unchanged' claim would fail"),
    }


def build_registry(protocol=None, kinetics=None):
    """Provenance registry: measured protocol vs illustrative kinetics."""
    p = protocol or ProtocolSpec()
    kin = kinetics or DAKinetics()
    reg = ParamRegistry("da_protocol")
    reg.define("n_stimulations", float(p.n_stimulations), "1", MEASURED,
               source=SOURCE, species=SPECIES, stage=STAGE,
               notes="6 consecutive ACh stimulations")
    reg.define("ach_dose_pmol", p.ach_dose_pmol, "pmol", MEASURED,
               source=SOURCE, species=SPECIES, stage=STAGE,
               notes="STIMULUS dose, not a dopamine amount")
    reg.define("inter_stimulus_interval_s", p.inter_stimulus_interval_s, "s",
               MEASURED, source=SOURCE, species=SPECIES, stage=STAGE,
               notes="10 min between stimulations")
    reg.define("recording_duration_s", p.recording_duration_s, "s", MEASURED,
               source=SOURCE, species=SPECIES, stage=STAGE)
    reg.define("representative_trace_s", p.representative_trace_s, "s", MEASURED,
               source=SOURCE, species=SPECIES, stage=STAGE)
    reg.define("bolus_offset_within_record_s", p.pre_bolus_baseline_s, "s",
               MEASURED, source=SOURCE, species=SPECIES, stage=STAGE,
               notes="bolus applied 5 s into the record")
    reg.define("post_dissection_rest_s", p.post_dissection_rest_s, "s", MEASURED,
               source=SOURCE, species=SPECIES, stage=STAGE,
               notes="about 20 min rest after dissection before recording")
    reg.define("voxel_volume_um3", kin.voxel_volume_um3, "um^3", ILLUSTRATIVE,
               notes="ASSUMED well-mixed volume for pmol -> nM; not measured")
    reg.define("pool_capacity_pmol", kin.pool_capacity_pmol, "pmol", ILLUSTRATIVE,
               notes="finite releasable pool; not measured anywhere in the source")
    reg.define("refill_tau_s", kin.refill_tau_s, "s", ILLUSTRATIVE,
               notes="pool refilling time constant; not measured")
    reg.define("stimulus_efficacy", kin.stimulus_efficacy, "pmol DA/pmol ACh",
               ILLUSTRATIVE,
               notes=("converts the measured ACh stimulus dose into a released DA "
                      "amount; a modelling choice, not a measured coupling"))
    reg.define("release_tau_s", kin.release_tau_s, "s", ILLUSTRATIVE,
               notes="extracellular appearance time constant")
    reg.define("clearance_tau_s", kin.clearance_tau_s, "s", ILLUSTRATIVE,
               notes="first-order extracellular DA removal; not measured")
    reg.define("t_half_clearance_s", kin.clearance_half_life_s, "s", DERIVED,
               derived_from=["clearance_tau_s"], notes="ln2 * clearance_tau_s")
    reg.define("release_shape_coupling", kin.release_shape_coupling, "1",
               ILLUSTRATIVE, notes="0 keeps amplitude-only pool effect")
    reg.define("noise_std_nM", kin.noise_std_nM, "nM", ILLUSTRATIVE,
               notes="observation noise on stored traces only")
    reg.validate()
    return reg


__all__ = [
    "SOURCE", "SPECIES", "STAGE", "PMOL_TO_MOL", "UM3_TO_LITRE",
    "MOL_PER_LITRE_TO_NM", "NM_PER_UM3_PER_PMOL", "NM_TO_UM",
    "ProtocolSpec", "DAKinetics", "DAProtocolResult",
    "amount_pmol_to_concentration_nM", "concentration_nM_to_amount_pmol",
    "curve_features", "measure_trace_features", "release_sequence",
    "simulate", "pattern_verdict", "sweep_pool_refill_efficacy",
    "sweep_clearance", "sweep_shape_coupling", "build_registry",
    "DEFAULT_DECLINE_THRESHOLD", "DEFAULT_T_HALF_TOLERANCE",
]
