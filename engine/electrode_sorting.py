"""SOURCE SEPARATION (spike sorting / demultiplexing) for the FIXED 20 um-pitch
electrode array of this project.

WHY THIS MODULE EXISTS
----------------------
The user fixed the array geometry at **shaft diameter 7 um, channel pitch
20 um** (channel count free) and DECLARED ``capture_radius_um = 50``.  Because
``pitch < 2 x capture_radius``, the capture spheres of neighbouring contacts
OVERLAP, and the access map reports the SAME soma point inside up to **21**
channel spheres at once (3186 of 3250 captured neurons are multi-channel;
``engine/embodied/access_map.py``, rebuilt at this geometry).  The geometric
assignment "channel <-> neuron" is therefore AMBIGUOUS by construction, and
some form of source separation stops being optional.  This module implements
one, measures whether it works, and states the limit at which it stops working.

WHAT THIS MODULE IS
-------------------
1. :class:`SignalModel` -- synthesises per-channel waveforms for a STATED
   ground-truth spike set over the REAL captured BANC somata, through the
   EXISTING pipeline and nothing else:

       volume conduction   engine.electrode.transfer_mV_per_nA  (REUSED; there
                           is NO second field solver anywhere in this module)
       area averaging      engine.electrode_frontend.contact_area_average's own
                           quadrature rule (ContactQuadrature), applied to a
                           point current source over the 7 um active patch
       interface           engine.electrode_frontend.InterfaceImpedance.transfer
       crosstalk           engine.electrode_frontend.CrosstalkMatrix
                           .added_crosstalk_matrix   (a channel-to-channel LINEAR
                           map, so it is applied in the TIME domain)
       noise               the Johnson-Nyquist floor of the same interface
                           (engine.electrode_frontend.thermal_noise_variance)

2. :func:`separate` -- a principled demultiplexing method plus two trivial
   baselines.  PRIMARY METHOD: **sparse least-squares / greedy matching pursuit
   (orthogonal matching pursuit) against the per-neuron channel templates
   derived from the real geometry**.  This is the natural inverse problem here:
   the forward map from "which neuron spiked" to "what the array reads" is
   LINEAR and KNOWN, so demultiplexing is a sparse linear inverse problem, not a
   blind-source-separation problem.  Baselines: (B1) nearest-channel
   thresholding with winner-take-all assignment, (B2) global thresholding with
   no assignment at all.

3. :func:`evaluate` -- detection and assignment metrics (detection rate,
   assignment accuracy, precision, false positives, and how many of the
   SHARED/multi-channel neurons are resolvable at all).

4. :func:`separability_limit` -- the noise level at which separation of the
   shared neurons stops working, MEASURED by this module's own sweep (adaptive
   bisection on a measured resolvable fraction), with the margin to the noise
   floor of this recording chain.

   NOISE-FLOOR STATUS -- stated because an earlier revision of this docstring
   got it wrong.  The number **24.02 uV RMS** carried by
   ``outputs/embodied_body/electrode_frontend.json`` is a **LEGACY DIAGNOSTIC,
   NOT a calibrated measured in-band floor**: it is a rectangular-ideal-band
   integral of ``4 k T Re[Z_e(f)]``, and the existing front end's own docstring
   says of that routine "no out-of-band tail or analog filter is included in
   this legacy diagnostic".  It also **does not reproduce** from the parameters
   its own JSON records -- integrating this interface over the stated
   1e-6..1e4 Hz band gives 5.6261 uV RMS, not 24.018 uV -- so it is treated as
   an unresolved inconsistency in an existing artefact, never as a measurement.
   This module therefore computes the noise **output-referred, on the
   recording's own rfft grid through the SAME interface transfer that shapes the
   spike** (:func:`output_referred_noise_uV`) and reports the legacy number
   separately.  A real probe/amplifier/tissue in-band noise budget is a
   separate, still-PENDING piece of work: NOTHING here may be quoted as a
   measured device noise floor.

5. :func:`SortingLedger` -- a provenance ledger that REFUSES
   ``MEASURED_CITED`` without a source, ``ASSUMED`` without a sweep, and
   ``MEASURED_LOCAL`` without a named artefact; the CC BY 4.0 attribution of
   the BANC soma data is carved into the module constants and into every output.

WHAT THIS MODULE IS NOT -- READ BEFORE QUOTING ANY NUMBER OUT OF IT
-------------------------------------------------------------------
* **Hand-built research prototype.**  It is NOT a validated device model and
  NOT a validated spike sorter.  It has never been checked against a real
  recording, a real probe or a real sorted dataset, and no number here is a
  prediction about any real experiment.
* **The somas are SINGLE VOXEL POINTS with no morphology.**  Consequences that
  this module CANNOT fix and that bound every result:
    - every neuron is a POINT current source, so there is no soma radius, no
      neurite, no axon, no dendrite, no synapse location, no membrane area;
    - a neurite passing a contact is INVISIBLE here, and a real extracellular
      spike's amplitude at a contact is set by the cell's geometry near that
      contact, which this model does not have;
    - **there is no spike waveform shape in the data.**  The template used is a
      DECLARED waveform (:data:`SPIKE_WAVEFORM_NOTE`), not a measured or
      simulated action potential, and it is FIXED across neurons.
* **The waveform is known to the estimator** (the primary method is given the
  same forward matrix that generated the data).  That makes it an ORACLE
  template-matching sorter: within THIS model it is the best case for
  demultiplexing, because no estimator can do better than being handed the
  generating forward operator.
  WHAT THAT DOES AND DOES NOT BOUND -- this is easy to overclaim, so it is
  stated precisely.  A failure here is a failure of **this model's** oracle: it
  shows that the information needed to attribute a spike to one of several somata
  a few micrometres apart is not present in this forward matrix at this noise
  level.  It is **NOT** a bound on a real sorter, because a real sorter is not
  limited by this model's input.  In particular this module has NO morphology
  (so no soma radius, no dendrite, no per-neuron waveform shape) and NO measured
  noise budget, and real separation leans on exactly those: a real spike carries
  a cell-specific waveform, and a real contact sits a few micrometres from a
  membrane whose geometry sets the amplitude.  So the honest reading is:
  **the conclusion is model-specific and must NOT be restated as "spike sorting
  of this array is impossible"** -- it is "separation is impossible for a
  POINT-SOURCE neuron model with one DECLARED waveform per neuron, at the noise
  level this chain computes".
  A separate arm of the runner measures what happens when the template is
  estimated from a short calibration recording instead of being handed over.
* **Neurons are current sources of a DECLARED magnitude, not spiking cells.**
  Nothing in this project models a membrane, a threshold or an action
  potential; the spike is an injected current waveform with a declared shape
  and a declared amplitude.  Every amplitude statement below is therefore
  amplitude-DECLARATION-dependent, and the runner reports the amplitude
  scaling explicitly.
* **No electrode-contact model beyond the existing front end**: no surface
  roughness/fractal area, no DC drift, no 1/f noise, no ADC, no quantisation,
  no common-mode rejection, no reference-electrode model.
* **The two crosstalk mechanisms are NOT summed.**  The medium-inherent
  coupling is ALREADY inside the volume-conduction term (adding it again would
  double-count it); only the ADDED electronic leakage is applied as a matrix.
* **Nothing about the fly.**  No behaviour, no perception, no attention, no
  recognition, no experience, and NO consciousness, identity or immortality
  claim of any kind is made or implied.

UNITS
-----
Coordinates in micrometres, currents in nanoamperes, potentials in
MILLIVOLTS inside the model (the solver's own unit) and reported in
MICROVOLTS where amplitudes are quoted, noise in uV RMS.  The conversion
between the BANC voxel frame and micrometres is the VERIFIED anisotropic
4 x 4 x 45 nm of :func:`engine.embodied.access_map.banc_voxel_resolution`.

ATTRIBUTION (REQUIRED -- CC BY 4.0, A LICENCE CONDITION)
--------------------------------------------------------
The soma coordinates consumed here come from

    "Distributed control circuits across a brain-and-cord connectome",
    Harvard Dataverse, doi:10.7910/DVN/7WTH1N, file ``somas_v1.parquet``
    (file id 13916460), Licensed CC BY 4.0
    (http://creativecommons.org/licenses/by/4.0).

Redistribution of this module's outputs, or of any derived coordinate, template
or spike train, MUST carry that attribution.
:data:`BANC_SOMA_ATTRIBUTION` is the record to reproduce; it is written into
every output dict produced here.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# The EXISTING pipeline is reused, never reimplemented.  Nothing below writes a
# field solver: every potential is a call into transfer_mV_per_nA.
from .electrode import Contact, transfer_mV_per_nA
from .electrode_frontend import (
    BOLTZMANN_J_PER_K, ContactGeometry, ContactQuadrature, CrosstalkMatrix,
    InterfaceImpedance, contact_points_um, thermal_noise_variance,
)
from .embodied.access_map import (
    BANC_SOMA_ATTRIBUTION, BANC_SOMA_DOI, BANC_SOMA_DATASET_TITLE,
    BANC_SOMA_FILE, BANC_SOMA_FILE_ID, BANC_SOMA_LICENCE, PROVENANCE,
)

__all__ = [
    "SPIKE_WAVEFORM_NOTE", "DISCLAIMER", "SORTING_ATTRIBUTION",
    "SortingLedger", "SignalModel", "SpikeTruth", "SeparationResult",
    "separate", "evaluate", "separability_limit", "sweep_noise",
    "matched_filter_snr", "output_referred_noise_uV", "make_trial_spikes", "run_noise_level",
    "resolvability_geometry", "per_neuron_gain_matrix", "gaussian_spike_current_nA",
    "noise_floor_uV_rms", "DEFAULT_NOISE_FLOOR_UV_RMS",
]

# ---------------------------------------------------------------------------
# statements that must travel with every output
# ---------------------------------------------------------------------------
#: The MEASURED noise floor of the existing front end, as reported by
#: ``outputs/embodied_body/electrode_frontend.json`` (band 1e-6 .. 1e4 Hz,
#: 7 um contact, 10 MOhm amplifier input, 300 K).  Carried as a constant here so
#: that the module cannot silently use a different floor than the front end did;
#: the runner re-checks it by recomputing it from the interface.
DEFAULT_NOISE_FLOOR_UV_RMS = 24.01816009741456

DISCLAIMER = (
    "HAND-BUILT RESEARCH PROTOTYPE, NOT A VALIDATED DEVICE MODEL AND NOT A "
    "VALIDATED SPIKE SORTER.  Somata are SINGLE VOXEL POINTS: every neuron is a "
    "point current source with NO morphology and NO spike waveform shape in the "
    "data, so the template used here is DECLARED, not measured or simulated.  "
    "The primary sorter is given the same forward matrix that generated the "
    "data, i.e. it is an ORACLE template matcher and its numbers are a BEST "
    "CASE for this model, not a real sorter's performance.  Neurons are DECLARED "
    "current sources, not spiking cells: no membrane, no threshold, no action "
    "potential and no refractory mechanism is modelled, so real spike trains "
    "would overlap differently.  Crosstalk here is a lumped geometric network, "
    "not an FEM/EM solve, and the two crosstalk mechanisms are NOT summed "
    "(the medium-inherent part is already inside volume conduction).  No claim "
    "is made that any neuron is recorded from a real animal, and no claim is "
    "made about behaviour, perception, attention, recognition, experience, "
    "consciousness, identity or immortality.")

#: Reproduce this block in any redistribution.  CC BY 4.0 makes attribution a
#: licence CONDITION, not a courtesy.
SORTING_ATTRIBUTION = dict(BANC_SOMA_ATTRIBUTION)
SORTING_ATTRIBUTION["consumed_by"] = (
    "engine/electrode_sorting.py -- per-neuron channel templates and synthetic "
    "spike waveforms built on the real BANC soma point cloud")

SPIKE_WAVEFORM_NOTE = (
    "DECLARED, NOT MEASURED.  The soma table carries ONE voxel per neuron: no "
    "morphology, no membrane, no action potential, so there is no waveform "
    "shape to be had.  The extracellular spike model here is therefore a "
    "single-lobe Gaussian current pulse of DECLARED width (sigma_ms) and "
    "DECLARED peak amplitude (peak_current_nA), identical for every neuron, "
    "handed to the EXISTING front end so that the front end's own filtering "
    "shapes it.  Consequences, stated plainly: (a) every template has the SAME "
    "shape, so the classifier separates neurons by POSITION and AMPLITUDE only, "
    "not by waveform; (b) a real spike's high-frequency content (which is what "
    "a real sorter leans on) is absent; (c) the declared amplitude sets every "
    "SNR number, so the amplitude is reported and scaled explicitly rather than "
    "being buried.")


# ---------------------------------------------------------------------------
# provenance vocabulary + ledger
# ---------------------------------------------------------------------------
class PROV:
    """The same four classes as the rest of the stack.  There is no fifth."""

    MEASURED_CITED = PROVENANCE.MEASURED_CITED
    MEASURED_LOCAL = PROVENANCE.MEASURED_LOCAL
    ASSUMED = PROVENANCE.ASSUMED
    ENGINEERING_DEFAULT = PROVENANCE.ENGINEERING_DEFAULT
    ALL = PROVENANCE.ALL


@dataclass(frozen=True)
class ProvenanceRecord:
    key: str
    value: object
    unit: str
    provenance: str
    source: Optional[str] = None
    note: str = ""
    sweep: Optional[tuple] = None

    def as_dict(self):
        return {"key": self.key, "value": self.value, "unit": self.unit,
                "provenance": self.provenance, "source": self.source,
                "note": self.note,
                "sweep": (None if self.sweep is None else list(self.sweep))}


class SortingLedger:
    """Provenance ledger that ENFORCES the honesty rules instead of recording them.

    ``record`` RAISES on:

    1. ``MEASURED_CITED`` without a non-blank ``source`` -- a cited measurement
       with no citation is fabricated provenance;
    2. ``ASSUMED`` without a ``sweep`` range (or the word "sweep"/"range" in the
       note) -- an assumption you refuse to vary is an unexamined constant;
    3. ``MEASURED_LOCAL`` without a named artefact -- "local" and "reproducible"
       are claims that need something to point at;
    4. an unknown provenance class, or a duplicate key.

    :meth:`grid_row` is the deliberate escape for one row of a computed sweep
    GRID (a repeated key is what a grid IS); it can only ever produce
    ``MEASURED_LOCAL``, so a grid row can never masquerade as a citation or as
    an assumption.
    """

    def __init__(self):
        self._recs = []
        self._keys = {}

    def record(self, key, value, unit, provenance, *, source=None, note="", sweep=None):
        if provenance not in PROV.ALL:
            raise ValueError("provenance must be one of %r; got %r" % (PROV.ALL, provenance))
        if provenance == PROV.MEASURED_CITED and not (source and str(source).strip()):
            raise ValueError(
                "refusing MEASURED_CITED without a source: %r. A cited measurement "
                "with no citation is fabricated provenance." % (key,))
        if provenance == PROV.MEASURED_LOCAL and not (source and str(source).strip()):
            raise ValueError(
                "refusing MEASURED_LOCAL without a named artefact: %r." % (key,))
        if provenance == PROV.ASSUMED and sweep is None and \
                not any(w in str(note).lower() for w in ("sweep", "range")):
            raise ValueError(
                "refusing ASSUMED without a declared sweep/range: %r. An assumption "
                "without a swept range is an unexamined constant." % (key,))
        if key in self._keys:
            raise ValueError("duplicate ledger key %r" % (key,))
        r = ProvenanceRecord(key, value, unit, provenance, source, note,
                             None if sweep is None else tuple(sweep))
        self._keys[key] = r
        self._recs.append(r)
        return r

    def grid_row(self, key, value, unit, *, source, note=""):
        """One row of a computed sweep grid: always MEASURED_LOCAL, key may repeat."""
        if not (source and str(source).strip()):
            raise ValueError("sweep rows are MEASURED_LOCAL and need a named source")
        r = ProvenanceRecord(key, value, unit, PROV.MEASURED_LOCAL, source, note, None)
        self._recs.append(r)
        return r

    def get(self, key):
        return self._keys[key]

    def __contains__(self, key):
        return key in self._keys

    def as_list(self):
        return [r.as_dict() for r in self._recs]

    def as_dict(self):
        return {r.key: r.as_dict() for r in self._recs}

    def counts(self):
        out = {p: 0 for p in PROV.ALL}
        for r in self._recs:
            out[r.provenance] += 1
        return out

    def unresolved(self):
        """Entries a reader must treat as NOT established from any source."""
        return [r.as_dict() for r in self._recs
                if r.provenance in (PROV.ASSUMED, PROV.ENGINEERING_DEFAULT)]


# ---------------------------------------------------------------------------
# geometry -> forward gain (the ONLY place potentials are produced)
# ---------------------------------------------------------------------------
def _as_contacts(points_um):
    return [Contact(tuple(float(v) for v in p), 1.0) for p in np.asarray(points_um, float)]


def per_neuron_gain_matrix(access_map, *, neuron_current_nA=0.01,
                           conductivity_S_m=0.3,
                           contact: ContactGeometry = None,
                           quadrature: ContactQuadrature = None,
                           normal=(0.0, 0.0, 1.0),
                           amplitude_floor_nA_per_channel=None,
                           neuron_rows=None,
                           progress=None):
    """Per-neuron, per-channel VOLTAGE gain of the array, in mV.

    Returns ``(G, info)`` with ``G`` of shape ``(n_channels, n_neurons)`` in mV
    per unit current (i.e. already multiplied by ``neuron_current_nA``), where
    ``n_neurons`` is ``len(neuron_rows)`` -- or every captured soma of the map
    when ``neuron_rows`` is None.

    PHYSICS AND CONVENTIONS, all inherited and all stated:

    * The potential is :func:`engine.electrode.transfer_mV_per_nA` -- the
      EXISTING solver.  NO second field solver is written here.
    * The contact is the FINITE 7 um active patch of
      :class:`engine.electrode_frontend.ContactGeometry`, and the recorded value
      is the patch AREA AVERAGE of that field, evaluated with the module's own
      :class:`ContactQuadrature` (the same tensor Gauss-Legendre x uniform-polar
      rule the front end uses).  This differs from a point sample at the contact
      centre by up to ~2x for a source a few micrometres away, which is exactly
      the regime this geometry puts most captured neurons in.
    * Only the CAPTURED (channel, neuron) pairs of the access map are given
      entries; everything else is exactly zero.  That is the map's own
      geometric predicate and it is NOT loosened or tightened here.
    * **A degenerate corner is REPORTED, NOT hidden**: the solver's point sphere
      has radius 1 um, so a source within 1 um of a contact returns the sphere's
      CENTRE value ``I/(8 pi sigma a)`` for that region -- a bounded
      regularisation, not a divergence.  The number of (channel, neuron) pairs
      whose quadrature points fall inside that sphere is counted and returned in
      ``info``; those entries are the model's approximation floor here.

    ``amplitude_floor_nA_per_channel``, when given, zeroes entries below a
    per-channel fraction of that channel's largest entry.  It defaults to None
    (keep everything), and any value actually used is written into ``info``.
    """
    if contact is None:
        contact = ContactGeometry.from_diameter(access_map.spec.diameter_um)
    q = quadrature or ContactQuadrature(4, 16)
    n_norm = np.asarray(normal, float)
    n_norm = n_norm / np.linalg.norm(n_norm)

    centres = np.asarray(access_map.channel_centres_um, float)
    n_ch = centres.shape[0]
    soma_um = access_map.resolution.voxels_to_um(
        np.asarray(access_map.soma_positions_vox, float))

    if neuron_rows is None:
        neuron_rows = np.flatnonzero(np.asarray(access_map.neuron_n_channels) > 0)
    neuron_rows = np.asarray(neuron_rows, dtype=np.int64)
    pos_um = soma_um[neuron_rows]
    row_of = {int(r): i for i, r in enumerate(neuron_rows.tolist())}
    n_neu = neuron_rows.size

    G = np.zeros((n_ch, n_neu), dtype=float)
    n_pairs = 0
    n_inside_sphere = 0
    # the solver's regularisation sphere: values are the constant I/(8 pi sigma a)
    sphere_radius_um = 1.0
    sphere_value_mV_per_nA = 1.0 / (8.0 * math.pi * conductivity_S_m * sphere_radius_um)

    for k in range(n_ch):
        rows = [int(r) for r in access_map.channel_neurons[k] if int(r) in row_of]
        if not rows:
            continue
        cols = np.array([row_of[r] for r in rows], dtype=np.int64)
        P = soma_um[np.array(rows, dtype=np.int64)]
        src = _as_contacts(P)
        xyz, w = contact_points_um(contact, source_um=tuple(centres[k]),
                                   normal=tuple(n_norm), quadrature=q)
        F = transfer_mV_per_nA(xyz, src, conductivity_S_m)      # (n_q, n_rows)
        mean = (w @ F) / w.sum()
        G[k, cols] = mean * float(neuron_current_nA)
        # count the pairs that touch the solver's own regularisation sphere
        dmin = np.sqrt(((P[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)).min(axis=1)
        n_inside_sphere += int((dmin <= sphere_radius_um).sum())
        n_pairs += len(rows)
        if progress is not None and (k % 32 == 0):
            progress(k, n_ch)

    n_zeroed = 0
    if amplitude_floor_nA_per_channel is not None:
        f = float(amplitude_floor_nA_per_channel)
        for k in range(n_ch):
            row = G[k]
            m = row.max() if row.size else 0.0
            if m > 0:
                cut = f * m
                z = row < cut
                n_zeroed += int(z.sum())
                row[z] = 0.0

    info = {
        "n_channels": int(n_ch),
        "n_neurons": int(n_neu),
        "n_nonzero_entries": int((G != 0).sum()),
        "n_channel_neuron_pairs_considered": int(n_pairs),
        "n_pairs_inside_solver_regularisation_sphere": int(n_inside_sphere),
        "solver_regularisation_sphere_radius_um": sphere_radius_um,
        "value_on_that_sphere_mV_per_nA": float(sphere_value_mV_per_nA),
        "gaussian_units": "mV",
        "gain_min_mV": float(G.min()), "gain_max_mV": float(G.max()),
        "gain_abs_max_mV": float(np.abs(G).max()),
        "n_entries_zeroed_by_amplitude_floor": int(n_zeroed),
        "amplitude_floor_fraction_of_channel_max": amplitude_floor_nA_per_channel,
        "per_neuron_max_gain_mV_median": float(np.median(np.abs(G).max(axis=0))),
        "quadrature": {"n_radial": q.n_radial, "n_azimuth": q.n_azimuth,
                       "n_points_per_contact": q.n_points,
                       "rule": "tensor Gauss-Legendre (radial) x uniform polar "
                               "(azimuthal), as engine.electrode_frontend"},
        "contact": contact.as_dict(),
        "note": ("area-averaged potential of a POINT current source at each captured "
                 "REAL soma position, from engine.electrode.transfer_mV_per_nA. "
                 "Sources closer than the solver's 1 um sphere return that sphere's "
                 "centre value; those pairs are counted above."),
        "no_second_field_solver": True,
    }
    return G, info


def noise_floor_uV_rms(interface: InterfaceImpedance, *, f_lo_Hz=1e-6,
                       f_hi_Hz=10e3, include_amplifier=False):
    """Johnson-Nyquist noise floor of the SAME interface, in uV RMS.

    Delegates to :func:`engine.electrode_frontend.thermal_noise_variance`; the
    band is a caller input and is returned with the number so the two can never
    drift apart.
    """
    d = thermal_noise_variance(interface, f_lo_Hz=f_lo_Hz, f_hi_Hz=f_hi_Hz,
                              include_amplifier=include_amplifier)
    return float(d["rms_uV"]), d


def output_referred_noise_uV(interface: InterfaceImpedance, *, n_steps, dt_ms,
                              f_lo_Hz=1e-6, n_grid=20000):
    """Noise RMS AT THE RECORDING'S OUTPUT, on the trace's OWN frequency grid.

    WHY THIS EXISTS AND WHY IT IS NOT THE FRONT END'S NUMBER
    -------------------------------------------------------
    ``engine.electrode_frontend.thermal_noise_variance`` integrates the electrode's
    open-circuit spectral density over a stated band.  A trace that is then
    SAMPLED and already filtered by the same interface does not carry that number:
    the interface transfer multiplies the noise too, and the discrete trace has a
    finite Nyquist.  On this geometry the difference is a factor of ~4.3 (the
    interface is a high pass with |H| ~ 0.05 at a few hundred Hz and ~0.98 near
    100 kHz), and it is the DIFFERENCE THAT DECIDES THIS QUESTION, so it is
    computed rather than borrowed.

    The integration matches the simulation exactly: the same rfft grid the spike
    waveform is filtered on, one-sided density ``4 k T Re Z_e(f) |H(f)|^2``,
    integrated from the first non-zero bin to Nyquist.  The DC bin is excluded
    because an rfft of a finite trace does not resolve it, which is stated in the
    returned record.

    Returns ``(rms_uV, detail)``.
    """
    n = int(n_steps)
    if n < 2:
        raise ValueError("n_steps must be >= 2")
    f = np.fft.rfftfreq(n, d=float(dt_ms) * 1e-3)
    pos = f > 0
    fg = f[pos]
    if fg.size < 2:
        raise ValueError("trace too short to resolve a noise band")
    H = interface.transfer(fg)
    s = (interface.noise_spectral_density_V_per_rtHz(fg) * np.abs(H)) ** 2
    var = float(np.trapezoid(s, fg) if hasattr(np, "trapezoid") else np.trapz(s, fg))
    rms_uV = math.sqrt(max(var, 0.0)) * 1e6
    f_lo_eff = float(fg[0])
    f_hi_eff = float(fg[-1])
    detail = {
        "rms_uV": rms_uV,
        "variance_V2": var,
        "f_lo_Hz_effective": f_lo_eff,
        "f_hi_Hz_effective": f_hi_eff,
        "n_grid": int(fg.size),
        "requested_f_lo_Hz": float(f_lo_Hz),
        "n_steps": int(n), "dt_ms": float(dt_ms),
        "nyquist_Hz": 1.0 / (2.0 * float(dt_ms) * 1e-3),
        "s_at_band_top_V2_per_Hz": float(s[-1]),
        "sqrt_s_at_band_top_nV_per_rtHz": math.sqrt(float(s[-1])) * 1e9,
        "include_amplifier": False,
        "law": "S_V(f) = 4 k T Re[Z_e(f)] * |H(f)|^2   (Johnson-Nyquist, output-referred)",
        "note": ("integrates on the RECORDING'S OWN rfft grid so the noise power and "
                 "the signal pass through exactly the same filter; the DC bin is "
                 "excluded because a finite trace does not resolve it. This is the "
                 "number the synthesised trace actually carries."),
    }
    return rms_uV, detail


def gaussian_spike_current_nA(t_ms, *, sigma_ms=1.0, peak_nA=1.0, centre_ms=0.0):
    """A DECLARED single-lobe Gaussian current pulse, in nA.

    See :data:`SPIKE_WAVEFORM_NOTE`: there is no waveform in the data, so this
    shape is a declaration.  It is normalised to unit PEAK, not unit area, so
    ``peak_nA`` is directly the peak injected current (and therefore the
    amplitude knob every SNR number scales with).
    """
    t = np.asarray(t_ms, float)
    if sigma_ms <= 0 or not math.isfinite(sigma_ms):
        raise ValueError("sigma_ms must be positive and finite")
    return float(peak_nA) * np.exp(-0.5 * ((t - centre_ms) / float(sigma_ms)) ** 2)


# ---------------------------------------------------------------------------
# ground truth and the synthesised recording
# ---------------------------------------------------------------------------
@dataclass
class SpikeTruth:
    """The DECLARED ground-truth spike set the recorded waveforms were built from.

    ``spikes`` is a list of ``(neuron_index, peak_time_ms)`` with
    ``neuron_index`` an index into the neuron axis of the model's gain matrix.
    ``neuron_rows`` are the soma-table rows behind that axis, so the ground
    truth can always be traced back to REAL measured neurons.

    HONEST LIMIT: the spike TIMES are drawn from a declared generator (uniform
    over the recording window, one spike per active neuron per trial).  They are
    an experiment design, NOT a measured firing rate for any neuron.  Refractory
    periods are not modelled because no spike generator of this project is
    reused here; the runner states the declared rate it implies.
    """

    neuron_rows: np.ndarray
    soma_root_ids: np.ndarray
    spikes: list = field(default_factory=list)
    n_channels_capturing: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    is_shared: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    params: dict = field(default_factory=dict)
    seed: int = 0

    def __len__(self):
        return len(self.spikes)

    def as_dict(self, *, max_list=0):
        d = {"n_spikes": len(self.spikes),
             "n_neurons_in_model": int(self.neuron_rows.size),
             "n_shared_neurons_in_model": int(np.sum(self.is_shared)),
             "params": dict(self.params), "seed": int(self.seed),
             "note": ("spike times are DECLARED, drawn from a generator -- no "
                      "measured firing rate of any neuron, and no refractory "
                      "mechanism is modelled")}
        if max_list:
            d["spikes_head"] = [[int(n), float(t)] for n, t in self.spikes[:max_list]]
        return d


class SignalModel:
    """Synthesises per-channel waveforms for a stated ground-truth spike set.

    The model is LINEAR and TIME-INVARIANT in its forward path, and every stage
    is the EXISTING one:

        I(t)  --(DECLARED Gaussian pulse)-->  current per neuron
              --(G, from the access map's real geometry + transfer_mV_per_nA)-->
                 unfiltered per-channel voltage
              --(InterfaceImpedance.transfer, evaluated on the recording's own
                 frequency grid)-->  interface-filtered voltage
              --(CrosstalkMatrix.added_crosstalk_matrix @ .)-->
                 crosstalk-mixed voltage
              --(+ Gaussian white noise at a DECLARED sigma)-->  recorded trace

    The noise sigma may be given DIRECTLY, or taken from the interface's own
    Johnson-Nyquist floor via ``noise_mode="interface_thermal"``; either way the
    number used and the band it belongs to are written into every result, so a
    swept sigma can never be mistaken for a measured noise floor, which this
    stack does NOT have.

    Parameters
    ----------
    gain_mV : (n_channels, n_neurons) mV
        per-neuron, per-channel voltage gain (already scaled by the source
        current amplitude).  From :func:`per_neuron_gain_matrix`.
    dt_ms, n_steps
        sampling interval and number of samples.  The interface filter is
        applied on this trace's own FFT grid, so the corner behaviour depends on
        the trace length; that is reported, not hidden.
    spike_sigma_ms, spike_peak_nA
        DECLARED waveform width and amplitude.
    """

    def __init__(self, gain_mV, access_map, *, dt_ms=1.0, n_steps=200,
                 contact: ContactGeometry = None,
                 interface: InterfaceImpedance = None,
                 conductivity_S_m=0.3,
                 spike_sigma_ms=1.0, spike_peak_nA=1.0,
                 waveform_half_width_ms=2.0, waveform_tail_tol=0.10,
                 crosstalk_f_Hz=1000.0, apply_crosstalk=True,
                 apply_interface=True,
                 noise_sd_mV: Optional[float] = None,
                 noise_mode="declared_sigma",
                 noise_band_Hz=(1e-6, 10e3), noise_include_amplifier=False,
                 seed=0):
        G = np.asarray(gain_mV, float)
        if G.ndim != 2:
            raise ValueError("gain_mV must be (n_channels, n_neurons)")
        if int(access_map.n_channels) != G.shape[0]:
            raise ValueError("gain matrix has %d channel rows but the access map has %d "
                             "channels" % (G.shape[0], access_map.n_channels))
        if noise_mode not in ("declared_sigma", "interface_thermal"):
            raise ValueError("noise_mode must be 'declared_sigma' or 'interface_thermal'")
        self.access_map = access_map
        self.G = G
        self.dt_ms = float(dt_ms)
        self.n_steps = int(n_steps)
        self.spike_sigma_ms = float(spike_sigma_ms)
        self.spike_peak_nA = float(spike_peak_nA)
        self.waveform_half_width_ms = float(waveform_half_width_ms)
        self.waveform_tail_tol = float(waveform_tail_tol)
        self.conductivity_S_m = float(conductivity_S_m)
        self.contact = contact or ContactGeometry.from_diameter(access_map.spec.diameter_um)
        self.interface = interface
        self.seed = int(seed)
        self.noise_mode = noise_mode
        self.noise_band_Hz = tuple(float(v) for v in noise_band_Hz)
        self.apply_crosstalk = bool(apply_crosstalk)
        self.apply_interface = bool(apply_interface)

        centres = np.asarray(access_map.channel_centres_um, float)
        self.centres_um = centres
        self.crosstalk = CrosstalkMatrix(
            centres_um=centres, pitch_um=float(access_map.spec.pitch_um),
            sigma_S_m=self.conductivity_S_m,
            amplifier_input_ohm=(self.interface.amplifier_input_ohm
                                 if self.interface is not None else math.inf),
            interface=self.interface)
        self.C_add = (self.crosstalk.added_crosstalk_matrix(
            float(crosstalk_f_Hz), contact=self.contact)
            if apply_crosstalk else np.eye(centres.shape[0]))

        # -- noise ---------------------------------------------------------
        floor_uV, floor_detail = (None, None)
        if interface is not None:
            floor_uV, floor_detail = noise_floor_uV_rms(
                interface, f_lo_Hz=self.noise_band_Hz[0], f_hi_Hz=self.noise_band_Hz[1],
                include_amplifier=noise_include_amplifier)
        if noise_mode == "interface_thermal":
            if floor_uV is None:
                raise ValueError("noise_mode='interface_thermal' needs an interface")
            noise_sd_mV = floor_uV * 1e-3
        if noise_sd_mV is None:
            raise ValueError("noise_sd_mV is required for noise_mode='declared_sigma'")
        self.noise_sd_mV = float(noise_sd_mV)
        self.noise_floor_detail = floor_detail
        self.interface_floor_uV_rms = floor_uV

        # -- spike waveform, filtered by the SAME front end -----------------
        #
        # TWO DECLARED CONVENTIONS live here, and both matter at the noise levels
        # this problem sits at, so neither is left implicit.
        #
        # (1) TRUNCATION.  The interface is a high pass, so a Gaussian current
        #     pulse leaves a DERIVATIVE-like response whose tail decays slowly
        #     (time constant R_spread*C_dl ~ 0.13 ms).  The waveform is therefore
        #     constructed over +/- waveform_half_width_ms and then TRUNCATED, with
        #     the truncation CHECKED: the norm of the truncated waveform is
        #     compared with the analytic whole-window value and with the
        #     small-signal derivative limit 2*pi*f_c*|w_current|.  Leaving a slow
        #     tail in would let a distant spike leak into every later sample, and
        #     would make the template's norm depend on the trace length.
        # (2) THE WAVEFORM LAGS ARE NEGATIVE.  With a high-pass front end the
        #     response PEAK sits at or before the current pulse's centre, so the
        #     waveform array is indexed with NEGATIVE lags: waveform[k] multiplies
        #     sample (t_index - k).  The spike TIME reported for a detection is
        #     therefore the arrival of the sharp leading feature, which can precede
        #     the injected peak by a fraction of a millisecond; the matching
        #     tolerance covers it and the convention is stated in the output.
        self.t_ms = np.arange(self.n_steps, dtype=float) * self.dt_ms
        self._H = self._interface_transfer_on_grid()
        w_full = self._apply_interface_to_waveform(gaussian_spike_current_nA(
            self.t_ms - 0.5 * self.n_steps * self.dt_ms,
            sigma_ms=self.spike_sigma_ms, peak_nA=self.spike_peak_nA,
            centre_ms=0.0))
        centre = int(np.argmax(np.abs(w_full)))
        half = int(round(self.waveform_half_width_ms / self.dt_ms))
        lo, hi = max(0, centre - half), min(self.n_steps, centre + half + 1)
        support = min(centre - lo, hi - 1 - centre)
        self.waveform_nA = w_full[centre - support: centre + support + 1][::-1].copy()
        self._w_unfiltered = np.array([float(self.spike_peak_nA)])
        self._w_full_norm = float(np.linalg.norm(w_full))
        self._w_asymptotic_norm = float(
            2.0 * math.pi * self.interface.f_corner_Hz * self.spike_peak_nA
            * self.spike_sigma_ms) if self.interface is not None else None
        self.waveform_truncation = {
            "half_width_ms_requested": self.waveform_half_width_ms,
            "support_samples_each_side": int(support),
            "n_waveform_samples": int(self.waveform_nA.size),
            "lag_convention": ("waveform[k] multiplies sample (i - k): NEGATIVE "
                              "lags, because a high-pass front end puts the response "
                              "peak at or before the current pulse centre"),
            "norm_truncated_nA_samples": float(np.linalg.norm(self.waveform_nA)),
            "norm_whole_window_nA_samples": self._w_full_norm,
            "norm_asymptotic_derivative_limit": self._w_asymptotic_norm,
            "relative_error_vs_whole_window": (
                abs(np.linalg.norm(self.waveform_nA) - self._w_full_norm)
                / max(self._w_full_norm, 1e-300)),
            "why_this_support": (
                "the high pass leaves an exponential-like tail with time constant "
                "~1/(2 pi f_corner) = %.4g ms; the support is the DECLARED "
                "physiological width (2 ms each side), and the error it costs is the "
                "relative_error_vs_whole_window above, which is reported rather than "
                "hidden.  Template norms -- and therefore every SNR -- are "
                "L2-normalised, so this truncation does not change the amplitude of a "
                "spike, only how much of its tail is modelled."
                % (1e3 / (2.0 * math.pi * self.interface.f_corner_Hz)
                   if self.interface is not None else float("nan"))),
        }
        # a "template" is one neuron's channel-space signature of ONE spike:
        # G[:, j] convolved with the filtered waveform.  Crosstalk pre-multiplied.
        self.templates = self.C_add @ G                      # (n_ch, n_neu)
        self._wnorm = float(np.linalg.norm(self.waveform_nA))
        self._tnorm = np.linalg.norm(self.templates, axis=0) * self._wnorm
        self._tnorm_safe = np.maximum(self._tnorm, 1e-300)

    # -- the interface stage ------------------------------------------------
    def _interface_transfer_on_grid(self):
        if not self.apply_interface or self.interface is None:
            return None
        n = self.n_steps
        freqs = np.fft.rfftfreq(n, d=self.dt_ms * 1e-3)
        H = np.zeros(freqs.shape[0], dtype=complex)
        pos = freqs > 0
        H[pos] = self.interface.transfer(freqs[pos])
        # DC BIN IS ZERO.  The interface is a high pass: Z_e/(Z_e+Z_L) -> 0 as
        # f -> 0 (the double layer gives no DC path), so the recording carries no
        # DC level and the filter must not inject one.  A finite trace does not
        # resolve the f -> 0 limit anyway; setting the bin to 0 is the declared,
        # documented choice, and it is why the filtered spike waveform is exactly
        # the derivative-like response rather than that plus a step.
        return H

    def _apply_interface_to_waveform(self, current_nA):
        if self._H is None:
            return np.asarray(current_nA, float)
        X = np.fft.rfft(np.asarray(current_nA, float))
        return np.fft.irfft(X * self._H, n=self.n_steps)

    # -- synthesis ----------------------------------------------------------
    def clean_trace_mV(self, spikes):
        """Per-channel UNFILTERED-BY-NOISE trace for a spike list, plus the filtered
        spike waveform actually used (so the caller can build templates).

        Returns ``(V, info)`` with ``V`` of shape ``(n_channels, n_steps)`` mV.
        The return contact is NOT needed here: the model works in the DIFFERENCE
        of potential between a contact and the common reference, and every stage
        is linear, so a single distant return contributes the same common-mode
        term to every channel (it is exactly the term the existing
        ``apply_frontend`` keeps in a separate bucket for that reason).  The
        gain matrix is the contact-to-reference potential of a unit source.
        """
        V = np.zeros((self.G.shape[0], self.n_steps), dtype=float)
        w = self.waveform_nA
        L = w.size
        t0 = float(self.t_ms[0])
        n_out_of_window = 0
        for j, t in spikes:
            j = int(j)
            if not 0 <= j < self.G.shape[1]:
                raise IndexError("neuron index %d outside the model" % j)
            i0 = int(round((float(t) - t0) / self.dt_ms))
            if i0 < 0 or i0 >= self.n_steps:
                n_out_of_window += 1
                continue
            # NEGATIVE LAGS: sample (i0 - k) gets w[k]
            a, b = max(0, i0 - (L - 1)), min(self.n_steps, i0 + 1)
            if a >= b:
                n_out_of_window += 1
                continue
            seg = w[i0 - np.arange(a, b)]
            V[:, a:b] += self.templates[:, j][:, None] * seg[None, :]
        info = {"n_spikes_in_window": int(len(spikes) - n_out_of_window),
                "n_spikes_outside_window": int(n_out_of_window),
                "waveform_sigma_ms": self.spike_sigma_ms,
                "waveform_peak_nA": self.spike_peak_nA,
                "waveform_norm_nA_samples": self._wnorm,
                "waveform_lag_convention": self.waveform_truncation["lag_convention"],
                "noise_added_here": False}
        return V, info

    def record(self, spikes, *, noise_sd_mV=None, seed=None):
        """The RECORDED trace: clean waveform plus Gaussian white noise.

        The noise is independent per channel and per sample, which is the
        declared model of the floor: the existing front end's noise is
        Johnson-Nyquist (band-limited, effectively white over the band) rather
        than 1/f, and that is a limitation of the front end, inherited here and
        reported, not repaired.
        """
        V, info = self.clean_trace_mV(spikes)
        sd = self.noise_sd_mV if noise_sd_mV is None else float(noise_sd_mV)
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        noise = rng.normal(0.0, sd, size=V.shape)
        info.update({"noise_sd_mV": sd, "noise_sd_uV": sd * 1e3,
                     "noise_is_white": True,
                     "noise_mode": self.noise_mode,
                     "interface_floor_uV_rms": self.interface_floor_uV_rms,
                     "interface_floor_matches_declared_sigma":
                         (None if self.interface_floor_uV_rms is None
                          else bool(abs(self.interface_floor_uV_rms - sd * 1e3)
                                    < 1e-9 * max(1.0, sd * 1e3))),
                     "noise_added_here": True,
                     "seed": int(self.seed if seed is None else int(seed))})
        return V + noise, info

    # -- reporting ----------------------------------------------------------
    def waveform_report(self):
        """What the front end did to the DECLARED waveform, with numbers."""
        return {
            "waveform_note": SPIKE_WAVEFORM_NOTE,
            "sigma_ms": self.spike_sigma_ms,
            "peak_nA": self.spike_peak_nA,
            "dt_ms": self.dt_ms,
            "n_steps": self.n_steps,
            "duration_ms": self.n_steps * self.dt_ms,
            "trace_fft_first_nonzero_bin_Hz": (1.0 / (self.n_steps * self.dt_ms * 1e-3)),
            "interface_applied": bool(self.apply_interface and self.interface is not None),
            "crosstalk_applied": bool(self.apply_crosstalk),
            "declared_peak_current_nA": self.spike_peak_nA,
            "waveform_peak_after_front_end_nA": float(
                np.abs(self.waveform_nA).max()),
            "interface_gain_at_waveform_peak_ratio": float(
                np.abs(self.waveform_nA).max() / self.spike_peak_nA),
            "truncation": self.waveform_truncation,
            "template_norm_mV_per_spike_median": float(np.median(self._tnorm)),
            "per_neuron_peak_channel_mV_median": float(np.median(np.abs(self.templates).max(0))),
            "per_neuron_peak_channel_mV_max": float(np.abs(self.templates).max()),
            "crosstalk_offdiag_ring1_median": float(np.median(
                self.C_add[np.abs(self.C_add) > 0][
                    np.abs(self.C_add[np.abs(self.C_add) > 0]) < 1.0])) if np.any(
                        (np.abs(self.C_add) > 0) & (np.abs(self.C_add) < 1)) else 0.0,
            "note": ("the interface is applied on the recording's own rFFT grid, so the "
                     "lowest non-zero bin is 1/duration; the filter's DC value is not "
                     "sampled by a finite trace.  A real spike's higher-frequency "
                     "content is ABSENT from this declared waveform."),
        }


# ---------------------------------------------------------------------------
# demultiplexing
# ---------------------------------------------------------------------------
@dataclass
class SeparationResult:
    """One demultiplexing run: what the method claimed, and what it was given."""

    method: str
    detections: list = field(default_factory=list)   # (neuron_index or None, t_ms)
    amp: list = field(default_factory=list)
    z: list = field(default_factory=list)
    n_iterations: int = 0
    stopped: str = ""
    threshold_z: float = 0.0
    params: dict = field(default_factory=dict)
    assumptions: tuple = ()

    def as_dict(self, *, max_list=0):
        d = {"method": self.method, "n_detections": len(self.detections),
             "n_iterations": int(self.n_iterations), "stopped": self.stopped,
             "threshold_z": float(self.threshold_z), "params": dict(self.params),
             "assumptions": list(self.assumptions)}
        if max_list:
            d["detections_head"] = [[(None if n is None else int(n)), float(t), float(z)]
                                    for n, t, z in list(zip(
                                        [d0[0] for d0 in self.detections],
                                        [d0[1] for d0 in self.detections], self.z))[:max_list]]
        return d


def _z_threshold(n_templates, n_time_samples, far_per_test=1e-6):
    """Detection threshold in units of the unit-norm matched filter's sigma.

    The statistic is ``<r, t_j> / (|t_j| sigma)`` with ``r`` pure white noise of
    sd ``sigma``; under H0 it is exactly standard normal for every neuron and
    every time sample.  The threshold is the two-sided normal quantile at
    ``far_per_test`` (Bonferroni over templates x time), computed by the
    standard asymptotic tail ``Phi^-1(1 - p) ~ sqrt(2 ln(1/p) - ln(2 pi ln(1/p)))``
    -- an APPROXIMATION, stated here, that is accurate to ~1e-3 for p <= 1e-3 and
    is swept in the runner.  Declared, not tuned.
    """
    if n_templates < 1 or n_time_samples < 1:
        raise ValueError("need at least one template and one time sample")
    p = float(far_per_test) / (float(n_templates) * float(n_time_samples))
    p = min(max(p, 1e-300), 0.5)
    u = math.log(1.0 / p)
    return math.sqrt(2.0 * u - math.log(2.0 * math.pi * u))


def _neuron_channel_index(templates_mV, *, channel_floor_fraction=1e-3):
    """Ragged (channel, neuron) index of the entries a matched filter should use.

    Entries whose magnitude is below ``channel_floor_fraction`` of THAT neuron's
    largest entry are dropped: at the real noise floor they contribute noise to
    the inner product and no signal.  The fraction is a declared parameter, and
    the number of dropped entries is reported by :func:`separate`.

    Returns ``(rows, cols, values, kept, total)`` where ``values`` is the
    (optionally zeroed) template value, so the "kept" index and the values can
    never disagree.
    """
    T = np.asarray(templates_mV, float)
    mx = np.abs(T).max(axis=0)
    keep = np.abs(T) >= float(channel_floor_fraction) * np.maximum(mx, 1e-300)[None, :]
    kept = T * keep
    rows, cols = np.nonzero(kept)
    return rows, cols, kept[rows, cols], int(keep.sum()), int(T.size)


def _matched_filter_pool(recorded_mV, rows, cols, values, w, n_neuron, out=None):
    """Cross-correlate the residual with every neuron's template, chunked by LAG.

    ``out[j, i] = sum_lag w[lag] * sum_c T[c, j] * R[c, i + lag]``, i.e. the
    un-normalised matched-filter output of neuron ``j`` at sample ``i``.  Only the
    (channel, neuron) entries that survived the channel floor are used.

    Chunk by lag AND time. Pair-by-full-time fancy indexing can allocate many
    gigabytes even without a channel x neuron x time tensor. The gather/product
    temporaries now target <=16 MiB. The persistent neuron x time output is
    separate and still needs a caller size budget; this is not a full memory cap.
    """
    n_ch, n_t = recorded_mV.shape
    # Fail predictably before allocating persistent output/statistic copies.
    # Full-scale work needs a streaming estimator rather than raising this guard.
    if n_neuron * n_t * 8 > 128 * 1024 * 1024:
        raise MemoryError("matched-filter neuron x time output exceeds 128 MiB; "
                          "use a bounded window/source batch")
    if out is None:
        out = np.zeros((n_neuron, n_t))
    L = w.size
    if n_t < L:
        raise ValueError("recording has %d samples, shorter than the %d-sample spike "
                         "waveform" % (n_t, L))
    # NEGATIVE LAGS, matching the synthesiser: waveform[k] multiplies sample (i-k).
    for lag in range(L):
        j0, j1 = lag, n_t - (L - 1 - lag)
        if j1 <= j0:
            continue
        # Bound the fancy-index temporary instead of pairs x FULL time.
        # Both gathered samples and product exist transiently: target <=16 MiB.
        block_t = max(1, min(256, (16 * 1024 * 1024) //
                             max(1, len(rows) * 8 * 2)))
        for start in range(j0, j1, block_t):
            stop = min(j1, start + block_t)
            prod = recorded_mV[rows[:, None], np.arange(start, stop)[None, :]] * values[:, None]
            prod *= w[lag]
            np.add.at(out[:, start:stop], cols, prod)
    # the search never presupposes which sign of the derivative it will see: the
    # statistic is a magnitude, so both polarities of the front end's response are
    # covered without introducing a second, unphysical template.
    return out


def _column_index(cols, rows, values, j):
    """The (channel, value) pairs of ONE neuron's template, from the ragged index."""
    sel = (cols == j)
    return rows[sel], values[sel]


def _waveform_contrib(residual, chans, vals, w, t_idx):
    """<residual_window, template_column . waveform> for one neuron at one time."""
    num = 0.0
    den = 0.0
    vv = float(vals @ vals)
    for lag in range(w.size):
        k = t_idx + lag
        if 0 <= k < residual.shape[1]:
            num += float(vals @ residual[chans, k]) * w[lag]
            den += vv * w[lag] ** 2
    return num, den


def _subtract_spike(residual, chans, vals, w, t_idx, amp):
    """Remove ``amp`` copies of one neuron's spike at ``t_idx`` from the residual."""
    for lag in range(w.size):
        k = t_idx + lag
        if 0 <= k < residual.shape[1]:
            residual[chans, k] -= amp * vals * w[lag]


def _joint_refine(original, rows, cols, values, w, entries, target, n_neuron,
                  min_rel_amp=0.05, max_group=24):
    """Joint non-negative least squares over the spikes whose windows OVERLAP.

    Greedy matching pursuit decides amplitudes one spike at a time, which is
    wrong as soon as two accepted spikes overlap in time (and at 1 ms sampling a
    1 ms-wide spike always overlaps its neighbours).  This recomputes the
    amplitudes of the overlapping group TOGETHER, on a scratch copy of the
    original recording, by projected gradient descent:

        minimise |r - sum_k a_k (G[:, j_k] . w)|^2   s.t. a_k >= 0

    and additionally prunes any member whose amplitude falls below
    ``min_rel_amp`` of the strongest member -- which is exactly what removes the
    duplicate/ghost detections a plain greedy search produces when neighbouring
    templates are highly coherent.

    ``target`` is the index, within ``entries``, of the candidate whose
    amplitude must be positive for the candidate to be accepted (use -1 to
    require only the last entry).  Returns a NEW list of ``(neuron, time,
    amplitude)``; the caller subtracts the returned amplitudes from its own
    residual.
    """
    if not entries:
        return []
    # group = entries whose support overlaps the candidate's spike window
    cand = entries[target]
    lo, hi = cand[1] - (w.size - 1), cand[1] + (w.size - 1)
    group = [e for e in entries if lo <= e[1] <= hi][:int(max_group)]
    if not group:
        group = [cand]
    # candidate must be in the group
    if cand not in group:
        group.append(cand)
    cols_j = []
    chans_j = []
    for (jj, _, _) in group:
        c, v = _column_index(cols, rows, values, jj)
        chans_j.append(c)
        cols_j.append(v)
    a = np.zeros(len(group))
    # scratch residual = the ORIGINAL recording minus everything NOT in the group
    scratch = original.copy()
    for e in entries:
        if e in group:
            continue
        c, v = _column_index(cols, rows, values, e[0])
        _subtract_spike(scratch, c, v, w, e[1], e[2])
    # gradient-descent with non-negativity, fixed small step (Lipschitz-safe)
    gram = np.zeros((len(group), len(group)))
    # diagonal of the normal matrix: |G[:,j] . w|^2 = (v.v) * |w|^2 per lag, with
    # the lag structure; the exact gram is computed on the overlap window only.
    L = w.size
    for p, (jj, ii, _) in enumerate(group):
        for q, (kk2, iq, _) in enumerate(group):
            # <g_p(t), g_q(t')> with the two spikes at ii and iq
            shift = ii - iq
            acc = 0.0
            for lag1 in range(L):
                l2 = lag1 - shift
                if 0 <= l2 < L:
                    acc += float(cols_j[p] @ cols_j[q]) * w[lag1] * w[l2]
            gram[p, q] = acc
    b = np.zeros(len(group))
    for p, (jj, ii, _) in enumerate(group):
        num, _ = _waveform_contrib(scratch, chans_j[p], cols_j[p], w, ii)
        b[p] = num
    Lipschitz = float(np.linalg.norm(gram, 2)) + 1e-30
    for _ in range(400):
        grad = gram @ a - b
        new = np.maximum(a - grad / Lipschitz, 0.0)
        if np.max(np.abs(new - a)) <= 1e-12 * max(1.0, float(np.max(np.abs(a)))):
            a = new
            break
        a = new
    strongest = float(np.max(a)) if a.size else 0.0
    out = []
    for (e, amp) in zip(group, a):
        if amp <= max(float(min_rel_amp) * strongest, 0.0):
            continue
        out.append((int(e[0]), int(e[1]), float(amp)))
    # the candidate must survive
    if int(cand[0]) not in [o[0] for o in out] or not out:
        return []
    if out[[o[0] for o in out].index(int(cand[0]))][2] <= 0:
        return []
    return out


def separate(recorded_mV, templates_mV, waveform_nA, *, method="omp_matched_filter",
             noise_sd_mV=None, dt_ms=1.0, t_offset_ms=0.0,
             far_per_test=1e-6, max_per_time=3, max_iterations=None,
             channel_floor_fraction=1e-3, neuron_mask=None,
             max_detections=None,
             n_time_bins=None, min_rel_amp=0.05):
    """Demultiplex one recorded window into spikes and neuron identities.

    METHODS
    -------
    ``"omp_matched_filter"``  (PRIMARY)
        Greedy sparse least-squares -- orthogonal matching pursuit -- against the
        per-neuron channel templates that the REAL geometry provides.  One
        iteration:

          1. cross-correlate the residual with every neuron's template over every
             sample (vectorised, chunked by lag);
          2. normalise by ``|template| * |waveform| * sigma``: under the model's
             white-noise null that statistic is standard normal for every neuron
             and every sample, so the acceptance threshold is a DECLARED normal
             quantile, not a hand-tuned constant;
          3. accept the single largest exceedance whose LEAST-SQUARES amplitude is
             positive, subtract it from the residual, repeat.

        This is the natural inverse problem for this array: the map from spike
        set to recording is LINEAR and the geometry writes its columns down
        explicitly, so demultiplexing is sparse linear inversion rather than
        blind source separation.

        ASSUMPTIONS (all listed in the returned result, all falsifiable):
          A1 ORACLE templates -- the inversion uses the SAME forward matrix that
             generated the data.  Real performance would be worse; the runner has
             a separate arm that estimates templates from a short calibration
             recording and measures the degradation.
          A2 sparsity -- at most ``max_per_time`` neurons overlap on a sample, and
             each neuron fires at most once per resolvable window;
          A3 white Gaussian noise of the stated sd, independent across channels
             and samples (the front end's Johnson-Nyquist floor is white over the
             band, which is a limitation of the front end, inherited);
          A4 the spike waveform SHAPE is known and identical for every neuron;
          A5 non-negative amplitudes: a solution with a negative amplitude is not
             a spike of that neuron and terminates the search.
        HONEST LABEL: A1 + A4 make this an ORACLE sorter -- within THIS model it
        is the best an estimator can do, because it is handed the operator that
        generated the data.  WHAT THAT DOES NOT MEAN: a failure below is a failure
        of this model's oracle, i.e. the information needed to attribute a spike
        to one of several somata a few micrometres apart is absent from THIS
        forward matrix at THIS noise level.  It is NOT a bound on a real sorter,
        which is not limited by this model's input: there is no morphology here,
        no per-neuron waveform shape, no membrane geometry and no measured noise
        budget, and real separation leans on exactly those.  The conclusion is
        therefore MODEL-SPECIFIC and must never be restated as "sorting this
        array is impossible".

    ``"nearest_channel_wta"``  (BASELINE B1)
        Threshold each channel on its own and assign the winner to whichever
        neuron has the largest gain on that channel.  The shared/multi-channel
        ambiguity is IGNORED by construction -- included because it is the obvious
        thing to do, and because it shows what ignoring the ambiguity costs.

    ``"global_threshold"``  (BASELINE B2)
        Threshold the array-wide maximum against the same declared false-alarm
        level and report a spike with NO neuron identity, so a detection that is
        right in time is still an assignment failure.  This is the "something
        happened" baseline.

    Returns a :class:`SeparationResult`.
    """
    R = np.asarray(recorded_mV, float)
    T = np.asarray(templates_mV, float)
    w = np.asarray(waveform_nA, float).ravel()
    if R.ndim != 2 or T.ndim != 2:
        raise ValueError("recorded_mV and templates_mV must be 2-D")
    if T.shape[0] != R.shape[0]:
        raise ValueError("templates and recording disagree on channel count")
    n_ch, n_t = R.shape
    n_neu = T.shape[1]
    if w.size < 1:
        raise ValueError("waveform_nA must have at least one sample")
    if noise_sd_mV is None:
        noise_sd_mV = float(np.median(np.std(R, axis=1)))
    noise_sd_mV = max(float(noise_sd_mV), 1e-15)
    wn = float(np.linalg.norm(w))

    mask = (np.ones(n_neu, bool) if neuron_mask is None
            else np.asarray(neuron_mask, bool).copy())
    if mask.shape != (n_neu,):
        raise ValueError("neuron_mask must have one entry per template column")

    n_bins = int(n_time_bins) if n_time_bins else n_t
    # the threshold is Bonferroni over EVERY test the search performs: every
    # template the caller allowed x every time sample.  Declared before the run.
    zthr = _z_threshold(int(mask.sum()), n_bins, far_per_test=far_per_test)
    common_params = {"n_channels": int(n_ch), "n_templates": int(n_neu),
                     "noise_sd_mV": float(noise_sd_mV), "dt_ms": float(dt_ms),
                     "t_offset_ms": float(t_offset_ms), "far_per_test": far_per_test,
                     "n_time_samples": int(n_t), "waveform_samples": int(w.size),
                     "noise_sd_uV": float(noise_sd_mV) * 1e3}

    if method == "global_threshold":
        zthr_g = _z_threshold(1, max(n_bins, 1), far_per_test=far_per_test)
        mag = np.abs(R).max(axis=0)
        idx = np.flatnonzero(mag > zthr_g * noise_sd_mV)
        dets = [(None, float(t_offset_ms + i * dt_ms)) for i in idx]
        return SeparationResult(
            method=method, detections=dets, amp=[float(mag[i]) for i in idx],
            z=[float(mag[i] / noise_sd_mV) for i in idx], n_iterations=len(dets),
            stopped="threshold", threshold_z=zthr_g, params=common_params,
            assumptions=("no inverse problem is solved: every detection carries NO "
                         "neuron identity, so it can never be a correct ASSIGNMENT; "
                         "reported as the 'something happened' baseline",))

    rows, cols, values, n_kept, n_total = _neuron_channel_index(
        T, channel_floor_fraction=channel_floor_fraction)
    colnorm = np.sqrt(np.bincount(cols, weights=values ** 2, minlength=n_neu))
    # A neuron whose ENTIRE template fell below the channel floor cannot be
    # detected: its matched-filter output is identically zero and dividing by its
    # (zero) norm would manufacture an infinite statistic out of nothing -- a
    # defect found by running this on the real geometry.  Such neurons are
    # removed from the search and COUNTED.
    detectable = mask & (colnorm > 0)
    n_undetectable = int(np.sum(mask & ~detectable))
    colnorm_safe = np.maximum(colnorm, 1e-300)
    if not detectable.any():
        raise ValueError("no template survived the channel floor "
                         "(channel_floor_fraction=%.3g): nothing is detectable"
                         % (channel_floor_fraction,))

    if method == "nearest_channel_wta":
        gain = np.zeros((n_ch, n_neu))
        gain[rows, cols] = np.abs(values)
        dets, amps, zs = [], [], []
        for k in range(n_ch):
            sig = R[k]
            i = int(np.argmax(np.abs(sig)))
            val = float(sig[i])
            if abs(val) <= zthr * noise_sd_mV:
                continue
            j = int(np.argmax(gain[k])) if np.any(gain[k] > 0) else None
            dets.append((j, float(t_offset_ms + i * dt_ms)))
            amps.append(val)
            zs.append(abs(val) / noise_sd_mV)
        return SeparationResult(
            method=method, detections=dets, amp=amps, z=zs, n_iterations=len(dets),
            stopped="per_channel_scan", threshold_z=zthr,
            params=dict(common_params, assignment_rule=(
                "largest |gain| on the winning channel; ties by lowest neuron index"),
                n_template_entries_kept=n_kept,
                n_template_entries_total=n_total,
                channel_floor_fraction=channel_floor_fraction),
            assumptions=("winner-take-all PER CHANNEL: the shared/multi-channel "
                         "ambiguity is ignored by construction, so the same neuron "
                         "can be claimed by many channels and by many times at once; "
                         "a 21-channel neuron is therefore counted 21 times",))

    if method != "omp_matched_filter":
        raise ValueError("unknown method %r" % (method,))

    # ---- PRIMARY: greedy sparse least squares -----------------------------
    resid = R.copy()
    entries = []          # accepted spikes: (neuron, time_index, amplitude)
    # A DETECTION BUDGET IS MANDATORY, and it is a BOUND, not a tuning knob: at a
    # low noise level the matched-filter statistic exceeds the threshold almost
    # everywhere, and an unbounded greedy search then spends its whole iteration
    # cap chasing the high-pass TAIL of spikes it has already found.  Measured
    # here: a single ~40 s stall on a 30-neuron fixture, because the loop ran to
    # its max_iterations cap.  The budget is derived from the DECLARED trial load
    # and is reported in the result.
    if max_detections is None:
        max_detections = int(max(4, max_per_time * (1 + int(n_t / 100))))
    max_detections = int(max_detections)
    max_it = (int(max_iterations) if max_iterations
              else min(int(max_per_time) * max(n_bins, 1), max_detections))
    stopped = "max_iterations"
    n_nonpositive_rejections = 0
    for it in range(max_it):
        # out[j, i] = sum_lag w[lag] * <T[:, j], resid[:, i + lag]>
        out = _matched_filter_pool(resid, rows, cols, values, w, n_neu)
        # ONE spike per neuron per window (assumption A2): a claimed neuron is
        # masked out, so the loop cannot spend its iterations re-detecting it
        for (jj, _, _) in entries:
            out[jj, :] = -np.inf
        stat = out / (colnorm_safe[:, None] * wn * noise_sd_mV)
        stat = np.where(detectable[:, None], np.abs(stat), -1.0)
        flat = int(np.argmax(stat))
        j, i = np.unravel_index(flat, stat.shape)
        z = float(stat[j, i])
        if z < zthr:
            stopped = "threshold"
            break
        cand = (int(j), int(i), 0.0)
        ref = _joint_refine(R, rows, cols, values, w, entries + [cand], target=-1,
                            n_neuron=n_neu, min_rel_amp=min_rel_amp)
        a_cand = None
        for (jj, ii, aa) in ref:
            if jj == int(j) and ii == int(i):
                a_cand = aa
        if a_cand is None or a_cand <= 0:
            n_nonpositive_rejections += 1
            stopped = "nonpositive_amplitude_after_joint_refinement"
            break
        # subtract the REFINED amplitudes of the whole refined group from the
        # residual, and keep the group's amplitudes (not the greedy ones)
        for (jj, ii, aa) in ref:
            c, v = _column_index(cols, rows, values, jj)
            _subtract_spike(resid, c, v, w, ii, aa)
        refined = {(jj, ii): aa for (jj, ii, aa) in ref}
        entries = [(jj, ii, refined.get((jj, ii), 0.0)) for (jj, _, _) in entries]
        entries = [e for e in entries if e[0] != int(j) or e[1] != int(i)]
        entries.extend(ref)
    entries = sorted(entries, key=lambda e: e[1])
    return SeparationResult(
        method=method,
        detections=[(int(j), float(t_offset_ms + i * dt_ms)) for (j, i, _) in entries],
        amp=[float(a) for (_, _, a) in entries], z=[float(z) for _ in entries],
        n_iterations=len(entries), stopped=stopped, threshold_z=zthr,
        params=dict(common_params, max_per_time=int(max_per_time),
                    n_template_entries_kept=n_kept,
                    n_template_entries_total=n_total,
                    channel_floor_fraction=channel_floor_fraction,
                    n_nonpositive_amplitude_rejections=int(n_nonpositive_rejections),
                    n_undetectable_templates=int(n_undetectable),
                    max_detections=int(max_detections),
                    amplitude_rule=(
                        "joint non-negative least squares (projected gradient) over "
                        "the detections whose spike windows OVERLAP in time, solved on "
                        "a scratch copy of the ORIGINAL recording; members below 5%% "
                        "of the group's strongest amplitude are pruned, which is what "
                        "removes the ghost/duplicate detections a plain greedy search "
                        "produces when neighbouring templates are coherent"),
                    normalisation=("<T[:,j], resid_window> / (|T[:,j]| |w| sigma): "
                                   "standard normal under the white-noise null")),
        assumptions=(
            "A1 ORACLE: the inversion uses the SAME forward matrix that generated the "
            "data; a real sorter must estimate its templates (measured separately)",
            "A2 sparsity: at most %d simultaneous neurons per sample, ONE spike per "
            "neuron per window" % int(max_per_time),
            "A3 white Gaussian noise of the stated sd, independent across channels "
            "and samples",
            "A4 spike waveform SHAPE known and identical for every neuron",
            "A5 non-negative amplitudes (a negative joint solution ends the search)"))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(truth: SpikeTruth, estimate: SeparationResult, *, dt_ms=1.0,
             tolerance_ms=1.5, neuron_is_shared=None):
    """Detection and ASSIGNMENT metrics of one estimate against the truth.

    Matching is greedy on TIME: a true spike and a detection match when their
    peaks differ by at most ``tolerance_ms``.  A detection that carries no neuron
    identity can still match on time, and is then counted as a DETECTION with a
    WRONG identity -- an assignment failure -- never as a clean hit.  "Something
    happened" is not source separation.

    The shared/multi-channel population is scored separately, because it is the
    population this module exists for: ``resolvable_neurons`` counts those whose
    recovered spikes are at least half correct, which is the criterion
    :func:`separability_limit` bisects on.
    """
    if not isinstance(truth, SpikeTruth):
        raise TypeError("truth must be a SpikeTruth")
    spikes = list(truth.spikes)
    dets = [tuple(d) for d in estimate.detections]
    n_true = len(spikes)
    n_det = len(dets)

    used = np.zeros(n_det, bool)
    n_det_hits = n_assign_ok = n_assign_bad = 0
    per_neuron = {}
    for (tj, tt) in spikes:
        best, best_dt = -1, None
        for di, (dj, dt) in enumerate(dets):
            if used[di]:
                continue
            d = abs(dt - tt)
            if d <= tolerance_ms and (best_dt is None or d < best_dt):
                best, best_dt = di, d
        rec = per_neuron.setdefault(int(tj), {"true": 0, "detected": 0,
                                             "assigned_ok": 0, "assigned_bad": 0,
                                             "n_detections_on_me": 0})
        rec["true"] += 1
        rec["n_detections_on_me"] += sum(
            1 for (dj, dt) in dets if dj is not None and int(dj) == int(tj)
            and abs(dt - tt) <= tolerance_ms)
        if best < 0:
            continue
        used[best] = True
        n_det_hits += 1
        rec["detected"] += 1
        dj = dets[best][0]
        if dj is not None and int(dj) == int(tj):
            n_assign_ok += 1
            rec["assigned_ok"] += 1
        else:
            n_assign_bad += 1
            rec["assigned_bad"] += 1
    n_false = int((~used).sum())

    shared = (np.asarray(neuron_is_shared, bool) if neuron_is_shared is not None
              else np.asarray(truth.is_shared, bool))
    if shared.size != truth.neuron_rows.size:
        raise ValueError("neuron_is_shared must have one entry per model neuron")
    shared_set = set(np.flatnonzero(shared).tolist())
    solo_set = set(np.flatnonzero(~shared).tolist())

    def _agg(idx_set):
        t = d = a = b = 0
        neu = 0
        resolvable = 0
        for j, rec in per_neuron.items():
            if j not in idx_set:
                continue
            neu += 1
            t += rec["true"]
            d += rec["detected"]
            a += rec["assigned_ok"]
            b += rec["assigned_bad"]
            if rec["true"] > 0 and rec["assigned_ok"] >= 0.5 * rec["true"]:
                resolvable += 1
        return {"n_neurons_with_spikes": neu, "n_true_spikes": t,
                "n_detected": d, "n_assigned_correctly": a,
                "n_assigned_wrongly": b,
                "detection_rate": (d / t if t else None),
                "assignment_accuracy": (a / d if d else None),
                "identity_given_truth": (a / t if t else None),
                "resolvable_neurons": resolvable,
                "resolvable_fraction_of_neurons_with_spikes": (
                    resolvable / neu if neu else None)}

    sh = _agg(shared_set)
    so = _agg(solo_set)
    return {
        "method": estimate.method,
        "n_true_spikes": n_true,
        "n_detections": n_det,
        "n_detections_matched_to_a_true_spike": n_det_hits,
        "n_detections_unmatched_false_positives": n_false,
        "false_positives_per_true_spike": (n_false / n_true if n_true else None),
        "precision": (n_det_hits / n_det if n_det else None),
        "detection_rate": (n_det_hits / n_true if n_true else None),
        "assignment_accuracy_of_detections": (n_assign_ok / n_det_hits
                                             if n_det_hits else None),
        "n_assigned_correctly": n_assign_ok,
        "n_assigned_wrongly": n_assign_bad,
        "identity_given_truth": (n_assign_ok / n_true if n_true else None),
        "tolerance_ms": float(tolerance_ms),
        "dt_ms": float(dt_ms),
        "shared": sh,
        "solo": so,
        "n_shared_neurons_in_model": int(shared.sum()),
        "n_shared_neurons_with_spikes": sh["n_neurons_with_spikes"],
        "threshold_z_used": float(estimate.threshold_z),
        "stopped": estimate.stopped,
        "note": ("detection and identity are scored SEPARATELY: a detection with the "
                 "wrong neuron counts as a detection and as an assignment failure, so "
                 "a method that only says 'a spike happened' cannot score on identity"),
    }


# ---------------------------------------------------------------------------
# geometry-only resolvability (no noise, no estimator)
# ---------------------------------------------------------------------------
def resolvability_geometry(templates_mV, waveform_nA, *, neuron_is_shared=None,
                           max_neurons=600, n_pairs_per_neuron=400, seed=0,
                           coherence_cutoffs=(0.5, 0.9, 0.99, 0.999, 0.9999)):
    """The coherent-pair structure of the templates: the MECHANISM of the ambiguity.

    For unit-norm templates ``u_j = g_j/|g_j|``, ``|<u_j, u_k>|`` is how
    indistinguishable two neurons are to a matched filter BEFORE any noise.  Two
    neurons with coherence ``c`` can only be told apart when the spike SNR
    ``|g||w|/sigma`` exceeds roughly ``1/(1-c)``: at equality, the response of
    the wrong neuron to the right neuron's spike has caught up with the detection
    threshold.  This function measures the coherence distribution by sampling
    pairs, and reports the ``1/(1-c)`` SNR requirement directly so the geometry's
    own demand can be compared with the SNR the front end actually delivers.

    HONEST LIMIT: pairs are SAMPLED (``max_neurons`` x ``n_pairs_per_neuron``),
    not exhaustively enumerated, so the reported maximum coherence is a sampled
    maximum and is written as such.
    """
    T = np.asarray(templates_mV, float)
    w = np.asarray(waveform_nA, float).ravel()
    wn = float(np.linalg.norm(w))
    n_neu = T.shape[1]
    cn = np.linalg.norm(T, axis=0)
    U = T / np.maximum(cn, 1e-300)[None, :]
    rng = np.random.default_rng(seed)
    idx = (np.arange(n_neu) if n_neu <= max_neurons
           else np.sort(rng.choice(n_neu, size=int(max_neurons), replace=False)))
    coh = []
    max_coh = np.zeros(n_neu)
    for j in idx:
        k = rng.choice(n_neu, size=min(int(n_pairs_per_neuron), n_neu), replace=False)
        c = np.abs(U[:, k].T @ U[:, j])
        c[k == j] = 0.0
        coh.append(c)
        max_coh[j] = float(c.max()) if c.size else 0.0
    coh = np.concatenate(coh) if coh else np.zeros(1)
    out = {
        "n_neurons": int(n_neu),
        "n_sampled_pairs": int(coh.size),
        "waveform_norm_A_samples": wn,
        "template_norm_mV_median": float(np.median(cn)),
        "template_norm_mV_max": float(cn.max()),
        "template_norm_mV_min": float(cn.min()),
        "self_z_at_1uV_noise_median": float(np.median(cn) * wn / 1e-3),
        "self_z_at_1uV_noise_max": float(cn.max() * wn / 1e-3),
        "sampled_pair_coherence_median": float(np.median(coh)),
        "sampled_pair_coherence_p99": float(np.percentile(coh, 99)),
        "sampled_pair_coherence_max": float(coh.max()),
        "sampled_pair_counts_by_cutoff": {str(c): int((coh >= c).sum())
                                          for c in coherence_cutoffs},
        "snr_required_for_cutoff": {str(c): (1.0 / (1.0 - c) if c < 1 else float("inf"))
                                    for c in coherence_cutoffs},
        "max_sampled_coherence_per_neuron_median": float(np.median(max_coh[idx])),
        "definition": ("|cos| = |<g_j, g_k>| / (|g_j||g_k|) between two neurons' "
                       "channel-space templates: how indistinguishable they are to a "
                       "matched filter, before any noise"),
        "note": ("a high coherence between two co-captured neurons is PURE GEOMETRY: "
                 "two somas a few micrometres apart project nearly the same pattern "
                 "onto the array, and no sorter can beat it with a linear filter. "
                 "Pairs are SAMPLED, not enumerated."),
    }
    if neuron_is_shared is not None:
        sh = np.asarray(neuron_is_shared, bool)
        if sh.size != n_neu:
            raise ValueError("neuron_is_shared must have one entry per neuron")
        out["n_shared_neurons"] = int(sh.sum())
        if sh.any():
            out["max_sampled_coherence_per_neuron_median_shared"] = float(
                np.median(max_coh[np.flatnonzero(sh)]))
    # the RAW sample, so a caller can plot the distribution it is quoting instead
    # of trusting a summary statistic
    out["_coherence_sample"] = coh.astype(float)
    return out


def matched_filter_snr(model_or_templates, waveform_nA=None, noise_sd_mV=None):
    """Per-neuron matched-filter SNR: ``|G[:,j] . w| / sigma``.

    This is THE quantity that decides whether a neuron's spike can be detected at
    all, and unlike a single-channel amplitude ratio it costs nothing to state in
    advance.  Under the model's white-noise null the matched-filter statistic of
    neuron ``j`` is standard normal, so its mean under H1 is exactly this SNR and
    the detection threshold is a declared normal quantile.

    Also returns, per neuron, the noise sd at which its own spike would reach a
    DECLARED threshold (given by the caller through ``noise_sd_mV`` for scaling
    only -- the crossing sd is reported at z = 1 so any threshold can be applied
    by multiplying: ``sd_crossing(z) = sd_at_z1 / z``).
    """
    if isinstance(model_or_templates, SignalModel):
        T = model_or_templates.templates
        w = model_or_templates.waveform_nA
    else:
        T = np.asarray(model_or_templates, float)
        if waveform_nA is None:
            raise ValueError("waveform_nA is required with a bare template matrix")
        w = np.asarray(waveform_nA, float).ravel()
    wn = float(np.linalg.norm(w))
    tn = np.linalg.norm(T, axis=0) * wn
    out = {
        "n_neurons": int(T.shape[1]),
        "template_norm_times_waveform_norm_mV": tn,
        "definition": ("|G[:,j] . w|: the mean of the matched-filter statistic, in mV, "
                       "i.e. the SNR in units of the noise sd"),
        "snr_at_1uV_noise_median": float(np.median(tn) / 1e-3),
        "snr_at_1uV_noise_max": float(tn.max() / 1e-3),
        "snr_at_1uV_noise_p99": float(np.percentile(tn, 99) / 1e-3),
        "amplitude_scaling": ("every entry scales LINEARLY with spike_peak_nA and "
                              "INVERSELY with conductivity_S_m and with the noise sd"),
    }
    if noise_sd_mV is not None:
        sd = float(noise_sd_mV)
        snr = tn / sd
        out["noise_sd_mV"] = sd
        out["snr_at_that_noise_median"] = float(np.median(snr))
        out["snr_at_that_noise_max"] = float(snr.max())
        out["n_neurons_snr_ge_1"] = int((snr >= 1.0).sum())
        out["n_neurons_snr_ge_5"] = int((snr >= 5.0).sum())
        out["n_neurons_snr_ge_6.5"] = int((snr >= 6.5).sum())
        out["snr_percentiles"] = {str(p): float(np.percentile(snr, p))
                                  for p in (1, 25, 50, 75, 99, 100)}
    return out


# ---------------------------------------------------------------------------
# the noise sweep and the separability limit
# ---------------------------------------------------------------------------
def make_trial_spikes(model: SignalModel, *, n_signal_neurons, rng,
                      neuron_is_shared=None, shared_only=False,
                      min_time_ms=1.0, edge_ms=5.0):
    """One DECLARED ground-truth spike set: each chosen neuron fires ONCE.

    The firing times are drawn uniformly inside the window with a declared edge
    margin so that each spike's whole waveform is inside the window.  This is an
    experiment design, NOT a measured firing rate: nothing in this project has a
    spike generator for these neurons and no refractory mechanism is modelled.
    """
    n_neu = model.G.shape[1]
    pool = np.arange(n_neu)
    if shared_only:
        if neuron_is_shared is None:
            raise ValueError("shared_only needs neuron_is_shared")
        pool = np.flatnonzero(np.asarray(neuron_is_shared, bool))
    k = min(int(n_signal_neurons), pool.size)
    chosen = rng.choice(pool, size=k, replace=False)
    t_max = max(min_time_ms + 1.0, model.n_steps * model.dt_ms - edge_ms)
    spikes = [(int(j), float(rng.uniform(min_time_ms, t_max))) for j in chosen]
    return sorted(spikes, key=lambda s: s[1])


def run_noise_level(model: SignalModel, *, noise_sd_mV, neuron_is_shared,
                    n_trials=8, neurons_in_window=6, method="omp_matched_filter",
                    tolerance_ms=1.5, dt_ms=1.0, far_per_test=1e-6,
                    max_per_time=3, seed=0, compare_methods=None,
                    shared_only=False, keep_per_trial=False,
                    max_detections=None):
    """One noise level: synthesise -> separate -> evaluate, ``n_trials`` times.

    The signal is FIXED and only the noise moves.  Because every stage of the
    chain is linear, this sweep is exactly equivalent to moving the source
    amplitude in the other direction, which is stated in the output.
    """
    rng = np.random.default_rng(seed)
    trials = []
    for it in range(int(n_trials)):
        spikes = make_trial_spikes(model, n_signal_neurons=neurons_in_window, rng=rng,
                                   neuron_is_shared=neuron_is_shared,
                                   shared_only=shared_only)
        V, sinfo = model.record(spikes, noise_sd_mV=noise_sd_mV,
                                seed=int(seed) * 100003 + it)
        est = separate(V, model.templates, model.waveform_nA, method=method,
                       noise_sd_mV=noise_sd_mV, dt_ms=dt_ms,
                       far_per_test=far_per_test, max_per_time=max_per_time,
                       max_detections=max_detections)
        truth = SpikeTruth(neuron_rows=np.arange(model.G.shape[1]),
                           soma_root_ids=np.zeros(model.G.shape[1], np.int64),
                           spikes=spikes, is_shared=neuron_is_shared,
                           params={"n_signal_neurons": len(spikes), "trial": it},
                           seed=int(seed))
        m = evaluate(truth, est, dt_ms=dt_ms, tolerance_ms=tolerance_ms,
                     neuron_is_shared=neuron_is_shared)
        m["_est"] = est
        trials.append(m)

    def _agg(key, sub=None, how="mean"):
        vals = []
        for t in trials:
            v = t[sub][key] if sub else t[key]
            if v is not None:
                vals.append(float(v))
        if not vals:
            return None
        return float(np.mean(vals)) if how == "mean" else float(np.median(vals))

    row = {
        "noise_sd_mV": float(noise_sd_mV), "noise_sd_uV": float(noise_sd_mV) * 1e3,
        "method": method, "n_trials": int(n_trials),
        "neurons_in_window": int(neurons_in_window),
        "n_true_spikes_per_trial_mean": _agg("n_true_spikes"),
        "detection_rate_mean": _agg("detection_rate"),
        "detection_rate_median": _agg("detection_rate", how="median"),
        "assignment_accuracy_mean": _agg("assignment_accuracy_of_detections"),
        "precision_mean": _agg("precision"),
        "identity_given_truth_mean": _agg("identity_given_truth"),
        "false_positives_per_true_spike_mean": _agg("false_positives_per_true_spike"),
        "shared_detection_rate_mean": _agg("detection_rate", "shared"),
        "shared_assignment_accuracy_mean": _agg("assignment_accuracy", "shared"),
        "shared_identity_given_truth_mean": _agg("identity_given_truth", "shared"),
        "shared_resolvable_neurons_mean": _agg("resolvable_neurons", "shared"),
        "shared_resolvable_fraction_mean": _agg(
            "resolvable_fraction_of_neurons_with_spikes", "shared"),
        "shared_neurons_with_spikes_mean": _agg("n_neurons_with_spikes", "shared"),
        "solo_resolvable_fraction_mean": _agg(
            "resolvable_fraction_of_neurons_with_spikes", "solo"),
        "n_detections_mean": _agg("n_detections"),
        "noise_is_the_swept_quantity": True,
    }
    if keep_per_trial:
        row["per_trial"] = [{k: v for k, v in t.items() if k != "_est"}
                            for t in trials]
        row["per_trial_reasons"] = [t["_est"].stopped for t in trials]
    if compare_methods:
        row["comparison"] = {}
        for meth in compare_methods:
            r2 = run_noise_level(
                model, noise_sd_mV=noise_sd_mV, neuron_is_shared=neuron_is_shared,
                n_trials=n_trials, neurons_in_window=neurons_in_window, method=meth,
                tolerance_ms=tolerance_ms, dt_ms=dt_ms, far_per_test=far_per_test,
                max_per_time=max_per_time, seed=seed, shared_only=shared_only)
            row["comparison"][meth] = {k: r2[k] for k in (
                "detection_rate_mean", "assignment_accuracy_mean", "precision_mean",
                "identity_given_truth_mean", "shared_resolvable_fraction_mean")}
    return row


def sweep_noise(model, *, noise_levels_uV, neuron_is_shared, n_trials=8,
                neurons_in_window=6, far_per_test=1e-6, max_per_time=3, seed=0,
                dt_ms=1.0, tolerance_ms=1.5, compare_methods=None,
                shared_only=False, progress=None, keep_per_trial=False):
    """Sweep the NOISE level with the signal fixed -- the binding constraint.

    The existing front end's LEGACY rectangular-band diagnostic puts 24.02 uV RMS
    (not reproducible from its own recorded band; see the module docstring) under
    a median neural signal
    of about 0.37 uV, so at the declared source amplitude the whole problem sits
    far in the noise; the sweep is how the transition is LOCATED rather than
    asserted.
    """
    rows = []
    for i, nl in enumerate(noise_levels_uV):
        row = run_noise_level(
            model, noise_sd_mV=float(nl) * 1e-3, neuron_is_shared=neuron_is_shared,
            n_trials=n_trials, neurons_in_window=neurons_in_window,
            tolerance_ms=tolerance_ms, dt_ms=dt_ms, far_per_test=far_per_test,
            max_per_time=max_per_time, seed=seed + 1000 * i,
            compare_methods=compare_methods, shared_only=shared_only,
            keep_per_trial=keep_per_trial)
        row["index"] = i
        rows.append(row)
        if progress is not None:
            progress(row)
    return rows


def separability_limit(model, *, neuron_is_shared, floor_uV=None, levels_uV=None,
                       n_trials=8, resolvable_target=0.5, neurons_in_window=6,
                       far_per_test=1e-6, max_per_time=3, seed=0, dt_ms=1.0,
                       tolerance_ms=1.5, max_refine=6, progress=None,
                       shared_only=False):
    """The noise level at which separation of the SHARED neurons stops working.

    DEFINITION (fixed before the search, never retuned): a shared (multi-channel)
    neuron is RESOLVABLE in a trial when at least half of its true spikes are both
    time-matched and assigned to the RIGHT neuron.  The SEPARABILITY LIMIT is the
    largest noise sd at which the mean resolvable fraction over the trials is
    still >= ``resolvable_target``.

    METHOD: a fixed coarse sweep first (so the shape is visible whatever the
    answer turns out to be), then BISECTION on the bracket that sweep measured.
    Every bisection step is a real trial run -- nothing is extrapolated or fitted.

    BOUNDS ARE REPORTED AS BOUNDS.  If the coarse sweep is resolvable at EVERY
    level run, the answer is a LOWER_BOUND_ONLY (the true limit is above the
    highest level tried); if it is resolvable at NONE, it is UPPER_BOUND_ONLY.
    """
    if levels_uV is None:
        levels_uV = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 24.02, 30.0, 100.0]
    rows = sweep_noise(model, noise_levels_uV=levels_uV,
                       neuron_is_shared=neuron_is_shared, n_trials=n_trials,
                       neurons_in_window=neurons_in_window,
                       far_per_test=far_per_test, max_per_time=max_per_time,
                       seed=seed, dt_ms=dt_ms, tolerance_ms=tolerance_ms,
                       progress=progress, shared_only=shared_only)

    def frac(row):
        v = row["shared_resolvable_fraction_mean"]
        return -1.0 if v is None else float(v)

    ordered = sorted(rows, key=lambda r: r["noise_sd_uV"])
    a = None    # highest level that PASSES
    b = None    # lowest level ABOVE a that FAILS
    frac_at_a = None
    frac_at_b = None
    for r in ordered:
        if frac(r) >= resolvable_target:
            a, frac_at_a = float(r["noise_sd_uV"]), frac(r)
        elif a is not None and b is None:
            b, frac_at_b = float(r["noise_sd_uV"]), frac(r)

    attempts = []
    if a is None:
        status, limit = "UPPER_BOUND_ONLY", float(ordered[0]["noise_sd_uV"])
        detail = ("the shared neurons were NOT resolvable at the LOWEST noise level "
                  "tried, so the limit is AT OR BELOW that level and the sweep "
                  "cannot bracket it from below")
    elif b is None:
        status, limit = "LOWER_BOUND_ONLY", float(ordered[-1]["noise_sd_uV"])
        detail = ("the shared neurons were resolvable at EVERY noise level tried, so "
                  "the limit is AT OR ABOVE the highest level and the sweep cannot "
                  "bracket it from above")
    else:
        for it in range(int(max_refine)):
            if b / a <= 1.05:
                break
            mid = math.sqrt(a * b)
            row = run_noise_level(
                model, noise_sd_mV=mid * 1e-3, neuron_is_shared=neuron_is_shared,
                n_trials=n_trials, neurons_in_window=neurons_in_window,
                tolerance_ms=tolerance_ms, dt_ms=dt_ms, far_per_test=far_per_test,
                max_per_time=max_per_time, seed=seed + 9000 + it,
                shared_only=shared_only)
            row["bisection_step"] = it
            attempts.append(row)
            if frac(row) >= resolvable_target:
                a, frac_at_a = mid, frac(row)
            else:
                b, frac_at_b = mid, frac(row)
        status, limit = "MEASURED", float(a)
        detail = ("bracketed by REAL trial runs between a passing level (%.4g uV, "
                  "resolvable fraction %.3f) and a failing one (%.4g uV, %.3f). The "
                  "limit reported is the highest level MEASURED to pass, so it is a "
                  "one-sided measurement of the boundary, not a fitted crossing."
                  % (a, -1.0 if frac_at_a is None else frac_at_a, b,
                     -1.0 if frac_at_b is None else frac_at_b))

    out = {
        "definition": ("largest noise sd at which the mean fraction of RESOLVABLE "
                       "shared/multi-channel neurons is still >= %.3f, where a shared "
                       "neuron is resolvable in a trial when >= 50%% of its true spikes "
                       "are both time-matched and assigned to the RIGHT neuron"
                       % resolvable_target),
        "resolvable_target": float(resolvable_target),
        "coarse_levels_uV": [float(v) for v in levels_uV],
        "coarse_rows": rows,
        "bisection_attempts": attempts,
        "status": status,
        "limit_noise_uV": limit,
        "resolvable_fraction_at_last_passing_level": (None if frac_at_a is None
                                                     else float(frac_at_a)),
        "resolvable_fraction_at_first_failing_level": (None if frac_at_b is None
                                                       else float(frac_at_b)),
        "n_trials_per_level": int(n_trials),
        "detail": detail,
        "note": ("the boundary is found by REAL trial runs and a bisection on the "
                 "interval they bracket; no curve is fitted and no value is "
                 "extrapolated past the measured bracket"),
    }
    if floor_uV is not None:
        f = float(floor_uV)
        out["measured_noise_floor_uV"] = f
        out["margin_floor_over_limit_factor"] = (f / limit if limit else None)
        out["margin_floor_over_limit_dB"] = (20.0 * math.log10(f / limit)
                                            if limit else None)
        out["amplitude_scale_needed_at_floor"] = (f / limit if limit else None)
        out["interpretation"] = (
            "if the limit is BELOW the noise level of interest, separation of the shared "
            "neurons at the real floor needs the noise to fall by the reported "
            "factor -- or, since every stage of the chain is linear, the DECLARED "
            "source amplitude to rise by the same factor.  The two directions are "
            "the SAME number.")
    return out


# ---------------------------------------------------------------------------
# template estimation from a calibration recording (the non-oracle arm)
# ---------------------------------------------------------------------------
def estimate_templates_from_calibration(model: SignalModel, *, n_calibrated,
                                        rng, min_snr=5.0, noise_sd_mV=None,
                                        max_channels_per_neuron=24):
    """Estimate each neuron's channel template from a recording where ONE neuron
    fires at a time -- the honest, non-oracle alternative to being handed the
    forward matrix.

    This is the standard calibration protocol: fire one source, average the
    aligned windows.  ``min_snr`` is the matched-filter SNR at which a neuron's
    AVERAGE must exceed the noise before its template is trusted (a declared
    threshold, and the count of neurons that fail it is returned, not hidden).

    Returns ``(templates, report)`` where ``templates`` has the same shape as
    ``model.templates`` with zeros in the columns of neurons whose template could
    not be estimated.
    """
    sd = model.noise_sd_mV if noise_sd_mV is None else float(noise_sd_mV)
    n_ch, n_neu = model.G.shape
    T_est = np.zeros((n_ch, n_neu))
    L = model.waveform_nA.size
    n_rep = 1  # one calibration spike per neuron; the model is noise-free per spike
    picked = np.arange(min(int(n_calibrated), n_neu))
    n_ok = 0
    n_failed = 0
    peak = int(np.argmax(np.abs(model.waveform_nA)))
    for j in picked:
        t_ms = 10.0
        V, _ = model.record([(int(j), t_ms)], noise_sd_mV=sd, seed=int(rng.integers(1, 2**31)))
        i0 = int(round(t_ms / model.dt_ms))
        # the waveform peak lands at the spike time; take the window around it
        lo = i0 - peak
        hi = lo + L
        if lo < 0 or hi > V.shape[1]:
            n_failed += 1
            continue
        w = model.waveform_nA
        win = V[:, lo:hi]
        num = win @ w
        den = float(w @ w)
        est = num / max(den, 1e-300)            # (n_ch,) per-spike amplitude estimate
        snr = float(np.linalg.norm(est) * np.linalg.norm(w) / sd)
        if snr < float(min_snr):
            n_failed += 1
            continue
        T_est[:, j] = est
        n_ok += 1
    report = {
        "n_neurons_calibrated": int(len(picked)),
        "n_templates_estimated": int(n_ok),
        "n_templates_failed_snr": int(n_failed),
        "min_snr_declared": float(min_snr),
        "noise_sd_mV": sd,
        "spearman_note": ("a per-spike amplitude estimate IS the template column here, "
                          "because the model is linear and the waveform is known; a "
                          "real calibration would have to average many repetitions"),
        "zero_columns": int(n_neu - n_ok),
    }
    return T_est, report


def build_sorting_ledger(*, model: SignalModel, gain_info: dict,
                         noise_floor_uV: float, noise_floor_detail: dict,
                         access_map, spike_sigma_ms: float, spike_peak_nA: float,
                         n_neurons_captured: int, n_shared: int,
                         sweep_noise_levels_uV: Sequence[float],
                         access_map_json_source: str, far_per_test: float = 1e-6):
    """Assemble the provenance ledger for a sorting run.

    Every entry states its class.  ``MEASURED_CITED`` entries carry the source
    actually consulted (and the ledger REFUSES one without it);
    ``MEASURED_LOCAL`` entries name the artefact on this machine; every
    ``ASSUMED`` entry carries a declared sweep range (and the ledger REFUSES one
    without it).  The CC BY 4.0 attribution of the soma data is carved in as a
    licence-condition record because attribution is a CONDITION of the licence.
    """
    led = SortingLedger()

    # ---- data provenance (CC BY 4.0 licence condition) --------------------
    led.record("dataset.banc_somas", BANC_SOMA_DATASET_TITLE, "name",
               PROV.MEASURED_CITED,
               source="%s, Harvard Dataverse, file %s (file id %d), %s"
                      % (BANC_SOMA_DOI, BANC_SOMA_FILE, BANC_SOMA_FILE_ID,
                         BANC_SOMA_LICENCE),
               note=("CC BY 4.0: attribution is a LICENCE CONDITION, not a courtesy. "
                     "Redistribution of these soma coordinates, or of any derived "
                     "template, spike train or figure, MUST carry this record."))
    led.record("dataset.banc_somas_doi", BANC_SOMA_DOI, "doi", PROV.MEASURED_CITED,
               source="https://doi.org/10.7910/DVN/7WTH1N")
    led.record("dataset.banc_somas_licence", BANC_SOMA_LICENCE, "licence",
               PROV.MEASURED_CITED,
               source="http://creativecommons.org/licenses/by/4.0",
               note="attribution mandatory on redistribution")
    led.record("dataset.attribution_block", SORTING_ATTRIBUTION, "record",
               PROV.MEASURED_CITED,
               source="engine.embodied.access_map.BANC_SOMA_ATTRIBUTION",
               note="carried into every output dict this module produces")

    # ---- geometry: FIXED inputs, inherited --------------------------------
    led.record("electrode.pitch_um", float(access_map.spec.pitch_um), "um",
               PROV.ENGINEERING_DEFAULT,
               source="FIXED USER DECISION, inherited from the access map unchanged",
               note=("FIXED. pitch (20 um) < 2 x capture radius (50 um), so capture "
                     "spheres overlap and the channel<->neuron assignment is ambiguous "
                     "BY CONSTRUCTION. This module does not change, sweep or optimise "
                     "the geometry."))
    led.record("electrode.shaft_diameter_um", float(access_map.spec.diameter_um), "um",
               PROV.ENGINEERING_DEFAULT,
               source="FIXED USER DECISION; consumed through ContactGeometry",
               note="FIXED; sets the contact area and the area-averaging correction")
    led.record("capture.radius_um", float(access_map.spec.capture_radius_um), "um",
               PROV.ASSUMED, sweep=(10.0, 25.0, 50.0, 100.0),
               source="DECLARED value inherited unchanged from the access map",
               note=("DECLARED, not measured. It sets how many channels see one neuron "
                     "and therefore the size of the demultiplexing problem; its sweep "
                     "is in run_access_map.py, and it is NOT swept or tuned here."))
    led.record("medium.conductivity_S_m", float(model.conductivity_S_m), "S/m",
               PROV.ASSUMED, sweep=(0.1, 0.2, 0.3, 0.5, 1.0),
               source="engine.electrode default, reused unchanged",
               note=("no conductivity was measured in this stack. Every potential, and "
                     "therefore every template, scales as 1/sigma; amplitudes are "
                     "reported so a reader can rescale."))

    # ---- the DECLARED spike waveform --------------------------------------
    led.record("spike.waveform_model", SPIKE_WAVEFORM_NOTE, "text", PROV.ASSUMED,
               sweep=(0.5, 1.0, 2.0),
               note=("DECLARED Gaussian lobe; NO waveform exists in the data (single "
                     "voxel somata). Sweep range = the sigma_ms values the runner "
                     "varies."))
    led.record("spike.sigma_ms", float(spike_sigma_ms), "ms", PROV.ASSUMED,
               sweep=(0.5, 1.0, 2.0),
               note=("DECLARED width of the current pulse; nothing in this stack "
                     "measured a spike width. Sweep range is the declared range the "
                     "runner varies."))
    led.record("spike.peak_current_nA", float(spike_peak_nA), "nA", PROV.ASSUMED,
               sweep=(0.01, 0.1, 1.0, 3.0, 10.0),
               note=("DECLARED source amplitude, the same convention as "
                     "engine.electrode_frontend's neuron_current_nA=0.01 nA. No "
                     "membrane current was measured here. Every SNR number scales "
                     "LINEARLY with it and the runner reports the amplitude needed at "
                     "the noise level of the recording chain as well as the value "
                     "used."))

    # ---- the noise floor (a MODEL number, not a bench measurement) --------
    led.record("noise.floor_uV_rms", float(noise_floor_uV), "uV RMS",
               PROV.MEASURED_LOCAL,
               source=("outputs/embodied_body/electrode_frontend.json :: "
                       "noise.electrode_only.rms_uV (band 1e-6..1e4 Hz)"),
               note=("Johnson-Nyquist floor of the 7 um contact with a 10 MOhm "
                     "amplifier input; RECOMPUTED in this run through "
                     "engine.electrode_frontend.thermal_noise_variance and checked "
                     "against the recorded value. MEASURED_LOCAL of THIS stack -- not "
                     "a measurement of any real electrode."))
    led.record("noise.band_Hz", [float(v) for v in model.noise_band_Hz], "Hz",
               PROV.ENGINEERING_DEFAULT,
               source="run_electrode_frontend.py's stated band",
               note="band of the floor above; kept identical to the front end's")
    led.record("noise.floor_detail", noise_floor_detail, "record",
               PROV.MEASURED_LOCAL,
               source="engine.electrode_frontend.thermal_noise_variance, this run")

    # ---- the forward model ------------------------------------------------
    led.record("forward.matrix_entries", int(gain_info["n_nonzero_entries"]), "entries",
               PROV.MEASURED_LOCAL,
               source=("engine/electrode_sorting.py::per_neuron_gain_matrix on the "
                       "access map built at pitch 20 um / diameter 7 um"),
               note=("area-averaged potentials of REAL soma positions from "
                     "engine.electrode.transfer_mV_per_nA; NO second field solver"))
    led.record("forward.pairs_inside_solver_sphere",
               int(gain_info["n_pairs_inside_solver_regularisation_sphere"]), "pairs",
               PROV.MEASURED_LOCAL, source="this run",
               note=("(channel, neuron) pairs whose patch quadrature points fall inside "
                     "the reused solver's own 1 um source sphere, where the field is "
                     "the bounded sphere value and not a 1/r law. Counted, not hidden: "
                     "these are the approximation floor of the forward model."))
    led.record("population.captured_neurons", int(n_neurons_captured), "neurons",
               PROV.MEASURED_LOCAL, source="access_map.neuron_n_channels > 0",
               note="neurons with at least one channel sphere containing the soma point")
    led.record("population.shared_neurons", int(n_shared), "neurons",
               PROV.MEASURED_LOCAL, source="access_map.neuron_n_channels > 1",
               note=("neurons picked up by MORE THAN ONE channel: the population this "
                     "module exists for. At pitch 20 um this is the majority state."))

    # ---- the methods ------------------------------------------------------
    led.record("sorter.primary_method", "omp_matched_filter", "enum",
               PROV.ENGINEERING_DEFAULT,
               source="declared in run_electrode_sorting.py BEFORE the run",
               note=("greedy sparse least-squares (orthogonal matching pursuit) against "
                     "geometry-derived per-neuron channel templates, because the forward "
                     "map is LINEAR and KNOWN. It is an ORACLE template matcher and NOT "
                     "a validated spike sorter."))
    led.record("sorter.detection_threshold_far", float(far_per_test), "probability",
               PROV.ENGINEERING_DEFAULT,
               source="declared in run_electrode_sorting.py BEFORE the run",
               note=("per-test false-alarm probability behind the Bonferroni threshold; "
                     "NEVER retuned after a result was seen. The normal-tail threshold "
                     "formula is an asymptotic approximation and is swept."))
    led.record("sorter.templates_are_oracle", True, "bool", PROV.ENGINEERING_DEFAULT,
               source="this module's design",
               note=("the inversion uses the same forward matrix that generated the "
                     "data: a BEST CASE. A separate arm of the runner estimates "
                     "templates from a calibration recording to measure the cost of "
                     "not being handed the answer."))

    # ---- the sweep is what makes the ASSUMED entries earned ---------------
    led.record("sweep.noise_levels_uV", [float(v) for v in sweep_noise_levels_uV],
               "uV RMS", PROV.ENGINEERING_DEFAULT,
               source="this module's own noise sweep, run in run_electrode_sorting.py",
               note=("the noise level is the SWEPT variable and the signal amplitude is "
                     "held fixed, because the noise (legacy diagnostic 24.02 uV, "
                     "output-referred value computed in the runner) against a ~0.37 uV median "
                     "neural signal is the binding constraint. Since the chain is "
                     "LINEAR this is exactly equivalent to sweeping the source "
                     "amplitude in the other direction."))
    led.record("access_map.source_json", access_map_json_source, "path",
               PROV.MEASURED_LOCAL, source=access_map_json_source,
               note="the access map artefact this forward matrix was built on")
    return led
