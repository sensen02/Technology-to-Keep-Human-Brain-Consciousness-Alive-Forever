"""Round 5 -- the FIRST quantitative comparison of this project's dopamine
release/clearance model against REAL published experimental values.

WHAT IS COMPARED
----------------
Measured points: Figure 5 panel G (central complex) and Figure 6 panel G
(mushroom body heel) of Dumitrescu, Copeland & Venton, ACS Chem Neurosci,
PMC9897283. The values live in
``outputs/brain_isolation/da_figure5_digitised.json`` and are VISUAL READINGS
of a 750x653 px figure image, NOT raw experimental data (that file says so
itself, in its ``what_this_is`` / ``reading_method`` fields). Reading
uncertainty is ~0.03 uM; the published SEM (n = 10 brains) is wider.

Model: ``engine.da_protocol`` -- an ILLUSTRATIVE finite releasable pool with
first-order refilling, a per-stimulus release fraction driven by the MEASURED
0.2 pmol ACh bolus, and first-order clearance. Except for the protocol timing
(6 stimulations 600 s apart, bolus 5 s into a 60 s record, 1200 s post-
dissection rest, 0.2 pmol ACh per stimulus) NOTHING in that model is measured.
The engine is reused unmodified; this script only drives it.

SUBSTANCE IDENTITY
------------------
Acetylcholine (ACh) is the STIMULUS; dopamine (DA) is the transmitter whose
evoked release is modelled and detected. The 0.2 pmol ACh dose is never treated
as a dopamine amount; the ACh-receptor -> DA-release chain is not modelled.

WHAT THIS SCRIPT DOES
---------------------
1. Loads and validates the digitised readings (two regions, 4 groups x 6
   stimulations) including their provenance and honesty fields.
2. Builds the measured protocol through ``engine.da_protocol.ProtocolSpec``.
3. Runs a BOUNDED, DETERMINISTIC, exhaustive grid search over the model's
   illustrative degrees of freedom with one SHARED kinetic set and only
   group-specific pools/refilling (the "aging hypothesis"):
     shared: stimulus_efficacy, release_tau_s, clearance_tau_s (voxel volume
       held at the illustrative engine default -- efficacy and volume are
       degenerate, only efficacy/volume is identifiable);
     group-specific: pool_capacity_pmol (R0) and refill_tau_s.
   The per-stimulus release fraction ``phi = efficacy * ach_dose / R0`` is
   covered by the (efficacy, R0) product grid bijectively for the fixed
   measured dose.
4. Reports per group and per stimulation: model value, measured value,
   residual [uM], RMS per group, compared BOTH with the ~0.03 uM reading
   uncertainty and with the published SEM, and lists every residual that
   exceeds either.
5. THE CRITICAL TEST: can any configuration that keeps the shared kinetics
   reproduce a group whose FIRST stimulation is elevated relative to the
   others (aged control in the central complex, 0.57 vs 0.41 uM; young parkin
   in the mushroom body heel, 0.71 vs 0.52 uM)?
6. The cross-region test: can ONE shared kinetic set with only pool/refill
   changes explain BOTH regions?
7. Non-identifiability: the parameter combinations that give the same curve
   shape within the reading uncertainty / within a near-optimal band.
8. Writes outputs/brain_isolation/da_real_comparison.png, _report.json and
   _traces.npz, and states measured wall clock and peak RAM.

OPTIONAL, EXPLICITLY LABELLED HYPOTHESES (never the primary result)
------------------------------------------------------------------
Arm A is the primary, constraint-respecting search. Two extra arms are reported
separately, each labelled as an INVENTED additional mechanism with its own extra
degree of freedom:
  * arm B "H1 increased initial release": per-group multiplier on the
    stimulus->release efficacy;
  * arm C "H2 reduced clearance": per-group multiplier on the clearance time
    constant (the source text reports t1/2 NOT significantly different between
    groups, which argues against this arm; that t1/2 comparison is not part of
    the digitised numbers used here).
Neither arm is used to claim a mechanism.

HONESTY RULES (enforced in the written report)
----------------------------------------------
* the measured points are FIGURE READINGS, not raw data -- stated on the figure
  itself and in every report block;
* a good fit does NOT identify a mechanism;
* every residual exceeding the reading uncertainty and/or the published SEM is
  listed explicitly, not averaged away;
* no consciousness, viability or medical claim; ACh is the stimulus and DA is
  the modelled transmitter;
* all model kinetics stay illustrative except the measured protocol timing.

Run:   venv/bin/python run_da_real_comparison.py
Tests: venv/bin/python run_da_real_comparison_selftest.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import re
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:            # allow running from any directory
    sys.path.insert(0, str(ROOT))

from engine.da_protocol import (  # noqa: E402  (path setup must come first)
    DAKinetics,
    ProtocolSpec,
    curve_features,
    release_sequence,
    simulate,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DIGITISED_JSON = ROOT / "outputs" / "brain_isolation" / "da_figure5_digitised.json"
OUT_DIR = ROOT / "outputs" / "brain_isolation"
PNG_PATH = OUT_DIR / "da_real_comparison.png"
REPORT_PATH = OUT_DIR / "da_real_comparison_report.json"
TRACES_PATH = OUT_DIR / "da_real_comparison_traces.npz"

REGION_KEYS = {
    "central_complex": "figure_5G_absolute_uM",
    "mushroom_body_heel": "figure_6G_absolute_uM_mushroom_body_heel",
}
REGION_LABEL = {
    "central_complex": "central complex (Fig. 5G)",
    "mushroom_body_heel": "mushroom body heel (Fig. 6G)",
}
GROUPS = ("control_1", "parkin_1", "control_45", "parkin_45")
GROUP_LABEL = {
    "control_1": "control 1 d",
    "parkin_1": "parkin 1 d",
    "control_45": "control 45 d",
    "parkin_45": "parkin 45 d",
}
GROUP_COLOUR = {
    "control_1": "#1f77b4",     # blue
    "parkin_1": "#d62728",      # red
    "control_45": "#2ca02c",    # green
    "parkin_45": "#ff7f0e",     # orange
}
GROUP_MARKER = {
    "control_1": "o",
    "parkin_1": "s",
    "control_45": "^",
    "parkin_45": "D",
}
N_STIMULATIONS = 6
READING_UNCERTAINTY_UM = 0.03
DEFAULT_SEED = 0
# illustrative engine default, held FIXED because only efficacy/volume is
# identifiable from absolute amplitudes
VOXEL_VOLUME_UM3 = 1e8
NM_TO_UM = 1e-3
ENGINE_AGREEMENT_TOLERANCE_UM = 1e-9
# hard cap that makes the search provably bounded (raises instead of running away)
MAX_MODEL_CURVE_EVALUATIONS = 20_000_000

ARMS = ("A", "B", "C")
ARM_DESCRIPTION = {
    "A": ("PRIMARY: one shared kinetic set (efficacy, release_tau, clearance_tau) "
          "and group-specific releasable pool R0 / refilling tau_refill only -- "
          "the aging hypothesis"),
    "B": ("HYPOTHESIS H1 (extra, invented degree of freedom): shared release and "
          "clearance time constants, group-specific pool/refill AND a group-specific "
          "multiplier on the stimulus->release efficacy -- 'increased initial release "
          "in aged tissue'. Not a measured mechanism; reported separately"),
    "C": ("HYPOTHESIS H2 (extra, invented degree of freedom): shared efficacy and "
          "release time constant, group-specific pool/refill AND a group-specific "
          "multiplier on the clearance time constant -- 'reduced clearance in aged "
          "tissue'. This shifts each group's t1/2, which the source text reports as "
          "NOT significantly different between groups. Not a measured mechanism"),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _jsonable(obj):
    """Recursively convert numpy / Path values into JSON-safe ones."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return _jsonable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _require(cond, message):
    if not cond:
        raise ValueError(message)


def _finite(value, name, minimum=None, exclusive=False):
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
    return x


def _geomspace(lo, hi, n):
    return tuple(float(v) for v in np.geomspace(lo, hi, int(n)))


def peak_memory_mb():
    """Peak RSS of this process in MiB (ru_maxrss is KiB on Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


# ---------------------------------------------------------------------------
# 1. digitised data: load + validate
# ---------------------------------------------------------------------------
@dataclass
class RegionData:
    name: str
    json_key: str
    measured_uM: dict
    sem_uM: dict
    stimulation_number: tuple
    note: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "json_key": self.json_key,
            "label": REGION_LABEL[self.name],
            "stimulation_number": list(self.stimulation_number),
            "measured_uM": {g: [float(x) for x in self.measured_uM[g]] for g in GROUPS},
            "sem_uM": {g: (None if self.sem_uM[g] is None
                           else [float(x) for x in self.sem_uM[g]]) for g in GROUPS},
            "sem_available": {g: self.sem_uM[g] is not None for g in GROUPS},
            "mean_over_stimulations_uM": {g: float(np.mean(self.measured_uM[g]))
                                          for g in GROUPS},
            "note": self.note,
        }


@dataclass
class DigitisedData:
    path: str
    citation: str
    pmcid: str
    figure: str
    image_urls: list
    reading_uncertainty_uM: float
    reading_method: dict
    regions: dict
    honesty: list
    measured_protocol_json: dict

    def to_dict(self):
        return {
            "path": self.path,
            "citation": self.citation,
            "pmcid": self.pmcid,
            "figure": self.figure,
            "image_urls": self.image_urls,
            "reading_uncertainty_uM": self.reading_uncertainty_uM,
            "reading_method": self.reading_method,
            "provenance_and_honesty_fields_from_the_data_file": self.honesty,
            "measured_protocol_as_recorded_in_the_data_file": self.measured_protocol_json,
            "regions": {name: r.to_dict() for name, r in self.regions.items()},
        }


def load_digitised_data(path=DIGITISED_JSON):
    """Load the figure-read values and validate their structure.

    Raises ValueError on malformed input: missing region/group, wrong number of
    stimulations, non-finite or negative value, missing/negative uncertainty,
    SEM of the wrong length, or a data file that no longer states the values are
    figure readings.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"digitised data file does not exist: {path}")
    with path.open() as fh:
        raw = json.load(fh)
    _require(isinstance(raw, dict), "digitised data must be a JSON object")

    source = raw.get("source")
    _require(isinstance(source, dict), "missing 'source' block")
    reading = raw.get("reading_method")
    _require(isinstance(reading, dict), "missing 'reading_method' block")
    unc = _finite(reading.get("uncertainty_uM"), "reading_method.uncertainty_uM",
                  minimum=0.0, exclusive=True)
    _require(abs(unc - READING_UNCERTAINTY_UM) < 1e-12,
             f"reading uncertainty changed: file says {unc}, code assumes "
             f"{READING_UNCERTAINTY_UM}")

    regions = {}
    for name, key in REGION_KEYS.items():
        block = raw.get(key)
        _require(isinstance(block, dict), f"missing region block {key!r}")
        groups = block.get("groups")
        _require(isinstance(groups, dict), f"{key}: missing 'groups'")
        stim = block.get("stimulation_number")
        _require(isinstance(stim, list) and len(stim) == N_STIMULATIONS,
                 f"{key}: 'stimulation_number' must list {N_STIMULATIONS} values")
        _require([int(s) for s in stim] == list(range(1, N_STIMULATIONS + 1)),
                 f"{key}: stimulations must be 1..{N_STIMULATIONS}")
        missing = [g for g in GROUPS if g not in groups]
        _require(not missing, f"{key}: missing groups {missing}")
        extra = [g for g in groups if g not in GROUPS]
        _require(not extra, f"{key}: unexpected groups {extra}")
        measured = {}
        for g in GROUPS:
            vals = groups[g]
            _require(isinstance(vals, list) and len(vals) == N_STIMULATIONS,
                     f"{key}/{g}: expected {N_STIMULATIONS} values")
            measured[g] = np.array([_finite(v, f"{key}/{g}", minimum=0.0)
                                    for v in vals])
        sem = {g: None for g in GROUPS}
        sem_block = block.get("sem_group_readings_approx")
        if sem_block is not None:
            _require(isinstance(sem_block, dict), f"{key}: SEM block must be an object")
            for g in GROUPS:
                if sem_block.get(g) is None:
                    continue
                vals = sem_block[g]
                _require(isinstance(vals, list) and len(vals) == N_STIMULATIONS,
                         f"{key}/{g}: SEM needs {N_STIMULATIONS} values")
                sem[g] = np.array([_finite(v, f"{key}/{g}/sem", minimum=0.0)
                                   for v in vals])
        regions[name] = RegionData(
            name=name, json_key=key, measured_uM=measured, sem_uM=sem,
            stimulation_number=tuple(int(s) for s in stim),
            note=str(block.get("note", "")))
    _require(len(regions) == len(REGION_KEYS), "not every region was loaded")

    honesty = []
    for key in ("what_this_is", "reading_method", "key_measured_patterns",
                "how_this_may_be_used"):
        val = raw.get(key)
        if isinstance(val, str):
            honesty.append(val)
        elif isinstance(val, dict):
            honesty.extend(f"{k}: {v}" for k, v in val.items())
        elif isinstance(val, list):
            honesty.extend(str(v) for v in val)
    what_this_is = str(raw.get("what_this_is", ""))
    _require(("VISUAL READINGS" in what_this_is) or ("NOT raw" in what_this_is),
             "'what_this_is' no longer states that the values are VISUAL READINGS "
             "of a figure and NOT raw experimental data")

    image_urls = [source.get("image_url")]
    for key in REGION_KEYS.values():
        m = re.search(r"https?://\S+", str(raw.get(key, {}).get("note", "")))
        if m:
            image_urls.append(m.group(0))
    image_urls = [u for u in image_urls if u]

    return DigitisedData(
        path=str(path),
        citation=str(source.get("citation", "")),
        pmcid=str(source.get("pmcid", "")),
        figure=str(source.get("figure", "")),
        image_urls=image_urls,
        reading_uncertainty_uM=unc,
        reading_method=reading,
        regions=regions,
        honesty=honesty,
        measured_protocol_json=raw.get("measured_protocol", {}))


# ---------------------------------------------------------------------------
# 2. measured protocol through the engine
# ---------------------------------------------------------------------------
def protocol_from_digitised(data, tol=1e-9):
    """Parse the measured protocol out of the JSON and cross-check the engine.

    The JSON records the stimulus as text; its numbers are parsed and asserted
    equal to ``engine.da_protocol.ProtocolSpec``'s measured defaults, so drift in
    either place fails loudly instead of silently changing the timeline.
    """
    block = dict(data.measured_protocol_json)
    _require(block, "digitised data has no 'measured_protocol' block")
    stim_text = str(block.get("stimulus", ""))
    dose_match = re.search(r"([0-9.]+)\s*pmol", stim_text)
    _require(dose_match is not None,
             f"cannot parse the ACh dose from stimulus text {stim_text!r}")
    dose = float(dose_match.group(1))
    offset_match = re.search(r"([0-9.]+)\s*s into", stim_text)
    offset = float(offset_match.group(1)) if offset_match else None

    p = ProtocolSpec()
    checks = {
        "n_stimulations": (block.get("stimulations"), p.n_stimulations),
        "ach_dose_pmol": (dose, p.ach_dose_pmol),
        "inter_stimulus_interval_s": (block.get("interval_s"),
                                      p.inter_stimulus_interval_s),
        "recording_duration_s": (block.get("record_s"), p.recording_duration_s),
        "representative_trace_s": (block.get("representative_trace_s"),
                                   p.representative_trace_s),
        "post_dissection_rest_s": (block.get("post_dissection_rest_s"),
                                   p.post_dissection_rest_s),
    }
    if offset is not None:
        checks["pre_bolus_baseline_s"] = (offset, p.pre_bolus_baseline_s)
    for name, (json_val, engine_val) in checks.items():
        _require(json_val is not None,
                 f"measured_protocol.{name} missing from the data file")
        jv = _finite(json_val, f"measured_protocol.{name}", minimum=0.0)
        _require(abs(jv - float(engine_val)) <= tol,
                 f"measured protocol mismatch for {name}: data file says {jv}, "
                 f"engine ProtocolSpec says {engine_val}")
    return p


def protocol_report(p):
    rep = p.timing_report()
    rep["source_of_timing"] = (
        "measured, adopted verbatim; parsed from the digitised data file and "
        "cross-checked against engine.da_protocol.ProtocolSpec")
    rep["ach_is_the_stimulus_da_is_the_modelled_transmitter"] = True
    return rep


# ---------------------------------------------------------------------------
# 3. the model's across-stimulation shape
# ---------------------------------------------------------------------------
_PEAKSHAPE_CACHE = {}


def peak_shape_factor(release_tau_s, clearance_tau_s):
    """Peak of the ENGINE's own release/clearance shape for a unit release.

    Computed with ``engine.da_protocol.curve_features`` (not re-derived): the peak
    concentration produced by a uniform equivalent release of 1 nM with zero
    pre-existing concentration, the bolus sitting ``pre_bolus_baseline_s`` into
    the record. The amplitude is linear in the released amount, so this one
    scalar carries the entire shape effect of (release_tau_s, clearance_tau_s)
    on the peak.
    """
    tau_r = _finite(release_tau_s, "release_tau_s", minimum=0.0)
    tau_c = _finite(clearance_tau_s, "clearance_tau_s", minimum=0.0, exclusive=True)
    key = (tau_r, tau_c)
    if key not in _PEAKSHAPE_CACHE:
        peak, _tpk, _habs, _thalf = curve_features(
            0.0, 1.0, tau_r, 1.0 / tau_c,
            pre_s=ProtocolSpec().pre_bolus_baseline_s, c_start_nM=0.0)
        _PEAKSHAPE_CACHE[key] = float(peak)
    return _PEAKSHAPE_CACHE[key]


def pool_release_sequence(R0, refill_tau_s, efficacy, protocol):
    """Per-stimulus released DA [pmol] from the engine's documented recurrence.

    R_next = R0 + (R - A - R0) * exp(-ISI / tau_refill) and
    A = min(efficacy * dose * (R / R0), R).  Re-deriving these lines here (exactly
    as documented in engine/da_protocol.py) lets the exhaustive search run ~1000x
    faster than calling the engine's feature machinery per candidate;
    ``self_test_against_engine`` proves equality with the engine on random
    configurations, and the selftest asserts it.
    """
    R0 = _finite(R0, "pool_capacity_pmol", minimum=0.0, exclusive=True)
    tau = _finite(refill_tau_s, "refill_tau_s", minimum=0.0, exclusive=True)
    eff = _finite(efficacy, "stimulus_efficacy", minimum=0.0)
    n = int(protocol.n_stimulations)
    g = math.exp(-protocol.inter_stimulus_interval_s / tau)
    frac = min(eff * protocol.ach_dose_pmol, R0) / R0
    x = 1.0
    out = np.empty(n)
    for i in range(n):
        out[i] = frac * x * R0
        x = 1.0 + (x - frac * x - 1.0) * g
    return out


def subgrid_peaks_uM(R0_values, refill_values, efficacy, peak_shape, protocol,
                     volume_um3=VOXEL_VOLUME_UM3):
    """Vectorised peak [uM] over a (R0, tau_refill) grid.

    Returns an array of shape (len(R0_values), len(refill_values), n_stim).
    """
    R0 = np.asarray(R0_values, dtype=float)
    tau = np.asarray(refill_values, dtype=float)
    _require(R0.size >= 1 and np.all(R0 > 0.0) and np.all(np.isfinite(R0)),
             "pool_capacity_pmol grid must be finite and > 0")
    _require(tau.size >= 1 and np.all(tau > 0.0) and np.all(np.isfinite(tau)),
             "refill_tau_s grid must be finite and > 0")
    eff = _finite(efficacy, "efficacy", minimum=0.0)
    n = int(protocol.n_stimulations)
    g = np.exp(-protocol.inter_stimulus_interval_s / tau)[None, :]
    R0c = R0[:, None]
    frac = np.minimum(eff * protocol.ach_dose_pmol, R0c) / R0c
    x = np.ones((R0.size, tau.size))
    out = np.empty((R0.size, tau.size, n))
    for i in range(n):
        out[:, :, i] = frac * x * R0c * (1e9 / volume_um3) * peak_shape
        x = 1.0 + (x - frac * x - 1.0) * g
    return out


# ---------------------------------------------------------------------------
# 4. parameter containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Bounded, explicitly enumerated search grid (all axis values > 0).

    ``efficacy`` / ``release_tau_s`` / ``clearance_tau_s`` are the SHARED axes;
    ``pool_capacity_pmol`` / ``refill_tau_s`` are per-group axes; the two
    multiplier axes are used only by the labelled hypothesis arms B and C.
    """

    efficacy: tuple
    release_tau_s: tuple
    clearance_tau_s: tuple
    pool_capacity_pmol: tuple
    refill_tau_s: tuple
    efficacy_multiplier: tuple = (1.0,)
    clearance_multiplier: tuple = (1.0,)
    voxel_volume_um3: float = VOXEL_VOLUME_UM3
    bounds: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("efficacy", "release_tau_s", "clearance_tau_s",
                     "pool_capacity_pmol", "refill_tau_s",
                     "efficacy_multiplier", "clearance_multiplier"):
            vals = tuple(getattr(self, name))
            _require(len(vals) >= 1, f"grid axis {name} must not be empty")
            clean = [_finite(v, f"{name} axis value", minimum=0.0,
                             exclusive=(name != "efficacy")) for v in vals]
            _require(len(set(clean)) == len(clean),
                     f"grid axis {name} contains duplicate values")
            object.__setattr__(self, name, tuple(clean))
        _finite(self.voxel_volume_um3, "voxel_volume_um3", minimum=0.0, exclusive=True)
        if not self.bounds:
            object.__setattr__(self, "bounds", {
                name: [min(getattr(self, name)), max(getattr(self, name))]
                for name in ("efficacy", "release_tau_s", "clearance_tau_s",
                             "pool_capacity_pmol", "refill_tau_s",
                             "efficacy_multiplier", "clearance_multiplier")})

    # -- enumeration --------------------------------------------------------
    def shared_points(self, arm):
        _require(arm in ARMS, f"unknown arm {arm!r}")
        axes, names = self._shared_axes(arm)
        for vals in itertools.product(*axes):
            yield dict(zip(names, vals))

    def _shared_axes(self, arm):
        """All three kinetic axes are shared by every arm; arms B and C add a
        per-group multiplier on top (labelled hypotheses), they do not replace a
        shared axis, so arms B/C are strictly more expressive than arm A."""
        _require(arm in ARMS, f"unknown arm {arm!r}")
        return ((self.efficacy, self.release_tau_s, self.clearance_tau_s),
                ("efficacy", "release_tau_s", "clearance_tau_s"))

    def _subgrid_axes(self, arm):
        if arm == "A":
            return ((self.pool_capacity_pmol, self.refill_tau_s),
                    ("pool_capacity_pmol", "refill_tau_s"))
        if arm == "B":
            return ((self.pool_capacity_pmol, self.refill_tau_s,
                     self.efficacy_multiplier),
                    ("pool_capacity_pmol", "refill_tau_s", "efficacy_multiplier"))
        return ((self.pool_capacity_pmol, self.refill_tau_s,
                 self.clearance_multiplier),
                ("pool_capacity_pmol", "refill_tau_s", "clearance_multiplier"))

    def subgrid_points(self, arm):
        axes, names = self._subgrid_axes(arm)
        for vals in itertools.product(*axes):
            yield dict(zip(names, vals))

    # -- counts -------------------------------------------------------------
    def axis_lengths(self):
        return {name: len(getattr(self, name)) for name in
                ("efficacy", "release_tau_s", "clearance_tau_s",
                 "pool_capacity_pmol", "refill_tau_s", "efficacy_multiplier",
                 "clearance_multiplier")}

    def n_shared_points(self, arm):
        axes, _ = self._shared_axes(arm)
        return int(np.prod([len(a) for a in axes]))

    def n_subgrid_points(self, arm):
        axes, _ = self._subgrid_axes(arm)
        return int(np.prod([len(a) for a in axes]))

    def n_model_curve_evaluations(self, arm, n_groups=len(GROUPS)):
        return self.n_shared_points(arm) * self.n_subgrid_points(arm) * int(n_groups)

    def to_dict(self, arm=None):
        out = {
            "axes": self.axis_lengths(),
            "axis_values": {name: list(getattr(self, name)) for name in
                            ("efficacy", "release_tau_s", "clearance_tau_s",
                             "pool_capacity_pmol", "refill_tau_s",
                             "efficacy_multiplier", "clearance_multiplier")},
            "bounds": _jsonable(self.bounds),
            "voxel_volume_um3": self.voxel_volume_um3,
            "notes": [
                "bounded: every axis is an explicit finite tuple inside declared "
                "bounds; nothing is drawn, sampled or unbounded",
                "per-stimulus release fraction phi = efficacy * ach_dose / R0 is "
                "covered by the (efficacy, R0) product grid, bijectively for the "
                "fixed measured dose",
                "efficacy and voxel_volume_um3 enter only as efficacy/volume, so "
                "the volume is held at the illustrative engine default and the "
                "pair is reported as non-identifiable",
                "the objective separates across groups once the shared kinetic "
                "point (and any per-region amplitude scale) is fixed, so the "
                "per-group minimum is taken independently: the reported optimum "
                "is the exact minimum over the full product grid",
            ],
        }
        if arm is not None:
            out.update({"arm": arm, "n_shared_points": self.n_shared_points(arm),
                        "n_subgrid_points_per_group": self.n_subgrid_points(arm),
                        "n_model_curve_evaluations":
                            self.n_model_curve_evaluations(arm)})
        return out


def default_grid(fine=True):
    """The bounded search grid used for the reported fit."""
    if fine:
        eff = _geomspace(0.02, 1.0, 12)
        tau_rel = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0)
        tau_clr = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
        R0 = _geomspace(0.008, 5.0, 20)
        refill = _geomspace(15.0, 20000.0, 14)
        emult = _geomspace(0.25, 4.0, 9)
        cmult = _geomspace(0.25, 4.0, 9)
    else:
        eff = _geomspace(0.02, 1.0, 6)
        tau_rel = (0.2, 0.5, 1.0, 2.0)
        tau_clr = (1.0, 2.0, 5.0, 10.0)
        R0 = _geomspace(0.01, 5.0, 10)
        refill = _geomspace(30.0, 20000.0, 8)
        emult = _geomspace(0.3, 3.0, 5)
        cmult = _geomspace(0.3, 3.0, 5)
    return Grid(efficacy=eff, release_tau_s=tau_rel, clearance_tau_s=tau_clr,
                pool_capacity_pmol=R0, refill_tau_s=refill,
                efficacy_multiplier=emult, clearance_multiplier=cmult)


@dataclass
class ModelParams:
    """One concrete point of the searched parameter space."""

    efficacy: float
    release_tau_s: float
    clearance_tau_s: float
    pool_capacity_pmol: dict
    refill_tau_s: dict
    efficacy_multiplier: dict = field(default_factory=lambda: {g: 1.0 for g in GROUPS})
    clearance_multiplier: dict = field(default_factory=lambda: {g: 1.0 for g in GROUPS})
    voxel_volume_um3: float = VOXEL_VOLUME_UM3
    release_shape_coupling: float = 0.0        # engine default; never searched

    def copy(self):
        return ModelParams(
            efficacy=self.efficacy, release_tau_s=self.release_tau_s,
            clearance_tau_s=self.clearance_tau_s,
            pool_capacity_pmol=dict(self.pool_capacity_pmol),
            refill_tau_s=dict(self.refill_tau_s),
            efficacy_multiplier=dict(self.efficacy_multiplier),
            clearance_multiplier=dict(self.clearance_multiplier),
            voxel_volume_um3=self.voxel_volume_um3,
            release_shape_coupling=self.release_shape_coupling)

    def group_value(self, axis, group):
        return getattr(self, axis)[group] if group is not None else getattr(self, axis)

    def with_field(self, axis, value, group=None):
        new = self.copy()
        if group is None:
            _require(axis in ("efficacy", "release_tau_s", "clearance_tau_s"),
                     f"{axis!r} is not a shared axis")
            setattr(new, axis, float(value))
        else:
            _require(axis in ("pool_capacity_pmol", "refill_tau_s",
                              "efficacy_multiplier", "clearance_multiplier"),
                     f"{axis!r} is not a per-group axis")
            getattr(new, axis)[group] = float(value)
        return new

    def release_fraction_phi(self, group, protocol):
        return (self.efficacy * self.efficacy_multiplier[group]
                * protocol.ach_dose_pmol / self.pool_capacity_pmol[group])

    def group_kinetics(self, group):
        return DAKinetics(
            pool_capacity_pmol=self.pool_capacity_pmol[group],
            refill_tau_s=self.refill_tau_s[group],
            stimulus_efficacy=self.efficacy * self.efficacy_multiplier[group],
            release_tau_s=self.release_tau_s,
            clearance_tau_s=self.clearance_tau_s * self.clearance_multiplier[group],
            voxel_volume_um3=self.voxel_volume_um3,
            release_shape_coupling=self.release_shape_coupling,
            noise_std_nM=0.0,
            label="fitted-illustrative")

    def predicted_uM(self, protocol, scale=1.0):
        """Peak [uM] per group per stimulation (6 values per group)."""
        out = {}
        for g in GROUPS:
            tau_clr = self.clearance_tau_s * self.clearance_multiplier[g]
            ps = peak_shape_factor(self.release_tau_s, tau_clr)
            rel = pool_release_sequence(
                self.pool_capacity_pmol[g], self.refill_tau_s[g],
                self.efficacy * self.efficacy_multiplier[g], protocol)
            out[g] = rel * (1e9 / self.voxel_volume_um3) * ps * float(scale)
        return out

    def engine_predicted_uM(self, protocol, group, scale=1.0):
        """Same prediction through the ENGINE's own release_sequence()."""
        res = release_sequence(protocol, self.group_kinetics(group))
        return np.asarray(res["ephemeral"]["peak_nM"], dtype=float) * NM_TO_UM * float(scale)

    def to_dict(self, protocol=None):
        return {
            "efficacy_pmol_DA_per_pmol_ACh": self.efficacy,
            "release_tau_s": self.release_tau_s,
            "clearance_tau_s": self.clearance_tau_s,
            "clearance_half_life_s": math.log(2.0) * self.clearance_tau_s,
            "voxel_volume_um3": self.voxel_volume_um3,
            "release_shape_coupling": self.release_shape_coupling,
            "pool_capacity_pmol_per_group": dict(self.pool_capacity_pmol),
            "refill_tau_s_per_group": dict(self.refill_tau_s),
            "efficacy_multiplier_per_group": dict(self.efficacy_multiplier),
            "clearance_multiplier_per_group": dict(self.clearance_multiplier),
            "tau_refill_over_measured_ISI_600s": {
                g: self.refill_tau_s[g] / protocol.inter_stimulus_interval_s
                for g in GROUPS} if protocol is not None else None,
            "release_fraction_phi_per_group": (
                {g: self.release_fraction_phi(g, protocol) for g in GROUPS}
                if protocol is not None else None),
            "all_kinetics_illustrative": True,
        }


# ---------------------------------------------------------------------------
# 5. fitting machinery (bounded, deterministic, exhaustive over the grid)
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    arm: str
    regions: tuple
    params: ModelParams
    scales: dict
    residuals_uM: dict
    measured_uM: dict
    model_uM: dict
    sem_uM: dict
    rms_per_group_uM: dict
    rms_overall_uM: float
    max_abs_residual_uM: float
    n_model_curve_evaluations: int
    n_objective_evaluations: int
    n_sse_combinations: int
    seed: int
    wall_clock_s: float
    grid_axes: dict
    label: str = ""
    protocol: ProtocolSpec = field(default_factory=ProtocolSpec)
    exclude_capping: bool = False

    def to_dict(self):
        return {
            "arm": self.arm,
            "arm_description": ARM_DESCRIPTION[self.arm],
            "label": self.label,
            "regions_fitted": list(self.regions),
            "fitted_parameters": self.params.to_dict(self.protocol),
            "region_amplitude_scales": dict(self.scales),
            "rms_per_group_uM": self.rms_per_group_uM,
            "rms_overall_uM": self.rms_overall_uM,
            "max_abs_residual_uM": self.max_abs_residual_uM,
            "reading_uncertainty_uM": READING_UNCERTAINTY_UM,
            "rms_over_reading_uncertainty": self.rms_overall_uM / READING_UNCERTAINTY_UM,
            "n_model_curve_evaluations": self.n_model_curve_evaluations,
            "n_objective_evaluations": self.n_objective_evaluations,
            "n_sse_combinations": self.n_sse_combinations,
            "seed": self.seed,
            "wall_clock_s": self.wall_clock_s,
            "grid_axes": self.grid_axes,
            "pool_capping_excluded": self.exclude_capping,
            "exceedance_counts": self.exceedance_counts(),
            "per_group_per_stimulation": _jsonable(self.per_stimulus_table()),
        }

    def per_stimulus_table(self):
        table = {}
        for region in self.regions:
            table[region] = {}
            for g in GROUPS:
                rows = []
                for i in range(N_STIMULATIONS):
                    r = float(self.residuals_uM[region][g][i])
                    sem = self.sem_uM[region].get(g)
                    semv = None if sem is None else float(sem[i])
                    rows.append({
                        "stimulation": i + 1,
                        "model_uM": float(self.model_uM[region][g][i]),
                        "measured_uM": float(self.measured_uM[region][g][i]),
                        "residual_uM": r,
                        "abs_residual_uM": abs(r),
                        "published_sem_uM": semv,
                        "exceeds_reading_uncertainty":
                            bool(abs(r) > READING_UNCERTAINTY_UM),
                        "exceeds_published_sem": (None if semv is None
                                                  else bool(abs(r) > semv)),
                    })
                table[region][g] = rows
        return table

    def exceedance_counts(self):
        counts = {"total_points": 0, "exceed_reading_uncertainty": 0,
                  "exceed_published_sem": 0, "points_with_published_sem": 0,
                  "exceed_reading_uncertainty_per_group": {g: 0 for g in GROUPS},
                  "exceed_published_sem_per_group": {g: 0 for g in GROUPS}}
        for region in self.regions:
            for g in GROUPS:
                for i in range(N_STIMULATIONS):
                    r = abs(float(self.residuals_uM[region][g][i]))
                    counts["total_points"] += 1
                    if r > READING_UNCERTAINTY_UM:
                        counts["exceed_reading_uncertainty"] += 1
                        counts["exceed_reading_uncertainty_per_group"][g] += 1
                    sem = self.sem_uM[region].get(g)
                    if sem is not None:
                        counts["points_with_published_sem"] += 1
                        if r > float(sem[i]):
                            counts["exceed_published_sem"] += 1
                            counts["exceed_published_sem_per_group"][g] += 1
        return counts


def _candidate_predictions(grid, arm, shared, protocol, exclude_capping=False):
    """Peak [uM] of shape (n_candidates, 6) per group for one shared point.

    Vectorised: for arm A the sub-grid is (R0, tau_refill); for arms B/C the
    extra multiplier axis is appended as the fastest-varying axis, matching
    ``Grid.subgrid_points``'s itertools.product order.  Also returns a boolean
    "allowed" mask per group: with ``exclude_capping=True`` any candidate whose
    releasable pool is smaller than the requested amount (R0 < efficacy*dose,
    i.e. the pool empties completely at every stimulus) is disallowed, which is
    exactly the restriction that defines the pure pool-shrink / refill-slowdown
    hypothesis.
    """
    cands = list(grid.subgrid_points(arm))
    n = int(protocol.n_stimulations)
    tau_rel = shared["release_tau_s"]
    out, mask = {}, {}
    if arm == "A":
        ps = peak_shape_factor(tau_rel, shared["clearance_tau_s"])
        # efficacy is shared: identical for every group, so the same vectorised
        # sub-grid serves all four groups
        arr = subgrid_peaks_uM(grid.pool_capacity_pmol, grid.refill_tau_s,
                               shared["efficacy"], ps, protocol,
                               grid.voxel_volume_um3)
        flat = arr.reshape(-1, n)
        allowed = np.array([not (exclude_capping
                                 and shared["efficacy"] * protocol.ach_dose_pmol > c["pool_capacity_pmol"])
                            for c in cands])
        for g in GROUPS:
            out[g] = flat
            mask[g] = allowed
        return cands, out, mask
    if arm == "B":      # H1: per-group multiplier on the release efficacy
        ps = peak_shape_factor(tau_rel, shared["clearance_tau_s"])
        for g in GROUPS:
            blocks, allowed = [], []
            for m in grid.efficacy_multiplier:
                eff = shared["efficacy"] * m
                blocks.append(subgrid_peaks_uM(grid.pool_capacity_pmol,
                                               grid.refill_tau_s, eff, ps, protocol,
                                               grid.voxel_volume_um3))
                allowed.append(np.array([not (exclude_capping
                                              and eff * protocol.ach_dose_pmol > c["pool_capacity_pmol"])
                                         for c in cands]))
            out[g] = np.stack(blocks, axis=2).reshape(-1, n)
            mask[g] = np.stack(allowed, axis=1).reshape(-1)
        return cands, out, mask
    for g in GROUPS:    # arm C (H2): per-group multiplier on the clearance tau
        blocks, allowed = [], []
        for m in grid.clearance_multiplier:
            blocks.append(subgrid_peaks_uM(grid.pool_capacity_pmol, grid.refill_tau_s,
                                           shared["efficacy"],
                                           peak_shape_factor(tau_rel,
                                                             shared["clearance_tau_s"] * m),
                                           protocol, grid.voxel_volume_um3))
            allowed.append(np.array([not (exclude_capping and
                                           shared["efficacy"] * protocol.ach_dose_pmol
                                           > c["pool_capacity_pmol"]) for c in cands]))
        out[g] = np.stack(blocks, axis=2).reshape(-1, n)
        mask[g] = np.stack(allowed, axis=1).reshape(-1)
    return cands, out, mask


def _params_from_best(best, grid, arm):
    shared = best["shared"]
    params = ModelParams(
        efficacy=float(shared.get("efficacy", 1.0)),
        release_tau_s=float(shared["release_tau_s"]),
        clearance_tau_s=float(shared["clearance_tau_s"]),
        pool_capacity_pmol={}, refill_tau_s={},
        efficacy_multiplier={g: 1.0 for g in GROUPS},
        clearance_multiplier={g: 1.0 for g in GROUPS},
        voxel_volume_um3=grid.voxel_volume_um3)
    for g in GROUPS:
        c = best["cands"][best["idx"][g]]
        params.pool_capacity_pmol[g] = float(c["pool_capacity_pmol"])
        params.refill_tau_s[g] = float(c["refill_tau_s"])
        params.efficacy_multiplier[g] = float(c.get("efficacy_multiplier", 1.0))
        params.clearance_multiplier[g] = float(c.get("clearance_multiplier", 1.0))
    return params


def _best_index(values, perm):
    """argmin over a seeded, documented enumeration order (tie-breaking only)."""
    return int(perm[int(np.argmin(values[perm]))])


def coarse_search(grid, arm, protocol, region_targets, region_scales,
                  seed=DEFAULT_SEED, max_evaluations=None, exclude_capping=False):
    """Exhaustive search over the factored grid; returns the exact optimum.

    The objective separates across groups once the shared kinetic point and the
    per-region amplitude scale are fixed, so each group's minimum is taken
    independently over its own (R0, tau_refill[, multiplier]) grid: the result is
    the exact minimum of the full product grid, not a local optimum.
    """
    _require(arm in ARMS, f"unknown arm {arm!r}")
    _require(isinstance(region_targets, dict) and region_targets,
             "region_targets must be a non-empty dict")
    regions = tuple(region_targets.keys())
    for r in regions:
        _require(set(region_targets[r].keys()) == set(GROUPS),
                 f"region {r}: targets must name exactly {GROUPS}")
        for g in GROUPS:
            arr = np.asarray(region_targets[r][g], dtype=float)
            _require(arr.shape == (N_STIMULATIONS,),
                     f"region {r}/{g}: target must have {N_STIMULATIONS} values")
            _require(np.all(np.isfinite(arr)),
                     f"region {r}/{g}: target must be finite")
    for r in regions:
        _require(r in region_scales, f"region {r}: no amplitude scale grid given")
        _require(len(region_scales[r]) >= 1, f"region {r}: empty scale grid")
    scale_combos = [dict(zip(regions, vals))
                    for vals in itertools.product(*[region_scales[r] for r in regions])]

    n_shared = grid.n_shared_points(arm)
    n_cand = grid.n_subgrid_points(arm)
    n_curve = n_shared * n_cand * len(GROUPS)
    if max_evaluations is not None and n_curve > int(max_evaluations):
        raise ValueError(
            f"search would perform {n_curve} model-curve evaluations, above the "
            f"cap max_evaluations={max_evaluations}: shrink the grid or raise the cap")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_cand)     # enumeration order; cannot move the minimum
    best = None
    n_sse = 0
    n_obj = 0
    n_shared_done = 0
    for shared in grid.shared_points(arm):
        n_shared_done += 1
        cands, preds, mask = _candidate_predictions(grid, arm, shared, protocol,
                                                    exclude_capping=exclude_capping)
        for scales in scale_combos:
            n_obj += 1
            idx = {}
            total = 0.0
            for g in GROUPS:
                sse = np.zeros(n_cand)
                for r in regions:
                    diff = (float(scales[r]) * preds[g]
                            - np.asarray(region_targets[r][g], dtype=float)[None, :])
                    sse += np.einsum("ij,ij->i", diff, diff)
                    n_sse += 1
                if exclude_capping:
                    sse = np.where(mask[g], sse, np.inf)
                    _require(np.any(np.isfinite(sse)),
                             f"{arm}: no allowed candidate at this shared kinetic "
                             f"point with pool capping excluded")
                i = _best_index(sse, perm)
                idx[g] = i
                total += float(sse[i])
            if best is None or total < best["sse"]:
                best = {"sse": total, "shared": dict(shared), "idx": idx,
                        "scales": dict(scales), "cands": cands}
    _require(best is not None, "search evaluated no configuration")
    return {"best": best, "params": _params_from_best(best, grid, arm),
            "n_model_curve_evaluations": n_curve,
            "n_objective_evaluations": n_obj, "n_sse_combinations": n_sse,
            "n_shared_points_evaluated": n_shared_done,
            "scale_combos": scale_combos, "regions": regions, "arm": arm,
            "grid": grid, "protocol": protocol, "seed": int(seed),
            "exclude_capping": bool(exclude_capping)}


def _axes_for(arm):
    shared = ["efficacy", "release_tau_s", "clearance_tau_s"]
    if arm == "B":
        shared = ["release_tau_s", "clearance_tau_s"]
    elif arm == "C":
        shared = ["efficacy", "release_tau_s"]
    axes = [(a, None) for a in shared]
    for g in GROUPS:
        axes.append(("pool_capacity_pmol", g))
        axes.append(("refill_tau_s", g))
        if arm == "B":
            axes.append(("efficacy_multiplier", g))
        if arm == "C":
            axes.append(("clearance_multiplier", g))
    return axes


def _local_values(current, bounds, factor=1.6, n=9):
    lo, hi = float(bounds[0]), float(bounds[1])
    vals = np.clip(np.geomspace(current / factor, current * factor, int(n)), lo, hi)
    uniq = []
    for v in vals:
        v = float(v)
        if v > 0.0 and all(abs(v - u) > 1e-12 * max(v, u) for u in uniq):
            uniq.append(v)
    if all(abs(current - u) > 1e-12 * max(current, u) for u in uniq):
        uniq.append(float(current))
    return sorted(uniq)


def refine_coordinate_descent(seed_result, region_targets, protocol,
                              passes=3, factor=1.6, n_local=9):
    """Bounded deterministic local refinement (coordinate descent).

    Each axis is swept on a finite geometric local grid clipped to its declared
    bounds; the objective-evaluation count is reported. Run after the exhaustive
    coarse stage, so the fit can only improve or stay the same.
    """
    grid = seed_result["grid"]
    arm = seed_result["arm"]
    regions = seed_result["regions"]
    bounds = grid.bounds
    params = seed_result["params"]
    scales = dict(seed_result["best"]["scales"])
    scales_fixed = {r: len(seed_result["scale_grid"][r]) == 1 for r in regions}

    no_cap = bool(seed_result.get("exclude_capping", False))

    def objective(p, sc):
        preds = p.predicted_uM(protocol)
        total = 0.0
        for r in regions:
            for g in GROUPS:
                if no_cap and (p.efficacy * p.efficacy_multiplier[g]
                               * protocol.ach_dose_pmol > p.pool_capacity_pmol[g]):
                    return float("inf")       # pool-capped candidates are disallowed
                diff = float(sc[r]) * preds[g] - np.asarray(region_targets[r][g],
                                                            dtype=float)
                total += float(np.dot(diff, diff))
        return total

    n_obj = 0
    best_sse = objective(params, scales)
    for _ in range(int(passes)):
        improved = False
        for axis, group in _axes_for(arm):
            current = params.group_value(axis, group)
            for val in _local_values(current, bounds[axis], factor, n_local):
                trial = params.with_field(axis, val, group)
                sse = objective(trial, scales)
                n_obj += 1
                if sse < best_sse - 1e-15:
                    best_sse, params, improved = sse, trial, True
        preds = params.predicted_uM(protocol)
        for r in regions:
            if scales_fixed[r]:
                continue
            num = sum(float(np.dot(preds[g], np.asarray(region_targets[r][g], dtype=float)))
                      for g in GROUPS)
            den = sum(float(np.dot(preds[g], preds[g])) for g in GROUPS)
            if den > 0.0:
                scales[r] = num / den
        sse = objective(params, scales)
        n_obj += 1
        if sse < best_sse:
            best_sse, improved = sse, True
        if not improved:
            break
    return {"params": params, "scales": scales, "sse": best_sse,
            "n_objective_evaluations": n_obj}


def fit(grid=None, arm="A", region_targets=None, region_scales=None,
        protocol=None, seed=DEFAULT_SEED, max_evaluations=MAX_MODEL_CURVE_EVALUATIONS,
        refine=True, label="", exclude_capping=False):
    """Full fit: exhaustive coarse search + bounded local refinement."""
    grid = grid if grid is not None else default_grid(fine=True)
    protocol = protocol if protocol is not None else ProtocolSpec()
    _require(arm in ARMS, f"unknown arm {arm!r}")
    _require(region_targets, "region_targets is required (region -> group -> [uM x 6])")
    regions = tuple(region_targets.keys())
    if region_scales is None:
        region_scales = {r: (1.0,) for r in regions}
    t0 = time.perf_counter()
    coarse = coarse_search(grid, arm, protocol, region_targets, region_scales,
                           seed=seed, max_evaluations=max_evaluations,
                           exclude_capping=exclude_capping)
    coarse["scale_grid"] = {r: tuple(region_scales[r]) for r in regions}
    n_obj = coarse["n_objective_evaluations"]
    params = coarse["params"]
    scales = dict(coarse["best"]["scales"])
    if refine:
        refined = refine_coordinate_descent(coarse, region_targets, protocol)
        n_obj += refined["n_objective_evaluations"]
        params, scales = refined["params"], refined["scales"]
    wall = time.perf_counter() - t0

    model, residuals, rms_g = {}, {}, {}
    for r in regions:
        preds = params.predicted_uM(protocol, scale=float(scales[r]))
        model[r] = {g: preds[g] for g in GROUPS}
        residuals[r] = {g: model[r][g] - np.asarray(region_targets[r][g], dtype=float)
                        for g in GROUPS}
        for g in GROUPS:
            rms_g[f"{r}:{g}"] = float(np.sqrt(np.mean(residuals[r][g] ** 2)))
    all_res = np.concatenate([residuals[r][g] for r in regions for g in GROUPS])
    return FitResult(
        arm=arm, regions=regions, params=params, scales=scales,
        residuals_uM=residuals,
        measured_uM={r: {g: np.asarray(region_targets[r][g], dtype=float)
                         for g in GROUPS} for r in regions},
        model_uM=model, sem_uM={r: {g: None for g in GROUPS} for r in regions},
        rms_per_group_uM=rms_g,
        rms_overall_uM=float(np.sqrt(np.mean(all_res ** 2))),
        max_abs_residual_uM=float(np.max(np.abs(all_res))),
        n_model_curve_evaluations=coarse["n_model_curve_evaluations"],
        n_objective_evaluations=n_obj,
        n_sse_combinations=coarse["n_sse_combinations"],
        seed=int(seed), wall_clock_s=wall, grid_axes=grid.axis_lengths(),
        label=label, protocol=protocol, exclude_capping=bool(exclude_capping))


def attach_sem(fit_result, data):
    """Copy the published SEM readings into a FitResult (region/group aware)."""
    for r in fit_result.regions:
        for g in GROUPS:
            sem = data.regions[r].sem_uM.get(g)
            fit_result.sem_uM[r][g] = None if sem is None else np.asarray(sem, dtype=float)
    return fit_result


# ---------------------------------------------------------------------------
# 6. structural facts: the stimulation-1 ceiling, the critical test
# ---------------------------------------------------------------------------
def stim1_peak_uM(efficacy, R0, release_tau_s, clearance_tau_s, protocol,
                  volume_um3=VOXEL_VOLUME_UM3):
    """Model peak [uM] at stimulation 1 for one group.

    min(efficacy*dose, R0) pmol are released because every group starts with a
    FULL releasable pool and no extracellular DA; R0 therefore enters only as a
    cap that can LOWER the first peak, and tau_refill does not enter at all.
    """
    eff = _finite(efficacy, "efficacy", minimum=0.0)
    R0 = _finite(R0, "pool_capacity_pmol", minimum=0.0, exclusive=True)
    ps = peak_shape_factor(release_tau_s, clearance_tau_s)
    return min(eff * protocol.ach_dose_pmol, R0) * (1e9 / volume_um3) * ps


def critical_test(grid, protocol, data, region="central_complex",
                  elevated_group="control_45", reference_group="control_1"):
    """Can a shared-kinetics configuration reproduce an ELEVATED first stimulus?"""
    meas = data.regions[region].measured_uM
    stim1 = [float(meas[g][0]) for g in GROUPS]
    v_hi = float(meas[elevated_group][0])
    v_lo = float(meas[reference_group][0])
    mean_v = float(np.mean(stim1))
    minimax_v = 0.5 * (max(stim1) + min(stim1))
    minimax_res = 0.5 * (max(stim1) - min(stim1))
    out = {
        "region": region,
        "region_label": REGION_LABEL[region],
        "elevated_group": elevated_group,
        "reference_group": reference_group,
        "measured_stim1_uM": {g: float(meas[g][0]) for g in GROUPS},
        "measured_elevation_uM": v_hi - v_lo,
        "reading_uncertainty_uM": READING_UNCERTAINTY_UM,
        "elevation_in_reading_uncertainty_units": (v_hi - v_lo) / READING_UNCERTAINTY_UM,
        "measured_stim1_order_highest_first": sorted(
            GROUPS, key=lambda g: -float(meas[g][0])),
        "structural_argument": [
            "every group starts stimulation 1 with a FULL releasable pool and no "
            "extracellular DA, so it releases min(efficacy*dose, R0) pmol and the "
            "peak is min(efficacy*dose, R0) * (1e9/V) * peak_shape",
            "R0 enters stimulation 1 ONLY as a cap that can LOWER that peak; "
            "tau_refill does not enter it at all; uncapped groups all sit on the "
            "same group-invariant ceiling efficacy*dose*(1e9/V)*peak_shape",
            "so the model can put one group above another at stimulation 1 only if "
            "the reference group is pool-capped (R0 < efficacy*dose) while the "
            "elevated group is not",
        ],
        "analytic_floor_uM": {
            "IF_NO_GROUP_IS_POOL_CAPPED_all_four_stim1_values_are_forced_equal": {
                "condition": ("R0 >= efficacy*dose for every group, i.e. no group "
                              "empties its pool at a stimulus: the pure "
                              "pool-shrink / refill-slowdown hypothesis"),
                "measured_stim1_spread_uM": float(max(
                    float(meas[g][0]) for g in GROUPS)
                    - min(float(meas[g][0]) for g in GROUPS)),
                "best_common_stim1_value_least_squares_uM": mean_v,
                "rms_over_the_four_stim1_points_uM": float(np.sqrt(np.mean(
                    [(mean_v - float(meas[g][0])) ** 2 for g in GROUPS]))),
                "minimax_common_stim1_value_uM": minimax_v,
                "minimax_max_abs_residual_uM": minimax_res,
                "rms_over_the_four_stim1_points_at_the_minimax_value_uM":
                    float(np.sqrt(np.mean([(minimax_v - float(meas[g][0])) ** 2
                                           for g in GROUPS]))),
                "rms_over_the_full_matrix_from_stim1_alone_uM": float(np.sqrt(
                    sum((mean_v - float(meas[g][0])) ** 2 for g in GROUPS)
                    / (len(GROUPS) * N_STIMULATIONS))),
                "in_reading_uncertainty_units": float(np.sqrt(np.mean(
                    [(mean_v - float(meas[g][0])) ** 2 for g in GROUPS]))
                    / READING_UNCERTAINTY_UM),
            },
            "FOR_THE_ELEVATED_PAIR_if_both_groups_uncapped": {
                "condition": ("both the elevated and the reference group are "
                              "uncapped, so they necessarily share one stim-1 peak"),
                "rms_over_the_two_groups_uM": abs(v_hi - v_lo) / 2.0,
                "max_abs_residual_uM": abs(v_hi - v_lo) / 2.0,
            },
        },
        "capping_route_is_the_only_way_out": (
            "capping lets a group sit BELOW the shared ceiling, so it is the only "
            "route to a non-equal stimulation-1 vector; it forces that group into "
            "an immediate drop to (1-exp(-600/tau_refill)) and then a flat "
            "plateau, whose cost is quantified in pool_capped_group_cost"),
    }
    n_pairs = 0
    n_elevating = 0
    examples = []
    elevating_all_capped = True
    for shared in grid.shared_points("A"):
        effD = shared["efficacy"] * protocol.ach_dose_pmol
        factor = (1e9 / grid.voxel_volume_um3) * peak_shape_factor(
            shared["release_tau_s"], shared["clearance_tau_s"])
        for R_hi in grid.pool_capacity_pmol:
            peak_hi = min(effD, R_hi) * factor
            for R_lo in grid.pool_capacity_pmol:
                peak_lo = min(effD, R_lo) * factor
                n_pairs += 1
                if peak_hi > peak_lo + 1e-15:
                    n_elevating += 1
                    ref_capped = R_lo < effD
                    elevating_all_capped = elevating_all_capped and ref_capped
                    if len(examples) < 6:
                        examples.append({
                            "efficacy": shared["efficacy"],
                            "release_tau_s": shared["release_tau_s"],
                            "clearance_tau_s": shared["clearance_tau_s"],
                            "R0_elevated_group_pmol": R_hi,
                            "R0_reference_group_pmol": R_lo,
                            "reference_group_is_pool_capped": bool(ref_capped),
                            "model_stim1_elevated_group_uM": peak_hi,
                            "model_stim1_reference_group_uM": peak_lo,
                        })
    out["grid_stim1_pairs_enumerated"] = n_pairs
    out["grid_stim1_pairs_with_elevation"] = n_elevating
    out["grid_stim1_pairs_with_elevation_fraction"] = n_elevating / float(n_pairs)
    out["elevating_examples"] = _jsonable(examples)
    out["every_elevating_pair_requires_pool_capping_of_the_reference_group"] = bool(
        elevating_all_capped)
    out["n_elevating_pairs_that_do_not_require_capping"] = int(
        0 if elevating_all_capped else -1) if n_elevating else 0
    return out


def capped_cost_report(grid, protocol, data, region="central_complex",
                       group="control_1"):
    """Cost of the only route to stimulation-1 elevation: capping a reference group.

    A capped group empties its whole pool at stimulation 1, so stimulation 2 can
    only reach what refilled during 600 s: the release drops by exactly
    1 - exp(-600/tau_refill) and then stays flat. This quantifies how far that
    shape is from the measured curve.
    """
    meas = np.asarray(data.regions[region].measured_uM[group], dtype=float)
    rows = []
    for tau_refill in grid.refill_tau_s:
        ratio = 1.0 - math.exp(-protocol.inter_stimulus_interval_s / tau_refill)
        shape = np.array([1.0] + [ratio] * (N_STIMULATIONS - 1))
        scales = np.linspace(0.5 * meas[0], 1.5 * meas[0], 401)
        errs = np.array([np.sqrt(np.mean((s * shape - meas) ** 2)) for s in scales])
        k = int(np.argmin(errs))
        res = scales[k] * shape - meas
        rows.append({"refill_tau_s": float(tau_refill),
                     "stim2_over_stim1": ratio,
                     "best_possible_rms_uM": float(errs[k]),
                     "best_possible_max_abs_residual_uM": float(np.max(np.abs(res))),
                     "amplitude_at_stim1_uM": float(scales[k]),
                     "residuals_uM": [float(x) for x in res]})
    best_row = min(rows, key=lambda r: r["best_possible_rms_uM"])
    return {
        "region": region, "group": group,
        "measured_uM": [float(x) for x in meas],
        "best_capped_fit": _jsonable(best_row),
        "rms_floor_for_a_capped_group_uM": best_row["best_possible_rms_uM"],
        "in_reading_uncertainty_units":
            best_row["best_possible_rms_uM"] / READING_UNCERTAINTY_UM,
        "rows": _jsonable(rows),
        "explanation": ("a pool-capped group drops by exactly (1-exp(-600/tau_refill)) "
                        "at stimulation 2 and then stays flat; no measured group has "
                        "that shape, so buying an elevated stimulation 1 by capping a "
                        "reference group costs at least this shape error"),
    }


# ---------------------------------------------------------------------------
# 7. non-identifiability
# ---------------------------------------------------------------------------
def near_optimal_set(grid, arm, protocol, region_targets, region_scales,
                     band_uM=0.01, seed=DEFAULT_SEED, max_rows=100000):
    """Grid configurations indistinguishable from the best achievable fit.

    Collects, for every shared-kinetic point, the per-group best sub-grid point,
    then reports how many configurations land within ``band_uM`` of the best
    maximum-absolute residual, together with the ranges of R0, tau_refill,
    efficacy and the two time constants inside that band.
    """
    regions = tuple(region_targets.keys())
    scale_combos = [dict(zip(regions, vals))
                    for vals in itertools.product(*[region_scales[r] for r in regions])]
    rng = np.random.default_rng(int(seed))
    rows = []
    for shared in grid.shared_points(arm):
        n_cand = grid.n_subgrid_points(arm)
        perm = rng.permutation(n_cand)
        cands, preds, _mask = _candidate_predictions(grid, arm, shared, protocol)
        for scales in scale_combos:
            idx, worst, sse = {}, 0.0, 0.0
            for g in GROUPS:
                cand_sse = np.zeros(n_cand)
                for r in regions:
                    diff = (float(scales[r]) * preds[g]
                            - np.asarray(region_targets[r][g], dtype=float)[None, :])
                    cand_sse += np.einsum("ij,ij->i", diff, diff)
                i = _best_index(cand_sse, perm)
                idx[g] = i
                sse += float(cand_sse[i])
                for r in regions:
                    d = np.abs(float(scales[r]) * preds[g][i]
                               - np.asarray(region_targets[r][g], dtype=float))
                    worst = max(worst, float(np.max(d)))
            rows.append({"shared": dict(shared), "scales": dict(scales),
                         "per_group": {g: cands[idx[g]] for g in GROUPS},
                         "rms_uM": math.sqrt(sse / (len(regions) * len(GROUPS)
                                                    * N_STIMULATIONS)),
                         "max_abs_residual_uM": worst})
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    rows.sort(key=lambda r: r["max_abs_residual_uM"])
    best = rows[0]
    keep = [r for r in rows
            if r["max_abs_residual_uM"] <= best["max_abs_residual_uM"] + float(band_uM)]
    within_reading = [r for r in rows
                      if r["max_abs_residual_uM"] <= READING_UNCERTAINTY_UM]

    def span(subset, getter):
        vals = [getter(r) for r in subset]
        return {"min": min(vals), "max": max(vals),
                "ratio_max_over_min": (max(vals) / min(vals) if min(vals) > 0 else None)}

    def group_span(subset, field):
        vals = [r["per_group"][g][field] for r in subset for g in GROUPS]
        return {"min": min(vals), "max": max(vals),
                "ratio_max_over_min": (max(vals) / min(vals) if min(vals) > 0 else None)}

    return {
        "band_uM": float(band_uM),
        "arm": arm,
        "n_configurations_scanned": len(rows),
        "best_max_abs_residual_uM": best["max_abs_residual_uM"],
        "best_rms_uM": best["rms_uM"],
        "n_within_band": len(keep),
        "n_within_reading_uncertainty": len(within_reading),
        "ranges_within_band": {
            "efficacy": span(keep, lambda r: r["shared"]["efficacy"]),
            "release_tau_s": span(keep, lambda r: r["shared"]["release_tau_s"]),
            "clearance_tau_s": span(keep, lambda r: r["shared"]["clearance_tau_s"]),
            "pool_capacity_pmol": group_span(keep, "pool_capacity_pmol"),
            "refill_tau_s": group_span(keep, "refill_tau_s"),
        },
        "example_equivalent_combinations": _jsonable(keep[:10]),
        "summary": [
            (f"{len(keep)} of {len(rows)} grid configurations land within "
             f"{band_uM} uM of the best achievable maximum residual, i.e. they are "
             f"indistinguishable at that level"),
            (f"{len(within_reading)} configurations put EVERY residual within the "
             f"{READING_UNCERTAINTY_UM} uM reading uncertainty"),
        ],
    }


def degeneracy_scan(grid, protocol, group="control_1"):
    """Explicit non-identifiability list for the amplitude-level degeneracies.

    (a) efficacy and voxel volume appear only as efficacy/volume, so any
        (efficacy, V) pair with the same ratio predicts identical uM curves;
    (b) release_tau_s and clearance_tau_s are not separately identifiable from
        peak amplitudes: only the joint peak-shape factor is observed.
    """
    base = ModelParams(efficacy=0.25, release_tau_s=0.5, clearance_tau_s=5.0,
                       pool_capacity_pmol={g: 0.1 for g in GROUPS},
                       refill_tau_s={g: 400.0 for g in GROUPS})
    ref = base.predicted_uM(protocol)[group]
    # (a) EXACT scaling family: scaling efficacy, voxel volume and every pool size
    #     together leaves every predicted curve unchanged (amplitude eff/V is
    #     preserved, and so is phi = efficacy*dose/R0)
    ev_pairs = []
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        pp = base.copy()
        pp.efficacy = base.efficacy * mult
        pp.voxel_volume_um3 = base.voxel_volume_um3 * mult
        pp.pool_capacity_pmol = {g: base.pool_capacity_pmol[g] * mult for g in GROUPS}
        y = pp.predicted_uM(protocol)
        ev_pairs.append({
            "scale_k": mult,
            "efficacy": pp.efficacy,
            "voxel_volume_um3": pp.voxel_volume_um3,
            "pool_capacity_pmol": dict(pp.pool_capacity_pmol),
            "efficacy_over_volume": pp.efficacy / pp.voxel_volume_um3,
            "release_fraction_phi": round(pp.release_fraction_phi(group, protocol), 15),
            "max_abs_curve_difference_uM":
                max(float(np.max(np.abs(y[g] - base.predicted_uM(protocol)[g])))
                    for g in GROUPS),
        })
    # (b) why the volume is FIXED rather than fitted: scaling efficacy and volume
    #     alone (leaving the pools) does change the curve, because the pool cap
    #     min(efficacy*dose, R0) no longer scales with the request
    vol_only = []
    for mult in (0.5, 2.0, 4.0):
        pp = base.copy()
        pp.efficacy = base.efficacy * mult
        pp.voxel_volume_um3 = base.voxel_volume_um3 * mult
        y = pp.predicted_uM(protocol)[group]
        vol_only.append({"scale_k": mult,
                         "efficacy": pp.efficacy,
                         "voxel_volume_um3": pp.voxel_volume_um3,
                         "efficacy_over_volume": pp.efficacy / pp.voxel_volume_um3,
                         "release_fraction_phi":
                             pp.release_fraction_phi(group, protocol),
                         "max_abs_curve_difference_uM":
                             float(np.max(np.abs(y - ref)))})
    table = {(tr, tc): peak_shape_factor(tr, tc)
             for tr in grid.release_tau_s for tc in grid.clearance_tau_s}
    # scaling release AND clearance time constants together rescales the whole
    # waveform in time, which leaves the PEAK amplitude exactly unchanged while
    # moving the trace half-decay: the absolute time scale is invisible to peaks
    time_scale_rows = []
    base_rel = grid.release_tau_s[len(grid.release_tau_s) // 2]
    base_clr = grid.clearance_tau_s[len(grid.clearance_tau_s) // 2]
    ref_ps = peak_shape_factor(base_rel, base_clr)
    for k in (0.25, 0.5, 1.0, 2.0, 4.0):
        tr = base_rel * k
        tc = base_clr * k
        ps = peak_shape_factor(tr, tc)
        pp = base.copy()
        pp.release_tau_s, pp.clearance_tau_s = tr, tc
        core = release_sequence(protocol, pp.group_kinetics(group))["ephemeral"]
        time_scale_rows.append({
            "k": k, "release_tau_s": tr, "clearance_tau_s": tc,
            "peak_shape_factor": ps,
            "relative_peak_shape_difference": abs(ps - ref_ps) / ref_ps,
            "t_half_s_first_stimulus": float(core["t_half_s"][0]),
            "peak_nM_first_stimulus": float(core["peak_nM"][0]),
            "max_abs_curve_difference_uM": float(np.max(np.abs(
                pp.predicted_uM(protocol)[group] - base.predicted_uM(protocol)[group]))),
        })
    keys = list(table)
    collisions = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if abs(table[a] - table[b]) <= 1e-3 * max(table[a], table[b]):
                collisions.append({"pair_a": list(a), "pair_b": list(b),
                                   "peak_shape_a": table[a], "peak_shape_b": table[b],
                                   "relative_difference":
                                       abs(table[a] - table[b]) / max(table[a], table[b])})
    return {
        "efficacy_volume_pool_scaling_degeneracy": {
            "exact_family": _jsonable(ev_pairs),
            "max_deviation_uM": max(p["max_abs_curve_difference_uM"] for p in ev_pairs),
            "family": "(efficacy, voxel_volume, every pool size) -> k * each",
            "statement": ("absolute amplitudes fix only efficacy/volume and the "
                          "released fraction fixes efficacy/R0, so the whole "
                          "(efficacy, volume, pool) triple can be scaled by any "
                          "k > 0 with IDENTICAL predictions: the voxel volume is "
                          "not identifiable and is therefore FIXED at the engine's "
                          "illustrative default, and the reported efficacy carries "
                          "that convention"),
            "if_only_efficacy_and_volume_are_scaled": {
                "rows": _jsonable(vol_only),
                "max_deviation_uM": max(r["max_abs_curve_difference_uM"]
                                        for r in vol_only),
                "statement": ("scaling only efficacy and volume does change the "
                              "curve whenever the requested amount crosses the "
                              "pool cap min(efficacy*dose, R0), which is why the "
                              "scaling family above must include the pool sizes"),
            },
        },
        "release_and_clearance_degeneracy": {
            "exact_time_scale_invariance": _jsonable(time_scale_rows),
            "n_pairs_with_identical_peak_shape": sum(
                1 for i, a in enumerate(keys) for b in keys[i + 1:]
                if table[a] == table[b]),
            "peak_shape_table": {f"{tr:g}s/{tc:g}s": table[(tr, tc)] for (tr, tc) in keys},
            "n_pairs_colliding_within_0.1_percent": len(collisions),
            "example_collisions": _jsonable(collisions[:8]),
            "statement": ("from peak amplitudes alone the two time constants enter "
                          "ONLY through the ratio clearance_tau/release_tau, so the "
                          "absolute time scale is not identifiable at all: "
                          "(0.1 s, 0.5 s), (0.2 s, 1.0 s), (1.0 s, 5.0 s) and "
                          "(4.0 s, 20.0 s) give identical peaks while their trace "
                          "half-decay times differ by 40x. Measured trace decay times "
                          "(t1/2), which the digitised panels do not contain, are "
                          "what would fix this; the source text reports t1/2 as NOT "
                          "significantly different between groups, which is itself a "
                          "constraint on the clearance time constant but is not a "
                          "number in this comparison"),
        },
    }


# ---------------------------------------------------------------------------
# 8. cross-region test and the normalised (panel H style) comparison
# ---------------------------------------------------------------------------
def cross_region_test(per_region_best, joint_with_scale, joint_without_scale,
                      grid, protocol, data):
    """One shared kinetic set for BOTH regions: does it work, and what is required?"""
    cc = data.regions["central_complex"].measured_uM
    mb = data.regions["mushroom_body_heel"].measured_uM
    pair = ("parkin_45", "control_45")
    cc_diff = [float(cc[pair[0]][i] - cc[pair[1]][i]) for i in range(N_STIMULATIONS)]
    mb_diff = [float(mb[pair[0]][i] - mb[pair[1]][i]) for i in range(N_STIMULATIONS)]
    cc_strict = [v < 0.0 for v in cc_diff]                 # parkin below control
    mb_non_below = [v >= 0.0 for v in mb_diff]             # parkin not below control
    n_pairs = n_cc = n_mb = n_both = 0
    for shared in grid.shared_points("A"):
        effD = shared["efficacy"] * protocol.ach_dose_pmol
        factor = (1e9 / grid.voxel_volume_um3) * peak_shape_factor(
            shared["release_tau_s"], shared["clearance_tau_s"])
        for R_a in grid.pool_capacity_pmol:          # parkin_45
            peak_a = min(effD, R_a) * factor
            for R_b in grid.pool_capacity_pmol:      # control_45
                peak_b = min(effD, R_b) * factor
                n_pairs += 1
                cc_ok = peak_a < peak_b          # model reproduces the CC ordering
                mb_ok = peak_a >= peak_b         # model reproduces the heel ordering
                n_cc += int(cc_ok)
                n_mb += int(mb_ok)
                n_both += int(cc_ok and mb_ok)
    ratios = {g: [float(mb[g][i] / cc[g][i]) for i in range(N_STIMULATIONS)]
              for g in GROUPS}
    return {
        "hypothesis_tested": ("ONE shared kinetic set -- shared efficacy, release and "
                              "clearance time constants, and group-specific pool/"
                              "refilling applied IDENTICALLY in both regions -- "
                              "explains BOTH regions"),
        "region_pair": list(pair),
        "central_complex_parkin45_minus_control45_uM": cc_diff,
        "mushroom_body_heel_parkin45_minus_control45_uM": mb_diff,
        "central_complex_parkin45_below_control45_all_6": bool(all(cc_strict)),
        "mushroom_body_heel_parkin45_not_below_control45_all_6":
            bool(all(mb_non_below)),
        "mushroom_body_heel_stimulations_with_parkin45_strictly_above":
            int(sum(1 for v in mb_diff if v > 0.0)),
        "mushroom_body_heel_stimulations_with_the_two_groups_equal":
            int(sum(1 for v in mb_diff if v == 0.0)),
        "ordering_conflict_confirmed_in_the_measured_data": bool(
            all(cc_strict) and all(mb_non_below)),
        "ordering_conflict_is_not_a_reading_artefact": bool(
            all(abs(v) > READING_UNCERTAINTY_UM for v in cc_diff)
            and mb_diff[0] > READING_UNCERTAINTY_UM),
        "smallest_absolute_central_complex_ordering_gap_uM":
            min(abs(v) for v in cc_diff),
        "central_complex_ordering_gap_at_stim1_uM": abs(cc_diff[0]),
        "heel_ordering_gap_at_stim1_uM": abs(mb_diff[0]),
        "stim1_R0_pairs_enumerated": n_pairs,
        "stim1_R0_pairs_matching_central_complex_ordering": n_cc,
        "stim1_R0_pairs_matching_mushroom_body_heel_ordering": n_mb,
        "stim1_R0_pairs_matching_BOTH_orderings": n_both,
        "structural_argument": [
            "at stimulation 1 the model's peak is min(efficacy*dose, R0) * "
            "(1e9/V) * peak_shape, so it depends on the group only through R0",
            "the ordering of two groups at stimulation 1 is therefore fixed by "
            "their R0 values and is IDENTICAL in both regions: a region amplitude "
            "scale (detected volume / electrode sensitivity) is a positive "
            "multiplier and cannot reorder groups",
            "the measured data require parkin-45d STRICTLY BELOW control-45d at all "
            "six stimulations in the central complex, and NOT BELOW it at any of the "
            "six in the mushroom body heel (equal at one of them), so with a shared "
            "kinetic set the required sign vector differs between regions and no "
            "configuration of arm A fits both",
        ],
        "cross_region_amplitude_ratios_heel_over_central_complex": ratios,
        "per_region_arm_A_best_fit_rms_uM": per_region_best,
        "joint_shared_kinetics_with_one_region_amplitude_scale": joint_with_scale,
        "joint_shared_kinetics_without_any_region_scale": joint_without_scale,
    }


def normalised_report(fit_result, data, region):
    """Panel-H style comparison: each group normalised to its own 1st stimulus.

    The paper normalises EACH FLY to that fly's own first stimulation and then
    averages; a deterministic model can only produce a ratio of means. The
    difference is reported, not hidden (the digitised data file records it as a
    known inconsistency of the source figure).
    """
    meas = data.regions[region].measured_uM
    out = {"region": region,
           "normalisation": ("model: ratio to the model's own stimulation 1 (ratio "
                             "of means). Paper: each fly normalised to its own "
                             "stimulation 1, then averaged (mean of ratios). These "
                             "are not identical; the digitised data file records "
                             "that inconsistency explicitly."),
           "per_group": {}}
    all_res = []
    for g in GROUPS:
        y = fit_result.model_uM[region][g]
        y = y / y[0]
        m = meas[g] / meas[g][0]
        res = y - m
        all_res.append(res ** 2)
        out["per_group"][g] = {
            "model_normalised": [float(v) for v in y],
            "measured_normalised": [float(v) for v in m],
            "residual_normalised": [float(v) for v in res],
            "rms_normalised": float(np.sqrt(np.mean(res ** 2))),
            "max_abs_residual_normalised": float(np.max(np.abs(res))),
            "reading_uncertainty_in_normalised_units":
                float(READING_UNCERTAINTY_UM / meas[g][0]),
            "n_points_exceeding_reading_uncertainty":
                int(np.sum(np.abs(res) > READING_UNCERTAINTY_UM / meas[g][0])),
        }
    out["rms_overall_normalised"] = float(np.sqrt(np.mean(np.concatenate(all_res))))
    out["note_first_point"] = ("stimulation 1 is 1.0 by construction for every "
                              "group, so the normalised comparison CANNOT test the "
                              "elevated first stimulation: normalisation removes it")
    return out


# ---------------------------------------------------------------------------
# 9. fast evaluator vs the engine
# ---------------------------------------------------------------------------
def self_test_against_engine(protocol=None, grid=None, n=24, seed=DEFAULT_SEED):
    """Compare the vectorised peak evaluator with engine.release_sequence()."""
    protocol = protocol if protocol is not None else ProtocolSpec()
    grid = grid if grid is not None else default_grid(fine=True)
    rng = np.random.default_rng(int(seed))
    rows = []
    worst = 0.0
    for _ in range(int(n)):
        R0 = float(rng.choice(grid.pool_capacity_pmol))
        tau_r = float(rng.choice(grid.refill_tau_s))
        eff = float(rng.choice(grid.efficacy))
        tau_rel = float(rng.choice(grid.release_tau_s))
        tau_clr = float(rng.choice(grid.clearance_tau_s))
        p = ModelParams(efficacy=eff, release_tau_s=tau_rel, clearance_tau_s=tau_clr,
                        pool_capacity_pmol={g: R0 for g in GROUPS},
                        refill_tau_s={g: tau_r for g in GROUPS})
        fast = p.predicted_uM(protocol)["control_1"]
        engine = p.engine_predicted_uM(protocol, "control_1")
        diff = float(np.max(np.abs(fast - engine)))
        worst = max(worst, diff)
        rows.append({"R0_pmol": R0, "tau_refill_s": tau_r, "efficacy": eff,
                     "release_tau_s": tau_rel, "clearance_tau_s": tau_clr,
                     "max_abs_difference_uM": diff})
    return {"n_configurations": int(n), "worst_abs_difference_uM": worst,
            "tolerance_uM": ENGINE_AGREEMENT_TOLERANCE_UM,
            "passed": bool(worst <= ENGINE_AGREEMENT_TOLERANCE_UM),
            "what_is_compared": ("the fast vectorised evaluator used by the "
                                 "exhaustive search vs the ENGINE's own "
                                 "release_sequence() peak_nM, same parameters, "
                                 "same measured protocol"),
            "rows": _jsonable(rows)}


# ---------------------------------------------------------------------------
# 10. the report
# ---------------------------------------------------------------------------
def build_report(data=None, protocol=None, grid=None, seed=DEFAULT_SEED,
                 quick=False, joint_scale_values=None, verbose=True):
    """Run every comparison and assemble the full machine-readable report."""
    t_start = time.perf_counter()
    data = data if data is not None else load_digitised_data()
    protocol = protocol if protocol is not None else protocol_from_digitised(data)
    grid = grid if grid is not None else default_grid(fine=not quick)
    timings = {}

    targets = {r: {g: data.regions[r].measured_uM[g] for g in GROUPS} for r in REGION_KEYS}

    fits, fits_heel = {}, {}
    for arm in ARMS:
        t0 = time.perf_counter()
        fits[arm] = attach_sem(fit(grid, arm,
                                   {"central_complex": targets["central_complex"]},
                                   protocol=protocol, seed=seed,
                                   label=f"arm {arm}: central complex only"), data)
        timings[f"fit_arm_{arm}_central_complex_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fits_heel[arm] = fit(grid, arm,
                             {"mushroom_body_heel": targets["mushroom_body_heel"]},
                             protocol=protocol, seed=seed,
                             label=f"arm {arm}: mushroom body heel only")
        fits_heel[arm].sem_uM["mushroom_body_heel"] = {g: None for g in GROUPS}
        timings[f"fit_arm_{arm}_heel_s"] = time.perf_counter() - t0

    # the pure pool-shrink / refill-slowdown hypothesis: pool capping disallowed
    t0 = time.perf_counter()
    nocap_cc = fit(grid, "A", {"central_complex": targets["central_complex"]},
                   protocol=protocol, seed=seed, exclude_capping=True,
                   label=("arm A with pool capping DISALLOWED (R0 >= efficacy*dose "
                          "for every group): the pure pool-shrink / refill-slowdown "
                          "hypothesis, central complex"))
    attach_sem(nocap_cc, data)
    nocap_mb = fit(grid, "A", {"mushroom_body_heel": targets["mushroom_body_heel"]},
                   protocol=protocol, seed=seed, exclude_capping=True,
                   label=("arm A with pool capping DISALLOWED, mushroom body heel"))
    nocap_mb.sem_uM["mushroom_body_heel"] = {g: None for g in GROUPS}
    timings["no_capping_fits_s"] = time.perf_counter() - t0

    scale_vals = (tuple(float(v) for v in joint_scale_values)
                  if joint_scale_values is not None else _geomspace(0.25, 4.0, 25))
    t0 = time.perf_counter()
    joint = fit(grid, "A", targets,
                {"central_complex": (1.0,), "mushroom_body_heel": scale_vals},
                protocol=protocol, seed=seed,
                label=("arm A over BOTH regions at once: one shared kinetic set, "
                       "one heel amplitude scale"))
    joint_noscale = fit(grid, "A", targets,
                        {"central_complex": (1.0,), "mushroom_body_heel": (1.0,)},
                        protocol=protocol, seed=seed, refine=False,
                        label=("arm A over BOTH regions: one shared kinetic set and "
                               "NO region amplitude scale (absolute amplitudes shared)"))
    timings["joint_cross_region_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crit_cc = critical_test(grid, protocol, data, "central_complex",
                            "control_45", "control_1")
    crit_mb = critical_test(grid, protocol, data, "mushroom_body_heel",
                            "parkin_1", "control_1")
    capped = {"central_complex": capped_cost_report(grid, protocol, data,
                                                   "central_complex", "control_1"),
              "mushroom_body_heel": capped_cost_report(grid, protocol, data,
                                                       "mushroom_body_heel", "control_1")}
    timings["critical_tests_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nonid = near_optimal_set(grid, "A", protocol,
                             {"central_complex": targets["central_complex"]},
                             {"central_complex": (1.0,)}, band_uM=0.01, seed=seed)
    nonid_heel = near_optimal_set(grid, "A", protocol,
                                  {"mushroom_body_heel":
                                   targets["mushroom_body_heel"]},
                                  {"mushroom_body_heel": (1.0,)}, band_uM=0.01,
                                  seed=seed)
    degeneracy = degeneracy_scan(grid, protocol)
    timings["non_identifiability_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cross = cross_region_test(
        per_region_best={"central_complex_arm_A_rms_uM": fits["A"].rms_overall_uM,
                         "mushroom_body_heel_arm_A_rms_uM":
                             fits_heel["A"].rms_overall_uM,
                         "central_complex_arm_A_max_abs_residual_uM":
                             fits["A"].max_abs_residual_uM,
                         "mushroom_body_heel_arm_A_max_abs_residual_uM":
                             fits_heel["A"].max_abs_residual_uM},
        joint_with_scale={"rms_overall_uM": joint.rms_overall_uM,
                          "max_abs_residual_uM": joint.max_abs_residual_uM,
                          "region_amplitude_scales": joint.scales,
                          "rms_over_reading_uncertainty":
                              joint.rms_overall_uM / READING_UNCERTAINTY_UM},
        joint_without_scale={"rms_overall_uM": joint_noscale.rms_overall_uM,
                             "max_abs_residual_uM": joint_noscale.max_abs_residual_uM,
                             "rms_over_reading_uncertainty":
                                 joint_noscale.rms_overall_uM / READING_UNCERTAINTY_UM},
        grid=grid, protocol=protocol, data=data)
    timings["cross_region_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    normalised = {"central_complex": normalised_report(fits["A"], data, "central_complex"),
                  "mushroom_body_heel": normalised_report(fits_heel["A"], data,
                                                          "mushroom_body_heel")}
    timings["normalised_panel_H_s"] = time.perf_counter() - t0

    engine_agreement = {}
    for region, f in (("central_complex", fits["A"]),
                      ("mushroom_body_heel", fits_heel["A"])):
        worst = 0.0
        for g in GROUPS:
            eng = f.params.engine_predicted_uM(protocol, g, scale=float(f.scales[region]))
            worst = max(worst, float(np.max(np.abs(eng - f.model_uM[region][g]))))
        engine_agreement[region] = {"max_abs_difference_uM": worst,
                                    "n_points": len(GROUPS) * N_STIMULATIONS,
                                    "tolerance_uM": ENGINE_AGREEMENT_TOLERANCE_UM}
    evaluator_check = self_test_against_engine(protocol, grid, n=24, seed=seed)

    violations = []
    for region, holder in (("central_complex", fits), ("mushroom_body_heel", fits_heel)):
        for arm, f in holder.items():
            sem_region = data.regions[region].sem_uM
            for g in GROUPS:
                for i in range(N_STIMULATIONS):
                    res = float(f.residuals_uM[region][g][i])
                    sem = sem_region.get(g)
                    semv = None if sem is None else float(sem[i])
                    if abs(res) > READING_UNCERTAINTY_UM or (
                            semv is not None and abs(res) > semv):
                        violations.append({
                            "region": region, "arm": arm, "group": g,
                            "stimulation": i + 1,
                            "model_uM": float(f.model_uM[region][g][i]),
                            "measured_uM": float(f.measured_uM[region][g][i]),
                            "residual_uM": res,
                            "reading_uncertainty_uM": READING_UNCERTAINTY_UM,
                            "published_sem_uM": semv,
                            "exceeds_reading_uncertainty":
                                bool(abs(res) > READING_UNCERTAINTY_UM),
                            "exceeds_published_sem": (None if semv is None
                                                      else bool(abs(res) > semv))})

    report = {
        "what_this_is": (
            "Round 5: the FIRST quantitative comparison in this project of a model "
            "prediction against REAL published experimental values. The model is "
            "engine.da_protocol driven by its measured protocol; the measured values "
            "are figure readings of PMC9897283 Figures 5G and 6G."),
        "honesty": {
            "measured_points_are_figure_readings_not_raw_data": True,
            "measured_points_statement": (
                "The measured points are VISUAL READINGS of a published figure image "
                "(750x653 px JPEG), NOT raw experimental data, with a ~0.03 uM "
                "reading uncertainty. This is stated on the figure itself."),
            "a_good_fit_does_not_identify_a_mechanism": True,
            "no_mechanism_is_claimed": True,
            "no_consciousness_viability_medical_claim": True,
            "ach_is_the_stimulus_da_is_the_modelled_transmitter": True,
            "measured_protocol_timing_only": (
                "Only the protocol timing (6 stimulations, 0.2 pmol ACh each, 600 s "
                "apart, bolus 5 s into a 60 s record, 1200 s post-dissection rest) is "
                "measured; every model kinetic constant is ILLUSTRATIVE and fitted, "
                "not measured."),
            "what_a_real_fit_would_still_require": [
                "raw per-fly data: the source values are figure-only, so only read "
                "means and SEMs exist here and per-fly variability is unavailable",
                "independently measured clearance per group, without which clearance "
                "and release scale are degenerate",
                "the detected volume / electrode sensitivity per region, without "
                "which absolute uM amplitudes cannot be attributed to release",
                "receptor expression and release-probability data, since the "
                "ACh-receptor -> DA-release chain is not modelled at all",
                "the regionally inconsistent effects the paper itself reports: the "
                "mushroom body heel differs from the central complex, and the "
                "measured ordering of parkin-45d vs control-45d differs between the "
                "two regions, which one shared kinetic set cannot reproduce",
                "measured trace half-decay times per group to test the reported "
                "'t1/2 not significantly different' claim against the model's "
                "clearance time constant instead of assuming it",
                "raw per-stimulus records (the digitised panels give peak amplitudes "
                "only, so no trace shape and no time axis were compared)",
            ],
        },
        "data_provenance": data.to_dict(),
        "measured_protocol": protocol_report(protocol),
        "search": {
            "grid": grid.to_dict(),
            "per_arm": {arm: grid.to_dict(arm) for arm in ARMS},
            "seed": int(seed),
            "seed_role": ("the seed only fixes the enumeration order used for "
                          "tie-breaking; the search is exhaustive, so the reported "
                          "minimum is order-independent -- the selftest checks both"),
            "bounded": ("every axis is an explicit finite tuple inside declared "
                        "bounds; model-curve evaluations, objective evaluations and "
                        "SSE combinations are counted exactly; a hard cap rejects "
                        "oversized searches"),
            "total_model_curve_evaluations_central_complex_all_arms":
                sum(fits[a].n_model_curve_evaluations for a in ARMS),
            "factored_exactness": (
                "the objective separates across groups once the shared kinetic point "
                "is fixed, so the per-group minimum is taken independently and the "
                "reported optimum is the exact minimum over the full product grid"),
        },
        "fits": {
            "central_complex": {a: fits[a].to_dict() for a in ARMS},
            "mushroom_body_heel": {a: fits_heel[a].to_dict() for a in ARMS},
            "cross_region_shared_kinetics": joint.to_dict(),
            "cross_region_no_region_scale": joint_noscale.to_dict(),
        },
        "no_capping_pure_pool_refill_hypothesis": {
            "central_complex": nocap_cc.to_dict(),
            "mushroom_body_heel": nocap_mb.to_dict(),
            "what_it_is": ("the same shared-kinetics search with pool capping "
                           "DISALLOWED (R0 >= efficacy*dose for every group), i.e. "
                           "exactly the pure pool-shrink / refill-slowdown reading "
                           "of the aging hypothesis"),
        },
        "critical_test": {"central_complex": crit_cc,
                          "mushroom_body_heel": crit_mb,
                          "pool_capped_group_cost": capped},
        "non_identifiability": {
            "near_optimal_set_central_complex_arm_A": nonid,
            "near_optimal_set_mushroom_body_heel_arm_A": nonid_heel,
            "explicit_degeneracies": degeneracy,
            "reading_of_these_lists": (
                "members of a near-optimal set are parameter combinations that "
                "predict the same curve shape to within the stated band, so the "
                "digitised readings cannot separate them; this is a statement about "
                "identifiability, not about any mechanism")},
        "cross_region": cross,
        "panel_H_normalised": normalised,
        "engine_agreement_at_fitted_parameters": engine_agreement,
        "fast_evaluator_vs_engine_check": evaluator_check,
        "residuals_exceeding_uncertainty": violations,
        "n_residuals_exceeding_uncertainty": len(violations),
        "run_stats": {
            "wall_clock_s": time.perf_counter() - t_start,
            "stage_wall_clock_s": timings,
            "peak_rss_mb": peak_memory_mb(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "argv": list(sys.argv),
        },
    }
    report["headline_findings"] = _headline(report, fits, fits_heel, joint,
                                            joint_noscale, crit_cc, crit_mb, cross,
                                            nocap_cc, nocap_mb)
    report["_data"] = data
    report["_protocol"] = protocol
    report["_grid"] = grid
    report["_fits"] = fits
    report["_fits_heel"] = fits_heel
    report["_nocap"] = {"central_complex": nocap_cc,
                        "mushroom_body_heel": nocap_mb}
    report["_joint"] = joint
    if verbose:
        for line in report["headline_findings"]["statements"]:
            print("[headline] " + line)
    return report


def _headline(report, fits, fits_heel, joint, joint_noscale, crit_cc, crit_mb,
              cross, nocap_cc, nocap_mb):
    ccA, hhA = fits["A"], fits_heel["A"]

    def holder_fit(name):
        return fits["A"] if name == "central_complex" else fits_heel["A"]

    def holder_rms(name):
        return holder_fit(name).rms_overall_uM

    def holder_maxres(name):
        return holder_fit(name).max_abs_residual_uM

    def n_viol(name):
        return holder_fit(name).exceedance_counts()["exceed_reading_uncertainty"]

    st = []
    for tag, f in (("central complex", ccA), ("mushroom body heel", hhA)):
        ec = f.exceedance_counts()
        st.append(
            f"Arm A (shared kinetics, group-specific pool/refill only) best fit on the "
            f"{tag}: RMS {f.rms_overall_uM:.4f} uM = "
            f"{f.rms_overall_uM / READING_UNCERTAINTY_UM:.2f}x the 0.03 uM reading "
            f"uncertainty, but the RMS hides the failure: "
            f"{ec['exceed_reading_uncertainty']} of {ec['total_points']} residuals "
            f"exceed the reading uncertainty, {ec['exceed_published_sem']} of "
            f"{ec['points_with_published_sem']} exceed the published SEM, and "
            f"max|residual| is {f.max_abs_residual_uM:.4f} uM = "
            f"{f.max_abs_residual_uM / READING_UNCERTAINTY_UM:.1f}x the uncertainty.")
    for tag, cr, noc, holder in (("central complex", crit_cc, nocap_cc, "central_complex"),
                                ("mushroom body heel", crit_mb, nocap_mb,
                                 "mushroom_body_heel")):
        fl = cr["analytic_floor_uM"][
            "IF_NO_GROUP_IS_POOL_CAPPED_all_four_stim1_values_are_forced_equal"]
        st.append(
            f"CRITICAL TEST -- {tag}: measured stimulation-1 values are "
            f"{cr['measured_stim1_uM']} (spread {fl['measured_stim1_spread_uM']:.2f} "
            f"uM). The model's stimulation-1 peak is GROUP-INVARIANT unless a group is "
            f"pool-capped, so with capping disallowed the four values are forced equal "
            f"and the best achievable RMS over those four points alone is "
            f"{fl['rms_over_the_four_stim1_points_uM']:.4f} uM = "
            f"{fl['in_reading_uncertainty_units']:.1f}x the reading uncertainty "
            f"(minimax max|residual| {fl['minimax_max_abs_residual_uM']:.4f} uM).")
        st.append(
            f"  {cr['grid_stim1_pairs_with_elevation']} of "
            f"{cr['grid_stim1_pairs_enumerated']} enumerated stimulation-1 grid pairs "
            f"CAN elevate one group above another, and every one of them does it by "
            f"pool-capping the other group (verified: "
            f"{cr['every_elevating_pair_requires_pool_capping_of_the_reference_group']}). "
            f"With capping allowed the best {tag} fit has RMS "
            f"{holder_rms(holder):.4f} uM with {n_viol(holder)} of 24 residuals above "
            f"the 0.03 uM reading uncertainty and max|residual| "
            f"{holder_maxres(holder):.4f} uM; with capping DISALLOWED it is RMS "
            f"{noc.rms_overall_uM:.4f} uM "
            f"({noc.rms_overall_uM / READING_UNCERTAINTY_UM:.1f}x), max|residual| "
            f"{noc.max_abs_residual_uM:.4f} uM, and the four stimulation-1 residuals "
            f"alone are "
            f"{[round(float(noc.residuals_uM[holder][g][0]), 3) for g in GROUPS]} uM "
            f"for {list(GROUPS)}.")
    st.append(
        f"CROSS-REGION: parkin-45d is strictly BELOW control-45d at all six "
        f"stimulations in the central complex, but NOT below it at any of the six in "
        f"the mushroom body heel (equal at one), while one shared kinetic set can "
        f"only produce the SAME ordering in both regions; "
        f"{cross['stim1_R0_pairs_matching_BOTH_orderings']} of "
        f"{cross['stim1_R0_pairs_enumerated']} enumerated stimulation-1 configurations "
        f"match both orderings. Shared-kinetics joint fit RMS "
        f"{joint.rms_overall_uM:.4f} uM (with one heel amplitude scale "
        f"{joint.scales['mushroom_body_heel']:.3f}); without any region scale "
        f"{joint_noscale.rms_overall_uM:.4f} uM.")
    for arm in ("B", "C"):
        st.append(
            f"{ARM_DESCRIPTION[arm].split(':')[0]} on the central complex: RMS "
            f"{fits[arm].rms_overall_uM:.4f} uM, max |residual| "
            f"{fits[arm].max_abs_residual_uM:.4f} uM (labelled hypothesis, extra "
            f"degree of freedom: "
            f"{'per-group release efficacy' if arm == 'B' else 'per-group clearance time constant'}).")
    return {"statements": st,
            "outcome_critical_test":
                report["critical_test"]["central_complex"][
                    "every_elevating_pair_requires_pool_capping_of_the_reference_group"],
            "outcome_cross_region_pairs_matching_both":
                cross["stim1_R0_pairs_matching_BOTH_orderings"],
            "no_mechanism_claimed": True}


# ---------------------------------------------------------------------------
# 11. outputs: figure + traces
# ---------------------------------------------------------------------------
def make_figure(report, out_path=PNG_PATH, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from matplotlib.lines import Line2D
    fits, fits_heel = report["_fits"], report["_fits_heel"]
    nocap = report["_nocap"]
    data = report["_data"]
    crit = report["critical_test"]
    x = np.arange(1, N_STIMULATIONS + 1)

    fig = plt.figure(figsize=(16.0, 15.4), dpi=dpi)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.05, 1.0],
                  hspace=0.50, wspace=0.20,
                  left=0.075, right=0.985, top=0.915, bottom=0.235)

    data_points = {}

    def record(ax, g, values, n_repeat=1):
        data_points.setdefault(id(ax), {})[g] = (
            np.tile(x, int(n_repeat)), np.asarray(values, dtype=float))

    def draw_absolute(ax, region, fa, fnc, title):
        meas = data.regions[region].measured_uM
        sem = data.regions[region].sem_uM
        top = 0.0
        for g in GROUPS:
            err = sem[g] if sem[g] is not None else np.full(N_STIMULATIONS,
                                                            READING_UNCERTAINTY_UM)
            ax.errorbar(x, meas[g], yerr=err, fmt=GROUP_MARKER[g], ms=9, lw=1.6,
                        color=GROUP_COLOUR[g], mfc="white", mew=2.0, capsize=4,
                        zorder=5,
                        label=f"{GROUP_LABEL[g]}: measured"
                              + ("" if sem[g] is not None else " (±0.03 read)"))
            ax.plot(x, fa.model_uM[region][g], "-", lw=2.6, color=GROUP_COLOUR[g],
                    zorder=4)
            ax.plot(x, fnc.model_uM[region][g], "--", lw=1.9, dashes=(6, 3),
                    color=GROUP_COLOUR[g], alpha=0.9, zorder=3)
            record(ax, g, np.concatenate([meas[g], fa.model_uM[region][g],
                                          fnc.model_uM[region][g]]), n_repeat=3)
            top = max(top, float(np.max(meas[g])),
                      float(np.max(fa.model_uM[region][g])),
                      float(np.max(fnc.model_uM[region][g])))
        ax.set_title(title, fontsize=14.5, fontweight="bold", pad=8)
        ax.set_xlabel("stimulation number (0.2 pmol ACh bolus each, 600 s apart)",
                      fontsize=12.5)
        ax.set_ylabel("peak [DA] (µM) — figure-read FSCV value", fontsize=12.5)
        ax.set_xticks(x)
        ax.grid(alpha=0.25, lw=0.7)
        ax.tick_params(labelsize=11.5)
        ax.set_ylim(0.0, 1.85 * top)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color="0.25", lw=2.6, ls="-"),
                    Line2D([], [], color="0.25", lw=1.9, ls="--")]
        labels += ["model arm A: shared kinetics, pool/refill per group "
                   "(pool capping allowed)",
                   "model arm A-nocap: pure pool shrink / refill slowdown only"]
        ax.legend(handles, labels, fontsize=8.6, ncol=2, loc="upper center",
                  framealpha=0.94, borderpad=0.5, columnspacing=0.8,
                  handlelength=2.4, labelspacing=0.35)

    def draw_residual(ax, region, f, title):
        ax.axhspan(-READING_UNCERTAINTY_UM, READING_UNCERTAINTY_UM, color="0.86",
                   zorder=0,
                   label=f"±{READING_UNCERTAINTY_UM} µM reading uncertainty")
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
        floor = crit[region]["analytic_floor_uM"][
            "FOR_THE_ELEVATED_PAIR_if_both_groups_uncapped"][
                "rms_over_the_two_groups_uM"]
        for s in (-1.0, 1.0):
            ax.axhline(s * floor, color="crimson", lw=1.7, ls="--", zorder=2,
                       label=(f"±{floor:.3f} µM floor for the elevated pair "
                              f"(both groups uncapped)") if s > 0 else None)
        for g in GROUPS:
            res = f.residuals_uM[region][g]
            sem = data.regions[region].sem_uM[g]
            if sem is not None:
                ax.errorbar(x, res, yerr=sem, fmt=GROUP_MARKER[g], ms=8.5, lw=1.4,
                            color=GROUP_COLOUR[g], mfc="none", mew=2.0, capsize=3.5,
                            zorder=4,
                            label=f"{GROUP_LABEL[g]} (±published SEM)")
            else:
                ax.plot(x, res, GROUP_MARKER[g], ms=8.5, color=GROUP_COLOUR[g],
                        mfc="none", mew=2.0, zorder=4, label=f"{GROUP_LABEL[g]}")
            record(ax, g, res)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel("stimulation number", fontsize=12.5)
        ax.set_ylabel("model − measured (µM)", fontsize=12.5)
        ax.set_xticks(x)
        ax.grid(alpha=0.25, lw=0.7)
        ax.tick_params(labelsize=11.5)
        ax.margins(y=0.45)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(list(uniq.values()), list(uniq.keys()), fontsize=8.6, ncol=2,
                  loc="best", framealpha=0.93, borderpad=0.5, labelspacing=0.35)

    def draw_normalised(ax, region, f, title):
        meas = data.regions[region].measured_uM
        for g in GROUPS:
            ax.plot(x, meas[g] / meas[g][0], GROUP_MARKER[g], ms=9,
                    color=GROUP_COLOUR[g], mfc="white", mew=2.0,
                    label=f"{GROUP_LABEL[g]}: measured")
            y = f.model_uM[region][g]
            ax.plot(x, y / y[0], "-", lw=2.4, color=GROUP_COLOUR[g],
                    label=f"{GROUP_LABEL[g]}: model arm A")
            record(ax, g, np.concatenate([meas[g] / meas[g][0], y / y[0]]),
                   n_repeat=2)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel("stimulation number", fontsize=12.5)
        ax.set_ylabel("normalised peak [DA]\n(to that group's 1st stimulation)",
                      fontsize=12.5)
        ax.set_xticks(x)
        ax.grid(alpha=0.25, lw=0.7)
        ax.tick_params(labelsize=11.5)
        ax.set_ylim(0.0, 1.18)
        ax.legend(fontsize=8.6, ncol=2, loc="lower left", framealpha=0.93,
                  borderpad=0.5, labelspacing=0.35)

    draw_absolute(fig.add_subplot(gs[0, 0]), "central_complex", fits["A"],
                  nocap["central_complex"],
                  "A  Central complex — absolute [DA] per stimulation\n"
                  "model curves use their measured group's colour and marker shape")
    draw_residual(fig.add_subplot(gs[1, 0]), "central_complex", fits["A"],
                  "B  Central complex — residual of arm A\n"
                  "grey band = ±0.03 µM reading uncertainty")
    draw_absolute(fig.add_subplot(gs[0, 1]), "mushroom_body_heel", fits_heel["A"],
                  nocap["mushroom_body_heel"],
                  "C  Mushroom body heel — absolute [DA] per stimulation\n"
                  "same model, second region: the group ordering differs")
    draw_residual(fig.add_subplot(gs[1, 1]), "mushroom_body_heel", fits_heel["A"],
                  "D  Mushroom body heel — residual of arm A\n"
                  "grey band = ±0.03 µM reading uncertainty")
    draw_normalised(fig.add_subplot(gs[2, 0]), "central_complex", fits["A"],
                    "E  Central complex — normalised (paper panel-H style)")
    draw_normalised(fig.add_subplot(gs[2, 1]), "mushroom_body_heel", fits_heel["A"],
                    "F  Mushroom body heel — normalised")

    cc = report["fits"]["central_complex"]["A"]
    hh = report["fits"]["mushroom_body_heel"]["A"]
    nc = report["no_capping_pure_pool_refill_hypothesis"]
    nc_cc, nc_mb = nc["central_complex"], nc["mushroom_body_heel"]
    fl_cc = crit["central_complex"]["analytic_floor_uM"][
        "IF_NO_GROUP_IS_POOL_CAPPED_all_four_stim1_values_are_forced_equal"]
    fl_mb = crit["mushroom_body_heel"]["analytic_floor_uM"][
        "IF_NO_GROUP_IS_POOL_CAPPED_all_four_stim1_values_are_forced_equal"]
    nc_cc_res = [round(float(nc_cc["per_group_per_stimulation"]["central_complex"][g][0]
                             ["residual_uM"]), 3) for g in GROUPS]
    footnote = "\n".join([
        "HONESTY / PROVENANCE — READ BEFORE USING THESE NUMBERS",
        "• MEASURED POINTS ARE FIGURE READINGS, NOT RAW DATA: read by eye from PMC9897283 Figure 5 panel G (central complex) and Figure 6 panel G (mushroom body heel), from a 750×653 px JPEG.",
        f"  Reading uncertainty ≈ ±0.03 µM on every point; the published error bars (SEM, n = 10 brains) are drawn where the digitised file records them and are wider.",
        "• ACh (0.2 pmol per stimulus) is the STIMULUS; dopamine is the transmitter whose evoked release is modelled. The ACh→receptor→DA-release chain is not modelled at all.",
        "• MEASURED AND USED VERBATIM: only the protocol timing — 6 stimulations, 600 s apart, bolus 5 s into each 60 s record, 1200 s post-dissection rest.",
        "• EVERY model kinetic constant (releasable pool, refilling, efficacy, release/clearance time constants, volume) remains ILLUSTRATIVE and was FITTED here,",
        "  not measured; several parameter combinations are non-identifiable. No mechanism is identified by this comparison.",
        f"• FIT QUALITY (arm A): central complex RMS {cc['rms_overall_uM']:.3f} µM ({cc['rms_overall_uM']/0.03:.1f}× the 0.03 µM reading uncertainty), heel {hh['rms_overall_uM']:.3f} µM ({hh['rms_overall_uM']/0.03:.1f}×) — but the RMS hides the failure:",
        f"  {cc['exceedance_counts']['exceed_reading_uncertainty']}/24 and {hh['exceedance_counts']['exceed_reading_uncertainty']}/24 residuals exceed that uncertainty, with max |residual| {cc['max_abs_residual_uM']:.3f} µM ({cc['max_abs_residual_uM']/0.03:.1f}×) and {hh['max_abs_residual_uM']:.3f} µM ({hh['max_abs_residual_uM']/0.03:.1f}×).",
        f"  Labelled hypothesis arms B (per-group efficacy) / C (per-group clearance) reach RMS {report['fits']['central_complex']['B']['rms_overall_uM']:.3f} / {report['fits']['central_complex']['C']['rms_overall_uM']:.3f} µM but add an invented mechanism; they are NOT the primary result.",
        "• CRITICAL TEST: the model's stimulation-1 peak is GROUP-INVARIANT unless a group is pool-capped, so a shared-kinetics model can raise one group above another only by capping the others —",
        f"  49 280 of 268 800 enumerated stimulation-1 configurations do so, and ALL of them cap the reference group. With capping DISALLOWED (the pure pool-shrink / refill-slowdown hypothesis) the four",
        f"  stimulation-1 values are forced equal and the best achievable RMS over them is {fl_cc['rms_over_the_four_stim1_points_uM']:.3f} µM (central complex, {fl_cc['in_reading_uncertainty_units']:.1f}× the uncertainty) and {fl_mb['rms_over_the_four_stim1_points_uM']:.3f} µM (heel, {fl_mb['in_reading_uncertainty_units']:.1f}×).",
        f"  That fit (arm A-nocap) has RMS {nc_cc['rms_overall_uM']:.3f} µM, and its four stimulation-1 residuals are {nc_cc_res} µM for {list(GROUPS)}: the aged-control elevation is missed by 0.125 µM = 4.2× the uncertainty.",
        "  That is evidence for an additional mechanism, NOT proof of one; no such mechanism was added to force a fit.",
        "• CROSS-REGION: parkin-45d is strictly BELOW control-45d at all 6 stimulations in the central complex but NOT below it at any of the 6 in the mushroom body heel",
        f"  (equal at one), while one shared kinetic set can only produce the same ordering in both regions — region-specific parameters (or a mechanism the model lacks) are REQUIRED. Shared-kinetics joint fit RMS {report['fits']['cross_region_shared_kinetics']['rms_overall_uM']:.3f} µM; {report['cross_region']['stim1_R0_pairs_matching_BOTH_orderings']} of {report['cross_region']['stim1_R0_pairs_enumerated']} configurations match both regions' ordering.",
        "• A good fit would NOT identify a mechanism: efficacy is degenerate with the detected volume, and the release/clearance time constants enter peak amplitudes only jointly.",
        "  No consciousness, viability or medical claim is made or implied.",
    ])

    note_artist = fig.text(0.04, 0.212, footnote, fontsize=8.8, va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="#fbfbf5",
                                     edgecolor="0.6", linewidth=0.9))
    fig.suptitle("Model vs REAL published values — [DA] across 6 ACh stimulations "
                 "(figure-read; two brain regions)",
                 fontsize=16, fontweight="bold", y=0.978)

    # ---- objective legibility checks (no clipping, no legend over data) ----
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi
    checks = {"figure_px": [float(fw), float(fh)], "panels": {}, "clipped": []}

    def bbox(artist):
        b = artist.get_window_extent(rend)
        return [float(b.x0), float(b.y0), float(b.x1), float(b.y1)]

    def markers_under(ax, leg):
        if ax.get_legend() is None:
            return 0
        lb = bbox(leg)
        n = 0
        for g in GROUPS:
            pts = data_points.get(id(ax), {}).get(g)
            if pts is None:
                continue
            xs, ys = pts
            disp = ax.transData.transform(np.column_stack([xs, ys]))
            for px, py in disp:
                if lb[0] <= px <= lb[2] and lb[1] <= py <= lb[3]:
                    n += 1
        return n

    # shrink the footnote font until it fits inside the canvas (max 4 tries)
    for _ in range(4):
        if bbox(note_artist)[2] <= fw - 8.0:
            break
        note_artist.set_fontsize(note_artist.get_fontsize() - 0.35)
        fig.canvas.draw()

    # move every legend to the first candidate position that covers no data
    legend_positions = {}
    for i, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        if leg is None:
            continue
        chosen, covered = "kept", markers_under(ax, leg)
        if covered:
            for loc in ("best", "upper left", "lower right", "upper right",
                        "center right", "lower left"):
                leg.set_loc(loc)
                fig.canvas.draw()
                n = markers_under(ax, leg)
                if n == 0:
                    chosen, covered = loc, 0
                    break
            else:
                chosen = "no_collision_free_position"
        legend_positions[f"axes_{i}"] = {"position": chosen,
                                         "data_markers_covered": int(covered)}

    def inside(b):
        return bool(b[0] >= -0.5 and b[2] <= fw + 0.5 and b[1] >= -0.5 and b[3] <= fh + 0.5)

    for name, artist in (("suptitle", fig._suptitle), ("footnote", note_artist)):
        b = bbox(artist)
        ok = inside(b)
        checks[name] = {"bbox_px": b, "fully_inside_canvas": ok}
        if not ok:
            checks["clipped"].append(name)

    foot_bbox = bbox(note_artist)
    for i, ax in enumerate(fig.axes):
        entry = {"title": ax.get_title().split("\n")[0]}
        ab = bbox(ax)
        entry["footnote_overlaps_axes"] = bool(not (foot_bbox[3] <= ab[1] + 0.5
                                                    or foot_bbox[1] >= ab[3] - 0.5
                                                    or foot_bbox[2] <= ab[0] + 0.5
                                                    or foot_bbox[0] >= ab[2] - 0.5))
        if ab[0] < -0.5 or ab[2] > fw + 0.5:
            checks["clipped"].append(f"axes_{i}")
        leg = ax.get_legend()
        if leg is not None:
            lb = bbox(leg)
            entry["legend_bbox_px"] = lb
            n_pts_in_legend = 0
            for g in GROUPS:
                pts = data_points.get(id(ax), {}).get(g)
                if pts is None:
                    continue
                xs, ys = pts
                disp = ax.transData.transform(np.column_stack([xs, ys]))
                for px, py in disp:
                    if lb[0] <= px <= lb[2] and lb[1] <= py <= lb[3]:
                        n_pts_in_legend += 1
            entry["data_markers_covered_by_legend"] = int(n_pts_in_legend)
            entry["legend_position"] = legend_positions.get(
                f"axes_{i}", {}).get("position")
            if n_pts_in_legend:
                checks["clipped"].append(f"legend_over_data_panel_{i}")
        checks["panels"][f"axes_{i}"] = entry

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return str(out_path), checks


def save_traces(report, out_path=TRACES_PATH, sample_dt_s=0.05):
    """Save measured/model curves, residuals and full 60 s model traces."""
    data, fits, fits_heel = report["_data"], report["_fits"], report["_fits_heel"]
    protocol = report["_protocol"]
    arrays = {}
    for region, holder in (("central_complex", fits), ("mushroom_body_heel", fits_heel)):
        for g in GROUPS:
            arrays[f"{region}__measured_uM__{g}"] = np.asarray(
                data.regions[region].measured_uM[g], dtype=float)
            sem = data.regions[region].sem_uM[g]
            if sem is not None:
                arrays[f"{region}__sem_uM__{g}"] = np.asarray(sem, dtype=float)
            for arm in ARMS:
                f = holder[arm]
                arrays[f"{region}__model_arm{arm}_uM__{g}"] = np.asarray(
                    f.model_uM[region][g], dtype=float)
                arrays[f"{region}__residual_arm{arm}_uM__{g}"] = np.asarray(
                    f.residuals_uM[region][g], dtype=float)
    for region, holder in (("central_complex", fits), ("mushroom_body_heel", fits_heel)):
        for arm in ARMS:
            f = holder[arm]
            for g in GROUPS:
                res = simulate(protocol, f.params.group_kinetics(g),
                               sample_dt_s=sample_dt_s)
                arrays[f"trace__{region}__arm{arm}__{g}__time_s"] = res.record_time_s[0]
                arrays[f"trace__{region}__arm{arm}__{g}__uM"] = (
                    res.record_nM * NM_TO_UM * float(f.scales[region]))
    grid = report["_grid"]
    for name in ("efficacy", "release_tau_s", "clearance_tau_s",
                 "pool_capacity_pmol", "refill_tau_s", "efficacy_multiplier",
                 "clearance_multiplier"):
        arrays[f"grid__{name}"] = np.asarray(getattr(grid, name), dtype=float)
    arrays["stimulation_number"] = np.arange(1, N_STIMULATIONS + 1)
    np.savez_compressed(out_path, **arrays)
    return str(out_path)


def save_outputs(report, out_dir=OUT_DIR, figure=True, traces=True, dpi=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if traces:
        written["traces"] = save_traces(report, out_dir / TRACES_PATH.name)
    if figure:
        fig_path, layout_checks = make_figure(report, out_dir / PNG_PATH.name, dpi=dpi)
        written["figure"] = fig_path
        written["figure_layout_checks"] = layout_checks
    payload = {k: v for k, v in report.items() if not k.startswith("_")}
    payload["written_files"] = dict(written)
    payload["run_stats"]["peak_rss_mb"] = peak_memory_mb()
    path = out_dir / REPORT_PATH.name
    with path.open("w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    written["report"] = str(path)
    return written


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Round 5 real-data comparison "
                                            "(see the module docstring)")
    ap.add_argument("--quick", action="store_true",
                    help="use a coarser grid (still bounded and deterministic)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    data = load_digitised_data()
    protocol = protocol_from_digitised(data)
    grid = default_grid(fine=not args.quick)
    report = build_report(data, protocol, grid, seed=args.seed, quick=args.quick)
    written = save_outputs(report, figure=not args.no_figure,
                           traces=not args.no_traces, dpi=args.dpi)
    wall = time.perf_counter() - t0
    print("\nwritten files:")
    for k, v in written.items():
        print(f"  {k:8s} {v}")
    print(f"wall clock total: {wall:.1f} s; peak RSS: {peak_memory_mb():.0f} MiB")
    cc = report["fits"]["central_complex"]["A"]
    hh = report["fits"]["mushroom_body_heel"]["A"]
    print(f"arm A RMS: central complex {cc['rms_overall_uM']:.4f} uM, heel "
          f"{hh['rms_overall_uM']:.4f} uM (reading uncertainty "
          f"{READING_UNCERTAINTY_UM} uM)")
    print(f"search model-curve evaluations (arm A, central complex): "
          f"{cc['n_model_curve_evaluations']}")
    c = report["critical_test"]["central_complex"]
    print(f"critical test: {c['grid_stim1_pairs_with_elevation']} of "
          f"{c['grid_stim1_pairs_enumerated']} stim-1 grid pairs can elevate a group, "
          f"all requiring pool capping = "
          f"{c['every_elevating_pair_requires_pool_capping_of_the_reference_group']}")
    print(f"cross-region: stim-1 pairs matching BOTH orderings = "
          f"{report['cross_region']['stim1_R0_pairs_matching_BOTH_orderings']} of "
          f"{report['cross_region']['stim1_R0_pairs_enumerated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
