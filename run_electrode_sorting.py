#!/usr/bin/env python
"""Runner: SOURCE SEPARATION (spike sorting / demultiplexing) -- BOUNDED, REDUCED SCOPE.

WHAT THIS RUNS AND WHY IT IS SMALL
----------------------------------
The full arm (256 channels x 20475 captured somata x 500 samples at 20 kHz) is
NOT run here: a 10 GB OOM was confirmed on it, and the instruction after that was
to inspect, report and run only TINY tests.  This runner therefore executes the
SAME code path as the full arm -- ``engine/electrode_sorting``, the access-map
geometry code, the reused solver -- on two explicitly REDUCED fixtures:

  FIXTURE A "real_tiny"      REAL BANC soma coordinates, %d channels on a
                             linear_shank_x at the FIXED 20 um pitch and the FIXED
                             7 um contact, capture radius 50 um DECLARED, and the
                             %d nearest real somata to the array centre.
  FIXTURE B "synthetic_tiny" A DECLARED synthetic geometry: %d channels on a 4x4
                             planar grid at a DECLARED pitch and %d sources at
                             DECLARED offsets, so the separation problem can be
                             turned up and down ON PURPOSE (sources 0 um apart up
                             to 120 um apart).  NO coordinate in this fixture is a
                             real neuron, and it is labelled as such everywhere.

SCOPE LIMITS, DECLARED UP FRONT
-------------------------------
  * <= 16 channels, <= 32 neurons, <= 0.5 s of trace per trial;
  * the process sets its own hard RLIMIT_AS and pins BLAS to ONE thread;
  * consequently the numbers here are NOT the whole-cloud numbers: 20475 captured
    somata, and the maximum multiplicity of 21 channels seeing ONE neuron, are
    NOT represented.  Fixture A reaches at most 8 channels per neuron, so the
    ambiguity is represented at REDUCED multiplicity and every result is
    fixture-local.

THE NOISE-FLOOR BOOKKEEPING, RESOLVED (this replaces a previously WRONG note)
-----------------------------------------------------------------------------
The front end's recorded 24.018 uV RMS is the OPEN-INPUT integral (R_in = inf) of
4 k T Re[Z_e] over 1e-6..1e4 Hz; the RECORDED SIGNAL was computed with the LOADED
input R_in = 10 MOhm, whose integral over the same band is 5.626 uV.  Both values
are CORRECT for their own configuration, and this run reproduces the open-input
one to <1e-3 relative from its own stated parameters.  The mismatch is therefore
open-vs-loaded between two artefacts -- NOT a unit error and NOT an unresolved
inconsistency (an earlier revision of this file wrongly called it unresolved).
Consequence: a "65x noise/signal" ratio is NOT this model's ratio, because it
divides an OPEN-input noise by a LOADED signal.  The like-for-like loaded figure
is ~15x, and the per-neuron matched-filter SNR is lower still.

WHAT THIS RUN IS NOT
--------------------
Hand-built research prototype.  NOT a validated device model and NOT a validated
spike sorter.  Somata are single voxel POINTS: no morphology, no per-neuron
waveform, no membrane.  The spike waveform is DECLARED.  The primary sorter is an
ORACLE template matcher -- it is handed the SAME forward matrix that generated the
data, so within this model it is the best an estimator can do, and its failure at
a given noise level does NOT bound a real sorter, which has morphology,
per-neuron waveforms and a different noise budget that this model lacks.  No
claim is made that any neuron is recorded from a real animal, and none whatever
about behaviour, perception, attention, recognition, experience, consciousness,
identity or immortality.

ATTRIBUTION (CC BY 4.0 -- A LICENCE CONDITION)
----------------------------------------------
Fixture A consumes real soma coordinates: "Distributed control circuits across a
brain-and-cord connectome", Harvard Dataverse, doi:10.7910/DVN/7WTH1N, file
somas_v1.parquet (file id 13916460), CC BY 4.0
(http://creativecommons.org/licenses/by/4.0).  Redistribution of this figure or
these numbers MUST carry that statement.

Run:  PYTHONPATH=/tmp/pq SORTING_FIXTURE=both venv/bin/python run_electrode_sorting.py
      (env: SORTING_FIXTURE=real|synthetic|both, MEM_LIMIT_GB default 2)
"""
from __future__ import annotations

import json
import math
import os
import platform
import resource
import sys
import time

# ---------------------------------------------------------------------------
# RESOURCE LIMITS FIRST, before numpy is imported: this process must not be able
# to repeat the 10 GB OOM.  Both are declared and both are printed in the output.
# ---------------------------------------------------------------------------
MEM_LIMIT_GB = float(os.environ.get("MEM_LIMIT_GB", "2"))
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    _want = int(MEM_LIMIT_GB * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS,
                       (_want if _hard == resource.RLIM_INFINITY
                        else min(_want, _hard), _hard))
    _applied = resource.getrlimit(resource.RLIMIT_AS)[0]
except (ValueError, OSError) as exc:      # pragma: no cover
    _applied = "NOT APPLIED: %s" % exc
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

MUJOCO_NOTE = "not needed: nothing here renders a scene, so MUJOCO_GL is irrelevant"

OUT_DIR = "outputs/embodied_body"
JSON_PATH = os.path.join(OUT_DIR, "electrode_sorting.json")
PNG_PATH = os.path.join(OUT_DIR, "electrode_sorting.png")

FIXTURE = os.environ.get("SORTING_FIXTURE", "both")
SHANK_N_CHANNELS = 8          # FIXTURE A: <= 16 by instruction
SHANK_NEURONS = 30            # FIXTURE A: <= 32 by instruction
SYNTH_N_CHANNELS = 16         # FIXTURE B: at the limit, not above
SYNTH_NEURONS = 16            # FIXTURE B
SYNTH_PITCH_UM = 16.0
SYNTH_OFFSETS_UM = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0,
                    22.0, 26.0, 32.0, 40.0, 55.0, 70.0, 90.0, 120.0)
MAX_TRACE_SECONDS = 0.5

# ---------------------------------------------------------------------------
# FIXED geometry (user decision) and DECLARED model inputs.
# ---------------------------------------------------------------------------
PITCH_UM = 20.0
DIAMETER_UM = 7.0
CAPTURE_RADIUS_UM = 50.0
CONDUCTIVITY_S_M = 0.3
C_DL_UF_PER_CM2 = 20.0
RHO_CT_OHM_CM2 = 385.0
TEMPERATURE_K = 300.0
AMPLIFIER_INPUT_OHM = 10.0e6
SPIKE_SIGMA_MS = 0.5
SPIKE_PEAK_NA = 0.01
DT_MS = 0.05                  # 20 kHz
N_STEPS = 500                 # 25 ms per trial (< 0.5 s)
WAVEFORM_HALF_WIDTH_MS = 2.0
TOLERANCE_MS = 0.6
FAR_PER_TEST = 1e-6
MAX_PER_TIME = 3
N_TRIALS = 2
NEURONS_IN_WINDOW = 2         # small: 2 of <=30 neurons fire per trial
NOISE_BAND_HZ = (1e-6, 10e3)
NOISE_LEVELS_UV = (1e-3, 0.1, 1.0, 5.6203, 24.0182, 50.0)
RECORDED_FLOOR_UV = 24.01816009741456
FRONTEND_RECORDED_RATIO = 65.37272265955913
FRONTEND_MEDIAN_SIGNAL_UV = 0.36740339273451544

# ===========================================================================
# PRE-REGISTERED PREDICTIONS -- declared in this source BEFORE anything is
# computed, never retuned afterwards.
# ===========================================================================
PRE_REGISTERED = [
    ("PS1", "At the noise levels this chain computes (5.62 uV output-referred, and "
            "the 24.02 uV open-input integral), ZERO percent of the SHARED "
            "(multi-channel) neurons are resolvable: no shared neuron has half its "
            "spikes recovered with the right identity. Declared as an exact zero. "
            "This is a MODEL-SPECIFIC statement (point sources, one DECLARED "
            "waveform, homogeneous ohmic medium, this noise budget) and is NOT a "
            "claim about any real device.",
     "resolvable_fraction_at_floor_eq", 0.0),
    ("PS2", "The detection rate never INCREASES with noise across the sweep "
            "(equality allowed).",
     "nonincreasing_in_noise", None),
    ("PS3", "Matched-filter SNR, not a per-channel amplitude ratio, is the binding "
            "quantity: the median matched-filter SNR of the fixture neurons at the "
            "chain's own noise level is below 1.0.",
     "median_mf_snr_lt", 1.0),
    ("PS4", "The ADDED electronic crosstalk is negligible next to the spatial "
            "overlap: removing it changes the detection rate by less than 10% "
            "relative.",
     "crosstalk_effect_le", 0.10),
    ("PS5", "Assignment, not detection, is the first thing to fail: whenever the "
            "shared detection rate is between 10% and 90%, assignment accuracy on "
            "the detected shared spikes is below 50%.",
     "assignment_lt_half_in_transition", 0.50),
    ("PS6", "Two sources CLOSER together than the channel pitch are harder to tell "
            "apart than two further apart: in the SAME synthetic fixture the "
            "template coherence of the closest declared pair exceeds that of the "
            "farthest pair.",
     "coherence_decreases_with_distance", None),
    ("PS7", "The ORACLE arm beats the calibration arm (templates ESTIMATED from "
            "one-neuron-at-a-time recordings) at the same noise level; the size of "
            "the gap is reported.",
     "calibration_worse_than_oracle", None),
    ("PS8", "The open-vs-loaded interface difference FULLY explains the 24.018 uV "
            "vs 5.626 uV discrepancy: the same interface with R_in = inf gives "
            "24.018 uV and with R_in = 10 MOhm gives 5.626 uV, to within 1%.",
     "open_loaded_explains_floor", 0.01),
]


def _wrap(text, width=76, indent="      "):
    out, line = [], ""
    for word in str(text).split():
        if len(line) + len(word) + 1 > width:
            out.append(indent + line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(indent + line)
    return out


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def verdict(kind, threshold, results):
    """Evaluate ONE pre-registered prediction.  Thresholds are read from the
    declaration above and cannot be changed here."""
    r = results or {}
    if kind == "resolvable_fraction_at_floor_eq":
        v = r.get("resolvable_fraction_at_floor")
        if v is None:
            return "EVIDENCE NOT FOUND", "the chain noise level was not measured"
        return ("PASS" if float(v) == 0.0 else "FAIL",
                "measured %.6f vs declared exactly 0.0" % v)
    if kind == "nonincreasing_in_noise":
        rows = sorted([x for x in (r.get("sweep_rows") or [])
                       if x.get("shared_detection_rate_mean") is not None],
                      key=lambda x: x["noise_sd_uV"])
        if len(rows) < 2:
            return "EVIDENCE NOT FOUND", "fewer than two sweep rows"
        bad = [(rows[i]["noise_sd_uV"], rows[i + 1]["noise_sd_uV"])
               for i in range(len(rows) - 1)
               if rows[i + 1]["shared_detection_rate_mean"]
               > rows[i]["shared_detection_rate_mean"] + 1e-12]
        return ("PASS" if not bad else "FAIL",
                "%d increases over %d successive pairs" % (len(bad), len(rows) - 1))
    if kind == "median_mf_snr_lt":
        v = r.get("median_mf_snr_at_chain_level")
        if v is None:
            return "EVIDENCE NOT FOUND", "no matched-filter SNR computed"
        return ("PASS" if float(v) < float(threshold) else "FAIL",
                "measured median MF-SNR %.4g vs declared < %.4g" % (v, threshold))
    if kind == "crosstalk_effect_le":
        v = r.get("crosstalk_relative_effect_low_noise")
        if v is None:
            return "EVIDENCE NOT FOUND", "the crosstalk arm did not run"
        return ("PASS" if float(v) <= float(threshold) else "FAIL",
                "measured relative change %.4g vs declared <= %.4g" % (v, threshold))
    if kind == "assignment_lt_half_in_transition":
        rows = [x for x in (r.get("sweep_rows") or [])
                if x.get("shared_detection_rate_mean") is not None
                and 0.10 < x["shared_detection_rate_mean"] < 0.90]
        if not rows:
            return "EVIDENCE NOT FOUND", ("no level put the shared detection rate "
                                          "strictly between 10% and 90%")
        worst = max((x.get("shared_assignment_accuracy_mean") or 0.0) for x in rows)
        return ("PASS" if worst < float(threshold) else "FAIL",
                "max shared assignment accuracy in the band %.4f vs declared < %.4f "
                "(%d row(s))" % (worst, threshold, len(rows)))
    if kind == "coherence_decreases_with_distance":
        c = r.get("synthetic_coherence") or {}
        if not c or c.get("closest_pair_coherence") is None:
            return "EVIDENCE NOT FOUND", "the synthetic fixture did not run"
        a, b = c["closest_pair_coherence"], c.get("far_pair_coherence")
        if b is None:
            return "EVIDENCE NOT FOUND", "no far pair in the fixture"
        return ("PASS" if a > b else "FAIL",
                "closest pair (%.4g um apart) |cos| %.6f vs far pair (%.4g um) "
                "%.6f" % (c.get("closest_pair_separation_um"), a,
                          c.get("far_pair_separation_um"), b))
    if kind == "calibration_worse_than_oracle":
        a = r.get("calibration_arm") or {}
        o = r.get("oracle_arm_at_same_noise") or {}
        fa, fo = a.get("resolvable_fraction_mean"), o.get("resolvable_fraction_mean")
        if fa is None or fo is None:
            return "EVIDENCE NOT FOUND", "the calibration arm produced no result"
        return ("PASS" if fa < fo else "FAIL",
                "calibration %.4f vs oracle %.4f" % (fa, fo))
    if kind == "open_loaded_explains_floor":
        v = r.get("open_loaded_relative_residual")
        if v is None:
            return "EVIDENCE NOT FOUND", "the two interface configurations did not run"
        return ("PASS" if abs(float(v)) <= float(threshold) else "FAIL",
                "open-input %.6f uV vs recorded %.6f uV, relative residual %.3g vs "
                "declared <= %.3g" % (r["open_input_band_uV"], RECORDED_FLOOR_UV,
                                      float(v), threshold))
    return "EVIDENCE NOT FOUND", "unknown test kind %r" % (kind,)


def _fake_access_map(access_map_mod, centres_um, n_ch_per_neuron, neuron_count,
                     pitch_um, spec, resolution):
    """An AccessMap carrying ONLY the channel geometry and the capture counts.

    The synthetic fixture has no parquet rows behind it, so the soma-table fields
    are empty and no coordinate in it is claimed to be real.  The CLASS is the
    same one the real fixture uses, so the separation code path is identical.
    """
    A = access_map_mod.AccessMap
    return A(spec=spec, resolution=resolution,
             channel_centres_vox=resolution.um_to_voxels(centres_um),
             channel_centres_um=centres_um,
             channel_neurons=tuple(() for _ in range(centres_um.shape[0])),
             channel_neurons_um=tuple(() for _ in range(centres_um.shape[0])),
             soma_root_ids=np.zeros(0, np.int64),
             soma_positions_vox=np.zeros((0, 3), np.int64),
             neuron_first_channel=np.zeros(neuron_count, np.int64),
             neuron_n_channels=np.asarray(n_ch_per_neuron, np.int64),
             reports={"synthetic_fixture": True, "pitch_um": pitch_um,
                      "note": ("DECLARED synthetic geometry: NO soma coordinate in "
                               "this fixture is real; only the electrode spec, the "
                               "voxel frame and the reused solver are.")},
             ledger=access_map_mod.Ledger())


def _run_fixture_a(access_map_mod, frontend, S, contact, interface, resolution, np):
    """FIXTURE A: real BANC coordinates, 8 channels, 30 somata.  BOUNDED."""
    from engine.embodied.adapters import REPO_ROOT, TierConfig, build_neural_tier
    from engine.neck_cut_data import load_banc

    info = {"name": "real_tiny", "soma_coordinates_real": True,
            "n_channels": SHANK_N_CHANNELS, "n_neurons": SHANK_NEURONS}
    cfg = TierConfig()
    ann = load_banc(os.path.join(REPO_ROOT, cfg.data_path))
    root_ids = np.asarray(ann["root_ids"])
    tier_built = build_neural_tier(cfg)
    tier_root_ids = root_ids[np.asarray(tier_built["global_index"])]
    somas_path = os.path.join(REPO_ROOT, "data", "banc", "somas_v1.parquet")
    soma = access_map_mod.load_soma_table(somas_path)

    spec = access_map_mod.ElectrodeArraySpec(
        pitch_um=PITCH_UM, diameter_um=DIAMETER_UM, n_channels=SHANK_N_CHANNELS,
        capture_radius_um=CAPTURE_RADIUS_UM, geometry="linear_shank_x",
        placement="densest_readout_soma", seed=0)
    am = access_map_mod.build_access_map(spec, tier_root_ids, somas_path,
                                        resolution, soma_table=soma)
    centres = np.asarray(am.channel_centres_um, float)
    centre = centres.mean(axis=0)
    all_um = resolution.voxels_to_um(soma.positions_vox)
    # SELECT FROM THE CAPTURED POPULATION.  A 50 um DECLARED capture radius means
    # only the somata inside a channel sphere are addressable at all; picking the
    # 30 nearest somata of the WHOLE 153892-row cloud would pick neurons hundreds
    # of micrometres away that no channel captures, and the gain matrix would be
    # empty (that is exactly how this was found).
    captured_rows = np.flatnonzero(np.asarray(am.neuron_n_channels) > 0)
    if captured_rows.size < SHANK_NEURONS:
        raise RuntimeError("the array captures only %d somata: increase the channel "
                           "count rather than the scope" % captured_rows.size)
    d_all = np.linalg.norm(all_um[captured_rows] - centre, axis=1)
    nearest = captured_rows[np.argsort(d_all)[:SHANK_NEURONS]]
    d = np.zeros(all_um.shape[0])
    d[captured_rows] = d_all
    info["array_extent_um"] = [float(v) for v in am.extent_um]
    info["array_origin_vox"] = [float(v) for v in am.ledger.get("array_origin_vox").value]
    info["radius_of_selected_somata_um"] = {
        "min": float(d[nearest].min()), "median": float(np.median(d[nearest])),
        "max": float(d[nearest].max())}
    info["n_captured_somata_available"] = int(captured_rows.size)
    info["selection_rule"] = ("the %d nearest CAPTURED somata to the array centre "
                              "(capture is the access map's own 50 um DECLARED "
                              "predicate)" % SHANK_NEURONS)
    info["access_map_report"] = {
        "captured_neurons_in_tier": int(np.sum(np.asarray(am.neuron_n_channels) > 0)),
        "note": ("the access map is built over the tier; the GAIN matrix below is "
                 "built only over the %d selected somata, which is this run's "
                 "bounded scope" % SHANK_NEURONS)}

    gain_matrix, gain_info = S.per_neuron_gain_matrix(
        am, neuron_current_nA=SPIKE_PEAK_NA, conductivity_S_m=CONDUCTIVITY_S_M,
        contact=contact, neuron_rows=nearest,
        quadrature=frontend.ContactQuadrature(4, 16))
    keep_ch = np.flatnonzero(np.abs(gain_matrix).max(axis=1) > 0)
    gain_matrix = gain_matrix[keep_ch]
    centres = centres[keep_ch]
    # the SignalModel checks the channel count against the map, so the map is
    # TRIMMED to the channels that actually have signal -- and the spec's
    # n_channels is updated so nothing reports a count the gain matrix lacks.
    import dataclasses as _dc
    am = _dc.replace(am, channel_centres_um=centres,
                     channel_centres_vox=np.asarray(am.channel_centres_vox)[keep_ch],
                     channel_neurons=tuple(am.channel_neurons[k] for k in keep_ch),
                     channel_neurons_um=tuple(am.channel_neurons_um[k] for k in keep_ch),
                     spec=_dc.replace(am.spec, n_channels=int(centres.shape[0])),
                     reports=dict(am.reports, trimmed_to_channels_with_signal=[
                         int(v) for v in keep_ch]))
    info["n_channels_with_signal"] = int(keep_ch.size)
    n_ch_of = (np.abs(gain_matrix) > 0).sum(axis=0)
    info["channels_per_neuron_histogram"] = {
        str(int(k)): int(v) for k, v in zip(*np.unique(n_ch_of, return_counts=True))}
    info["max_channels_per_neuron"] = int(n_ch_of.max())
    info["gain_info"] = gain_info
    info["gain_peak_mV"] = float(np.abs(gain_matrix).max())
    return info, gain_matrix, centres, n_ch_of > 1, am


def _run_fixture_b(S, resolution, access_map_mod, frontend, contact, np):
    """FIXTURE B: DECLARED synthetic geometry, 16 channels, 16 sources."""
    n_side = int(round(math.sqrt(SYNTH_N_CHANNELS)))
    p_vox = SYNTH_PITCH_UM * 1000.0 / resolution.nm_per_voxel
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="xy")
    gx = gx.ravel().astype(float) - (n_side - 1) / 2.0
    gy = gy.ravel().astype(float) - (n_side - 1) / 2.0
    centres_vox = np.column_stack((gx * p_vox[0], gy * p_vox[1],
                                  np.zeros_like(gx)))[:SYNTH_N_CHANNELS]
    centres_um = resolution.voxels_to_um(centres_vox)
    offs = np.asarray(SYNTH_OFFSETS_UM, float)
    src_vox = np.zeros((offs.size, 3))
    src_vox[:, 0] = offs * 1000.0 / resolution.nm_per_voxel[0]
    src_um = resolution.voxels_to_um(src_vox)
    info = {"name": "synthetic_tiny", "soma_coordinates_real": False,
            "n_channels": int(centres_um.shape[0]), "n_neurons": int(src_um.shape[0]),
            "pitch_um": SYNTH_PITCH_UM,
            "declared_source_offsets_um": [float(v) for v in offs],
            "note": ("DECLARED synthetic geometry. The electrode spec, the voxel "
                     "frame and engine.electrode.transfer_mV_per_nA are the real "
                     "ones; NO source coordinate is a real neuron.")}
    spec = access_map_mod.ElectrodeArraySpec(
        pitch_um=SYNTH_PITCH_UM, diameter_um=DIAMETER_UM,
        n_channels=int(centres_um.shape[0]), capture_radius_um=CAPTURE_RADIUS_UM,
        geometry="planar_grid_xy", placement="explicit_origin",
        origin_vox=(0.0, 0.0, 0.0), seed=0)
    quad = frontend.ContactQuadrature(4, 16)
    src = [frontend.Contact(tuple(float(v) for v in p), 1.0) for p in src_um]
    G = np.zeros((centres_um.shape[0], src_um.shape[0]))
    for k in range(centres_um.shape[0]):
        xyz, w = frontend.contact_points_um(contact, source_um=tuple(centres_um[k]),
                                           normal=(0.0, 0.0, 1.0), quadrature=quad)
        F = frontend.transfer_mV_per_nA(xyz, src, CONDUCTIVITY_S_M)
        G[k] = ((w @ F) / w.sum()) * SPIKE_PEAK_NA
    n_ch_of = (np.abs(G) > 0).sum(axis=0)
    info["max_channels_per_neuron"] = int(n_ch_of.max())
    info["gain_peak_mV"] = float(np.abs(G).max())
    return info, G, centres_um, n_ch_of > 1, offs, spec


def _evaluate_fixture(tag, model, is_shared, S, contact, interface, extra=None):
    """Sweep, baselines, crosstalk arm and calibration arm -- all bounded."""
    out = {"tag": tag}
    rows = []
    for i, nl in enumerate(NOISE_LEVELS_UV):
        row = S.run_noise_level(
            model, noise_sd_mV=nl * 1e-3, neuron_is_shared=is_shared,
            n_trials=N_TRIALS, neurons_in_window=NEURONS_IN_WINDOW,
            method="omp_matched_filter", tolerance_ms=TOLERANCE_MS, dt_ms=DT_MS,
            far_per_test=FAR_PER_TEST, max_per_time=MAX_PER_TIME, seed=500 + i,
            compare_methods=("nearest_channel_wta", "global_threshold"))
        row["index"] = i
        rows.append(row)
        print("  [%s] noise %8.4g uV : det %5.3f assign %5s prec %5s "
              "shared-resolvable %5.3f (dets/trial %.1f)"
              % (tag, nl, row["detection_rate_mean"] or 0.0,
                 "n/a" if row["assignment_accuracy_mean"] is None
                 else "%.3f" % row["assignment_accuracy_mean"],
                 "n/a" if row["precision_mean"] is None
                 else "%.3f" % row["precision_mean"],
                 row["shared_resolvable_fraction_mean"] or 0.0,
                 row["n_detections_mean"] or 0.0))
    out["sweep_rows"] = rows
    floor_row = min(rows, key=lambda r: abs(r["noise_sd_uV"] - 5.6203))
    out["resolvable_fraction_at_floor"] = floor_row["shared_resolvable_fraction_mean"]
    out["floor_level_uV"] = floor_row["noise_sd_uV"]
    band = [x for x in rows if x.get("shared_detection_rate_mean") is not None
            and 0.10 < x["shared_detection_rate_mean"] < 0.90]
    out["assignment_in_transition_band"] = (
        max((x.get("shared_assignment_accuracy_mean") or 0.0) for x in band)
        if band else None)

    mf = S.matched_filter_snr(model, noise_sd_mV=5.6203e-3)
    out["matched_filter_snr"] = mf
    out["matched_filter_snr_at_chain_level"] = mf["snr_at_that_noise_median"]

    geo = S.resolvability_geometry(model.templates, model.waveform_nA,
                                   neuron_is_shared=is_shared, max_neurons=64,
                                   n_pairs_per_neuron=64, seed=0)
    out["geometry_resolvability"] = {k: v for k, v in geo.items()
                                     if k != "_coherence_sample"}
    out["coherence_sample"] = [float(v) for v in geo["_coherence_sample"][:4000]]

    # ---- crosstalk arm: what does the ADDED leakage contribute? -----------
    saved_C, saved_T = model.C_add, model.templates
    try:
        model.C_add = np.eye(model.C_add.shape[0])
        model.templates = model.G.copy()
        row_nc = S.run_noise_level(
            model, noise_sd_mV=NOISE_LEVELS_UV[0] * 1e-3,
            neuron_is_shared=is_shared, n_trials=N_TRIALS,
            neurons_in_window=NEURONS_IN_WINDOW, tolerance_ms=TOLERANCE_MS,
            dt_ms=DT_MS, far_per_test=FAR_PER_TEST, max_per_time=MAX_PER_TIME,
            seed=500)
        b, a = rows[0]["detection_rate_mean"], row_nc["detection_rate_mean"]
        out["crosstalk_relative_effect_low_noise"] = (
            None if (b is None or a is None or abs(b) < 1e-12)
            else float(abs(a - b) / abs(b)))
        out["crosstalk_arm"] = row_nc
        out["crosstalk_arm_note"] = (
            "the medium-inherent coupling is NOT added anywhere: it is already "
            "inside the volume-conduction term. Only the ADDED electronic leakage "
            "(ring-1 median 7.42e-4) is applied as a channel-to-channel matrix, and "
            "this arm removes exactly that.")
    finally:
        model.C_add, model.templates = saved_C, saved_T

    # ---- calibration (non-oracle) arm -------------------------------------
    rng = np.random.default_rng(7)
    T_est, cal_report = S.estimate_templates_from_calibration(
        model, n_calibrated=min(8, model.G.shape[1]), rng=rng, min_snr=1.0,
        noise_sd_mV=1e-3)
    cols = np.flatnonzero(np.abs(T_est).max(axis=0) > 0)
    if cols.size:
        m_est = S.SignalModel(model.G[:, cols], model.access_map, dt_ms=DT_MS,
                              n_steps=N_STEPS, contact=contact, interface=interface,
                              conductivity_S_m=CONDUCTIVITY_S_M,
                              spike_sigma_ms=SPIKE_SIGMA_MS,
                              spike_peak_nA=SPIKE_PEAK_NA, noise_sd_mV=1e-3,
                              noise_band_Hz=NOISE_BAND_HZ)
        m_est.C_add = model.C_add.copy()
        m_est.templates = T_est[:, cols]
        m_est._tnorm = np.linalg.norm(m_est.templates, axis=0) * m_est._wnorm
        m_or = S.SignalModel(model.G[:, cols], model.access_map, dt_ms=DT_MS,
                             n_steps=N_STEPS, contact=contact, interface=interface,
                             conductivity_S_m=CONDUCTIVITY_S_M,
                             spike_sigma_ms=SPIKE_SIGMA_MS,
                             spike_peak_nA=SPIKE_PEAK_NA, noise_sd_mV=1e-3,
                             noise_band_Hz=NOISE_BAND_HZ)
        sh = is_shared[cols]
        n_win = 1
        cal = S.run_noise_level(m_est, noise_sd_mV=1e-3, neuron_is_shared=sh,
                                n_trials=N_TRIALS, neurons_in_window=n_win,
                                tolerance_ms=TOLERANCE_MS, dt_ms=DT_MS,
                                far_per_test=FAR_PER_TEST, max_per_time=MAX_PER_TIME,
                                seed=31)
        orc = S.run_noise_level(m_or, noise_sd_mV=1e-3, neuron_is_shared=sh,
                                n_trials=N_TRIALS, neurons_in_window=n_win,
                                tolerance_ms=TOLERANCE_MS, dt_ms=DT_MS,
                                far_per_test=FAR_PER_TEST, max_per_time=MAX_PER_TIME,
                                seed=31)
        out["calibration_arm"] = {
            "report": cal_report, "n_neurons_kept": int(cols.size),
            "resolvable_fraction_mean": cal["shared_resolvable_fraction_mean"],
            "detection_rate_mean": cal["detection_rate_mean"]}
        out["oracle_arm_at_same_noise"] = {
            "resolvable_fraction_mean": orc["shared_resolvable_fraction_mean"],
            "detection_rate_mean": orc["detection_rate_mean"]}
    else:
        out["calibration_arm"] = {"report": cal_report,
                                  "resolvable_fraction_mean": None}
    if extra:
        out.update(extra)
    return out


def main():
    t_start = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    os.chdir(here)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from engine.embodied import access_map as access_map_mod
    from engine.embodied.access_map import banc_voxel_resolution
    from engine.electrode_frontend import (ContactGeometry, InterfaceImpedance,
                                          SOMA_ATTRIBUTION, thermal_noise_variance)
    import engine.electrode_frontend as frontend
    import engine.electrode_sorting as S

    print("=" * 78)
    print("SOURCE SEPARATION -- BOUNDED RUN, REDUCED SCOPE (NOT the whole-cloud arm)")
    print("=" * 78)
    print("FIXED geometry: shaft diameter %.4g um, channel pitch %.4g um, capture "
          "radius %.4g um DECLARED" % (DIAMETER_UM, PITCH_UM, CAPTURE_RADIUS_UM))
    print("SCOPE LIMITS  : <= %d channels, <= %d neurons, %.0f ms per trial, "
          "RLIMIT_AS=%s bytes, BLAS threads=1"
          % (max(SHANK_N_CHANNELS, SYNTH_N_CHANNELS),
             max(SHANK_NEURONS, SYNTH_NEURONS), N_STEPS * DT_MS, _applied))
    print("  -> a 10 GB OOM was confirmed on the full 256-channel x 20475-neuron arm;")
    print("     this run is deliberately bounded and says so. The whole-cloud numbers")
    print("     (up to 21 channels per neuron) are NOT reproduced here.")
    print()
    print("attribution (CC BY 4.0, REQUIRED on redistribution):")
    print("  " + SOMA_ATTRIBUTION["citation_string"])
    print()
    print("PRE-REGISTERED PREDICTIONS (declared in this source before the run):")
    for pid, stmt, kind, thr in PRE_REGISTERED:
        print("  %s [%s thr=%r]" % (pid, kind, thr))
        for line in _wrap(stmt, indent="      "):
            print(line)
    print()

    out = {
        "generated_by": "run_electrode_sorting.py (bounded, reduced scope)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "host_cwd": os.getcwd(),
        "mujoco_note": MUJOCO_NOTE,
        "PROTOTYPE_NOTICE": S.DISCLAIMER,
        "attribution": dict(S.SORTING_ATTRIBUTION),
        "attribution_required": True,
        "scope": {
            "fixture": FIXTURE,
            "bounded": True,
            "why": ("a 10 GB OOM was confirmed on the full arm (256 channels x 20475 "
                    "captured somata x 500 samples at 20 kHz per OMP iteration); the "
                    "instruction after that was inspect/report plus tiny tests only"),
            "limits": {"max_channels": SYNTH_N_CHANNELS,
                       "max_neurons": SHANK_NEURONS,
                       "max_trace_seconds": MAX_TRACE_SECONDS,
                       "rlimit_as_bytes": _applied, "blas_threads": 1},
            "not_reproduced_here": [
                "the whole-cloud population (20475 captured somata)",
                "the maximum multiplicity: up to 21 channels seeing ONE neuron",
                "the 256-channel array and its full crosstalk matrix",
                "any whole-cloud sweep statistic; every number below is fixture-local",
            ],
            "run_profile": {
                "n_noise_levels": len(NOISE_LEVELS_UV),
                "noise_levels_uV": list(NOISE_LEVELS_UV),
                "n_trials_per_level": N_TRIALS,
                "neurons_firing_per_trial": NEURONS_IN_WINDOW,
                "dt_ms": DT_MS, "n_steps": N_STEPS,
                "spike_sigma_ms": SPIKE_SIGMA_MS, "spike_peak_nA": SPIKE_PEAK_NA},
        },
        "pre_registered": [{"id": p[0], "statement": p[1], "test": p[2],
                            "threshold": p[3]} for p in PRE_REGISTERED],
    }

    # ---- the interface, both ways: this is the open-vs-loaded fix ---------
    resolution = banc_voxel_resolution()
    contact = ContactGeometry.from_diameter(DIAMETER_UM)
    loaded = InterfaceImpedance(contact, conductivity_S_m=CONDUCTIVITY_S_M,
                                c_dl_uF_per_cm2=C_DL_UF_PER_CM2,
                                rho_ct_ohm_cm2=RHO_CT_OHM_CM2,
                                temperature_K=TEMPERATURE_K,
                                amplifier_input_ohm=AMPLIFIER_INPUT_OHM)
    opened = InterfaceImpedance(contact, conductivity_S_m=CONDUCTIVITY_S_M,
                                c_dl_uF_per_cm2=C_DL_UF_PER_CM2,
                                rho_ct_ohm_cm2=RHO_CT_OHM_CM2,
                                temperature_K=TEMPERATURE_K,
                                amplifier_input_ohm=math.inf)
    band = (1e-6, 1e4)
    open_uV = thermal_noise_variance(opened, f_lo_Hz=band[0], f_hi_Hz=band[1])["rms_uV"]
    loaded_uV = thermal_noise_variance(loaded, f_lo_Hz=band[0], f_hi_Hz=band[1])["rms_uV"]
    out_ref_uV, out_ref_detail = S.output_referred_noise_uV(
        loaded, n_steps=N_STEPS, dt_ms=DT_MS)
    resid = abs(open_uV - RECORDED_FLOOR_UV) / RECORDED_FLOOR_UV
    print("-" * 78)
    print("1. NOISE FLOOR RESOLVED: THE 24.018 vs 5.626 DISCREPANCY")
    print("-" * 78)
    print("OPEN  input (R_in=inf,     dc_transfer %.4f): %.6f uV RMS over %.0f..%.4g Hz"
          % (opened.dc_transfer, open_uV, band[0], band[1]))
    print("LOADED input (R_in=10 MOhm, dc_transfer %.4f): %.6f uV RMS, same band"
          % (loaded.dc_transfer, loaded_uV))
    print("recorded in the front-end JSON             : %.6f uV RMS" % RECORDED_FLOOR_UV)
    print("  -> the recorded value IS the OPEN-input integral (relative residual %.3g)"
          % resid)
    print("  -> the RECORDED SIGNAL was computed with the LOADED input. Both numbers")
    print("     are correct for their own configuration: this is an OPEN-vs-LOADED")
    print("     mismatch between two artefacts, NOT a unit error and NOT an")
    print("     unexplained inconsistency (an earlier revision wrongly said so).")
    print("  -> a '65x noise/signal' ratio is therefore NOT like-for-like: it divides")
    print("     OPEN-input noise by a LOADED signal. Loaded-consistent ratio: %.4g x;"
          % (loaded_uV / FRONTEND_MEDIAN_SIGNAL_UV))
    print("     the per-neuron matched-filter SNR is lower still (see below).")
    print("output-referred noise on THIS trace's grid (dc excluded, to Nyquist): "
          "%.6f uV RMS" % out_ref_uV)
    print()
    out["noise_floor"] = {
        "open_input_band_uV": open_uV, "loaded_interface_band_uV": loaded_uV,
        "recorded_floor_uV": RECORDED_FLOOR_UV,
        "open_loaded_relative_residual": float(resid), "band_Hz": list(band),
        "open_dc_transfer": opened.dc_transfer,
        "loaded_dc_transfer": loaded.dc_transfer,
        "output_referred_uV": out_ref_uV, "output_referred_detail": out_ref_detail,
        "explanation": ("the front-end JSON's %.6f uV RMS is the OPEN-input (R_in=inf) "
                        "integral of 4 k T Re[Z_e] over 1e-6..1e4 Hz, reproduced here "
                        "to %.3g relative. The recorded SIGNAL used the LOADED input "
                        "R_in = 10 MOhm, whose integral over the same band is %.6f uV. "
                        "The mismatch is open-vs-loaded, not a unit error."
                        % (RECORDED_FLOOR_UV, resid, loaded_uV)),
        "legacy_ratio_65x_is_not_like_for_like": {
            "frontend_recorded_ratio": FRONTEND_RECORDED_RATIO,
            "loaded_consistent_ratio": loaded_uV / FRONTEND_MEDIAN_SIGNAL_UV,
            "note": ("65x divides OPEN-input noise by a signal computed with the "
                     "LOADED input. The like-for-like loaded ratio is ~15x, and the "
                     "per-neuron matched-filter SNR is lower still because it also "
                     "pays for coherent neighbours.")},
    }

    fixtures = {}
    if FIXTURE in ("real", "both"):
        info_a, G_a, centres_a, shared_a, am_a = _run_fixture_a(
            access_map_mod, frontend, S, contact, loaded, resolution, np)
        print("-" * 78)
        print("2. FIXTURE A  real BANC coordinates, BOUNDED")
        print("-" * 78)
        print("%d channels, %d somata selected, max %d channels per neuron, "
              "peak |gain| %.4g mV"
              % (G_a.shape[0], G_a.shape[1], info_a["max_channels_per_neuron"],
                 info_a["gain_peak_mV"]))
        print("selected somata distance from the array centre: min %.3g um, median "
              "%.3g um, max %.3g um" % (info_a["radius_of_selected_somata_um"]["min"],
                                        info_a["radius_of_selected_somata_um"]["median"],
                                        info_a["radius_of_selected_somata_um"]["max"]))
        model_a = S.SignalModel(G_a, am_a, dt_ms=DT_MS, n_steps=N_STEPS,
                                contact=contact, interface=loaded,
                                conductivity_S_m=CONDUCTIVITY_S_M,
                                spike_sigma_ms=SPIKE_SIGMA_MS,
                                spike_peak_nA=SPIKE_PEAK_NA,
                                waveform_half_width_ms=WAVEFORM_HALF_WIDTH_MS,
                                noise_sd_mV=out_ref_uV * 1e-3,
                                noise_band_Hz=NOISE_BAND_HZ)
        res_a = _evaluate_fixture("real_tiny", model_a, shared_a, S, contact, loaded)
        res_a["fixture"] = info_a
        res_a["waveform_report"] = model_a.waveform_report()
        fixtures["real_tiny"] = res_a

    if FIXTURE in ("synthetic", "both"):
        info_b, G_b, centres_b, shared_b, offs, spec_b = _run_fixture_b(
            S, resolution, access_map_mod, frontend, contact, np)
        print("-" * 78)
        print("3. FIXTURE B  DECLARED SYNTHETIC geometry, BOUNDED")
        print("-" * 78)
        print("%d channels at a DECLARED %.4g um pitch, %d sources at DECLARED offsets "
              "%s um; max %d channels per source"
              % (G_b.shape[0], SYNTH_PITCH_UM, G_b.shape[1],
                 [float(v) for v in offs], info_b["max_channels_per_neuron"]))
        am_b = _fake_access_map(access_map_mod, centres_b,
                                (np.abs(G_b) > 0).sum(axis=0), G_b.shape[1],
                                SYNTH_PITCH_UM, spec_b, resolution)
        model_b = S.SignalModel(G_b, am_b, dt_ms=DT_MS, n_steps=N_STEPS,
                                contact=contact, interface=loaded,
                                conductivity_S_m=CONDUCTIVITY_S_M,
                                spike_sigma_ms=SPIKE_SIGMA_MS,
                                spike_peak_nA=SPIKE_PEAK_NA,
                                waveform_half_width_ms=WAVEFORM_HALF_WIDTH_MS,
                                noise_sd_mV=out_ref_uV * 1e-3,
                                noise_band_Hz=NOISE_BAND_HZ)
        U = model_b.templates / np.maximum(
            np.linalg.norm(model_b.templates, axis=0), 1e-300)[None, :]
        pairs = []
        for i in range(offs.size):
            for j in range(i + 1, offs.size):
                pairs.append((abs(offs[i] - offs[j]), float(abs(U[:, i] @ U[:, j])),
                              int(i), int(j)))
        pairs.sort(key=lambda p: p[0])
        syn_coh = {"closest_pair_separation_um": pairs[0][0],
                   "closest_pair_coherence": pairs[0][1],
                   "closest_pair_indices": [pairs[0][2], pairs[0][3]],
                   "far_pair_separation_um": pairs[-1][0],
                   "far_pair_coherence": pairs[-1][1],
                   "far_pair_indices": [pairs[-1][2], pairs[-1][3]],
                   "n_pairs": len(pairs),
                   "all_pairs": [[float(a), float(b)] for a, b, _, _ in pairs]}
        res_b = _evaluate_fixture("synthetic_tiny", model_b, shared_b, S, contact,
                                  loaded, extra={"synthetic_coherence": syn_coh})
        res_b["fixture"] = info_b
        res_b["waveform_report"] = model_b.waveform_report()
        fixtures["synthetic_tiny"] = res_b
        print("DECLARED-geometry coherence: closest pair (%.4g um apart) |cos| %.6f; "
              "farthest pair (%.4g um apart) |cos| %.6f"
              % (syn_coh["closest_pair_separation_um"], syn_coh["closest_pair_coherence"],
                 syn_coh["far_pair_separation_um"], syn_coh["far_pair_coherence"]))

    out["fixtures"] = fixtures
    primary = fixtures.get("real_tiny") or fixtures.get("synthetic_tiny")
    for k in ("resolvable_fraction_at_floor", "sweep_rows",
              "median_mf_snr_at_chain_level", "crosstalk_relative_effect_low_noise",
              "calibration_arm", "oracle_arm_at_same_noise",
              "assignment_in_transition_band", "matched_filter_snr",
              "geometry_resolvability", "coherence_sample"):
        if primary and k in primary:
            out[k] = primary[k]
    out["primary_fixture"] = primary.get("tag") if primary else None
    if "synthetic_tiny" in fixtures:
        out["synthetic_coherence"] = fixtures["synthetic_tiny"]["synthetic_coherence"]

    verdicts = []
    for pid, stmt, kind, thr in PRE_REGISTERED:
        v, detail = verdict(kind, thr, out)
        verdicts.append({"id": pid, "verdict": v, "detail": detail,
                         "statement": stmt, "test": kind, "threshold": thr})
    out["pre_registered_verdicts"] = verdicts
    out["pre_registered_summary"] = {
        "pass": sum(1 for v in verdicts if v["verdict"] == "PASS"),
        "fail": sum(1 for v in verdicts if v["verdict"] == "FAIL"),
        "evidence_not_found": sum(1 for v in verdicts
                                  if v["verdict"] == "EVIDENCE NOT FOUND"),
        "total": len(verdicts)}
    print()
    print("-" * 78)
    print("4. VERDICTS (thresholds as declared, never retuned)")
    print("-" * 78)
    for v in verdicts:
        print("  %s  %-19s %s" % (v["id"], v["verdict"], v["detail"]))
    print("  summary: %d PASS, %d FAIL, %d EVIDENCE NOT FOUND of %d"
          % (out["pre_registered_summary"]["pass"],
             out["pre_registered_summary"]["fail"],
             out["pre_registered_summary"]["evidence_not_found"],
             out["pre_registered_summary"]["total"]))

    # ---- ledger ----------------------------------------------------------
    led = S.SortingLedger()
    led.record("dataset.banc_somas", S.SORTING_ATTRIBUTION["dataset_title"], "name",
               S.PROV.MEASURED_CITED,
               source="%s, Harvard Dataverse, file %s (file id %d), %s"
                      % (S.SORTING_ATTRIBUTION["doi"], S.SORTING_ATTRIBUTION["file"],
                         S.SORTING_ATTRIBUTION["file_id"],
                         S.SORTING_ATTRIBUTION["licence"]),
               note=("CC BY 4.0: attribution is a LICENCE CONDITION. Fixture A uses "
                     "REAL coordinates from this dataset; fixture B uses none."))
    led.record("dataset.banc_somas_doi", S.SORTING_ATTRIBUTION["doi"], "doi",
               S.PROV.MEASURED_CITED, source="https://doi.org/10.7910/DVN/7WTH1N")
    led.record("dataset.banc_somas_licence", S.SORTING_ATTRIBUTION["licence"],
               "licence", S.PROV.MEASURED_CITED,
               source="http://creativecommons.org/licenses/by/4.0")
    led.record("geometry.pitch_um", PITCH_UM, "um", S.PROV.ENGINEERING_DEFAULT,
               source="FIXED USER DECISION, unchanged and not optimised",
               note=("FIXED. Fixture B uses a DECLARED %.4g um pitch for contrast "
                     "only." % SYNTH_PITCH_UM))
    led.record("geometry.shaft_diameter_um", DIAMETER_UM, "um",
               S.PROV.ENGINEERING_DEFAULT, source="FIXED USER DECISION",
               note="FIXED; consumed through ContactGeometry for area and noise")
    led.record("capture.radius_um", CAPTURE_RADIUS_UM, "um", S.PROV.ASSUMED,
               sweep=(10.0, 25.0, 50.0, 100.0),
               note="DECLARED, not measured; sets how many channels see one neuron")
    led.record("spike.peak_current_nA", SPIKE_PEAK_NA, "nA", S.PROV.ASSUMED,
               sweep=(0.01, 0.1, 1.0, 3.0, 10.0),
               note=("DECLARED source amplitude; every SNR scales linearly with it. "
                     "No membrane current was measured in this stack."))
    led.record("spike.sigma_ms", SPIKE_SIGMA_MS, "ms", S.PROV.ASSUMED,
               sweep=(0.25, 0.5, 1.0),
               note="DECLARED waveform width; no waveform exists in the data")
    led.record("waveform.half_width_ms", WAVEFORM_HALF_WIDTH_MS, "ms",
               S.PROV.ASSUMED, sweep=(1.0, 2.0, 4.0),
               note=("DECLARED truncation of the high-pass tail; the retained fraction "
                     "and the resulting norm error are reported in each fixture's "
                     "waveform report. Sweep range = the widths that would be tried."))
    led.record("medium.conductivity_S_m", CONDUCTIVITY_S_M, "S/m", S.PROV.ASSUMED,
               sweep=(0.1, 0.2, 0.3, 0.5, 1.0),
               note="reused unchanged from engine.electrode; not measured here")
    led.record("noise.output_referred_uV", out_ref_uV, "uV RMS",
               S.PROV.MEASURED_LOCAL,
               source=("engine.electrode_sorting.output_referred_noise_uV on this "
                       "trace's own rfft grid"),
               note=("Johnson-Nyquist through the SAME loaded interface that shapes "
                     "the spike, dc bin excluded, to Nyquist. This is the noise the "
                     "synthesised trace carries; it is a MODEL number, NOT a bench "
                     "measurement and NOT a device noise floor."))
    led.record("noise.open_input_band_uV", open_uV, "uV RMS", S.PROV.MEASURED_LOCAL,
               source="engine.electrode_frontend.thermal_noise_variance, R_in=inf",
               note=("the OPEN-input integral; it reproduces the front-end JSON's "
                     "%.6f uV to %.3g relative, which RESOLVES the open-vs-loaded "
                     "mismatch. Both configurations are correct for their own setup."
                     % (RECORDED_FLOOR_UV, resid)))
    led.record("noise.in_band_budget_status", "PENDING", "status",
               S.PROV.ENGINEERING_DEFAULT, source="this run",
               note=("a real probe/amplifier/tissue in-band noise budget does not "
                     "exist in this stack: no 1/f drift, quantisation, interference, "
                     "common-mode conversion or amplifier voltage/current noise is "
                     "modelled."))
    led.record("sorter.primary_method", "omp_matched_filter", "enum",
               S.PROV.ENGINEERING_DEFAULT, source="declared in this file before the run",
               note=("oracle template matcher: handed the SAME forward matrix that "
                     "generated the data. Within this model that is the best an "
                     "estimator can do, and a failure is MODEL-SPECIFIC -- it does NOT "
                     "bound a real sorter, which has morphology, per-neuron waveforms "
                     "and a different noise budget."))
    led.record("sorter.detection_threshold_far", FAR_PER_TEST, "probability",
               S.PROV.ENGINEERING_DEFAULT,
               source="declared before the run; never retuned")
    led.record("scope.bounded", True, "bool", S.PROV.ENGINEERING_DEFAULT,
               source="this run",
               note=("<=%d channels, <=%d neurons, <=%.1f s, RLIMIT_AS=%s. The "
                     "whole-cloud arm (20475 somata, up to 21 channels per neuron) is "
                     "NOT reproduced." % (SYNTH_N_CHANNELS, SHANK_NEURONS,
                                          MAX_TRACE_SECONDS, _applied)))
    led.record("sweep.noise_levels_uV", list(NOISE_LEVELS_UV), "uV RMS",
               S.PROV.ENGINEERING_DEFAULT,
               source="this run's sweep grid, declared before the run",
               note=("the noise is the swept quantity and the DECLARED signal is fixed; "
                     "because the chain is linear this equals sweeping the source "
                     "amplitude the other way. Sweep range = the listed levels."))
    out["provenance_ledger"] = led.as_list()
    out["provenance_counts"] = led.counts()
    out["unresolved_entries"] = led.unresolved()

    out["could_not_do"] = [
        "Run the whole-cloud arm: a 10 GB OOM was confirmed on 256 channels x 20475 "
        "somata x 500 samples at 20 kHz, so this run is bounded to <=16 channels and "
        "<=32 neurons and every number here is fixture-local.",
        ("Represent the maximum multiplicity of the real geometry: fixture A reaches "
         "at most %d channels per neuron against 21 in the whole-cloud arm."
         % (fixtures["real_tiny"]["fixture"]["max_channels_per_neuron"]
            if "real_tiny" in fixtures else 0)),
        "Validate the sorter against a real recording or a real sorted dataset: the "
        "primary method is an ORACLE given the generating forward matrix, and no "
        "spike sorter of this project has ever been checked against data.",
        "Report a MEASURED in-band noise floor: the budget here is a Johnson-Nyquist "
        "model number over a stated grid, and the probe/amplifier/tissue budget is "
        "separate work that is still PENDING.",
        "Give any neuron a waveform, amplitude, firing rate or refractory behaviour "
        "from data: the somata are single voxel points with no electrophysiology, so "
        "the waveform is DECLARED and shared by every neuron.",
        "Claim any device or animal result. No claim is made that any neuron is "
        "recorded from a real animal, and no claim about behaviour, perception, "
        "attention, recognition, experience, consciousness, identity or immortality.",
    ]
    print()
    print("-" * 78)
    print("5. WHAT THIS RUN COULD NOT DO")
    print("-" * 78)
    for x in out["could_not_do"]:
        for line in _wrap(x, indent="  - "):
            print(line)

    out["headline"] = {
        "question": ("can a soma point seen by SEVERAL channels at the fixed 7 um / "
                     "20 um geometry be separated back out?"),
        "scope": "bounded fixtures; NOT the whole-cloud arm",
        "noise_open_input_uV": open_uV,
        "noise_loaded_output_referred_uV": out_ref_uV,
        "resolvable_fraction_of_shared_neurons_at_chain_noise_level":
            out.get("resolvable_fraction_at_floor"),
        "median_matched_filter_snr_at_chain_noise_level":
            out.get("median_mf_snr_at_chain_level"),
        "answer": ("within THIS model -- point sources, one DECLARED waveform per "
                   "neuron, this noise budget -- the shared neurons are not "
                   "attributable at the noise level of this chain. That is a "
                   "MODEL-SPECIFIC result and NOT a claim that spike sorting of this "
                   "array is impossible in reality."),
    }
    out["elapsed_s"] = time.time() - t_start

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    print()
    print("wrote %s (%.1f kB)" % (os.path.abspath(JSON_PATH),
                                  os.path.getsize(JSON_PATH) / 1024.0))

    _make_figure(plt, out, fixtures, out_ref_uV, open_uV)
    print("wrote %s (%.1f kB)" % (os.path.abspath(PNG_PATH),
                                  os.path.getsize(PNG_PATH) / 1024.0))
    print()
    print("HEADLINE (bounded scope): at the chain's %.4f uV noise level the resolvable "
          "fraction of shared neurons is %s; median matched-filter SNR is %s"
          % (out_ref_uV, out.get("resolvable_fraction_at_floor"),
             out.get("median_mf_snr_at_chain_level")))
    print("elapsed %.1f s" % out["elapsed_s"])
    return out


def _make_figure(plt, out, fixtures, out_ref_uV, open_uV):
    fig, ax = plt.subplots(2, 3, figsize=(19.0, 10.5))
    fig.suptitle(
        "SOURCE SEPARATION AT THE FIXED GEOMETRY -- BOUNDED, REDUCED-SCOPE RUN\n"
        "shaft %.4g um, pitch %.4g um, capture radius %.4g um DECLARED | fixtures: "
        "REAL-tiny (<=%d ch, <=%d real somata) + SYNTHETIC-tiny (<=%d ch, <=%d declared "
        "sources) | NOT the whole-cloud arm (20475 somata, 21 ch/neuron not represented)"
        % (DIAMETER_UM, PITCH_UM, CAPTURE_RADIUS_UM, SHANK_N_CHANNELS, SHANK_NEURONS,
           SYNTH_N_CHANNELS, SYNTH_NEURONS), fontsize=11)

    p = fixtures.get("real_tiny") or fixtures.get("synthetic_tiny")
    noise = np.array([r["noise_sd_uV"] for r in p["sweep_rows"]])

    def col(key, sub=None):
        v = []
        for r in p["sweep_rows"]:
            x = r[sub][key] if sub else r[key]
            v.append(np.nan if x is None else float(x))
        return np.asarray(v)

    a = ax[0, 0]
    a.semilogx(noise, col("detection_rate_mean"), "o-", label="detection (time only)")
    a.semilogx(noise, col("assignment_accuracy_mean"), "s-",
               label="assignment accuracy of detections")
    a.semilogx(noise, col("precision_mean"), "^-", label="precision")
    a.axvline(out_ref_uV, color="crimson", ls="--", lw=2,
              label="this chain's level %.2f uV" % out_ref_uV)
    a.axvline(open_uV, color="darkred", ls=":", lw=2,
              label="open-input integral %.2f uV" % open_uV)
    a.set_xlabel("noise sd (uV RMS, log)")
    a.set_ylabel("rate")
    a.set_ylim(-0.05, 1.05)
    a.grid(alpha=0.3, which="both")
    a.set_title("P1 detection vs assignment (%s fixture)\ndetection is 'a spike "
                "happened'; assignment is 'and it was THIS neuron'" % p["tag"])
    a.legend(fontsize=7, loc="center left")

    a = ax[0, 1]
    a.semilogx(noise, col("shared_resolvable_fraction_mean"), "o-", color="darkgreen",
               label="shared neurons half-recovered with RIGHT identity")
    a.semilogx(noise, col("shared_detection_rate_mean"), "s--", color="seagreen",
               label="shared spikes detected (time only)")
    a.axvline(out_ref_uV, color="crimson", ls="--", lw=2)
    a.set_xlabel("noise sd (uV RMS, log)")
    a.set_ylabel("fraction")
    a.set_ylim(-0.05, 1.05)
    a.grid(alpha=0.3, which="both")
    a.set_title("P2 THE HEADLINE, BOUNDED SCOPE\nresolvable = >=50%% of its spikes "
                "recovered with the right identity")
    a.legend(fontsize=7)

    a = ax[0, 2]
    tn = np.asarray(p["matched_filter_snr"]["template_norm_times_waveform_norm_mV"])
    a.hist(np.log10(np.clip(tn / (out_ref_uV * 1e-3), 1e-12, None)), bins=30,
           color="slateblue", alpha=0.85)
    a.axvline(0.0, color="k", ls="--", label="SNR = 1")
    a.axvline(math.log10(max(p.get("median_mf_snr_at_chain_level") or 0.0, 1e-12)),
              color="crimson", lw=2,
              label="median MF-SNR %.3g" % (p.get("median_mf_snr_at_chain_level") or 0.0))
    a.set_xlabel("log10( per-neuron matched-filter SNR |G[:,j].w| / sigma )")
    a.set_ylabel("neurons")
    a.grid(alpha=0.3)
    a.set_title("P3 WHY: per-neuron detection SNR at %.2f uV\na per-channel amplitude "
                "ratio is NOT the detection SNR" % out_ref_uV)
    a.legend(fontsize=7)

    a = ax[1, 0]
    cs = np.asarray(p.get("coherence_sample") or [0.0])
    a.hist(cs, bins=40, color="darkorange", alpha=0.85)
    a.set_xlabel("|cos| between two neurons' channel templates")
    a.set_ylabel("sampled pairs")
    a.grid(alpha=0.3)
    a.set_title("P4 THE MECHANISM (%s): template coherence is PURE GEOMETRY\n"
                "separable only if SNR > ~1/(1-cos)" % p["tag"])

    a = ax[1, 1]
    sc = (fixtures.get("synthetic_tiny") or {}).get("synthetic_coherence")
    if sc:
        ap = np.asarray(sc["all_pairs"])
        a.semilogx(np.clip(ap[:, 0], 1e-2, None), ap[:, 1], "o", ms=4, color="purple")
        a.plot([sc["closest_pair_separation_um"]], [sc["closest_pair_coherence"]],
               "r*", ms=18,
               label="closest pair %.4g um" % sc["closest_pair_separation_um"])
        a.plot([sc["far_pair_separation_um"]], [sc["far_pair_coherence"]],
               "g*", ms=16, label="farthest pair %.4g um" % sc["far_pair_separation_um"])
        a.set_title("P5 DECLARED-SYNTHETIC CONTROL: coherence vs declared separation\n"
                    "closer sources are less distinguishable, turned on purpose")
        a.legend(fontsize=7)
    else:
        a.text(0.5, 0.5, "synthetic fixture not run", ha="center")
        a.set_title("P5 synthetic control (not run)")
    a.set_xlabel("declared source separation (um, log)")
    a.set_ylabel("|cos|")
    a.grid(alpha=0.3, which="both")

    a = ax[1, 2]
    labels = ["added crosstalk effect\n(relative, low noise)",
              "median matched-filter SNR\nat this chain's level",
              "loaded-consistent\nnoise/signal ratio",
              "legacy 65x (OPEN noise /\nLOADED signal)"]
    mf = p.get("median_mf_snr_at_chain_level") or 0.0
    vals = [p.get("crosstalk_relative_effect_low_noise") or 0.0, mf,
            out["noise_floor"]["loaded_interface_band_uV"] / FRONTEND_MEDIAN_SIGNAL_UV,
            FRONTEND_RECORDED_RATIO]
    a.barh(range(len(vals)), np.log10(np.clip(vals, 1e-12, None)),
           color=["steelblue", "slateblue", "seagreen", "crimson"])
    a.set_yticks(range(len(vals)))
    a.set_yticklabels(labels, fontsize=7.5)
    for i, v in enumerate(vals):
        a.text(np.log10(max(v, 1e-12)), i, "  %.4g" % v, va="center", fontsize=8)
    a.set_xlabel("log10(value) -- dimensionless, compared for ORDER only")
    a.set_title("P6 WHAT SETS THIS: the added crosstalk is a ~1e-3 fraction;\n"
                "the 65x headline mixes an OPEN-input noise with a LOADED signal")
    a.grid(alpha=0.3, axis="x")

    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    fig.text(0.01, 0.004,
             "Attribution (CC BY 4.0, LICENCE CONDITION): fixture A uses real soma "
             "coordinates from \"Distributed control circuits across a brain-and-cord "
             "connectome\", Harvard Dataverse, doi:10.7910/DVN/7WTH1N, file "
             "somas_v1.parquet (file id 13916460). Fixture B is DECLARED synthetic and "
             "contains no real coordinate. Hand-built research prototype, NOT a "
             "validated device model and NOT a validated spike sorter; the primary "
             "method is an ORACLE template matcher.  No claim about behaviour, "
             "perception, attention, recognition, experience, consciousness, identity "
             "or immortality; no claim that any neuron is recorded from a real animal.",
             fontsize=7)
    fig.savefig(PNG_PATH, dpi=115)
    plt.close(fig)


if __name__ == "__main__":
    main()
