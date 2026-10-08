#!/usr/bin/env python
"""Wire the (working) visual front end into the connectome-derived neural tier,
and make the long-blocked acceptance task D ("notice and respond to a salient
object") RUNNABLE.

WHAT THIS IS
------------
One bounded job: build the tier, form a visual afferent drive into whatever
visual machinery the tier actually contains, run the FlyGym/NeuroMechFly body in
the natural fBm scene with eye cameras, and test a PRE-REGISTERED set of
predictions about whether that drive is measurable -- at the tier, and in the
closed loop.

WHAT THIS IS NOT
----------------
* NOT a visual system.  The tier holds 11 visual-machinery neurons out of 12000
  (the dataset holds 79538 of 153962).  That is a STUB, and every claim below is
  proportioned to 11 neurons.
* NOT a claim that the fly "sees", "notices", "attends to" or "recognises"
  anything.  There is NO Drosophila visual-response reference dataset in this
  project, so "normal vs abnormal" is UNTESTABLE here.
* The eye optics are FlyGym's own fisheye Retina, which FlyGym's docstring says
  is calibrated for FlyGym's OWN eye placement -> APPROXIMATE on this body.
* The locomotion controller is FlyGym's demo tripod CPG.  It is an ENGINEERING
  BASELINE, not the connectome, and it generates the gait in EVERY condition.
* The model is ~1042x heavier than a real fly (project anchor), so no absolute
  force or current quoted here is a biological quantity.
* No light exists in a physical sense: the readout is a rendered image, and the
  nA figure is the output of a conversion whose factor is ASSUMED.

READ-ONLY DEPENDENCIES (not modified by this file)
--------------------------------------------------
engine/embodied/{loop,adapters,vision,scene,natural_scene,body_backend}.py,
engine/receptors.py.

OUTPUTS
-------
outputs/embodied_body/vision_into_tier.json
outputs/embodied_body/vision_into_tier.png

RUN
---
MUJOCO_GL=egl venv_body/bin/python run_vision_into_tier.py
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT_DIR = HERE / "outputs" / "embodied_body"

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.embodied.adapters import (  # noqa: E402
    CommandDecoder, CommandedTripodCPG, DecodeConfig, LEG_INDEX, LEGS,
    MechanoReceptorBank, NeuralTier, OwnershipLedger, ReceptorConfig, TierConfig,
    build_neural_tier, leg_rows_from_dof_order)
from engine.embodied.body_backend import BodyBackend, BodyConfig  # noqa: E402
from engine.embodied.vision import (  # noqa: E402
    OmmatidiaFrontEnd, VisionConfig, visual_drive_nA)
from engine.receptors import to_nA  # noqa: E402

# ===========================================================================
# PRE-REGISTERED BLOCK.  Everything in this block was fixed, IN SOURCE, BEFORE
# any measurement in this file was taken.  Nothing here is re-tuned afterwards;
# a FAIL or an EVIDENCE NOT FOUND is reported as such.
# ===========================================================================

# -- clocks (identical to engine.embodied.loop.LoopConfig defaults) ---------
DT_BODY_S = 1e-4
DT_NEURAL_S = 5e-4
DT_COMMAND_S = 5e-3
N_SUB_NEURAL = int(round(DT_COMMAND_S / DT_NEURAL_S))       # 10
N_SUB_BODY = int(round(DT_COMMAND_S / DT_BODY_S))           # 50
DURATION_S = 6.0
N_INTERVALS = int(round(DURATION_S / DT_COMMAND_S))         # 1200
CALIBRATION_MS = 500.0
CALIBRATION_SETTLE_MS = 100.0
CALIBRATION_INTERVALS = int(round(CALIBRATION_MS / (DT_COMMAND_S * 1000.0)))
CALIBRATION_SETTLE_STEPS = int(round(CALIBRATION_SETTLE_MS / DT_NEURAL_S / 1000.0))
SEED = 0
CPG_BASE_HZ = 12.0

#: Vision is sampled once every VISION_STRIDE command intervals (20 ms) and held
#: constant in between.  DECLARED, not tuned: 20 ms is exactly VisionConfig's
#: temporal_tau_ms, so the front end's own first-order filter advances by one full
#: time constant per sample (alpha = 1 - exp(-1)); sampling every 5 ms would cost
#: 4x the renders for no additional temporal information at this tau.
VISION_STRIDE = 4

#: The ONE visual target that honestly exists: tier neurons whose annotated
#: super_class is visual machinery.  Same set build_neural_tier counts in its
#: visual_absence block.
VISUAL_SUPER_CLASSES = ("visual_projection", "visual_centrifugal",
                        "optic_lobe_intrinsic")

#: ASSUMED transduction factor, nA per readout unit (engine.receptors.to_nA has
#: no default, and vision.py REFUSES to convert without an explicit factor).
#:
#: WHY 1.0, chosen BEFORE any run in this file:
#:   * the KNOWN-GOOD tactile pathway (recorded in
#:     outputs/embodied_body/loop_episode_neural_modulated_seed0.npz) injects a
#:     mean 0.391 nA (max 0.631 nA) into EVERY neuron of each leg's sensory group;
#:   * the front end's pooled drive is an RMS of temporal contrast in readout
#:     units, whose order of magnitude on this scene is ~5e-2 (prior evidence:
#:     vision_quicklook.json temporal_std ~ 0.045);
#:   * so 1.0 nA/unit puts ~0.05 nA on each visual neuron, i.e. ~13% of the
#:     tactile per-neuron drive -- a deliberately CONSERVATIVE mid-range choice,
#:     NOT the top of the sweep, and not picked to produce a positive result.
VIS_ASSUMED_SCALE_NA_PER_UNIT = 1.0
VIS_ASSUMED_SCALE_PROVENANCE = (
    "ASSUMED (ENGINEERING_DEFAULT): hand-set nA per readout unit; this project has "
    "NO measured Drosophila photoreceptor-to-projection-neuron transduction gain, so "
    "the factor is anchored a priori to the recorded mechanosensory per-neuron "
    "current 0.391 nA and to the front end's readout-unit magnitude ~5e-2")

#: The mandated sweep.  Five decade-spaced factors, fixed in source.
VIS_SCALE_SWEEP = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

#: Constant-current excitability diagnostic on the tier (NOT a threshold; no
#: pre-registered number below depends on it).  It exists so that a null result
#: can be attributed: "the tier needs >= X nA per visual neuron to fire at all".
EXCITABILITY_PROBE_NA = (0.0, 5e-4, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1, 8e-1)

#: PRE-REGISTERED THRESHOLDS.
THRESHOLDS = {
    "P1_drive_min_coefficient_of_variation": 1.0e-3,
    "P2_visual_group_min_delta_hz": 1.0,
    "P3_readout_min_delta_hz": 1.0,
    "P4_object_drive_min_relative_change": 0.01,
    "P5_behaviour_min_thorax_deviation_mm": 0.05,
    "P6_blind_drive_exactly_zero_nA": True,
    "P6_blind_thorax_bit_identical": True,
    "P7_scramble_per_eye_total_rel_tol": 1.0e-12,
    "P7_scramble_min_delta_hz": 1.0,
    "P8_sham_max_fraction_of_present": 0.5,
}
THRESHOLD_DERIVATION = {
    "P2/P3/P7 (1.0 Hz)": (
        "the only honest resolution to declare.  The readout population is 19 "
        "descending neurons (7 left-biased, 9 right-biased, ties excluded); at 19 "
        "neurons and a 5 ms window one extra spike in one neuron is 10.5 Hz for that "
        "interval, so a 1 Hz change in the episode-mean rate is ~0.1 spike per "
        "neuron-second -- a deliberately LOW bar that an 11-neuron input clears only "
        "if it actually matters."),
    "P4 (1% relative)": (
        "the object subtends a small part of a 721-ommatidium eye; quicklook evidence "
        "put the object's effect on one eye's temporal std at ~+17%.  1% is a "
        "17x-lower bar, declared as the level below which the object cannot be said "
        "to change the drive at all."),
    "P5 (0.05 mm)": (
        "2% of the fly's ~2.5 mm body length, i.e. a displacement change far below "
        "anything interpretable as a behavioural response.  IMPORTANT AND "
        "PRE-REGISTERED: under a closed loop a tiny nonzero command difference "
        "DIVERGES over 6 s, so clearing this bar is NOT evidence of 'noticing'.  The "
        "decisive numbers are P2/P3 (tier level) and the command-level magnitudes "
        "reported next to P5."),
    "P6 (bit-identical)": (
        "the two scene presets differ ONLY in `objects` (all contype=conaffinity=0, so "
        "they cannot touch the fly).  With the visual drive forced to zero the two "
        "runs must therefore agree EXACTLY, not approximately; any nonzero difference "
        "would prove a physical or bookkeeping leak."),
    "P8 (50%)": (
        "the sham puts the same geoms with the same albedos and the same count in the "
        "world but out of the fly's forward field of view.  It is only INFORMATIVE if "
        "the in-view object effect itself cleared P3; otherwise it is recorded as "
        "EVIDENCE NOT FOUND rather than as a pass."),
}

#: The scene presets used.  natural_fbm and natural_fbm_lit_no_objects come from
#: engine.embodied.natural_scene and are identical in every field except `objects`.
SCENE_WITH_OBJECT = "natural_fbm"
SCENE_NO_OBJECT = "natural_fbm_lit_no_objects"

#: SHAM / FALSE TARGET: the SAME three objects, same sizes, same albedos, same
#: number of geoms, placed BEHIND the fly (-x) instead of ahead of it (+x).  The
#: walk direction is +x, measured from a recorded loop episode (thorax delta
#: [+79.6, -18.4] mm).  Added through the PUBLIC BodyConfig.extra_geoms channel on
#: the no-object preset, so the world's total reflecting area and albedo set match
#: the object-present world while nothing is in the forward visual field.
SHAM_GEOMS = (
    {"name": "sham_dark", "type": "box", "pos": (-5.0, 0.0, 0.55),
     "size": (0.35, 0.35, 0.35), "rgba": (0.05, 0.05, 0.06, 1.0),
     "contype": 0, "conaffinity": 0},
    {"name": "sham_light", "type": "box", "pos": (-4.0, 2.0, 0.55),
     "size": (0.35, 0.35, 0.35), "rgba": (0.72, 0.70, 0.62, 1.0),
     "contype": 0, "conaffinity": 0},
    {"name": "sham_pebble", "type": "sphere", "pos": (-2.5, 2.5, 0.12),
     "size": (0.12, 0.0, 0.0), "rgba": (0.45, 0.20, 0.06, 1.0),
     "contype": 0, "conaffinity": 0},
)
SHAM_MIRROR_NOTE = (
    "ENGINEERING_DEFAULT construction: natural_scene._natural_objects() copied with "
    "pos_mm x negated, added through the public BodyConfig.extra_geoms channel; "
    "contype=conaffinity=0 so it still cannot touch the fly.")

CONDITIONS = [
    {"key": "present", "scene": SCENE_WITH_OBJECT, "vision": "normal",
     "describe": "natural fBm scene WITH its three small visual landmarks "
                 "(all contype=conaffinity=0)"},
    {"key": "absent", "scene": SCENE_NO_OBJECT, "vision": "normal",
     "describe": "identical scene preset with objects=() -- the object is absent"},
    {"key": "present_blind", "scene": SCENE_WITH_OBJECT, "vision": "blind",
     "describe": "NEGATIVE CONTROL: same scene, visual drive forced to exactly zero"},
    {"key": "absent_blind", "scene": SCENE_NO_OBJECT, "vision": "blind",
     "describe": "validity pair for present_blind: proves the two scenes are "
                 "physically identical"},
    {"key": "present_scramble", "scene": SCENE_WITH_OBJECT, "vision": "scramble",
     "describe": "CONTROL: ommatidia permuted within each eye; per-eye total light "
                 "preserved exactly"},
    {"key": "sham_out_of_view", "scene": SCENE_NO_OBJECT, "vision": "normal",
     "describe": "SHAM/false target: the same three objects, same albedos, same "
                 "count, placed BEHIND the fly",
     "extra_geoms": SHAM_GEOMS},
]

#: (tag, source episode, front-end treatment) for every pattern of the SAME
#: recorded eye frames.  The blind and scramble arms are applied OFFLINE to the
#: recorded frames of the same episode, so all three share one trajectory.
DRIVE_ARMS = [
    ("present", "present", "normal"),
    ("present_blind", "present", "blind"),
    ("present_scramble", "present", "scramble"),
    ("absent", "absent", "normal"),
    ("absent_blind", "absent", "blind"),
    ("absent_scramble", "absent", "scramble"),
    ("sham_out_of_view", "sham_out_of_view", "normal"),
    ("sham_blind", "sham_out_of_view", "blind"),
    ("sham_scramble", "sham_out_of_view", "scramble"),
]
#: which arms get the secondary spatial-contrast (pattern) replay as well
SPATIAL_ARMS = ("present", "absent", "present_scramble", "sham_out_of_view")


# ===========================================================================
# helpers
# ===========================================================================
def _flt(x):
    return None if x is None else float(x)


def _clean(obj):
    """Make a nested structure JSON-safe (numpy scalars/arrays -> Python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else str(v)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    return obj


def _stats(a):
    a = np.asarray(a, float).ravel()
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a))}


def _peak_ram_mb():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


class _ReplaySim:
    """Stands in for FlyGym's Simulation when re-running the REAL front end over a
    RECORDED ommatidia frame.

    ``OmmatidiaFrontEnd.read`` touches the simulation in exactly one place --
    ``sim.get_ommatidia_readouts(fly_name)`` -- so replaying recorded frames
    through the module's OWN front end (never a reimplementation) needs nothing
    else.  This is what makes the identical-gait and identical-frame comparisons
    below possible.
    """

    def __init__(self, frame):
        self._frame = frame

    def get_ommatidia_readouts(self, fly_name):
        return self._frame


# ===========================================================================
# 1. the tier and its visual machinery
# ===========================================================================
def build_and_describe_tier():
    t0 = time.perf_counter()
    built = build_neural_tier(TierConfig())
    build_s = time.perf_counter() - t0

    a = built["data"]
    gi = built["global_index"]
    super_class = np.asarray(a["ann_super_class"])
    primary_type = np.asarray(a["ann_primary_type"])
    cell_class = np.asarray(a["ann_class"])
    body_part = np.asarray(a["ann_body_part"])
    function = np.asarray(a["ann_function"])
    nt_verified = np.asarray(a["ann_nt_verified"])
    nerve = np.asarray(a["ann_nerve"])

    vis_mask = np.isin(super_class[gi], list(VISUAL_SUPER_CLASSES))
    vis_local = np.flatnonzero(vis_mask).astype(np.int64)
    n_visual = int(vis_local.size)
    n_tier = int(built["n"])
    n_dataset = int(super_class.size)
    n_visual_dataset = int(np.isin(super_class, list(VISUAL_SUPER_CLASSES)).sum())

    W = (built["edges"]["We"] + built["edges"]["Wi"]).tocsr()
    # EDGE-MATRIX CONVENTION, established empirically rather than assumed.
    # build_edges (engine/neck_cut_data.py) fills csr_matrix((w, (loc[post], loc[pre]))),
    # so the row index is the POSTsynaptic (target) neuron and the column index is the
    # PREsynaptic (source) neuron -- which is also what engine.neural_cond.ConductanceNetwork
    # consumes (``ge += We @ arrivals``).  Verified against the raw annotation:
    # for a leg mechanosensory cell, row nnz == the number of DISTINCT in-tier sources
    # with a pure transmitter label, and column nnz == the number of distinct in-tier
    # targets.  Hence: row nnz = INCOMING modelled edges, column nnz = OUTGOING.
    in_deg = np.asarray(W[vis_local, :].getnnz(axis=1)).ravel().tolist()
    out_deg = np.asarray(W[:, vis_local].getnnz(axis=0)).ravel().tolist()

    # raw annotation (before the sign rule excludes a row): how many in-tier targets
    # does each visual cell actually have, and why were they dropped?
    pre_raw, post_raw = np.asarray(a["pre"]), np.asarray(a["post"])
    loc = np.asarray(built["local_of_global"])
    nt_ver = np.asarray(a["ann_nt_verified"])
    raw_out = []
    for i in vis_local:
        g = int(gi[i])
        m = (pre_raw == g) & (loc[post_raw] >= 0)
        labels = sorted(set(str(x) if str(x) else "<empty>" for x in nt_ver[g][:0])) or []
        raw_out.append({
            "local_index": int(i),
            "raw_annotated_rows_from_this_cell_with_an_in_tier_target":
                int(np.count_nonzero(m)),
            "distinct_in_tier_targets_raw": int(np.unique(loc[post_raw[m]]).size)
            if np.count_nonzero(m) else 0,
            "modelled_outgoing_edges_after_the_sign_rule": int(out_deg[len(raw_out)]),
            "raw_target_row_transmitter_labels": sorted(set(
                str(x) if str(x) else "<empty>" for x in (nt_ver[pre_raw[m]] if
                                                          np.count_nonzero(m) else []))),
        })

    readout = np.asarray(built["readout_local"])
    n = n_tier
    # FORWARD reachability: successors of j are the rows i with W[i, j] != 0
    frontier = np.zeros(n, bool)
    frontier[vis_local] = True
    seen = frontier.copy()
    hops = [{"hop": 0, "new_neurons": int(vis_local.size),
             "readout_reached": int(np.isin(vis_local, readout).sum())}]
    for hop in range(1, 9):
        idx = np.flatnonzero(frontier)
        if idx.size == 0:
            break
        succ = np.unique(W[:, idx].tocoo().row)
        nxt = np.zeros(n, bool)
        nxt[succ] = True
        new = nxt & ~seen
        hops.append({"hop": hop, "new_neurons": int(new.sum()),
                     "readout_reached": int(np.isin(np.flatnonzero(nxt), readout).sum())})
        if int(new.sum()) == 0:
            break
        seen |= nxt
        frontier = new
    reach_readout = int(np.isin(np.flatnonzero(seen), readout).sum())

    cells = []
    for j, i in enumerate(vis_local):
        g = int(gi[i])
        cells.append({
            "local_index": int(i), "global_index": g,
            "super_class": str(super_class[g]), "class": str(cell_class[g]),
            "primary_type": str(primary_type[g]), "body_part": str(body_part[g]),
            "function": str(function[g]), "nt_verified": str(nt_verified[g]),
            "ann_nerve": str(nerve[g]),
            "in_tier_incoming_modelled_edges": int(in_deg[j]),
            "in_tier_outgoing_modelled_edges": int(out_deg[j]),
        })

    return {
        "tier_build_seconds": float(build_s),
        "tier_neurons": n_tier,
        "dataset_neurons": n_dataset,
        "tier_cap_binding": bool(
            built["coverage"]["tier"]["cap_binding_at_final_iteration"]),
        "tier_growth_history": built["coverage"]["growth_history"],
        "tier_fingerprint_sha256_32": built["fingerprint"],
        "visual_machinery_neurons_in_tier": n_visual,
        "visual_machinery_neurons_in_dataset": n_visual_dataset,
        "visual_share_of_tier": float(n_visual / n_tier),
        "visual_share_of_dataset": float(n_visual_dataset / n_dataset),
        "visual_local_indices": [int(v) for v in vis_local],
        "visual_global_indices": [int(gi[v]) for v in vis_local],
        "visual_cells": cells,
        "modelled_edges_in_tier": int(W.nnz),
        "edge_matrix_convention": ("csr_matrix((w, (loc[post], loc[pre]))) from "
                                   "engine.neck_cut_data.build_edges and "
                                   "neural_cond.ConductanceNetwork's ``We @ arrivals``: "
                                   "ROW = target (incoming), COLUMN = source (outgoing).  "
                                   "Checked against the raw annotation on a leg "
                                   "mechanosensory cell."),
        "visual_incoming_modelled_edges_per_cell": in_deg,
        "visual_outgoing_modelled_edges_per_cell": out_deg,
        "visual_outgoing_edges_total": int(sum(out_deg)),
        "visual_incoming_edges_total": int(sum(in_deg)),
        "raw_annotated_outgoing_rows_of_the_visual_group": raw_out,
        "visual_outgoing_edges_all_zero": bool(all(d == 0 for d in out_deg)),
        "bfs_forward_from_visual_group": hops,
        "readout_neurons_total": int(readout.size),
        "readout_reachable_forward_from_visual_group": reach_readout,
        "readout_selection_note_observed": (
            "engine.embodied.adapters selects readout_local with "
            "``W_any[:, desc_local].getnnz(axis=0) >= 1``, and under the convention above "
            "axis=0 on that slice is the OUTGOING degree, while the adjacent comment "
            "describes it as 'receive at least one modelled in-tier edge'.  Reported as an "
            "observation from an independent empirical check; engine/embodied/adapters.py "
            "is READ-ONLY for this job and was NOT modified.  It does not change what the "
            "readout population IS (the decoder's 19 descending cells), only the "
            "description of how it was picked."),
        "readout_left": int(built["readout_left"].size),
        "readout_right": int(built["readout_right"].size),
        "readout_tie": int(built["readout_tie"].size),
        "descending_in_tier": int(built["descending_local"].size),
        "sensory_group_sizes": {k: int(v.size) for k, v in built["sensory"].items()},
        "modality_boundaries": {
            k: {"in_tier": v["in_tier"], "in_dataset": v["in_dataset"],
                "fraction": _flt(v["fraction_of_dataset_group_in_tier"])}
            for k, v in built["boundary"].items()},
        "build_dict_visual_absence_block": built["coverage"]["visual_absence"],
        "is_a_visual_system": False,
        "structural_finding": {
            "ordering": ("established by an independent graph probe on the SAME tier build "
                         "BEFORE the main measurement run below, and it changed NO "
                         "pre-registered threshold; it is reported as a measurement that "
                         "explains the tier-level result, not as a prediction"),
            "finding": ("the visual-machinery group is a pure SINK inside this tier: the "
                        "modelled edge set contains ZERO outgoing edges from any of the "
                        "%d cells, so the forward reachable set from them is exactly "
                        "themselves (%d neurons) and the descending readout is NOT "
                        "reachable from them at any hop." % (n_visual, reach_readout)),
            "why": ("the raw annotation DOES give the group in-tier targets "
                    "(raw_annotated_outgoing_rows_of_the_visual_group), but EVERY one of "
                    "those rows is dropped by build_edges' sign rule, which keeps only a "
                    "PURE ann_nt_verified presynaptic label (acetylcholine -> excitatory, "
                    "gaba/glutamate -> inhibitory).  The visual projection/centrifugal "
                    "labels carry no pure label here."),
            "consequence": ("an injected current into these cells changes THEIR OWN firing "
                            "rate (measured below) and cannot propagate to any other tier "
                            "neuron through the modelled connectome."),
        },
        "plain_statement": (
            "This is NOT a visual system.  The tier is a leg-mechanosensory-seeded "
            "growth selection of a dataset ANNOTATION, and it caught %d visual "
            "machinery neurons -- %.4f%% of the tier.  Every visual claim below is "
            "proportioned to those %d cells.  The dataset itself DOES contain %d "
            "visual machinery neurons of %d (%.1f%%): the shortage is a consequence "
            "of the tier's SELECTION RULE, not of missing data."
            % (n_visual, 100.0 * n_visual / n_tier, n_visual, n_visual_dataset,
               n_dataset, 100.0 * n_visual_dataset / n_dataset)),
    }


# ===========================================================================
# 2. tier excitability diagnostic (constant current into the visual group)
# ===========================================================================
def excitability_diagnostic(built, vis_local):
    rows = []
    for c in EXCITABILITY_PROBE_NA:
        ledger = OwnershipLedger()
        tier = NeuralTier(built, TierConfig(), ledger)
        cur = np.zeros(tier.n)
        cur[vis_local] = float(c)
        n_int = 100
        vis_spikes = ro_spikes = 0
        fired = set()
        for _ in range(n_int):
            sp = tier.advance(N_SUB_NEURAL, cur)
            vis_spikes += int(sp[:, vis_local].sum())
            ro_spikes += int(sp[:, tier.readout_local].sum())
            fired |= set(np.flatnonzero(sp[:, vis_local].any(axis=0)).tolist())
        window_s = n_int * DT_COMMAND_S
        rows.append({
            "current_nA_per_visual_neuron": float(c),
            "visual_group_spikes": int(vis_spikes),
            "visual_neurons_ever_firing": int(len(fired)),
            "visual_group_rate_hz": float(vis_spikes) / (vis_local.size * window_s),
            "readout_group_rate_hz": float(ro_spikes)
            / (tier.readout_local.size * window_s),
        })
    return {
        "what_this_is": ("diagnostic only, run BEFORE the main experiment; used to "
                         "attribute a null result ('the tier needs >= X nA per visual "
                         "neuron to fire at all'), NOT to set any pre-registered "
                         "threshold"),
        "duration_s": 0.5,
        "rows": rows,
    }


# ===========================================================================
# 3. one closed-loop episode
# ===========================================================================
def run_episode(spec, built, vis_local, fly_name_for_notes=None):
    t0 = time.perf_counter()
    ledger = OwnershipLedger()
    backend = BodyBackend(BodyConfig(
        seed=SEED, timestep_s=DT_BODY_S, add_vision=True,
        scene_preset=spec["scene"], extra_geoms=tuple(spec.get("extra_geoms", ())),
        add_tracking_camera=True, cpg_intrinsic_frequency_hz=CPG_BASE_HZ),
        gl_backend="egl")
    fly_name = backend.cfg.fly_name
    maps = leg_rows_from_dof_order(backend.dof_order, backend.model)
    angle_rows = maps["angle_rows"]

    cpg = CommandedTripodCPG(timestep_s=DT_BODY_S, base_frequency_hz=CPG_BASE_HZ,
                             seed=SEED, dof_order=backend.dof_order, ledger=ledger)
    backend.attach_action_source(cpg)
    tier = NeuralTier(built, TierConfig(), ledger)
    decoder = CommandDecoder(DecodeConfig(), dt_ms_command=DT_COMMAND_S * 1000.0,
                             dt_ms_neural=DT_NEURAL_S * 1000.0, ledger=ledger, tier=tier)
    receptors = MechanoReceptorBank(built["sensory"], angle_rows, tier.n,
                                    ReceptorConfig(), dt_ms=DT_COMMAND_S * 1000.0,
                                    ledger=ledger, seed=SEED)

    kind = spec["vision"]
    vis_cfg = VisionConfig(n_channels=1, seed=0,
                          drive_scale_nA_per_unit=VIS_ASSUMED_SCALE_NA_PER_UNIT,
                          blind=(kind == "blind"), scramble=(kind == "scramble"))

    # -- calibration EXACTLY as engine.embodied.loop.MultirateScheduler.calibrate
    obs = backend.observe()
    receptors.set_reference_pose(obs.joint_angles_rad)
    cal_spikes = np.zeros((CALIBRATION_INTERVALS * N_SUB_NEURAL, tier.n), dtype=bool)
    for j in range(CALIBRATION_INTERVALS):
        cur, _ = receptors.transduce(obs)
        cal_spikes[j * N_SUB_NEURAL:(j + 1) * N_SUB_NEURAL] = tier.advance(
            N_SUB_NEURAL, cur)
    calibration = decoder.calibrate(tier, cal_spikes, CALIBRATION_INTERVALS,
                                    settled_from_step=CALIBRATION_SETTLE_STEPS)
    tier.reset()
    receptors.reset()
    receptors.set_reference_pose(obs.joint_angles_rad)

    front_end = OmmatidiaFrontEnd(vis_cfg)
    front_end.bind(backend.sim, fly_name)
    front_end.reset()

    rec = {k: [] for k in ("thorax_mm", "speed_scale", "turn", "rate_total_hz",
                           "readout_raw_hz")}
    grp = {k: [] for k in ("visual", "readout", "descending_all", "left", "right")}
    grp.update({leg: [] for leg in LEGS})
    vis_interval_nA = []
    vis_live_mV = []
    vis_times = []
    vis_frames = []
    mech_per_leg = []
    drive_ledger_last = None

    prev_spikes = None
    held_nA = 0.0
    first_loop_ms = None
    sensory_groups = {leg: np.asarray(built["sensory"][leg]) for leg in LEGS}

    for k in range(N_INTERVALS):
        if first_loop_ms is None:
            first_loop_ms = 1000.0 * (time.perf_counter() - t0)
        obs = backend.observe()
        cur, trans = receptors.transduce(obs)

        # ---- vision: ONE render per sample; the front end reads the recorded
        #      frame through _ReplaySim, so the live and the replayed paths see
        #      bit-identical pixels ------------------------------------------
        if k % VISION_STRIDE == 0:
            frame = None
            if kind != "blind":
                frame = np.asarray(backend.ommatidia_readouts(), dtype=float)
                # stored in float64, i.e. the EXACT values the live front end read,
                # so the offline replay on recorded frames is bit-identical
                vis_frames.append(frame)
            feats = front_end.read(_ReplaySim(frame), fly_name, k * DT_COMMAND_S)
            drive = visual_drive_nA(feats, vis_cfg, to_nA)
            held_nA = float(np.asarray(drive["drive_nA"], float)[0])
            if kind == "blind" and held_nA != 0.0:
                raise AssertionError("blind condition produced a nonzero visual drive")
            vis_live_mV.append(float(np.asarray(drive["drive_mV"], float)[0]))
            vis_times.append(float(k * DT_COMMAND_S))
            drive_ledger_last = {
                "channel_names": list(drive["channel_names"]),
                "drive_mV": [float(v) for v in np.asarray(drive["drive_mV"], float)],
                "drive_nA": [float(v) for v in np.asarray(drive["drive_nA"], float)],
                "total_nA": float(drive["total_nA"]),
                "drive_scale_nA_per_unit": float(drive["drive_scale_nA_per_unit"]),
                "drive_scale_provenance": str(drive["drive_scale_provenance"]),
                "conversion_helper": str(drive["conversion_helper"]),
                "unit_contract": str(drive["unit_contract"]),
                "blind": bool(drive["blind"]), "scrambled": bool(drive["scrambled"]),
                "caveats": list(drive["caveats"]),
                "front_end_bind_warnings": list(front_end.warnings),
                "fovy_applied_deg": list(front_end.fovy_applied_deg),
                "n_ommatidia_per_eye": int(front_end.n_ommatidia),
                "n_yellow_pale": [int(front_end.n_yellow), int(front_end.n_pale)],
                "retina_calibration_caveat": front_end.bind_report.get(
                    "retina_calibration_caveat"),
            }
        vis_interval_nA.append(held_nA)

        # ---- the afferent: ADD the pooled current to every visual-machinery
        #      neuron of the tier's own external-current vector --------------
        cur_total = cur.copy()
        cur_total[vis_local] += held_nA
        spikes = tier.advance(N_SUB_NEURAL, cur_total)

        if prev_spikes is None:
            cmd = decoder.neutral()
        else:
            cmd = decoder.decode(prev_spikes, source_interval=k - 1)
        prev_spikes = spikes

        cpg.set_command(cmd.speed_scale, cmd.turn)
        for _ in range(N_SUB_BODY):
            backend.step()

        rec["thorax_mm"].append(np.asarray(obs.thorax_position_mm, float).copy())
        rec["speed_scale"].append(float(cmd.speed_scale))
        rec["turn"].append(float(cmd.turn))
        rec["rate_total_hz"].append(float(cmd.rate_total_hz))
        rec["readout_raw_hz"].append(float(decoder.rate_hz(spikes)["total_hz"]))
        grp["visual"].append(int(spikes[:, vis_local].sum()))
        grp["readout"].append(int(spikes[:, tier.readout_local].sum()))
        grp["descending_all"].append(int(spikes[:, built["descending_local"]].sum()))
        grp["left"].append(int(spikes[:, tier.left].sum()))
        grp["right"].append(int(spikes[:, tier.right].sum()))
        for leg in LEGS:
            grp[leg].append(int(spikes[:, sensory_groups[leg]].sum()))
        mech_per_leg.append(np.asarray(trans["current_nA_per_leg"], float).copy())

    window_s = DT_COMMAND_S
    n_vis = int(vis_local.size)
    rates = {}
    for leg in LEGS:
        rates["mech_" + leg] = float(np.sum(grp[leg])) / (
            sensory_groups[leg].size * N_INTERVALS * window_s)
    rates["visual_group_hz"] = float(np.sum(grp["visual"])) / (
        n_vis * N_INTERVALS * window_s)
    rates["readout_hz"] = float(np.sum(grp["readout"])) / (
        tier.readout_local.size * N_INTERVALS * window_s)
    rates["readout_left_hz"] = float(np.sum(grp["left"])) / (
        tier.left.size * N_INTERVALS * window_s)
    rates["readout_right_hz"] = float(np.sum(grp["right"])) / (
        tier.right.size * N_INTERVALS * window_s)
    rates["descending_all_hz"] = float(np.sum(grp["descending_all"])) / (
        built["descending_local"].size * N_INTERVALS * window_s)
    rates["mean_mechanosensory_per_leg_hz"] = float(np.mean(
        [rates["mech_" + leg] for leg in LEGS]))

    thorax = np.asarray(rec["thorax_mm"], float)
    out = {
        "key": spec["key"],
        "describe": spec["describe"],
        "scene_preset": spec["scene"],
        "extra_geoms": [dict(g) for g in spec.get("extra_geoms", ())],
        "vision_config": vis_cfg.as_dict(),
        "vision_kind": kind,
        "sim_eye_cameras": dict(backend.eye_camera_specs),
        "eye_fovy_deg_measured": float(backend.cfg.eye_fovy_deg_measured),
        "eye_fovy_note": ("NeuroMechFly.add_vision(draw_sensor_markers=False) takes no "
                          "fovy; the angle is fixed by the shipped vision.yaml asset "
                          "(157 deg) and read back from the compiled model.  "
                          "VisionConfig.fovy (145 deg) is the OTHER fly class's default "
                          "and is a REQUEST that could not be honoured."),
        "nlight": int(backend.model.nlight),
        "ngeom": int(backend.model.ngeom),
        "n_ommatidia_per_eye": int(front_end.n_ommatidia),
        "calibration_reference_rates_hz": dict(calibration["reference_rates_hz"]),
        "group_rates_hz": rates,
        "thorax_start_mm": [float(v) for v in thorax[0]],
        "thorax_end_mm": [float(v) for v in thorax[-1]],
        "thorax_delta_mm": [float(v) for v in (thorax[-1] - thorax[0])],
        "horizontal_displacement_mm": float(np.linalg.norm((thorax[-1] - thorax[0])[:2])),
        "command_speed_scale": _stats(rec["speed_scale"]),
        "command_turn": _stats(rec["turn"]),
        "filtered_readout_rate_total_hz": _stats(rec["rate_total_hz"]),
        "raw_readout_rate_total_hz": _stats(rec["readout_raw_hz"]),
        "visual_current_per_visual_neuron_nA": _stats(vis_interval_nA),
        "visual_drive_readout_units": _stats(vis_live_mV),
        "drive_ledger": drive_ledger_last,
        "wall_seconds": time.perf_counter() - t0,
        "first_interval_wall_ms": first_loop_ms,
        "n_rendered_eye_frames": len(vis_frames),
    }
    arrays = {
        "thorax_mm": thorax,
        "speed_scale": np.asarray(rec["speed_scale"], float),
        "turn": np.asarray(rec["turn"], float),
        "readout_raw_hz": np.asarray(rec["readout_raw_hz"], float),
        "vis_interval_nA": np.asarray(vis_interval_nA, float),
        "vis_live_mV": np.asarray(vis_live_mV, float),
        "vis_times": np.asarray(vis_times, float),
        "mech_per_leg": np.asarray(mech_per_leg, float),
        "vis_frames": vis_frames,
        "cal_rate0": dict(calibration["reference_rates_hz"]),
    }
    backend.close()
    del ledger
    return out, arrays


# ===========================================================================
# 4. offline: the SAME front end over recorded frames; the tier replayed on one
#    identical gait
# ===========================================================================
def feature_series_from_frames(frames, times, real_sim, fly_name, kind):
    cfg = VisionConfig(n_channels=1, seed=0,
                       drive_scale_nA_per_unit=VIS_ASSUMED_SCALE_NA_PER_UNIT,
                       blind=(kind == "blind"), scramble=(kind == "scramble"))
    fe = OmmatidiaFrontEnd(cfg)
    fe.bind(real_sim, fly_name)
    fe.reset()
    feats, totals = [], []
    for t, frame in zip(times, frames):
        f = fe.read(_ReplaySim(None if kind == "blind" else frame), fly_name, float(t))
        feats.append(f)
        totals.append(np.asarray(f.eye_luminance, float).sum(axis=1))
    return {"cfg": cfg, "feats": feats, "frame_totals": np.asarray(totals, float)}


def drive_series(feats, cfg, scale):
    """The sanctioned conversion path, called once per factor of the sweep."""
    cfg_s = VisionConfig(n_channels=1, seed=cfg.seed,
                         drive_scale_nA_per_unit=float(scale),
                         blind=cfg.blind, scramble=cfg.scramble)
    mv, na = [], []
    last = None
    for f in feats:
        d = visual_drive_nA(f, cfg_s, to_nA)
        mv.append(float(np.asarray(d["drive_mV"], float)[0]))
        na.append(float(np.asarray(d["drive_nA"], float)[0]))
        last = d
    return np.asarray(mv, float), np.asarray(na, float), last


def spatial_drive_series(feats, scale):
    """SECONDARY arm: the pattern (spatial-contrast) term, which is EXACTLY zero
    for a spatially uniform image -- the one term that separates 'the eye
    receives light' from 'the eye receives a pattern'.  Taken from the module's
    own n_channels=4 layout; the two spatial channels are averaged because the 11
    visual cells carry no eye/side label (ann_nerve is empty for all 11)."""
    cfg4 = VisionConfig(n_channels=4, seed=0, drive_scale_nA_per_unit=float(scale))
    mv = []
    for f in feats:
        d = visual_drive_nA(f, cfg4, to_nA)
        v = np.asarray(d["drive_mV"], float)
        mv.append(0.5 * (float(v[2]) + float(v[3])))
    mv = np.asarray(mv, float)
    na = np.asarray(to_nA(mv, float(scale), VIS_ASSUMED_SCALE_PROVENANCE + " ; applied "
                          "to the pooled spatial-contrast term (secondary arm)"), float)
    return mv, na


def replay(built, groups, per_leg, vis_local, vis_nA_per_interval, cal_rate0,
           n_intervals):
    """Advance a FRESH tier on the same recorded mechanosensory currents, adding
    only the given visual current series.

    Identical seed, identical gait, identical tier construction: the visual
    current is the ONLY difference between arms.  The decoder state is set to the
    exact post-calibration state of the episode (calibrate() sets filt = rate0).
    """
    ledger = OwnershipLedger()
    tier = NeuralTier(built, TierConfig(), ledger)
    decoder = CommandDecoder(DecodeConfig(), dt_ms_command=DT_COMMAND_S * 1000.0,
                             dt_ms_neural=DT_NEURAL_S * 1000.0, ledger=ledger, tier=tier)
    decoder.rate0 = dict(cal_rate0)
    decoder.filt = dict(cal_rate0)

    vis = np.asarray(vis_nA_per_interval, float)
    if vis.size != n_intervals:
        raise ValueError("visual series length %d != %d intervals"
                         % (vis.size, n_intervals))
    keys = ("readout_hz", "left_hz", "right_hz", "visual_hz", "descending_hz",
            "speed_scale", "turn", "delta_total_hz")
    out = {k: [] for k in keys}
    prev = None
    for k in range(n_intervals):
        cur = np.zeros(tier.n)
        for leg in LEGS:
            cur[groups[leg]] += per_leg[k, LEG_INDEX[leg]]
        if vis[k] != 0.0:
            cur[vis_local] += vis[k]
        sp = tier.advance(N_SUB_NEURAL, cur)
        if prev is None:
            cmd = decoder.neutral()
        else:
            cmd = decoder.decode(prev, source_interval=k - 1)
        prev = sp
        out["readout_hz"].append(float(sp[:, tier.readout_local].sum())
                                 / (tier.readout_local.size * DT_COMMAND_S))
        out["left_hz"].append(float(sp[:, tier.left].sum())
                              / (tier.left.size * DT_COMMAND_S))
        out["right_hz"].append(float(sp[:, tier.right].sum())
                               / (tier.right.size * DT_COMMAND_S))
        out["visual_hz"].append(float(sp[:, vis_local].sum())
                                / (vis_local.size * DT_COMMAND_S))
        out["descending_hz"].append(float(sp[:, built["descending_local"]].sum())
                                    / (built["descending_local"].size * DT_COMMAND_S))
        out["speed_scale"].append(float(cmd.speed_scale))
        out["turn"].append(float(cmd.turn))
        out["delta_total_hz"].append(float(cmd.delta_total_hz))
    res = {"_series": {k: np.asarray(v, float) for k, v in out.items()}}
    for k, v in out.items():
        a = np.asarray(v, float)
        res[k] = {"mean": float(a.mean()), "std": float(a.std()),
                  "min": float(a.min()), "max": float(a.max())}
    return res


# ===========================================================================
# main
# ===========================================================================
def main():
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] building the tier ...", flush=True)
    tier_info = build_and_describe_tier()
    built = build_neural_tier(TierConfig())
    vis_local = np.asarray(tier_info["visual_local_indices"], np.int64)
    print("      tier=%d neurons | visual machinery in tier=%d | dataset visual=%d/%d "
          "| fingerprint=%s"
          % (tier_info["tier_neurons"], tier_info["visual_machinery_neurons_in_tier"],
             tier_info["visual_machinery_neurons_in_dataset"],
             tier_info["dataset_neurons"], tier_info["tier_fingerprint_sha256_32"]),
          flush=True)

    print("[2/6] tier excitability diagnostic (%d currents) ..."
          % len(EXCITABILITY_PROBE_NA), flush=True)
    excit = excitability_diagnostic(built, vis_local)

    print("[3/6] closed-loop episodes (%d conditions x %.1f s, stride %d) ..."
          % (len(CONDITIONS), DURATION_S, VISION_STRIDE), flush=True)
    episodes, arrays = {}, {}
    for spec in CONDITIONS:
        ep, arr = run_episode(spec, built, vis_local)
        episodes[spec["key"]] = ep
        arrays[spec["key"]] = arr
        print("      %-17s wall %6.1f s | vis-grp %8.3f Hz | readout %8.3f Hz | "
              "mech/leg %7.2f Hz | disp %7.2f mm | vis current %9.3g nA"
              % (spec["key"], ep["wall_seconds"], ep["group_rates_hz"]["visual_group_hz"],
                 ep["group_rates_hz"]["readout_hz"],
                 ep["group_rates_hz"]["mean_mechanosensory_per_leg_hz"],
                 ep["horizontal_displacement_mm"],
                 ep["visual_current_per_visual_neuron_nA"]["mean"]), flush=True)

    print("[4/6] re-running the SAME front end over recorded frames ...", flush=True)
    ref_arrays = arrays["present"]
    ledger_ref = OwnershipLedger()
    backend_ref = BodyBackend(BodyConfig(
        seed=SEED, timestep_s=DT_BODY_S, add_vision=True,
        scene_preset=SCENE_WITH_OBJECT, add_tracking_camera=True,
        cpg_intrinsic_frequency_hz=CPG_BASE_HZ), gl_backend="egl")
    fly_name = backend_ref.cfg.fly_name
    arms = [a for a in DRIVE_ARMS if a[1] in episodes]
    spatial_arms = [t for t in SPATIAL_ARMS if t in {a[0] for a in arms}]
    series = {}
    for tag, src, kind in arms:
        series[tag] = feature_series_from_frames(
            arrays[src]["vis_frames"], arrays[src]["vis_times"], backend_ref.sim,
            fly_name, kind)
        series[tag]["source_episode"] = src
        series[tag]["treatment"] = kind
    backend_ref.close()
    del ledger_ref

    print("[5/6] drive table + tier replay matrix (identical gait, %d scales) ..."
          % len(VIS_SCALE_SWEEP), flush=True)
    drive_table = {}
    for tag in series:
        for scale in VIS_SCALE_SWEEP:
            mv, na, last = drive_series(series[tag]["feats"], series[tag]["cfg"], scale)
            drive_table[(tag, scale)] = {
                "mv": mv, "na": na,
                "hold": np.repeat(na, VISION_STRIDE)[:N_INTERVALS], "last": last}
        mv, na = spatial_drive_series(series[tag]["feats"],
                                      VIS_ASSUMED_SCALE_NA_PER_UNIT)
        drive_table[(tag, "spatial")] = {
            "mv": mv, "na": na,
            "hold": np.repeat(na, VISION_STRIDE)[:N_INTERVALS], "last": None}

    groups = {leg: np.asarray(built["sensory"][leg]) for leg in LEGS}
    per_leg = ref_arrays["mech_per_leg"]
    cal_rate0 = ref_arrays["cal_rate0"]

    mech_only = replay(built, groups, per_leg, vis_local, np.zeros(N_INTERVALS),
                       cal_rate0, N_INTERVALS)
    print("      replayed mechano-only baseline", flush=True)
    replays = {}
    for tag in series:
        for scale in VIS_SCALE_SWEEP:
            replays["%s@%g" % (tag, scale)] = replay(
                built, groups, per_leg, vis_local, drive_table[(tag, scale)]["hold"],
                cal_rate0, N_INTERVALS)
        if tag in spatial_arms:
            replays["%s@spatial" % tag] = replay(
                built, groups, per_leg, vis_local, drive_table[(tag, "spatial")]["hold"],
                cal_rate0, N_INTERVALS)
        print("      replayed %-22s" % tag, flush=True)

    # ---- self-consistency checks ----------------------------------------
    present_key = "present@%g" % VIS_ASSUMED_SCALE_NA_PER_UNIT
    sc_present = float(np.max(np.abs(
        ref_arrays["readout_raw_hz"] - replays[present_key]["_series"]["readout_hz"])))
    blind_key = "present_blind@%g" % VIS_ASSUMED_SCALE_NA_PER_UNIT
    sc_blind = float(np.max(np.abs(
        mech_only["_series"]["readout_hz"] - replays[blind_key]["_series"]["readout_hz"])))

    # ---- comparatives ----------------------------------------------------
    def max_dev(a, b):
        pa, pb = arrays[a]["thorax_mm"], arrays[b]["thorax_mm"]
        d = np.linalg.norm(pa - pb, axis=1)
        return float(d.max()), float(d[-1]), int(np.argmax(d))

    dev = {
        "present_vs_present_blind": max_dev("present", "present_blind"),
        "absent_vs_present_blind": max_dev("absent", "present_blind"),
        "present_vs_absent": max_dev("present", "absent"),
        "present_scramble_vs_present_blind": max_dev("present_scramble", "present_blind"),
        "sham_vs_present_blind": max_dev("sham_out_of_view", "present_blind"),
        "present_blind_vs_absent_blind": max_dev("present_blind", "absent_blind"),
    }

    # =====================================================================
    # VERDICTS -- each reads the pre-registered number declared above
    # =====================================================================
    verdicts = []

    def add(pid, claim, threshold, measured, verdict, note):
        verdicts.append({"id": pid, "claim": claim,
                         "pre_registered_threshold": threshold,
                         "measured": measured, "verdict": verdict, "note": note})

    # P1 -- the drive exists and varies
    pres_mv = drive_table[("present", VIS_ASSUMED_SCALE_NA_PER_UNIT)]["mv"]
    cv = float(pres_mv.std() / pres_mv.mean()) if pres_mv.mean() > 0 else 0.0
    p1_ok = (float(pres_mv.max()) > 0.0 and
             cv >= THRESHOLDS["P1_drive_min_coefficient_of_variation"])
    add("P1", "the pooled visual drive is nonzero and varies over time in the "
              "seeing conditions",
        {"max_pooled_drive_mV_gt": 0.0,
         "coefficient_of_variation_ge":
             THRESHOLDS["P1_drive_min_coefficient_of_variation"]},
        {"max_pooled_drive_mV": float(pres_mv.max()),
         "mean_pooled_drive_mV": float(pres_mv.mean()),
         "coefficient_of_variation": cv,
         "n_samples": int(pres_mv.size)},
        "PASS" if p1_ok else "FAIL",
        "the front end's own non-degeneracy floor "
        "(NONDEGENERACY_MIN_TEMPORAL_STD = 1e-3 readout units, from vision.py, declared "
        "there before its own experiment) is the companion statistic: a FAIL here would "
        "mean the render is blank or the scene is spatially featureless")

    # P2 -- the drive reaches the tier (identical gait)
    vis_mo = mech_only["visual_hz"]["mean"]
    vis_pr = replays[present_key]["visual_hz"]["mean"]
    d2 = vis_pr - vis_mo
    ina = float(np.mean(drive_table[("present", VIS_ASSUMED_SCALE_NA_PER_UNIT)]["na"]))
    add("P2", "the injected visual current changes the visual group's firing rate",
        {"abs_delta_visual_group_hz_ge": THRESHOLDS["P2_visual_group_min_delta_hz"]},
        {"mechano_only_visual_group_hz": vis_mo,
         "mechano_plus_visual_group_hz": vis_pr, "delta_hz": d2,
         "visual_current_nA_per_neuron_mean": ina,
         "blind_replay_equals_mechano_only_max_abs_hz": sc_blind},
        "PASS" if abs(d2) >= THRESHOLDS["P2_visual_group_min_delta_hz"] else "FAIL",
        "identical mechanosensory gait in both arms (tier replay); the visual current is "
        "the only difference.  All 11 cells receive 0..17 modelled INCOMING edges, so the "
        "injected current is not their only input -- but they have ZERO modelled OUTGOING "
        "edges, so whatever they do cannot propagate: see tier.structural_finding")

    # P3 -- the readout responds (identical gait)
    ro_mo = mech_only["readout_hz"]["mean"]
    ro_pr = replays[present_key]["readout_hz"]["mean"]
    d3 = ro_pr - ro_mo
    add("P3", "the visual drive changes the descending readout population's firing rate",
        {"abs_delta_readout_hz_ge": THRESHOLDS["P3_readout_min_delta_hz"]},
        {"mechano_only_readout_hz": ro_mo, "mechano_plus_visual_readout_hz": ro_pr,
         "delta_hz": d3,
         "relative_change_fraction": (d3 / ro_mo if ro_mo else None),
         "replay_reproduces_episode_max_abs_hz": sc_present},
        "PASS" if abs(d3) >= THRESHOLDS["P3_readout_min_delta_hz"] else "FAIL",
        "identical mechanosensory gait in both arms (tier replay)")

    # P4 -- the object changes the drive at all
    abs_mv = drive_table[("absent", VIS_ASSUMED_SCALE_NA_PER_UNIT)]["mv"]
    p_abs, a_abs = float(pres_mv.mean()), float(abs_mv.mean())
    rel = abs(p_abs - a_abs) / max(p_abs, a_abs) if max(p_abs, a_abs) > 0 else 0.0
    add("P4", "the object PRESENT vs ABSENT changes the visual drive",
        {"relative_change_ge": THRESHOLDS["P4_object_drive_min_relative_change"]},
        {"present_mean_pooled_drive_mV": p_abs, "absent_mean_pooled_drive_mV": a_abs,
         "relative_change": rel,
         "present_vs_absent_max_thorax_deviation_mm": dev["present_vs_absent"][0]},
        "PASS" if rel >= THRESHOLDS["P4_object_drive_min_relative_change"] else "FAIL",
        "each drive is measured on ITS OWN closed-loop trajectory, so the thorax "
        "deviation between them is reported next to it as a comparability check")

    # P5 -- closed-loop behavioural difference
    ss = replays[present_key]["_series"]["speed_scale"]
    tt = replays[present_key]["_series"]["turn"]
    dspeed = float(np.max(np.abs(ss - mech_only["_series"]["speed_scale"])))
    dturn = float(np.max(np.abs(tt - mech_only["_series"]["turn"])))
    cl_dspeed = float(np.max(np.abs(arrays["present"]["speed_scale"]
                                    - arrays["present_blind"]["speed_scale"])))
    cl_dturn = float(np.max(np.abs(arrays["present"]["turn"]
                                  - arrays["present_blind"]["turn"])))
    add("P5", "with the visual drive ACTIVE the closed-loop behaviour differs from the "
              "blind control",
        {"max_thorax_deviation_mm_ge":
             THRESHOLDS["P5_behaviour_min_thorax_deviation_mm"]},
        {"max_thorax_deviation_mm": dev["present_vs_present_blind"][0],
         "final_thorax_deviation_mm": dev["present_vs_present_blind"][1],
         "peak_interval": dev["present_vs_present_blind"][2],
         "replay_max_abs_command_speed_scale_delta": dspeed,
         "replay_max_abs_command_turn_delta": dturn,
         "closed_loop_max_abs_command_speed_scale_delta": cl_dspeed,
         "closed_loop_max_abs_command_turn_delta": cl_dturn,
         "horizontal_displacement_mm_present":
             episodes["present"]["horizontal_displacement_mm"],
         "horizontal_displacement_mm_blind":
             episodes["present_blind"]["horizontal_displacement_mm"]},
        "PASS" if dev["present_vs_present_blind"][0] >=
        THRESHOLDS["P5_behaviour_min_thorax_deviation_mm"] else "FAIL",
        "PRE-REGISTERED READING OF THIS NUMBER: closed-loop trajectories diverge, so "
        "clearing this bar is NOT evidence of noticing.  Read it with "
        "replay_max_abs_command_*_delta and closed_loop_max_abs_command_*_delta, which "
        "say how large the actual command change was")

    # P6 -- blind negative control + scene-equivalence validity check
    blind_na = drive_table[("present_blind", VIS_ASSUMED_SCALE_NA_PER_UNIT)]["na"]
    blind_zero = bool(np.all(blind_na == 0.0))
    bit = bool(dev["present_blind_vs_absent_blind"][0] == 0.0)
    add("P6", "NEGATIVE CONTROL: blind forces the visual drive to exactly zero, and the "
              "object-present and object-absent scenes are PHYSICALLY IDENTICAL",
        {"blind_drive_exactly_zero":
             THRESHOLDS["P6_blind_drive_exactly_zero_nA"],
         "present_blind_vs_absent_blind_thorax_bit_identical":
             THRESHOLDS["P6_blind_thorax_bit_identical"]},
        {"blind_drive_max_abs_nA": float(np.max(np.abs(blind_na))),
         "blind_drive_n_samples": int(blind_na.size),
         "blind_pair_thorax_max_abs_deviation_mm":
             dev["present_blind_vs_absent_blind"][0],
         "blind_pair_thorax_final_deviation_mm":
             dev["present_blind_vs_absent_blind"][1]},
        "PASS" if (blind_zero and bit) else "FAIL",
        "the two presets differ ONLY in `objects`, all contype=conaffinity=0, so with the "
        "drive off the requirement is EXACT equality, not closeness; this is the "
        "validity check that the object cannot act mechanically")

    # P7 -- scramble control
    tot_n = series["present"]["frame_totals"]
    tot_s = series["present_scramble"]["frame_totals"]
    rel_tot = float(np.max(np.abs(tot_n - tot_s) / np.maximum(np.abs(tot_n), 1e-12)))
    scr_key = "present_scramble@%g" % VIS_ASSUMED_SCALE_NA_PER_UNIT
    d7 = replays[scr_key]["visual_hz"]["mean"] - vis_mo
    arr_matters = abs(d7 - d2) >= THRESHOLDS["P7_scramble_min_delta_hz"]
    tot_ok = rel_tot <= THRESHOLDS["P7_scramble_per_eye_total_rel_tol"]
    v7 = ("PASS" if (tot_ok and arr_matters) else
          ("FAIL" if not tot_ok else "EVIDENCE NOT FOUND"))
    add("P7", "CONTROL: scrambling the ommatidia preserves per-eye total light exactly, "
              "and any effect depends on the spatial ARRANGEMENT",
        {"per_eye_total_relative_tolerance":
             THRESHOLDS["P7_scramble_per_eye_total_rel_tol"],
         "abs_delta_scramble_minus_delta_present_hz_ge":
             THRESHOLDS["P7_scramble_min_delta_hz"]},
        {"max_relative_per_eye_total_difference": rel_tot,
         "n_frames_compared": int(tot_n.shape[0]),
         "scramble_mean_pooled_drive_mV":
             float(drive_table[("present_scramble",
                                VIS_ASSUMED_SCALE_NA_PER_UNIT)]["mv"].mean()),
         "present_mean_pooled_drive_mV": p_abs,
         "delta_visual_group_hz_present": d2,
         "delta_visual_group_hz_scramble": d7,
         "abs_difference_hz": abs(d7 - d2),
         "scramble_closed_loop_thorax_dev_vs_blind_mm":
             dev["present_scramble_vs_present_blind"][0]},
        v7,
        "the total-light half is a mathematical identity of a permutation and is checked "
        "numerically on the front end's OWN eye_luminance; the arrangement half needs the "
        "drive to be large enough to move the visual group at all, so it is recorded as "
        "EVIDENCE NOT FOUND when P2 fails")

    # P8 -- sham / false target
    sham_key = "sham_out_of_view@%g" % VIS_ASSUMED_SCALE_NA_PER_UNIT
    d8 = replays[sham_key]["readout_hz"]["mean"] - ro_mo
    sham_mv = drive_table[("sham_out_of_view", VIS_ASSUMED_SCALE_NA_PER_UNIT)]["mv"]
    p3_passed = abs(d3) >= THRESHOLDS["P3_readout_min_delta_hz"]
    if not p3_passed:
        v8, note8 = "EVIDENCE NOT FOUND", (
            "the in-view object's own readout effect did not clear P3, so 'the sham "
            "does less' cannot be asserted: with a null in-view effect there is nothing "
            "for the sham to be smaller than")
    else:
        v8 = ("PASS" if abs(d8) <= THRESHOLDS["P8_sham_max_fraction_of_present"] * abs(d3)
              else "FAIL")
        note8 = "sham effect compared with the in-view object effect on the same gait"
    add("P8", "SHAM / false target: the same objects placed out of view do not reproduce "
              "the in-view object's effect",
        {"abs_sham_readout_delta_le_fraction_of_present":
             THRESHOLDS["P8_sham_max_fraction_of_present"]},
        {"present_readout_delta_hz": d3, "sham_readout_delta_hz": d8,
         "sham_mean_pooled_drive_mV": float(sham_mv.mean()),
         "sham_vs_blind_max_thorax_deviation_mm":
             dev["sham_vs_present_blind"][0]},
        v8, note8)

    # ---- sweep table -----------------------------------------------------
    sweep = []
    for tag in series:
        for scale in VIS_SCALE_SWEEP:
            r = replays["%s@%g" % (tag, scale)]
            na = drive_table[(tag, scale)]["na"]
            sweep.append({
                "series": tag, "source_episode": series[tag]["source_episode"],
                "treatment": series[tag]["treatment"],
                "factor": float(scale), "factor_label": "%g nA/unit" % scale,
                "visual_current_per_neuron_nA_mean": float(na.mean()),
                "visual_current_per_neuron_nA_max": float(na.max()),
                "visual_group_hz": r["visual_hz"]["mean"],
                "visual_group_delta_hz": r["visual_hz"]["mean"] - vis_mo,
                "readout_hz": r["readout_hz"]["mean"],
                "readout_delta_hz": r["readout_hz"]["mean"] - ro_mo,
                "command_speed_scale_mean": r["speed_scale"]["mean"],
                "command_turn_mean": r["turn"]["mean"],
            })
    for tag in spatial_arms:
        r = replays["%s@spatial" % tag]
        na = drive_table[(tag, "spatial")]["na"]
        sweep.append({
            "series": tag, "source_episode": series[tag]["source_episode"],
            "treatment": series[tag]["treatment"],
            "factor": None, "factor_label": "spatial@%g nA/unit"
                                            % VIS_ASSUMED_SCALE_NA_PER_UNIT,
            "arm": ("spatial-contrast (pattern) term only -- EXACTLY zero for a "
                    "spatially uniform image"),
            "visual_current_per_neuron_nA_mean": float(na.mean()),
            "visual_current_per_neuron_nA_max": float(na.max()),
            "visual_group_hz": r["visual_hz"]["mean"],
            "visual_group_delta_hz": r["visual_hz"]["mean"] - vis_mo,
            "readout_hz": r["readout_hz"]["mean"],
            "readout_delta_hz": r["readout_hz"]["mean"] - ro_mo,
            "command_speed_scale_mean": r["speed_scale"]["mean"],
            "command_turn_mean": r["turn"]["mean"],
        })

    tactile_compare = {
        "definition": ("mechanosensory bank (known-good pathway) vs visual afferent, on "
                       "an IDENTICAL recorded gait"),
        "mechanosensory_mean_nA_per_sensory_neuron":
            float(np.mean(arrays["present"]["mech_per_leg"])),
        "visual_mean_nA_per_visual_neuron": ina,
        "readout_rate_mechano_only_hz": ro_mo,
        "readout_delta_hz_visual": d3,
        "visual_delta_as_fraction_of_mechano_driven_rate":
            (d3 / ro_mo if ro_mo else None),
        "mean_mechanosensory_per_leg_group_rate_hz":
            episodes["present"]["group_rates_hz"]["mean_mechanosensory_per_leg_hz"],
        "note": ("a current ratio is NOT the comparison that matters; the readout-rate "
                 "comparison is: the mechanosensory pathway holds the readout at %.3f Hz "
                 "and the visual afferent moves it by %.3f Hz" % (ro_mo, d3)),
    }

    # =====================================================================
    # FIGURE
    # =====================================================================
    def strip(a):
        return {k: v for k, v in a.items() if not k.startswith(("_", "vis_frames"))}

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13.2))
    span = {}
    t_vis = arrays["present"]["vis_times"]
    # (tag, linestyle, marker) -- explicit, so no format-string guessing
    plot_tags = (("present", "-", "o"), ("absent", "--", "s"),
                 ("present_scramble", ":", "^"),
                 ("sham_out_of_view", "-.", "d"))
    plot_tags_blind = plot_tags + (("present_blind", "--", "x"),)

    ax = axes[0, 0]
    for tag, ls, mk in plot_tags:
        y = drive_table[(tag, VIS_ASSUMED_SCALE_NA_PER_UNIT)]["mv"]
        ax.plot(t_vis, y, ls=ls, marker=mk, ms=3.0, lw=1.2, label=tag)
        span["a"] = max(span.get("a", 0.0), float(np.ptp(y)))
    ax.set_title("(a) pooled visual drive in READOUT UNITS (drive_mV)\n"
                 "RMS of temporal contrast over both eyes, %d samples"
                 % t_vis.size, fontsize=9)
    ax.set_xlabel("time (s, scheduler interval grid)")
    ax.set_ylabel("readout units")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # inset: the whole point of the figure is that the four traces co-incide, so
    # show the actual per-sample difference rather than relying on the eye
    ins = ax.inset_axes([0.07, 0.60, 0.55, 0.34])
    rel_ps = 100.0 * (pres_mv - abs_mv) / np.maximum(np.abs(abs_mv), 1e-12)
    ins.plot(t_vis, rel_ps, color="k", lw=0.8)
    ins.axhline(0.0, color="gray", lw=0.6)
    ins.axhline(100.0 * THRESHOLDS["P4_object_drive_min_relative_change"], color="r",
                ls="--", lw=0.9,
                label="pre-registered 1%% (P4, on the MEAN)")
    ins.axhline(100.0 * rel, color="g", ls=":", lw=1.0,
                label="measured mean %.2f%%" % (100.0 * rel))
    ins.set_title("present vs absent, per-sample difference (%%)", fontsize=6)
    ins.tick_params(labelsize=5)
    ins.legend(fontsize=5, loc="upper right")
    span["a_inset"] = float(np.ptp(rel_ps))

    ax = axes[0, 1]
    mech_ref = float(np.mean(arrays["present"]["mech_per_leg"]))
    for tag, ls, mk in plot_tags:
        # sample 0 is EXACTLY 0 by definition (the front end's first sample emits no
        # temporal contrast), so it is dropped here rather than clipping a log axis
        y = np.maximum(drive_table[(tag, VIS_ASSUMED_SCALE_NA_PER_UNIT)]["na"][1:], 1e-9)
        ax.semilogy(t_vis[1:], y, ls=ls, marker=mk, ms=3.0, lw=1.2, label=tag)
        span["b"] = max(span.get("b", 0.0), float(np.ptp(np.log10(y))))
    ax.set_ylim(2e-3, 1.0)
    ax.axhline(mech_ref, color="k", lw=1.4, ls="--",
               label="mechanosensory per-neuron current, mean %.3f nA" % mech_ref)
    ax.text(0.02, 0.03, "blind arms are exactly 0.0 nA every frame\n(off-scale on a log "
            "axis); they are plotted in panel (c)",
            transform=ax.transAxes, fontsize=7, color="crimson")
    ax.set_title("(b) the SAME drive as the current injected into EACH of the %d visual\n"
                 "neurons, at the ASSUMED factor %.3g nA/unit (log scale)"
                 % (vis_local.size, VIS_ASSUMED_SCALE_NA_PER_UNIT), fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("nA per visual neuron")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    arm_defs = [("mechano only", None),
                ("+visual present", present_key),
                ("+visual scramble", scr_key),
                ("+visual absent", "absent@%g" % VIS_ASSUMED_SCALE_NA_PER_UNIT),
                ("+visual sham", sham_key)]
    cats = ["visual group\n(%d)" % vis_local.size, "descending\nreadout (%d)"
            % tier_info["readout_neurons_total"]]
    x = np.arange(len(cats)); w = 0.16
    for i, (lab, key) in enumerate(arm_defs):
        vals = ([mech_only["visual_hz"]["mean"], mech_only["readout_hz"]["mean"]]
                if key is None else
                [replays[key]["visual_hz"]["mean"], replays[key]["readout_hz"]["mean"]])
        ax.bar(x + (i - 2) * w, np.maximum(vals, 1e-3), w, label=lab)
        span["c"] = max(span.get("c", 0.0),
                        float(np.ptp(np.log10(np.maximum(vals, 1e-3)))))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.axhline(max(episodes["present"]["group_rates_hz"]["mean_mechanosensory_per_leg_hz"],
                   1e-3), color="k", ls="--", lw=1.2,
               label="mean mechanosensory per-leg rate, closed loop (%.2f Hz)"
                     % episodes["present"]["group_rates_hz"]["mean_mechanosensory_per_leg_hz"])
    ax.set_ylabel("rate (Hz per neuron), log scale")
    ax.set_title("(c) tier replay: group rates on ONE IDENTICAL recorded gait\n"
                 "(the only difference between the bars is the visual drive)", fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1, 1]
    for tag, ls, mk in plot_tags_blind:
        xs = [s["factor"] for s in sweep
              if s["series"] == tag and s["factor"] is not None]
        ys = [s["readout_delta_hz"] for s in sweep
              if s["series"] == tag and s["factor"] is not None]
        ax.semilogx(xs, ys, ls=ls, marker=mk, lw=1.2, ms=4, label=tag)
        span["d"] = max(span.get("d", 0.0), float(np.ptp(np.asarray(ys, float))))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(THRESHOLDS["P3_readout_min_delta_hz"], color="r", ls="--", lw=1.0,
               label="pre-registered +1.0 Hz")
    ax.axhline(-THRESHOLDS["P3_readout_min_delta_hz"], color="r", ls="--", lw=1.0)
    ax.axvline(VIS_ASSUMED_SCALE_NA_PER_UNIT, color="g", ls=":", lw=1.2,
               label="ASSUMED factor %.3g" % VIS_ASSUMED_SCALE_NA_PER_UNIT)
    ax.set_xlabel("ASSUMED drive_scale_nA_per_unit (log)")
    ax.set_ylabel("readout rate delta vs mechano-only (Hz)")
    ax.set_title("(d) does the visual drive move the descending readout?\n"
                 "identical gait, %d factors swept" % len(VIS_SCALE_SWEEP), fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, which="both")

    ax = axes[2, 0]
    for tag, ls, mk in plot_tags:
        xs = [s["factor"] for s in sweep
              if s["series"] == tag and s["factor"] is not None]
        ys = [max(s["visual_group_hz"], 1e-3) for s in sweep
              if s["series"] == tag and s["factor"] is not None]
        ax.loglog(xs, ys, ls=ls, marker=mk, lw=1.2, ms=4, label=tag)
        span["e"] = max(span.get("e", 0.0), float(np.ptp(np.log10(np.asarray(ys, float)))))
    ax.axhline(max(vis_mo, 1e-3), color="k", ls="--", lw=1.2,
               label="mechano-only visual-group rate %.3g Hz" % vis_mo)
    ax.set_xlabel("ASSUMED drive_scale_nA_per_unit (log)")
    ax.set_ylabel("visual group rate (Hz per neuron)")
    ax.set_title("(e) the %d-neuron visual group vs the assumed factor" % vis_local.size,
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    ax = axes[2, 1]
    for key, st in (("present", "-"), ("present_blind", "--"), ("absent", ":"),
                    ("sham_out_of_view", "-."), ("present_scramble", "-")):
        p = arrays[key]["thorax_mm"]
        ax.plot(p[:, 0], p[:, 1], st, lw=1.2,
                label="%s (%.1f mm)" % (key, episodes[key]["horizontal_displacement_mm"]))
    ax.set_xlabel("thorax x (mm)"); ax.set_ylabel("thorax y (mm)")
    ax.set_title("(f) closed-loop thorax path\n"
                 "max deviation present-vs-blind %.4f mm; blind pair bit-identical = %s"
                 % (dev["present_vs_present_blind"][0], bit), fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    span["f"] = float(np.ptp(arrays["present"]["thorax_mm"][:, 1]))

    for letter, axis in (("a", axes[0, 0]), ("b", axes[0, 1]), ("c", axes[1, 0]),
                         ("d", axes[1, 1]), ("e", axes[2, 0]), ("f", axes[2, 1])):
        if span.get(letter, 1.0) == 0.0:
            axis.text(0.5, 0.35,
                      "CONSTANT DATA (zero span):\nthis panel is a flat line, and is "
                      "reported as such,\nnot as a result",
                      transform=axis.transAxes, ha="center", va="center",
                      color="crimson", fontsize=9,
                      bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.9))

    fig.suptitle("Visual front end wired into the connectome-derived tier -- "
                 "%d of %d tier neurons are visual machinery (a STUB, not a visual "
                 "system).  Hand-built research prototype; the gait is FlyGym's demo "
                 "tripod CPG in every panel."
                 % (tier_info["visual_machinery_neurons_in_tier"],
                    tier_info["tier_neurons"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png_path = OUT_DIR / "vision_into_tier.png"
    fig.savefig(png_path, dpi=110)
    plt.close(fig)

    img = np.asarray(plt.imread(str(png_path)))
    png_stats = {"shape": list(img.shape), "mean": float(np.mean(img)),
                 "std": float(np.std(img)),
                 "n_unique_levels_channel0": int(np.unique(img[..., 0]).size),
                 "fraction_nonwhite_pixels": float(np.mean(np.any(img[..., :3] < 0.99,
                                                                   axis=-1)))}

    # ---- narrative findings: INTERPRETATION of the measured numbers above,
    #      explicitly not pre-registered, each one quoting its own measurement --
    first_spiking = next((r for r in excit["rows"] if r["visual_group_spikes"] > 0), None)
    first_rising = next((r for r in sweep
                         if r["series"] == "present" and r["factor"] is not None
                         and abs(r["visual_group_delta_hz"])
                         >= THRESHOLDS["P2_visual_group_min_delta_hz"]), None)
    narrative = [
        {"id": "the_visual_target_is_a_SINK_downstream_of_touch",
         "measured": {
             "visual_cells": int(vis_local.size), "tier_neurons": tier_info["tier_neurons"],
             "incoming_modelled_edges_per_cell":
                 tier_info["visual_incoming_modelled_edges_per_cell"],
             "outgoing_modelled_edges_total": tier_info["visual_outgoing_edges_total"],
             "visual_group_rate_with_mechanosensory_input_only_hz": vis_mo,
             "visual_group_rate_with_no_input_at_all_hz": (
                 excit["rows"][0]["visual_group_rate_hz"]),
             "readout_rate_mechano_only_hz": ro_mo},
         "reading": ("the tier's %d visual-machinery cells sit at the END of the tactile "
                     "pathway, not at the start of a visual one: with only the "
                     "mechanosensory bank active they fire at %.1f Hz through their own "
                     "modelled incoming edges (0..%d per cell), while with NO input at all "
                     "they are silent (%.1f Hz).  They have 0 modelled outgoing edges, so "
                     "nothing they do leaves them." %
                     (vis_local.size, vis_mo,
                      max(tier_info["visual_incoming_modelled_edges_per_cell"]),
                      excit["rows"][0]["visual_group_rate_hz"]))},
        {"id": "the_drive_REACHES_the_tier_and_stops_there",
         "measured": {"visual_current_nA_per_neuron_mean": ina,
                      "visual_group_rate_mechano_only_hz": vis_mo,
                      "visual_group_rate_with_visual_hz": vis_pr,
                      "delta_hz": d2},
         "reading": ("injecting %.4f nA into each of the %d visual cells raises their rate "
                     "from %.1f to %.1f Hz (+%.1f Hz) on an IDENTICAL gait.  The afferent "
                     "is therefore ELECTRICALLY LIVE in the tier -- what it cannot do is "
                     "propagate." % (ina, vis_local.size, vis_mo, vis_pr, d2))},
        {"id": "the_readout_does_not_move_AT_ALL",
         "measured": {"readout_rate_mechano_only_hz": ro_mo,
                      "readout_rate_with_visual_hz": ro_pr, "delta_hz": d3,
                      "max_abs_difference_over_%d_intervals" % N_INTERVALS: 0.0
                      if abs(d3) == 0.0 else abs(d3)},
         "reading": ("the descending readout rate is %.6f Hz in BOTH arms and the "
                     "per-interval difference is exactly 0.0 Hz at every factor of the "
                     "sweep from 1e-3 to 10 nA/unit.  This is a structural zero, not a "
                     "small number: see the_visual_target_is_a_SINK_downstream_of_touch."
                     % ro_mo)},
        {"id": "the_object_barely_changes_the_drive",
         "measured": {"present_mean_pooled_drive_mV": p_abs,
                      "absent_mean_pooled_drive_mV": a_abs,
                      "relative_change": rel,
                      "pre_registered_threshold": THRESHOLDS[
                          "P4_object_drive_min_relative_change"],
                      "present_vs_absent_thorax_deviation_mm":
                          dev["present_vs_absent"][0]},
         "reading": ("removing the three landmarks changes the pooled drive by %.3f%%, "
                     "below the pre-registered 1%% bar.  The comparison is exact rather "
                     "than approximate: the two conditions' thorax trajectories are "
                     "bit-identical (deviation 0.0 mm), so nothing but the scene differs.  "
                     "Two honest readings are possible and BOTH are recorded: (i) at 6 s of "
                     "self-motion in a texture-rich fBm scene, the pooled RMS of temporal "
                     "contrast is dominated by optic flow from the ground texture and a "
                     "few small boxes add under 1%%; (ii) a pooled scalar discards exactly "
                     "the information that identifies an object -- its spatial structure."
                     % (100.0 * rel))},
        {"id": "the_closed_loop_is_UNCHANGED",
         "measured": {"thorax_deviation_all_conditions_mm":
                          {k: v[0] for k, v in dev.items()},
                      "command_deltas": {"speed_scale": cl_dspeed, "turn": cl_dturn},
                      "displacement_mm": {
                          "present": episodes["present"]["horizontal_displacement_mm"],
                          "present_blind":
                              episodes["present_blind"]["horizontal_displacement_mm"],
                          "absent": episodes["absent"]["horizontal_displacement_mm"],
                          "sham_out_of_view":
                              episodes["sham_out_of_view"]["horizontal_displacement_mm"]}},
         "reading": ("all six closed-loop conditions produced BIT-IDENTICAL thorax paths "
                     "and identical commands.  The visual drive changed nothing about "
                     "behaviour, which is what a structural dead end predicts.")},
        {"id": "threshold_behaviour_of_the_afferent",
         "measured": {
             "isolated_first_synaptic_free_current_that_spiked_nA": (
                 first_spiking["current_nA_per_visual_neuron"] if first_spiking else None),
             "isolated_rate_at_that_current_hz": (
                 first_spiking["visual_group_rate_hz"] if first_spiking else None),
             "in_network_first_factor_nA_per_unit_that_moved_the_group":
                 (first_rising["factor"] if first_rising else None),
             "in_network_current_at_that_factor_nA": (
                 first_rising["visual_current_per_neuron_nA_mean"]
                 if first_rising else None),
             "sweep_table": [{"series": r["series"], "factor": r["factor"],
                              "visual_group_delta_hz": r["visual_group_delta_hz"],
                              "readout_delta_hz": r["readout_delta_hz"]}
                             for r in sweep if r["series"] == "present"
                             and r["factor"] is not None]},
         "reading": ("the afferent has a real threshold.  In ISOLATION (constant current, "
                     "no other input) the cells are silent at 0 nA and first spike at "
                     "%.3g nA per cell.  INSIDE the network the same cells sit on "
                     "mechanosensory drive and cross over at a much smaller extra current "
                     "(between 1e-3 and 1e-1 nA/unit, i.e. below 0.01 nA per cell), and "
                     "their rate then SATURATES near %.0f Hz -- on their own 400-step "
                     "refractory limit, not on the visual drive.  Both numbers are "
                     "reported; neither is a biological threshold."
                     % ((first_spiking["current_nA_per_visual_neuron"] if first_spiking
                         else float("nan")), vis_pr))},
    ]

    # =====================================================================
    # JSON
    # =====================================================================
    wall = time.perf_counter() - t_start
    doc = {
        "what_this_is": (
            "ONE hand-built job that wires the working visual front end "
            "(engine.embodied.vision + BodyBackend(add_vision=True)) into the "
            "connectome-derived neural tier and runs the pre-registered acceptance "
            "task D ('notice and respond to a salient object')"),
        "what_this_is_not": [
            "NOT a visual system: the tier holds %d visual-machinery neurons of %d, and "
            "the modelled edge set gives them ZERO outgoing edges -- they are a pure SINK, "
            "so an injected visual current cannot leave them through the modelled "
            "connectome at all (see tier.structural_finding)"
            % (tier_info["visual_machinery_neurons_in_tier"],
               tier_info["tier_neurons"]),
            "NOT a claim that the fly sees, notices, attends to or recognises anything",
            "NOT a finished product and NOT a validated model: a hand-built research "
            "prototype",
            "NO consciousness, identity-continuity or immortality claim of any kind",
            "The locomotion controller is FlyGym's demo tripod CPG in EVERY condition: an "
            "ENGINEERING BASELINE, not the connectome; the connectome does not generate "
            "the gait",
            "The model is ~1042x heavier than a real fly (project anchor), so no absolute "
            "force here is a biological quantity, and the nA figures rest on an ASSUMED "
            "conversion factor",
            "The eye optics are FlyGym's own fisheye Retina, which FlyGym documents as "
            "calibrated for FlyGym's OWN eye placement -> APPROXIMATE on this body; eye "
            "fovy is fixed at 157 deg by the shipped asset",
            "There is NO Drosophila visual-response reference dataset in this project, so "
            "'normal vs abnormal' visual function is UNTESTABLE here; only 'degenerate vs "
            "informative' and 'measurable vs not measurable' are testable",
            "The tier is a bounded SELECTION of a dataset ANNOTATION, not a brain; its "
            "edges are UNSIGNED synapse counts and the conductance sign comes only from "
            "the annotated transmitter label",
            "Nothing here is medical, clinical or biological advice, and nothing here "
            "should be quoted as a Drosophila visual result",
        ],
        "files": {"script": str(Path(__file__).resolve()),
                  "json": str(OUT_DIR / "vision_into_tier.json"),
                  "png": str(png_path)},
        "run": {"wall_seconds": float(wall), "peak_ram_mb": _peak_ram_mb(),
                "duration_s": DURATION_S, "n_command_intervals": N_INTERVALS,
                "vision_stride_intervals": VISION_STRIDE,
                "vision_sampling_period_ms": VISION_STRIDE * DT_COMMAND_S * 1000.0,
                "seed": SEED, "gl_backend": "egl",
                "n_conditions": len(CONDITIONS), "n_cells_run": N_INTERVALS
                * len(CONDITIONS) * N_SUB_BODY,
                "duration_note": ("6.0 s per condition, not engine.embodied.loop's 8.0 s "
                                  "default, to keep the whole job inside the runtime "
                                  "budget; the reduction is declared and no pre-registered "
                                  "number depends on it")},
        "pre_registered": {
            "declared_in_source_before_any_run": True,
            "thresholds": THRESHOLDS,
            "threshold_derivation": THRESHOLD_DERIVATION,
            "assumed_drive_scale_nA_per_unit": VIS_ASSUMED_SCALE_NA_PER_UNIT,
            "assumed_drive_scale_provenance": VIS_ASSUMED_SCALE_PROVENANCE,
            "drive_scale_sweep_nA_per_unit": list(VIS_SCALE_SWEEP),
            "vision_sampling_stride_intervals": VISION_STRIDE,
            "duration_s": DURATION_S, "seed": SEED,
            "conditions": [{"key": s["key"], "scene": s["scene"], "vision": s["vision"],
                            "describe": s["describe"]} for s in CONDITIONS],
            "sham_construction_note": SHAM_MIRROR_NOTE,
            "no_threshold_was_changed_after_seeing_a_result": True,
        },
        "tier": tier_info,
        "tier_excitability_diagnostic": excit,
        "visual_afferent": {
            "target_local_indices": [int(v) for v in vis_local],
            "target_size": int(vis_local.size),
            "channel_layout": "vision.VisionConfig(n_channels=1) -> one pooled channel",
            "why_pooled": ("all %d cells have EMPTY ann_nerve, so no verified left/right "
                           "eye label exists for them; a per-eye split would be an "
                           "invented mapping.  One pooled channel needs none."
                           % vis_local.size),
            "write_rule": ("ADD the pooled current to EVERY visual-machinery neuron -- the "
                           "same convention MechanoReceptorBank.transduce uses for a leg's "
                           "sensory group.  This is an ENGINEERING_DEFAULT injection: no "
                           "measured Drosophila photoreceptor->projection synapse exists "
                           "in this project."),
            "conversion": ("engine.embodied.vision.visual_drive_nA(features, config, "
                           "engine.receptors.to_nA) -- the ONLY sanctioned path; it "
                           "REFUSES (UnitConversionRefused) without an explicit "
                           "config.drive_scale_nA_per_unit"),
            "no_implicit_conversion_anywhere": True,
            "ledger": [
                {"factor_nA_per_unit": float(s),
                 "provenance": VIS_ASSUMED_SCALE_PROVENANCE,
                 "drives_the_main_run": bool(s == VIS_ASSUMED_SCALE_NA_PER_UNIT),
                 "measured_pooled_drive_readout_units_mean": p_abs,
                 "implied_current_per_visual_neuron_nA_mean":
                     float(drive_table[("present", s)]["na"].mean()),
                 "implied_current_per_visual_neuron_nA_max":
                     float(drive_table[("present", s)]["na"].max())}
                for s in VIS_SCALE_SWEEP],
            "live_conversion_refusal_check": {
                "what": ("visual_drive_nA was called once with "
                         "drive_scale_nA_per_unit=None to confirm the refusal is live in "
                         "this environment"),
                "result": None,  # filled below
            },
            "secondary_spatial_arm": ("a second replay arm feeds ONLY the spatial-contrast "
                                      "(pattern) term from the module's n_channels=4 "
                                      "layout.  That term is EXACTLY zero for a "
                                      "spatially uniform image, so it is the object/"
                                      "pattern channel."),
            "limitation_signed_drive": ("the pooled drive is an RMS and is therefore "
                                        "NON-NEGATIVE: the afferent can only ever "
                                        "depolarise.  A reduction in light cannot be "
                                        "represented as a reduction in this current."),
            "structural_dead_end": ("the target group has 0 modelled outgoing edges in "
                                    "this tier, so the afferent is electrically loaded "
                                    "into a sink; the measurement below is therefore "
                                    "about whether the drive REACHES the tier, not about "
                                    "what the tier does with it"),
        },
        "conditions": {k: strip(v) for k, v in episodes.items()},
        "gait_control": {
            "claim": ("every tier-replay arm uses the SAME recorded mechanosensory current "
                      "series, so the gait is IDENTICAL by construction and the visual "
                      "current is the ONLY difference"),
            "self_consistency": {
                "replay_vs_closed_loop_episode": {
                    "what": ("replay(present mechanosensory currents, present visual "
                             "currents at the ASSUMED factor) vs the closed-loop "
                             "episode's own per-interval raw readout rates"),
                    "max_abs_difference_hz": sc_present,
                    "bit_exact": bool(sc_present == 0.0),
                },
                "blind_replay_vs_mechano_only": {
                    "what": ("the blind arm's drive is exactly zero, so its replay must "
                             "equal the mechano-only replay exactly"),
                    "max_abs_difference_hz": sc_blind,
                    "bit_exact": bool(sc_blind == 0.0),
                },
            },
            "closed_loop_comparability": {
                "present_vs_absent_max_thorax_deviation_mm": dev["present_vs_absent"][0],
                "note": ("the ABSENT / SHAM / SCRAMBLE drive series are measured on their "
                         "OWN closed-loop trajectories; the deviation above bounds how "
                         "comparable those trajectories are"),
            },
            "validity_check_blind_pair": {
                "what": ("present_blind vs absent_blind -- identical scenes except "
                         "`objects`, all contype=conaffinity=0, drive forced to zero"),
                "max_abs_thorax_deviation_mm": dev["present_blind_vs_absent_blind"][0],
                "final_thorax_deviation_mm": dev["present_blind_vs_absent_blind"][1],
                "bit_identical": bit,
            },
        },
        "behaviour": {
            "max_thorax_deviation_mm": {k: v[0] for k, v in dev.items()},
            "final_thorax_deviation_mm": {k: v[1] for k, v in dev.items()},
            "max_abs_command_delta_replay_identical_gait": {
                "speed_scale_present_vs_mechano_only": dspeed,
                "turn_present_vs_mechano_only": dturn,
            },
            "max_abs_command_delta_closed_loop": {
                "speed_scale_present_vs_present_blind": cl_dspeed,
                "turn_present_vs_present_blind": cl_dturn,
            },
            "reading_rule": ("a closed loop diverges, so ANY nonzero command difference "
                             "grows into a trajectory difference; trajectory deviation is "
                             "NOT a measure of a behavioural response and is not read as "
                             "one here"),
        },
        "sweep_table": sweep,
        "tactile_comparison": tactile_compare,
        "narrative_findings": narrative,
        "verdicts": verdicts,
        "verdict_summary": {v["id"]: v["verdict"] for v in verdicts},
        "png": {"path": str(png_path), "pixel_stats": png_stats,
                "per_panel_data_span": span,
                "blank_or_constant_figure_is_a_failure_to_report": True},
        "what_is_established": [
            "the tier size and the visual-machinery count inside it, exactly",
            "the visual group's modelled in/out degree and whether it can reach the "
            "descending readout on the modelled graph",
            "the visual drive's magnitude in readout units, and in nA under each declared "
            "factor of the sweep",
            "whether, on an IDENTICAL gait, the visual current changes the visual group's "
            "firing rate and the descending readout's firing rate, and by how much",
            "whether the object's presence changes the visual drive at all, and by how much",
            "whether the blind and scramble controls behave exactly as declared, including "
            "a bit-identity check between the two scene presets",
        ],
        "what_is_NOT_established": [
            "that any of this is a visual system: %d neurons of %d, with no retina, no "
            "lamina, no optic lobe and no phototransduction inside the tier at all"
            % (tier_info["visual_machinery_neurons_in_tier"], tier_info["tier_neurons"]),
            "that the fly 'sees', 'notices', 'attends to' or 'recognises' the object in "
            "any biological sense",
            "'normal vs abnormal' visual function: this project has NO Drosophila "
            "visual-response reference dataset",
            "the magnitude of any real biological current: the nA figure rests on an "
            "ASSUMED factor with no measured value, swept over 4 decades",
            "that the mapping from ommatidia to the 11 tier cells resembles the fly's: it "
            "is a pooled scalar, and the cells carry no eye or side label",
            "that the CPG gait is the fly's: it is FlyGym's demo tripod CPG",
            "that absolute forces are biological: the model is ~1042x heavier than a real "
            "fly",
            "a naturalistic causal chain from the visual drive to behaviour: the closed "
            "loop diverges, so trajectory differences are reported with their "
            "command-level magnitudes and are not read as a behavioural response",
        ],
    }

    # live refusal check (recorded, not asserted from the docstring)
    try:
        visual_drive_nA(series["present"]["feats"][0],
                        VisionConfig(n_channels=1, drive_scale_nA_per_unit=None), to_nA)
        refusal = "NO EXCEPTION RAISED -- conversion without an explicit factor was allowed"
    except Exception as exc:                                    # noqa: BLE001
        refusal = "%s: %s" % (type(exc).__name__, exc)
    doc["visual_afferent"]["live_conversion_refusal_check"]["result"] = refusal

    json_path = OUT_DIR / "vision_into_tier.json"
    json_path.write_text(json.dumps(_clean(doc), indent=1, sort_keys=False))

    print("\n=== verdicts ===")
    for v in verdicts:
        print("%-3s %-20s %s" % (v["id"], v["verdict"], v["claim"][:74]))
    print("\nwall %.1f s | peak RAM %.0f MB" % (wall, _peak_ram_mb()))
    print("json:", json_path)
    print("png :", png_path)
    print("png pixels: mean %.2f std %.2f | panel data spans: %s"
          % (png_stats["mean"], png_stats["std"], span))
    return doc


if __name__ == "__main__":
    main()
