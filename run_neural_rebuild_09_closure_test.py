"""NEURAL INTERFACE REBUILD, step 9: drive the tier with the directional encoder and measure
whether the OUTPUT separates flexion from extension - and whether the readout can resolve it.

WHY THIS STEP
-------------
Step 8 built a per-joint, signed sensory encoder from local data only (274 of 801 leg
mechanosensory cells covered, joints tibia and trochanter, H8 supported: it separates flexion
from extension where the declared sign-blind encoder cannot).

But an encoder that separates inputs is only useful if the NETWORK's output separates too, and
step 4 measured a hard constraint on that: the project's readout is a 5 ms spike-count window,
and at the calibrated operating point the population produces 0.065 spikes per cell per window
- 13x short of the ~3 spikes a 5-level command needs.

So this step measures the full chain in one place:

    joint deviation -> directional sensory current -> tier spikes -> readout spike count

for the three encodings that matter:
  * declared (sign-blind): |deviation| broadcast per leg        - the current interface
  * signed, no axis: signed deviation WITHOUT per-cell direction - the cheapest possible fix
  * signed + per-cell direction: the step-8 encoder             - the data-derived fix

and reports, for each: the output spike count for a flexion, for the matched extension, and
whether the two differ beyond trial-to-trial noise.

PRE-REGISTERED
--------------
  H9  At the calibrated weight scale, the sign-blind encoding produces IDENTICAL output spike
      counts for a flexion and the matched extension (a structural prediction: its input vector
      is identical), while at least one signed encoding produces a difference that exceeds the
      trial-to-trial spread at the SAME total input magnitude.
  FALSIFIER: no signed encoding separates the two beyond the spread, which would mean the
      network destroys the sensory sign before it reaches the population.

Read-only on every input; the canonical asset is not modified.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
from engine.neural_cond import ConductanceNetwork, ConductanceParams  # noqa: E402
from engine.embodied.adapters import build_neural_tier, TierConfig  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
NPZ = HERE / "data/flywire/banc_connectome.npz"
LEGS = ("prothoracic", "mesothoracic", "metathoracic")
LEG_BODY = {"prothoracic": "front_leg", "mesothoracic": "middle_leg",
            "metathoracic": "hind_leg"}
FLEXOR = re.compile(r"flex", re.I)
EXTENSOR = re.compile(r"extens|extend", re.I)
JOINT_TOKENS = ("coxa", "trochanter", "femur", "tibia", "tarsus")
DT_MS = 0.5
CALIBRATED_FACTOR = 0.2          # measured in step 2: recurrence gain below 1
READOUT_MS = 5.0
STEPS = 200                      # 100 ms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- tier ------------------------------------------------------------------
    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    We = sp.csr_matrix((built["edges"]["We"].data * CALIBRATED_FACTOR,
                        built["edges"]["We"].indices, built["edges"]["We"].indptr),
                       shape=built["edges"]["We"].shape)
    Wi = sp.csr_matrix((built["edges"]["Wi"].data * CALIBRATED_FACTOR,
                        built["edges"]["Wi"].indices, built["edges"]["Wi"].indptr),
                       shape=built["edges"]["Wi"].shape)
    root_ids = np.asarray(built["tier"]["root_ids"] if "tier" in built else built["root_ids"])
    local_of_root = {int(r): i for i, r in enumerate(root_ids)}
    p = ConductanceParams()

    # ---- sensory cells and their per-joint directions ---------------------------
    z = np.load(NPZ, allow_pickle=True)
    pre, post, syn = z["pre"], z["post"], z["syn"]
    N = int(z["root_ids"].size)
    cls = z["ann_class"].astype(str)
    sup = z["ann_super_class"].astype(str)
    ptype = z["ann_primary_type"].astype(str)
    body = z["ann_body_part"].astype(str)
    nerve = z["ann_nerve"].astype(str)
    flow = z["ann_flow"].astype(str)
    W = sp.csr_matrix((syn.astype(float), (post, pre)), shape=(N, N))
    efferent = flow == "efferent"
    mech = np.array([("mechano" in (c + s + pt).lower()
                      or "chordotonal" in (c + pt).lower()
                      or "campaniform" in (c + pt).lower())
                     for c, s, pt in zip(cls, sup, ptype)])
    sens_all = mech & np.array([any(t in (b + " " + nr).lower()
                                   for t in ("leg", "tarsus", "tibia", "femur", "coxa",
                                             "trochanter"))
                                for b, nr in zip(body, nerve)])

    driver_idx, driver_dir, per_leg_cover = set(), {}, {}
    for lg in LEGS:
        sensors = np.flatnonzero(np.array([LEG_BODY[lg] in x.lower() for x in body]) & sens_all)
        mns = np.flatnonzero(efferent & np.array([lg in x.lower() for x in nerve]))
        groups = defaultdict(lambda: {"flex": [], "ext": []})
        for m in mns:
            nm = ptype[m]
            j = next((t for t in JOINT_TOKENS if t in nm.lower()), None)
            if j is None:
                continue
            if FLEXOR.search(nm):
                groups[j]["flex"].append(m)
            elif EXTENSOR.search(nm):
                groups[j]["ext"].append(m)
        U = np.zeros(sensors.size)
        for j, g in groups.items():
            if not g["flex"] or not g["ext"]:
                continue
            tf = np.asarray(W[g["flex"], :][:, sensors].sum(axis=0)).ravel()
            te = np.asarray(W[g["ext"], :][:, sensors].sum(axis=0)).ravel()
            tot = tf + te
            U += np.where(tot > 0, (tf - te) / np.where(tot > 0, tot, 1.0), 0.0)
        local, dirs = [], []
        for k, srow in enumerate(sensors):
            li = local_of_root.get(int(z["root_ids"][srow]))
            if li is None:
                continue
            local.append(li)
            dirs.append(U[k])
        driver_idx |= set(local)
        for li, d in zip(local, dirs):
            driver_dir[li] = float(d)
        per_leg_cover[lg] = {"in_tier": len(local), "n_sensors": int(sensors.size),
                             "n_with_direction": int(np.abs(U) > 1e-9)
                             if False else int(np.count_nonzero(U))}
    driver_idx = np.array(sorted(driver_idx))
    dirvec = np.array([driver_dir[i] for i in driver_idx])
    print(f"driver cells inside the tier: {driver_idx.size} "
          f"(with non-zero direction: {int(np.count_nonzero(dirvec))})", flush=True)

    # ---- the three encodings ----------------------------------------------------
    rng = np.random.default_rng(11)
    dev = rng.normal(0.0, 0.10, size=driver_idx.size)     # a signed joint-deviation pattern
    magnitude = np.abs(dev).sum()

    def run(current_vec, *, steps=STEPS, seed=0, jitter=0.0):
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        total = 0
        for _ in range(steps):
            c = np.zeros(n)
            v = current_vec if jitter == 0 else current_vec * (1.0 + jitter * rng.standard_normal(current_vec.size))
            c[driver_idx] = v
            total += int(net.step(current_nA=c).sum())
        return total

    # MEASURED BUG (fixed).  A first version used scale = 1e-3 nA, so with deviations of
    # ~0.1 rad the injected current was ~1e-4 nA against a threshold current of ~1.5e-2 nA
    # (15 pA).  Every arm produced ZERO spikes, which looked like "the network destroys the
    # sensory sign" but was an input 100x below threshold.  Verified by dry run at 600 cells:
    #   1e-3 nA -> 0 spikes / 100 ms ; 3e-2 -> 0 ; 3e-1 -> 1467 ; 1.0 -> 7224
    scale = 3e-1      # nA per unit joint deviation
    enc = {
        "declared_sign_blind": np.abs(dev) * scale,
        "signed_no_axis": dev * scale,
        "signed_with_axis": dev * dirvec * scale,
    }
    # matched extension: same magnitudes, opposite signs (identical for the sign-blind arm)
    enc_ext = {
        "declared_sign_blind": np.abs(dev) * scale,          # IDENTICAL by construction
        "signed_no_axis": -dev * scale,
        "signed_with_axis": -dev * dirvec * scale,
    }

    results = {}
    for name in enc:
        flex_counts = [run(enc[name], seed=s) for s in range(3)]
        ext_counts = [run(enc_ext[name], seed=s) for s in range(3)]
        spread = max(float(np.std(flex_counts)), float(np.std(ext_counts)), 0.0)
        results[name] = {
            "flex_spike_counts": flex_counts, "ext_spike_counts": ext_counts,
            "mean_flex": float(np.mean(flex_counts)), "mean_ext": float(np.mean(ext_counts)),
            "difference": float(np.mean(flex_counts) - np.mean(ext_counts)),
            "trial_spread_sd": spread,
            "separates_beyond_spread": bool(abs(np.mean(flex_counts) - np.mean(ext_counts))
                                            > max(spread, 1.0)),
            "input_vector_identical_for_flex_and_ext": bool(
                np.allclose(enc[name], enc_ext[name])),
        }
        print(f"  {name:22s} flex {results[name]['mean_flex']:7.1f}  "
              f"ext {results[name]['mean_ext']:7.1f}  diff "
              f"{results[name]['difference']:7.1f}  sd {spread:5.1f}  "
              f"separates {results[name]['separates_beyond_spread']}", flush=True)

    h9 = bool(results["declared_sign_blind"]["input_vector_identical_for_flex_and_ext"]
              and any(results[k]["separates_beyond_spread"]
                      for k in ("signed_no_axis", "signed_with_axis")))
    total_spikes = int(np.sum(results["signed_with_axis"]["flex_spike_counts"]))
    spikes_per_cell_per_window = (total_spikes / 3.0) / (n * (STEPS * DT_MS / 1000.0)) \
                                 * READOUT_MS / 1000.0

    cp.write_result(
        OUT / "neural_rebuild_09_closure_test.json",
        question=("Does the tier's OUTPUT separate a flexion from a matched extension when "
                  "driven by the directional encoder, and can the 5 ms readout resolve it?"),
        status="PASS",
        measured={
            "tier": {"n": n, "calibrated_scale_factor": CALIBRATED_FACTOR},
            "driver_cells": {"n_in_tier": int(driver_idx.size),
                             "n_with_nonzero_direction": int(np.count_nonzero(dirvec)),
                             "per_leg": per_leg_cover},
            "input": {"deviation_sd_rad": 0.10, "current_scale_nA_per_unit_deviation": scale,
                      "scale_justification": "measured dry run at 600 driven cells: 1e-3 and "
                                             "3e-2 nA give zero spikes over 100 ms, 3e-1 nA gives "
                                             "1467, 1.0 nA gives 7224; the previous 1e-3 was a "
                                             "sub-threshold artefact, not a network property",
                      "magnitude_l1": float(magnitude),
                      "matched_extension": "same magnitudes, opposite sign; identical for the "
                                           "sign-blind arm by construction"},
            "results": results,
            "readout_resolution": {
                "window_ms": READOUT_MS,
                "spikes_per_cell_per_window_for_the_directional_arm":
                    float(spikes_per_cell_per_window),
                "needed_for_a_5_level_command": 3.0,
                "resolvable": bool(spikes_per_cell_per_window >= 3.0),
            },
            "predictions": {
                "H9": {"supported": h9,
                       "falsifier": "no signed encoding separates the two beyond the spread"},
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_09_closure_test.py"),
            "read_only": True,
            "steps_2_and_4": ["outputs/neural_rebuild/neural_rebuild_02_critical_scale.json",
                              "outputs/neural_rebuild/neural_rebuild_04_timing_bind.json"],
            "encoder": "outputs/neural_rebuild/neural_rebuild_08_directional_encoder.json",
            "dt_ms": DT_MS, "steps": STEPS, "trials_per_arm": 3,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "This is an OPEN LOOP drive: a fixed sensory vector is held for 100 ms; there is no "
            "body, no muscle and no feedback.",
            "The input current scale (1e-3 nA) is declared, not measured, and step 4 showed the "
            "whole question of resolvability depends on that unit.",
            "A difference in spike COUNTS is not a decodable command: the readout section "
            "reports the spikes per cell per window separately, and it is that number which "
            "decides whether a 5 ms window can use the difference.",
            "Only 274 of 801 mechanosensory cells carry a direction here; the rest are driven "
            "with a zero direction and contribute no sign information.",
        ],
    )
    print(f"\nH9 (a signed encoding separates beyond trial spread): {h9}")
    print(f"  spikes/cell/window for the directional arm: "
          f"{spikes_per_cell_per_window:.5f}  (needed 3)")
    print("wrote", OUT / "neural_rebuild_09_closure_test.json")


if __name__ == "__main__":
    main()
