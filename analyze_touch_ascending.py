"""How body touch reaches the brain (BANC), and what a neck cut actually severs.

THE ANSWER THIS PRODUCES (weighted by real synapse counts)
----------------------------------------------------------
BODY mechanosensory neurons do not send most of their output to the brain. They
land predominantly on LOCAL interneurons of the ventral nerve cord, with a
sizable monosynaptic fraction onto ascending neurons, and a small local reflex
fraction onto motor neurons. The brain is reached by the relay chain
sensory -> VNC local interneuron -> ascending neuron -> brain.

HEAD mechanosensory neurons (antennal Johnston's organ, interommatidial
sensilla) enter the BRAIN directly and barely touch the VNC, so a neck cut does
not remove them at all.  An earlier version of this analysis mixed head and body
sources and therefore concluded that severing every ascending neuron barely
isolates the brain; the mixture, not the anatomy, produced that.

WHAT IS REAL vs INFERRED
------------------------
REAL: `ann_class` / `ann_nerve` / `ann_super_class` / `ann_primary_type` labels,
and the cached pre->post edge list with synapse counts.
INFERRED (stated in the report, not as fact):
  * "ascending"/"descending" is the dataset's label; this cache has no soma or
    arbor coordinates, so no single cell's polarity is independently verified.
  * Every edge counted here is unsinged: the cached `nt_pair` is -1 for every
    row, so these are synapse counts, not excitation/inhibition.
  * The neck cut is an edge removal over annotated classes, not a cut plane
    reconstructed from anatomy.
  * Conditional on the cached snapshot, which disagrees with the current
    FlyWire `synapse_table` build for at least one measured cell.

Run: venv/bin/python analyze_touch_ascending.py
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "brain_isolation"

MECH_LABELS = ("bristle_neuron", "chordotonal_organ_neuron",
               "campaniform_sensillum_neuron", "hair_plate_neuron",
               "multidendritic_neuron", "taste_bristle_tactile_neuron")
GUSTATORY_LABELS = ("taste_bristle_gustatory_neuron",)

BODY_NERVE_KEYWORDS = ("leg_nerve", "abdominal_nerve", "dorsal_mesothoracic_nerve",
                       "dorsal_metathoracic_nerve", "thoracic")
HEAD_NERVE_KEYWORDS = ("antennal_nerve", "eye_nerve", "maxillary-labial_nerve",
                       "occipital_nerve", "frontal_nerve")

# Annotated populations whose axons run through the cervical connective.
CROSSING = ("ascending", "sensory_ascending", "ascending_visceral_circulatory",
            "descending", "sensory_descending")
BRAIN_TARGETS = ("central_brain_intrinsic", "optic_lobe_intrinsic",
                 "visual_projection", "visual_centrifugal")


def weighted_profile(pre, post, syn, mask_src, labels, total=None):
    """Synapse-weighted target profile, grouped by a per-neuron label array."""
    e = mask_src[pre]
    tot = float(syn[e].sum()) if total is None else float(total)
    prof = collections.Counter()
    for c, w in zip(labels[post][e], syn[e]):
        prof[str(c)] += float(w)
    return tot, {k: {"synapses": v, "fraction": v / tot if tot else 0.0}
                 for k, v in prof.most_common()}


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "data" / "flywire" / "banc_connectome.npz", allow_pickle=False)
    pre, post, syn = d["pre"], d["post"], d["syn"]
    n = len(d["root_ids"])
    sc, cls = d["ann_super_class"], d["ann_class"]
    nerve, primary = d["ann_nerve"], d["ann_primary_type"]
    verified, nt_pair = d["ann_nt_verified"], d["nt_pair"]

    mech = np.isin(cls, list(MECH_LABELS))
    body_nerve = np.array([any(k in s for k in BODY_NERVE_KEYWORDS) for s in nerve])
    head_nerve = np.array([any(k in s for k in HEAD_NERVE_KEYWORDS) for s in nerve])
    body_mech = mech & body_nerve & ~head_nerve
    head_mech = mech & head_nerve & ~body_nerve
    crossing = np.isin(sc, list(CROSSING))
    brain_target = np.isin(sc, list(BRAIN_TARGETS))
    vnc = sc == "ventral_nerve_cord_intrinsic"
    ascending_like = np.isin(sc, ["ascending", "sensory_ascending",
                                  "ascending_visceral_circulatory"])

    report = {
        "question": "how does BODY touch reach the brain, and what does a neck cut remove?",
        "scope": {
            "real": ["ann_class / ann_nerve / ann_super_class / ann_primary_type labels",
                     "cached pre->post pairs with synapse counts"],
            "inferred_not_measured": [
                "ascending/descending polarity is the dataset's label; no soma or arbor coordinates exist in this cache",
                "all edges are UNSIGNED: cached nt_pair is -1 on every row, so these are synapse counts, not excitation/inhibition",
                "the neck cut is an annotated-class edge removal, not a reconstructed cut plane",
                "conditional on the cached snapshot, which disagrees with the current synapse_table build for at least one measured cell"],
        },
        "dataset": {"neurons": int(n), "unique_pairs": int(len(pre)),
                    "nt_pair_all_minus_one": bool(np.all(nt_pair == -1)),
                    "neurons_with_verified_label": int(np.count_nonzero(verified != ""))},
        "source_partition": {
            "mechanosensory_total": int(mech.sum()),
            "by_class": {lab: int((cls == lab).sum()) for lab in MECH_LABELS},
            "excluded_gustatory": {lab: int((cls == lab).sum()) for lab in GUSTATORY_LABELS},
            "body_route": int(body_mech.sum()),
            "head_route": int(head_mech.sum()),
            "unassigned_nerve": int((mech & ~body_mech & ~head_mech).sum()),
        },
    }

    for tag, mask in (("body", body_mech), ("head", head_mech)):
        tot, prof = weighted_profile(pre, post, syn, mask, sc)
        top_types = collections.Counter()
        for c, w in zip(primary[post][mask[pre]], syn[mask[pre]]):
            top_types[str(c)] += float(w)
        report[f"{tag}_touch_output_profile"] = {
            "source_neurons": int(mask.sum()),
            "outgoing_edges": int(mask[pre].sum()),
            "outgoing_synapses": tot,
            "by_target_super_class": {k: v for k, v in list(prof.items())[:10]},
            "top_target_cell_types": [
                {"type": k, "fraction": v / tot if tot else 0.0}
                for k, v in top_types.most_common(8)] if tot else [],
        }

    # Relay bottleneck: where does VNC local output go?
    e = vnc[pre]
    vnc_out = float(syn[e].sum())
    report["relay_bottleneck"] = {
        "vnc_intrinsic_outgoing_synapses": vnc_out,
        "to_ascending_like": {"synapses": float(syn[e & ascending_like[post]].sum()),
                              "fraction": float(syn[e & ascending_like[post]].sum() / vnc_out)},
        "to_vnc_intrinsic": {"synapses": float(syn[e & vnc[post]].sum()),
                             "fraction": float(syn[e & vnc[post]].sum() / vnc_out)},
        "to_motor": {"synapses": float(syn[e & (sc[post] == "motor")].sum()),
                     "fraction": float(syn[e & (sc[post] == "motor")].sum() / vnc_out)},
        "interpretation": "only this fraction of local nerve-cord output is addressed to ascending neurons; the rest stays local",
    }

    # Brain-side input budget and how much of it a neck cut removes.
    cbi = sc == "central_brain_intrinsic"
    e_in = cbi[post]
    total_in = float(syn[e_in].sum())
    report["brain_input_budget"] = {
        "central_brain_intrinsic_incoming_synapses": total_in,
        "from_ascending_like": {"synapses": float(syn[e_in & ascending_like[pre]].sum()),
                                "fraction": float(syn[e_in & ascending_like[pre]].sum() / total_in)},
        "from_sensory_head_direct": {"synapses": float(syn[e_in & head_mech[pre]].sum()),
                                     "fraction": float(syn[e_in & head_mech[pre]].sum() / total_in)},
        "from_body_mech_direct": {"synapses": float(syn[e_in & body_mech[pre]].sum()),
                                  "fraction": float(syn[e_in & body_mech[pre]].sum() / total_in)},
    }

    # The neck cut as a bandwidth number, not a reachability claim.
    cut_mask = crossing[pre]
    report["neck_cut"] = {
        "definition": "remove ALL outgoing edges of every annotated neck-crossing population (ascending-like and descending-like)",
        "crossing_neurons": int(crossing.sum()),
        "crossing_neurons_ascending_like": int(ascending_like.sum()),
        "crossing_neurons_descending_like": int((sc == "descending").sum()),
        "edges_severed": int(cut_mask.sum()),
        "synapses_severed": float(syn[cut_mask].sum()),
        "of_which_ascending_like_to_brain": float(syn[cut_mask & brain_target[post]].sum()),
        "reachability_is_NOT_used_as_evidence": (
            "removing these edges still leaves almost every central-brain-intrinsic neuron "
            "reachable from touch sensors within a few hops, because the cached graph is "
            "dense and polysynaptic; therefore connectivity alone cannot demonstrate a "
            "labelled ascending pathway. The synapse-weighted profile above is the usable "
            "measurement."),
    }

    report["sign_coverage"] = {
        "all_edges": {"edges": int(len(pre)),
                      "with_pure_verified_label": int((verified[pre] != "").sum())},
        "body_touch_edges": {"edges": int(body_mech[pre].sum()),
                             "with_pure_verified_label": int((body_mech[pre] & (verified[pre] != "")).sum())},
        "note": "a pure label is necessary but not sufficient for a sign; receptor identity and reversal potential still decide it",
    }
    report["run"] = {"wall_clock_s": time.perf_counter() - t0}
    (OUT / "touch_ascending_report.json").write_text(json.dumps(report, indent=2, default=str))
    np.savez_compressed(OUT / "touch_ascending_arrays.npz",
                        body_mech=body_mech, head_mech=head_mech,
                        crossing=crossing, vnc_intrinsic=vnc)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
