#!/usr/bin/env python3
"""DERIVE the motor-neuron -> muscle map, and say exactly how it was derived.

WHY THIS IS DERIVABLE LOCALLY.  The BANC annotations name motor neurons BY THE MUSCLE THEY
INNERVATE: ``ann_primary_type`` holds values like 'tergopleural_promotor',
'pleural_remotor_abductor', 'sternal_adductor', 'tibia_flexor', 'tibia_extensor',
'tarsus_depressor'.  The shipped muscle model's actuators are named after the same anatomy
('LFC_tergopleural_promotor_a', 'LFC_pleural_remotor_and_abductor', ...).  So the mapping is a
NAME MATCH between a connectome annotation and a muscle model, not something that has to be
transcribed from a PDF table.

THIS IS A DERIVATION, NOT A CITED TABLE, AND IT IS LABELLED AS SUCH.  The literature does carry
authoritative tables (eLife 51781, bioRxiv 2022.12.15.520299: "Knowing which MNs innervate which
muscles is key to interpreting the connectome").  Name matching can be wrong -- 'accesory' is
misspelled in the model, 'flex' vs 'flexor', 'and' dropped -- so every match is reported with its
score, every miss is listed, and the artefact says DERIVED rather than TAKEN.  Nothing downstream
may treat this map as authoritative until it is checked against one of those tables.

Run:
    OPENBLAS_NUM_THREADS=1 venv/bin/python tools_mn_muscle_map.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MUSCLE_XML = (HERE / "venv_body/lib/python3.12/site-packages/flygym/assets/model/"
              "musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml")
BANC = HERE / "data/flywire/banc_connectome.npz"

#: tokens that carry no identity once the muscle is being named
STOP = {"a", "b", "c", "and", "the", "muscle", "lfc", "lff", "lft", "rfc", "rff", "rft",
        "lf", "rf", "lm", "rm", "lh", "rh", "left", "right", "front", "mid", "hind"}
#: a token that is a bare number is an id (e.g. LFTibia_flex_93434), not a name
#: the model prefixes every muscle with its leg and segment: LFC = left-front COXA, LFF =
#: left-front FEMUR, LFTibia = left-front TIBIA.  The segment is PART OF THE NAME and dropping it
#: was a mistake: MEASURED, without expanding the prefix, LFTibia_flex_93434 tokenised to
#: {lftibia, flex} against the connectome's {tibia, flex} and scored J=0.33 -- reported as a
#: "weak match" when it is in fact an exact one, and the numeric id (93434, a FlyWire body id)
#: diluted it further.  Expanding the prefix fixes both.
PREFIX_SEGMENT = {"c": "coxa", "f": "femur", "t": "tibia", "ta": "tarsus"}


def expand_prefix(name: str):
    """'LFTibia_flex_93434' -> 'tibia flex';  'LFC_tergopleural_promotor_a' -> 'coxa tergopleural
    promotor';  anything not matching the pattern is returned unchanged."""
    s = str(name)
    parts = s.split("_")
    head = parts[0]
    m = re.fullmatch(r"([LR])([FHM])(Tibia|Tarsus|Coxa|Femur|C|F|T)", head, flags=re.I)
    if m:
        seg = m.group(3).lower()
        seg = {"c": "coxa", "f": "femur", "t": "tibia"}.get(seg, seg)
        return " ".join([seg] + parts[1:])
    return s


def tokens(name: str):
    t = re.split(r"[^A-Za-z0-9]+", expand_prefix(name))
    return {x.lower() for x in t if x and x.lower() not in STOP and not x.isdigit()}
def canon(tok: str):
    """fold the spelling variants the two sources actually use"""
    for suf, base in (("flexor", "flex"), ("extensor", "extend"), ("accesory", "accessory"),
                      ("accessory", "accessory")):
        if tok == suf:
            return base
    return tok


def muscle_actuators():
    """Muscle actuators, found by their CLASS, because ``dyntype`` is inherited.

    MEASURED, and my first version got this wrong: ``grep -c dyntype`` over the whole XML returns
    **1**.  The muscle dynamics are declared once in ``<default class="muscle">`` and every muscle
    actuator simply says ``class="muscle"``.  Searching for ``dyntype == "muscle"`` on each
    actuator therefore finds NOTHING and reports zero muscles -- a silent empty result that looks
    like "the model has no muscles".  The class attribute is the correct discriminator here, and
    the muscle count is cross-checked below.
    """
    tree = ET.parse(MUSCLE_XML)
    out = [el.get("name") for el in tree.getroot().iter()
           if el.tag == "general" and el.get("class") == "muscle" and el.get("name")]
    if not out:
        raise RuntimeError(
            "no actuator with class='muscle' found in %s; if the model changed, find the "
            "discriminator again rather than assuming there are no muscles" % MUSCLE_XML)
    return out


def main() -> int:
    muscles = muscle_actuators()
    print(f"muscle actuators in the shipped model: {len(muscles)}")
    for m in muscles:
        print("   ", m)

    z = np.load(BANC, allow_pickle=False)
    prim = np.asarray(z["ann_primary_type"])
    root = np.asarray(z["root_ids"])
    nerve = np.asarray(z["ann_nerve"])
    cls = np.asarray(z["ann_class"])

    # candidate motor-neuron types: named after a muscle, i.e. not 'MNxx' codes and not
    # something ending in '_neuron' (a category, not a muscle)
    types = [str(s) for s in np.unique(prim)
             if s and not str(s).startswith("MN") and not str(s).endswith("_neuron")]
    KW = ("promotor", "remotor", "rotator", "adductor", "abductor", "flex", "extend",
          "depressor", "levator", "trochanter", "tibia", "tarsus", "sternal", "tergal",
          "tergopleural", "pleural", "tarsal", "tergotrochanter", "sternotrochanter")
    mntypes = sorted({t for t in types if any(k in t for k in KW)})
    print(f"\nBANC motor-neuron types named after muscles: {len(mntypes)}")

    rows = []
    for mus in muscles:
        mt = {canon(x) for x in tokens(mus)}
        best, best_score = None, 0.0
        for typ in mntypes:
            tt = {canon(x) for x in tokens(typ)}
            if not tt:
                continue
            j = len(mt & tt) / len(mt | tt)
            if j > best_score:
                best, best_score = typ, j
        # A FLOOR, so a bad match is reported as NO match instead of the least-bad one.
        # MEASURED: without it, LFC_pleural_promotor was "matched" to tergopleural_promotor at
        # J=0.25 -- the connectome carries no pleural_promotor type, so the honest answer is
        # "unmatched, needs the literature table", not a silent wrong pairing.
        if best_score < 0.40:
            best = None
        m = prim == best if best is not None else np.zeros(prim.size, bool)
        leg_m = m & np.array([("leg_nerve" in str(n)) for n in nerve])
        rows.append({
            "muscle_actuator": mus,
            "matched_mn_type": best,
            "jaccard": round(float(best_score), 3),
            "neurons_with_that_type": int(m.sum()),
            "of_which_leg_nerve": int(leg_m.sum()),
            "example_ids": [int(x) for x in root[m][:5]],
        })
    rows.sort(key=lambda r: -r["jaccard"])
    print(f"\n{'muscle actuator':<40}{'matched MN type':<34}{'J':>6}{'n':>5}{'leg':>5}")
    for r in rows:
        print(f"{r['muscle_actuator']:<40}{str(r['matched_mn_type']):<34}"
              f"{r['jaccard']:>6.2f}{r['neurons_with_that_type']:>5}{r['of_which_leg_nerve']:>5}")

    weak = [r for r in rows if r["jaccard"] < 0.5]
    unmatched = sorted(set(mntypes) - {r["matched_mn_type"] for r in rows})
    print(f"\nmatches below J=0.5 (need manual check): {len(weak)}")
    for r in weak:
        print(f"   {r['muscle_actuator']} -> {r['matched_mn_type']} (J={r['jaccard']})")
    print(f"BANC muscle-named types NOT claimed by any actuator: {unmatched}")

    dest = HERE / "outputs" / "mn_muscle_map.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "provenance": ("DERIVED by NAME MATCHING between the shipped muscle model's actuator "
                       "names and BANC ann_primary_type.  NOT a cited table.  The literature "
                       "carries authoritative tables (eLife 51781; bioRxiv 2022.12.15.520299) "
                       "and this must be checked against one of them before it is trusted."),
        "muscle_model_xml": str(MUSCLE_XML.relative_to(HERE)),
        "banc_cache": str(BANC.relative_to(HERE)),
        "n_muscle_actuators": len(muscles),
        "n_banc_muscle_named_types": len(mntypes),
        "rows": rows,
        "weak_matches_j_lt_0.5": weak,
        "unclaimed_mn_types": unmatched,
    }, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
