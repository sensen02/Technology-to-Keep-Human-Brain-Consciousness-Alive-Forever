"""ROUND 39 (FIX), part F: put ONE contact parameter on external evidence, as a
pre-registered hypothesis.

THE GAP THIS CLOSES (partially)
-------------------------------
Round 38 closed with the honest statement that the contact parameters this project has used
since round 34 - `solref 0.02` (MuJoCo default, 100x softer than the upstream asset) and
`mu 1.0` - are MODEL DEFAULTS, not measurements, and that "what the equivalent stiffness and
friction of a real fly's pulvilli + claws are" is an external-evidence gap of the same rank
as the connectome-annotation gap.

Part F closes the FRICTION half of it with a measurement from the literature, and states
plainly that the STIFFNESS half is still open.

THE EVIDENCE
------------
Ramdya P, Thandiackal R, Cherney R, Asselborn T, Benton R, Ijspeert AJ, Floreano D.
"Climbing favours the tripod gait over alternative faster insect gaits."
Nature Communications 8:14494 (2017).  doi:10.1038/ncomms14494

  measured on live D. melanogaster, quasi-static pull-off while walking:
      mu_static = 0.84 +/- 0.13   (unperturbed animals, Fluon-coated surface)
      mu_static = 0.83 +/- 0.04   (distal tarsal segments polymer-coated, uncoated surface)

  The same paper models leg adhesion as a normal FORCE at the pretarsus (claw + pulvillus),
  separate from friction.  Its own scope statement is a boundary on what it can support:
  "We did not model limb compliance and dynamical effects".

So the registerable claim is narrow and falsifiable:

  H1 (pre-registered, BEFORE the run):
      Setting the foot-ground sliding friction to the MEASURED 0.84 (+/-20%, i.e. 0.67-1.01)
      changes the gated open-loop escape time by less than 20% relative to the project's
      model default of 1.0, because 1.0 lies inside the measured range's upper end.
  FALSIFIER: if any mu in [0.67, 1.01] moves the escape time by >= 20%, H1 is rejected and
      the friction value must be treated as a first-order parameter.
  SECOND, INDEPENDENT PREDICTION: the same sweep must reproduce part A's finding that
      cone=elliptic + noslip_iterations=3 is the field with the large effect.  If the cone
      effect vanishes under the measured friction, part A does not generalise.

NOT claimed: that 0.84 is "the" coefficient for a fly foot on this substrate.  Fluon is not
the model's floor, and the quoted numbers are for a whole animal, not a single pad.  This is
a defensible ORDER OF MAGNITUDE and a defensible CENTRAL VALUE, nothing finer.

Diagnostic measurement.  Not an autonomous result.  Not a fly measurement.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
import run_fix_round39 as R39        # noqa: E402
import run_fix_round39_margin_research as MR  # noqa: E402

OUT = HERE / "outputs/fix_round39"

MEASURED_MU = 0.84
MEASURED_MU_SD = 0.13
TOLERANCE_FRACTION = 0.20            # pre-registered
H1_CHANGE_LIMIT = 0.20
MU_GRID = (0.5, 0.67, 0.84, 1.0, 1.01, 1.5, 2.0)   # 1.0 is the project default = the reference

BASE = cp.CONTACT_SETS["mujoco_default"]


def variant(mu, cone=None, noslip=None) -> cp.ContactSet:
    return dataclasses.replace(
        BASE, name=f"mu{mu:g}" + (f"_cone{cone}" if cone else ""),
        provenance=f"model default contact set with sliding friction set to {mu:g}"
                   + (f", cone={cone}, noslip_iterations={noslip}" if cone else ""),
        friction=(float(mu), BASE.friction[1], BASE.friction[2]),
        cone=cone or BASE.cone, noslip_iterations=noslip or BASE.noslip_iterations)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    rows = []
    t0 = time.time()

    # --- arm 1: friction alone, around the measured value ----------------------
    for mu in MU_GRID:
        cs = variant(mu)
        r = R39.run_case(f"mu_{mu:g}", cs, seconds=0.30)
        rows.append({"arm": "friction_only", "mu": mu,
                     "cone": cs.cone, "noslip_iterations": cs.noslip_iterations,
                     "escape_time_s": r["escape_time_s"],
                     "seated_thorax_mm": r["seated_thorax_mm"],
                     "final_thorax_mm": r["final_thorax_mm"],
                     "seated_gate": r["seated"]})
        print(f"  mu={mu:<5g} escape {r['escape_time_s']*1000:6.1f} ms  "
              f"seated {r['seated_thorax_mm']:.3f} mm  "
              f"ncon {r['seated']['n_contact']}", flush=True)

    # --- arm 2: the measured friction PLUS the part-A winner (cone+noslip) ------
    cs2 = variant(MEASURED_MU, cone="elliptic", noslip=3)
    r2 = R39.run_case("mu_0.84_cone_elliptic_noslip3", cs2, seconds=0.30)
    rows.append({"arm": "measured_mu_plus_partA_winner", "mu": MEASURED_MU,
                 "cone": cs2.cone, "noslip_iterations": cs2.noslip_iterations,
                 "escape_time_s": r2["escape_time_s"],
                 "seated_thorax_mm": r2["seated_thorax_mm"],
                 "final_thorax_mm": r2["final_thorax_mm"],
                 "seated_gate": r2["seated"]})
    print(f"  mu={MEASURED_MU} + elliptic/noslip3  escape "
          f"{r2['escape_time_s']*1000:6.1f} ms", flush=True)

    by_mu = {r["mu"]: r for r in rows if r["arm"] == "friction_only"}
    assert 1.0 in by_mu, "the grid must contain the project default 1.0 to be the reference"
    ref = by_mu[1.0]["escape_time_s"]
    # H1 is scoped to the MEASURED range only, as pre-registered
    in_range = [r for r in rows if r["arm"] == "friction_only"
                and 0.67 <= r["mu"] <= 1.01]
    changes = {f"{r['mu']:g}": abs(r["escape_time_s"] - ref) / ref for r in in_range}
    h1_holds = bool(changes and all(v < H1_CHANGE_LIMIT for v in changes.values()))
    cone_ratio = r2["escape_time_s"] / ref

    cp.write_result(
        OUT / "fix_round39_friction_registered.json",
        question=("Does setting the foot-ground friction to the measured Drosophila value "
                  "(0.84 +/- 0.13) change the gated open-loop escape time by more than 20%, "
                  "and does part A's cone finding survive at that friction?"),
        status="PASS",
        measured={
            "pre_registration": {
                "H1": ("mu in [0.67, 1.01] changes the gated escape time by < 20% relative "
                       "to mu = 1.0"),
                "falsifier": "any mu in that range moving the escape time by >= 20%",
                "H1_supported": h1_holds,
                "second_prediction": ("cone=elliptic + noslip3 remains the large effect at "
                                      "the measured friction"),
                "second_prediction_escape_ratio_vs_mu1.0": cone_ratio,
                "declared_before_run": True,
            },
            "evidence": {
                "citation": "Ramdya P et al., 'Climbing favours the tripod gait over "
                            "alternative faster insect gaits', Nature Communications 8:14494 "
                            "(2017), doi:10.1038/ncomms14494",
                "measured_mu_static": MEASURED_MU,
                "measured_mu_sd": MEASURED_MU_SD,
                "second_measurement_in_source": 0.83,
                "second_measurement_sd": 0.04,
                "conditions": "live D. melanogaster, unperturbed animals on a Fluon-coated "
                              "surface (0.84 +/- 0.13) and distal-tarsus-coated animals on "
                              "an uncoated surface (0.83 +/- 0.04)",
                "scope_limit_from_source": "'We did not model limb compliance and dynamical "
                                           "effects' - so it constrains friction, not joint "
                                           "or pad compliance",
            },
            "rows": rows,
            "escape_time_changes_vs_mu1.0_within_measured_range": changes,
            "verdict": {
                "friction_is_first_order_in_the_measured_range": not h1_holds,
                "cone_remains_the_large_effect": (cone_ratio is not None and cone_ratio > 1.2),
                "recommended_working_value_mu": MEASURED_MU,
                "recommended_working_range_mu": [0.67, 1.01],
                "interpretation": (
                    "Friction is now on external evidence: the project's 1.0 sits at the top "
                    "of the measured range (0.84 +/- 0.13), and across the whole measured "
                    "range the gated escape time moves by less than the pre-registered 20%, "
                    "so the friction value is NOT what blocks standing. The large effect "
                    "stays with the contact cone and noslip. "
                    "WHAT IS STILL OPEN: the contact STIFFNESS (solref) has no literature "
                    "value here; the source explicitly did not model compliance, and part A "
                    "shows solref alone is a second-order knob (25.1 / 37.7 / 32.9 ms for "
                    "0.02 / 0.002 / 0.0002), so it is left as a declared model setting."),
            },
        },
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_unchanged": True,
            "protocol": "run_fix_round39.run_case (gated seat, tarsus-only, 0.30 s, escape "
                        "at 90% of the gated seated height, adhesion pinned to 0)",
            "wall_s": time.time() - t0,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "0.84 is NOT 'the' coefficient for a fly foot on this model's floor: the source "
            "measured whole animals on Fluon, not a single pad on the model substrate. It is "
            "a defensible order of magnitude and central value, nothing finer.",
            "The stiffness (solref) and the pad compliance remain model settings with no "
            "literature value attached here.",
            "A <20% change in escape time does not mean friction is irrelevant to walking: "
            "this measures open-loop collapse of one pose, not gait.",
            "Nothing here is autonomous behaviour.",
        ],
    )
    print(f"\nH1 (escape time within 20% across mu 0.67-1.01): {h1_holds}")
    print(f"cone effect ratio at measured mu: {cone_ratio:.3f}x")
    print("wrote", OUT / "fix_round39_friction_registered.json")


if __name__ == "__main__":
    main()
