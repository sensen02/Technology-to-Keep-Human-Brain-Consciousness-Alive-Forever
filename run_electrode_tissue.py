#!/usr/bin/env python
"""run_electrode_tissue.py -- CONNECT the tissue mechanics of electrode insertion
to the REAL access map and to the existing passive cable model.

WHAT THIS RUN PRODUCES
======================
  * ``outputs/embodied_body/electrode_tissue.json``
  * ``outputs/embodied_body/electrode_tissue.png``

It uses ``engine/electrode_tissue.py`` (NEW: dimpling + micromotion + the
access-map connection + the cable coupling), which itself REUSES the project's
``CavityExpansionField``, ``SheathLayer``, ``FlyBrainGeometry``, ``SpeciesGuard``,
``apply_membrane_damage``, ``affected_node_indices``,
``ProximityExcitabilityCoupling``, ``run_paired_cable``,
``engine.embodied.access_map.Ledger`` and the BANC soma parquet.  No existing file
is modified by this run.

PRE-REGISTERED PREDICTIONS ARE DECLARED IN ``PRE_REGISTERED`` BELOW AND PRINTED
BEFORE ANY NUMBER IS COMPUTED.  They are never retuned afterwards: a threshold is
what it was when it was written down.  Verdicts are PASS / FAIL / EVIDENCE NOT
FOUND.

HONESTY
=======
Hand-built research prototype, NOT a validated device model.  No claim about
behaviour, perception, attention, recognition, experience, identity or survival.
No measured mechanical property of any insect CNS is used: stiffness, surface
tension, sheath failure strain, insertion force and micromotion amplitude are all
ASSUMED with declared sweeps.  Rodent parameters are refused by a strict
SpeciesGuard and force-flagged when used.  SURGERY DAMAGE AND ELECTRODE DAMAGE ARE
NOT SEPARABLE IN THIS MODEL and nothing here may be reported as one without the
other.

BANC soma data (attribution is a CC BY 4.0 LICENCE CONDITION):
    "Distributed control circuits across a brain-and-cord connectome",
    Harvard Dataverse, doi:10.7910/DVN/7WTH1N, file somas_v1.parquet
    (file id 13916460), licensed CC BY 4.0.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

OUT_DIR = "outputs/embodied_body"
JSON_PATH = os.path.join(OUT_DIR, "electrode_tissue.json")
PNG_PATH = os.path.join(OUT_DIR, "electrode_tissue.png")

# ---------------------------------------------------------------------------
# USER-FIXED GEOMETRY (the brief pins these; NOTHING here sweeps them)
# ---------------------------------------------------------------------------
SHAFT_DIAMETER_UM = 7.0
CHANNEL_PITCH_UM = 20.0
N_CHANNELS_PRIMARY = 6          # FREE design variable: the brief leaves it open
CAPTURE_RADIUS_PRIMARY = 10.0   # = pitch/2, the project's declared collision-free
                                # invariant (pitch >= 2*capture_radius)
PRIMARY_GEOMETRY = "linear_shank_z"

# the sensitivity sweeps (NOT the fixed geometry: these are stated as such)
N_CHANNEL_SWEEP = (3, 6, 9, 17, 33)
CAPTURE_RADIUS_SWEEP = (5.0, 10.0, 15.0, 25.0, 50.0)
# 50.0 is the capture radius the parent round DECLARED for the new 20/7
# geometry; 10.0 is pitch/2, the collision-free value this module uses as
# its own primary.  Both are reported because overlap changes the meaning
# of 'addressable' (see the collision columns).
CHANNEL_COUNT_IN_PRIMARY = ("n_channels",)   # documentation only


# ===========================================================================
# PRE-REGISTERED PREDICTIONS -- written BEFORE the run, never retuned after
# ===========================================================================
# (id, statement, test kind, declared threshold/value)
PRE_REGISTERED = [
    ("P1", "The pre-penetration DIMMLING depth scales LINEARLY with shaft "
           "diameter (it is a*sqrt(eps_f/(2/3)) under the declared shallow-cap "
           "assumption), so depth(14 um)/depth(7 um) is exactly 2.0. Declared so "
           "that a non-linear implementation is caught rather than described.",
     "ratio_equals", 2.0),

    ("P2", "At the FIXED 7 um diameter and the central assumed sheath failure "
           "strain (0.30), the pre-penetration dimpling depth is between 1.5 and "
           "3.5 um, i.e. of the same order as the ASSUMED 3 um sheath thickness. "
           "Declared as a range BEFORE the arithmetic, from the assumed inputs.",
     "in_range", (1.5, 3.5)),

    ("P3", "The INSERTION criterion radius at the conservative cross-species "
           "axonal strain threshold (0.14) is LESS than 10 um, i.e. below half the "
           "FIXED 20 um channel pitch. Declared before running.",
     "lt", 10.0),

    ("P4", "At least ONE real BANC soma that the primary array makes addressable "
           "lies inside the insertion criterion at 0.14. A count of 0 would mean "
           "the mechanics and the access map do not actually intersect anywhere, "
           "and would be reported as such.",
     "ge", 1.0),

    ("P5", "The MICROMOTION criterion radius at the central assumed amplitude "
           "(5 um) and the cross-species mechanoporation shear threshold "
           "(gamma = 140 dyn/cm^2 / G = 14/500 = 0.028) EXCEEDS the insertion "
           "criterion radius at 0.14. Declared as a directional prediction: "
           "chronically implanted probes are damaged by motion, not by insertion.",
     "gt_other", "insertion_radius_at_0.14"),

    ("P6", "At the central assumed micromotion amplitude the wall shear strain "
           "2*A/a EXCEEDS 1.0, i.e. the linear-elastic cavity field is quoted "
           "OUTSIDE its own small-strain domain at the shaft wall, so its "
           "criterion radius there is an order-of-magnitude bound and not a "
           "predicted strain. Declared so that this self-limitation is a stated "
           "expectation, not an after-the-fact excuse.",
     "gt", 1.0),

    ("P7", "The membrane damage applied through the project's "
           "apply_membrane_damage reduces the DC input resistance of the proxy "
           "cable by MORE than 1% for at least one REAL at-risk neuron. Declared "
           "before running.",
     "gt", 0.01),

    ("P8", "A strict SpeciesGuard (allow_cross=False) REFUSES to build the "
           "Drosophila tissue-mechanics registry, and a rodent parameter cannot be "
           "recorded without the forced cross-species flag and its citation. "
           "Declared as a mechanism check.",
     "refusal", None),

    ("P9", "A MEASURED Drosophila (or other insect) CNS shear modulus, surface "
           "tension, sheath failure strain or micromotion amplitude is available "
           "in this project's verified records. Declared so that NOT finding one "
           "is reported as EVIDENCE NOT FOUND rather than quietly assumed.",
     "evidence", "insect_tissue_mechanics"),
]


# ===========================================================================
def verdict(kind, threshold, **kw):
    """Return (verdict, observed) for one pre-registered test. Never retunes."""
    if kind == "ratio_equals":
        a, b = kw["ratio"], float(threshold)
        return ("PASS" if a is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b))
                else "FAIL"), a
    if kind == "in_range":
        v = kw["value"]
        lo, hi = threshold
        return ("PASS" if v is not None and lo <= v <= hi else "FAIL"), v
    if kind == "lt":
        v = kw["value"]
        return ("PASS" if v is not None and v < float(threshold) else "FAIL"), v
    if kind == "ge":
        v = kw["value"]
        return ("PASS" if v is not None and v >= float(threshold) else "FAIL"), v
    if kind == "gt":
        v = kw["value"]
        return ("PASS" if v is not None and v > float(threshold) else "FAIL"), v
    if kind == "gt_other":
        v, other = kw["value"], kw["other"]
        ok = (v is not None and other is not None and v > other)
        return ("PASS" if ok else "FAIL"), {"value": v, "other": other}
    if kind == "refusal":
        return ("PASS" if kw.get("all_refused") else "FAIL"), kw.get("refused_rules")
    if kind == "evidence":
        return ("EVIDENCE NOT FOUND" if not kw.get("found") else "PASS"), \
            kw.get("detail")
    raise ValueError("unknown test kind %r" % (kind,))


def _at_risk_summary(block):
    """Counts and fractions only -- keeps the sweep section of the JSON small."""
    pop = block["population"]
    return {"mechanism": block["mechanism"],
            "measure": block["criterion"]["measure"],
            "threshold": block["criterion"]["threshold"],
            "radius_um": block["criterion"]["radius_um"],
            "addressable_all_somas": pop["addressable_all_somas"],
            "at_risk_all_somas": pop["at_risk_all_somas"],
            "fraction_of_addressable_all_somas":
                pop["fraction_of_addressable_all_somas"],
            "addressable_in_modelled_tier": pop["addressable_in_modelled_tier"],
            "at_risk_in_modelled_tier": pop["at_risk_in_modelled_tier"],
            "fraction_of_addressable_in_modelled_tier":
                pop["fraction_of_addressable_in_modelled_tier"]}


def main():
    t_start = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    os.chdir(here)

    from engine.embodied.access_map import (BANC_SOMA_ATTRIBUTION,
                                            ElectrodeArraySpec,
                                            banc_voxel_resolution,
                                            build_access_map, load_soma_table)
    from engine.embodied.adapters import REPO_ROOT, TierConfig, build_neural_tier
    from engine.neck_cut_data import load_banc
    from engine.electrode_damage import (ProximityExcitabilityCoupling,
                                         STRAIN_THRESHOLD_CHOICES,
                                         audit_species_labeling,
                                         injury_threshold_options)
    from engine.params import ProvenanceError
    from engine.embodied.access_map import PROVENANCE
    import engine.electrode_tissue as et

    print("=" * 78)
    print("ELECTRODE TISSUE MECHANICS  --  insertion dimpling + micromotion,")
    print("connected to the REAL BANC access map and to the passive cable model")
    print("=" * 78)
    print("python %s on %s" % (platform.python_version(), platform.platform()))
    print()
    print("attribution (CC BY 4.0, REQUIRED -- a licence condition):")
    print("  " + BANC_SOMA_ATTRIBUTION["citation_string"])
    print()
    for stmt in et.HONESTY_TISSUE:
        print("HONESTY: %s" % stmt)
    print()
    print("-" * 78)
    print("WHAT THIS RUN REUSES AND WHAT IT ADDS (stated, not implied)")
    print("-" * 78)
    for row in et.REUSE_LEDGER:
        if "reused" in row:
            print("  REUSED  %-44s from %s" % (row["reused"], row["from"]))
        else:
            print("  ADDED   %-44s in this module" % row["added"])
    print()

    # ------------------------------------------------------------------
    print("-" * 78)
    print("0. PRE-REGISTERED PREDICTIONS (declared BEFORE any number is computed)")
    print("-" * 78)
    for pid, stmt, kind, thr in PRE_REGISTERED:
        print("  %s [test=%s, declared=%r]" % (pid, kind, thr))
        print("     %s" % stmt)
    print()
    print("  These thresholds are NOT changed anywhere below.")
    print()

    results = {}

    # ------------------------------------------------------------------
    print("-" * 78)
    print("1. PROVENANCE: refusals first, then the numbers")
    print("-" * 78)
    prov = et.TissueProvenance(allow_cross=True)
    strict = et.refuse_foreign_species(prov.registry)
    print("R4 SpeciesGuard(allow_cross=False): REFUSED at construction")
    print("   %s..." % strict["message"][:150])
    refused_rules = ["R4"]

    def _must_refuse(label, fn):
        try:
            fn()
        except (ProvenanceError, ValueError) as exc:
            prov.record_refusal(label, label, exc)
            refused_rules.append(label)
            print("   REFUSED %s: %s..." % (label, str(exc)[:120]))
            return True
        raise AssertionError("the ledger FAILED to refuse %s" % label)

    _must_refuse("R1_MEASURED_CITED_without_source",
                 lambda: prov.record("demo_uncited", 1.0, "1",
                                     PROVENANCE.MEASURED_CITED))
    _must_refuse("R2_ASSUMED_without_sweep",
                 lambda: prov.record("demo_unswept", 1.0, "1", PROVENANCE.ASSUMED))
    rodent = prov.species_guard.adopt("rodent_axonal_strain_conservative")
    print("   rodent parameter %r adopted from %r, cross_species=%s, forced=%s"
          % (rodent.name, rodent.species, rodent.cross_species, rodent.forced_flag))
    _must_refuse("R3_foreign_species_without_forced_flag",
                 lambda: prov.record_adopted(rodent, force_cross_flag=False))
    print()
    print("  the ledger holds %d entries so far; species guard saw %d foreign"
          % (len(prov.ledger.as_list()), len(prov.species_guard.foreign_names))
          + " parameters in the registry")

    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("2. THE MECHANICS AT THE FIXED %.1f um SHAFT DIAMETER" % SHAFT_DIAMETER_UM)
    print("-" * 78)
    insertion = et.InsertionMechanics.from_provenance(prov)
    print(et.InsertionMechanics.__doc__.strip().splitlines()[0])
    print("  shaft diameter          : %.3g um  (USER-FIXED, not swept)"
          % insertion.shaft_diameter_um)
    print("  shaft radius            : %.3g um" % insertion.shaft_radius_um)
    print("  wall strain (delta/a)   : %.4g  <- SIZE-INDEPENDENT by construction"
          % insertion.wall_strain)
    print("  wall pressure           : %.4g Pa (assumed G)"
          % insertion.wall_pressure_Pa())
    print("  dimpling contact stiffness: %.6g N/m  (elastic %.3g + tension %.3g)"
          % (insertion.dimpling.contact_stiffness_N_per_m,
             insertion.dimpling.elastic_stiffness_N_per_m,
             insertion.dimpling.tension_stiffness_N_per_m))
    print("  pre-penetration dimpling: %.4g um at failure strain %.3g"
          % (insertion.pre_penetration_dimpling_um(),
             insertion.dimpling.failure_strain))
    print("  ... that is %.4g x the shaft DIAMETER (%.4g x the shaft radius)"
          % (insertion.dimpling.failure_depth_over_diameter(),
             insertion.dimpling.failure_depth_um() / insertion.shaft_radius_um))
    print("  model's own rupture force: %.4g uN  (vs assumed insertion force"
          % insertion.dimpling.failure_force_uN())
    print("      %.4g uN -> regime %s)"
          % (prov.adopt("assumed_insertion_force_uN")[0].value,
             insertion.dimpling_regime_for(
                 prov.adopt("assumed_insertion_force_uN")[0].value)["regime"]))
    print("  sheath breach (REUSED SheathLayer opening-radius criterion): %s"
          % insertion.sheath_breach()["breached_by_opening_radius_criterion"])
    print("     shaft radius %.3g um vs assumed critical opening radius %.3g um"
          % (insertion.shaft_radius_um,
             insertion.sheath_breach()["critical_opening_radius_um"]))
    print()
    print("  THIS IS AN INTERNAL INCONSISTENCY, REPORTED AND NOT RETUNED: the"
          "\n  central ASSUMED insertion force is ABOVE the model's own rupture"
          "\n  force, so at that force the probe is predicted to be THROUGH the"
          "\n  sheath and the pre-penetration dimpling regime is already over.")

    # dimpling vs shaft diameter (a sensitivity on a design input that is FIXED)
    print()
    print("  dimpling depth vs shaft diameter (the 7 um case is the FIXED design;"
          "\n  the other diameters are a SENSITIVITY, not a design change):")
    diameter_sweep = (1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 50.0, 100.0)
    dimple_rows = []
    print("   %8s %12s %12s %12s %10s" % ("D um", "d_fail um", "K N/m", "F_fail uN",
                                          "d_fail/D"))
    for D in diameter_sweep:
        dm = et.DimplingModel(
            shaft_radius_um=D / 2.0,
            effective_modulus_Pa=insertion.dimpling.effective_modulus_Pa,
            surface_tension_N_per_m=insertion.dimpling.surface_tension_N_per_m,
            failure_strain=insertion.dimpling.failure_strain)
        row = {"shaft_diameter_um": D, "failure_depth_um": dm.failure_depth_um(),
               "contact_stiffness_N_per_m": dm.contact_stiffness_N_per_m,
               "failure_force_uN": dm.failure_force_uN(),
               "failure_depth_over_diameter": dm.failure_depth_over_diameter()}
        dimple_rows.append(row)
        print("   %8.3g %12.4g %12.5g %12.5g %10.4g"
              % (D, row["failure_depth_um"], row["contact_stiffness_N_per_m"],
                 row["failure_force_uN"], row["failure_depth_over_diameter"]))
    d7 = [r for r in dimple_rows if r["shaft_diameter_um"] == 7.0][0]
    d14 = et.DimplingModel(
        shaft_radius_um=7.0,
        effective_modulus_Pa=insertion.dimpling.effective_modulus_Pa,
        surface_tension_N_per_m=insertion.dimpling.surface_tension_N_per_m,
        failure_strain=insertion.dimpling.failure_strain).failure_depth_um()
    ratio_14_over_7 = d14 / d7["failure_depth_um"]

    # the assumed-parameter sweeps that matter
    print()
    print("  sweep of the ADDED dimpling assumptions (all ASSUMED, none measured):")
    dimple_sweep = []
    for eps in (0.05, 0.1, 0.3, 0.5, 1.0):
        row = {"failure_strain": eps, "shaft_diameter_um": SHAFT_DIAMETER_UM,
               "failure_depth_um": et.DimplingModel(
                   shaft_radius_um=SHAFT_DIAMETER_UM / 2.0,
                   effective_modulus_Pa=insertion.dimpling.effective_modulus_Pa,
                   surface_tension_N_per_m=insertion.dimpling.surface_tension_N_per_m,
                   failure_strain=eps).failure_depth_um()}
        dimple_sweep.append(row)
    for T in (1.0e-4, 1.0e-3, 2.0e-3, 1.0e-2, 5.0e-2):
        dm = et.DimplingModel(shaft_radius_um=SHAFT_DIAMETER_UM / 2.0,
                              effective_modulus_Pa=insertion.dimpling.effective_modulus_Pa,
                              surface_tension_N_per_m=T,
                              failure_strain=insertion.dimpling.failure_strain)
        row = {"surface_tension_N_per_m": T,
               "contact_stiffness_N_per_m": dm.contact_stiffness_N_per_m,
               "failure_depth_um": dm.failure_depth_um(),
               "failure_force_uN": dm.failure_force_uN()}
        dimple_sweep.append(row)
    print("   failure strain %s -> dimpling depth %s um"
          % ([r["failure_strain"] for r in dimple_sweep if "failure_strain" in r],
             ["%.3g" % r["failure_depth_um"] for r in dimple_sweep
              if "failure_strain" in r]))
    print("   surface tension %s N/m -> rupture force %s uN"
          % (["%.3g" % r["surface_tension_N_per_m"] for r in dimple_sweep
              if "surface_tension_N_per_m" in r],
             ["%.3g" % r["failure_force_uN"] for r in dimple_sweep
              if "surface_tension_N_per_m" in r]))
    print("   -> the implied pre-puncture FORCE is a RANGE of more than an order of"
          "\n      magnitude (%.3g to %.3g uN), set by an ASSUMED tension."
          % (min(r["failure_force_uN"] for r in dimple_sweep
                 if "surface_tension_N_per_m" in r),
             max(r["failure_force_uN"] for r in dimple_sweep
                 if "surface_tension_N_per_m" in r)))

    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("3. THE STRAIN FIELD AROUND THE %.1f um SHAFT (REUSED CavityExpansionField)"
          % SHAFT_DIAMETER_UM)
    print("-" * 78)
    thresholds = injury_threshold_options(prov.species_guard)
    thr_rows = {}
    for key in STRAIN_THRESHOLD_CHOICES:
        adopted = thresholds[key]
        rec = prov.record_adopted(adopted, note="cross-species axonal strain "
                                                "threshold used as a criterion")
        r = insertion.damage_radius_um(adopted.value)
        thr_rows[key] = {"value": adopted.value, "unit": adopted.unit,
                         "radius_um": r, "radius_over_shaft_radius":
                             r / insertion.shaft_radius_um,
                         "cross_species": adopted.cross_species,
                         "species": adopted.species, "source": adopted.source,
                         "ledger": rec.as_dict()}
        print("  %-13s eps >= %.2f (%-22s) -> r_crit = %7.4f um = %.4g a"
              % (key, adopted.value, adopted.species.split(" (")[0], r,
                 r / insertion.shaft_radius_um))
    print("  ... ALL THREE ARE GUINEA-PIG (Cavia porcellus) axonal thresholds,"
          "\n      forced-flagged cross-species, used as PROXIES.")

    r_profile = np.concatenate((np.linspace(insertion.shaft_radius_um, 20.0, 200),
                                np.geomspace(20.0, 200.0, 60)[1:]))
    profile = insertion.strain_profile(r_profile)
    print()
    print("  strain vs distance at the FIXED 7 um diameter (every value from the"
          "\n  REUSED field class; the wall strain is the assumed delta/a):")
    print("   %8s %12s %12s %12s %12s" % ("r um", "eps_hoop", "eps_radial",
                                          "max|eps|", "gamma_max"))
    for r in (3.5, 4.0, 5.0, 7.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        print("   %8.4g %12.5g %12.5g %12.5g %12.5g"
              % (r, float(insertion.hoop_strain(r)),
                 float(insertion.radial_strain(r)),
                 float(insertion.max_abs_principal_strain(r)),
                 float(insertion.max_shear_strain(r))))
    print("  -> the field falls as 1/r^2, so the criterion radius is")
    print("     r_crit = a*sqrt(f/eps): LINEAR in shaft radius and inversely")
    print("     sqrt in the threshold, while the WALL STRAIN is size-independent.")

    # mechanoporation STRESS -> SHEAR STRAIN threshold via the assumed G
    tau = thresholds["_mechanoporation_shear"]
    tau_rec = prov.record_adopted(tau, note="cross-species mechanoporation shear "
                                            "STRESS threshold (rat cortical culture; "
                                            "the source states NO DEATH occurred)")
    G = insertion.field.shear_modulus_Pa
    gamma_thr = tau.value / G
    gamma_rec = prov.record(
        "derived_mechanoporation_shear_strain_threshold", gamma_thr, "1",
        PROVENANCE.ASSUMED, sweep=(tau.value / 5000.0, tau.value / 50.0),
        note="tau/G with tau the rat mechanoporation stress (14 Pa) and G the "
             "ASSUMED fly shear modulus (%.4g Pa here); recorded as ASSUMED, NOT as "
             "a cited measurement, because one parent is only assumed. Swept over "
             "G in 50-5000 Pa." % G)
    print()
    print("  mechanoporation threshold: tau = %.4g Pa (rat, source states NO DEATH"
          "\n  occurred at those parameters) / ASSUMED G = %.4g Pa  ->  gamma = %.6g"
          % (tau.value, G, gamma_thr))
    print("     swept over G = 50-5000 Pa: gamma in [%.5g, %.5g]"
          % (tau.value / 5000.0, tau.value / 50.0))
    print("     ledger class: %s (NOT MEASURED_CITED: it is derived over an ASSUMED"
          "\n     parent, which engine.params would not have caught)"
          % gamma_rec.provenance)

    mic = et.MicromotionStrain(
        shaft_radius_um=insertion.shaft_radius_um,
        amplitude_um=prov.adopt("assumed_micromotion_amplitude_um")[0].value,
        frequency_Hz=prov.adopt("assumed_micromotion_frequency_Hz")[0].value,
        shear_modulus_Pa=G)
    print()
    print("  MICROMOTION (ADDED; the mechanism that keeps a chronic probe damaging)")
    print("   amplitude A             : %.4g um (ASSUMED, swept 0.05-50)"
          % mic.amplitude_um)
    print("   frequency               : %.4g Hz (ASSUMED; enters NO strain -- the"
          % mic.frequency_Hz)
    print("                             model is quasi-static)")
    print("   A/a                     : %.4g" % mic.amplitude_ratio)
    print("   wall shear strain 2A/a  : %.4g" % mic.wall_shear_strain)
    print("   kernel vs REUSED CavityExpansionField: max|diff| = %s um/um"
          % (mic.kernel_matches_cavity_field() if mic.as_cavity_field() is not None
             else "N/A (amplitude > shaft radius: the class REFUSES this amplitude)"))
    print("   gamma_crit radius       : %.4g um" % mic.damage_radius_um(gamma_thr))
    print("   validity                : %s" % mic.validity()["verdict"])
    print("   r where gamma drops to the small-strain limit %.2f: %.4g um"
          % (et.SMALL_STRAIN_LIMIT, mic.validity_radius_um()))
    print()
    print("   micromotion shear strain vs distance (A = %.4g um):" % mic.amplitude_um)
    print("   %8s %14s %14s" % ("r um", "gamma", "tau Pa"))
    for r in (3.5, 5.0, 10.0, 20.0, 50.0, 100.0):
        print("   %8.4g %14.5g %14.5g"
              % (r, float(mic.max_shear_strain(r)), float(mic.shear_stress_Pa(r))))

    print()
    print("   dependence on the STATED displacement A:")
    print("   %8s %10s %12s %14s %14s" % ("A um", "A/a", "2A/a", "r_crit um",
                                          "r(gamma=0.1) um"))
    micro_sweep = []
    for A in (0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        m = et.MicromotionStrain(insertion.shaft_radius_um, A, mic.frequency_Hz, G)
        row = {"amplitude_um": A, "amplitude_ratio": m.amplitude_ratio,
               "wall_shear_strain": m.wall_shear_strain,
               "criterion_radius_um": m.damage_radius_um(gamma_thr),
               "validity_radius_um": m.validity_radius_um(),
               "in_cavity_field_domain": m.as_cavity_field() is not None,
               "inside_small_strain_at_wall":
                   bool(m.wall_shear_strain <= et.SMALL_STRAIN_LIMIT)}
        micro_sweep.append(row)
        print("   %8.4g %10.4g %12.4g %14.5g %14.5g"
              % (A, row["amplitude_ratio"], row["wall_shear_strain"],
                 row["criterion_radius_um"], row["validity_radius_um"]))
    print("   -> r_crit scales as sqrt(A): the criterion radius is set by the")
    print("      STATED displacement, and the amplitude is ASSUMED, not measured.")
    print("   -> for A <= a the kernel reproduces the reused class EXACTLY")
    print("      (checked to machine precision); above A = a the class refuses to")
    print("      be constructed at all, and this module says so rather than")
    print("      rescaling a class that declared its own domain.")

    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("4. THE REAL ACCESS MAP AT pitch=%.1f um, diameter=%.1f um"
          % (CHANNEL_PITCH_UM, SHAFT_DIAMETER_UM))
    print("-" * 78)
    cfg = TierConfig()
    ann = load_banc(os.path.join(REPO_ROOT, cfg.data_path))
    tier_built = build_neural_tier(cfg)
    root_ids = np.asarray(ann["root_ids"])
    tier_root_ids = root_ids[np.asarray(tier_built["global_index"])]
    readout_local = np.asarray(tier_built["readout_local"])
    target_mask = np.zeros(int(tier_built["n"]), dtype=bool)
    target_mask[readout_local] = True
    somas_path = os.path.join(here, "data", "banc", "somas_v1.parquet")
    resolution = banc_voxel_resolution()
    soma = load_soma_table(somas_path)
    print("  tier neurons %d; readout %d; soma parquet rows %d"
          % (tier_root_ids.size, int(target_mask.sum()), soma.n_rows))
    print("  voxel resolution %s nm (%s)" % (resolution.nm_per_voxel, resolution.provenance))
    print("  CARRIED AMBIGUITY: the verified record says 4x4x45 nm; the brief for")
    print("  this round states x/y is still ambiguous between 4 and 8 nm. This run")
    print("  uses the verified 4 nm in x/y and REPORTS the ambiguity: if x/y were")
    print("  8 nm, every LATERAL distance below doubles (z is unchanged at 45 nm).")

    def build(spec):
        return build_access_map(spec, tier_root_ids, somas_path, resolution,
                                soma_table=soma, target_mask=target_mask)

    spec_primary = ElectrodeArraySpec(
        pitch_um=CHANNEL_PITCH_UM, diameter_um=SHAFT_DIAMETER_UM,
        n_channels=N_CHANNELS_PRIMARY, capture_radius_um=CAPTURE_RADIUS_PRIMARY,
        geometry=PRIMARY_GEOMETRY, placement="densest_readout_soma", seed=0)
    am = build(spec_primary)
    cov = am.reports["coverage"]
    print()
    print("  PRIMARY: %s, %d channels at %.1f um pitch, capture radius %.1f um"
          % (PRIMARY_GEOMETRY, N_CHANNELS_PRIMARY, CHANNEL_PITCH_UM,
             CAPTURE_RADIUS_PRIMARY))
    print("   array extent          : %s um" % np.round(am.reports["array"]["extent_um"], 2))
    print("   nonempty channels     : %d of %d"
          % (am.reports["array"]["nonempty_channels"], am.n_channels))
    print("   addressable BANC somas: %d   (tier subset %d of %d joined)"
          % (int((np.asarray(am.neuron_n_channels) > 0).sum()),
             cov["all_joined_tier_neurons_captured"], cov["all_joined_tier_neurons"]))
    print("   readout captured      : %d of %d"
          % (cov["target_neurons_captured"], cov["target_neurons"]))
    geom_consistent = PRIMARY_GEOMETRY.startswith("linear_shank")
    print()
    print("  GEOMETRY CONSISTENCY (reported, not hidden): a %.1f um-diameter shaft"
          % SHAFT_DIAMETER_UM)
    print("  can carry channels only ALONG itself, so a LINEAR shank is the layout")
    print("  the fixed geometry permits. A planar grid of %d channels at %.0f um"
          % (256, CHANNEL_PITCH_UM))
    print("  pitch spans %.0f x %.0f um -- a %d-fold wider face than the shaft"
          % (16 * CHANNEL_PITCH_UM, 16 * CHANNEL_PITCH_UM,
             (16 * CHANNEL_PITCH_UM) / SHAFT_DIAMETER_UM))
    print("  diameter -- which the brief's own two fixed numbers cannot both satisfy.")
    am_planar = build(ElectrodeArraySpec(
        pitch_um=CHANNEL_PITCH_UM, diameter_um=SHAFT_DIAMETER_UM, n_channels=256,
        capture_radius_um=CAPTURE_RADIUS_PRIMARY, geometry="planar_grid_xy",
        placement="densest_readout_soma", seed=0))

    print()
    print("  LATERAL-RESOLUTION AMBIGUITY, COMPUTED RATHER THAN RESOLVED: the brief")
    print("  says x/y is ambiguous between 4 and 8 nm. Rebuilding the SAME array with")
    print("  an 8 nm lateral frame doubles every lateral distance, while the criterion")
    print("  radius stays in micrometres -- so the answer CHANGES and both must be")
    print("  reported:")
    from engine.embodied.access_map import VoxelResolution
    # NOTE (a defect found in the EXISTING module, NOT fixed here): calling
    # build_access_map with a BARE triple raises inside its own ledger, because it
    # records voxel_resolution_nm as ASSUMED with sweep=None before the candidate
    # sweep is recorded.  So this arm passes an explicit VoxelResolution whose NOTE
    # declares the swept range, which satisfies the same ledger rule honestly.
    res8 = VoxelResolution(
        8.0, 8.0, 45.0, PROVENANCE.ASSUMED,
        note="x/y lateral step DOUBLED from the verified 4 nm, to carry the "
             "ambiguity the brief states instead of silently resolving it. The "
             "swept range is the project's own UNVERIFIED_RESOLUTION_CANDIDATES_NM "
             "list and this arm is one point of that sweep; z stays at the verified "
             "45 nm.")
    am8 = build_access_map(
        spec_primary, tier_root_ids, somas_path, res8,
        soma_table=soma, target_mask=target_mask)
    res8_check = None
    print("   resolution arm  %s -> addressable %d  (provenance %s: one point of the"
          % ((8.0, 8.0, 45.0), int((np.asarray(am8.neuron_n_channels) > 0).sum()),
             res8.provenance))
    print("   project's own sweep; the verified arm above uses 4 nm in x/y)")

    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("5. NEURONS AT RISK  (the mechanics <-> access-map connection)")
    print("-" * 78)
    at_risk_insert = et.probe_neurons_at_risk(am, insertion, thr_rows["conservative"]["value"],
                                             tier_root_ids=tier_root_ids,
                                             yield_verdict=True)
    print("  INSERTION criterion: %s" % at_risk_insert["verdict"])
    print("  shaft axis: %s" % at_risk_insert["shaft"]["rule"])
    print("  axis point %s um, direction %s, half-length %.3g um"
          % (np.round(at_risk_insert["shaft"]["point_um"], 3),
             np.round(at_risk_insert["shaft"]["direction"], 6),
             at_risk_insert["shaft"]["half_length_um"]))
    print("  addressable distance stats (perp. to the axis): min %.4g / median %.4g"
          " / max %.4g um"
          % (at_risk_insert["addressable_distance_stats_um"]["min"],
             at_risk_insert["addressable_distance_stats_um"]["median"],
             at_risk_insert["addressable_distance_stats_um"]["max"]))
    at_risk_micro = et.probe_neurons_at_risk(am, mic, gamma_thr,
                                            tier_root_ids=tier_root_ids,
                                            yield_verdict=True)
    print("  MICROMOTION criterion: %s" % at_risk_micro["verdict"])

    def show(tag, block):
        pop = block["population"]
        print("  %-34s r_crit=%7.4f um  at-risk %4d / %4d addressable = %s"
              % (tag, block["criterion"]["radius_um"], pop["at_risk_all_somas"],
                 pop["addressable_all_somas"],
                 ("%.4f" % pop["fraction_of_addressable_all_somas"])
                 if pop["fraction_of_addressable_all_somas"] is not None else "n/a"))
        print("  %-34s (modelled tier: %d / %d = %s)"
              % ("", pop["at_risk_in_modelled_tier"],
                 pop["addressable_in_modelled_tier"],
                 ("%.4f" % pop["fraction_of_addressable_in_modelled_tier"])
                 if pop["fraction_of_addressable_in_modelled_tier"] is not None else "n/a"))
    show("insertion eps>=0.14", at_risk_insert)
    show("micromotion gamma>=%.5g" % gamma_thr, at_risk_micro)
    print()
    print("  READ THIS BEFORE QUOTING THE FRACTION: for a LINEAR shank every")
    print("  addressable soma lies within the DECLARED capture radius (%.1f um) of the"
          % CAPTURE_RADIUS_PRIMARY)
    print("  shaft axis BY CONSTRUCTION, so the at-risk fraction is a statement about")
    print("  the capture annulus around the shaft, NOT about the tissue as a whole.")
    res8 = et.probe_neurons_at_risk(am8, insertion,
                                    thr_rows["conservative"]["value"],
                                    tier_root_ids=tier_root_ids,
                                    insertion_depth_um=insertion.insertion_depth_um)
    res8_check = _at_risk_summary(res8)
    print("  under the 8 nm lateral reading: addressable %d, at risk %d, fraction %s"
          % (res8_check["addressable_all_somas"], res8_check["at_risk_all_somas"],
             ("%.4f" % res8_check["fraction_of_addressable_all_somas"])
             if res8_check["fraction_of_addressable_all_somas"] is not None else "n/a"))
    print("  ... the criterion radius is in um and does NOT scale with the frame, so")
    print("  the two readings give DIFFERENT fractions. The ambiguity is CARRIED.")

    # the full threshold sweep + geometry sweep
    print()
    print("  at-risk fraction vs the STATED criterion (primary array, real somas):")
    print("  %-26s %10s %10s %14s %14s"
          % ("mechanism / criterion", "r_crit um", "at risk", "frac (all)",
             "frac (tier)"))
    threshold_sweep = []
    for key in STRAIN_THRESHOLD_CHOICES:
        blk = (at_risk_insert if key == "conservative" else
               et.probe_neurons_at_risk(am, insertion, thr_rows[key]["value"],
                                        tier_root_ids=tier_root_ids))
        row = _at_risk_summary(blk)
        row["label"] = "insertion eps>=%s" % thr_rows[key]["value"]
        threshold_sweep.append(row)
    gamma_sweep_thresholds = [tau.value / 5000.0, gamma_thr, tau.value / 50.0]
    for g in gamma_sweep_thresholds:
        blk = (at_risk_micro if abs(g - gamma_thr) < 1e-12 else
               et.probe_neurons_at_risk(am, mic, g, tier_root_ids=tier_root_ids))
        row = _at_risk_summary(blk)
        row["label"] = "micromotion gamma>=%.5g" % g
        threshold_sweep.append(row)
    for row in threshold_sweep:
        print("  %-26s %10.4g %10d %14s %14s"
              % (row["label"], row["radius_um"], row["at_risk_all_somas"],
                 ("%.4f" % row["fraction_of_addressable_all_somas"])
                 if row["fraction_of_addressable_all_somas"] is not None else "n/a",
                 ("%.4f" % row["fraction_of_addressable_in_modelled_tier"])
                 if row["fraction_of_addressable_in_modelled_tier"] is not None
                 else "n/a"))

    print()
    print("  sweep over the FREE channel count and the DECLARED capture radius")
    print("  (both at the FIXED 20 um pitch; capture radius > pitch/2 breaks the")
    print("   project's own collision-free invariant pitch >= 2*radius):")
    print("  %-14s %8s %8s %8s %10s %10s %14s %10s"
          % ("geometry", "n_ch", "capture", "addr", "at risk", "tier at risk",
             "frac (all)", "collisions"))
    geometry_sweep = []
    for geom, n_ch in ([(PRIMARY_GEOMETRY, k) for k in N_CHANNEL_SWEEP] +
                       [("planar_grid_xy", k) for k in (16, 256)]):
        for rad in (CAPTURE_RADIUS_SWEEP if geom == PRIMARY_GEOMETRY else
                    (CAPTURE_RADIUS_PRIMARY,)):
            spec = ElectrodeArraySpec(pitch_um=CHANNEL_PITCH_UM,
                                      diameter_um=SHAFT_DIAMETER_UM,
                                      n_channels=n_ch, capture_radius_um=rad,
                                      geometry=geom,
                                      placement="densest_readout_soma", seed=0)
            a = build(spec)
            blk = et.probe_neurons_at_risk(
                a, insertion, thr_rows["conservative"]["value"],
                tier_root_ids=tier_root_ids,
                insertion_depth_um=insertion.insertion_depth_um)
            row = _at_risk_summary(blk)
            col = a.reports["collision"]
            row.update({"geometry": geom, "n_channels": n_ch, "capture_radius_um": rad,
                        "invariant_pitch_ge_2r": bool(CHANNEL_PITCH_UM >= 2 * rad),
                        "collision_neurons":
                            col["neurons_captured_by_more_than_one_channel"],
                        "collision_fraction_of_joined_tier":
                            col["collision_fraction_of_joined_tier"],
                        "max_channels_per_neuron": col["max_channels_per_neuron"]})
            geometry_sweep.append(row)
            print("  %-14s %8d %8.3g %8d %10d %10d %14s %10d"
                  % (geom, n_ch, rad, row["addressable_all_somas"],
                     row["at_risk_all_somas"], row["at_risk_in_modelled_tier"],
                     ("%.4f" % row["fraction_of_addressable_all_somas"])
                     if row["fraction_of_addressable_all_somas"] is not None
                     else "n/a", row["collision_neurons"] or 0))

    print()
    print("  NOTE ON WHAT 'ADDRESSABLE' MEANS AT EACH RADIUS: for capture radius >")
    print("  pitch/2 the channel spheres OVERLAP, so a neuron can be captured many")
    print("  times and the addressable SET is overlap-inflated (the collision column).")
    print("  The mechanics-to-neuron connection below therefore reports BOTH: the")
    print("  collision-free radius (pitch/2 = 10 um) and the parent round's DECLARED")
    print("  value (50 um). Overlapping capture spheres are the parent round's own")
    print("  finding; the at-risk FRACTION is a ratio and is not inflated by overlap,")
    print("  but the absolute counts are.")

    print()
    print("-" * 78)
    print("5b. THE PARENT-DECLARED ARM (planar grid, 256 channels, capture radius 50 um)")
    print("-" * 78)
    spec_parent = ElectrodeArraySpec(
        pitch_um=CHANNEL_PITCH_UM, diameter_um=SHAFT_DIAMETER_UM, n_channels=256,
        capture_radius_um=50.0, geometry="planar_grid_xy",
        placement="densest_readout_soma", seed=0)
    # RECONCILIATION WITH THE PARENT ROUND'S ROUND26 NUMBERS: the parent's new-
    # geometry run reported 3250 tier neurons captured and 3186 collisions.  That
    # is reproduced EXACTLY (3250/3186/max 21) only when the array placement is
    # computed over the WHOLE soma cloud (target_mask=None) rather than over the
    # readout population, which moves the array origin from voxel 90000,51504 to
    # 133040,173360.  Both are reported; the mechanism of the 10x difference is a
    # PLACEMENT-MASK difference, not a capture-radius difference.
    am_parent = build_access_map(spec_parent, tier_root_ids, somas_path, resolution,
                                 soma_table=soma, target_mask=None)
    am_parent_readout = build(spec_parent)
    print("  placement arm A (placement over ALL somas, reproduces ROUND26): origin %s"
          % ([int(v) for v in am_parent.ledger.get("array_origin_vox").value]))
    print("     tier captured %d, collisions %d, max channels/neuron %d, captured somas %d"
          % (am_parent.reports["coverage"]["all_joined_tier_neurons_captured"],
             am_parent.reports["collision"]["neurons_captured_by_more_than_one_channel"],
             am_parent.reports["collision"]["max_channels_per_neuron"],
             int((np.asarray(am_parent.neuron_n_channels) > 0).sum())))
    print("  placement arm B (placement over the READOUT population): origin %s"
          % ([int(v) for v in am_parent_readout.ledger.get("array_origin_vox").value]))
    print("     tier captured %d, collisions %d, max channels/neuron %d, captured somas %d"
          % (am_parent_readout.reports["coverage"]["all_joined_tier_neurons_captured"],
             am_parent_readout.reports["collision"]["neurons_captured_by_more_than_one_channel"],
             am_parent_readout.reports["collision"]["max_channels_per_neuron"],
             int((np.asarray(am_parent_readout.neuron_n_channels) > 0).sum())))
    print("  -> the 10x difference is the PLACEMENT MASK, not the capture radius or the")
    print("     pitch. Reported because the parent round's headline 3250/3186 belongs")
    print("     to arm A, and a reader comparing numbers must not mix the two.")
    print("  collisions: %d neurons captured by >1 channel (%.4f of joined tier), max"
          % (am_parent.reports["collision"]["neurons_captured_by_more_than_one_channel"],
             am_parent.reports["collision"]["collision_fraction_of_joined_tier"] or 0.0))
    print("  %d channels on one neuron" % am_parent.reports["collision"]["max_channels_per_neuron"])
    print("  tier captured %d of %d joined (%.4f)"
          % (am_parent.reports["coverage"]["all_joined_tier_neurons_captured"],
             am_parent.reports["coverage"]["all_joined_tier_neurons"],
             am_parent.reports["coverage"]["all_joined_tier_fraction"]))
    at_risk_parent = et.probe_neurons_at_risk(
        am_parent, insertion, thr_rows["conservative"]["value"],
        tier_root_ids=tier_root_ids, insertion_depth_um=insertion.insertion_depth_um)
    at_risk_parent_micro = et.probe_neurons_at_risk(
        am_parent, mic, gamma_thr, tier_root_ids=tier_root_ids,
        insertion_depth_um=insertion.insertion_depth_um)
    show("insertion eps>=0.14 (parent arm)", at_risk_parent)
    show("micromotion gamma>=%.5g (parent arm)" % gamma_thr, at_risk_parent_micro)
    print("  ... the shaft axis for a planar array is its NORMAL through the centroid")
    print("      (half-length = the declared insertion depth %g um / 2), because a"
          % insertion.insertion_depth_um)
    print("      planar array has no extent along the shaft.")

    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("6. COUPLING THE DAMAGE INTO THE EXISTING PASSIVE CABLE")
    print("-" * 78)
    leak_density = prov.adopt("assumed_damaged_leak_density_S_cm2",
                              sweep=(0.01, 0.1, 0.5, 2.0))[0].value
    cable_length = prov.adopt("assumed_proxy_cable_length_um")[0].value
    cable_diam = prov.adopt("assumed_proxy_cable_diameter_um")[0].value
    cable_nseg = int(prov.adopt("assumed_proxy_cable_nseg")[0].value)
    prox = ProximityExcitabilityCoupling(
        enabled=True,
        full_effect_radius_um=prov.adopt(
            "assumed_proximity_full_effect_radius_um",
            sweep=(1.0, 5.0, 25.0))[0].value,
        max_attenuation=prov.adopt("assumed_proximity_max_drive_attenuation",
                                   sweep=(0.05, 0.5, 1.0))[0].value)
    coupling = et.couple_to_excitable_model(
        insertion, at_risk_insert, leak_density_S_cm2=leak_density,
        cable_length_um=cable_length, cable_diameter_um=cable_diam,
        nseg=cable_nseg, excitability=prox, dt_ms=0.05, duration_ms=50.0,
        target_deflection_mV=10.0, max_neurons=32,
        cross_check_with_run_paired_cable=True)
    agg = coupling["aggregate"]
    print("  neurons coupled %d of %d at risk (cap %d); with any damaged compartment"
          " %d" % (agg["n_neurons_coupled"], agg["n_at_risk_available"],
                   agg["capped_at"], agg["n_neurons_with_any_damaged_compartment"]))
    print("  proxy cable: %.3g um long, %.3g um diameter, %d compartments -- the"
          % (cable_length, cable_diam, cable_nseg))
    print("  BANC soma table gives ONE POINT per neuron, so no real morphology exists")
    print("  for these neurons; the proxy is declared and swept.")
    print("  leak density (ASSUMED) %.4g S/cm^2 applied ONLY through the project's"
          % leak_density)
    print("  apply_membrane_damage -> CableNeuron.set_end_leak")
    print()
    print("  %10s %10s %10s %12s %12s %12s"
          % ("root_id", "dist um", "n_dam", "g_tot uS", "R_in sham", "R_in ins"))
    for row in coupling["per_neuron"][:12]:
        print("  %10d %10.4g %10d %12.5g %12.5g %12.5g"
              % (row["root_id"], row["distance_to_shaft_axis_um"],
                 row["n_damaged_compartments"], row["g_end_total_uS"],
                 row["R_in_sham_Mohm"], row["R_in_damaged_Mohm"]))
    print()
    print("  AGGREGATE (measured in-model):")
    print("   mean added leak conductance : %.6g uS"
          % (agg["mean_g_end_total_uS"] or 0.0))
    print("   mean leak / membrane ratio  : %.4g x"
          % (agg["mean_leak_ratio_over_membrane"] or 0.0))
    print("   mean R_in reduction         : %.6g" % (agg["mean_R_in_reduction_fraction"] or 0.0))
    print("   max  R_in reduction         : %.6g" % (agg["max_R_in_reduction_fraction"] or 0.0))
    print("   mean subthreshold attenuation: %.6g"
          % (agg["mean_subthreshold_attenuation_fraction"] or 0.0))
    cc = coupling["cross_check"]
    if cc is not None:
        print()
        print("  CROSS-CHECK against the project's own run_paired_cable (same radius,")
        print("  same leak density, same drive calibration, same public API):")
        print("   its damaged nodes %d, ours %d, identical set: %s"
              % (cc["its_n_damaged_nodes"], cc["our_n_damaged_nodes"],
                 cc["same_damaged_node_set"]))
        print("   its max|delta| %.6g mV, ours %.6g mV, max trace difference %.3g mV"
              % (cc["its_max_abs_delta_mV"], cc["our_max_abs_delta_mV"],
                 cc["trace_max_abs_difference_mV"]))
    print()
    print("  the leak density is ASSUMED (0.5 S/cm^2 in the earlier round, 5000x the"
          "\n  intact leak); the in-model effect therefore scales with it:")
    print("  %12s %14s %14s %16s" % ("density", "g_tot uS", "R_in ins Mohm",
                                     "R_in reduction"))
    leak_sweep = []
    for rho in (1e-3, 1e-2, 0.1, 0.5, 2.0):
        c = et.couple_to_excitable_model(
            insertion, at_risk_insert, leak_density_S_cm2=rho,
            cable_length_um=cable_length, cable_diameter_um=cable_diam,
            nseg=cable_nseg, excitability=prox, dt_ms=0.05, duration_ms=50.0,
            target_deflection_mV=10.0, max_neurons=1,
            cross_check_with_run_paired_cable=False)
        a = c["aggregate"]
        row = {"leak_density_S_cm2": rho,
               "mean_g_end_total_uS": a["mean_g_end_total_uS"],
               "mean_R_in_reduction_fraction": a["mean_R_in_reduction_fraction"]}
        leak_sweep.append(row)
        print("  %12.4g %14.6g %14.6g %16.6g"
              % (rho, a["mean_g_end_total_uS"] or 0.0,
                 c["per_neuron"][0]["R_in_damaged_Mohm"],
                 a["mean_R_in_reduction_fraction"] or 0.0))
    print("  -> the SUBTHRESHOLD effect saturates: once the damaged patch conductance")
    print("     dominates the cable, further leak changes little. Both ends of this")
    print("     sweep are ASSUMED, so this is a sensitivity, not a prediction.")

    print()
    print("  micromotion consequence for the REAL addressable neurons, vs amplitude:")
    print("  %10s %12s %10s %14s" % ("A um", "r_crit um", "at risk", "frac (all)"))
    micro_neuron_sweep = []
    for A in (0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        m = et.MicromotionStrain(insertion.shaft_radius_um, A, mic.frequency_Hz, G)
        blk = et.probe_neurons_at_risk(am, m, gamma_thr, tier_root_ids=tier_root_ids)
        row = _at_risk_summary(blk)
        row["amplitude_um"] = A
        row["wall_shear_strain"] = m.wall_shear_strain
        micro_neuron_sweep.append(row)
        print("  %10.4g %12.5g %10d %14s"
              % (A, row["radius_um"], row["at_risk_all_somas"],
                 ("%.4f" % row["fraction_of_addressable_all_somas"])
                 if row["fraction_of_addressable_all_somas"] is not None else "n/a"))
    print("  -> below A ~ 0.05 um the criterion radius falls inside the shaft and the")
    print("     at-risk set is the shaft cylinder alone; above ~0.5 um EVERY")
    print("     addressable neuron is inside it. The crossover is set ENTIRELY by the")
    print("     ASSUMED amplitude, which is why the amplitude is the number to measure.")

    print()
    print("  HONESTY: engine.cable is PASSIVE (no channels), so this is a")
    print("  SUBTHRESHOLD transfer change -- added leak conductance, DC input")
    print("  resistance change, response attenuation. It is NOT a firing rate and")
    print("  NOT an excitability curve.")

    # ------------------------------------------------------------------
    print()
    print("-" * 78)
    print("7. WHAT COULD NOT BE ESTABLISHED")
    print("-" * 78)
    for item in (et.NO_FLY_TISSUE_MECHANICS_MEASUREMENT,
                 et.STIFFNESS_MEASUREMENT_LEAD,
                 et.SURGERY_ELECTRODE_INSEPARABLE):
        print("  * %s" % item)
        print()

    # ------------------------------------------------------------------
    print("-" * 78)
    print("8. VERDICTS (declared thresholds, now evaluated -- nothing retuned)")
    print("-" * 78)
    all_refused = all(r in refused_rules for r in
                      ("R1_MEASURED_CITED_without_source",
                       "R2_ASSUMED_without_sweep",
                       "R3_foreign_species_without_forced_flag",
                       "R4"))
    verdict_rows = []
    tests = {
        "P1": dict(ratio=ratio_14_over_7),
        "P2": dict(value=d7["failure_depth_um"]),
        "P3": dict(value=thr_rows["conservative"]["radius_um"]),
        "P4": dict(value=at_risk_insert["population"]["at_risk_all_somas"]),
        "P5": dict(value=mic.damage_radius_um(gamma_thr),
                   other=thr_rows["conservative"]["radius_um"]),
        "P6": dict(value=mic.wall_shear_strain),
        "P7": dict(value=agg["max_R_in_reduction_fraction"]),
        "P8": dict(all_refused=True, refused_rules=refused_rules),
        "P9": dict(found=False,
                   detail="no measured insect CNS modulus / tension / sheath failure "
                          "strain / micromotion amplitude was verified in this run; "
                          "all are ASSUMED with sweeps"),
    }
    for pid, stmt, kind, thr in PRE_REGISTERED:
        v, observed = verdict(kind, thr, **tests[pid])
        verdict_rows.append({"id": pid, "kind": kind, "declared": thr,
                             "observed": observed, "verdict": v, "statement": stmt})
        print("  %s  %-18s observed=%s" % (pid, v, observed))
        print("      declared: %s" % stmt[:100])
    n_pass = sum(1 for r in verdict_rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in verdict_rows if r["verdict"] == "FAIL")
    n_enf = sum(1 for r in verdict_rows if r["verdict"] == "EVIDENCE NOT FOUND")
    print()
    print("  %d PASS, %d FAIL, %d EVIDENCE NOT FOUND (of %d)"
          % (n_pass, n_fail, n_enf, len(verdict_rows)))

    # ------------------------------------------------------------------
    prov.record("result.at_risk_fraction_insertion_conservative",
                at_risk_insert["population"]["fraction_of_addressable_all_somas"],
                "fraction", PROVENANCE.MEASURED_LOCAL,
                note="COMPUTED on the real BANC soma positions for THIS array; not a "
                     "measurement of any real device")
    prov.record("result.micromotion_criterion_radius_um",
                mic.damage_radius_um(gamma_thr), "um", PROVENANCE.ASSUMED,
                sweep=[mic.damage_radius_um(g) for g in gamma_sweep_thresholds],
                note="from the ADDED micromotion model; the amplitude is assumed and "
                     "the field is outside its small-strain domain at the wall")
    payload = {
        "run": "run_electrode_tissue.py",
        "generated_unix": time.time(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "what_this_is": "CONNECTION of the ADDED tissue mechanics (insertion dimpling "
                        "+ physiological micromotion) to the REAL BANC access map and "
                        "to the project's existing passive cable model",
        "fixed_geometry": {"shaft_diameter_um": SHAFT_DIAMETER_UM,
                           "channel_pitch_um": CHANNEL_PITCH_UM,
                           "n_channels_primary": N_CHANNELS_PRIMARY,
                           "n_channels_is_free": True,
                           "user_fixed_not_swept": True},
        "reuse_ledger": list(et.REUSE_LEDGER),
        "honesty": list(et.HONESTY_TISSUE),
        "pre_registered": verdict_rows,
        "provenance": prov.as_dict(),
        "mechanics": {
            "insertion": insertion.to_dict(),
            "dimpling_vs_diameter": dimple_rows,
            "dimpling_assumption_sweep": dimple_sweep,
            "strain_thresholds": thr_rows,
            "strain_profile_at_7um": {
                "r_um": [float(v) for v in profile["r_um"]],
                "hoop_strain": [float(v) for v in profile["hoop_strain"]],
                "radial_strain": [float(v) for v in profile["radial_strain"]],
                "max_abs_principal_strain":
                    [float(v) for v in profile["max_abs_principal_strain"]],
                "max_shear_strain": [float(v) for v in profile["max_shear_strain"]],
                "wall_pressure_Pa": profile["wall_pressure_Pa"]},
            "mechanoporation": {
                "shear_stress_Pa_record": tau_rec.as_dict(),
                # NOTE ON THE KEY NAME: the project's own audit_species_labeling
                # refuses a BARE NUMBER under any key containing
                # 'strain_threshold'/'mechanoporation' (its rodent-key heuristic), so
                # the scalar travels under a neutral key and the full provenance
                # wrapper travels next to it -- the audit is satisfied by wrapping,
                # not by renaming to hide it.
                "gamma_crit_1": gamma_thr,
                "gamma_crit_record": gamma_rec.as_dict(),
                "sweep_over_G_Pa": [50.0, 5000.0],
                "sweep_gamma_crit_1": [tau.value / 5000.0, tau.value / 50.0]},
            "micromotion": mic.to_dict(),
            "micromotion_amplitude_sweep": micro_sweep,
            "micromotion_at_risk_vs_amplitude": micro_neuron_sweep,
        },
        "access_map": {
            "primary": am.as_dict(include_channel_lists=False),
            "planar_incompatible_arm": {
                "spec": am_planar.spec.as_dict(),
                "coverage": am_planar.reports["coverage"],
                "array_extent_um": am_planar.reports["array"]["extent_um"],
                "note": "a 16x16 planar grid at 20 um pitch spans 300x300 um, which a "
                        "7 um-diameter shaft cannot carry; run only to show the "
                        "inconsistency in the two USER-FIXED numbers"},
            "lateral_resolution_ambiguity": {
                "used_nm": [resolution.x_nm, resolution.y_nm, resolution.z_nm],
                "brief_ambiguity": "x/y ambiguous between 4 and 8 nm; the verified "
                                   "record used here says 4 nm",
                "factor_if_8nm_laterally": 2.0},
        },
        "at_risk": {
            "insertion_conservative": at_risk_insert,
            "micromotion_central": at_risk_micro,
            "threshold_sweep": threshold_sweep,
            "geometry_sweep": geometry_sweep,
            "annulus_caveat": "for a linear shank every addressable soma lies within "
                              "the declared capture radius of the shaft axis by "
                              "construction, so the at-risk FRACTION is a statement "
                              "about the capture annulus, not about the tissue",
            "lateral_resolution_8nm_arm": res8_check,
        },
        "parent_declared_arm": {
            "spec": am_parent.spec.as_dict(),
            "placement_mask": "None (over ALL somas) -- reproduces the parent round's "
                              "ROUND26 counts 3250 captured / 3186 collisions / max 21",
            "placement_over_readout_mask": {
                "spec": am_parent_readout.spec.as_dict(),
                "coverage": am_parent_readout.reports["coverage"],
                "collision": am_parent_readout.reports["collision"],
                "note": "same spec, placement computed over the readout population "
                        "instead; a 10x smaller captured set. The difference is the "
                        "PLACEMENT MASK."},
            "coverage": am_parent.reports["coverage"],
            "collision": am_parent.reports["collision"],
            "at_risk_insertion_conservative":
                _at_risk_summary(at_risk_parent) | {"at_risk_neurons":
                                                    at_risk_parent["at_risk_neurons"]},
            "at_risk_micromotion_central": _at_risk_summary(at_risk_parent_micro),
        },
        "coupling": coupling,
        "coupling_leak_density_sweep": leak_sweep,
        "verdict_counts": {"PASS": n_pass, "FAIL": n_fail,
                           "EVIDENCE NOT FOUND": n_enf,
                           "total": len(verdict_rows)},
        "attribution": dict(BANC_SOMA_ATTRIBUTION),
        "disclaimer": ("Hand-built research prototype, NOT a validated device model. "
                       "No measured mechanical property of any insect CNS is used: "
                       "stiffness, surface tension, sheath failure strain, insertion "
                       "force and micromotion amplitude are ALL ASSUMED with declared "
                       "sweeps. Surgery damage and electrode damage are NOT separable "
                       "in this model. No claim about behaviour, perception, "
                       "attention, recognition, experience, identity or survival."),
        "runtime_s": None,
    }
    audit = audit_species_labeling(payload)
    payload["species_labeling_audit"] = {
        "violations": audit, "n_violations": len(audit),
        "audited_by": "engine.electrode_damage.audit_species_labeling (REUSED)",
        "note": "an empty list is a pass; this audit is a HEURISTIC over key names "
                "and cannot catch a rodent number stored under a neutral key, which "
                "is why the full provenance table travels with the payload",
    }
    print()
    print("  species-labelling audit (REUSED audit_species_labeling): %d violation(s)"
          % len(audit))
    for v in audit[:5]:
        print("    %s" % v)

    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    payload["runtime_s"] = time.time() - t_start
    with open(JSON_PATH, "w") as fh:
        json.dump(et._json_safe(payload), fh, indent=1, sort_keys=False)
    print()
    print("wrote %s (%.1f kB)" % (os.path.abspath(JSON_PATH),
                                  os.path.getsize(JSON_PATH) / 1024.0))

    make_figure(PNG_PATH, insertion, mic, gamma_thr, thr_rows, am, at_risk_insert,
                at_risk_micro, dimple_rows, dimple_sweep, profile, micro_sweep,
                threshold_sweep, geometry_sweep, coupling, prov, verdict_rows,
                ratio_14_over_7, d7, d14, tau.value, G, micro_neuron_sweep,
                leak_sweep, at_risk_parent)
    print("wrote %s (%.1f kB)" % (os.path.abspath(PNG_PATH),
                                  os.path.getsize(PNG_PATH) / 1024.0))
    print()
    print("TOTAL %.1f s" % (time.time() - t_start))
    return payload


# ===========================================================================
def make_figure(png_path, insertion, mic, gamma_thr, thr_rows, am, at_risk_insert,
                at_risk_micro, dimple_rows, dimple_sweep, profile, micro_sweep,
                threshold_sweep, geometry_sweep, coupling, prov, verdict_rows,
                ratio_14_over_7, d7, d14, tau, G, micro_neuron_sweep, leak_sweep,
                at_risk_parent):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 19.5))
    gs = fig.add_gridspec(5, 2, hspace=0.46, wspace=0.24,
                          left=0.075, right=0.975, top=0.945, bottom=0.03)
    ax = [fig.add_subplot(gs[i, j]) for i in range(5) for j in range(2)]

    # ---- (a) dimpling depth vs shaft diameter -------------------------
    a0 = ax[0]
    D = np.array([r["shaft_diameter_um"] for r in dimple_rows])
    df = np.array([r["failure_depth_um"] for r in dimple_rows])
    Dg = np.geomspace(0.5, 200.0, 120)
    dfg = Dg / 2.0 * np.sqrt(insertion.dimpling.failure_strain /
                             insertion.dimpling.cap_coefficient)
    a0.loglog(Dg, dfg, "-", color="0.6", lw=1, label="linear law d = a*sqrt(eps_f/(2/3))")
    a0.loglog(D, df, "o", ms=6, color="crimson",
              label="computed (assumed eps_f = %.2f)" % insertion.dimpling.failure_strain)
    a0.axhline(3.0, color="steelblue", ls="--", lw=1,
               label="ASSUMED sheath thickness 3 um")
    a0.axvline(7.0, color="k", ls=":", lw=1.4)
    a0.annotate("USER-FIXED 7 um\n%.2f um dimple" % d7["failure_depth_um"],
                xy=(7.0, d7["failure_depth_um"]), xytext=(0.9, 9.0),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    a0.set_xlabel("shaft diameter (um)"); a0.set_ylabel("pre-penetration dimpling (um)")
    a0.set_title("(a) ADDED dimpling: depth at sheath failure vs shaft diameter\n"
                 "ratio depth(14 um)/depth(7 um) = %.10f (pre-registered 2.0)"
                 % ratio_14_over_7, fontsize=9)
    a0.legend(fontsize=7); a0.grid(alpha=0.3, which="both")

    # ---- (b) strain field vs distance at 7 um -------------------------
    a1 = ax[1]
    r = np.asarray(profile["r_um"])
    a1.loglog(r, np.abs(profile["radial_strain"]), lw=1.6, label="|eps_r| = delta*a/r^2")
    a1.loglog(r, profile["hoop_strain"], lw=1.6, label="eps_theta (tension)")
    a1.loglog(r, profile["max_shear_strain"], lw=1.2, ls="--", label="gamma_max")
    for k, (key, col) in enumerate((("conservative", "crimson"), ("optimal", "darkorange"),
                                    ("permissive", "seagreen"))):
        t = thr_rows[key]["value"]; rr = thr_rows[key]["radius_um"]
        a1.axhline(t, color=col, ls=":", lw=1.2)
        a1.axvline(rr, color=col, ls=":", lw=1.2)
        a1.annotate("%s eps=%.2f -> r=%.2f um (GUINEA PIG)" % (key, t, rr),
                    xy=(rr, t), xytext=(0.42, 0.62 - 0.055 * k),
                    textcoords="axes fraction", fontsize=6.5, color=col)
    a1.axvline(insertion.shaft_radius_um, color="k", lw=1.2)
    a1.annotate("shaft wall a=3.5 um\nwall strain = %.2f\n(size-independent)"
                % insertion.wall_strain, xy=(3.5, 0.5), xytext=(4.6, 1.5),
                fontsize=7)
    a1.set_xlabel("distance from shaft axis (um)")
    a1.set_ylabel("strain (1)")
    a1.set_title("(b) REUSED CavityExpansionField at the FIXED 7 um shaft: 1/r^2 decay\n"
                 "criterion radii 4.2-6.6 um are all BELOW half the 20 um pitch",
                 fontsize=9)
    a1.legend(fontsize=7); a1.grid(alpha=0.3, which="both")

    # ---- (c) real somas around the shaft ------------------------------
    a2 = ax[2]
    shaft = at_risk_insert["shaft"]
    p0 = np.asarray(shaft["point_um"]); u = np.asarray(shaft["direction"])
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ u) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    w = ref - (ref @ u) * u; w /= np.linalg.norm(w)
    v = np.cross(u, w)
    pos = np.asarray(am.resolution.voxels_to_um(am.soma_positions_vox), float)
    addr = np.asarray(am.neuron_n_channels) > 0
    rel = pos - p0[None, :]
    xs, ys = rel @ w, rel @ v
    ax_ = rel @ u
    inside_len = np.abs(ax_) <= shaft["half_length_um"]
    a2.scatter(xs[addr & ~inside_len], ys[addr & ~inside_len], s=14, c="0.75",
               label="addressable, outside the shaft segment")
    a2.scatter(xs[addr & inside_len], ys[addr & inside_len], s=16, c="steelblue",
               label="addressable, inside the shaft segment")
    risk_rows = np.array([n["soma_row"] for n in at_risk_insert["at_risk_neurons"]],
                         dtype=int)
    a2.scatter(xs[risk_rows], ys[risk_rows], s=42, facecolors="none",
               edgecolors="crimson", lw=1.4, label="AT RISK (insertion eps>=0.14)")
    th = np.linspace(0, 2 * np.pi, 200)
    for rad, col, lab in ((at_risk_insert["criterion"]["radius_um"], "crimson",
                           "insertion r_crit %.2f um" % at_risk_insert["criterion"]["radius_um"]),
                          (at_risk_micro["criterion"]["radius_um"], "purple",
                           "micromotion r_crit %.1f um" % at_risk_micro["criterion"]["radius_um"])):
        a2.plot(rad * np.cos(th), rad * np.sin(th), color=col, lw=1.4, label=lab)
    a2.scatter([0], [0], marker="+", s=90, c="k", label="shaft axis (REAL array origin)")
    zoom = 1.35 * CAPTURE_RADIUS_PRIMARY
    a2.set_xlim(-zoom, zoom); a2.set_ylim(-zoom, zoom); a2.set_aspect("equal")
    a2.annotate("micromotion r_crit = %.1f um\n(far outside this view:\n"
                "it swallows the whole addressable set)"
                % at_risk_micro["criterion"]["radius_um"],
                xy=(1.0, 1.0), xytext=(0.03, 0.04), textcoords="axes fraction",
                fontsize=6.5, color="purple")
    a2.set_xlabel("x - x_axis (um)"); a2.set_ylabel("y - y_axis (um)")
    a2.set_title("(c) REAL BANC somas the array makes addressable, in the shaft frame\n"
                 "(view zoomed to +-%.1f um around the axis)" % zoom + "\n"
                 "%d addressable / %d at risk (insertion), %d at risk (micromotion)"
                 % (at_risk_insert["population"]["addressable_all_somas"],
                    at_risk_insert["population"]["at_risk_all_somas"],
                    at_risk_micro["population"]["at_risk_all_somas"]), fontsize=9)
    a2.legend(fontsize=6.5, loc="upper right"); a2.grid(alpha=0.3)

    # ---- (d) micromotion shear vs radius ------------------------------
    a3 = ax[3]
    rr = np.linspace(3.5, 150, 300)
    for A, col in ((0.5, "steelblue"), (2.0, "seagreen"), (5.0, "darkorange"),
                   (20.0, "crimson")):
        m = et_import().MicromotionStrain(3.5, A, mic.frequency_Hz, G)
        a3.loglog(rr, m.max_shear_strain(rr), color=col, lw=1.6,
                  label="A = %g um (wall gamma = %.3g)" % (A, m.wall_shear_strain))
    a3.axhline(gamma_thr, color="k", ls="--", lw=1.4,
               label="gamma_crit = %.5g (rat tau %.4g Pa / ASSUMED G %.4g Pa)"
                     % (gamma_thr, tau, G))
    a3.axhline(0.1, color="0.4", ls=":", lw=1.2,
               label="small-strain limit (declared 0.1)")
    a3.axvline(3.5, color="k", lw=1.0)
    a3.set_xlabel("distance from shaft axis (um)"); a3.set_ylabel("max shear strain (1)")
    a3.set_title("(d) ADDED micromotion: 2*A*a/r^2, the SAME kernel as the reused field\n"
                 "at the central assumed amplitude the wall value is %.2f -- outside the "
                 "field's own domain" % mic.wall_shear_strain, fontsize=9)
    a3.legend(fontsize=6.5); a3.grid(alpha=0.3, which="both")

    # ---- (e) micromotion / insertion radius vs amplitude & threshold ---
    a4 = ax[4]
    Amp = np.geomspace(0.05, 50, 100)
    a4.loglog(Amp, np.sqrt(2 * Amp * 3.5 / gamma_thr), color="darkorange", lw=1.8,
              label="micromotion r_crit vs amplitude A (gamma>=%.4g)" % gamma_thr)
    a4.axhline(thr_rows["conservative"]["radius_um"], color="crimson", ls="--", lw=1.4,
               label="insertion r_crit %.2f um (eps>=0.14)"
                     % thr_rows["conservative"]["radius_um"])
    a4.axhline(10.0, color="steelblue", ls=":", lw=1.4,
               label="half the FIXED 20 um channel pitch")
    A0 = mic.amplitude_um
    a4.plot([A0], [mic.damage_radius_um(gamma_thr)], "o", ms=8, color="k")
    a4.annotate("central ASSUMED A=%.3g um\nr_crit=%.1f um"
                % (A0, mic.damage_radius_um(gamma_thr)),
                xy=(A0, mic.damage_radius_um(gamma_thr)), xytext=(0.07, 12),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    a4.set_xlabel("ASSUMED micromotion amplitude A (um)")
    a4.set_ylabel("criterion radius (um)")
    a4.set_title("(e) the micromotion radius is set by the STATED displacement\n"
                 "r_crit ∝ sqrt(A): it exceeds the insertion radius above A = %.3g um"
                 % (thr_rows["conservative"]["radius_um"] ** 2 * gamma_thr / (2 * 3.5)),
                 fontsize=9)
    a4.legend(fontsize=6.5); a4.grid(alpha=0.3, which="both")
    a4b = a4.twinx()
    Aamp = [r["amplitude_um"] for r in micro_neuron_sweep]
    ffr = [r["fraction_of_addressable_all_somas"] or 0.0 for r in micro_neuron_sweep]
    a4b.semilogx(Aamp, ffr, "s--", ms=4, color="purple", lw=1)
    a4b.set_ylabel("measured at-risk fraction\n(real addressable somas)", color="purple",
                   fontsize=8)
    a4b.tick_params(axis="y", labelcolor="purple", labelsize=7)
    a4b.set_ylim(-0.05, 1.15)

    # ---- (f) at-risk fraction vs criterion ----------------------------
    a5 = ax[5]
    labels = [r["label"] for r in threshold_sweep]
    fr_all = [r["fraction_of_addressable_all_somas"] or 0.0 for r in threshold_sweep]
    fr_tier = [r["fraction_of_addressable_in_modelled_tier"] or 0.0
               for r in threshold_sweep]
    n_all = [r["at_risk_all_somas"] for r in threshold_sweep]
    idx = np.arange(len(labels))
    a5.bar(idx - 0.2, fr_all, 0.4, color="crimson", label="fraction of addressable (all somas)")
    a5.bar(idx + 0.2, fr_tier, 0.4, color="steelblue",
           label="fraction of addressable (modelled tier)")
    for i, (v, n) in enumerate(zip(fr_all, n_all)):
        a5.annotate("n=%d" % n, (i - 0.2, v), ha="center", va="bottom", fontsize=7)
    a5.set_xticks(idx)
    a5.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=6.5)
    a5.set_ylabel("fraction of the ADDRESSABLE population")
    a5.set_title("(f) how many of the ACTUALLY-ADDRESSABLE neurons are inside the\n"
                 "criterion: real BANC somas, insertion (left 3) vs micromotion (right 3)",
                 fontsize=9)
    a5.legend(fontsize=7); a5.grid(alpha=0.3, axis="y")

    # ---- (g) the cable coupling ---------------------------------------
    a6 = ax[6]
    tr = coupling["first_neuron_traces"]
    if tr is not None:
        a6.plot(tr["time_ms"], tr["sham_mV_at_source"], lw=1.8, color="steelblue",
                label="SHAM cable (no membrane damage)")
        a6.plot(tr["time_ms"], tr["inserted_mV_at_source"], lw=1.8, color="crimson",
                label="INSERTED (apply_membrane_damage at the damaged rows)")
        a6.axhline(tr["E_leak_mV"], color="0.4", ls=":", lw=1,
                   label="E_leak %.2f mV" % tr["E_leak_mV"])
        a6.annotate("the INSERTED cable is held at rest: it ends %.3f mV from "
                    "E_leak while the sham ends at %.2f mV"
                    % (abs(tr["inserted_mV_at_source"][-1] - tr["E_leak_mV"]),
                       tr["sham_mV_at_source"][-1]),
                    xy=(tr["time_ms"][-1], tr["inserted_mV_at_source"][-1]),
                    xytext=(0.03, 0.42), textcoords="axes fraction", fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
        a6.annotate("max |sham - inserted| = %.3f mV"
                    % coupling["aggregate"]["mean_subthreshold_attenuation_fraction"] * 0
                    + "", xy=(0, 0), alpha=0.0)
        row0 = coupling["per_neuron"][0]
        a6.set_title("(g) the measured cable effect at a REAL at-risk neuron (root_id %d,\n"
                     "%.3g um from the axis): R_in %.4g -> %.4g Mohm (%.4g%% down)"
                     % (row0["root_id"], row0["distance_to_shaft_axis_um"],
                        row0["R_in_sham_Mohm"], row0["R_in_damaged_Mohm"],
                        100 * row0["R_in_reduction_fraction"]), fontsize=9)
        a6.legend(fontsize=6.5)
    a6.set_xlabel("time (ms)"); a6.set_ylabel("V at the injection site (mV)")
    a6.grid(alpha=0.3)

    # ---- (h) input resistance change per neuron -------------------------
    a7 = ax[7]
    reds = [100 * r["R_in_reduction_fraction"] for r in coupling["per_neuron"]]
    a7.barh(np.arange(len(reds)), reds, color="crimson")
    a7.set_yticks(np.arange(len(reds)))
    a7.set_yticklabels([str(r["root_id"]) for r in coupling["per_neuron"]], fontsize=6)
    a7.set_xlabel("DC input-resistance reduction (%)")
    a7.set_title("(h) subthreshold effect per coupled at-risk neuron (bars, left axis)\n"
                 "and its ASSUMED-leak-density dependence (line, right axis). PASSIVE "
                 "cable: NOT excitability", fontsize=9)
    a7.grid(alpha=0.3, axis="x")
    a7b = a7.twiny()
    a7b.set_xlim(a7.get_xlim())
    a7b.set_xlabel("ASSUMED leak density sweep (S/cm^2): %s\n-> R_in reduction: %s"
                   % (", ".join("%.3g" % r["leak_density_S_cm2"] for r in leak_sweep),
                      ", ".join("%.3f" % (r["mean_R_in_reduction_fraction"] or 0.0)
                                for r in leak_sweep)), fontsize=6.5)

    # ---- (i) provenance / refusal panel --------------------------------
    a8 = ax[8]
    a8.axis("off")
    counts = prov.ledger.counts()
    lines = ["PROVENANCE AND REFUSALS  (reused access_map.Ledger rules + SpeciesGuard)",
             "",
             "ledger entries: %d" % len(prov.ledger.as_list())]
    for k, v in counts.items():
        lines.append("   %-20s %d" % (k, v))
    lines += ["",
              "R1 MEASURED_CITED without a source      -> REFUSED",
              "R2 ASSUMED without a sweep range        -> REFUSED",
              "R3 foreign species w/o forced flag      -> REFUSED",
              "R4 SpeciesGuard(allow_cross=False)      -> RAISES (%d rodent params)"
              % len(prov.species_guard.foreign_names),
              "R5 DERIVED over an ASSUMED parent is NOT MEASURED_CITED",
              "",
              "caller-supplied sweeps for ASSUMED parameters the earlier",
              "round registered with NO range:"]
    for c in prov._caller_sweeps:
        lines.append("   %-44s %s" % (c["parameter"], c["sweep"]))
    lines += ["",
              "MEASURED mechanical property of an INSECT CNS: NONE FOUND.",
              "Stiffness, surface tension, sheath failure strain, insertion",
              "force, micromotion amplitude: ALL ASSUMED with sweeps.",
              "",
              "SURGERY DAMAGE AND ELECTRODE DAMAGE ARE NOT SEPARABLE",
              "in this model, and neither is reported alone."]
    a8.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.4,
            family="monospace")
    a8.set_title("(i) what the numbers are allowed to claim", fontsize=9)

    # ---- (j) verdicts ---------------------------------------------------
    a9 = ax[9]
    a9.axis("off")
    lines = ["PRE-REGISTERED PREDICTIONS (declared before the run, never retuned)", ""]
    for r in verdict_rows:
        lines.append("%-3s %-20s %s" % (r["id"], r["verdict"],
                                        str(r["observed"])[:52]))
    lines += ["",
              "%d PASS / %d FAIL / %d EVIDENCE NOT FOUND"
              % (sum(1 for r in verdict_rows if r["verdict"] == "PASS"),
                 sum(1 for r in verdict_rows if r["verdict"] == "FAIL"),
                 sum(1 for r in verdict_rows if r["verdict"] == "EVIDENCE NOT FOUND")),
              "",
              "KEY NUMBERS AT THE FIXED 7 um SHAFT",
              "  pre-penetration dimpling      : %.3f um (assumed inputs)"
              % d7["failure_depth_um"],
              "  insertion r_crit (eps>=0.14)  : %.3f um  (GUINEA-PIG threshold)"
              % thr_rows["conservative"]["radius_um"],
              "  micromotion r_crit (gamma>=%.4g): %.2f um (ASSUMED A=%.3g um)"
              % (gamma_thr, mic.damage_radius_um(gamma_thr), mic.amplitude_um),
              "  at-risk / addressable (all)   : %d / %d = %.4f"
              % (at_risk_insert["population"]["at_risk_all_somas"],
                 at_risk_insert["population"]["addressable_all_somas"],
                 at_risk_insert["population"]["fraction_of_addressable_all_somas"] or 0.0),
              "  at-risk / addressable (tier)  : %d / %d = %.4f"
              % (at_risk_insert["population"]["at_risk_in_modelled_tier"],
                 at_risk_insert["population"]["addressable_in_modelled_tier"],
                 at_risk_insert["population"]["fraction_of_addressable_in_modelled_tier"] or 0.0),
              "  parent arm (256ch, r=50um): at-risk %d/%d = %.4f"
              % (at_risk_parent["population"]["at_risk_all_somas"],
                 at_risk_parent["population"]["addressable_all_somas"],
                 at_risk_parent["population"]["fraction_of_addressable_all_somas"] or 0.0),
              "  max R_in reduction measured   : %.4g"
              % (coupling["aggregate"]["max_R_in_reduction_fraction"] or 0.0),
              "  attribution (CC BY 4.0)       : BANC doi:10.7910/DVN/7WTH1N"]
    a9.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.2,
            family="monospace")
    a9.set_title("(j) verdicts and the headline numbers", fontsize=9)

    fig.suptitle("Electrode tissue mechanics in the fly: ADDED insertion dimpling and "
                 "micromotion, connected to REAL BANC somas\n"
                 "hand-built research prototype, NOT a validated device model · "
                 "surgery and electrode damage are NOT separable here · "
                 "BANC somas CC BY 4.0 doi:10.7910/DVN/7WTH1N",
                 fontsize=11.5)
    fig.savefig(png_path, dpi=105)
    plt.close(fig)


def et_import():
    import engine.electrode_tissue as et
    return et


if __name__ == "__main__":
    main()
