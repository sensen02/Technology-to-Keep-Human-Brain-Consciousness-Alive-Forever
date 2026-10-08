"""engine.electrode_damage -- mechanical damage from inserting a micrometre-scale
electrode into Drosophila nervous tissue.

HONEST SCOPE -- READ THIS BEFORE USING ANY NUMBER FROM THIS MODULE
------------------------------------------------------------------
There is **no quantitative measurement of electrode-insertion damage in any
insect CNS** (explicit negative result, `EXTERNAL_EVIDENCE_TOUCH_ELECTRODE.md`
section 2, item 1).  Every number that enters this module is therefore one of:

  ``measured``      quoted from that evidence file, always WITH its species and
                    citation.  All of them are rodent / guinea-pig / rat.
                    **None of them is a fly number.**
  ``derived``       pure arithmetic on registered parents; inherits the parents'
                    species, so a derived rodent number is still rodent.
  ``assumed`` /
  ``illustrative``  chosen by us because nothing is measurable.

Three mechanisms are combined, and each is labelled:

1. ``FlyBrainGeometry`` -- a 500 x 300 x 100 um box.  Even this is registered
   *illustrative*: the evidence file states the approximate scale but gives no
   primary citation for it, so it must not be called measured here.  Damage is
   reported as a FRACTION of this box, never as a "damage radius" imported from
   rodent histology.
2. ``CavityExpansionField`` -- plane-strain, incompressible cavity expansion:
   u(r) = delta*a/r, eps_r = -delta*a/r^2, eps_theta = +delta*a/r^2,
   sigma_rr = -2G*delta*a/r^2, sigma_tt = +2G*delta*a/r^2.  This is an
   **ASSUMPTION**: no measured strain field around an inserted probe exists
   (evidence file section 2, item 5).  Its analytic limits (1/r displacement,
   zero at infinity, continuity at the shaft wall, zero incompressibility
   residual, Lame equilibrium, traction at the wall) are all checked in
   ``run_electrode_damage_selftest.py``.
3. ``InjuryThresholds`` -- guinea-pig axonal strain 0.14/0.21/0.34, rat
   mechanoporation shear 140 dyn/cm^2 for 300 ms (the source itself states no
   death occurred at those parameters), rat small-pore resealing (~1 min) and
   guinea-pig transected-axon resealing (20 +/- 5 min).  All are used as
   **cross-species proxies** and are force-flagged as such.

SPECIES GUARD
-------------
``SpeciesGuard`` is constructed on ``DROSOPHILA_CNS``.  Building it without
``allow_cross=True`` **raises** ``ProvenanceError`` (via
``ParamRegistry.check_species``) listing every foreign-species measured/derived
parameter in the registry.  With ``allow_cross=True`` every use of such a
parameter comes back through ``SpeciesGuard.adopt`` as an ``Adopted`` record
with ``cross_species=True`` and ``forced_cross_species_flag=True``; the report
audit ``audit_species_labeling`` then refuses a report in which a rodent number
appears without that wrapper and its citation.

COUPLING (damage actually propagates into the existing models)
-------------------------------------------------------------
* membrane damage -> extra leak conductance at the affected cable compartments,
  applied only through the public ``CableNeuron.set_end_leak`` API
  (``cable.py`` is NOT modified);
* chemistry damage -> ``InjuryPotassium.apply_injury`` (membrane permeability /
  reservoir conductance) and ``InjuryLigand.switch_barrier`` (sheath/barrier
  conductance) from ``injury_tissue.py``;
* an OPTIONAL proximity -> reduced-drive coupling, **default OFF**; it is a
  drive-scaling placeholder, NOT an excitability model (the cable is passive).

WHAT THIS MODULE DOES NOT DO
----------------------------
No consciousness, viability, survival or medical claim.  No prediction of real
fly electrode damage.  Units: um, s, ms, mV, uS, nA, Pa, mM, nM (mM*um^3 and
nM*um^3 for amounts), exactly as the reused modules declare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve

from .cable import CableNeuron, Morphology
from .injury_tissue import InjuryLigand, InjuryPotassium, PotassiumConfig
from .params import (ASSUMED, DERIVED, ILLUSTRATIVE, MEASURED, Param,
                     ParamRegistry, ProvenanceError)
from .profile import DROSOPHILA_CNS, OrganismProfile, ProfileViolation

__all__ = [
    "FLY_SPECIES", "EVIDENCE_DOC", "CROSS_SPECIES_CAVEAT", "HONESTY_STATEMENTS",
    "ADDENDUM_TAG", "SHEATH_LOAD_BEARING_EVIDENCE",
    "NO_DROSOPHILA_LAMELLA_MEASUREMENT",
    "NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION",
    "NO_SECONDS_SCALE_SHEATH_RESEALING",
    "build_registry", "SpeciesGuard", "Adopted", "audit_species_labeling",
    "provenance_tier",
    "FlyBrainGeometry", "CavityExpansionField", "AffineDamageRadius",
    "SheathLayer", "PreparationVariant", "preparation_variants",
    "parenchymal_damage_estimate", "rodent_radius_transfer_table",
    "rodent_radius_records",
    "SheathRepairModel", "chemistry_config_for_variant",
    "circle_rectangle_overlap_area_um2", "injury_threshold_options",
    "STRAIN_THRESHOLD_CHOICES",
    "ResealingTimescales", "ProximityExcitabilityCoupling", "CableDamageConfig",
    "ChemistryDamageConfig", "affected_node_indices", "apply_membrane_damage",
    "apply_chemistry_coupling", "run_paired_cable", "run_paired_chemistry",
]

FLY_SPECIES = "Drosophila melanogaster"
EVIDENCE_DOC = "outputs/brain_isolation/EXTERNAL_EVIDENCE_TOUCH_ELECTRODE.md"

# Species strings are deliberately verbose: the guard compares them, and a
# report reader must never have to guess which animal a number came from.
SP_MOUSE = "Mus musculus (mouse)"
SP_GUINEA = "Cavia porcellus (guinea pig)"
SP_RAT = "Rattus norvegicus (rat)"
# insect species that are NOT the model organism: the species guard fires on
# these too, because "insect" is not "Drosophila".
SP_COCKROACH = "Periplaneta americana (cockroach)"
SP_MANDUCA = "Manduca sexta (tobacco hawkmoth)"
SP_LOCUST = "Locusta migratoria (locust)"
SP_INSECT = "insect (species not specified in the addendum)"

CIT_GUINEA_STRAIN = "10.1115/1.1324667 (guinea-pig optic nerve)"
CIT_RAT_MECHANOPORATION = "PMC7385830 (rat cortical culture)"
CIT_GUINEA_RESEAL = "10.1152/jn.2000.84.4.1763 (guinea-pig spinal cord, 37 C)"
CIT_MOUSE_PUNCTURE = "PMC13037841 (PNAS 2026; mouse, tungsten 7.5-100 um)"
CIT_MOUSE_HISTOLOGY = "PMC6546924 (mouse V1m; Michigan A16, 15 um thick x 123 um wide)"
CIT_RAT_SCAR = "PMC6965008 (rat striatum; polyimide 12 um x 380 um, 70 d)"
CIT_RAT_SPINE = "PMC10441615 (rat M1, 1 week)"
CIT_MOUSE_BBB = "PMC3164482 (mouse cortex, 30 min after insertion)"
CIT_FLY_PIPETTE = "PMC9884108 (Drosophila brain, ex vivo whole-cell)"

# --- insect data that arrived in a parent-agent ADDENDUM after this round ------
# It was reported to this agent second-hand mid-round (with citations, and with
# the explicit instruction to model the sheath as the load-bearing layer), and it
# is NOW recorded in section 2b of the authoritative evidence file.  Every
# parameter built from it carries ADDENDUM_TAG so a reader can see the chain of
# custody.  This agent did NOT independently re-verify these numbers.
EVIDENCE_FILE_NAME = "EXTERNAL_EVIDENCE_TOUCH_ELECTRODE.md"
ADDENDUM_TAG = ("arrived via the post-hoc parent addendum and is now recorded in "
                "section 2b of " + EVIDENCE_FILE_NAME + "; not re-verified by this agent")
DOI_LAMELLA_COCKROACH = "doi:10.2307/1539019 (Twarog & Roeder 1956)"
DOI_LAMELLA_MANDUCA = "doi:10.1007/s00359-025-01755-4"
DOI_DROSOPHILA_SHEATH = ("doi:10.1016/j.cris.2025.100113; "
                        "doi:10.3389/fnins.2014.00365")
DOI_SHEATH_REPAIR = ("doi:10.1523/JNEUROSCI.04-11-02689.1984; doi:10.1242/jcs.95.4.599")
DOI_SHEATH_REMOVED = "doi:10.7554/eLife.03293; doi:10.1016/j.cell.2025.11.040"
DOI_FLY_ELECTRODES = ("doi:10.1016/j.neuron.2020.06.022; doi:10.7554/eLife.81780; "
                     "doi:10.1016/j.neuron.2016.12.043")
DOI_INSECT_TIPS = "doi:10.1371/journal.pone.0010019; doi:10.1152/jn.00298.2010"
CIT_TWAROG_ROEDER = ("10.2307/1539019 (Twarog & Roeder 1956: nerve substance bulged "
                     "'almost explosively' out of a small sheath hole in hypotonic "
                     "saline -- the SHEATH, not the parenchyma, is load-bearing)")
SHEATH_LOAD_BEARING_EVIDENCE = (
    "Twarog & Roeder 1956, doi:10.2307/1539019: after a small hole was made in the "
    "sheath, the nerve substance bulged 'almost explosively' out of it under "
    "hypotonic saline. The load-bearing element is the sheath, NOT the parenchyma, "
    "so a rodent-style parenchymal kill-zone radius is the wrong primary insult.")
NO_DROSOPHILA_LAMELLA_MEASUREMENT = (
    "There is NO verified Drosophila neural-lamella thickness measurement. The "
    "commonly quoted '2-3 um' is the WHOLE surface-glia + ECM barrier, and the "
    "'2 um'/'1 um' figures in Stork 2008 are SCALE BARS, not measurements. No "
    "Drosophila lamella thickness is tagged measured anywhere in this module.")
NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION = (
    "No insect study quantifies electrode-insertion damage: no cell counts, no "
    "damage radius, no scar thickness, no track histology. This negative result "
    "was confirmed by two independent literature searches (evidence file section 2 "
    "item 1 and the parent-agent addendum item 6).")
NO_SECONDS_SCALE_SHEATH_RESEALING = (
    "There is NO seconds-scale sheath-resealing measurement in insects (negative "
    "result). Sheath repair is slow: phagocytes persist >1 month, perineurial "
    "proliferation peaks at 6-8 d and continues >=17 d (cockroach connective). "
    "Sheath resealing on a seconds timescale is therefore a FREE PARAMETER and is "
    "NOT applied in this model.")

CROSS_SPECIES_CAVEAT = (
    "跨物种借用：本模块用到的每一个 measured/derived 数值都来自哺乳动物"
    "（小鼠/大鼠/豚鼠）。没有任何昆虫 CNS 的电极损伤定量测量存在。这些数值"
    "只能作为函数形式或假设阈值使用，不得表述为适用于果蝇的实测值。")

HONESTY_STATEMENTS = (
    "No quantitative electrode-damage measurement exists for ANY insect CNS "
    "(no cell counts, no damage radius, no scar thickness, no track histology; "
    "confirmed by two independent literature searches).",
    "The displacement/strain field is an ASSUMED plane-strain incompressible "
    "cavity expansion; no measured strain field around a probe exists.",
    "The SHEATH (neural lamella + perineurium) is modelled as the load-bearing "
    "layer, because breaching it lets the nerve substance bulge out almost "
    "explosively (Twarog & Roeder 1956). Its fly thickness and its breach "
    "threshold are NOT measured: both are free parameters swept here.",
    "No Drosophila neural-lamella thickness is used as measured: the quoted "
    "'2-3 um' is whole surface-glia + ECM and the Stork 2008 2/1 um are scale "
    "bars; measured lamella values exist only for cockroach/Manduca/locust and "
    "are therefore cross-species too.",
    "Every threshold is a CROSS-SPECIES PROXY: strain 0.14/0.21/0.34 = "
    "guinea-pig optic nerve; shear 140 dyn/cm2 x 300 ms = rat cortical culture, "
    "and the source states NO DEATH occurred at those parameters; membrane "
    "resealing ~1 min = rat, 20+/-5 min = guinea pig.",
    "Sheath repair has NO seconds-scale measurement in insects; membrane "
    "resealing constants must NOT be reused for the sheath (they differ by "
    "~3 orders of magnitude in time).",
    "Force F=0.0346+10.07*d and compression dp=0.212+5.06*d are MOUSE numbers "
    "used as FUNCTION FORM only; bleeding thresholds (none at 15 um, always "
    ">=100 um) are mouse numbers, and the fly has no vertebrate vasculature.",
    "Damage is reported as a FRACTION of the 500x300x100 um fly brain; a "
    "rodent-style 100 um damage radius is 2x the whole fly brain thickness.",
    "The parenchymal damage radius is an UNMEASURED FREE PARAMETER with a "
    "sensitivity sweep, not a fitted or borrowed value.",
    "No consciousness, viability, survival or medical claim; this model does "
    "NOT predict real fly electrode damage.",
)

# keys in a report whose value must never be a bare number: any such quantity
# has to be a provenance wrapper carrying species + source (see the audit).
RODENT_GUARD_KEY_PATTERNS = (
    "rodent", "axonal_injury", "strain_threshold", "mechanoporation", "reseal",
    "bleed", "probe_force", "probe_compress", "mouse",
)

UNITS = {
    "length": "um", "strain": "1", "stress": "Pa", "force": "mN", "time": "s",
    "conductance": "uS", "leak_density": "S/cm^2", "volume": "um^3",
    "concentration": "mM", "ligand": "nM",
}


def _finite(value, name, minimum=None, strict=False):
    x = float(value)
    if not math.isfinite(x) or (minimum is not None and
                                (x <= minimum if strict else x < minimum)):
        raise ValueError(f"invalid {name}: {value!r}")
    return x


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


# ----------------------------------------------------------------------
# 1. provenance registry
# ----------------------------------------------------------------------
def build_registry(name="electrode_damage"):
    """Every number this module can use, with enforced provenance.

    `measured` requires a citation (enforced by ``engine.params.Param``), and
    every measured number here carries the species it was measured in.  Read
    `engine.params` for why an unlabelled number is treated as a bug.
    """
    r = ParamRegistry(name)

    # ---- measured, ALL cross-species (rodent / guinea pig / rat) --------
    r.add(Param("rodent_axonal_strain_conservative", 0.14, "1", MEASURED,
                source=CIT_GUINEA_STRAIN, species=SP_GUINEA,
                notes="conservative axonal injury strain threshold, guinea-pig "
                      "optic nerve. Used here as a MEMBRANE/AXONAL rupture proxy "
                      "for Drosophila: that transfer is an ASSUMPTION."))
    r.add(Param("rodent_axonal_strain_optimal", 0.21, "1", MEASURED,
                source=CIT_GUINEA_STRAIN, species=SP_GUINEA,
                notes="'optimal' axonal injury strain threshold, guinea-pig optic nerve."))
    r.add(Param("rodent_axonal_strain_permissive", 0.34, "1", MEASURED,
                source=CIT_GUINEA_STRAIN, species=SP_GUINEA,
                notes="'permissive' axonal injury strain threshold, guinea-pig optic nerve."))
    r.add(Param("rodent_mechanoporation_shear_dyn_cm2", 140.0, "dyn/cm^2", MEASURED,
                source=CIT_RAT_MECHANOPORATION, species=SP_RAT,
                notes="mechanoporation (shear) threshold, rat cortical culture. "
                      "The source explicitly states NO CELL DEATH occurred at "
                      "these parameters; it is a permeabilisation threshold, not "
                      "a death threshold."))
    r.add(Param("rodent_mechanoporation_duration_ms", 300.0, "ms", MEASURED,
                source=CIT_RAT_MECHANOPORATION, species=SP_RAT,
                notes="required load duration for the shear threshold above."))
    r.add(Param("rodent_reseal_small_pore_reduction_time_s", 60.0, "s", MEASURED,
                source=CIT_RAT_MECHANOPORATION, species=SP_RAT,
                notes="source reports a 4-fold decrease of permeabilised cells "
                      "within 1 min and return to control by 10 min."))
    r.add(Param("rodent_reseal_transected_tau_min", 20.0, "min", MEASURED,
                source=CIT_GUINEA_RESEAL, species=SP_GUINEA,
                notes="exponential resealing time constant of a TRANSECTED axon "
                      "at 37 C; blocked at <=25 C or [Ca]o <=0.5 mM."))
    r.add(Param("rodent_reseal_transected_tau_sd_min", 5.0, "min", MEASURED,
                source=CIT_GUINEA_RESEAL, species=SP_GUINEA,
                notes="reported spread of the transected-axon resealing constant."))
    r.add(Param("rodent_probe_force_intercept_mN", 0.0346, "mN", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE,
                notes="pia-puncture force fit intercept Fp[mN] = 0.0346 + 10.07*d[mm]."))
    r.add(Param("rodent_probe_force_slope_mN_per_mm", 10.07, "mN/mm", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE, notes="force fit slope."))
    r.add(Param("rodent_probe_compress_intercept_mm", 0.212, "mm", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE,
                notes="pre-puncture compression fit dp[mm] = 0.212 + 5.06*d[mm]."))
    r.add(Param("rodent_probe_compress_slope_mm_per_mm", 5.06, "1", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE, notes="compression fit slope."))
    r.add(Param("rodent_probe_force_15um_reported_uN", 185.0, "uN", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE, notes="source-quoted point."))
    r.add(Param("rodent_probe_force_15um_sd_uN", 40.0, "uN", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE, notes="source-quoted spread."))
    r.add(Param("rodent_probe_force_100um_reported_uN", 1610.0, "uN", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE, notes="source-quoted point."))
    r.add(Param("rodent_probe_force_100um_sd_uN", 239.0, "uN", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE, notes="source-quoted spread."))
    r.add(Param("rodent_bleed_absent_diameter_um", 15.0, "um", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE,
                notes="no bleeding in any 15 um case (mouse)."))
    r.add(Param("rodent_bleed_always_diameter_um", 100.0, "um", MEASURED,
                source=CIT_MOUSE_PUNCTURE, species=SP_MOUSE,
                notes="bleeding in every >=100 um case (mouse). Drosophila has no "
                      "vertebrate-style vasculature, so this row has no fly analogue."))
    r.add(Param("rodent_histology_neuron_loss_zone_um", 50.0, "um", MEASURED,
                source=CIT_MOUSE_HISTOLOGY, species=SP_MOUSE,
                notes="largest NeuN+ reduction at 0-50 um from the track, 1-28 d."))
    r.add(Param("rodent_histology_degeneration_um", 150.0, "um", MEASURED,
                source=CIT_MOUSE_HISTOLOGY, species=SP_MOUSE,
                notes="significantly elevated NeuN+Casp3+ up to 150 um, 28 d."))
    r.add(Param("rodent_histology_device_thickness_um", 15.0, "um", MEASURED,
                source=CIT_MOUSE_HISTOLOGY, species=SP_MOUSE, notes="Michigan A16 thickness."))
    r.add(Param("rodent_histology_device_width_um", 123.0, "um", MEASURED,
                source=CIT_MOUSE_HISTOLOGY, species=SP_MOUSE, notes="Michigan A16 width."))
    r.add(Param("rodent_glial_scar_fwhm_um", 129.0, "um", MEASURED,
                source=CIT_RAT_SCAR, species=SP_RAT, notes="GFAP FWHM, 70 d."))
    r.add(Param("rodent_glial_scar_fwhm_sd_um", 10.0, "um", MEASURED,
                source=CIT_RAT_SCAR, species=SP_RAT, notes="reported spread (113-150 um)."))
    r.add(Param("rodent_spine_loss_um", 100.0, "um", MEASURED,
                source=CIT_RAT_SPINE, species=SP_RAT,
                notes="~50% spine-density loss within 100 um of the device."))
    r.add(Param("rodent_bbb_damage_area_targeted_um2", 57000.0, "um^2", MEASURED,
                source=CIT_MOUSE_BBB, species=SP_MOUSE, notes="track aimed at a large vessel."))
    r.add(Param("rodent_bbb_damage_area_avoided_um2", 9300.0, "um^2", MEASURED,
                source=CIT_MOUSE_BBB, species=SP_MOUSE, notes="track avoiding vessels."))
    r.add(Param("rodent_bbb_damage_reduction_percent", 82.8, "%", MEASURED,
                source=CIT_MOUSE_BBB, species=SP_MOUSE, notes="reduction by avoiding vessels."))
    r.add(Param("rodent_bbb_damage_reduction_sd_percent", 14.3, "%", MEASURED,
                source=CIT_MOUSE_BBB, species=SP_MOUSE, notes="reported spread."))

    # ---- measured, SAME species: the only Drosophila measured electrode row --
    r.add(Param("fly_patch_pipette_outer_diameter_mm", 1.5, "mm", MEASURED,
                source=CIT_FLY_PIPETTE, species=FLY_SPECIES,
                notes="borosilicate SHANK outer diameter for fly whole-cell "
                      "recording. The source does NOT give the pulled tip "
                      "diameter, so this cannot be used as a shaft diameter."))
    r.add(Param("fly_patch_pipette_inner_diameter_mm", 1.12, "mm", MEASURED,
                source=CIT_FLY_PIPETTE, species=FLY_SPECIES,
                notes="shank inner diameter; tip diameter not stated in the source."))

    # ---- measured, PARENT-ADDENDUM insect sheath/electrode data --------------
    # Every source string below carries ADDENDUM_TAG.  Note carefully: the
    # lamella values are cockroach / Manduca / locust measurements, so they are
    # ALSO cross-species for a Drosophila model, and the guard fires on them.
    add = f"{ADDENDUM_TAG}; "
    r.add(Param("insect_lamella_thickness_cockroach_um_lo", 2.0, "um", MEASURED,
                source=add + f"cockroach neural lamella 2-5 um, perineurium 1-3 um ({DOI_LAMELLA_COCKROACH})",
                species=SP_COCKROACH,
                notes="directly measured in cockroach. CROSS-SPECIES for a fly "
                      "model: the guard flags it."))
    r.add(Param("insect_lamella_thickness_cockroach_um_hi", 5.0, "um", MEASURED,
                source=add + f"cockroach neural lamella 2-5 um, perineurium 1-3 um ({DOI_LAMELLA_COCKROACH})",
                species=SP_COCKROACH, notes="upper end of the measured cockroach range."))
    r.add(Param("insect_perineurium_thickness_cockroach_um_lo", 1.0, "um", MEASURED,
                source=add + f"cockroach perineurium 1-3 um ({DOI_LAMELLA_COCKROACH})",
                species=SP_COCKROACH,
                notes="the perineurium is a distinct layer under the lamella."))
    r.add(Param("insect_perineurium_thickness_cockroach_um_hi", 3.0, "um", MEASURED,
                source=add + f"cockroach perineurium 1-3 um ({DOI_LAMELLA_COCKROACH})",
                species=SP_COCKROACH, notes="upper end of the measured range."))
    r.add(Param("insect_lamella_thickness_manduca_um_lo", 4.0, "um", MEASURED,
                source=add + f"Manduca neck connective 4-5 um by TEM ({DOI_LAMELLA_MANDUCA})",
                species=SP_MANDUCA, notes="TEM measurement, neck connective."))
    r.add(Param("insect_lamella_thickness_manduca_um_hi", 5.0, "um", MEASURED,
                source=add + f"Manduca neck connective 4-5 um by TEM ({DOI_LAMELLA_MANDUCA})",
                species=SP_MANDUCA, notes="'on par with locust ~5 um' per the addendum."))
    r.add(Param("insect_lamella_thickness_locust_um", 5.0, "um", MEASURED,
                source=add + f"locust ~5 um, quoted as 'on par' with Manduca ({DOI_LAMELLA_MANDUCA})",
                species=SP_LOCUST, notes="approximate value as reported in the addendum."))
    r.add(Param("insect_sheath_perineurial_proliferation_peak_days", 7.0, "day", MEASURED,
                source=add + f"cockroach connective, perineurial proliferation peaks 6-8 d "
                             f"({DOI_SHEATH_REPAIR})", species=SP_COCKROACH,
                notes="midpoint 7 d of a reported 6-8 d peak; the addendum gives "
                      "the range, not a point estimate."))
    r.add(Param("insect_sheath_repair_persists_days", 17.0, "day", MEASURED,
                source=add + f"cockroach connective, perineurial proliferation continues "
                             f">=17 d ({DOI_SHEATH_REPAIR})", species=SP_COCKROACH,
                notes="lower bound on the repair duration."))
    r.add(Param("insect_sheath_phagocyte_persistence_days", 30.0, "day", MEASURED,
                source=add + f"cockroach connective, phagocytes persist >1 month ({DOI_SHEATH_REPAIR})", species=SP_COCKROACH,
                notes="reported as '>1 month'; 30 d is used as a LOWER BOUND."))
    r.add(Param("insect_tungsten_tip_diameter_um", 1.0, "um", MEASURED,
                source=add + f"the only insect electrode TIP diameter in um cited in the "
                             f"addendum: a tungsten 1 um tip at 1 Mohm ({DOI_INSECT_TIPS})",
                species=SP_INSECT,
                notes="one of only two insect tip diameters found in um at all."))
    r.add(Param("insect_protease_pipette_broken_diameter_um", 10.0, "um", MEASURED,
                source=add + f"a protease (collagenase) pipette broken to ~10 um ({DOI_INSECT_TIPS})", species=SP_INSECT,
                notes="the second, and the only other, insect tip size in um."))
    r.add(Param("rodent_15um_as_percent_of_fly_brain_width", 5.0, "%", DERIVED,
                derived_from=["rodent_bleed_absent_diameter_um", "fly_brain_width_um"],
                species=SP_MOUSE,
                notes="15/300*100 = 5%: the addendum's scale statement. It mixes a "
                      "rodent diameter with fly geometry and is used ONLY as a scale "
                      "illustration, so it is flagged cross-species."))

    r.add(Param("fly_surface_glia_thickness_um_max", 1.0, "um", MEASURED,
                source=add + "sub-epidermal glia <1 um, quoted WITH the warning that "
                             f"no reliable Drosophila lamella measurement exists ({DOI_DROSOPHILA_SHEATH})",
                species=FLY_SPECIES,
                notes="the only Drosophila-species sheath-ADJACENT number available, and "
                      "it is an UPPER BOUND from a '<1 um' statement, not a lamella "
                      "thickness. It is NOT used as the sheath thickness."))

    # ---- measured, SAME species: Drosophila electrode sizes ------------------
    r.add(Param("fly_brain_patch_resistance_Mohm_lo", 5.0, "MOhm", MEASURED,
                source=add + f"fly brain patch 5-15 MOhm ({DOI_FLY_ELECTRODES})",
                species=FLY_SPECIES,
                notes="Drosophila electrode resistance. The addendum gives NO tip "
                      "diameter for it, so it cannot be converted into a shaft size."))
    r.add(Param("fly_brain_patch_resistance_Mohm_hi", 15.0, "MOhm", MEASURED,
                source=add + f"fly brain patch 5-15 MOhm ({DOI_FLY_ELECTRODES})",
                species=FLY_SPECIES, notes="upper end of the fly brain patch range."))
    r.add(Param("fly_al_mb_resistance_Mohm_lo", 3.0, "MOhm", MEASURED,
                source=add + f"fly antennal lobe / mushroom body 3-5 MOhm ({DOI_FLY_ELECTRODES})", species=FLY_SPECIES,
                notes="lower resistance than the brain patch range, i.e. a blunter "
                      "tip by the usual inverse relation; no diameter is given."))
    r.add(Param("fly_al_mb_resistance_Mohm_hi", 5.0, "MOhm", MEASURED,
                source=add + f"fly antennal lobe / mushroom body 3-5 MOhm ({DOI_FLY_ELECTRODES})", species=FLY_SPECIES,
                notes="upper end of the AL/MB range."))
    r.add(Param("fly_vnc_sharp_resistance_Mohm_lo", 80.0, "MOhm", MEASURED,
                source=add + f"fly VNC sharp electrode 80-100 MOhm ({DOI_FLY_ELECTRODES})",
                species=FLY_SPECIES,
                notes="the finest of the three fly electrodes by resistance; its "
                      "tip diameter is again not given."))
    r.add(Param("fly_vnc_sharp_resistance_Mohm_hi", 100.0, "MOhm", MEASURED,
                source=add + f"fly VNC sharp electrode 80-100 MOhm ({DOI_FLY_ELECTRODES})",
                species=FLY_SPECIES, notes="upper end of the VNC sharp range."))

    # ---- derived (pure arithmetic; species inherited from the parents) -----
    r.add(Param("rodent_mechanoporation_shear_Pa", 14.0, "Pa", DERIVED,
                derived_from=["rodent_mechanoporation_shear_dyn_cm2"],
                species=SP_RAT,
                notes="1 dyn/cm^2 = 0.1 Pa exactly, so 140 dyn/cm^2 = 14 Pa. "
                      "Conversion only, no new information."))
    r.add(Param("rodent_reseal_small_pore_tau_s", 60.0 / math.log(4.0), "s", DERIVED,
                derived_from=["rodent_reseal_small_pore_reduction_time_s"],
                species=SP_RAT,
                notes="ASSUMES single-exponential resealing with a 4-fold decrease "
                      "in 60 s; the source only states 'within 1 min' and 'back to "
                      "control by 10 min'. The exponential form is our assumption."))
    r.add(Param("rodent_reseal_transected_tau_s", 1200.0, "s", DERIVED,
                derived_from=["rodent_reseal_transected_tau_min"], species=SP_GUINEA,
                notes="20 min -> 1200 s, no modelling content."))
    r.add(Param("rodent_reseal_transected_tau_sd_s", 300.0, "s", DERIVED,
                derived_from=["rodent_reseal_transected_tau_sd_min"], species=SP_GUINEA,
                notes="5 min -> 300 s, no modelling content."))
    r.add(Param("rodent_probe_force_15um_fit_uN", 185.65, "uN", DERIVED,
                derived_from=["rodent_probe_force_intercept_mN",
                              "rodent_probe_force_slope_mN_per_mm"], species=SP_MOUSE,
                notes="(0.0346 + 10.07*0.015) mN = 0.18565 mN -> 185.65 uN."))
    r.add(Param("rodent_probe_force_100um_fit_uN", 1041.6, "uN", DERIVED,
                derived_from=["rodent_probe_force_intercept_mN",
                              "rodent_probe_force_slope_mN_per_mm"], species=SP_MOUSE,
                notes="(0.0346 + 10.07*0.1) mN = 1.0416 mN -> 1041.6 uN."))
    r.add(Param("rodent_probe_compress_15um_fit_um", 287.9, "um", DERIVED,
                derived_from=["rodent_probe_compress_intercept_mm",
                              "rodent_probe_compress_slope_mm_per_mm"], species=SP_MOUSE,
                notes="(0.212 + 5.06*0.015) mm = 0.2879 mm -> 287.9 um of "
                      "pre-puncture surface compression."))
    r.add(Param("rodent_probe_compress_100um_fit_um", 718.0, "um", DERIVED,
                derived_from=["rodent_probe_compress_intercept_mm",
                              "rodent_probe_compress_slope_mm_per_mm"], species=SP_MOUSE,
                notes="(0.212 + 5.06*0.1) mm = 0.718 mm -> 718 um."))
    r.add(Param("fit_vs_reported_force_15um_rel_error", 0.0035135, "1", DERIVED,
                derived_from=["rodent_probe_force_15um_fit_uN",
                              "rodent_probe_force_15um_reported_uN"], species=SP_MOUSE,
                notes="|185.65-185|/185. The linear fit reproduces the source's "
                      "own 15 um point to 0.35%."))
    r.add(Param("fit_vs_reported_force_100um_rel_error", 0.352795, "1", DERIVED,
                derived_from=["rodent_probe_force_100um_fit_uN",
                              "rodent_probe_force_100um_reported_uN"], species=SP_MOUSE,
                notes="|1041.6-1610|/1610 = 35%. UNRESOLVED INTERNAL INCONSISTENCY "
                      "in the evidence file: the quoted linear fit does not pass "
                      "through the quoted 100 um point. We do not explain it away."))
    r.add(Param("fit_vs_reported_force_100um_sd_units", 2.3782, "1", DERIVED,
                derived_from=["rodent_probe_force_100um_fit_uN",
                              "rodent_probe_force_100um_reported_uN",
                              "rodent_probe_force_100um_sd_uN"], species=SP_MOUSE,
                notes="(1610-1041.6)/239 = 2.38 sigma: the discrepancy is larger "
                      "than the quoted spread, so it is not explained by scatter "
                      "reported in the same row."))

    # ---- illustrative: fly-scale geometry (no primary citation in the file) --
    r.add(Param("fly_brain_length_um", 500.0, "um", ILLUSTRATIVE, species=FLY_SPECIES,
                notes="Approximate adult fly brain long axis. The evidence file "
                      "states 'about 500x300x100 um' but gives no primary citation "
                      "for it, so it is registered ILLUSTRATIVE, not measured."))
    r.add(Param("fly_brain_width_um", 300.0, "um", ILLUSTRATIVE, species=FLY_SPECIES,
                notes="same source and same caveat as fly_brain_length_um."))
    r.add(Param("fly_brain_thickness_um", 100.0, "um", ILLUSTRATIVE, species=FLY_SPECIES,
                notes="same source and same caveat; this is the dimension a "
                      "rodent-style 100 um damage radius destroys completely."))
    r.add(Param("mouse_cortical_thickness_um", 900.0, "um", ILLUSTRATIVE, species=SP_MOUSE,
                notes="Quoted for scale comparison in the evidence file section 3 "
                      "WITHOUT a citation, so ILLUSTRATIVE here; used only for the "
                      "thickness-ratio comparison, never as a model parameter."))

    # ---- assumed / illustrative: model choices with no measurement ----------
    r.add(Param("assumed_shaft_diameter_um", 10.0, "um", ASSUMED, species=SP_MOUSE,
                range=(1.0, 50.0),
                notes="The smallest diameter in the ONLY systematic measured series "
                      "(7.5-100 um tungsten, mouse). Used as the primary demo shaft "
                      "because its force/displacement data exist; it is NOT a "
                      "recommended fly electrode and NOT a fly measurement.  "
                      "NOTE: the user has since FIXED the simulated electrode at 7 um, "
                      "so this entry is retained only as the origin of the field."))
    r.add(Param("assumed_glass_tip_diameter_um", 1.0, "um", ASSUMED, species=None,
                range=(0.2, 10.0),
                notes="Assumed pulled-glass tip diameter for the second demo case. "
                      "The fly electrophysiology source gives the 1.5 mm shank but "
                      "NOT the tip, so this is a pure assumption."))
    r.add(Param("assumed_insertion_depth_um", 100.0, "um", ASSUMED, species=FLY_SPECIES,
                range=(20.0, 300.0),
                notes="assumed full-thickness penetration of the fly brain."))
    r.add(Param("assumed_wall_displacement_fraction", 0.5, "1", ASSUMED, species=None,
                range=(0.05, 1.0),
                notes="delta/a: radial displacement imposed on the tissue at the "
                      "shaft wall, as a fraction of the shaft radius. NO measured "
                      "strain field around a probe exists; this is the single most "
                      "influential assumption in the damage radius."))
    r.add(Param("assumed_tissue_shear_modulus_Pa", 500.0, "Pa", ASSUMED, species=None, range=(50.0, 5000.0),
                notes="Shear modulus of fly CNS tissue. NOT measured anywhere in "
                      "this project's verified records. Used only to turn the "
                      "assumed strain field into an assumed stress for the "
                      "mechanoporation proxy (radius ~ sqrt(G))."))
    r.add(Param("assumed_damaged_leak_density_S_cm2", 0.5, "S/cm^2", ASSUMED, species=None, range=(0.001, 2.0),
                notes="Extra leak conductance density of a ruptured membrane patch. "
                      "Intact cable leak used in the demo is 1e-4 S/cm^2, so this is "
                      "5000x the intact leak; it is a free parameter and the cable "
                      "effect scales with it."))
    r.add(Param("assumed_damaged_permeability_um_s", 0.2, "um/s", ASSUMED, species=None, range=(0.01, 5.0),
                notes="K membrane permeability after damage, fed to "
                      "InjuryPotassium.apply_injury. The intact value in the paired "
                      "run is exactly 0, so the sham has no membrane exchange at all."))
    r.add(Param("assumed_sheath_conductance_damaged_um3_s", 5.0, "um^3/s", ASSUMED,
                species=None,
                range=(0.1, 50.0),
                notes="Sheath/barrier ligand conductance after damage, fed to "
                      "InjuryLigand.switch_barrier. The sham uses exactly 0 "
                      "(idealised fully tight sheath, range=(0.1, 50.0)). For Drosophila this is a "
                      "glial/tracheal sheath restriction, NOT a vertebrate BBB."))
    r.add(Param("assumed_proximity_full_effect_radius_um", 5.0, "um", ASSUMED, species=None, range=(1.0, 20.0),
                notes="OPTIONAL proximity->drive coupling, default OFF."))
    r.add(Param("assumed_proximity_max_drive_attenuation", 0.5, "1", ASSUMED, species=None, range=(0.0, 1.0),
                notes="OPTIONAL proximity->drive coupling, default OFF. This is a "
                      "drive-scaling placeholder, NOT excitability."))
    r.add(Param("assumed_affine_damage_offset_um", 20.0, "um", ASSUMED, species=None, range=(1.0, 100.0),
                notes="Offset of the AFFINE damage-radius alternative. Chosen only "
                      "to be smaller than the rodent 0-50 um zone and to make the "
                      "size dependence visible; not fitted to anything."))
    r.add(Param("assumed_affine_damage_slope", 0.5, "1", ASSUMED, species=None, range=(0.1, 2.0),
                notes="Slope of the affine alternative, pure assumption."))

    # ---- addendum-driven sheet/sheath model (every value ASSUMED) -----------
    r.add(Param("fly_sheath_thickness_um", 3.0, "um", ASSUMED, species=FLY_SPECIES, range=(0.5, 10.0),
                notes="Free parameter standing in for a Drosophila neural-lamella + "
                      "perineurium thickness. NOT measured: there is no verified "
                      "Drosophila lamella thickness, the quoted '2-3 um' is the "
                      "whole surface-glia+ECM barrier and Stork 2008's 2/1 um are "
                      "scale bars. 3 um sits inside the measured CROSS-SPECIES "
                      "cockroach range (2-5 um) only for scale."))
    r.add(Param("assumed_sheath_critical_opening_radius_um", 2.0, "um", ASSUMED,
                species=None, range=(0.1, 20.0),
                notes="PRIMARY sheath breach criterion: a rigid shaft tears / "
                      "delaminates the load-bearing sheet once its radius exceeds "
                      "this opening radius. NO insect measurement exists (negative "
                      "result), so this is a free parameter swept over 0.5-5 um; "
                      "2.0 um is only a default inside that sweep."))
    r.add(Param("assumed_sheath_delamination_multiple", 2.0, "1", ASSUMED, species=None, range=(1.0, 5.0),
                notes="Sheath delamination radius = critical_opening_radius + this "
                      "multiple * sheath thickness. Pure assumption."))
    r.add(Param("assumed_sheath_herniation_multiplier", 5.0, "1", ASSUMED, species=None,
                range=(1.0, 50.0),
                notes="Multiplier on the parenchymal damage radius used ONLY for the "
                      "sheath-INTACT variant once the sheath is breached: losing the "
                      "load-bearing layer is a discontinuous change in the boundary "
                      "condition (Twarog & Roeder 1956), and no measurement sizes its "
                      "parenchymal consequence. Swept over 1-50x, never fitted."))
    r.add(Param("assumed_parenchymal_free_radius_um", 10.0, "um", ASSUMED, species=None, range=(1.0, 50.0),
                notes="Free-parameter parenchymal damage radius used in the "
                      "sensitivity sweep. Explicitly UNMEASURED for insects."))
    r.add(Param("assumed_sheath_repair_tau_s", 1296000.0, "s", ASSUMED, species=None, range=(86400.0, 2592000.0),
                notes="15 d expressed in s, a free parameter for sheath repair; it is "
                      "NOT a measured time constant. The only measured anchors are "
                      "the cockroach repair durations (>=17 d, phagocytes >1 month). "
                      "No seconds-scale sheath resealing is applied anywhere."))

    r.validate()
    return r


# ----------------------------------------------------------------------
# 2. species guard
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Adopted:
    """One parameter's value plus the provenance a report must carry."""
    name: str
    value: float
    unit: str
    provenance: str
    species: str | None
    source: str | None
    cross_species: bool
    forced_flag: bool
    note: str = ""

    def to_dict(self):
        return {"value": self.value, "unit": self.unit,
                "provenance": self.provenance, "species": self.species,
                "source": self.source, "cross_species": bool(self.cross_species),
                "forced_cross_species_flag": bool(self.forced_flag),
                "parameter": self.name, "note": self.note}

    def label(self):
        """Short human label used on figures."""
        if self.cross_species:
            return f"{self.value:g} {self.unit} [CROSS-SPECIES {self.species}]"
        return f"{self.value:g} {self.unit} [{self.provenance}" + \
               (f" {self.species}]" if self.species else "]")


class SpeciesGuard:
    """Refuses, or force-flags, foreign-species numbers in a fly model.

    ``allow_cross=False`` (default) raises ``ProvenanceError`` from
    ``ParamRegistry.check_species``.  ``allow_cross=True`` keeps the offenders in
    ``forced_flags`` and every ``adopt`` of such a parameter returns
    ``cross_species=True`` -- a flag the report audit then requires.
    """

    def __init__(self, profile= DROSOPHILA_CNS, registry=None, allow_cross=False):
        if not isinstance(profile, OrganismProfile):
            raise TypeError("profile must be an engine.profile.OrganismProfile")
        if profile.species != FLY_SPECIES:
            raise ProfileViolation(
                f"engine.electrode_damage models Drosophila only; got profile "
                f"{profile.name!r} for species {profile.species!r}")
        self.profile = profile
        self.registry = registry if registry is not None else build_registry()
        self.allow_cross = bool(allow_cross)
        # Raises ProvenanceError here when allow_cross=False: this IS the guard.
        self.offenders = self.registry.check_species(profile.species,
                                                    allow_cross=self.allow_cross)
        self.foreign_names = tuple(sorted(n for n, _ in self.offenders))
        self.caveats = list(profile.required_caveats())
        if self.foreign_names:
            self.caveats.append(CROSS_SPECIES_CAVEAT)

    def adopt(self, name):
        snapshot = self.registry.snapshot()
        if name not in snapshot:
            raise KeyError(f"refusing to use undeclared parameter {name!r}")
        d = snapshot[name]
        provenance = d["provenance"]
        species = d["species"]
        source = d["source"]
        if provenance == DERIVED and not source:
            # a derived number inherits its citation from its parents; spell that
            # out rather than leaving the report with an uncited number
            trace = []
            for parent in d["derived_from"]:
                if parent not in snapshot:
                    raise ProvenanceError(f"{name}: parent {parent!r} is not registered")
                parent_source = snapshot[parent]["source"] or "no citation"
                trace.append(f"{parent} [{parent_source}]")
            source = "derived from " + "; ".join(trace)
        foreign = bool(species and species != self.profile.species
                       and provenance in (MEASURED, DERIVED))
        if foreign and not self.allow_cross:
            raise ProvenanceError(
                f"{name}: measured in {species!r} but the model species is "
                f"{self.profile.species!r}. Pass allow_cross=True and carry the "
                f"cross-species flag into the report, or do not use this number.")
        if provenance in (MEASURED, DERIVED) and not source:
            raise ProvenanceError(f"{name}: {provenance} without a citation")
        return Adopted(name=name, value=float(d["value"]), unit=d["unit"],
                       provenance=provenance, species=species, source=source,
                       cross_species=foreign, forced_flag=bool(foreign and self.allow_cross),
                       note=d["notes"])

    def adopted_record(self, adopted):
        """Verify a record really is flagged, then return its report dict."""
        if adopted.cross_species and not (adopted.forced_flag and adopted.source):
            raise ProvenanceError(
                f"{adopted.name}: cross-species use must carry a forced flag and "
                f"a citation")
        return adopted.to_dict()

    def foreign_parameters_used(self, adopted_list):
        used = sorted(a.name for a in adopted_list if a.cross_species)
        unflagged = [n for n in used if n not in self.foreign_names]
        if unflagged:
            raise ProvenanceError(f"foreign parameters not in the offender list: {unflagged}")
        return used

    def record(self, adopted_list=()):
        return {
            "profile": self.profile.name,
            "model_species": self.profile.species,
            "allow_cross": self.allow_cross,
            "n_foreign_parameters_in_registry": len(self.foreign_names),
            "foreign_parameters_in_registry": list(self.foreign_names),
            "foreign_parameters_used": self.foreign_parameters_used(adopted_list),
            "guard_default_would_refuse": True,
            "caveats": self.caveats,
        }


def audit_species_labeling(obj, path="$"):
    """Return the list of species-labelling violations in a report object.

    Rules:
      A. any provenance wrapper with a foreign species must set
         ``cross_species: true`` and carry a non-empty ``source``;
      B. a ``measured`` wrapper without a source is a violation;
      C. any key matching ``RODENT_GUARD_KEY_PATTERNS`` must not hold a bare
         number -- a rodent-derived quantity has to be a wrapper with species
         and citation, never a naked float that a reader would take as a fly
         number.  This rule is a HEURISTIC over key names: it catches the keys
         whose names announce their origin, and it cannot catch a rodent number
         stored under a neutral key name, so the report also carries the full
         provenance table.
    """
    bad = []

    def walk(node, path):
        if isinstance(node, dict):
            is_wrapper = "value" in node and "provenance" in node
            if is_wrapper:
                provenance, species = node.get("provenance"), node.get("species")
                source = node.get("source")
                foreign = bool(species and species != FLY_SPECIES
                               and provenance in (MEASURED, DERIVED))
                if foreign and node.get("cross_species") is not True:
                    bad.append(f"{path}: foreign-species {provenance} value from "
                               f"{species!r} is not flagged cross_species=true")
                if foreign and not source:
                    bad.append(f"{path}: foreign-species {provenance} value without a source")
                if provenance == MEASURED and not source:
                    bad.append(f"{path}: measured value without a citation")
            for key, value in node.items():
                if isinstance(key, str) and any(p in key.lower()
                                                for p in RODENT_GUARD_KEY_PATTERNS):
                    if isinstance(value, (bool, int, float, np.floating, np.integer)):
                        bad.append(f"{path}.{key}: bare number {value!r} under a "
                                   f"rodent/cross-species key; it must be a "
                                   f"provenance wrapper with species and citation")
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(obj, path)
    return bad


# ----------------------------------------------------------------------
# 3. fly-scaled geometry
# ----------------------------------------------------------------------
def circle_rectangle_overlap_area_um2(radius_um, center_x_um, center_y_um,
                                      width_um, height_um):
    """Exact (quadrature, kink-split) area of disc n rectangle, in um^2.

    Used instead of a pixel grid so that damage fractions are reproducible and
    so that a grid refinement can be used as an independent cross-check.
    """
    R = _finite(radius_um, "radius_um", 0.0)
    cx = _finite(center_x_um, "center_x_um")
    cy = _finite(center_y_um, "center_y_um")
    w = _finite(width_um, "width_um", 0.0, True)
    h = _finite(height_um, "height_um", 0.0, True)
    if R == 0.0:
        return 0.0
    x_lo, x_hi = max(cx - R, 0.0), min(cx + R, w)
    if x_hi <= x_lo:
        return 0.0
    th_lo = math.asin(max(-1.0, min(1.0, (x_lo - cx) / R)))
    th_hi = math.asin(max(-1.0, min(1.0, (x_hi - cx) / R)))
    breaks = [th_lo, th_hi]
    for c in ((h - cy) / R, cy / R):          # top and bottom clipping
        if -1.0 < c < 1.0:
            t = math.acos(c)
            for cand in (-t, t):
                if th_lo < cand < th_hi:
                    breaks.append(cand)
    breaks = sorted(breaks)

    def chord(theta):
        half = R * math.cos(theta)
        top = min(cy + half, h)
        bottom = max(cy - half, 0.0)
        return max(top - bottom, 0.0) * R * math.cos(theta)

    total = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a <= 1e-15:
            continue
        total += quad(chord, a, b, epsabs=1e-12, epsrel=1e-12, limit=120)[0]
    return float(total)


@dataclass(frozen=True)
class FlyBrainGeometry:
    """A rectangular fly-brain box; damage is reported as a fraction of it."""
    length_um: float = 500.0
    width_um: float = 300.0
    thickness_um: float = 100.0

    def __post_init__(self):
        for name in ("length_um", "width_um", "thickness_um"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, True))

    @classmethod
    def from_adopt(cls, adopt):
        return cls(adopt("fly_brain_length_um").value,
                   adopt("fly_brain_width_um").value,
                   adopt("fly_brain_thickness_um").value)

    @property
    def volume_um3(self):
        return self.length_um * self.width_um * self.thickness_um

    @property
    def footprint_um2(self):
        return self.length_um * self.width_um

    def track_area_um2(self, radius_um):
        """Cross-sectional area of a centred cylindrical damage zone."""
        return circle_rectangle_overlap_area_um2(radius_um, self.length_um / 2,
                                                 self.width_um / 2,
                                                 self.length_um, self.width_um)

    def track_damage_volume_um3(self, radius_um, depth_um=None):
        """Cylinder n brain box (exact lateral clip) + hemispherical tip cap.

        The cap is added only for a partial-depth track (a through-track leaves
        the tissue, so it has no distal cap).  The cap is not laterally clipped;
        that is documented as a small approximation and never matters for the
        fly-scaled radii used here (they are <1 um to a few um).
        """
        R = _finite(radius_um, "radius_um", 0.0)
        depth = self.thickness_um if depth_um is None else _finite(depth_um, "depth_um", 0.0, True)
        if depth > self.thickness_um:
            raise ValueError(
                f"insertion depth {depth} um exceeds the fly brain thickness "
                f"{self.thickness_um} um; a track cannot damage tissue it is not in")
        if R == 0.0:
            return 0.0
        volume = self.track_area_um2(R) * depth
        if depth < self.thickness_um:
            volume += (2.0 / 3.0) * math.pi * R ** 3
        return float(volume)

    def damage_fraction(self, radius_um, depth_um=None):
        fraction = self.track_damage_volume_um3(radius_um, depth_um) / self.volume_um3
        return float(min(1.0, max(0.0, fraction)))

    def thickness_ratio(self, radius_um):
        """2R / thickness: 2.0 means the zone spans the whole brain thickness."""
        return 2.0 * _finite(radius_um, "radius_um", 0.0) / self.thickness_um

    def footprint_fraction(self, radius_um):
        return self.track_area_um2(radius_um) / self.footprint_um2

    def to_dict(self):
        return {"length_um": self.length_um, "width_um": self.width_um,
                "thickness_um": self.thickness_um, "volume_um3": self.volume_um3,
                "shape": "rectangular box; NOT a segmented brain reconstruction"}


# ----------------------------------------------------------------------
# 4. damage field (ASSUMED)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CavityExpansionField:
    """ASSUMED plane-strain incompressible cavity expansion around a shaft.

    With a prescribed radial wall displacement ``delta`` at a shaft of radius
    ``a``, local area preservation forces ``d(r*u)/dr = 0``, so::

        u(r)     = delta*a/r                (1/r decay; -> 0 as r -> inf)
        eps_r    = du/dr    = -delta*a/r^2
        eps_th   = u/r      = +delta*a/r^2  (tension)
        gamma    = eps_th-eps_r = 2*delta*a/r^2
        sigma_rr = -2G*delta*a/r^2, sigma_tt = +2G*delta*a/r^2
        wall pressure p = 2G*delta/a, traction-free/consistency checked.

    This is NOT a measured strain field: none exists around an inserted probe
    (evidence file section 2, item 5).  It is linear-elastic, small-strain and
    plane-strain, none of which is verified for insect CNS.  The solution is
    defined only for r >= a; inside the shaft it is refused instead of clamped.
    """

    shaft_radius_um: float
    wall_displacement_fraction: float = 0.5
    shear_modulus_Pa: float = 500.0

    def __post_init__(self):
        object.__setattr__(self, "shaft_radius_um",
                           _finite(self.shaft_radius_um, "shaft_radius_um", 0.0, True))
        f = _finite(self.wall_displacement_fraction, "wall_displacement_fraction", 0.0, True)
        if f > 1.0:
            raise ValueError("wall_displacement_fraction > 1 exceeds the "
                             "small-strain range this field is derived for")
        object.__setattr__(self, "wall_displacement_fraction", f)
        object.__setattr__(self, "shear_modulus_Pa",
                           _finite(self.shear_modulus_Pa, "shear_modulus_Pa", 0.0, True))

    # -- basic quantities ------------------------------------------------
    @property
    def wall_displacement_um(self):
        return self.wall_displacement_fraction * self.shaft_radius_um

    @property
    def _c(self):
        """delta*a, the constant of the 1/r field (units um^2)."""
        return self.wall_displacement_um * self.shaft_radius_um

    @property
    def wall_pressure_Pa(self):
        return 2.0 * self.shear_modulus_Pa * self.wall_displacement_um / self.shaft_radius_um

    def _r(self, r_um):
        a = np.asarray(r_um, dtype=float)
        if not np.isfinite(a).all():
            raise ValueError("radius must be finite")
        if np.any(a < self.shaft_radius_um):
            raise ValueError(
                f"the cavity-expansion solution is defined only outside the shaft "
                f"wall (r >= a = {self.shaft_radius_um:g} um); got r_min = "
                f"{float(np.min(a)):g} um")
        return a

    @staticmethod
    def _wrap(r_um, out):
        return float(out) if np.ndim(r_um) == 0 else out

    def displacement_um(self, r_um):
        return self._wrap(r_um, self._c / self._r(r_um))

    def hoop_strain(self, r_um):
        return self._wrap(r_um, self._c / self._r(r_um) ** 2)

    def radial_strain(self, r_um):
        return self._wrap(r_um, -self._c / self._r(r_um) ** 2)

    def max_abs_principal_strain(self, r_um):
        return self._wrap(r_um, self._c / self._r(r_um) ** 2)

    def max_shear_strain(self, r_um):
        return self._wrap(r_um, 2.0 * self._c / self._r(r_um) ** 2)

    def radial_stress_Pa(self, r_um):
        return self._wrap(r_um, -2.0 * self.shear_modulus_Pa * self._c / self._r(r_um) ** 2)

    def hoop_stress_Pa(self, r_um):
        return self._wrap(r_um, 2.0 * self.shear_modulus_Pa * self._c / self._r(r_um) ** 2)

    def max_shear_stress_Pa(self, r_um):
        return self._wrap(r_um, 2.0 * self.shear_modulus_Pa * self._c / self._r(r_um) ** 2)

    # -- thresholds ------------------------------------------------------
    def strain_radius_um(self, strain_threshold):
        """r where |eps| = threshold, floored at the shaft radius a."""
        e = _finite(strain_threshold, "strain_threshold", 0.0, True)
        return float(max(self.shaft_radius_um, math.sqrt(self._c / e)))

    def shear_stress_radius_um(self, shear_stress_Pa):
        t = _finite(shear_stress_Pa, "shear_stress_Pa", 0.0, True)
        return float(max(self.shaft_radius_um,
                         math.sqrt(2.0 * self.shear_modulus_Pa * self._c / t)))

    # -- analytic-limit diagnostics (used by the selftest) ---------------
    def incompressibility_residual_um2(self, r_um, h_rel=1e-3):
        """d(r*u)/dr, exactly zero analytically; central difference here."""
        r = float(_finite(r_um, "r_um", self.shaft_radius_um))
        if r < self.shaft_radius_um:
            raise ValueError("r must be >= shaft radius")
        h = float(_finite(h_rel, "h_rel", 0.0, True)) * r
        g = lambda x: x * float(self.displacement_um(max(x, self.shaft_radius_um)))
        return (g(r + h) - g(r - h)) / (2 * h)

    def equilibrium_residual_Pa_per_um(self, r_um, h_rel=1e-3):
        """d(sigma_rr)/dr + (sigma_rr - sigma_tt)/r, zero analytically."""
        r = float(_finite(r_um, "r_um", self.shaft_radius_um))
        if r < self.shaft_radius_um:
            raise ValueError("r must be >= shaft radius")
        h = float(_finite(h_rel, "h_rel", 0.0, True)) * r
        sr = lambda x: float(self.radial_stress_Pa(max(x, self.shaft_radius_um)))
        d_sr = (sr(r + h) - sr(r - h)) / (2 * h)
        return d_sr + (float(self.radial_stress_Pa(r)) - float(self.hoop_stress_Pa(r))) / r

    def profile_arrays(self, r_um):
        r = np.asarray(r_um, dtype=float)
        return {"r_um": r, "displacement_um": self.displacement_um(r),
                "hoop_strain": self.hoop_strain(r), "radial_strain": self.radial_strain(r),
                "max_abs_principal_strain": self.max_abs_principal_strain(r),
                "max_shear_strain": self.max_shear_strain(r),
                "radial_stress_Pa": self.radial_stress_Pa(r),
                "hoop_stress_Pa": self.hoop_stress_Pa(r),
                "wall_pressure_Pa": self.wall_pressure_Pa,
                "wall_displacement_um": self.wall_displacement_um}

    def to_dict(self):
        return {"model": "plane-strain incompressible cavity expansion (Lame)",
                "provenance": "ASSUMED; no measured strain field around a probe exists",
                "shaft_radius_um": self.shaft_radius_um,
                "wall_displacement_fraction": self.wall_displacement_fraction,
                "wall_displacement_um": self.wall_displacement_um,
                "shear_modulus_Pa": self.shear_modulus_Pa,
                "wall_pressure_Pa": self.wall_pressure_Pa,
                "validity": "linear-elastic small strain; defined only for r >= a; "
                            "not verified for insect CNS; no plasticity, no "
                            "friction, no tissue tearing, no viscoelasticity"}


@dataclass(frozen=True)
class AffineDamageRadius:
    """AFFINE damage-radius alternative r = offset + slope*a (pure assumption).

    The evidence file states that no measured damage-radius-vs-diameter scaling
    exists, so any scaling law must be assumed.  This one is reported only as a
    sensitivity: it does NOT vanish at zero shaft radius, which is unphysical and
    is exactly why it is not used for the headline numbers.
    """
    offset_um: float = 20.0
    slope: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "offset_um", _finite(self.offset_um, "offset_um", 0.0))
        object.__setattr__(self, "slope", _finite(self.slope, "slope", 0.0))

    def radius_um(self, shaft_radius_um):
        a = _finite(shaft_radius_um, "shaft_radius_um", 0.0)
        return self.offset_um + self.slope * a

    def to_dict(self):
        return {"model": "affine r_damage = offset + slope*a",
                "provenance": "ASSUMED; reported as a sensitivity only",
                "offset_um": self.offset_um, "slope": self.slope,
                "defect": "does not vanish at zero shaft radius"}


# ----------------------------------------------------------------------
# 4b. the SHEATH: the load-bearing layer, and its breach criterion
# ----------------------------------------------------------------------
def provenance_tier(source):
    """Which evidence tier a citation string belongs to.

    ``model_assumption``            no source at all: our choice.
    ``evidence_file_section_2b``    insect data that arrived in the post-hoc
                                    parent addendum and is now in section 2b of
                                    the authoritative evidence file.
    ``evidence_file_primary``       citations listed in the same evidence file
                                    (sections 1 and 2) or in it via section 2b.
    """
    if not source:
        return "model_assumption"
    if ADDENDUM_TAG in source:
        return "evidence_file_section_2b"
    return "evidence_file_primary"


@dataclass(frozen=True)
class SheathLayer:
    """The sheath (neural lamella + perineurium) as a DISTINCT load-bearing layer.

    Why this exists: breaching the sheath lets the nerve substance bulge out
    "almost explosively" (Twarog & Roeder 1956, doi:10.2307/1539019), so the
    primary mechanical insult is sheath breach/delamination, not a rodent-style
    parenchymal kill-zone radius.
    """

    thickness_um: float = 3.0
    critical_opening_radius_um: float = 2.0
    delamination_multiple: float = 2.0
    wall_displacement_fraction: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "thickness_um",
                           _finite(self.thickness_um, "thickness_um", 0.0, True))
        object.__setattr__(self, "critical_opening_radius_um",
                           _finite(self.critical_opening_radius_um,
                                   "critical_opening_radius_um", 0.0, True))
        object.__setattr__(self, "delamination_multiple",
                           _finite(self.delamination_multiple, "delamination_multiple", 0.0))
        f = _finite(self.wall_displacement_fraction, "wall_displacement_fraction", 0.0, True)
        if f > 1.0:
            raise ValueError("wall_displacement_fraction > 1 is outside the small-strain range")
        object.__setattr__(self, "wall_displacement_fraction", f)

    def breached(self, shaft_radius_um):
        """PRIMARY criterion: the sheet must open wider than it can sustain."""
        a = _finite(shaft_radius_um, "shaft_radius_um", 0.0)
        return bool(a >= self.critical_opening_radius_um)

    def hole_edge_stretch(self, shaft_radius_um):
        """Degenerate second criterion: hoop strain at the hole rim = delta/a.

        With a wall displacement proportional to the shaft radius this is exactly
        ``wall_displacement_fraction``, i.e. SIZE-INDEPENDENT.  That degeneracy is
        a direct consequence of the assumed delta ~ a form and is reported rather
        than hidden; it is why the opening-radius criterion is the primary one.
        """
        _finite(shaft_radius_um, "shaft_radius_um", 0.0, True)
        return float(self.wall_displacement_fraction)

    def stretch_criterion_breached(self, shaft_radius_um, critical_stretch):
        return bool(self.hole_edge_stretch(shaft_radius_um) >=
                    _finite(critical_stretch, "critical_stretch", 0.0))

    def delamination_radius_um(self, shaft_radius_um):
        a = _finite(shaft_radius_um, "shaft_radius_um", 0.0)
        if not self.breached(a):
            return 0.0
        return float(max(a, self.critical_opening_radius_um +
                         self.delamination_multiple * self.thickness_um))

    def breach_area_um2(self, shaft_radius_um):
        r = self.delamination_radius_um(shaft_radius_um)
        return float(math.pi * r * r) if r > 0 else 0.0

    def opening_radius_sweep(self, shaft_radii_um, critical_values):
        """Breach state for every (shaft radius, assumed criterion) pair."""
        out = {}
        for c in critical_values:
            c = _finite(c, "critical_opening_radius_um", 0.0, True)
            out[float(c)] = [bool(_finite(a, "shaft_radius_um", 0.0) >= c)
                             for a in shaft_radii_um]
        return out

    def to_dict(self):
        return {"thickness_um": self.thickness_um,
                "critical_opening_radius_um": self.critical_opening_radius_um,
                "delamination_multiple": self.delamination_multiple,
                "primary_criterion": "shaft radius >= critical opening radius "
                                     "(tear/delamination); the criterion value is a "
                                     "FREE PARAMETER, no insect measurement exists",
                "secondary_criterion": "hole-rim hoop strain = delta/a = "
                                       "wall_displacement_fraction, which is "
                                       "size-independent under the assumed delta ~ a "
                                       "form and therefore reported, not used",
                "provenance": "ASSUMED; lamella thickness measured only in cockroach "
                              "(2-5 um), Manduca (4-5 um TEM) and locust (~5 um), all "
                              "CROSS-SPECIES for Drosophila",
                "no_drosophila_measurement": NO_DROSOPHILA_LAMELLA_MEASUREMENT,
                "load_bearing_evidence": SHEATH_LOAD_BEARING_EVIDENCE}


@dataclass(frozen=True)
class PreparationVariant:
    """Which preparation the electrode is inserted into.

    Most Drosophila recordings remove or enzymatically breach the sheath first
    ("peri-neural sheath gently removed"; 0.5% collagenase), so the electrode
    often never penetrates an intact sheath.  The two variants are therefore not
    cosmetic: they decide whether the sheath can be the primary insult at all.
    """
    name: str
    sheath_present: bool
    herniation_multiplier: float = 1.0
    note: str = ""

    VALID = ("sheath_intact", "sheath_removed")
    SHEATH_REMOVED_EVIDENCE = ("most Drosophila recordings breach the sheath first "
                              "('perineural sheath gently removed'; 0.5% collagenase): "
                              + DOI_SHEATH_REMOVED)

    def __post_init__(self):
        if self.name not in self.VALID:
            raise ValueError(f"unknown preparation variant {self.name!r}; use {self.VALID}")
        object.__setattr__(self, "sheath_present", bool(self.sheath_present))
        object.__setattr__(self, "herniation_multiplier",
                           _finite(self.herniation_multiplier, "herniation_multiplier", 1.0))

    def to_dict(self):
        return {"name": self.name, "sheath_present": self.sheath_present,
                "herniation_multiplier": self.herniation_multiplier,
                "note": self.note,
                "surgery_note": "in the sheath-removed preparation the insertion "
                                "damage is plausibly smaller than the SURGERY damage "
                                "(sheath removal / enzymatic breach), and no insect "
                                "study quantifies surgery damage either, so that term "
                                "is not modelled",
                "sheath_removal_evidence": PreparationVariant.SHEATH_REMOVED_EVIDENCE,
                "damage_quantified_for_this_variant_in_insects": False}


def preparation_variants(adopt):
    """The two preparations requested by the addendum, with provenance attached."""
    mult = adopt("assumed_sheath_herniation_multiplier")
    return {
        "sheath_intact": PreparationVariant(
            name="sheath_intact", sheath_present=True, herniation_multiplier=mult.value),
        "sheath_removed": PreparationVariant(
            name="sheath_removed", sheath_present=False, herniation_multiplier=1.0),
    }


def parenchymal_damage_estimate(field, strain_threshold, geometry, sheath,
                               variant, free_radius_um=None, threshold_record=None):
    """Damage as a FRACTION of the fly brain, for one preparation variant.

    ``free_radius_um`` overrides the mechanistic radius with the explicitly
    UNMEASURED free parameter; the mechanistic radius is the strain-threshold
    crossing of the assumed field.  Both are reported, never merged silently.
    """
    if not isinstance(field, CavityExpansionField):
        raise TypeError("field must be a CavityExpansionField")
    if not isinstance(geometry, FlyBrainGeometry):
        raise TypeError("geometry must be a FlyBrainGeometry")
    if not isinstance(sheath, SheathLayer):
        raise TypeError("sheath must be a SheathLayer")
    if not isinstance(variant, PreparationVariant):
        raise TypeError("variant must be a PreparationVariant")
    a = field.shaft_radius_um
    mechanistic = field.strain_radius_um(strain_threshold)
    free = mechanistic if free_radius_um is None else \
        _finite(free_radius_um, "free_radius_um", 0.0)
    breached = bool(variant.sheath_present and sheath.breached(a))
    multiplier = variant.herniation_multiplier if breached else 1.0
    delamination = sheath.delamination_radius_um(a) if variant.sheath_present else 0.0
    breach_area = sheath.breach_area_um2(a) if variant.sheath_present else 0.0
    out = {
        "variant": variant.name,
        "shaft_radius_um": a,
        "strain_criterion": float(strain_threshold),
        "mechanistic_damage_radius_um": mechanistic,
        "free_parameter_radius_um": free,
        "sheath_present": variant.sheath_present,
        "sheath_breached": breached,
        "parenchymal_multiplier": multiplier,
        "parenchymal_radius_mechanistic_um": mechanistic * multiplier,
        "parenchymal_radius_free_parameter_um": free * multiplier,
        "parenchymal_fraction_mechanistic": geometry.damage_fraction(mechanistic * multiplier),
        "parenchymal_fraction_free_parameter": geometry.damage_fraction(free * multiplier),
        "sheath_delamination_radius_um": delamination,
        "sheath_breach_area_um2": breach_area,
        "sheath_breach_area_fraction_of_footprint": breach_area / geometry.footprint_um2,
        "sheath_breach_area_fraction_of_brain_volume": (
            breach_area * geometry.thickness_um / geometry.volume_um3),
        "units": "um, um^2, fraction of the fly brain box (1)",
        "provenance": "ASSUMED model output, NOT a measurement: the parenchymal "
                      "radius, the sheath thickness and the sheath breach "
                      "criterion are all free parameters",
    }
    if threshold_record is not None:
        out["strain_criterion_record"] = threshold_record.to_dict()
    return out


def rodent_radius_records(adopt):
    """Provenance records for the rodent 'damage radius' numbers people quote.

    Returns {radius_um: wrapper}.  None of these is a fly number, and one of them
    (100 um) is not even a primary measurement -- it is a review paraphrase.
    """
    out = {}
    a = adopt("rodent_histology_neuron_loss_zone_um")
    out[50.0] = {**a.to_dict(),
                 "note": a.note + " | used here as a radial DISTANCE from the track"}
    out[100.0] = {
        "value": 100.0, "unit": "um", "provenance": ASSUMED, "species": SP_MOUSE,
        "source": "evidence file section 2 item 3: the widely circulated 'kill zone "
                  "up to 100 um' is a PARAPHRASE in the Polikov 2005 review, NOT an "
                  "original measurement",
        "cross_species": True, "forced_cross_species_flag": True,
        "parameter": "commonly_cited_100um_damage_zone",
        "note": "kept in the comparison precisely because it is the number people "
                "transfer; it has the weakest provenance of the four"}
    b = adopt("rodent_glial_scar_fwhm_um")
    out[129.0] = {**b.to_dict(),
                  "note": b.note + " | this is a FULL WIDTH at half maximum, so the "
                                   "scar RADIUS is about half of it; listed at its "
                                   "full width to show how large the rodent number is"}
    c = adopt("rodent_histology_degeneration_um")
    out[150.0] = {**c.to_dict(),
                  "note": c.note + " | the study's own upper distance, used here as a radius"}
    return out


def rodent_radius_transfer_table(geometry, adopt=None, mouse_thickness_um=900.0,
                                radii_um=None):
    """Why a rodent damage radius cannot be transferred to the fly.

    For each rodent-reported lesion radius: the fraction of the FLY brain volume a
    single full-thickness track would destroy, next to the fraction of a MOUSE
    cortical thickness (900 um) the same radius spans.  The same number is a
    modest lesion in the mouse and a large fraction of the fly.
    """
    if not isinstance(geometry, FlyBrainGeometry):
        raise TypeError("geometry must be a FlyBrainGeometry")
    thickness = _finite(mouse_thickness_um, "mouse_thickness_um", 0.0, True)
    records = rodent_radius_records(adopt) if adopt is not None else {}
    if radii_um is None:
        radii_um = sorted(records)
    rows = []
    for r in radii_um:
        r = _finite(r, "radius_um", 0.0)
        record = records.get(r)
        if record is None:
            record = {"value": r, "unit": "um", "provenance": ASSUMED,
                      "species": "not specified by the caller",
                      "source": "radius supplied by the caller, not a registered "
                                "measurement",
                      "cross_species": True, "forced_cross_species_flag": True,
                      "parameter": "unregistered_radius"}
        rows.append({
            "radius_record": record,
            "fly_brain_volume_fraction": geometry.damage_fraction(r),
            "fly_brain_volume_percent": 100.0 * geometry.damage_fraction(r),
            "fly_thickness_ratio_2R_over_100um": geometry.thickness_ratio(r),
            "reference_cortical_thickness_fraction_2R_over_900um": 2.0 * r / thickness,
            "transfer_factor": (geometry.thickness_ratio(r) / (2.0 * r / thickness)),
            "units": "um, fraction (1), percent (%)",
        })
    thickness_record = (adopt("mouse_cortical_thickness_um").to_dict() if adopt is not None
                        else {"value": thickness, "unit": "um", "provenance": ILLUSTRATIVE,
                              "species": SP_MOUSE,
                              "source": "quoted in the evidence file without a citation",
                              "cross_species": True, "parameter": "mouse_cortical_thickness_um"})
    return {"rows": rows, "reference_cortical_thickness_record": thickness_record,
            "radii_used_um": [row["radius_record"]["value"] for row in rows],
            "conclusion": "the SAME rodent radius is a small fraction of a 900 um "
                          "cortex and is 1-3x the ENTIRE fly brain thickness; the "
                          "transfer factor is exactly 9.0x, so rodent radii cannot be "
                          "transferred to a 100 um-thick fly brain",
            "provenance": "fly geometry ILLUSTRATIVE; mouse cortical thickness "
                          "quoted in the evidence file without a citation "
                          "(registered illustrative)"}


@dataclass(frozen=True)
class SheathRepairModel:
    """Slow sheath repair vs fast membrane resealing: do NOT reuse one for the other."""
    membrane_small_pore_tau_s: float
    membrane_transected_tau_s: float
    sheath_repair_tau_s: float
    insect_repair_persists_days: float

    def __post_init__(self):
        for name in ("membrane_small_pore_tau_s", "membrane_transected_tau_s",
                     "sheath_repair_tau_s", "insect_repair_persists_days"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, True))

    @property
    def sheath_over_pore_ratio(self):
        return self.sheath_repair_tau_s / self.membrane_small_pore_tau_s

    @property
    def sheath_over_transected_ratio(self):
        return self.sheath_repair_tau_s / self.membrane_transected_tau_s

    def to_dict(self):
        return {"membrane_small_pore_tau_s": self.membrane_small_pore_tau_s,
                "membrane_transected_tau_s": self.membrane_transected_tau_s,
                "sheath_repair_tau_s": self.sheath_repair_tau_s,
                "insect_repair_persists_days": self.insect_repair_persists_days,
                "sheath_over_transected_ratio": self.sheath_over_transected_ratio,
                "sheath_over_pore_ratio": self.sheath_over_pore_ratio,
                "negative_result": NO_SECONDS_SCALE_SHEATH_RESEALING,
                "cross_species": True,
                "source": f"{CIT_RAT_MECHANOPORATION}; {CIT_GUINEA_RESEAL}; {ADDENDUM_TAG}",
                "applied": "no seconds-scale sheath resealing is applied anywhere in "
                           "this model; the membrane resealing constants are used only "
                           "for the membrane leak, where they were measured"}


def chemistry_config_for_variant(variant, adopt):
    """Chemistry coupling parameters for a preparation variant.

    For ``sheath_removed`` the sheath restriction is ALREADY gone before the
    electrode arrives, so its intact and damaged conductances are set equal: the
    coupling then changes the K permeability only, and changes the sheath term by
    exactly nothing.  That equality is a testable, exact statement.
    """
    if not isinstance(variant, PreparationVariant):
        raise TypeError("variant must be a PreparationVariant")
    damaged_sheath = adopt("assumed_sheath_conductance_damaged_um3_s").value
    if variant.sheath_present:
        intact_sheath = 0.0          # idealised fully tight sheath in the sham
    else:
        intact_sheath = damaged_sheath
    return ChemistryDamageConfig(
        permeability_intact_um_s=0.0,
        permeability_damaged_um_s=adopt("assumed_damaged_permeability_um_s").value,
        reservoir_conductance_um3_s=0.0,
        sheath_conductance_intact_um3_s=intact_sheath,
        sheath_conductance_damaged_um3_s=damaged_sheath)



# ----------------------------------------------------------------------
# 5. thresholds and resealing
# ----------------------------------------------------------------------
STRAIN_THRESHOLD_CHOICES = {
    "conservative": "rodent_axonal_strain_conservative",
    "optimal": "rodent_axonal_strain_optimal",
    "permissive": "rodent_axonal_strain_permissive",
}


def injury_threshold_options(guard):
    """The externally verified thresholds, force-flagged as cross-species."""
    out = {key: guard.adopt(name) for key, name in STRAIN_THRESHOLD_CHOICES.items()}
    out["_mechanoporation_shear"] = guard.adopt("rodent_mechanoporation_shear_Pa")
    out["_mechanoporation_shear_dyn_cm2"] = guard.adopt("rodent_mechanoporation_shear_dyn_cm2")
    out["_mechanoporation_duration_ms"] = guard.adopt("rodent_mechanoporation_duration_ms")
    return out


@dataclass(frozen=True)
class ResealingTimescales:
    """Membrane resealing constants, applied exactly as exp(-t/tau)."""
    small_pore_tau_s: float
    transected_tau_s: float
    transected_tau_sd_s: float

    KINDS = ("small_pore", "transected")

    def __post_init__(self):
        for name in ("small_pore_tau_s", "transected_tau_s"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0, True))
        object.__setattr__(self, "transected_tau_sd_s",
                           _finite(self.transected_tau_sd_s, "transected_tau_sd_s", 0.0))

    @classmethod
    def from_adopt(cls, adopt):
        return cls(adopt("rodent_reseal_small_pore_tau_s").value,
                   adopt("rodent_reseal_transected_tau_s").value,
                   adopt("rodent_reseal_transected_tau_sd_s").value)

    def tau_s(self, kind):
        if kind not in self.KINDS:
            raise KeyError(f"unknown resealing kind {kind!r}; use {self.KINDS}")
        return self.small_pore_tau_s if kind == "small_pore" else self.transected_tau_s

    def decay_factor(self, t_since_damage_s, kind="transected"):
        t = _finite(t_since_damage_s, "t_since_damage_s", 0.0)
        return float(math.exp(-t / self.tau_s(kind)))

    def decay_factors(self, times_s, kind="transected"):
        t = np.asarray(times_s, dtype=float)
        if not np.isfinite(t).all() or np.any(t < 0):
            raise ValueError("times must be finite and nonnegative")
        return np.exp(-t / self.tau_s(kind))

    def to_dict(self):
        return {"small_pore_tau_s": self.small_pore_tau_s,
                "transected_tau_s": self.transected_tau_s,
                "transected_tau_sd_s": self.transected_tau_sd_s,
                "form": "single exponential exp(-t/tau); the source for the small "
                        "pore only bounds a 4-fold decrease within 1 min and full "
                        "recovery by 10 min (PMC7385830), so the exponential is ours",
                "cross_species": True,
                "source": f"{CIT_RAT_MECHANOPORATION}; {CIT_GUINEA_RESEAL}",
                "conditions": "transected-axon resealing is blocked at <=25 C or "
                              "[Ca]o <=0.5 mM; nothing here models calcium"}


# ----------------------------------------------------------------------
# 6. coupling
# ----------------------------------------------------------------------
def affected_node_indices(morph, shaft_point_um, shaft_direction, damage_radius_um):
    """Cable rows whose centre lies within damage_radius of the shaft AXIS."""
    if not isinstance(morph, Morphology):
        raise TypeError("morph must be an engine.cable.Morphology")
    point = np.asarray(shaft_point_um, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("shaft_point_um must be three finite coordinates")
    direction = np.asarray(shaft_direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("shaft_direction must be three finite coordinates, nonzero")
    radius = _finite(damage_radius_um, "damage_radius_um", 0.0)
    unit = direction / np.linalg.norm(direction)
    centres = np.column_stack((morph.x, morph.y, morph.z))
    rel = centres - point
    perpendicular = rel - np.outer(rel @ unit, unit)
    distance = np.linalg.norm(perpendicular, axis=1)
    return np.flatnonzero(distance <= radius)


@dataclass(frozen=True)
class ProximityExcitabilityCoupling:
    """OPTIONAL proximity -> reduced-drive coupling.  DEFAULT OFF.

    This is NOT an excitability model: ``engine.cable`` is passive and has no
    channels, so all it can do is scale the injected drive found in the damaged
    neighbourhood.  With ``enabled=False`` the scale array is exactly 1.0, so a
    disabled run is bit-for-bit identical to a run without the coupling.
    """
    enabled: bool = False
    full_effect_radius_um: float = 5.0
    max_attenuation: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "full_effect_radius_um",
                           _finite(self.full_effect_radius_um, "full_effect_radius_um", 0.0, True))
        f = _finite(self.max_attenuation, "max_attenuation", 0.0)
        if f > 1.0:
            raise ValueError("max_attenuation must be <= 1")
        object.__setattr__(self, "max_attenuation", f)

    def scale(self, distances_um):
        d = np.asarray(distances_um, dtype=float)
        if not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("distances must be finite and nonnegative")
        if not self.enabled:
            return np.ones_like(d)
        proximity = np.clip(1.0 - d / self.full_effect_radius_um, 0.0, 1.0)
        return 1.0 - self.max_attenuation * proximity

    def to_dict(self):
        return {"enabled": self.enabled, "default": "OFF",
                "full_effect_radius_um": self.full_effect_radius_um,
                "max_attenuation": self.max_attenuation,
                "provenance": "ASSUMED; placeholder, NOT an excitability model "
                              "(the passive cable has no channels)"}


@dataclass(frozen=True)
class CableDamageConfig:
    """Leak density + reversal for the cable end-leak coupling."""
    leak_density_S_cm2: float = 0.5
    E_rev_mV: float | None = None          # None -> the cable's own E_leak
    resealing_kind: str | None = None      # None -> static leak
    excitability: ProximityExcitabilityCoupling = \
        field(default_factory=ProximityExcitabilityCoupling)

    def __post_init__(self):
        object.__setattr__(self, "leak_density_S_cm2",
                           _finite(self.leak_density_S_cm2, "leak_density_S_cm2", 0.0))
        if self.E_rev_mV is not None:
            object.__setattr__(self, "E_rev_mV", _finite(self.E_rev_mV, "E_rev_mV"))
        if self.resealing_kind is not None and self.resealing_kind not in ResealingTimescales.KINDS:
            raise KeyError(f"unknown resealing kind {self.resealing_kind!r}")

    def to_dict(self):
        return {"leak_density_S_cm2": self.leak_density_S_cm2,
                "E_rev_mV": self.E_rev_mV,
                "E_rev_note": "None means 'shunt at the cable's own E_leak', which "
                              "isolates the CONDUCTANCE effect. A real "
                              "mechanoporation pore has a nonspecific reversal "
                              "near 0 mV and would also depolarise; that is an "
                              "unmodelled additional assumption.",
                "resealing_kind": self.resealing_kind,
                "coupling_api": "CableNeuron.set_end_leak (cable.py not modified)"}


def apply_membrane_damage(cable, node_indices, leak_density_S_cm2, E_rev_mV):
    """Add an end leak to the damaged rows via the public set_end_leak API.

    The conductance per compartment is ``density * area_cm2 * 1e6`` (uS), i.e.
    area-weighted, and rows with an identical float conductance are set in one
    grouped call.  ``set_end_leak`` SETS rather than increments, so this is a
    replacement of those rows' end leak -- the caller owns that choice.
    """
    if not isinstance(cable, CableNeuron):
        raise TypeError("cable must be an engine.cable.CableNeuron")
    indices = np.asarray(node_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("at least one damaged node index is required")
    if np.any(indices < 0) or np.any(indices >= cable.m.n):
        raise ValueError("damaged node index out of range")
    if len(set(indices.tolist())) != indices.size:
        raise ValueError("duplicate node indices would double-count the damage")
    density = _finite(leak_density_S_cm2, "leak_density_S_cm2", 0.0)
    reversal = _finite(E_rev_mV, "E_rev_mV")
    per_node = density * cable.area_cm2[indices] * 1e6
    groups = {}
    for index, g in zip(indices, per_node):
        groups.setdefault(float(g), []).append(int(index))
    for g, members in groups.items():
        cable.set_end_leak(members, g, reversal)
    return {"node_indices": tuple(int(i) for i in indices),
            "g_uS_per_node": tuple(float(g) for g in per_node),
            "total_g_uS": float(per_node.sum()),
            "E_rev_mV": reversal,
            "leak_density_S_cm2": density,
            "n_rows_set": int(indices.size),
            "grouped_calls": len(groups)}


def run_paired_cable(morphology, field, shaft_point_um, shaft_direction,
                     strain_threshold, config=None, inject_index=0,
                     target_deflection_mV=10.0, dt_ms=0.05, duration_ms=50.0,
                     cable_kwargs=None, resealing=None):
    """Inserted vs sham on two identical passive cables.

    Only the membrane-damage coupling differs: the sham keeps ``g_end == 0``
    exactly.  The injected current is calibrated once from the sham's DC input
    resistance to a stated target deflection, so the comparison is not a hidden
    comparison of two different stimuli.
    """
    if not isinstance(field, CavityExpansionField):
        raise TypeError("field must be a CavityExpansionField")
    config = config or CableDamageConfig()
    cable_kwargs = dict(cable_kwargs or {})
    dt_ms = _finite(dt_ms, "dt_ms", 0.0, True)
    duration_ms = _finite(duration_ms, "duration_ms", 0.0, True)
    steps = int(round(duration_ms / dt_ms))
    if steps < 1:
        raise ValueError("duration shorter than one timestep")
    if not isinstance(inject_index, (int, np.integer)) or not 0 <= inject_index < morphology.n:
        raise ValueError("inject_index out of range")
    target = _finite(target_deflection_mV, "target_deflection_mV", 0.0, True)

    radius = field.strain_radius_um(strain_threshold)
    indices = affected_node_indices(morphology, shaft_point_um, shaft_direction, radius)

    def build():
        return CableNeuron(morphology, dt_ms=dt_ms, **cable_kwargs)

    # DC calibration on a throw-away cable, so the two compared cables are
    # untouched.  At DC, G * dv = i (C/dt terms drop out), so one sparse solve of
    # the PUBLIC conductance matrix gives the exact input resistance used to set
    # the injected current for the stated target deflection.
    calibration = build()
    i_unit = np.zeros(morphology.n)
    i_unit[inject_index] = 1.0              # 1 nA inward
    dv = spsolve(calibration.G.tocsc(), i_unit)
    if not np.isfinite(dv).all() or dv[inject_index] <= 0:
        raise FloatingPointError("sham DC input resistance is not positive finite")
    # mV/nA = 1e-3 V / 1e-9 A = 1e6 ohm = 1 Mohm, so this ratio IS in Mohm.
    resistance_Mohm = float(dv[inject_index])
    drive_nA = target / resistance_Mohm

    sham, damaged = build(), build()
    record = None
    if indices.size:
        record = apply_membrane_damage(damaged, indices, config.leak_density_S_cm2,
                                       sham.E_leak if config.E_rev_mV is None
                                       else config.E_rev_mV)

    centres = np.column_stack((morphology.x, morphology.y, morphology.z))
    direction = np.asarray(shaft_direction, dtype=float)
    unit = direction / np.linalg.norm(direction)
    rel = centres - np.asarray(shaft_point_um, dtype=float)
    distances = np.linalg.norm(rel - np.outer(rel @ unit, unit), axis=1)
    scale = config.excitability.scale(distances)
    i_inject = np.zeros(morphology.n)
    i_inject[inject_index] = drive_nA
    i_effective = i_inject * scale

    traces = {"sham": np.empty((steps, morphology.n)),
              "inserted": np.empty((steps, morphology.n))}
    g_end_final = np.zeros(morphology.n)
    factor_record = []
    for k in range(steps):
        t_ms = (k + 1) * dt_ms
        if config.resealing_kind is not None and indices.size:
            if resealing is None:
                raise ValueError("resealing_kind set but no ResealingTimescales given")
            factor = resealing.decay_factor(t_ms / 1000.0, config.resealing_kind)
            per_node = config.leak_density_S_cm2 * damaged.area_cm2[indices] * 1e6
            groups = {}
            for index, g in zip(indices, per_node):
                groups.setdefault(float(g * factor), []).append(int(index))
            for g, members in groups.items():
                damaged.set_end_leak(members, g, record["E_rev_mV"])
            g_end_final = damaged.g_end.copy()
            factor_record.append(factor)
        # the optional proximity coupling scales the drive in BOTH runs, because
        # the probe is present in both; the inserted-vs-sham difference therefore
        # stays attributable to the membrane damage alone.
        traces["sham"][k] = sham.step(i_inject=i_effective)
        traces["inserted"][k] = damaged.step(i_inject=i_effective)
    if not factor_record:
        g_end_final = damaged.g_end.copy()
        factor_record = [1.0]

    delta = traces["sham"] - traces["inserted"]
    return {
        "damage_radius_um": radius,
        "distances_to_shaft_axis_um": distances.copy(),
        "damaged_node_indices": indices,
        "damaged_node_centres_um": centres[indices].copy() if indices.size else np.zeros((0, 3)),
        "g_end_uS": g_end_final,
        "membrane_damage_record": record,
        "sham": traces["sham"], "inserted": traces["inserted"],
        "delta_mV": delta,
        "drive_nA": drive_nA, "drive_effective_nA": i_effective.copy(),
        "i_inject_sham_nA": i_inject,
        "scale": scale,
        "inject_index": int(inject_index),
        "input_resistance_Mohm": resistance_Mohm,
        "target_deflection_mV": target,
        "dt_ms": dt_ms, "duration_ms": duration_ms, "steps": steps,
        "resealing_factor_range": (float(min(factor_record)), float(max(factor_record))),
        "resealing_kind": config.resealing_kind,
        "E_leak_mV": sham.E_leak,
        "max_abs_delta_mV": float(np.abs(delta).max()),
        "max_local_attenuation_fraction": float(
            0.0 if not indices.size else
            np.max(np.abs(delta[:, indices])) /
            max(1e-12, np.max(np.abs(traces["sham"][:, indices] - sham.E_leak)))),
    }


@dataclass(frozen=True)
class ChemistryDamageConfig:
    permeability_intact_um_s: float = 0.0
    permeability_damaged_um_s: float = 0.2
    reservoir_conductance_um3_s: float = 0.0
    sheath_conductance_intact_um3_s: float = 0.0
    sheath_conductance_damaged_um3_s: float = 5.0

    def __post_init__(self):
        for name in ("permeability_intact_um_s", "permeability_damaged_um_s",
                     "reservoir_conductance_um3_s", "sheath_conductance_intact_um3_s",
                     "sheath_conductance_damaged_um3_s"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, 0.0))

    def scaled_values(self, damage_fraction):
        f = _finite(damage_fraction, "damage_fraction", 0.0)
        if f > 1.0:
            raise ValueError("damage_fraction must be in [0, 1]")
        p = self.permeability_intact_um_s + \
            (self.permeability_damaged_um_s - self.permeability_intact_um_s) * f
        g = self.sheath_conductance_intact_um3_s + \
            (self.sheath_conductance_damaged_um3_s - self.sheath_conductance_intact_um3_s) * f
        return float(p), float(g)

    def to_dict(self):
        return {"permeability_intact_um_s": self.permeability_intact_um_s,
                "permeability_damaged_um_s": self.permeability_damaged_um_s,
                "reservoir_conductance_um3_s": self.reservoir_conductance_um3_s,
                "sheath_conductance_intact_um3_s": self.sheath_conductance_intact_um3_s,
                "sheath_conductance_damaged_um3_s": self.sheath_conductance_damaged_um3_s,
                "provenance": "ALL FOUR damaged/intact values are ASSUMED; the "
                              "intact values are 0, i.e. the sham has exactly zero "
                              "exchange, which is what makes the zero-crossing "
                              "check exact",
                "fly_note": "for Drosophila this is a glial/tracheal sheath "
                            "restriction, NOT a vertebrate blood-brain barrier",
                "coupling_api": "InjuryPotassium.apply_injury + "
                                "InjuryLigand.switch_barrier"}


def apply_chemistry_coupling(potassium, ligand, barrier_edge, damage_fraction,
                            config=None):
    """Drive the existing injury_tissue public API from a damage fraction."""
    if not isinstance(potassium, InjuryPotassium):
        raise TypeError("potassium must be an engine.injury_tissue.InjuryPotassium")
    if not isinstance(ligand, InjuryLigand):
        raise TypeError("ligand must be an engine.injury_tissue.InjuryLigand")
    config = config or ChemistryDamageConfig()
    if not (isinstance(barrier_edge, tuple) and len(barrier_edge) == 2):
        raise ValueError("barrier_edge must be a tuple of two voxel indices")
    permeability, sheath = config.scaled_values(damage_fraction)
    potassium.apply_injury(permeability, config.reservoir_conductance_um3_s)
    ligand.switch_barrier({tuple(barrier_edge): sheath})
    return {"damage_fraction": float(damage_fraction),
            "permeability_um_s": permeability,
            "reservoir_conductance_um3_s": config.reservoir_conductance_um3_s,
            "barrier_edge": tuple(barrier_edge),
            "sheath_conductance_um3_s": sheath,
            "potassium_events": len(potassium.events),
            "ligand_events": len(ligand.events),
            "api": "InjuryPotassium.apply_injury / InjuryLigand.switch_barrier"}


def run_paired_chemistry(damage_fraction, config=None, barrier_edge=(1, 2),
                        dt_s=0.05, duration_s=20.0, shape=(4, 1, 1),
                        spacing_um=(5.0, 5.0, 5.0), diffusion_um2_s=10.0,
                        source_amount_s=1.0, potassium_config=None,
                        ligand_extra=None, include_supported=False,
                        supported_reservoir_um3_s=5.0, supported_buffer_amount=4000.0):
    """Inserted vs sham for the chemistry: identical models, only coupling differs.

    ``sham``   -> ``apply_chemistry_coupling(..., damage_fraction=0)``
    ``inserted`` -> ``apply_chemistry_coupling(..., damage_fraction)``

    Both go through the SAME public API (``InjuryPotassium.apply_injury`` /
    ``InjuryLigand.switch_barrier``), so the two runs differ by a value and not
    by a code path.  In the sham the intact permeability is exactly 0 and the
    intact sheath conductance is exactly 0, which makes "no exchange at all" an
    exact claim rather than an approximate one.
    """
    dt_s = _finite(dt_s, "dt_s", 0.0, True)
    duration_s = _finite(duration_s, "duration_s", 0.0, True)
    steps = int(round(duration_s / dt_s))
    if steps < 1:
        raise ValueError("duration shorter than one timestep")
    config = config or ChemistryDamageConfig()
    base = potassium_config or PotassiumConfig(
        intracellular_volume_um3=1000.0, extracellular_volume_um3=200.0,
        initial_inside_mM=120.0, initial_outside_mM=3.0, membrane_area_um2=100.0,
        buffer_capacity_amount=0.0, reservoir_conductance_um3_s=0.0,
        temperature_K=298.15)
    source = np.zeros(shape)
    source[0, 0, 0] = _finite(source_amount_s, "source_amount_s", 0.0)
    ligand_kwargs = dict(shape=shape, spacing_um=spacing_um,
                         diffusion_um2_s=diffusion_um2_s,
                         initial_nM=0.0, receptor_capacity_nM=0.0,
                         kon_nM_inv_s=0.1, koff_s=0.1, response_tau_s=10.0,
                         source_amount_s=source,
                         barrier_edges={tuple(barrier_edge): config.sheath_conductance_intact_um3_s})
    ligand_kwargs.update(ligand_extra or {})

    # The two variants differ for BOTH runs in the same way (reservoir/buffer),
    # so the inserted-vs-sham difference is attributable to the damage coupling
    # alone.  The 'closed' variant has no reservoir and no buffer, and therefore
    # shows the documented closed-two-pool artifact
    # (engine/injury_tissue.py docstring) at full strength.
    variants = {"closed": (base, config)}
    if include_supported:
        variants["supported"] = (
            replace(base, buffer_capacity_amount=supported_buffer_amount),
            replace(config, reservoir_conductance_um3_s=supported_reservoir_um3_s))

    out = {"damage_fraction": float(damage_fraction), "dt_s": dt_s,
           "duration_s": duration_s, "steps": steps,
           "barrier_edge": tuple(barrier_edge), "config": config.to_dict(),
           "variant_definitions": {},
           "trace_columns": ["time_s", "Ki_mM", "Ke_mM", "Ek_mV",
                             "source_voxel_free_nM", "far_voxel_free_nM",
                             "ligand_boundary_net_nM_um3"],
           "variants": {}}
    for variant, (kconfig, cconfig) in variants.items():
        out["variant_definitions"][variant] = {
            "potassium_config": {k: v for k, v in vars(kconfig).items()},
            "chemistry_damage_config": cconfig.to_dict(),
            "note": "both runs in this variant use the same potassium config; only "
                    "the damage-scaled permeability/sheath conductance differs",
        }
        models = {}
        for name, fraction in (("sham", 0.0), ("inserted", float(damage_fraction))):
            m = InjuryPotassium(kconfig)
            l = InjuryLigand(**ligand_kwargs)
            coupling = apply_chemistry_coupling(m, l, barrier_edge, fraction, cconfig)
            models[name] = {"potassium": m, "ligand": l, "coupling": coupling,
                            "errors": (0.0, 0.0)}
        traces = {name: [] for name in models}
        for _ in range(steps):
            for name, mm in models.items():
                p, l = mm["potassium"], mm["ligand"]
                p.step(dt_s)
                l.step(dt_s)
                err = (abs(p.mass_balance_error()), abs(l.mass_balance_error()))
                mm["errors"] = (max(mm["errors"][0], err[0]), max(mm["errors"][1], err[1]))
                far = float(np.mean(l.model.free_nM[2:])) if np.prod(shape) > 2 else \
                    float(l.model.free_nM[-1])
                traces[name].append([p.time_s, p.inside_mM, p.outside_mM, p.ek_mV,
                                     float(l.model.free_nM[0, 0, 0]), far,
                                     l.ledger["boundary_in"] - l.ledger["boundary_out"]])
        inserted = models["inserted"]
        out["variants"][variant] = {
            "traces": {k: np.asarray(v, dtype=float) for k, v in traces.items()},
            "sham_coupling": models["sham"]["coupling"],
            "inserted_coupling": inserted["coupling"],
            "max_potassium_mass_error": inserted["errors"][0],
            "max_ligand_mass_error": inserted["errors"][1],
            "sham_permeability_um_s": models["sham"]["potassium"].permeability_um_s,
            "inserted_permeability_um_s": inserted["potassium"].permeability_um_s,
            "sham_far_ligand_nM": float(np.mean(models["sham"]["ligand"].model.free_nM[2:])),
            "inserted_far_ligand_nM": float(np.mean(inserted["ligand"].model.free_nM[2:])),
            "final_Ki_sham_mM": models["sham"]["potassium"].inside_mM,
            "final_Ki_inserted_mM": inserted["potassium"].inside_mM,
            "final_Ke_sham_mM": models["sham"]["potassium"].outside_mM,
            "final_Ke_inserted_mM": inserted["potassium"].outside_mM,
            "final_Ek_sham_mV": models["sham"]["potassium"].ek_mV,
            "final_Ek_inserted_mV": inserted["potassium"].ek_mV,
            "inserted_sheath_events": list(inserted["ligand"].events),
            "sham_sheath_events": list(models["sham"]["ligand"].events),
        }
    return out
