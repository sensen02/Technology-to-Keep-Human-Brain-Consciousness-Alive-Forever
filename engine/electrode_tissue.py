"""engine.electrode_tissue -- TISSUE MECHANICS of electrode insertion into
Drosophila nervous tissue, and its connection to the REAL access map.

WHAT THIS MODULE ADDS, AND WHAT IT REUSES  (read this first)
===========================================================
The project already had a PARTIAL mechanical model in ``engine/electrode_damage.py``:
an assumed cavity-expansion strain field, an assumed sheath layer, provenance
machinery, a species guard and two coupling functions.  This module does NOT
reinvent any of that.  It adds exactly the two mechanisms that were missing --
**insertion dimpling** and **physiological micromotion** -- and connects the
mechanics to the access map and to the cable model through the project's own
APIs.

REUSED UNCHANGED (imported and called, not copied):
  * ``CavityExpansionField``       the shaft strain FIELD.  Every strain-vs-radius
                                   number for insertion in this module IS this
                                   class's output; ``InsertionMechanics.field``
                                   is an instance of it and all strain methods
                                   delegate to it.
  * ``FlyBrainGeometry``           damage as a fraction of the 500x300x100 um box.
  * ``SheathLayer``                the load-bearing-layer breach criterion.
  * ``build_registry`` / ``Param`` / ``ParamRegistry`` / ``ProvenanceError``
                                   parameter provenance.
  * ``SpeciesGuard`` / ``Adopted`` / ``audit_species_labeling``
                                   the foreign-species refusal.
  * ``affected_node_indices``      which cable rows lie near the shaft axis.
  * ``apply_membrane_damage``      THE damage application into the cable (via
                                   ``CableNeuron.set_end_leak``).  This module
                                   contains NO second membrane-damage model.
  * ``ProximityExcitabilityCoupling``  the optional proximity drive-scaling.
  * ``ResealingTimescales`` / ``CableDamageConfig``
  * ``run_paired_cable``           the project's own inserted-vs-sham paired run,
                                   called here as an independent cross-check of
                                   this module's paired run.
  * ``engine.embodied.access_map.Ledger`` / ``PROVENANCE``
                                   the ledger that refuses MEASURED_CITED without
                                   a source and ASSUMED without a sweep.
  * ``engine.cable.CableNeuron`` / ``Morphology``
  * ``BANC_SOMA_ATTRIBUTION``      the CC BY 4.0 attribution record (see below).

ADDED HERE (new mechanisms, new parameters, new connections):
  * ``DimplingModel``        surface indentation BEFORE sheath penetration:
                             a two-term contact stiffness (flat-punch elastic
                             2*a*E* plus tensed-surface 2*pi*T) for the depth at
                             a stated force, and a shallow-spherical-cap stretch
                             criterion for the depth at which the sheet fails.
                             Both coefficients are ASSUMED and swept; there is
                             NO measured dimpling depth for any insect.
  * ``InsertionMechanics``   (a) + (b) of the brief in one object, scaled by the
                             USER-FIXED 7 um shaft diameter.
  * ``MicromotionStrain``    shear strain from physiological micromotion acting
                             on tissue that ALREADY CONTAINS the probe.  The
                             quasi-static kernel is the SAME incompressible 1/r
                             field the project already uses (reused and checked
                             for exact numerical agreement in the valid regime);
                             the cyclic amplitude replaces the static wall
                             displacement.
  * ``probe_neurons_at_risk``  the REAL captured neurons of an ``AccessMap``
                             tested against a stated strain criterion, with the
                             count and the fraction of the addressable population.
  * ``couple_to_excitable_model``  drives ``apply_membrane_damage`` and
                             ``ProximityExcitabilityCoupling`` from the mechanics
                             and reports the measured in-model effect.
  * ``TissueProvenance``     a ledger wrapper that uses the project's ``Ledger``
                             refusal rules AND the ``SpeciesGuard``, and that
                             maps a DERIVED value to MEASURED_CITED only when
                             EVERY parent is itself measured.

THE TWO HONESTY LIMITS THAT MATTER MOST HERE
============================================
1. **SURGERY DAMAGE AND ELECTRODE DAMAGE ARE NOT SEPARABLE IN THIS MODEL.**
   The project previously established that an assumed surgery radius already
   reproduces the electrode's transmission loss.  Nothing in this module
   separates the two, and no number here may be reported as "the electrode's
   damage" as distinct from the surgery's.
2. **NO MEASURED MECHANICAL PROPERTY OF ANY INSECT CNS IS USED.**  Stiffness,
   surface tension, sheath failure strain, insertion force and micromotion
   amplitude/frequency are ALL ``ASSUMED`` with declared sweep ranges.  The only
   measured mechanical thresholds available are RODENT/GUINEA-PIG ones carried by
   ``engine/electrode_damage.py``; they are cross-species, they are forced-flagged
   on every use, and a strict ``SpeciesGuard`` REFUSES them outright (see
   ``refuse_foreign_species``).

UNITS: um, s, Hz, N, N/m, Pa, uS, nA, mV, as the reused modules declare.

ATTRIBUTION (CC BY 4.0 -- a LICENCE CONDITION, not a courtesy)
==============================================================
The neuron positions consumed by ``probe_neurons_at_risk`` come from
``data/banc/somas_v1.parquet``:

    "Distributed control circuits across a brain-and-cord connectome",
    Harvard Dataverse, doi:10.7910/DVN/7WTH1N, file ``somas_v1.parquet``
    (file id 13916460), licensed CC BY 4.0
    (http://creativecommons.org/licenses/by/4.0).

Every output dict from this module carries the attribution record
(``BANC_SOMA_ATTRIBUTION``); any redistribution must carry it too.

WHAT THIS MODULE DOES NOT DO
============================
No consciousness, perception, attention, recognition, identity, survival or
medical claim.  A hand-built research prototype, not a validated device model.
It does NOT predict real fly electrode damage.  Nothing here says the fly
notices anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from .cable import CableNeuron, Morphology
from .params import ASSUMED, DERIVED, ILLUSTRATIVE, MEASURED, Param, ProvenanceError
from .profile import DROSOPHILA_CNS, OrganismProfile
from .embodied.access_map import BANC_SOMA_ATTRIBUTION, Ledger, PROVENANCE
from .electrode_damage import (Adopted, CableDamageConfig, CavityExpansionField,
                               FlyBrainGeometry, HONESTY_STATEMENTS,
                               NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION,
                               ProximityExcitabilityCoupling,
                               ResealingTimescales, SP_RAT, SheathLayer,
                               SpeciesGuard, affected_node_indices,
                               apply_membrane_damage, build_registry,
                               preparation_variants, run_paired_cable)

__all__ = [
    "FLY_SPECIES", "SURGERY_ELECTRODE_INSEPARABLE", "HONESTY_TISSUE",
    "NO_FLY_TISSUE_MECHANICS_MEASUREMENT", "STIFFNESS_MEASUREMENT_LEAD",
    "REUSE_LEDGER", "ADDED_LEDGER", "USER_FIXED_SHAFT_DIAMETER_UM",
    "USER_FIXED_CHANNEL_PITCH_UM", "SMALL_STRAIN_LIMIT",
    "build_tissue_registry", "TissueProvenance", "refuse_foreign_species",
    "DimplingModel", "InsertionMechanics", "MicromotionStrain",
    "probe_axis_from_access_map", "probe_neurons_at_risk",
    "couple_to_excitable_model",
]

FLY_SPECIES = "Drosophila melanogaster"

#: USER-FIXED geometry.  NOT swept anywhere in this module: the brief pins them.
USER_FIXED_SHAFT_DIAMETER_UM = 7.0
USER_FIXED_CHANNEL_PITCH_UM = 20.0

#: The strain above which the linear-elastic small-strain field the project
#: already uses must not be quoted as a strain any more.  0.1 is a DECLARED
#: modelling limit (a common rule of thumb for the validity of small-strain
#: linear elasticity), not a biological threshold and not a citation.
SMALL_STRAIN_LIMIT = 0.1

SURGERY_ELECTRODE_INSEPARABLE = (
    "SURGERY DAMAGE AND ELECTRODE DAMAGE ARE NOT SEPARABLE HERE. The project "
    "previously established that an ASSUMED surgery radius already reproduces the "
    "electrode's transmission loss, so no number in this module may be reported as "
    "'the damage caused by the electrode' as distinct from the damage caused by the "
    "surgery that placed it. In the sheath-removed preparation the insertion term is "
    "plausibly SMALLER than the surgery term.")

NO_FLY_TISSUE_MECHANICS_MEASUREMENT = (
    "There is NO measured mechanical property of any Drosophila (or other insect) "
    "CNS in this module: not a shear/Young's modulus, not a surface tension, not a "
    "sheath tearing strain, not an insertion force, not a micromotion amplitude or "
    "frequency. Every one of those is ASSUMED with a declared sweep range. This "
    "agent also did not find and did not verify any such measurement in the time "
    "available, so no citation is attached to any of them.")

STIFFNESS_MEASUREMENT_LEAD = (
    "UNVERIFIED LEAD, NOT A PARAMETER VALUE: an open-access protocol exists for "
    "measuring Drosophila ventral-nerve-cord stiffness by AFM with a spherical "
    "probe and a Hertz fit (STAR Protocols 2022, doi:10.1016/j.xpro.2022.101901; "
    "the DOI and title were read on the publisher/PMC page during this run). This "
    "agent did NOT read a numeric modulus out of it and therefore uses NO number "
    "from it. It is recorded only as where a future round could get a real "
    "fly-species stiffness instead of the ASSUMED one used here.")

HONESTY_TISSUE = tuple(HONESTY_STATEMENTS) + (
    "The insertion dimpling model is ADDED here and is ASSUMED: no dimpling depth "
    "has been measured in any insect. Its two stiffness coefficients (tissue "
    "modulus, surface tension) and its sheath failure strain are free parameters "
    "with declared sweeps, and the pre-puncture dimpling depth scales LINEARLY with "
    "shaft radius, so the 7 um shaft number is a consequence of that assumption, "
    "not a measurement.",
    "MICROMOTION is ADDED here. No pulsation/breathing amplitude or frequency has "
    "been verified for the fly, so both are ASSUMED with sweeps. At the central "
    "assumed amplitude the wall shear strain predicted by the linear-elastic field "
    "exceeds 1.0, which VIOLATES the field's own small-strain assumption: the "
    "micromotion damage radius is therefore an order-of-magnitude bound, NOT a "
    "predicted strain. That limitation is reported, not hidden.",
    "The cavity-expansion field makes the WALL STRAIN size-independent (it is the "
    "assumed wall displacement fraction) while the criterion RADIUS scales linearly "
    "with shaft radius. So the model cannot say anything size-dependent about wall "
    "strain, and every size-dependent damage radius here inherits that assumption.",
    SURGERY_ELECTRODE_INSEPARABLE,
    NO_FLY_TISSUE_MECHANICS_MEASUREMENT,
)

#: Machine-readable statement of what came from where.  Written into the output.
REUSE_LEDGER = (
    {"reused": "CavityExpansionField", "from": "engine/electrode_damage.py",
     "for": "the SHAFT strain field (strain/stress/displacement vs radius) and the "
            "strain-criterion radius. InsertionMechanics.field IS an instance of it; "
            "no second field is implemented."},
    {"reused": "FlyBrainGeometry", "from": "engine/electrode_damage.py",
     "for": "damage as a fraction of the fly brain box."},
    {"reused": "SheathLayer", "from": "engine/electrode_damage.py",
     "for": "the load-bearing-sheet breach criterion (independent of, and reported "
            "next to, the dimpling criterion added here)."},
    {"reused": "affected_node_indices", "from": "engine/electrode_damage.py",
     "for": "selecting the cable rows inside the damage radius."},
    {"reused": "apply_membrane_damage", "from": "engine/electrode_damage.py",
     "for": "APPLYING the damage to the cable (-> CableNeuron.set_end_leak). This "
            "module has no membrane-damage model of its own."},
    {"reused": "ProximityExcitabilityCoupling", "from": "engine/electrode_damage.py",
     "for": "the optional proximity drive-scaling arm (default OFF)."},
    {"reused": "run_paired_cable", "from": "engine/electrode_damage.py",
     "for": "an INDEPENDENT cross-check of this module's paired inserted-vs-sham "
            "run at the same radius and leak density."},
    {"reused": "SpeciesGuard / Adopted / audit_species_labeling",
     "from": "engine/electrode_damage.py",
     "for": "refusing (or force-flagging) every rodent parameter."},
    {"reused": "build_registry / ParamRegistry / Param / ProvenanceError",
     "from": "engine/params.py",
     "for": "parameter provenance; this module EXTENDS the existing registry rather "
            "than replacing it."},
    {"reused": "Ledger / PROVENANCE", "from": "engine/embodied/access_map.py",
     "for": "the ledger rules that refuse MEASURED_CITED without a source and "
            "ASSUMED without a sweep range."},
    {"reused": "Morphology / CableNeuron", "from": "engine/cable.py",
     "for": "the passive multicompartment cable."},
    {"reused": "BANC_SOMA_ATTRIBUTION", "from": "engine/embodied/access_map.py",
     "for": "the CC BY 4.0 attribution record carried into every output."},
    {"added": "DimplingModel", "in": "this module",
     "for": "pre-penetration surface indentation (was NOT modelled anywhere)."},
    {"added": "MicromotionStrain", "in": "this module",
     "for": "cyclic shear from physiological motion acting on tissue that already "
            "contains the probe (was NOT modelled anywhere)."},
    {"added": "probe_neurons_at_risk", "in": "this module",
     "for": "connecting the mechanics to the REAL captured neurons of the access "
            "map (the access map itself declares 'No tissue mechanics')."},
    {"added": "couple_to_excitable_model", "in": "this module",
     "for": "driving apply_membrane_damage / ProximityExcitabilityCoupling from the "
            "mechanics and reporting the measured in-model effect."},
    {"added": "TissueProvenance", "in": "this module",
     "for": "holding the Ledger refusal rules and the SpeciesGuard together, and "
            "for refusing to label a DERIVED value as cited unless every parent is "
            "measured."},
)


# ----------------------------------------------------------------------
# small numeric helpers (NOT models -- guard rails)
# ----------------------------------------------------------------------
def _num(value, name, minimum=None, strict=False):
    x = float(value)
    if not math.isfinite(x) or (minimum is not None and
                                (x <= minimum if strict else x < minimum)):
        raise ValueError("invalid %s: %r" % (name, value))
    return x


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------
# 1. parameter registry: EXTENDS engine.electrode_damage's registry
# ----------------------------------------------------------------------
def build_tissue_registry(name="electrode_tissue"):
    """The existing electrode-damage registry PLUS this module's new parameters.

    Extending rather than replacing is deliberate: every parameter the previous
    round registered (including every RODENT measured value and its citation)
    stays visible to the guard and to the report.
    """
    r = build_registry(name)

    # ---- the USER-FIXED geometry.  ILLUSTRATIVE = "chosen by us", no sweep,
    # ---- because the brief pins these two numbers and forbids changing them.
    r.add(Param("user_fixed_shaft_diameter_um", USER_FIXED_SHAFT_DIAMETER_UM, "um",
                ILLUSTRATIVE, species=FLY_SPECIES,
                notes="USER-FIXED DESIGN INPUT, NOT a measurement and NOT swept: the "
                      "brief fixes the shaft diameter at 7 um. Recorded in the ledger "
                      "as ENGINEERING_DEFAULT for exactly that reason."))
    r.add(Param("user_fixed_channel_pitch_um", USER_FIXED_CHANNEL_PITCH_UM, "um",
                ILLUSTRATIVE, species=FLY_SPECIES,
                notes="USER-FIXED DESIGN INPUT, NOT a measurement and NOT swept: the "
                      "brief fixes the channel pitch at 20 um."))

    # ---- ADDED, all ASSUMED, all with a declared sweep range -----------------
    r.add(Param("assumed_tissue_effective_modulus_Pa", 1000.0, "Pa", ASSUMED,
                species=None, range=(20.0, 20000.0),
                notes="REDUCED (plane-strain) elastic modulus E* = E/(1-nu^2) of fly "
                      "CNS tissue, used ONLY in the ADDED dimpling contact stiffness. "
                      "NO insect CNS modulus was verified by this agent (see "
                      "STIFFNESS_MEASUREMENT_LEAD for where a real one could come "
                      "from). Swept 20-20000 Pa."))
    r.add(Param("assumed_surface_tension_N_per_m", 2.0e-3, "N/m", ASSUMED,
                species=None, range=(1e-4, 5e-2),
                notes="Effective in-plane TENSION of the tissue surface plus the "
                      "load-bearing sheath, resisting the dimple. Used only in the "
                      "ADDED dimpling contact stiffness 2*pi*T. NO measured insect "
                      "value is claimed; swept 1e-4 to 5e-2 N/m (two and a half "
                      "decades), which is what makes the implied contact stiffness "
                      "and therefore the implied pre-puncture FORCE a range and not a "
                      "number."))
    r.add(Param("assumed_sheath_failure_strain", 0.30, "1", ASSUMED, species=None,
                range=(0.05, 1.0),
                notes="In-plane stretch at which the load-bearing sheet tears during "
                      "dimpling. This is the ADDED dimpling rupture criterion; the "
                      "project's existing SheathLayer uses a separate, independent "
                      "OPENING-RADIUS criterion. Neither is measured in any insect. "
                      "Swept 0.05-1.0."))
    r.add(Param("assumed_insertion_force_uN", 1.0, "uN", ASSUMED, species=None,
                range=(1e-3, 100.0),
                notes="Axial force pushing the probe into the tissue before it "
                      "penetrates. NO insect insertion force is measured anywhere "
                      "(negative result). Swept 1e-3 to 100 uN; the pre-puncture "
                      "regime ends at the model's own rupture force, so this force is "
                      "only meaningful below it."))
    r.add(Param("assumed_micromotion_amplitude_um", 5.0, "um", ASSUMED, species=None,
                range=(0.05, 50.0),
                notes="Peak relative DISPLACEMENT between the tissue and the "
                      "implanted probe (pulsation / breathing / locomotion), which is "
                      "the input of the ADDED MicromotionStrain. NO fly value was "
                      "verified by this agent and NONE is taken from the rodent "
                      "literature. Swept 0.05-50 um; the sweep IS the result, because "
                      "the criterion radius scales with it."))
    r.add(Param("assumed_micromotion_frequency_Hz", 3.0, "Hz", ASSUMED, species=None,
                range=(0.05, 50.0),
                notes="Frequency of that cyclic displacement. Reported for the "
                      "loading-rate context ONLY: the model is quasi-static, so this "
                      "parameter does NOT enter any strain. Swept 0.05-50 Hz."))
    r.add(Param("assumed_proxy_cable_length_um", 120.0, "um", ASSUMED, species=None,
                range=(20.0, 400.0),
                notes="Length of the PROXY passive cable used to couple the damage "
                      "into engine.cable. The BANC soma table gives ONE POINT per "
                      "neuron, so no real morphology exists for these neurons; the "
                      "proxy is a declared stand-in, swept 20-400 um."))
    r.add(Param("assumed_proxy_cable_diameter_um", 1.0, "um", ASSUMED, species=None,
                range=(0.1, 5.0),
                notes="Diameter of that proxy cable. ASSUMED: no verified Drosophila "
                      "neurite diameter is used here. Swept 0.1-5 um. It sets the "
                      "membrane area, and therefore the per-compartment leak "
                      "conductance, linearly."))
    r.add(Param("assumed_proxy_cable_nseg", 50.0, "compartments", ASSUMED, species=None,
                range=(10.0, 400.0),
                notes="Discretisation of the proxy cable (float because every "
                      "registered parameter is a float; used as int()). Swept "
                      "10-400 compartments so that the damaged patch length is not a "
                      "discretisation artefact."))

    # ---- DERIVED, but NOT all parents measured => must be swept ---------------
    r.add(Param("derived_mechanoporation_shear_strain_threshold", 14.0 / 500.0, "1",
                DERIVED,
                derived_from=["rodent_mechanoporation_shear_Pa",
                              "assumed_tissue_shear_modulus_Pa"],
                species=SP_RAT,
                notes="gamma = tau/G: the rat mechanoporation STRESS threshold "
                      "divided by the ASSUMED fly shear modulus. Because one parent "
                      "is only ASSUMED, this value is recorded in the ledger as "
                      "ASSUMED with an explicit sweep (0.0028-0.28 for G = "
                      "5000-50 Pa), NOT as a cited measurement. The rat threshold "
                      "also carries the source's own caveat: NO cell death occurred "
                      "at those parameters."))

    # ---- a lead, not a parameter: recorded so a reader can see what was
    # ---- searched for and NOT found.  Deliberately ILLUSTRATIVE and unused.
    r.add(Param("fly_cns_modulus_measurement_status", 0.0, "flag", ILLUSTRATIVE,
                species=FLY_SPECIES,
                notes="PLACEHOLDER ROW (value 0, never used as a number): it exists "
                      "so that the honesty statements and STIFFNESS_MEASUREMENT_LEAD "
                      "travel with the registry. " + NO_FLY_TISSUE_MECHANICS_MEASUREMENT))

    r.validate()
    return r


# ----------------------------------------------------------------------
# 2. the ledger: Ledger refusal rules + SpeciesGuard, in one object
# ----------------------------------------------------------------------
class TissueProvenance:
    """A provenance ledger for tissue mechanics that REFUSES, not merely records.

    Refusals enforced here (all four are exercised in ``run_electrode_tissue.py``
    and the refusals are reported, not just claimed):

      R1  MEASURED_CITED with no source  -> refused by the reused
          ``access_map.Ledger`` (re-raised here as ``ProvenanceError``).
      R2  ASSUMED with no sweep range    -> refused by the reused Ledger.
      R3  a foreign-species parameter used in a Drosophila model without the
          forced cross-species flag and its citation -> refused here.
      R4  building the model at all with a strict SpeciesGuard (allow_cross=False)
          while the registry holds rodent parameters -> refused by ``SpeciesGuard``
          (``ProvenanceError`` at construction).  See ``refuse_foreign_species``.

    One rule is STRICTER than the underlying machinery: a ``DERIVED`` parameter is
    labelled MEASURED_CITED only when EVERY parent is itself measured (recursively).
    Otherwise it is recorded as ASSUMED and must carry a sweep.  The registry's own
    ``Param`` class accepts ``derived`` without that check, so this closes a real
    gap: ``gamma = rat_stress / assumed_modulus`` is not a measured quantity.
    """

    #: engine.params provenance -> the ledger's four classes
    def __init__(self, profile=DROSOPHILA_CNS, registry=None, allow_cross=True):
        if not isinstance(profile, OrganismProfile):
            raise TypeError("profile must be an engine.profile.OrganismProfile")
        self.registry = registry if registry is not None else build_tissue_registry()
        self.species_guard = SpeciesGuard(profile, self.registry, allow_cross=allow_cross)
        self.ledger = Ledger()                 # REUSED refusal rules live here
        self._snapshot = self.registry.snapshot()
        self._adopted = []
        self._species_entries = []
        self._refusals = []
        self._caller_sweeps = []
        self._cache = {}
        self._readoptions = []

    # -- adoption ----------------------------------------------------------
    def adopt(self, name, *, sweep=None, note="", ledger_class=None):
        """guard.adopt + ledger.record, with the ledger's refusals preserved.

        IDEMPOTENT: adopting the same parameter twice returns the FIRST ledger
        record.  A later call with a DIFFERENT sweep is recorded as a re-adoption
        note instead of being dropped or silently overwriting the first one.
        """
        adopted = self.species_guard.adopt(name)     # R4 / R3 machinery
        return adopted, self.record_adopted(adopted, sweep=sweep, note=note,
                                            ledger_class=ledger_class)

    def record_adopted(self, adopted, *, sweep=None, note="", ledger_class=None,
                       force_cross_flag=True):
        if not isinstance(adopted, Adopted):
            raise TypeError("record_adopted expects an Adopted record")
        if not any(a is adopted for a in self._adopted):
            self._adopted.append(adopted)
        if adopted.name in self._cache:               # idempotent adoption
            prior = self._cache[adopted.name]
            if sweep is not None and [float(v) for v in sweep] != list(prior.sweep or []):
                self._readoptions.append({
                    "parameter": adopted.name, "first_sweep": prior.sweep,
                    "later_sweep": [float(v) for v in sweep],
                    "note": "the FIRST adoption's sweep is the one in the ledger"})
            return prior
        if adopted.cross_species and not force_cross_flag:
            raise ProvenanceError(
                "refusing to record %r: it was measured in %r, not in %r, and the "
                "forced cross-species flag was explicitly withheld. A foreign-species "
                "parameter may only enter a Drosophila model force-flagged and "
                "carrying its citation." % (adopted.name, adopted.species,
                                            self.species_guard.profile.species))
        provenance, kw = self._ledger_class(adopted, ledger_class, sweep)
        if provenance == PROVENANCE.ASSUMED and sweep is None:
            # a registered range IS a declared sweep: use it rather than demanding
            # that every caller repeat what the registry already states
            declared = self._snapshot[adopted.name].get("range")
            if declared is not None:
                sweep = tuple(float(v) for v in declared)
        if (provenance == PROVENANCE.ASSUMED and sweep is not None
                and self._snapshot[adopted.name].get("range") is None):
            # the parameter is ASSUMED and the EARLIER round registered no range for
            # it; the sweep therefore comes from this round and is recorded as such
            self._caller_sweeps.append({"parameter": adopted.name,
                                        "sweep": [float(v) for v in sweep],
                                        "why": "ASSUMED in the existing registry with "
                                               "NO declared range; a sweep had to be "
                                               "supplied for the ledger to accept it"})
        src = adopted.source
        full_note = " ".join(x for x in (note, adopted.note) if x)
        try:
            record = self.ledger.record(adopted.name, adopted.value, adopted.unit,
                                        provenance, source=src, note=full_note,
                                        sweep=sweep)
        except ValueError as exc:                    # R1 / R2, re-raised
            raise ProvenanceError("%s (ledger refusal: %s)" % (exc, adopted.name))
        self._species_entries.append({
            "parameter": adopted.name, "value": adopted.value, "unit": adopted.unit,
            "engine_params_provenance": adopted.provenance,
            "ledger_class": provenance, "species": adopted.species,
            "cross_species": bool(adopted.cross_species),
            "forced_cross_species_flag": bool(adopted.forced_flag),
            "source": src, "sweep": None if sweep is None else list(sweep),
        })
        self._cache[adopted.name] = record
        return record

    def _all_parents_measured(self, name, _seen=None):
        _seen = set() if _seen is None else _seen
        if name in _seen:
            return False
        _seen.add(name)
        p = self._snapshot.get(name)
        if p is None:
            return False
        if p["provenance"] == MEASURED:
            return bool(p["source"])
        if p["provenance"] == DERIVED:
            parents = p.get("derived_from") or []
            return bool(parents) and all(self._all_parents_measured(x, _seen)
                                         for x in parents)
        return False

    def _ledger_class(self, adopted, ledger_class, sweep):
        if ledger_class is not None:
            if ledger_class not in PROVENANCE.ALL:
                raise ValueError("unknown ledger class %r" % (ledger_class,))
            return ledger_class, {}
        if adopted.provenance == MEASURED:
            return PROVENANCE.MEASURED_CITED, {}
        if adopted.provenance == DERIVED:
            if self._all_parents_measured(adopted.name):
                return PROVENANCE.MEASURED_CITED, {}
            # a derived value over an ASSUMED parent is NOT a cited measurement
            if sweep is None:
                param_range = self._snapshot[adopted.name].get("range")
                if param_range is None:
                    raise ProvenanceError(
                        "refusing %r: derived from a non-measured parent, so it is "
                        "recorded as ASSUMED -- and an ASSUMED entry needs a declared "
                        "sweep range, which neither the caller nor the registry "
                        "supplies." % (adopted.name,))
            return PROVENANCE.ASSUMED, {}
        if adopted.provenance == ILLUSTRATIVE:
            return PROVENANCE.ENGINEERING_DEFAULT, {}
        return PROVENANCE.ASSUMED, {}          # engine ASSUMED -> ledger ASSUMED

    # -- direct recording, used for computed results -----------------------
    def record(self, key, value, unit, provenance, *, source=None, note="",
               sweep=None):
        try:
            return self.ledger.record(key, value, unit, provenance, source=source,
                                      note=note, sweep=sweep)
        except ValueError as exc:
            raise ProvenanceError(str(exc))

    # -- reports -----------------------------------------------------------
    def foreign_parameters_used(self):
        return [e for e in self._species_entries if e["cross_species"]]

    def record_refusal(self, rule, attempted, message):
        entry = {"rule": rule, "attempted": attempted, "refused": True,
                 "message": str(message)}
        self._refusals.append(entry)
        return entry

    def as_dict(self):
        return {
            "ledger_entries": self.ledger.as_list(),
            "ledger_counts": self.ledger.counts(),
            "unresolved_entries": [e["key"] for e in self.ledger.unresolved()],
            "species_guard": self.species_guard.record(self._adopted),
            "species_entries": self._species_entries,
            "foreign_parameters_used": [e["parameter"] for e in
                                        self.foreign_parameters_used()],
            "refusal_demonstrations": self._refusals,
            "caller_supplied_sweeps": self._caller_sweeps,
            "re_adoptions_with_a_different_sweep": self._readoptions,
            "rules": {
                "R1": "MEASURED_CITED without a source: REFUSED (reused "
                      "access_map.Ledger rule).",
                "R2": "ASSUMED without a declared sweep range: REFUSED (reused "
                      "access_map.Ledger rule).",
                "R3": "foreign-species parameter without the forced cross-species "
                      "flag + citation: REFUSED here.",
                "R4": "SpeciesGuard(allow_cross=False) on this registry: RAISES at "
                      "construction, listing every rodent parameter.",
                "R5": "DERIVED is labelled MEASURED_CITED only when EVERY parent is "
                      "measured (stricter than engine.params, which does not check "
                      "parents); otherwise it must be swept.",
            },
            "attribution": dict(BANC_SOMA_ATTRIBUTION),
        }


def refuse_foreign_species(registry=None, profile=DROSOPHILA_CNS):
    """Demonstrate rule R4: build a STRICT guard and return its refusal.

    Returns a dict with the refusal message.  Never silently succeeds: if the
    strict guard does NOT raise, that is itself an error worth failing on, so a
    ``ProvenanceError`` is raised to say so.
    """
    try:
        SpeciesGuard(profile, registry, allow_cross=False)
    except ProvenanceError as exc:
        return {"rule": "R4", "refused": True, "raised": "ProvenanceError",
                "message": str(exc)[:1200]}
    raise ProvenanceError(
        "the strict SpeciesGuard did NOT refuse: the registry apparently holds no "
        "foreign-species measured parameter, which contradicts the documented state "
        "of engine/electrode_damage.py")


# ----------------------------------------------------------------------
# 3. the ADDED dimpling model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DimplingModel:
    """Surface indentation (dimpling) of the sheath BEFORE the probe penetrates.

    TWO independent, both-ASSUMED statements, reported separately:

    (1) DEPTH AT A STATED FORCE -- two contact springs in parallel::

            k    = 2*a*E*        (rigid flat punch on an elastic half-space)
                 + 2*pi*T        (tensed surface / load-bearing sheet)
            d(F) = F / k

        ``k`` has units N/m (a stiffness), both terms are stiffnesses, and the sum
        is the model's ASSUMPTION.  The flat-punch modulus form and the 2*pi*T
        sheet term are standard FUNCTION FORMS chosen here; this agent did not
        verify a primary source for either in the time available, so neither is
        registered as MEASURED_CITED.  The elastic term scales with shaft radius;
        the tension term does not, which is what makes the depth-vs-diameter curve
        bend rather than being exactly 1/a.

    (2) DEPTH AT SHEET FAILURE -- a shallow spherical cap of footprint radius ``a``
        and depth ``d`` has meridian arc length ``a*(1 + (2/3)*(d/a)^2)``, so the
        in-plane stretch is ``eps = (2/3)*(d/a)^2`` (the 2/3 is pure geometry of the
        arc-length expansion, and is registered ILLUSTRATIVE for that reason).  A
        failure stretch ``eps_f`` (ASSUMED) therefore gives::

            d_fail = a * sqrt(eps_f / (2/3))

        This is the pre-penetration dimpling depth: beyond it the sheet tears, the
        probe is through, and the dimpling phase is over.

    NUMBERS FROM THIS MODEL ARE NOT MEASUREMENTS.  No dimpling depth has been
    measured in any insect.
    """

    shaft_radius_um: float
    effective_modulus_Pa: float
    surface_tension_N_per_m: float
    failure_strain: float
    cap_coefficient: float = 2.0 / 3.0

    #: geometric factor of the shallow-cap arc-length expansion
    CAP_COEFFICIENT_GEOMETRIC = 2.0 / 3.0

    def __post_init__(self):
        object.__setattr__(self, "shaft_radius_um",
                           _num(self.shaft_radius_um, "shaft_radius_um", 0.0, True))
        object.__setattr__(self, "effective_modulus_Pa",
                           _num(self.effective_modulus_Pa, "effective_modulus_Pa",
                                0.0, True))
        object.__setattr__(self, "surface_tension_N_per_m",
                           _num(self.surface_tension_N_per_m,
                                "surface_tension_N_per_m", 0.0))
        object.__setattr__(self, "failure_strain",
                           _num(self.failure_strain, "failure_strain", 0.0, True))
        object.__setattr__(self, "cap_coefficient",
                           _num(self.cap_coefficient, "cap_coefficient", 0.0, True))

    # -- (1) depth at a stated force --------------------------------------
    @property
    def elastic_stiffness_N_per_m(self):
        return 2.0 * self.shaft_radius_um * 1e-6 * self.effective_modulus_Pa

    @property
    def tension_stiffness_N_per_m(self):
        return 2.0 * math.pi * self.surface_tension_N_per_m

    @property
    def contact_stiffness_N_per_m(self):
        """Sum of the two terms: the model's ASSUMED total stiffness."""
        return self.elastic_stiffness_N_per_m + self.tension_stiffness_N_per_m

    @property
    def elastic_fraction_of_stiffness(self):
        return self.elastic_stiffness_N_per_m / self.contact_stiffness_N_per_m

    def depth_um(self, force_uN):
        f = _num(force_uN, "force_uN", 0.0)
        return f * 1e-6 / self.contact_stiffness_N_per_m * 1e6

    def force_uN_for_depth(self, depth_um):
        d = _num(depth_um, "depth_um", 0.0)
        return self.contact_stiffness_N_per_m * d * 1e-6 * 1e6

    # -- (2) depth at sheet failure ---------------------------------------
    def failure_depth_um(self):
        """Pre-penetration dimpling depth: LINEAR in shaft radius (assumed form)."""
        return self.shaft_radius_um * math.sqrt(self.failure_strain /
                                                self.cap_coefficient)

    def failure_depth_over_diameter(self):
        return self.failure_depth_um() / (2.0 * self.shaft_radius_um)

    def failure_force_uN(self):
        return self.force_uN_for_depth(self.failure_depth_um())

    def cap_strain_at(self, depth_um):
        d = np.asarray(depth_um, dtype=float)
        return self.cap_coefficient * (d / self.shaft_radius_um) ** 2

    def to_dict(self):
        return {
            "model": "ADDED: two-term contact stiffness (2*a*E* + 2*pi*T) plus a "
                     "shallow-spherical-cap failure strain",
            "provenance": "ASSUMED; no dimpling depth, no insect CNS modulus, no "
                          "insect surface tension and no insect sheath failure strain "
                          "was verified. The two contact stiffness terms are standard "
                          "FUNCTION FORMS chosen here, not verified citations.",
            "shaft_radius_um": self.shaft_radius_um,
            "shaft_diameter_um": 2.0 * self.shaft_radius_um,
            "effective_modulus_Pa": self.effective_modulus_Pa,
            "surface_tension_N_per_m": self.surface_tension_N_per_m,
            "failure_strain": self.failure_strain,
            "cap_coefficient": self.cap_coefficient,
            "elastic_stiffness_N_per_m": self.elastic_stiffness_N_per_m,
            "tension_stiffness_N_per_m": self.tension_stiffness_N_per_m,
            "contact_stiffness_N_per_m": self.contact_stiffness_N_per_m,
            "elastic_fraction_of_stiffness": self.elastic_fraction_of_stiffness,
            "failure_depth_um": self.failure_depth_um(),
            "failure_depth_over_diameter": self.failure_depth_over_diameter(),
            "failure_force_uN": self.failure_force_uN(),
            "validity": "depth at a stated force is meaningful only BELOW the failure "
                        "depth; above it the probe has penetrated and the "
                        "pre-penetration regime is over",
            "no_fly_measurement": NO_FLY_TISSUE_MECHANICS_MEASUREMENT,
            "measurement_lead": STIFFNESS_MEASUREMENT_LEAD,
        }


# ----------------------------------------------------------------------
# 4. ADDED: InsertionMechanics  (a) dimpling + (b) the shaft strain field
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class InsertionMechanics:
    """Tissue displacement during insertion, at the USER-FIXED shaft diameter.

    (a) DIMPLING before penetration  -> ``self.dimpling`` (ADDED here).
    (b) STRAIN FIELD around the shaft -> ``self.field``, an instance of the
        project's EXISTING ``CavityExpansionField``, evaluated at
        ``shaft_radius_um = 7/2 um`` (REUSED; every strain number below is that
        class's output, not a re-derivation).

    The field's own documented limits apply unchanged: plane strain, linear
    elastic, small strain, incompressible, no plasticity, no friction, no tearing,
    defined only for r >= a.  Read ``CavityExpansionField``'s docstring.
    """

    field: CavityExpansionField
    dimpling: DimplingModel
    sheath: SheathLayer
    geometry: FlyBrainGeometry
    insertion_depth_um: float
    shaft_diameter_um: float
    variant_name: str = "sheath_intact"

    MECHANISM = "insertion_cavity_expansion"

    def __post_init__(self):
        if not isinstance(self.field, CavityExpansionField):
            raise TypeError("field must be the project's CavityExpansionField")
        if not isinstance(self.dimpling, DimplingModel):
            raise TypeError("dimpling must be a DimplingModel")
        if abs(2.0 * self.field.shaft_radius_um - self.shaft_diameter_um) > 1e-9:
            raise ValueError("shaft_diameter_um must equal twice the field's radius")
        object.__setattr__(self, "insertion_depth_um",
                           _num(self.insertion_depth_um, "insertion_depth_um",
                                0.0, True))
        object.__setattr__(self, "shaft_diameter_um",
                           _num(self.shaft_diameter_um, "shaft_diameter_um", 0.0, True))

    # -- factory -----------------------------------------------------------
    @classmethod
    def from_provenance(cls, prov, *, shaft_diameter_um=None, insertion_depth_um=None,
                        variant_name="sheath_intact"):
        """Build from a :class:`TissueProvenance`, adopting every parameter.

        Every number enters through ``prov.adopt`` so it is registered, got a
        ledger class, and -- if it is rodent -- got the forced cross-species flag.
        """
        if not isinstance(prov, TissueProvenance):
            raise TypeError("prov must be a TissueProvenance")
        if shaft_diameter_um is None:
            shaft_diameter_um = prov.adopt("user_fixed_shaft_diameter_um")[0].value
        else:
            _num(shaft_diameter_um, "shaft_diameter_um", 0.0, True)
        a = 0.5 * float(shaft_diameter_um)
        # NOTE: several ASSUMED parameters in the EXISTING engine/electrode_damage.py
        # registry declare NO range.  This module's ledger refuses to record an
        # ASSUMED value without a sweep, so a sweep is supplied HERE for each of
        # them, and every such entry is listed in ``caller_supplied_sweeps`` so a
        # reader can see which ranges came from this round rather than from the
        # earlier one.
        wall = prov.adopt("assumed_wall_displacement_fraction",
                          sweep=(0.05, 0.25, 0.5, 1.0))[0].value
        shear = prov.adopt("assumed_tissue_shear_modulus_Pa",
                           sweep=(50.0, 200.0, 500.0, 2000.0, 5000.0))[0].value
        modulus = prov.adopt("assumed_tissue_effective_modulus_Pa")[0].value
        tension = prov.adopt("assumed_surface_tension_N_per_m")[0].value
        failure = prov.adopt("assumed_sheath_failure_strain")[0].value
        thickness = prov.adopt("fly_sheath_thickness_um",
                               sweep=(1.0, 3.0, 5.0, 10.0))[0].value
        opening = prov.adopt("assumed_sheath_critical_opening_radius_um")[0].value
        delam = prov.adopt("assumed_sheath_delamination_multiple",
                           sweep=(0.0, 2.0, 5.0))[0].value
        depth = (prov.adopt("assumed_insertion_depth_um",
                            sweep=(25.0, 50.0, 100.0))[0].value
                 if insertion_depth_um is None else float(insertion_depth_um))
        geometry = FlyBrainGeometry.from_adopt(lambda n: prov.adopt(n)[0])
        variants = preparation_variants(lambda n: prov.adopt(n)[0])
        if variant_name not in variants:
            raise KeyError("unknown preparation variant %r" % (variant_name,))
        prov.record("variant_used", variant_name, "enum", PROVENANCE.ENGINEERING_DEFAULT,
                    note="declared preparation; sheath_intact vs sheath_removed is a "
                         "DECLARED choice, not a measurement")
        field_ = CavityExpansionField(shaft_radius_um=a,
                                      wall_displacement_fraction=float(wall),
                                      shear_modulus_Pa=float(shear))
        dimpling = DimplingModel(shaft_radius_um=a,
                                 effective_modulus_Pa=float(modulus),
                                 surface_tension_N_per_m=float(tension),
                                 failure_strain=float(failure))
        sheath = SheathLayer(thickness_um=float(thickness),
                             critical_opening_radius_um=float(opening),
                             delamination_multiple=float(delam),
                             wall_displacement_fraction=float(wall))
        return cls(field=field_, dimpling=dimpling, sheath=sheath, geometry=geometry,
                   insertion_depth_um=float(depth),
                   shaft_diameter_um=float(shaft_diameter_um),
                   variant_name=variant_name)

    # -- (a) dimpling ------------------------------------------------------
    @property
    def shaft_radius_um(self):
        return self.field.shaft_radius_um

    def dimpling_depth_um(self, force_uN=None):
        return self.dimpling.depth_um(force_uN)

    def pre_penetration_dimpling_um(self):
        """The dimpling depth at which the sheet fails: the pre-penetration depth."""
        return self.dimpling.failure_depth_um()

    def dimpling_regime_for(self, force_uN):
        f_break = self.dimpling.failure_force_uN()
        below = float(force_uN) < f_break
        return {"force_uN": float(force_uN),
                "rupture_force_uN": f_break,
                "regime": "pre_penetration" if below else "POST_RUPTURE",
                "note": ("the two-term contact model is only a DIMMLING model below "
                         "the rupture force; above it the probe has penetrated and "
                         "the model is out of its own domain")}

    def cap_strain_at_depth(self, depth_um):
        return self.dimpling.cap_strain_at(depth_um)

    # -- (b) the REUSED shaft strain field ---------------------------------
    def displacement_um(self, r_um):
        return self.field.displacement_um(r_um)

    def hoop_strain(self, r_um):
        return self.field.hoop_strain(r_um)

    def radial_strain(self, r_um):
        return self.field.radial_strain(r_um)

    def max_abs_principal_strain(self, r_um):
        return self.field.max_abs_principal_strain(r_um)

    def max_shear_strain(self, r_um):
        return self.field.max_shear_strain(r_um)

    @property
    def wall_strain(self):
        """eps at r = a: exactly the assumed wall displacement fraction.

        SIZE-INDEPENDENT by construction of the assumed delta ~ a form.  Reported
        because a reader will otherwise read a size trend into it.
        """
        return self.field.wall_displacement_fraction

    def wall_pressure_Pa(self):
        return self.field.wall_pressure_Pa

    def strain_profile(self, r_um):
        return self.field.profile_arrays(r_um)

    # -- criterion ---------------------------------------------------------
    def damage_radius_um(self, strain_threshold):
        """REUSED: the strain-threshold crossing of the cavity-expansion field."""
        return self.field.strain_radius_um(strain_threshold)

    def damage_radius_over_shaft_radius(self, strain_threshold):
        return self.damage_radius_um(strain_threshold) / self.shaft_radius_um

    def damage_fraction_of_brain(self, strain_threshold):
        """REUSED FlyBrainGeometry: damage as a fraction of the fly brain box."""
        r = self.damage_radius_um(strain_threshold)
        sheath_breached = bool(self.sheath.breached(self.shaft_radius_um))
        mult = (self.sheath.herniation_multiplier if sheath_breached else 1.0)
        return {"damage_radius_um": r, "sheath_breached": sheath_breached,
                "parenchymal_multiplier": mult,
                "parenchymal_radius_um": r * mult,
                "fraction_of_brain_volume": self.geometry.damage_fraction(r * mult)}

    def sheath_breach(self):
        return {"sheath_present": self.variant_name == "sheath_intact",
                "shaft_radius_um": self.shaft_radius_um,
                "critical_opening_radius_um": self.sheath.critical_opening_radius_um,
                "breached_by_opening_radius_criterion":
                    bool(self.sheath.breached(self.shaft_radius_um)),
                "delamination_radius_um":
                    self.sheath.delamination_radius_um(self.shaft_radius_um),
                "reused_from": "SheathLayer (engine/electrode_damage.py)",
                "provenance": "the opening-radius criterion is an UNMEASURED free "
                              "parameter; this is INDEPENDENT of the dimpling-failure "
                              "criterion added in this module and the two are "
                              "reported side by side"}

    def to_dict(self):
        return {"mechanism": self.MECHANISM,
                "reused": {"strain_field": "CavityExpansionField",
                           "sheath": "SheathLayer", "geometry": "FlyBrainGeometry"},
                "added": ["DimplingModel (pre-penetration surface indentation)"],
                "shaft_diameter_um": self.shaft_diameter_um,
                "shaft_radius_um": self.shaft_radius_um,
                "shaft_diameter_is_user_fixed": True,
                "field": self.field.to_dict(),
                "dimpling": self.dimpling.to_dict(),
                "wall_strain_size_independent": self.wall_strain,
                "sheath_breach": self.sheath_breach(),
                "insertion_depth_um": self.insertion_depth_um,
                "surgery_electrode_caveat": SURGERY_ELECTRODE_INSEPARABLE}


# ----------------------------------------------------------------------
# 5. ADDED: MicromotionStrain
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MicromotionStrain:
    """Shear strain from PHYSIOLOGICAL MICROMOTION, on tissue already containing
    the probe.  This is the mechanism by which a chronically implanted probe keeps
    damaging tissue, and the project did not model it.

    MODEL (one assumption, stated):
      the probe is anchored while the tissue moves by a cyclic peak relative
      displacement ``A``.  The relative displacement field is taken to be the SAME
      incompressible plane-strain kernel the project already uses, with the static
      wall displacement ``delta`` replaced by the CYCLIC amplitude ``A``::

          u(r) = A*a/r,      gamma(r) = 2*A*a/r^2,      tau(r) = G*gamma(r)

      so the wall shear strain is ``gamma(a) = 2*(A/a)/... `` -- explicitly,
      ``gamma_wall = 2*A/a`` at r = a.  The kernel is IDENTICAL to
      ``CavityExpansionField.max_shear_strain`` (that class's ``_c`` is
      ``delta*a``); ``kernel_matches_cavity_field()`` checks the numerical
      agreement to machine precision in the regime where the class accepts the
      amplitude, and ``as_cavity_field()`` returns the class instance there.

    THE LIMIT THAT MATTERS: this kernel is linear-elastic and small-strain.  A few
    micrometres of relative motion across a 3.5 um shaft radius gives a wall shear
    strain of order 1, which is FAR outside that assumption.  The model therefore
    reports, for every result, whether the field is being quoted outside its own
    validity, and the criterion radius in that regime is an ORDER-OF-MAGNITUDE
    BOUND rather than a predicted strain.

    NOT MODELLED: viscoelasticity (frequency does not enter any strain), tissue
    sliding along the probe, the fibrous capsule that forms around a chronic
    implant, and any anchor stiffness.  No fly amplitude or frequency is verified.
    """

    shaft_radius_um: float
    amplitude_um: float
    frequency_Hz: float
    shear_modulus_Pa: float

    MECHANISM = "micromotion_shear"

    def __post_init__(self):
        for name in ("shaft_radius_um", "amplitude_um", "frequency_Hz",
                     "shear_modulus_Pa"):
            object.__setattr__(self, name, _num(getattr(self, name), name, 0.0, True))

    # -- the kernel (identical to the reused field's; see the class docstring) --
    @property
    def _c_um2(self):
        return self.amplitude_um * self.shaft_radius_um

    def _r(self, r_um):
        r = np.asarray(r_um, dtype=float)
        if not np.isfinite(r).all():
            raise ValueError("radius must be finite")
        if np.any(r < self.shaft_radius_um):
            raise ValueError(
                "the relative-displacement field is defined only outside the shaft "
                "wall (r >= a = %g um); got r_min = %g um"
                % (self.shaft_radius_um, float(np.min(r))))
        return r

    @staticmethod
    def _wrap(r_um, out):
        return float(out) if np.ndim(r_um) == 0 else out

    def displacement_um(self, r_um):
        return self._wrap(r_um, self._c_um2 / self._r(r_um))

    def max_shear_strain(self, r_um):
        return self._wrap(r_um, 2.0 * self._c_um2 / self._r(r_um) ** 2)

    def shear_stress_Pa(self, r_um):
        return self.shear_modulus_Pa * np.asarray(self.max_shear_strain(r_um))

    # -- the reuse checks --------------------------------------------------
    @property
    def amplitude_ratio(self):
        """A/a: the dimensionless driver of the whole mechanism."""
        return self.amplitude_um / self.shaft_radius_um

    @property
    def wall_shear_strain(self):
        return 2.0 * self.amplitude_ratio

    def as_cavity_field(self):
        """The REUSED field class, when the amplitude is inside ITS domain.

        ``CavityExpansionField`` refuses ``wall_displacement_fraction > 1``, i.e. it
        refuses an amplitude larger than the shaft radius.  Returning ``None`` there
        is the honest answer: the module does not silently rescale a class that has
        declared its own domain.
        """
        if self.amplitude_um > self.shaft_radius_um:
            return None
        return CavityExpansionField(shaft_radius_um=self.shaft_radius_um,
                                    wall_displacement_fraction=self.amplitude_ratio,
                                    shear_modulus_Pa=self.shear_modulus_Pa)

    def kernel_matches_cavity_field(self, r_um=()):
        """Max |difference| between this kernel and the reused class, in-domain."""
        field_ = self.as_cavity_field()
        if field_ is None:
            return None
        r = np.asarray(r_um if np.size(r_um) else
                       np.linspace(self.shaft_radius_um,
                                   20.0 * self.shaft_radius_um, 64), dtype=float)
        return float(np.max(np.abs(self.max_shear_strain(r) -
                                   field_.max_shear_strain(r))))

    # -- validity ----------------------------------------------------------
    def small_strain_valid_at(self, r_um):
        return np.asarray(self.max_shear_strain(r_um)) <= SMALL_STRAIN_LIMIT

    def validity_radius_um(self):
        """r beyond which gamma <= SMALL_STRAIN_LIMIT (0 if never above it)."""
        if self.wall_shear_strain <= SMALL_STRAIN_LIMIT:
            return 0.0
        return float(math.sqrt(2.0 * self._c_um2 / SMALL_STRAIN_LIMIT))

    def validity(self):
        wall = self.wall_shear_strain
        return {
            "wall_shear_strain": wall,
            "small_strain_limit": SMALL_STRAIN_LIMIT,
            "wall_inside_small_strain": bool(wall <= SMALL_STRAIN_LIMIT),
            "valid_outside_radius_um": self.validity_radius_um(),
            "verdict": ("INSIDE the linear-elastic small-strain domain"
                        if wall <= SMALL_STRAIN_LIMIT else
                        "OUTSIDE the field's own domain at the shaft wall: the "
                        "criterion radius below is an ORDER-OF-MAGNITUDE bound, not "
                        "a predicted strain"),
        }

    # -- criterion ---------------------------------------------------------
    def damage_radius_um(self, shear_strain_threshold):
        g = _num(shear_strain_threshold, "shear_strain_threshold", 0.0, True)
        return float(max(self.shaft_radius_um,
                         math.sqrt(2.0 * self._c_um2 / g)))

    def damage_radius_um_from_stress(self, shear_stress_Pa):
        tau = _num(shear_stress_Pa, "shear_stress_Pa", 0.0, True)
        return self.damage_radius_um(tau / self.shear_modulus_Pa)

    def cycles_per_day(self):
        return self.frequency_Hz * 86400.0

    def to_dict(self):
        return {
            "mechanism": self.MECHANISM,
            "added": "this mechanism was NOT modelled anywhere in the project",
            "reused": "the incompressible 1/r plane-strain KERNEL of "
                      "CavityExpansionField (its _c = delta*a); the cyclic amplitude "
                      "replaces the static wall displacement",
            "kernel_agreement_with_cavity_field": self.kernel_matches_cavity_field(),
            "shaft_diameter_um": 2.0 * self.shaft_radius_um,
            "shaft_radius_um": self.shaft_radius_um,
            "amplitude_um": self.amplitude_um,
            "frequency_Hz": self.frequency_Hz,
            "amplitude_ratio_over_shaft_radius": self.amplitude_ratio,
            "wall_shear_strain": self.wall_shear_strain,
            "shear_modulus_Pa": self.shear_modulus_Pa,
            "wall_shear_stress_Pa": float(self.shear_stress_Pa(self.shaft_radius_um)),
            "cycles_per_day": self.cycles_per_day(),
            "validity": self.validity(),
            "not_modelled": ["viscoelasticity (frequency enters NO strain)",
                             "tissue sliding along the probe",
                             "the fibrous capsule of a chronic implant",
                             "any anchor stiffness",
                             "any measured fly amplitude or frequency"],
            "provenance": "amplitude and frequency are ASSUMED with sweeps; no fly "
                          "value was verified and NO rodent amplitude is transferred",
            "surgery_electrode_caveat": SURGERY_ELECTRODE_INSEPARABLE,
        }


# ----------------------------------------------------------------------
# 6. the probe axis, taken from the REAL access map
# ----------------------------------------------------------------------
def probe_axis_from_access_map(access_map, *, extension_fraction=0.5,
                               insertion_depth_um=None):
    """The shaft axis, DERIVED from the array the access map actually built.

    Declared rule (no hidden choice):

      * ``linear_shank_*``: the shank IS the array, so the axis is the line the
        channel centres lie on (exact: they are collinear by construction), and
        the shaft segment spans that line extended at each end by
        ``extension_fraction * pitch`` for the tip and the tail.
      * ``planar_grid_xy``: the array is a plane, so the axis is its NORMAL through
        the array centroid, and the segment length must come from the declared
        insertion depth (``insertion_depth_um``), centred on the array plane.

    The sign is fixed deterministically (largest-magnitude component positive) so
    that two runs cannot disagree about the direction.
    """
    centres = np.asarray(access_map.channel_centres_um, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 3 or centres.shape[0] == 0:
        raise ValueError("access_map.channel_centres_um must be (n,3)")
    centroid = centres.mean(axis=0)
    geometry = access_map.spec.geometry
    centred = centres - centroid
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    pitch = float(access_map.spec.pitch_um)
    if geometry.startswith("linear_shank"):
        axis = vt[0].copy()
        span = centred @ axis
        half = float(np.max(np.abs(span))) + extension_fraction * pitch
        rule = ("linear shank: axis = the line through the channel centres; the shaft "
                "segment spans the contacts plus extension_fraction*pitch at each end")
    else:
        axis = vt[2].copy()
        if insertion_depth_um is None:
            raise ValueError(
                "a planar array has no extent along the shaft axis, so the shaft "
                "segment length must be supplied as insertion_depth_um; this module "
                "refuses to invent one here")
        half = 0.5 * _num(insertion_depth_um, "insertion_depth_um", 0.0, True)
        rule = ("planar grid: axis = the array-plane NORMAL through the centroid; the "
                "segment is the declared insertion depth centred on the array plane")
    # deterministic sign convention: the largest-magnitude component is positive,
    # so two runs (or two machines) cannot disagree about the direction
    k = int(np.argmax(np.abs(axis)))
    if axis[k] < 0:
        axis = -axis
    axis = axis / np.linalg.norm(axis)
    return {"point_um": centroid, "direction": axis, "half_length_um": half,
            "rule": rule, "geometry": geometry,
            "singular_values_um": [float(v) for v in sv],
            "pitch_um": pitch,
            "extension_fraction": float(extension_fraction)}


def _check_probe_consistency(access_map, mechanics):
    """The mechanics and the access map MUST describe the same probe."""
    d_map = float(access_map.spec.diameter_um)
    d_mech = 2.0 * float(mechanics.shaft_radius_um)
    if abs(d_map - d_mech) > 1e-9:
        raise ValueError(
            "the mechanics describe a %.6g um shaft but the access map was built "
            "with diameter_um=%.6g: refusing to connect two different probes"
            % (d_mech, d_map))
    return {"access_map_diameter_um": d_map, "mechanics_diameter_um": d_mech,
            "consistent": True}


def probe_neurons_at_risk(access_map, mechanics, strain_threshold, *,
                          shaft=None, tier_root_ids=None, extension_fraction=0.5,
                          insertion_depth_um=None, yield_verdict=False):
    """How many of the ACTUALLY-ADDRESSABLE neurons fall inside a strain criterion.

    This is the number that connects the mechanics to the access map.  It is
    COMPUTED from the real BANC soma points the access map captured -- not
    described, not estimated from a density.

    Parameters
    ----------
    access_map
        an ``engine.embodied.access_map.AccessMap`` (built at the same shaft
        diameter; a mismatch RAISES).
    mechanics
        ``InsertionMechanics`` or ``MicromotionStrain``.  Both expose
        ``damage_radius_um(threshold)`` and a mechanism name.
    strain_threshold
        the stated criterion.  For insertion it is a maximum principal strain
        (cross-species guinea-pig values 0.14/0.21/0.34 are what the project
        carries); for micromotion it is a maximum SHEAR strain.
    tier_root_ids
        optional: the root ids of the modelled tier, so the at-risk fraction can be
        reported for the model's OWN population as well as for the whole soma cloud.

    Returns a dict; ``at_risk_neurons`` carries the per-neuron distance to the
    shaft axis, which is what the coupling step consumes.
    """
    consistency = _check_probe_consistency(access_map, mechanics)
    threshold = _num(strain_threshold, "strain_threshold", 0.0, True)
    radius = float(mechanics.damage_radius_um(threshold))
    if shaft is None:
        shaft = probe_axis_from_access_map(
            access_map, extension_fraction=extension_fraction,
            insertion_depth_um=insertion_depth_um)
    point = np.asarray(shaft["point_um"], dtype=float)
    axis = np.asarray(shaft["direction"], dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = float(shaft["half_length_um"])

    resolution = access_map.resolution
    positions_vox = np.asarray(access_map.soma_positions_vox, dtype=float)
    all_um = resolution.voxels_to_um(positions_vox)
    root_ids = np.asarray(access_map.soma_root_ids, dtype=np.int64)
    n_ch = np.asarray(access_map.neuron_n_channels, dtype=np.int64)
    addressable = n_ch > 0

    rel = all_um - point[None, :]
    axial = rel @ axis
    perp = np.linalg.norm(rel - np.outer(axial, axis), axis=1)
    inside_radius = perp <= radius
    inside_length = np.abs(axial) <= half
    at_risk = addressable & inside_radius & inside_length

    if tier_root_ids is not None:
        tier = np.isin(root_ids, np.asarray(tier_root_ids, dtype=np.int64).ravel())
    else:
        tier = np.zeros(root_ids.size, dtype=bool)

    def _frac(k, n):
        return (float(k) / float(n)) if n else None

    n_addr = int(addressable.sum())
    n_addr_tier = int((addressable & tier).sum())
    n_risk = int(at_risk.sum())
    n_risk_tier = int((at_risk & tier).sum())
    rows = np.flatnonzero(at_risk)
    neurons = [{"soma_row": int(r), "root_id": int(root_ids[r]),
                "position_um": [float(v) for v in all_um[r]],
                "distance_to_shaft_axis_um": float(perp[r]),
                "axial_coordinate_um": float(axial[r]),
                "n_channels_capturing": int(n_ch[r]),
                "in_modelled_tier": bool(tier[r])}
               for r in rows]
    # the distance distribution of the addressable population, so a reader can see
    # how close the criterion radius is to a real spatial gap rather than a density
    addr_perp = perp[addressable]
    out = {
        "mechanism": getattr(mechanics, "MECHANISM", "unknown"),
        "criterion": {
            "measure": ("max_abs_principal_strain" if
                        getattr(mechanics, "MECHANISM", "") ==
                        InsertionMechanics.MECHANISM else "max_shear_strain"),
            "threshold": threshold,
            "radius_um": radius,
            "collapsed": bool(radius <= mechanics.shaft_radius_um + 1e-12),
            "note": ("radius is floored at the shaft radius by the reused field class, "
                     "so a threshold above the wall strain reports the whole shaft "
                     "cylinder and nothing more"),
        },
        "shaft": {"point_um": [float(v) for v in point],
                  "direction": [float(v) for v in axis],
                  "half_length_um": half,
                  "diameter_um": 2.0 * mechanics.shaft_radius_um,
                  "rule": shaft["rule"],
                  "geometry": shaft["geometry"],
                  "singular_values_um": shaft["singular_values_um"]},
        "probe_consistency": consistency,
        "population": {
            "addressable_all_somas": n_addr,
            "addressable_all_somas_description":
                "BANC somas whose single point lies inside >=1 channel capture "
                "sphere of THIS array",
            "at_risk_all_somas": n_risk,
            "fraction_of_addressable_all_somas": _frac(n_risk, n_addr),
            "addressable_in_modelled_tier": n_addr_tier,
            "at_risk_in_modelled_tier": n_risk_tier,
            "fraction_of_addressable_in_modelled_tier": _frac(n_risk_tier, n_addr_tier),
            "tier_supplied": bool(tier_root_ids is not None),
        },
        "addressable_distance_stats_um": {
            "min": float(addr_perp.min()) if addr_perp.size else None,
            "median": float(np.median(addr_perp)) if addr_perp.size else None,
            "max": float(addr_perp.max()) if addr_perp.size else None,
            "n_within_criterion": int((addr_perp <= radius).sum()),
        },
        "at_risk_neurons": neurons,
        "frame": {"voxel_resolution_nm": [resolution.x_nm, resolution.y_nm, resolution.z_nm],
                  "voxel_resolution_provenance": resolution.provenance,
                  "voxel_resolution_source": resolution.source,
                  "note": "the x/y value is 4 nm in the verified record; the brief for "
                          "this round states x/y is still ambiguous between 4 and 8 nm, "
                          "and that ambiguity is CARRIED here, not resolved: if x/y "
                          "were 8 nm every lateral distance in this block doubles "
                          "(and the z value is unchanged at 45 nm).",
                  "lateral_ambiguity_factor_if_8nm": 2.0},
        "attribution": dict(BANC_SOMA_ATTRIBUTION),
        "surgery_electrode_caveat": SURGERY_ELECTRODE_INSEPARABLE,
        "provenance": {
            "measured_local": "the soma positions and the captured sets come from the "
                              "access map built on the real BANC parquet",
            "assumed": "the criterion radius comes from the ASSUMED mechanics; the "
                       "shaft-axis placement rule is a DECLARED choice",
        },
    }
    if yield_verdict:
        out["verdict"] = (
            "no addressable neuron lies inside the criterion"
            if n_risk == 0 else
            "%d of %d addressable somas (%.4f) lie inside the %.4g strain criterion "
            "at r <= %.4g um" % (n_risk, n_addr, _frac(n_risk, n_addr) or 0.0,
                                 threshold, radius))
    return out


# ----------------------------------------------------------------------
# 7. coupling the mechanics into the EXISTING cable model
# ----------------------------------------------------------------------
def _place_proxy_cable(position_um, axis, length_um, diameter_um, nseg):
    """A passive proxy cable through ``position_um``, perpendicular to the shaft.

    DECLARED WORST CASE: the BANC soma table gives one point per neuron and no
    morphology, so the neurite's true path is unknown.  A cable running
    PERPENDICULAR to the shaft axis through the soma point maximises the length of
    membrane inside the damage cylinder, so it is the worst case for the counted
    damaged area.  That choice is stated rather than hidden.
    """
    nseg = int(nseg)
    base = Morphology.cylinder(length_um, diameter_um, nseg)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    w = ref - float(ref @ axis) * axis
    w = w / np.linalg.norm(w)                       # perpendicular to the shaft
    u = np.cross(axis, w)
    rot = np.column_stack((w, u, axis))
    local = np.column_stack((base.x - 0.5 * length_um, base.y, base.z))
    xyz = local @ rot.T + np.asarray(position_um, dtype=float)
    return Morphology(xyz[:, 0], xyz[:, 1], xyz[:, 2], base.d, base.parent)


def couple_to_excitable_model(mechanics, at_risk, *,
                              leak_density_S_cm2=0.5, E_rev_mV=None,
                              cable_length_um=None, cable_diameter_um=None,
                              nseg=None, excitability=None, resealing=None,
                              dt_ms=0.05, duration_ms=50.0,
                              target_deflection_mV=10.0, max_neurons=32,
                              cross_check_with_run_paired_cable=True,
                              cable_kwargs=None):
    """Feed the mechanical damage into the EXISTING passive cable and measure it.

    EVERYTHING THAT APPLIES DAMAGE IS REUSED:

      * ``affected_node_indices``     picks the cable rows inside the radius;
      * ``apply_membrane_damage``     applies the leak (``CableNeuron.set_end_leak``);
      * ``ProximityExcitabilityCoupling`` optionally scales the injected drive;
      * ``run_paired_cable``          is called as an INDEPENDENT cross-check of the
                                      same radius/density and its traces are compared
                                      against this function's own paired run.

    THE MODEL IS PASSIVE.  ``engine.cable`` has no voltage-gated channels, so there
    is NO firing-rate, threshold or "excitability" quantity available.  What is
    measured and reported is the SUBTHRESHOLD transfer: the added leak conductance,
    the DC input resistance change and the response attenuation.  Calling that
    "excitability" would overstate it, so it is called what it is.

    Returns a dict with per-neuron results for the neurons at risk and an aggregate.
    """
    if not isinstance(at_risk, dict) or "at_risk_neurons" not in at_risk:
        raise ValueError("at_risk must be the dict returned by probe_neurons_at_risk")
    shaft = at_risk["shaft"]
    point = np.asarray(shaft["point_um"], dtype=float)
    axis = np.asarray(shaft["direction"], dtype=float)
    radius = float(at_risk["criterion"]["radius_um"])
    leak_density = _num(leak_density_S_cm2, "leak_density_S_cm2", 0.0, True)
    if cable_length_um is None:
        cable_length_um = 120.0
    if cable_diameter_um is None:
        cable_diameter_um = 1.0
    if nseg is None:
        nseg = 50
    nseg = int(nseg)
    if nseg < 2:
        raise ValueError("nseg must be >= 2")
    excitability = excitability or ProximityExcitabilityCoupling()
    if not isinstance(excitability, ProximityExcitabilityCoupling):
        raise TypeError("excitability must be a ProximityExcitabilityCoupling")
    cable_kwargs = dict(cable_kwargs or {})
    dt_ms = _num(dt_ms, "dt_ms", 0.0, True)
    duration_ms = _num(duration_ms, "duration_ms", 0.0, True)
    steps = int(round(duration_ms / dt_ms))
    if steps < 1:
        raise ValueError("duration shorter than one timestep")
    target = _num(target_deflection_mV, "target_deflection_mV", 0.0, True)
    max_neurons = int(max_neurons)
    if max_neurons < 1:
        raise ValueError("max_neurons must be >= 1")

    neurons = list(at_risk["at_risk_neurons"])[:max_neurons]
    per_neuron = []
    first_traces = None
    cross_check = None
    for k, neuron in enumerate(neurons):
        morph = _place_proxy_cable(neuron["position_um"], axis, cable_length_um,
                                   cable_diameter_um, nseg)
        indices = affected_node_indices(morph, point, axis, radius)   # REUSED
        inject = nseg // 2

        sham = CableNeuron(morph, dt_ms=dt_ms, **cable_kwargs)
        damaged = CableNeuron(morph, dt_ms=dt_ms, **cable_kwargs)
        record = None
        if indices.size:
            record = apply_membrane_damage(damaged, indices, leak_density,
                                           sham.E_leak if E_rev_mV is None
                                           else E_rev_mV)          # REUSED

        centres = np.column_stack((morph.x, morph.y, morph.z))
        rel = centres - point[None, :]
        axial = rel @ axis
        distance = np.linalg.norm(rel - np.outer(axial, axis), axis=1)
        scale = excitability.scale(distance)                        # REUSED

        i_unit = np.zeros(morph.n)
        i_unit[inject] = 1.0
        r_sham = float(spsolve(sham.G.tocsc(), i_unit)[inject])
        r_damaged = float(spsolve(damaged.G.tocsc(), i_unit)[inject])
        drive_nA = target / r_sham
        i_inject = np.zeros(morph.n)
        i_inject[inject] = drive_nA
        i_eff = i_inject * scale

        v_sham = np.empty((steps, morph.n))
        v_dam = np.empty((steps, morph.n))
        for s in range(steps):
            t_ms = (s + 1) * dt_ms
            if resealing is not None and indices.size:
                if not isinstance(resealing, ResealingTimescales):
                    raise TypeError("resealing must be a ResealingTimescales")
                raise NotImplementedError(
                    "this coupling reports a STATIC leak; time-dependent resealing "
                    "is available through run_paired_cable(resealing=...) and is "
                    "deliberately not duplicated here")
            v_sham[s] = sham.step(i_inject=i_eff)
            v_dam[s] = damaged.step(i_inject=i_eff)
        delta = v_sham - v_dam
        denom = max(1e-12, float(np.max(np.abs(v_sham[:, indices] - sham.E_leak)))
                    if indices.size else 1e-12)
        atten = (float(np.max(np.abs(delta[:, indices])) / denom)
                 if indices.size else 0.0)
        row = {
            "soma_row": neuron["soma_row"], "root_id": neuron["root_id"],
            "distance_to_shaft_axis_um": neuron["distance_to_shaft_axis_um"],
            "in_modelled_tier": neuron["in_modelled_tier"],
            "n_damaged_compartments": int(indices.size),
            "damaged_compartments": [int(i) for i in indices],
            "damaged_patch_length_um": float(2.0 * math.sqrt(max(
                0.0, radius ** 2 - neuron["distance_to_shaft_axis_um"] ** 2))),
            "g_end_uS_per_compartment": ([] if record is None else
                                         list(record["g_uS_per_node"])),
            "g_end_total_uS": (0.0 if record is None else record["total_g_uS"]),
            "g_membrane_total_uS": float(damaged.g_mem.sum()),
            "leak_ratio_over_membrane": (0.0 if record is None else
                                         float(record["total_g_uS"] /
                                               max(1e-30, damaged.g_mem.sum()))),
            "R_in_sham_Mohm": r_sham, "R_in_damaged_Mohm": r_damaged,
            "R_in_ratio_damaged_over_sham": r_damaged / r_sham,
            "R_in_reduction_fraction": 1.0 - r_damaged / r_sham,
            "subthreshold_attenuation_fraction": atten,
            "max_abs_delta_mV": float(np.abs(delta).max()),
            "drive_nA_at_target_deflection": drive_nA,
            "proximity_scale_at_damaged_rows": ([] if not indices.size else
                                                [float(v) for v in scale[indices]]),
        }
        per_neuron.append(row)

        if k == 0:
            first_traces = {
                "time_ms": [float((s + 1) * dt_ms) for s in range(steps)],
                "sham_mV_at_source": [float(v) for v in v_sham[:, inject]],
                "inserted_mV_at_source": [float(v) for v in v_dam[:, inject]],
                "E_leak_mV": float(sham.E_leak),
                "inject_index": int(inject),
            }
            if cross_check_with_run_paired_cable and isinstance(mechanics,
                                                               InsertionMechanics):
                cc = run_paired_cable(morph, mechanics.field, point, axis,
                                      at_risk["criterion"]["threshold"],
                                      config=CableDamageConfig(
                                          leak_density_S_cm2=leak_density,
                                          E_rev_mV=E_rev_mV,
                                          excitability=excitability),
                                      inject_index=inject,
                                      target_deflection_mV=target,
                                      dt_ms=dt_ms, duration_ms=duration_ms,
                                      cable_kwargs=cable_kwargs)
                cross_check = {
                    "called": "engine.electrode_damage.run_paired_cable",
                    "its_damage_radius_um": cc["damage_radius_um"],
                    "its_n_damaged_nodes": int(cc["damaged_node_indices"].size),
                    "our_n_damaged_nodes": int(indices.size),
                    "same_damaged_node_set": bool(
                        np.array_equal(np.asarray(cc["damaged_node_indices"]),
                                       np.asarray(indices))),
                    "its_max_abs_delta_mV": cc["max_abs_delta_mV"],
                    "our_max_abs_delta_mV": float(np.abs(delta).max()),
                    "trace_max_abs_difference_mV": float(np.max(np.abs(
                        cc["delta_mV"] - delta))),
                    "its_g_end_total_uS": float(cc["g_end_uS"].sum()),
                    "our_g_end_total_uS": float(damaged.g_end.sum()),
                    "note": ("two INDEPENDENT code paths through the same public "
                             "apply_membrane_damage API; agreement is a check on this "
                             "module's paired loop, not new evidence"),
                }

    with_damage = [r for r in per_neuron if r["n_damaged_compartments"] > 0]
    def _mean(key, rows):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "n_neurons_coupled": len(per_neuron),
        "n_neurons_with_any_damaged_compartment": len(with_damage),
        "n_at_risk_available": len(at_risk["at_risk_neurons"]),
        "capped_at": max_neurons,
        "mean_g_end_total_uS": _mean("g_end_total_uS", with_damage),
        "mean_R_in_reduction_fraction": _mean("R_in_reduction_fraction", with_damage),
        "mean_subthreshold_attenuation_fraction":
            _mean("subthreshold_attenuation_fraction", with_damage),
        "max_R_in_reduction_fraction": (None if not with_damage else
                                        float(np.max([r["R_in_reduction_fraction"]
                                                      for r in with_damage]))),
        "mean_leak_ratio_over_membrane": _mean("leak_ratio_over_membrane", with_damage),
    }
    return {
        "api": "engine.electrode_damage.apply_membrane_damage -> "
               "CableNeuron.set_end_leak; affected_node_indices; "
               "ProximityExcitabilityCoupling; run_paired_cable (cross-check)",
        "mechanism": at_risk["mechanism"],
        "criterion_radius_um": radius,
        "criterion_threshold": at_risk["criterion"]["threshold"],
        "leak_density_S_cm2": leak_density,
        "E_rev_mV": E_rev_mV,
        "E_rev_note": "None means the shunt is at the cable's own E_leak, which "
                      "isolates the CONDUCTANCE effect; a real mechanoporation pore "
                      "has a nonspecific reversal near 0 mV and would also "
                      "depolarise -- that additional assumption is NOT made here",
        "proxy_cable": {
            "length_um": cable_length_um, "diameter_um": cable_diameter_um,
            "nseg": nseg, "n_nodes": nseg + 1,
            "orientation": "PERPENDICULAR to the shaft axis through the soma point "
                           "(declared worst case for damaged membrane length)",
            "why_proxy": "the BANC soma table provides ONE POINT per neuron and no "
                         "morphology, so no real neurite geometry exists for these "
                         "neurons; the proxy is a declared stand-in with swept size",
        },
        "excitability_coupling": excitability.to_dict(),
        "dt_ms": dt_ms, "duration_ms": duration_ms, "steps": steps,
        "target_deflection_mV": target,
        "per_neuron": per_neuron,
        "aggregate": aggregate,
        "first_neuron_traces": first_traces,
        "cross_check": cross_check,
        "honesty": (
            "engine.cable is PASSIVE: it has no channels, so nothing here is a "
            "firing rate, a threshold or an excitability curve. The reported "
            "quantities are the added leak conductance, the DC input-resistance "
            "change and the subthreshold response attenuation.",
            SURGERY_ELECTRODE_INSEPARABLE,
        ),
    }
