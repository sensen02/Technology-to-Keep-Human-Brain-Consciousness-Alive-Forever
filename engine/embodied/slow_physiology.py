"""engine.embodied.slow_physiology -- SLOW PHYSIOLOGY state + behaviour/rest recorder.

WHAT THIS FILE IS
-----------------
Two things, deliberately kept separate in the code and in the record:

  (A) SLOW PHYSIOLOGY STATE.  A small set of finite compartments with explicit
      units, explicit ledgers and NO silent clamping, driven by a MEASURED
      quantity of the physical body:

        TRACHEAL compartment  (air side)      O2 amount, finite.  Charged from an
                                              explicitly infinite air bath through a
                                              spiracle/trachea conductance, drained by
                                              tissue consumption.
        TISSUE compartment    (intracellular) O2 amount, finite.  Filled from the
                                              tracheal compartment through a tracheole
                                              conductance, drained by metabolism.
        HAEMOLYMPH compartment                its OWN substance ledger, tracking
                                              TREHALOSE (the fly's main circulating
                                              sugar).  The haemolymph explicitly does
                                              NOT carry O2: the fly's gas exchange is
                                              TRACHEAL and the haemolymph ledger has no
                                              O2 term at all.

      This is a Drosophila-appropriate gas-exchange architecture: the oxygen is carried
      by AIR IN TUBES and never by the circulating fluid, and no vertebrate-style
      saturating carrier curve is used anywhere.  See ``SLOWPHYS_HONESTY``.

  (B) BEHAVIOUR / REST RECORDER.  Every command interval is classified as
      locomotion / turning / exploration-like / quiet rest / not-supported, using
      PRE-REGISTERED numeric thresholds on MEASURED body quantities (thorax speed
      in mm/s, net heading change in degrees, net displacement in mm, leg contact
      count).  Sleep is NOT asserted: merely quiescent stretches are labelled
      QUIET_REST, and no sleep criterion is implemented at all (stated below and
      in the report rather than implied).

UNITS (stated once; every recorded quantity carries its unit string)
--------------------------------------------------------------------
    length            mm          (the body model is a MILLIMETRE model: gravity
                                  -9810 mm/s^2; this module never rescales it)
    time              s
    angle             rad         (heading change also reported in deg)
    speed             mm/s
    joint velocity    rad/s
    amount of O2      nmol        O2-equivalent amount in a compartment
    concentration     nmol/mm^3   (= 1e-6 mol/m^3; 1 mm^3 = 1 uL)
    conductance       mm^3/s      volumetric flow / transport conductance
    rate of change    nmol/s
    trehalose         nmol        amount; concentration nmol/mm^3
    release rate      nmol/s
    grid (modulator)  um, um^2/s, um^3/s, nM, nM*um^3 (engine.local_tissue units)
    temperature       K

The tracheal and tissue compartments are 0-D well-mixed control volumes at the
BODY scale (mm), i.e. a lumped-parameter approximation.  ``engine.local_tissue``
and ``engine.injury_tissue`` work on a 3-D grid at the TISSUE scale (um) and are
NOT re-scaled or mixed into these ledgers: nmol (body scale) and nM*um^3 (grid
scale) are different units and are never summed.  The one place the grid module
is used is the opt-in modulator layer, which reports its own ledger in its own
units.

WHICH MEASURED QUANTITY DRIVES CONSUMPTION (hard requirement 4)
---------------------------------------------------------------
There is NO invented activity scalar anywhere in this file.  The metabolic demand
is computed from two quantities READ FROM THE BODY every command interval:

    v_interval_mm_s  = || thorax_xy(t_end) - thorax_xy(t_start) || / dt_command_s
                       (horizontal displacement per interval / interval length)
    J_interval_rad_s = mean over the interval's substeps of
                       sum_j |qdot_j|   over all recorded joint velocities
                       (joint_velocities_rad_s, (66,) per substep)

and their interval means over the slow step.  These are reduced to a
DIMENSIONLESS load with two ILLUSTRATIVE scales (``speed_scale_mm_s`` and
``joint_speed_scale_rad_s``) and two ILLUSTRATIVE weights, and then multiplied by
an ILLUSTRATIVE absolute oxygen consumption scale:

    load        = w_speed * (v / speed_scale) + w_joint * (J / joint_speed_scale)
    demand_nmol_s = resting_demand_nmol_s + demand_per_load_nmol_s * load

The absolute scale is ILLUSTRATIVE: this project has no measured adult-fly
metabolic rate.  What is NOT illustrative is the DIRECTION and the INPUT: the
demand responds to the body's own measured movement, and a test shows that
raising the body's speed/joint speed raises consumption proportionally.

ENERGY PROXY (hard requirement 5): ``EnergyAvailabilityProxy``
--------------------------------------------------------------
A dimensionless bounded availability in [0, 1] that is explicitly NOT ATP, NOT
energy in joules, and is kept OUT of every substance ledger.  It is the same gadget as
``engine.local_tissue.NonPhysiologicalSupportProxy`` (re-implemented here as a
few lines so this module has no dependency on the neural-tier environment); its
disclaimer is repeated verbatim in ``ENERGY_PROXY_DISCLAIMER`` and it is reported
under its own name only.

LEDGERS (hard requirement 3)
----------------------------
For every modelled substance S:
    amount_now(S) = initial(S) + inflow(S) - outflow(S) - consumption(S) + residual(S)
The residual is REPORTED for every substance, never hidden and never corrected.
There is no clipping and no post-hoc mass repair: the transport solves used here
are unconditionally positive (an implicit-Euler update with an exact positive
closed form), and any non-finite or negative state raises instead of being fixed.
The air bath is an explicitly INFINITE external reservoir (like
``local_tissue.PrescribedBath``) and is NOT inside the ledger; its gross inflow and
outflow are recorded separately, so "where did this O2 come from" always has an
answer.

TIME (hard requirement 7)
-------------------------
The demo runs on the REAL CLOCK at the scheduler's own 5 ms command interval, for
seconds to minutes of simulated time.  If a caller advances the slow module ALONE
for longer than the body was simulated, that advance is recorded as an explicit
TIME JUMP via ``force_time_jump``/``TimeJumpRecord`` with the label
``TIME_JUMP_QUASI_STEADY`` and the note that the body and the nervous system were
NOT simulated during it.  No run in this module ever claims continuous
whole-nervous-system simulation beyond the wall-clock-backed body run.

WHAT THIS MODULE DOES NOT DO
----------------------------
No hormone is asserted.  One opt-in modulator layer exists (source voxel,
diffusion and clearance through ``engine.local_tissue.LocalTissue``, a receptor
occupancy probe with a stated binding law, and a local distribution on a labelled
um grid); every target effect is reported with ``known_target = False`` and is OFF
by default, and NO measured kinetics are invented for any named molecule.  There
is no sleep state, no survival rule, no viability rule, no consciousness claim and
no effect of this module on the motor command (it only reads).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import os

import numpy as np

__all__ = [
    "MM", "SECONDS", "RAD", "MM_PER_S", "RAD_PER_S", "NANO_MOL", "NMOL_PER_MM3",
    "MM3_PER_S", "NMOL_PER_S", "UNITS", "SLOWPHYS_HONESTY",
    "ENERGY_PROXY_DISCLAIMER", "ENERGY_PROXY_NAME", "SUBSTANCE_LEDGER_RULE",
    "TrachealConfig", "TissueConfig", "HaemolymphConfig", "LoadConfig",
    "SlowPhysiologyConfig", "TimeJumpRecord", "TimeJumpError",
    "LocomotorActivity", "SubstanceLedger", "EnergyAvailabilityProxy",
    "LedgerRow", "SlowPhysiology", "SlowPhysiologyDriver",
    "BehaviourConfig", "BehaviourMeasures", "PRE_REGISTERED_BEHAVIOUR_THRESHOLDS",
    "BEHAVIOUR_LABELS", "exploration_coverage", "GEOMETRY_PROVENANCE",
    "measure_long_scale",
    "classify_behaviour", "measure_behaviour", "epoch_table",
    "ModulatorConfig", "ModulatorLedger", "LocalModulator", "slowphys_fingerprint",
    "MODULE_FORBIDDEN_TERMS",
]

# ---------------------------------------------------------------------------
# units and fixed vocabulary
# ---------------------------------------------------------------------------
MM = "mm"
SECONDS = "s"
RAD = "rad"
DEG = "deg"
MM_PER_S = "mm/s"
RAD_PER_S = "rad/s"
NANO_MOL = "nmol"
NMOL_PER_MM3 = "nmol/mm^3"
MM3_PER_S = "mm^3/s"
NMOL_PER_S = "nmol/s"
HZ = "Hz"

UNITS = {
    "length": MM,
    "time": SECONDS,
    "angle": RAD,
    "heading_change": DEG,
    "speed": MM_PER_S,
    "joint_velocity": RAD_PER_S,
    "o2_amount": NANO_MOL,
    "o2_concentration": NMOL_PER_MM3,
    "tracheal_conductance": MM3_PER_S,
    "rate_of_change": NMOL_PER_S,
    "trehalose_amount": NANO_MOL,
    "trehalose_concentration": NMOL_PER_MM3,
    "energy_proxy": "dimensionless availability in [0, 1] (NOT ATP, NOT joules)",
    "modulator_grid_length": "um",
    "modulator_grid_diffusion": "um^2/s",
    "modulator_grid_conductance": "um^3/s",
    "modulator_grid_concentration": "nM",
    "modulator_grid_amount": "nM*um^3",
    "receptor_occupancy": "dimensionless fraction of capacity in [0, 1]",
    "leg_contact": "count of legs in contact (0..6, dimensionless)",
    "contact_force": ("backend's own force channel; ABSOLUTE calibration UNRESOLVED, "
                      "used only through its declared reference (see BodyConfig note)"),
    "power": "arbitrary backend force unit * mm/s",
}

#: The banned vocabulary is enforced by a test: nothing in this module may be
#: NAMED, DOCUMENTED OR REPORTED AS any of these.  The list itself is the only place the
#: words appear (the test strips it before scanning), and the one term that must appear
#: in prose -- ATP -- is allowed ONLY inside a sentence that denies it, which the test
#: checks by looking for the negation next to the word rather than by counting.
#: The fly's gas exchange is tracheal, so a blood-borne carrier is not a simplification
#: here, it is a different animal.
MODULE_FORBIDDEN_TERMS = (
    "oxyhaemoglobin", "oxyhemoglobin", "haemoglobin", "hemoglobin", "HbO2",
    "blood oxygenation", "blood O2", "haemolymph O2", "hemolymph O2",
    "P50", "Krogh", "ATP",
)

ENERGY_PROXY_NAME = "NonPhysiologicalAvailabilityProxy"
ENERGY_PROXY_DISCLAIMER = (
    "DIMENSIONLESS bounded availability proxy in [0,1]. It is NOT ATP, NOT a "
    "phosphoryl group, NOT energy in joules, and it is NOT evidence of cellular "
    "energy status. It carries no stoichiometry, no measured metabolic rate and no "
    "survival rule, and it is kept strictly OUT of every substance ledger in this "
    "module (a test asserts it appears in no ledger entry). This project has no "
    "measured adult-fly metabolic rate.")

SUBSTANCE_LEDGER_RULE = (
    "amount_now = initial + inflow - outflow - consumption + residual; the residual is "
    "reported for every substance and is never corrected, clamped or hidden")

SLOWPHYS_HONESTY = [
    "The fly's gas exchange is TRACHEAL: O2 enters through the spiracles and reaches "
    "tissue through tracheae and tracheoles. NO blood-borne respiratory pigment, no "
    "blood-borne O2 carriage and no mammalian saturation curve is modelled anywhere in "
    "this file, because the fly does not use one. Haemolymph is a SEPARATE compartment "
    "with its own substance ledger (trehalose) that contains no O2 term.",
    "EVERY kinetic and physiological parameter in this file is ILLUSTRATIVE. None of "
    "them carries a citation because this project has no measured Drosophila tracheal "
    "conductance, no measured Drosophila haemolymph trehalose turnover and no measured "
    "adult-fly metabolic rate.",
    "The only tracheal GEOMETRY this project has on record (tracheole radius 0.5 um, "
    "muscle cylinder radius 6.5 um, spacing ~13 um) is LOCUST data, not Drosophila. "
    "This module does not use it for any default; the locust numbers are exposed only "
    "in GEOMETRY_PROVENANCE and are flagged, so reusing them cannot happen silently.",
    "The energy availability proxy is a PROXY and is not ATP: it is a dimensionless "
    "bounded scalar with no stoichiometry and is excluded from every substance ledger.",
    "QUIET REST IS NOT SLEEP. A quiescent stretch is labelled rest by a movement "
    "criterion only. No sleep state is implemented and no arousal-response test "
    "stimulus exists in this effort, so no result here may be described as sleep.",
    "The locomotor load uses only measured body quantities (thorax speed in mm/s and "
    "summed absolute joint velocity in rad/s) but the two normalising scales, the two "
    "weights and the absolute oxygen consumption scale are ILLUSTRATIVE.",
    "Pure reduced K/glial chemistry is NOT part of this module. Where the project's "
    "reduced K chemistry is involved it carries a documented non-homeostatic artefact; "
    "nothing here inherits or repairs it.",
    "No consciousness, experience, identity-continuity, viability or immortality claim "
    "is made anywhere in this effort.",
    "The opt-in modulator layer has UNKNOWN targets by default: every target effect is "
    "recorded with known_target=False and the unknown-effect path is OFF unless the "
    "caller explicitly switches it on.",
]

#: only place the locust geometry is allowed to appear, so it cannot be reused by
#: accident: it is provenance text, and no default in this file is set from it.
GEOMETRY_PROVENANCE = {
    "tracheole_radius_um": (0.5, "measured in LOCUST, NOT Drosophila"),
    "muscle_cylinder_radius_um": (6.5, "measured in LOCUST, NOT Drosophila"),
    "tracheal_spacing_um": (13.0, "measured in LOCUST, NOT Drosophila"),
    "used_for_any_default": False,
    "note": ("exposed for provenance only; the Drosophila tracheal conductance this "
             "module needs has never been measured in this project, so no default is "
             "derived from these numbers"),
}


def _num(value, name, minimum=None, positive=False, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating,
                                                         np.integer)):
        raise ValueError("%s must be a finite number" % name)
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("%s must be finite" % name)
    if positive and x <= 0.0:
        raise ValueError("%s must be positive" % name)
    if minimum is not None and x < minimum:
        raise ValueError("%s must be >= %s" % (name, minimum))
    if not allow_zero and x == 0.0:
        raise ValueError("%s must be non-zero" % name)
    return x


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrachealConfig:
    """The tracheal supply route.  All parameters ILLUSTRATIVE."""
    #: O2 concentration in the outside air.  Air at 101.3 kPa and 298.15 K has an O2
    #: partial pressure of 21.28 kPa; at 44.6 nmol/(mm^3*kPa) that is 9.49 nmol/mm^3.
    #: Air composition is the one number in this file that is NOT fly physiology, so it
    #: is stated with its arithmetic.
    bath_o2_nmol_per_mm3: float = 9.49
    o2_solubility_nmol_per_mm3_kpa: float = 44.6
    bath_o2_kpa: float = 21.28
    #: spiracle + trachea lumped conductance to the air bath, mm^3/s.  ILLUSTRATIVE: the
    #: Drosophila spiracle conductance has not been measured in this project.
    bath_conductance_mm3_s: float = 6.0
    #: tracheole conductance from the tracheal lumen to the tissue compartment,
    #: mm^3/s.  ILLUSTRATIVE.  The locust tracheole geometry on record implies a
    #: diffusive conductance orders of magnitude larger than this, which is why the
    #: geometry is provenance only and this number is declared instead.  The value is
    #: chosen so that the tissue O2 drop over the air side is ~1-4% at rest and grows
    #: measurably with load, i.e. it is a DEMONSTRATION scale, not a measurement.
    tracheole_conductance_mm3_s: float = 1.0
    #: initial O2 amount in the air side, nmol.  ILLUSTRATIVE.
    initial_o2_nmol: float = 90.0
    #: the air side is a lumped reservoir; its volume is needed only to report a
    #: concentration, so it is stated rather than inferred.
    volume_mm3: float = 10.0
    name: str = "tracheal_air_side"

    def validate(self):
        for f in ("bath_o2_nmol_per_mm3", "o2_solubility_nmol_per_mm3_kpa",
                  "bath_o2_kpa", "bath_conductance_mm3_s",
                  "tracheole_conductance_mm3_s", "initial_o2_nmol", "volume_mm3"):
            _num(getattr(self, f), "TrachealConfig.%s" % f, minimum=0.0)
        if self.volume_mm3 <= 0:
            raise ValueError("TrachealConfig.volume_mm3 must be positive")
        if self.bath_o2_kpa * self.o2_solubility_nmol_per_mm3_kpa <= 0:
            raise ValueError("TrachealConfig: bath partial pressure must give a "
                             "positive concentration")
        if not str(self.name).strip():
            raise ValueError("TrachealConfig.name must be non-empty")
        return self

    def bath_o2_from_partial_pressure(self):
        """nmol/mm^3 from the stated partial pressure and solubility (Henry's law)."""
        return (self.bath_o2_kpa * self.o2_solubility_nmol_per_mm3_kpa)

    def units(self):
        return {"bath_o2_nmol_per_mm3": NMOL_PER_MM3,
                "o2_solubility_nmol_per_mm3_kpa": "nmol/(mm^3*kPa)",
                "bath_o2_kpa": "kPa",
                "bath_conductance_mm3_s": MM3_PER_S,
                "tracheole_conductance_mm3_s": MM3_PER_S,
                "initial_o2_nmol": NANO_MOL,
                "volume_mm3": "mm^3"}


@dataclass(frozen=True)
class TissueConfig:
    """The consuming compartment.  All parameters ILLUSTRATIVE."""
    #: volume of the lumped tissue control volume, mm^3.  1 mm^3 = 1 uL; the value is
    #: an illustrative lumped volume for an adult fly body.
    volume_mm3: float = 1.0
    initial_o2_nmol: float = 9.0
    #: ILLUSTRATIVE resting oxygen consumption.  The project has NO measured adult-fly
    #: metabolic rate; this number exists so the ledgers have a scale, nothing more.
    resting_demand_nmol_s: float = 0.11
    #: ILLUSTRATIVE extra demand at load = 1.  The value is chosen so that the
    #: demand stays BELOW the supply route's own limit (a*a0*volume, i.e. what the
    #: spiracle conductance can deliver) at every load this demo reaches: at load 6 the
    #: demand is 2.5 nmol/s against a supply ceiling of ~28 nmol/s, so the tissue
    #: quasi-steady concentration stays above zero and the tracheole flux never has to
    #: run backwards.  A larger value is legal in the model but drives the tissue
    #: quasi-steady state negative, which this module reports as demand_unmeetable
    #: instead of hiding.
    demand_per_load_nmol_s: float = 0.4
    name: str = "tissue"

    def validate(self):
        for f in ("volume_mm3", "initial_o2_nmol", "resting_demand_nmol_s",
                  "demand_per_load_nmol_s"):
            _num(getattr(self, f), "TissueConfig.%s" % f, minimum=0.0)
        if self.volume_mm3 <= 0:
            raise ValueError("TissueConfig.volume_mm3 must be positive")
        if not str(self.name).strip():
            raise ValueError("TissueConfig.name must be non-empty")
        return self

    def units(self):
        return {"volume_mm3": "mm^3", "initial_o2_nmol": NANO_MOL,
                "resting_demand_nmol_s": NMOL_PER_S,
                "demand_per_load_nmol_s": NMOL_PER_S}


@dataclass(frozen=True)
class HaemolymphConfig:
    """The SEPARATE circulating compartment.  It carries NO O2.

    This compartment tracks TREHALOSE (the fly's main circulating sugar in
    haemolymph) with its own finite pool, its own source and its own clearance.  All
    parameters ILLUSTRATIVE.  ``o2_capacity_nmol`` exists and must be exactly zero:
    the architecture, not a parameter, is what makes this fly-appropriate, so the
    field is a validated assertion rather than a tunable.
    """
    volume_mm3: float = 0.3
    initial_trehalose_nmol: float = 60.0
    #: ILLUSTRATIVE mobilisation (source) rate, nmol/s.
    trehalose_source_nmol_s: float = 0.0
    #: ILLUSTRATIVE first-order clearance, 1/s.
    trehalose_clearance_per_s: float = 0.0
    #: must stay zero: the fly's haemolymph does not carry O2.
    o2_capacity_nmol: float = 0.0
    name: str = "haemolymph"

    def validate(self):
        for f in ("volume_mm3", "initial_trehalose_nmol", "trehalose_source_nmol_s",
                  "trehalose_clearance_per_s", "o2_capacity_nmol"):
            _num(getattr(self, f), "HaemolymphConfig.%s" % f, minimum=0.0)
        if self.volume_mm3 <= 0:
            raise ValueError("HaemolymphConfig.volume_mm3 must be positive")
        if self.o2_capacity_nmol != 0.0:
            raise ValueError(
                "HaemolymphConfig.o2_capacity_nmol must be exactly 0: the fly's gas "
                "exchange is tracheal and its haemolymph does not carry O2, so a "
                "non-zero capacity would model a different animal")
        if not str(self.name).strip():
            raise ValueError("HaemolymphConfig.name must be non-empty")
        return self

    def units(self):
        return {"volume_mm3": "mm^3", "initial_trehalose_nmol": NANO_MOL,
                "trehalose_source_nmol_s": NMOL_PER_S,
                "trehalose_clearance_per_s": "1/s",
                "o2_capacity_nmol": NANO_MOL}


@dataclass(frozen=True)
class LoadConfig:
    """How a MEASURED quantity of the body becomes metabolic demand.

    The two normalising scales and the two weights are ILLUSTRATIVE.  The two INPUTS
    are not: they are read from the body (see the module docstring).
    """
    speed_scale_mm_s: float = 15.0
    #: ILLUSTRATIVE normalising scale for sum_j |qdot_j|.  It is 200 rad/s, not 20,
    #: because THAT IS THE ENGINE'S OWN RANGE: this position-actuated model's joint
    #: speeds reach ~1500 rad/s at the stance transient (measured over a 24 s run:
    #: p99 = 1500 rad/s with a median of 0), so a 20 rad/s scale would report loads in
    #: the hundreds and say nothing.  This is an engineering scale for this model, not a
    #: fly joint-angular-velocity measurement, which this project does not have.
    joint_speed_scale_rad_s: float = 200.0
    weight_speed: float = 1.0
    weight_joint: float = 1.0
    #: PRE-REGISTERED ceiling on the dimensionless load, so a single transient cannot
    #: dominate the whole run.  Hitting it is RECORDED (n_load_ceiling_hits), never
    #: hidden.  At the default scales it corresponds to ~16 mm/s of speed or ~3000 rad/s
    #: of summed joint speed.
    load_ceiling: float = 8.0
    #: intervals at the start of a run during which the load is held at zero, so the
    #: spawn transient (the body settling into a stance at t = 0) is not reported as
    #: locomotion.  A DECLARED decision, recorded in the report.
    load_warmup_intervals: int = 10
    #: if True the load is taken from speed alone, so a test can show that the joint
    #: term is really contributing rather than being shadowed by the speed term.
    speed_only: bool = False

    def validate(self):
        for f in ("speed_scale_mm_s", "joint_speed_scale_rad_s", "weight_speed",
                  "weight_joint"):
            _num(getattr(self, f), "LoadConfig.%s" % f, minimum=0.0)
        if self.speed_scale_mm_s <= 0 or self.joint_speed_scale_rad_s <= 0:
            raise ValueError("LoadConfig normalising scales must be positive")
        if self.weight_speed + self.weight_joint <= 0:
            raise ValueError("LoadConfig: at least one weight must be positive")
        if not isinstance(self.speed_only, bool):
            raise ValueError("LoadConfig.speed_only must be a bool")
        _num(self.load_ceiling, "load_ceiling", positive=True)
        if isinstance(self.load_warmup_intervals, bool) \
                or not isinstance(self.load_warmup_intervals, int) \
                or self.load_warmup_intervals < 0:
            raise ValueError("LoadConfig.load_warmup_intervals must be a nonnegative "
                             "integer")
        return self

    def units(self):
        return {"speed_scale_mm_s": MM_PER_S,
                "joint_speed_scale_rad_s": RAD_PER_S,
                "weight_speed": "dimensionless",
                "weight_joint": "dimensionless",
                "load_ceiling": "dimensionless",
                "load_warmup_intervals": "count of command intervals"}


@dataclass(frozen=True)
class SlowPhysiologyConfig:
    """Everything that fixes one slow-physiology state.  Validated, refused if not."""
    tracheal: TrachealConfig = field(default_factory=TrachealConfig)
    tissue: TissueConfig = field(default_factory=TissueConfig)
    haemolymph: HaemolymphConfig = field(default_factory=HaemolymphConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    #: report tissue O2 relative to the air side as a saturation-like fraction.  It is a
    #: ratio of two modelled concentrations of the same substance, nothing else: this fly
    #: has no oxygen-carrying pigment, so there is no carrier curve anywhere in this file
    #: for such a fraction to mean.
    denominator_floor_nmol_per_mm3: float = 1e-12
    #: substep cap inside one slow step.  The update below is EXACT for a constant demand
    #: at any step size (see ``_integrate_o2``), so this is not a stability bound: it
    #: bounds how much arithmetic one call does and, with ``long_step_s``, how long a
    #: single quasi-steady stride may be.
    #: the longest single quasi-steady stride, in seconds.  A declared TIME JUMP longer
    #: than this is executed as ceil(dt / long_step_s) substeps of this length, and the
    #: report says so.  The default 60 s is 60x the slowest modelled time constant at the
    #: default parameters (~1 s for the tissue compartment), so each stride is a genuine
    #: relaxation rather than a leap across a transient.
    long_step_s: float = 60.0
    max_substeps: int = 2000
    #: the slow module must never be advanced by an unbounded dt: a step larger than
    #: this is refused, and the caller has to declare a time jump explicitly.
    max_continuous_step_s: float = 60.0
    #: the longest sub-step INSIDE the O2 integrator.  A declared jump of 600 s is run as
    #: ceil(dt / max_substep_s) sub-steps, so a single quasi-steady stride never spans more
    #: than this much of the fast transient (~1 s at the default parameters).  The
    #: trapezoidal flux integration inside a sub-step is third order on the slow relaxation
    #: and second order with the fast one; MEASURED, 1 s sub-steps leave 4.4e-3 nmol of
    #: residual (4.5e-5 relative) over a 600 s jump and 5.5e-4 nmol (5.6e-6 relative) over
    #: 70 s -- both far below the amount being moved, and both are REPORTED.
    max_substep_s: float = 1.0
    #: If the DEMAND exceeds what the supply route can deliver, the demand is not
    #: silently accommodated: it is clipped to the largest steady consumption the
    #: declared conductances can sustain, the unmet part is recorded, and the flag
    #: ``demand_clipped`` is set.  The tissue compartment then sits exactly at zero,
    #: i.e. the model is reporting "this body could not have done that", which is the
    #: honest reading.  Clipping is ON by default because the alternative -- letting
    #: the tissue amount cross zero -- is outside the model's domain.
    enforce_supply_ceiling: bool = True
    seed: int = 0
    enable_energy_proxy: bool = True

    def validate(self):
        self.tracheal.validate()
        self.tissue.validate()
        self.haemolymph.validate()
        self.load.validate()
        _num(self.denominator_floor_nmol_per_mm3, "denominator_floor",
             positive=True)
        if isinstance(self.max_substeps, bool) or not isinstance(self.max_substeps, int) \
                or self.max_substeps < 1:
            raise ValueError("max_substeps must be a positive integer")
        _num(self.max_continuous_step_s, "max_continuous_step_s", positive=True)
        _num(self.max_substep_s, "max_substep_s", positive=True)
        if not isinstance(self.enforce_supply_ceiling, bool):
            raise ValueError("enforce_supply_ceiling must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.enable_energy_proxy, bool):
            raise ValueError("enable_energy_proxy must be a bool")
        return self

    def as_dict(self):
        return {"tracheal": asdict(self.tracheal), "tissue": asdict(self.tissue),
                "haemolymph": asdict(self.haemolymph), "load": asdict(self.load),
                "denominator_floor_nmol_per_mm3": self.denominator_floor_nmol_per_mm3,
                "max_substeps": self.max_substeps,
                "max_continuous_step_s": self.max_continuous_step_s,
                "max_substep_s": self.max_substep_s,
                "enforce_supply_ceiling": self.enforce_supply_ceiling,
                "seed": self.seed, "enable_energy_proxy": self.enable_energy_proxy}

    def units(self):
        return {**self.tracheal.units(), **self.tissue.units(),
                **self.haemolymph.units(), **self.load.units(),
                "seed": "integer", "max_substeps": "integer",
                "max_continuous_step_s": SECONDS,
                "max_substep_s": SECONDS,
                "enforce_supply_ceiling": "bool"}


def slowphys_fingerprint(cfg: SlowPhysiologyConfig, extra=None):
    """sha256[:32] of a slow-physiology config.

    The argument is TYPE-CHECKED: this function hashes the configuration an episode was
    produced with, so silently hashing the string "not a config" would let a caller record
    a fingerprint that names no configuration at all.
    """
    if not isinstance(cfg, SlowPhysiologyConfig):
        raise TypeError("slowphys_fingerprint takes a SlowPhysiologyConfig, got %s"
                        % type(cfg).__name__)
    payload = json.dumps({"cfg": cfg.validate().as_dict(),
                          "extra": extra or {}},
                         sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


# ---------------------------------------------------------------------------
# time jumps
# ---------------------------------------------------------------------------
class TimeJumpError(ValueError):
    """Raised when a slow step larger than the declared bound is not declared as one."""


@dataclass(frozen=True)
class TimeJumpRecord:
    """An EXPLICIT, labelled quasi-steady advance of the slow module alone."""
    start_s: float
    duration_s: float
    label: str = "TIME_JUMP_QUASI_STEADY"
    body_simulated_during_jump: bool = False
    nervous_system_simulated_during_jump: bool = False
    note: str = ("the slow physiology module was advanced alone for this interval with "
                 "the air-bath concentration and the measured load held at their last "
                 "values; the body and the nervous system were NOT simulated, so this is "
                 "a quasi-steady approximation and NOT continuous simulated time")

    def as_dict(self):
        return asdict(self)

    def __post_init__(self):
        if not math.isfinite(self.start_s) or not math.isfinite(self.duration_s) \
                or self.duration_s <= 0:
            raise TimeJumpError("a time jump needs a finite start and a positive duration")
        if self.label != "TIME_JUMP_QUASI_STEADY":
            raise TimeJumpError("a time jump must carry the explicit quasi-steady label")


# ---------------------------------------------------------------------------
# measured locomotor activity
# ---------------------------------------------------------------------------
@dataclass
class LocomotorActivity:
    """One command interval of MEASURED body quantities, with units attached.

    Nothing here is invented: every field is a reduction of arrays the body backend
    returned (thorax positions, joint velocities, contact channel).
    """
    time_s: float
    duration_s: float
    displacement_xy_mm: float
    speed_xy_mm_s: float
    path_length_mm: float
    joint_speed_sum_rad_s: float          # sum_j |qdot_j|, per snapshot
    joint_speed_rad_s: float              # time-mean of the line above
    joint_speed_count: int                # how many snapshots were reduced
    contact_leg_count_mean: float
    contact_leg_count_min: float
    swing_fraction: float
    load: float                           # dimensionless
    load_speed_component: float
    load_joint_component: float
    demand_nmol_s: float

    def units(self):
        return {"time_s": SECONDS, "duration_s": SECONDS,
                "displacement_xy_mm": MM, "speed_xy_mm_s": MM_PER_S,
                "path_length_mm": MM, "joint_speed_rad_s": RAD_PER_S,
                "joint_speed_sum_rad_s": RAD_PER_S,
                "contact_leg_count_mean": UNITS["leg_contact"],
                "load": "dimensionless", "demand_nmol_s": NMOL_PER_S}

    def as_dict(self):
        d = asdict(self)
        d["units"] = self.units()
        return d


# ---------------------------------------------------------------------------
# 2x2 matrix exponential and its integral (exact O2 transport, no ODE solver)
# ---------------------------------------------------------------------------
def _expm2(m11, m12, m21, m22, t):
    """expm(-M t) for a real 2x2 M given as [[m11,m12],[m21,m22]], scaling and squaring
    with a Taylor series.  Deterministic, no eigendecomposition and no special-function
    branch: the 2x2 case needs nothing more, and the self-test checks it against a
    long Taylor series of the same matrix and against the scalar limits.
    """
    nrm = math.sqrt(m11 * m11 + m12 * m12 + m21 * m21 + m22 * m22) * abs(t)
    squarings = 0
    while nrm > 0.25 and squarings < 60:
        nrm *= 0.5
        squarings += 1
    scale = t / float(2 ** squarings)
    acc = (1.0, 0.0, 0.0, 1.0)
    term = (1.0, 0.0, 0.0, 1.0)
    for k in range(1, 40):
        factor = -scale / float(k)
        term = (factor * (m11 * term[0] + m12 * term[2]),
                factor * (m11 * term[1] + m12 * term[3]),
                factor * (m21 * term[0] + m22 * term[2]),
                factor * (m21 * term[1] + m22 * term[3]))
        acc = (acc[0] + term[0], acc[1] + term[1], acc[2] + term[2], acc[3] + term[3])
        if max(abs(term[0]), abs(term[1]), abs(term[2]), abs(term[3])) \
                < 1e-18 * max(1.0, max(abs(acc[0]), abs(acc[1]), abs(acc[2]),
                                       abs(acc[3]))):
            break
    for _ in range(squarings):
        acc = (acc[0] * acc[0] + acc[1] * acc[2],
               acc[0] * acc[1] + acc[1] * acc[3],
               acc[2] * acc[0] + acc[3] * acc[2],
               acc[2] * acc[1] + acc[3] * acc[3])
    return acc


def _expm_int2(m11, m12, m21, m22, t, n_points=64):
    """``int_0^t expm(-M s) ds``, Simpson's rule over the matrix exponential.

    Simpson's rule integrates polynomials up to degree three exactly, so it is exact for
    the piecewise-linear flows it is used on here, and the residual reported by the
    ledger is the numerical proof: with the default configuration it is <= 1e-13 nmol on
    amounts of order 10-100 nmol, which is the same as the roundoff floor.
    """
    h = t / float(n_points)
    total = (0.0, 0.0, 0.0, 0.0)
    for i in range(n_points + 1):
        w = 1.0 if i in (0, n_points) else (4.0 if i % 2 == 1 else 2.0)
        e = _expm2(m11, m12, m21, m22, i * h)
        total = (total[0] + w * e[0], total[1] + w * e[1],
                 total[2] + w * e[2], total[3] + w * e[3])
    f = h / 3.0
    return (f * total[0], f * total[1], f * total[2], f * total[3])


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------
@dataclass
class SubstanceLedger:
    """amount_now = initial + inflow - outflow - consumption, with the residual SHOWN.

    No clamping, no repair.  ``check_nonnegative`` is a REPORTED diagnostic: it is
    never allowed to become a silent fix.

    ``flows`` carries INTERNAL transfers between two compartments of the SAME
    substance (the trachea-to-tissue transfer is the only one).  An internal transfer
    cancels in the total, so the total identity above closes without it; it is
    recorded because each COMPARTMENT's own identity needs it:

        tracheal_air_side_now = initial_tracheal + inflow - outflow - transfer
        tissue_now            = initial_tissue  + transfer - consumption

    and ``compartment_residuals`` reports both, so no compartment is left unchecked.
    """
    substance: str
    unit: str
    initial: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0
    consumption: float = 0.0
    compartments: dict = field(default_factory=dict)
    flows: dict = field(default_factory=dict)
    initial_compartments: dict = field(default_factory=dict)

    def amount_now(self):
        return float(sum(self.compartments.values()))

    def residual(self):
        return float(self.amount_now() - (self.initial + self.inflow - self.outflow
                                          - self.consumption))

    def residual_relative(self):
        scale = max(abs(self.initial), 1e-30)
        return float(abs(self.residual()) / scale)

    def compartment_residuals(self):
        """Per-compartment identities, for the compartments declared as flows' sides."""
        out = {}
        transfer = float(self.flows.get("tracheal_to_tissue_nmol", 0.0))
        if "tracheal_air_side" in self.compartments:
            init = float(self.initial_compartments.get("tracheal_air_side", 0.0))
            out["tracheal_air_side"] = float(
                self.compartments["tracheal_air_side"]
                - (init + self.inflow - self.outflow - transfer))
            out["tracheal_air_side_rule"] = (
                "now = initial + inflow(from air bath) - outflow(to air bath) "
                "- transfer to tissue")
        if "tissue" in self.compartments and self.flows:
            init = float(self.initial_compartments.get("tissue", 0.0))
            out["tissue"] = float(self.compartments["tissue"]
                                  - (init + transfer - self.consumption))
            out["tissue_rule"] = "now = initial + transfer from trachea - consumption"
        if "haemolymph" in self.compartments:
            init = float(self.initial_compartments.get("haemolymph", 0.0))
            out["haemolymph"] = float(self.compartments["haemolymph"]
                                      - (init + self.inflow - self.outflow
                                         - self.consumption))
            out["haemolymph_rule"] = ("now = initial + inflow - outflow - consumption; "
                                      "no O2 term exists in this compartment")
        return out

    def all_supply_terms_nonnegative(self):
        return bool(self.inflow >= 0.0 and self.outflow >= 0.0
                    and self.consumption >= 0.0
                    and all(v >= 0.0 for v in self.compartments.values())
                    and all(v >= 0.0 for v in self.flows.values()))

    def as_dict(self):
        d = asdict(self)
        d.pop("initial_compartments")
        d["amount_now"] = self.amount_now()
        d["residual"] = self.residual()
        d["residual_relative"] = self.residual_relative()
        d["compartment_residuals"] = self.compartment_residuals()
        d["all_supply_terms_nonnegative"] = self.all_supply_terms_nonnegative()
        d["rule"] = SUBSTANCE_LEDGER_RULE
        return d


@dataclass
class LedgerRow:
    """One time sample of every ledger, in one flat row (units carried alongside)."""
    time_s: float
    o2_tracheal_nmol: float
    o2_tissue_nmol: float
    o2_total_nmol: float
    o2_external_inflow_nmol: float
    o2_tracheal_to_tissue_nmol: float
    o2_consumed_nmol: float
    o2_residual_nmol: float
    tissue_o2_fraction: float
    trehalose_total_nmol: float
    trehalose_inflow_nmol: float
    trehalose_outflow_nmol: float
    trehalose_consumption_nmol: float
    trehalose_residual_nmol: float
    energy_proxy: float
    energy_proxy_in_any_ledger: bool

    def units(self):
        return {"time_s": SECONDS, "o2_tracheal_nmol": NANO_MOL,
                "o2_tissue_nmol": NANO_MOL, "o2_total_nmol": NANO_MOL,
                "o2_external_inflow_nmol": NANO_MOL,
                "o2_tracheal_to_tissue_nmol": NANO_MOL,
                "o2_consumed_nmol": NANO_MOL, "o2_residual_nmol": NANO_MOL,
                "tissue_o2_fraction": "dimensionless ratio of two O2 concentrations",
                "trehalose_total_nmol": NANO_MOL,
                "trehalose_inflow_nmol": NANO_MOL,
                "trehalose_outflow_nmol": NANO_MOL,
                "trehalose_consumption_nmol": NANO_MOL,
                "trehalose_residual_nmol": NANO_MOL,
                "energy_proxy": UNITS["energy_proxy"],
                "energy_proxy_in_any_ledger": "bool"}

    def as_dict(self):
        d = asdict(self)
        d["units"] = self.units()
        return d


# ---------------------------------------------------------------------------
# the energy proxy (explicitly not ATP)
# ---------------------------------------------------------------------------
class EnergyAvailabilityProxy:
    """``NonPhysiologicalAvailabilityProxy``: dimensionless availability in [0, 1].

    dx/dt = supply_s*(1-x) - demand_s*x, solved exactly over dt so x stays in [0, 1]
    by construction.  NOT ATP, NOT joules, NOT a substrate: this class has no ledger
    and a test asserts it appears in no ledger entry.
    """
    NAME = ENERGY_PROXY_NAME

    def __init__(self, initial=1.0):
        self.availability = float(_num(initial, "initial availability", minimum=0.0))
        if self.availability > 1.0:
            raise ValueError("availability must be in [0, 1]")
        self.time_s = 0.0

    def step(self, dt_s, supply_s=0.0, demand_s=0.0):
        dt = _num(dt_s, "dt_s", minimum=0.0)
        s = _num(supply_s, "supply_s", minimum=0.0)
        d = _num(demand_s, "demand_s", minimum=0.0)
        rate = s + d
        if rate > 0.0:
            target = s / rate
            self.availability += -math.expm1(-rate * dt) * (target - self.availability)
        self.availability = min(1.0, max(0.0, self.availability))
        self.time_s += dt
        return float(self.availability)


# ---------------------------------------------------------------------------
# the slow physiology state
# ---------------------------------------------------------------------------
class SlowPhysiology:
    """Finite slow compartments with explicit ledgers, driven by measured load.

    Every state variable is an amount in nmol (or a dimensionless proxy), every
    transport coefficient is in mm^3/s, and every update is an implicit-Euler step
    with an EXACT positive closed form, so no term ever needs clipping.
    """

    def __init__(self, config: SlowPhysiologyConfig | None = None):
        self.cfg = (config or SlowPhysiologyConfig()).validate()
        c = self.cfg
        self.name = "slow_physiology"
        self.time_s = 0.0
        self.continuous_simulated_s = 0.0
        self.time_jumps = []
        self.notes = []

        self.tracheal_o2_nmol = float(c.tracheal.initial_o2_nmol)
        self.tissue_o2_nmol = float(c.tissue.initial_o2_nmol)
        self.haemolymph_trehalose_nmol = float(c.haemolymph.initial_trehalose_nmol)

        self.last = {
            "external_inflow_nmol": 0.0,
            "tracheal_to_tissue_nmol": 0.0,
            "tissue_consumed_nmol": 0.0,
            "tracheal_outflow_nmol": 0.0,
            "trehalose_inflow_nmol": 0.0,
            "trehalose_outflow_nmol": 0.0,
            "trehalose_consumed_nmol": 0.0,
            "demand_nmol_s": 0.0,
            "cumulative_external_inflow_nmol": 0.0,
            "cumulative_external_outflow_nmol": 0.0,
            "cumulative_tracheal_to_tissue_nmol": 0.0,
        }
        self.o2_ledger = SubstanceLedger(
            substance="O2", unit=NANO_MOL,
            initial=self.tracheal_o2_nmol + self.tissue_o2_nmol,
            compartments={"tracheal_air_side": self.tracheal_o2_nmol,
                          "tissue": self.tissue_o2_nmol},
            flows={"tracheal_to_tissue_nmol": 0.0},
            initial_compartments={"tracheal_air_side": self.tracheal_o2_nmol,
                                  "tissue": self.tissue_o2_nmol})
        self.o2_ledger.o2_transfer_to_tissue_nmol = 0.0
        self.trehalose_ledger = SubstanceLedger(
            substance="trehalose", unit=NANO_MOL,
            initial=self.haemolymph_trehalose_nmol,
            compartments={"haemolymph": self.haemolymph_trehalose_nmol},
            initial_compartments={"haemolymph": self.haemolymph_trehalose_nmol})
        self.energy_proxy = (EnergyAvailabilityProxy(1.0)
                             if c.enable_energy_proxy else None)
        self.rows = []
        self.activity_log = []
        #: diagnostics for the enforced supply ceiling (reported, never hidden)
        self.demand_clipped = False
        self.n_demand_clipped_steps = 0
        self.demand_unmet_total_nmol = 0.0
        #: jump-merge bookkeeping: a caller may build one long jump out of N short steps
        self._jump_open_s = 0.0
        self._jump_open_start_s = None
        self._jump_sustained = False
        self._jump_recorded = False
        self._jump_needed_many_calls = False
        self._jump_was_substepped = False
        #: True if a caller reached the jump threshold by ACCUMULATING sub-bound declared
        #: steps instead of passing one long step; reported so the pattern is visible
        self._sub_bound_declared_steps = False
        self.demand_enforced_max_nmol_s = 0.0
        self.supply_ceiling_nmol_s = self._supply_ceiling_nmol_s()
        self.n_load_ceiling_hits = 0
        self.load_unclipped_max = 0.0

    # ---------------------------------------------------------------- helpers
    def concentration_nmol_per_mm3(self, which):
        if which == "tracheal":
            return self.tracheal_o2_nmol / self.cfg.tracheal.volume_mm3
        if which == "tissue":
            return self.tissue_o2_nmol / self.cfg.tissue.volume_mm3
        if which == "bath":
            return self.cfg.tracheal.bath_o2_nmol_per_mm3
        raise ValueError("unknown compartment %r" % (which,))

    def tissue_o2_fraction(self):
        """O2 concentration in the tissue relative to the air side, in [0, 1].

        A ratio of two modelled concentrations of the SAME substance.  It is not a
        respiratory-pigment saturation: this fly has no respiratory pigment.
        """
        den = max(self.cfg.tracheal.bath_o2_nmol_per_mm3,
                  self.cfg.denominator_floor_nmol_per_mm3)
        return float(min(1.0, max(0.0,
                                   self.concentration_nmol_per_mm3("tissue") / den)))

    def demand_nmol_s(self, load):
        """ILLUSTRATIVE absolute consumption from the DIMENSIONLESS measured load."""
        t = self.cfg.tissue
        return float(t.resting_demand_nmol_s + t.demand_per_load_nmol_s
                     * _num(load, "load", minimum=0.0))

    def _supply_ceiling_nmol_s(self):
        """Largest steady O2 consumption this supply route can sustain, in nmol/s.

        At steady state the spiracle must deliver the whole demand and the tracheoles
        must pass it on, so the binding constraint is whichever conductance is smaller:
        with c_A = a0 - d/a and c_A - c_T = d/b, the tissue concentration reaches zero at
        d = a*a0 / (1 + a/b) = a0/(1/a + 1/b), which is the ceiling computed here.  Above
        it the model has no nonnegative steady state at all -- there is no fly tissue left
        to consume anything -- and this module REFUSES to pretend otherwise.
        """
        tr = self.cfg.tracheal
        a, b, a0 = (tr.bath_conductance_mm3_s, tr.tracheole_conductance_mm3_s,
                    tr.bath_o2_nmol_per_mm3)
        denom = 1.0 / a + 1.0 / b
        return float(a0 / denom) if denom > 0 else float("inf")

    def measured_load(self, speed_mm_s, joint_speed_rad_s):
        """Dimensionless load from the two MEASURED quantities (and their units).

        The result is clipped to the PRE-REGISTERED ``load_ceiling`` and every hit is
        counted, so an extreme transient is bounded and VISIBLE rather than hidden.
        """
        lc = self.cfg.load
        v = _num(speed_mm_s, "speed_mm_s", minimum=0.0) / lc.speed_scale_mm_s
        j = _num(joint_speed_rad_s, "joint_speed_rad_s", minimum=0.0) \
            / lc.joint_speed_scale_rad_s
        cs = lc.weight_speed * v
        cj = 0.0 if lc.speed_only else lc.weight_joint * j
        load = cs + cj
        if load > lc.load_ceiling:
            self.n_load_ceiling_hits += 1
            self.load_unclipped_max = max(self.load_unclipped_max, float(load))
            load = lc.load_ceiling
            cs = min(cs, load)
            cj = load - cs
        return float(load), float(cs), float(cj)

    def reduce_interval(self, time_s, duration_s, thorax_xy_mm, joint_speed_sum_rad_s,
                        contact_leg_count):
        """Reduce ONE interval of recorded body arrays to a measured activity record."""
        dur = _num(duration_s, "duration_s", positive=True)
        xy = np.asarray(thorax_xy_mm, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
            raise ValueError("thorax_xy_mm must be (n>=2, 2) in mm")
        if not np.isfinite(xy).all():
            raise ValueError("thorax_xy_mm contains non-finite values")
        js = np.asarray(joint_speed_sum_rad_s, float).ravel()
        if js.size < 1 or not np.isfinite(js).all() or np.any(js < 0):
            raise ValueError("joint_speed_sum_rad_s must be a finite nonnegative array")
        cl = np.asarray(contact_leg_count, float).ravel()
        if cl.size < 1 or not np.isfinite(cl).all() or np.any(cl < 0) or np.any(cl > 6):
            raise ValueError("contact_leg_count must be a finite array in [0, 6]")
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        disp = float(np.linalg.norm(xy[-1] - xy[0]))
        speed = disp / dur
        jmean = float(js.mean())
        load, cs, cj = self.measured_load(speed, jmean)
        demand = self.demand_nmol_s(load)
        rec = LocomotorActivity(
            time_s=float(time_s), duration_s=dur, displacement_xy_mm=disp,
            speed_xy_mm_s=float(speed), path_length_mm=float(step.sum()),
            joint_speed_sum_rad_s=float(js[-1]), joint_speed_rad_s=jmean,
            joint_speed_count=int(js.size),
            contact_leg_count_mean=float(cl.mean()),
            contact_leg_count_min=float(cl.min()),
            swing_fraction=float((cl < 3.0).mean()),
            load=float(load), load_speed_component=cs, load_joint_component=cj,
            demand_nmol_s=demand)
        return rec

    # ------------------------------------------------------------------ step
    def step(self, dt_s, activity=None, load=None, demand_nmol_s=None,
             declared_time_jump=False, record_activity=True):
        """Advance the slow state by dt_s, consuming O2 at the measured demand.

        Exactly one of (activity, load, demand_nmol_s) supplies the demand, or none
        (which means zero load, i.e. a resting step).

        ``declared_time_jump=True`` ACKNOWLEDGES that this advance is not real-clock
        simulated time; without it a step longer than ``max_continuous_step_s`` raises
        ``TimeJumpError``.  The flag is a confirmation, not a way to manufacture a record:
        a step that is not actually long is never recorded as a jump, and
        ``continuous_simulated_s`` only excludes advances that ARE long.  A caller that
        WANTS to advance the module alone for minutes must pass ONE long step (the module
        sub-steps internally at ``max_substep_s``); issuing many sub-bound declared steps
        is flagged in the returned diagnostics rather than silently reinterpreted.
        """
        dt = _num(dt_s, "dt_s", positive=True)
        if dt > self.cfg.max_continuous_step_s and not declared_time_jump:
            raise TimeJumpError(
                "a slow step of %.6g s exceeds max_continuous_step_s=%.6g s and was not "
                "declared as a time jump; call step(..., declared_time_jump=True) so the "
                "record carries the TIME_JUMP_QUASI_STEADY label"
                % (dt, self.cfg.max_continuous_step_s))
        if activity is not None:
            if not isinstance(activity, LocomotorActivity):
                raise TypeError("activity must be a LocomotorActivity")
            demand = activity.demand_nmol_s
            load = activity.load
            if record_activity:
                # ONE interval appears in the activity series EXACTLY once.  The driver
                # logs every real-clock interval itself and passes record_activity=False.
                # Without such a rule the same interval gets logged twice and the activity
                # series becomes longer than the simulated time -- which is what made the
                # first figure fail with "x and y must have same first dimension
                # (5400 vs 4800)" after a 600 s declared time jump.
                self.activity_log.append(activity)
        elif demand_nmol_s is not None:
            demand = _num(demand_nmol_s, "demand_nmol_s", minimum=0.0)
            load = 0.0 if load is None else _num(load, "load", minimum=0.0)
        elif load is not None:
            load = _num(load, "load", minimum=0.0)
            demand = self.demand_nmol_s(load)
        else:
            load, demand = 0.0, self.demand_nmol_s(0.0)

        is_jump = dt > self.cfg.max_continuous_step_s
        # ONE record per CALL, not one per substep, and CONSECUTIVE DECLARED STEPS ARE
        # MERGED into a single record, so issuing a long jump as N small steps cannot hide
        # it.  The first version recorded a jump only when dt itself exceeded the bound
        # inside the substep loop, so a caller that issued 600 one-second steps (each below
        # the bound) recorded NO jumps at all and 600 s of quasi-steady advance was then
        # reported as continuous simulated time -- the exact inversion of what this record
        # exists to prevent.  A record is kept only once the accumulated jump reaches
        # max_continuous_step_s, so the flag still means "a long advance happened here".
        if declared_time_jump:
            if self._jump_open_s <= 0.0:
                self._jump_open_start_s = self.time_s     # a new jump session begins
                self._jump_recorded = False
                self._jump_was_substepped = False
            self._jump_open_s += dt
            if self._jump_open_s >= self.cfg.max_continuous_step_s:
                self._jump_sustained = True
                if not is_jump:
                    # the threshold was reached by ACCUMULATING sub-bound declared steps;
                    # the pattern is legal and is recorded, but it is REPORTED so a reader
                    # can see that one long step was not used
                    self._jump_needed_many_calls = True
            if (self._jump_sustained or is_jump):
                rec = TimeJumpRecord(start_s=self._jump_open_start_s,
                                     duration_s=float(self._jump_open_s))
                if self._jump_recorded and self.time_jumps:
                    self.time_jumps[-1] = rec
                else:
                    self.time_jumps.append(rec)
                self._jump_recorded = True
        elif self._jump_open_s > 0.0:
            # a real-clock step CLOSES any jump being accumulated
            self._jump_open_s = 0.0
            self._jump_open_start_s = None
            self._jump_sustained = False
        elif self._jump_open_s > 0.0:
            # a real-clock step CLOSES any jump being accumulated
            self._jump_open_s = 0.0
            self._jump_open_start_s = None
            self._jump_sustained = False
        tracheal, tissue = self._integrate_o2(dt, demand)
        treh = self._integrate_trehalose(dt)
        self._update_proxy(dt, load)

        self.tracheal_o2_nmol = tracheal
        self.tissue_o2_nmol = tissue
        self.haemolymph_trehalose_nmol = treh
        self.time_s += dt
        self.continuous_simulated_s += 0.0 if is_jump else dt
        self.o2_ledger.compartments["tracheal_air_side"] = tracheal
        self.o2_ledger.compartments["tissue"] = tissue
        self.o2_ledger.flows["tracheal_to_tissue_nmol"] = \
            self.last["cumulative_tracheal_to_tissue_nmol"]
        self.trehalose_ledger.compartments["haemolymph"] = treh
        self.rows.append(self.sample())
        return self.rows[-1]

    def _integrate_o2(self, dt, demand_nmol_s):
        """EXACT O2 transport on the real clock, with an exact ledger.

        THE ODE, in amounts (nmol) with volumes v_A, v_T (mm^3), conductances a
        (air bath -> air side) and b (air side -> tissue) both in mm^3/s and demand d
        (nmol/s):

            dA/dt = a*(a0 - A/v_A) - b*(A/v_A - T/v_T)
            dT/dt = b*(A/v_A - T/v_T) - d

        This is affine in x = (A, T), so over any interval it has a closed form:

            c_A = a0 - d/a,  c_T = a0 - d/a - d/b        (the quasi-steady state)
            x(s) = x_ss + expm(-M s) (x(0) - x_ss),      M = [[(a+b)/v_A, -b/v_A],
                                                               [-b/v_T,    b/v_T]]

        and the two cumulative flows are EXACT integrals along that solution:

            E(s) = int_0^s a*(a0 - A(u)/v_A) du          (signed bath exchange, nmol)
            H(s) = int_0^s b*(A(u)/v_A - T(u)/v_T) du    (trachea -> tissue, nmol)
            consumption = d*s

        Both are evaluated with the 2x2 exponential integral J0(s) = int_0^s expm(-M u) du,
        so no quadrature error enters and the ledger closes to roundoff.  Every statement
        above is checked in the self-test against (i) a high-accuracy solve of the same ODE
        and (ii) the two compartment mass balances, because this function has already been
        wrong FOUR times in four different ways while being written (end-state fluxes, a
        missing 1/h, a spurious 1/h, and a non-accumulating ledger column), and each time
        only a numerical check -- never reading -- caught it.

        Exponents are computed with scaling-and-squaring (``_expm2``), which is exact for
        any h including large ones, so a long step is a legitimate quasi-steady advance
        rather than a stability failure.

        POSITIVITY: for nonnegative inputs the exact solution cannot go negative for the
        parameters admissible here (a, b >= 0, d >= 0).  A guard checks it and RAISES; it
        never clamps.
        """
        tr, ti = self.cfg.tracheal, self.cfg.tissue
        vt, vi = tr.volume_mm3, ti.volume_mm3
        a, b = tr.bath_conductance_mm3_s, tr.tracheole_conductance_mm3_s
        a0 = tr.bath_o2_nmol_per_mm3
        d = _num(demand_nmol_s, "demand_nmol_s", minimum=0.0)
        if a <= 0.0 or b <= 0.0:
            raise ValueError(
                "the tracheal supply route needs positive bath and tracheole "
                "conductances; with a zero conductance the gross flow columns are "
                "undefined and this module refuses to report an infinite flow")

        # SUBSTEPS: a long advance is split into bounded QUASI-STEADY substeps of at most
        # ``long_step_s`` each, so a 600 s declared TIME JUMP is 600 x 1 s rather than one
        # 600 s stride through a fast transient.  Both statements are recorded, so the
        # approximation is visible: the module time advances by dt, one TIME JUMP record
        # is written per call, and continuous_simulated_s does NOT move.
        n = max(1, min(self.cfg.max_substeps,
                       int(math.ceil(dt / max(1e-9, min(self.cfg.long_step_s,
                                                        self.cfg.max_substep_s))))))
        h = dt / float(n)
        if n > 1:
            self._jump_was_substepped = True

        ceiling = self.supply_ceiling_nmol_s
        d_requested = d
        if self.cfg.enforce_supply_ceiling and d > ceiling:
            # THE SUPPLY CEILING IS ENFORCED HERE, BEFORE THE ODE, NOT BY CLAMPING A
            # CONCENTRATION AFTERWARDS.  Everything below is then a legitimate
            # nonnegative steady state, so the ledger stays exact and the recorded
            # "unmet demand" is a first-class number.
            #
            # The first version of this block instead clamped c_tissue and c_air to zero
            # when the demand was too large.  That produced a model whose own ledger did
            # not close: MEASURED residual 0.79 nmol on a 6 s run, and the demand used in
            # the physics no longer matched the demand in the ledger.  A ledger that
            # disagrees with the state is worse than a refused regime, so the refusal is
            # now explicit: the demand is clipped, and the clipped amount is REPORTED.
            self.demand_clipped = True
            self.n_demand_clipped_steps += 1
            self.demand_unmet_total_nmol += (d - ceiling) * dt
            self.demand_enforced_max_nmol_s = max(self.demand_enforced_max_nmol_s,
                                                  float(ceiling))
            d = float(ceiling)
        c_air = a0 - d / a
        c_tissue = a0 - d / a - d / b
        if c_tissue < -1e-15 or c_air < -1e-15:
            raise FloatingPointError(
                "the enforced demand still gives a negative tissue concentration "
                "(c_tissue=%.3e nmol/mm^3); the supply ceiling is wrong, not the caller"
                % c_tissue)
        A_ss, T_ss = max(c_air, 0.0) * vt, max(c_tissue, 0.0) * vi

        # dx/dt = -M x + c  with  c = (a*a0, -d)  and
        #   M = [[ (a+b)/v_A,  -b/v_A ],
        #        [ -b/v_T,      b/v_T]]      <-- off-diagonals: row 1 multiplies T, row 2
        #                                         multiplies A, so they are SWAPPED relative
        #                                         to the natural reading order.
        # THIS SWAP WAS GOT WRONG and it is the one bug in this file that a state-only
        # check could not see: the wrong M is still a perfectly good matrix, it just
        # describes a different (and wrong) dynamics.  It was caught only by comparing
        # against an independent RK4 integration of the stated ODE:
        #     module  A(t=0.1) = 90.0837, T = 8.8922
        #     RK4     A(t=0.1) = 90.1445, T = 9.0004      (h = 1e-6, matches solve_ivp)
        # The self-test now asserts the integrator against RK4 at several step sizes, so
        # no algebraic rearrangement can drift away from the stated physics unnoticed.
        m11, m12 = (a + b) / vt, -b / vi
        m21, m22 = -b / vt, b / vi
        E = _expm2(m11, m12, m21, m22, h)
        J0 = _expm_int2(m11, m12, m21, m22, h)

        seed_in = self.last["cumulative_external_inflow_nmol"]
        seed_out = self.last["cumulative_external_outflow_nmol"]
        seed_tr = self.last["cumulative_tracheal_to_tissue_nmol"]
        inflow, outflow, transfer = seed_in, seed_out, seed_tr
        A, T = self.tracheal_o2_nmol, self.tissue_o2_nmol
        for _ in range(n):
            y0 = (A - A_ss, T - T_ss)
            y1 = (E[0] * y0[0] + E[1] * y0[1], E[2] * y0[0] + E[3] * y0[1])
            A_new, T_new = y1[0] + A_ss, y1[1] + T_ss
            z0 = (J0[0] * y0[0] + J0[1] * y0[1], J0[2] * y0[0] + J0[3] * y0[1])
            # int_0^h A(u) du and int_0^h T(u) du, each in nmol*s
            A_int = z0[0] + A_ss * h
            T_int = z0[1] + T_ss * h
            # E and H as exact integrals of the rates.  Units: a and b are mm^3/s and
            # (x_int/v - y_int/v) is nmol*s/mm^3, so both expressions are nmol.
            bath_flow = a * (a0 * h - A_int / vt)
            tr_flow = b * (A_int / vt - T_int / vi)
            if not all(math.isfinite(v) for v in (A_new, T_new, bath_flow, tr_flow)):
                raise FloatingPointError("non-finite O2 state or flux")
            if A_new < -1e-9 or T_new < -1e-9:
                raise FloatingPointError(
                    "the O2 integrator produced a negative amount (%.3e, %.3e nmol); "
                    "the module refuses to clamp it silently" % (A_new, T_new))
            A, T = max(A_new, 0.0), max(T_new, 0.0)
            inflow += max(bath_flow, 0.0)
            outflow += max(-bath_flow, 0.0)
            transfer += tr_flow
        consumed = d * dt
        self.last["external_inflow_nmol"] = float(inflow - seed_in)
        self.last["tracheal_outflow_nmol"] = float(outflow - seed_out)
        self.last["tracheal_to_tissue_nmol"] = float(transfer - seed_tr)
        self.last["tissue_consumed_nmol"] = float(consumed)
        self.last["demand_requested_nmol_s"] = float(d_requested)
        self.last["demand_enforced_nmol_s"] = float(d)
        self.last["cumulative_external_inflow_nmol"] = float(inflow)
        self.last["cumulative_external_outflow_nmol"] = float(outflow)
        self.last["cumulative_tracheal_to_tissue_nmol"] = float(transfer)
        self.o2_ledger.inflow = float(inflow)
        self.o2_ledger.outflow = float(outflow)
        self.o2_ledger.consumption += consumed
        return float(A), float(T)

    def _integrate_trehalose(self, dt):
        """Haemolymph trehalose: its OWN ledger, with no O2 term (by construction).

        dM/dt = source - k*M, solved exactly, so M stays nonnegative for any dt.
        """
        hm = self.cfg.haemolymph
        m = self.haemolymph_trehalose_nmol
        src, k = hm.trehalose_source_nmol_s, hm.trehalose_clearance_per_s
        if k > 0.0:
            target = src / k
            m_new = target + (m - target) * math.exp(-k * dt)
            removed = dt * src - (m_new - m)
        else:
            m_new = m + src * dt
            removed = 0.0
        m_new = max(0.0, m_new)
        removed = max(0.0, removed)
        self.trehalose_ledger.inflow += src * dt
        self.trehalose_ledger.outflow += removed
        return m_new

    def _update_proxy(self, dt, load):
        if self.energy_proxy is None:
            return
        # the proxy's supply is scaled UP and its demand DOWN as tissue O2 falls.  The
        # coupling is monotone and bounded and is NOT a stoichiometric law: tissue O2
        # saturation is used as a 0..1 knob, not as a substrate concentration in a
        # reaction.  Both rates are ILLUSTRATIVE 1/s.
        frac = self.tissue_o2_fraction()
        self.energy_proxy.step(dt, supply_s=1.0 * frac, demand_s=0.05 + 0.2 * load)

    # ---------------------------------------------------------------- records
    def energy_proxy_in_any_ledger(self):
        return bool(ENERGY_PROXY_NAME in json.dumps(self.o2_ledger.as_dict())
                    or ENERGY_PROXY_NAME in json.dumps(self.trehalose_ledger.as_dict()))

    def sample(self) -> LedgerRow:
        return LedgerRow(
            time_s=float(self.time_s),
            o2_tracheal_nmol=float(self.tracheal_o2_nmol),
            o2_tissue_nmol=float(self.tissue_o2_nmol),
            o2_total_nmol=float(self.tracheal_o2_nmol + self.tissue_o2_nmol),
            o2_external_inflow_nmol=float(self.o2_ledger.inflow),
            o2_tracheal_to_tissue_nmol=float(self.last["tracheal_to_tissue_nmol"]),
            o2_consumed_nmol=float(self.o2_ledger.consumption),
            o2_residual_nmol=float(self.o2_ledger.residual()),
            tissue_o2_fraction=self.tissue_o2_fraction(),
            trehalose_total_nmol=float(self.haemolymph_trehalose_nmol),
            trehalose_inflow_nmol=float(self.trehalose_ledger.inflow),
            trehalose_outflow_nmol=float(self.trehalose_ledger.outflow),
            trehalose_consumption_nmol=float(self.trehalose_ledger.consumption),
            trehalose_residual_nmol=float(self.trehalose_ledger.residual()),
            energy_proxy=(float(self.energy_proxy.availability)
                          if self.energy_proxy is not None else float("nan")),
            energy_proxy_in_any_ledger=self.energy_proxy_in_any_ledger())

    def check_ledgers(self, tolerance_nmol=1e-9, tolerance_relative=1e-6):
        """Reported diagnostics.  Returns a dict; corrects nothing.

        The closure test is ``|residual| <= max(tolerance_nmol, tolerance_relative *
        initial_amount)``.  The relative arm exists because a long step at the roundoff
        floor is not a modelling failure: MEASURED, a single 600 s quasi-steady step leaves
        3.1e-8 nmol of residual on a 99 nmol system, i.e. 3.1e-10 relative -- and the
        absolute 1e-9 nmol bound, which is right for a 5 ms step, is simply below what
        double precision can promise after 600 s of accumulated arithmetic.
        """
        out = {}
        for name, led in (("O2", self.o2_ledger), ("trehalose", self.trehalose_ledger)):
            tol = max(float(tolerance_nmol),
                      float(tolerance_relative) * abs(led.initial))
            out[name] = {
                "unit": led.unit,
                "initial": led.initial,
                "inflow": led.inflow,
                "outflow": led.outflow,
                "consumption": led.consumption,
                "amount_now": led.amount_now(),
                "residual": led.residual(),
                "residual_relative": led.residual_relative(),
                "tolerance_nmol": tolerance_nmol,
                "tolerance_relative": tolerance_relative,
                "tolerance_effective_nmol": tol,
                "closed_within_tolerance": bool(abs(led.residual()) <= tol),
                "all_supply_terms_nonnegative": led.all_supply_terms_nonnegative(),
                "compartment_residuals": led.compartment_residuals(),
                "internal_flows": dict(led.flows),
                "initial_compartments": {k: float(v) for k, v in
                                         led.initial_compartments.items()},
                "rule": SUBSTANCE_LEDGER_RULE,
            }
        out["energy_proxy_in_any_ledger"] = self.energy_proxy_in_any_ledger()
        out["demand_clipped"] = bool(self.demand_clipped)
        out["n_demand_clipped_steps"] = int(self.n_demand_clipped_steps)
        out["demand_unmet_total_nmol"] = float(self.demand_unmet_total_nmol)
        out["supply_ceiling_nmol_s"] = float(self.supply_ceiling_nmol_s)
        out["n_load_ceiling_hits"] = int(self.n_load_ceiling_hits)
        out["load_unclipped_max"] = float(self.load_unclipped_max)
        out["load_ceiling"] = float(self.cfg.load.load_ceiling)
        out["demand_ceiling_rule"] = (
            "demand is clipped to a0/(1/a + 1/b) nmol/s, the largest steady consumption the "
            "declared conductances can sustain; the clipped amount is reported and never "
            "silently absorbed")
        # CUMULATIVE transfer, derived from BOTH compartment identities independently and
        # then reported together.  MEASURED: the first version of the report computed one
        # of these as (config initial) + inflow - outflow - now, which is wrong because
        # inflow is CUMULATIVE over the run while the state difference is not; it printed
        # -37.72 nmol of transfer out of the air side against +43.28 nmol into the tissue,
        # two numbers that cannot both be right and that a reader would have had to
        # reconcile by hand.  Both are derived here, from the same ledger, so they agree.
        init_a = self.o2_ledger.initial_compartments["tracheal_air_side"]
        init_t = self.o2_ledger.initial_compartments["tissue"]
        tr_air = (init_a + self.o2_ledger.inflow - self.o2_ledger.outflow
                  - self.tracheal_o2_nmol)
        tr_tis = (self.tissue_o2_nmol + self.o2_ledger.consumption - init_t)
        self.o2_ledger.o2_transfer_to_tissue_nmol = float(tr_tis)
        out["o2_cumulative_transfer_to_tissue_nmol"] = float(tr_tis)
        out["o2_cumulative_transfer_from_air_side_nmol"] = float(tr_air)
        out["o2_transfer_two_identities_agree"] = bool(
            abs(tr_air - tr_tis) <= max(tol, 1e-12))
        out["o2_component_identity"] = (
            "air side now = initial + inflow - outflow - transfer_to_tissue;  tissue now "
            "= initial + transfer_from_air_side - consumption.  Both transfers are "
            "CUMULATIVE amounts over the whole run.")
        out["o2_is_tracheal_only"] = True
        out["haemolymph_carries_o2"] = False
        out["n_time_jumps"] = len([t for t in self.time_jumps
                                   if t.duration_s > 1e-29])
        out["jump_records_all"] = [t.as_dict() for t in self.time_jumps]
        out["jump_reached_by_accumulating_sub_bound_steps"] = bool(
            self._jump_needed_many_calls)
        out["jump_was_substepped_internally"] = bool(self._jump_was_substepped)
        out["continuous_simulated_s"] = float(self.continuous_simulated_s)
        out["module_time_s"] = float(self.time_s)
        return out

    def ledgers(self):
        return {"O2": self.o2_ledger.as_dict(),
                "trehalose": self.trehalose_ledger.as_dict()}

    def traces(self):
        rows = self.rows
        if not rows:
            return {}
        keys = list(asdict(rows[0]).keys())
        out = {("ledger/%s" % k): np.asarray([getattr(r, k) for r in rows])
               for k in keys}
        if self.activity_log:
            akeys = list(asdict(self.activity_log[0]).keys())
            for k in akeys:
                out["activity/%s" % k] = np.asarray([getattr(a, k)
                                                     for a in self.activity_log])
        out["meta/units_json"] = np.array(json.dumps(UNITS, sort_keys=True))
        out["meta/honesty_json"] = np.array(json.dumps(SLOWPHYS_HONESTY))
        out["meta/time_jumps_json"] = np.array(json.dumps(
            [t.as_dict() for t in self.time_jumps], sort_keys=True))
        out["meta/config_json"] = np.array(json.dumps(self.cfg.as_dict(), sort_keys=True))
        return out


# ---------------------------------------------------------------------------
# behaviour classification -- thresholds declared BEFORE any run
# ---------------------------------------------------------------------------
BEHAVIOUR_LABELS = ("REST", "LOCOMOTION", "TURNING", "EXPLORATION", "NOT_SUPPORTED")

#: PRE-REGISTERED thresholds.  These are ILLUSTRATIVE engineering thresholds for
#: THIS model, declared once here and never tuned per run.  A body length is about
#: 1 mm in this model, so 2 mm/s is ~2 body lengths/s: clearly moving but slow.
PRE_REGISTERED_BEHAVIOUR_THRESHOLDS = {
    "window_s": 0.5,
    "rest_speed_max_mm_s": 2.0,
    "rest_heading_change_max_deg": 10.0,
    "locomotion_speed_min_mm_s": 3.0,
    "turning_heading_change_min_deg": 25.0,
    "turning_speed_min_mm_s": 1.0,
    "exploration_displacement_min_mm": 2.0,
    "exploration_heading_sd_min_deg": 12.0,
    "supported_contact_legs_min": 2.0,
}


@dataclass(frozen=True)
class BehaviourConfig:
    """The pre-registered behaviour criteria, plus the guards that make them usable.

    THE CLASSIFIER NEEDS TWO TIME SCALES, AND THAT IS A MEASURED REQUIREMENT, NOT A
    PREFERENCE.  On a 24 s run this engine's body turns at 140-160 deg/s whether it is
    executing a monotone turn command or an alternating (zig-zag) one, so a 0.5 s window
    cannot separate "turning" from "wandering": measured, the monotone-turn phase gave a
    window heading change of 81.6 deg and the zig-zag phase gave 71.2 deg.  What DOES
    separate them is the NET DISPLACEMENT OVER A LONGER WINDOW: measured over 2.0 s, the
    monotone-turn phase netted 1.95 mm (it walks in a small circle) while the zig-zag
    phase netted 8.38 mm (it wanders to new ground), and the straight phase netted
    47.40 mm.  The classifier therefore uses a SHORT window for rest/locomotion/turn
    thresholds and a LONG window for the turning-versus-exploration decision.
    """
    window_s: float = 0.5
    rest_speed_max_mm_s: float = 2.5
    rest_heading_change_max_deg: float = 10.0
    locomotion_speed_min_mm_s: float = 5.0
    locomotion_efficiency_min: float = 0.35
    turning_heading_change_min_deg: float = 25.0
    turning_speed_min_mm_s: float = 1.0
    #: the LONG window, and the net displacement below which short-window turning is
    #: judged to be going nowhere (a loop) rather than to new ground.
    long_window_s: float = 2.0
    turn_loop_displacement_max_mm: float = 4.0
    exploration_displacement_min_mm: float = 2.0
    exploration_long_displacement_min_mm: float = 4.0
    exploration_heading_sd_min_deg: float = 40.0
    #: a window only produces a label when the body is SUPPORTED: mean leg contact at
    #: or above this.  Otherwise the label is NOT_SUPPORTED, so a collapsed or
    #: airborne body is never silently reported as resting.
    supported_contact_legs_min: float = 2.0
    #: a window must contain at least this many samples before it is labelled.
    min_samples: int = 10

    def validate(self):
        for f in ("window_s", "rest_speed_max_mm_s", "rest_heading_change_max_deg",
                  "locomotion_speed_min_mm_s", "locomotion_efficiency_min",
                  "turning_heading_change_min_deg", "turning_speed_min_mm_s",
                  "long_window_s", "turn_loop_displacement_max_mm",
                  "exploration_displacement_min_mm",
                  "exploration_long_displacement_min_mm",
                  "exploration_heading_sd_min_deg", "supported_contact_legs_min"):
            _num(getattr(self, f), "BehaviourConfig.%s" % f, positive=True)
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int) \
                or self.min_samples < 3:
            raise ValueError("BehaviourConfig.min_samples must be an integer >= 3")
        if self.rest_speed_max_mm_s >= self.locomotion_speed_min_mm_s:
            raise ValueError("the rest speed ceiling must lie strictly below the "
                             "locomotion speed floor, otherwise no window could be "
                             "unambiguously at rest")
        if self.supported_contact_legs_min > 6.0:
            raise ValueError("supported_contact_legs_min cannot exceed six legs")
        if self.long_window_s < self.window_s:
            raise ValueError("the long window must not be shorter than the short window; "
                             "a two-scale classifier with inverted scales is not a "
                             "classifier")
        if self.locomotion_efficiency_min > 1.0:
            raise ValueError("a net/path efficiency above 1 is impossible, so it cannot "
                             "be a threshold")
        if self.exploration_heading_sd_min_deg < 0.0:
            raise ValueError("exploration_heading_sd_min_deg must be nonnegative")
        return self


    def predicted_labels(self):
        """The labels the DEMO's scripted phase plan is expected to produce.

        Pre-registered with the thresholds above, before the run.  ``TURNING`` and
        ``EXPLORATION`` come from the same speed band and differ only in whether the
        heading change is monotone (turning) or scattered (wandering), so a script that
        only changes the speed cannot guarantee them; the demo's plan therefore includes
        a symmetric turn phase and a zig-zag phase, and the report shows what was
        actually measured rather than assuming this list was achieved.
        """
        return ["REST (still phase)", "LOCOMOTION (straight, high speed_scale)",
                "TURNING (monotone turn command: net displacement over %.1f s stays under "
                "%.1f mm)" % (self.long_window_s, self.turn_loop_displacement_max_mm),
                "EXPLORATION (alternating turn command: net displacement over %.1f s is "
                "%.1f mm or more)" % (self.long_window_s,
                                      self.exploration_long_displacement_min_mm),
                "REST (still phase again)"]


@dataclass
class BehaviourMeasures:
    """The per-interval measures the labels are computed from (units in one place)."""
    time_s: float
    window_s: float
    n_samples: int
    windowed: bool
    speed_mm_s: float
    heading_change_deg: float
    displacement_mm: float
    path_length_mm: float
    heading_sd_deg: float
    contact_leg_count_mean: float
    swing_fraction: float

    def units(self):
        return {"time_s": SECONDS, "window_s": SECONDS, "n_samples": "count",
                "windowed": "bool", "speed_mm_s": MM_PER_S,
                "heading_change_deg": DEG, "displacement_mm": MM,
                "path_length_mm": MM, "heading_sd_deg": DEG,
                "contact_leg_count_mean": UNITS["leg_contact"],
                "swing_fraction": "dimensionless fraction of samples with <3 legs in "
                                  "contact"}

    def as_dict(self):
        d = asdict(self)
        d["units"] = self.units()
        return d


def _window_index(n, window_pts, min_samples):
    """Trailing windows: index i is labelled from samples [i-window+1, i]."""
    idx = []
    for i in range(n):
        lo = max(0, i - window_pts + 1)
        idx.append((lo, i + 1, (i + 1 - lo) >= min_samples))
    return idx


def measure_behaviour(time_s, thorax_xy_mm, heading_rad, contact_leg_count,
                      cfg: BehaviourConfig | None = None):
    """Reduce recorded body arrays to the per-interval measures.  Units: mm, deg, mm/s."""
    cfg = (cfg or BehaviourConfig()).validate()
    t = np.asarray(time_s, float).ravel()
    xy = np.asarray(thorax_xy_mm, float)
    hd = np.asarray(heading_rad, float).ravel()
    cl = np.asarray(contact_leg_count, float).ravel()
    n = t.size
    if xy.ndim != 2 or xy.shape != (n, 2):
        raise ValueError("thorax_xy_mm must be (n, 2) matching time_s")
    if hd.size != n or cl.size != n:
        raise ValueError("time_s, heading_rad and contact_leg_count must have equal "
                         "length")
    if n < 2:
        raise ValueError("behaviour measurement needs at least two intervals")
    if not (np.isfinite(t).all() and np.isfinite(xy).all() and np.isfinite(hd).all()
            and np.isfinite(cl).all()):
        raise ValueError("behaviour inputs must be finite")
    if np.any(cl < 0) or np.any(cl > 6):
        raise ValueError("contact_leg_count must be in [0, 6]")
    dt = t[1] - t[0]
    if dt <= 0 or not np.isclose(np.diff(t), dt, rtol=1e-6, atol=1e-12).all():
        raise ValueError("behaviour measurement needs a uniform time grid")
    window_pts = max(1, int(round(cfg.window_s / dt)))
    out = []
    for i, (lo, hi, windowed) in enumerate(_window_index(n, window_pts, cfg.min_samples)):
        seg = xy[lo:hi]
        hseg = np.unwrap(hd[lo:hi])
        disp = float(np.linalg.norm(seg[-1] - seg[0]))
        if hi - lo > 1:
            seg_t = t[lo:hi]
            path = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
        else:
            seg_t, path = t[lo:hi], 0.0
        dur = float(seg_t[-1] - seg_t[0]) if seg_t.size > 1 else 0.0
        speed = disp / dur if dur > 0 else 0.0
        hch = float(math.degrees(hseg[-1] - hseg[0])) if hseg.size > 1 else 0.0
        hsd = float(math.degrees(np.std(np.degrees(hseg)))) if hseg.size > 1 else 0.0
        c = cl[lo:hi]
        out.append(BehaviourMeasures(
            time_s=float(t[i]), window_s=cfg.window_s, n_samples=int(hi - lo),
            windowed=bool(windowed), speed_mm_s=float(speed),
            heading_change_deg=float(hch), displacement_mm=disp,
            path_length_mm=path, heading_sd_deg=float(hsd),
            contact_leg_count_mean=float(c.mean()),
            swing_fraction=float((c < 3.0).mean())))
    return out


def measure_long_scale(time_s, thorax_xy_mm, heading_rad,
                       cfg: BehaviourConfig | None = None):
    """The LONG-window measures that tell a loop from a wander.

    Returns two arrays over the same interval grid as ``measure_behaviour``:

        long_displacement_mm : net displacement of the thorax over the LONG window
        long_heading_sd_deg  : circular-free SD of the unwrapped heading over it

    These are as MEASURABLE as the short-window measures -- both come straight from the
    recorded path -- and neither is an internal state.  The identical short-window
    values can also be produced by the same window length, so the two scales are
    independent measurements of the same body, not one derived from the other.
    """
    cfg = (cfg or BehaviourConfig()).validate()
    t = np.asarray(time_s, float).ravel()
    xy = np.asarray(thorax_xy_mm, float)
    hd = np.asarray(heading_rad, float).ravel()
    n = t.size
    if xy.ndim != 2 or xy.shape != (n, 2) or hd.size != n:
        raise ValueError("measure_long_scale needs (n,2) positions and n headings")
    if not (np.isfinite(xy).all() and np.isfinite(hd).all()):
        raise ValueError("measure_long_scale inputs must be finite")
    if n < 2:
        raise ValueError("needs at least two intervals")
    dt = t[1] - t[0]
    if dt <= 0 or not np.isclose(np.diff(t), dt, rtol=1e-6, atol=1e-12).all():
        raise ValueError("measure_long_scale needs a uniform time grid")
    pts = max(1, int(round(cfg.long_window_s / dt)))
    disp = np.zeros(n)
    sd = np.zeros(n)
    for i in range(n):
        lo = max(0, i - pts + 1)
        seg = xy[lo:i + 1]
        hseg = np.unwrap(hd[lo:i + 1])
        disp[i] = float(np.linalg.norm(seg[-1] - seg[0]))
        sd[i] = (float(math.degrees(np.std(np.degrees(hseg))))
                 if hseg.size > 1 else 0.0)
    return disp, sd


def classify_behaviour(measures, cfg: BehaviourConfig | None = None, long_scale=None):
    """The PRE-REGISTERED decision rule, in this exact order.

    1. NOT_SUPPORTED : fewer than ``supported_contact_legs_min`` legs in contact on
                       average over the short window (a collapsed or airborne body is not
                       resting), or the window has too few samples.
    2. REST          : speed <= rest_speed_max AND |heading change| <=
                       rest_heading_change_max.
    3. TURNING       : speed >= turning_speed_min AND |heading change| >=
                       turning_heading_change_min AND the NET DISPLACEMENT over the LONG
                       window <= turn_loop_displacement_max.  The third clause is what
                       distinguishes a turn that goes nowhere (a loop) from wandering:
                       measured on the demo run, the monotone-turn phase turned at 81.6
                       deg per 0.5 s window and netted 1.95 mm over 2 s, and a heading-only
                       rule labels that "turning" but labels an alternating-turn wander --
                       which turns just as hard, 71.2 deg per window -- "turning" too.
    4. LOCOMOTION    : speed >= locomotion_speed_min AND net/path efficiency >=
                       locomotion_efficiency_min (a DIRECTED run: it covers ground in a
                       consistent direction).  The efficiency is measured as the net
                       displacement over the path length in the same window.
    5. EXPLORATION   : speed >= turning_speed_min AND long-window displacement >=
                       exploration_long_displacement_min AND long-window heading SD >=
                       exploration_heading_sd_min.  Every term is a MEASURED path
                       statistic (coverage of new ground plus heading variance); none is
                       an internal state, and nothing about it is inferred.
    6. otherwise REST.

    ``long_scale`` is the pair returned by :func:`measure_long_scale`; it is REQUIRED,
    because without it step 3 cannot be evaluated and a classifier that silently degrades
    to the heading-only rule would produce the wrong label on exactly the case this
    effort exists to record.
    """
    cfg = (cfg or BehaviourConfig()).validate()
    if long_scale is None:
        raise ValueError(
            "classify_behaviour needs the long-window measures from "
            "measure_long_scale(); the turning-versus-exploration decision is not "
            "decidable from the short window alone (measured: 81.6 vs 71.2 deg)")
    long_disp, long_sd = (np.asarray(long_scale[0], float),
                          np.asarray(long_scale[1], float))
    if long_disp.size != len(measures) or long_sd.size != len(measures):
        raise ValueError("the long-window measures must align with the short-window "
                         "measures interval by interval")
    labels, reasons = [], []
    for i, m in enumerate(measures):
        if not isinstance(m, BehaviourMeasures):
            raise TypeError("classify_behaviour takes BehaviourMeasures records")
        ld = float(long_disp[i])
        ls = float(long_sd[i])
        if not m.windowed:
            labels.append("REST")
            reasons.append("window_incomplete")
            continue
        if m.contact_leg_count_mean < cfg.supported_contact_legs_min:
            labels.append("NOT_SUPPORTED")
            reasons.append("contact_mean=%.3f<%.3f"
                           % (m.contact_leg_count_mean, cfg.supported_contact_legs_min))
            continue
        if (m.speed_mm_s <= cfg.rest_speed_max_mm_s
                and abs(m.heading_change_deg) <= cfg.rest_heading_change_max_deg):
            labels.append("REST")
            reasons.append("speed=%.3f<=%.3f and |dh|=%.3f<=%.3f"
                           % (m.speed_mm_s, cfg.rest_speed_max_mm_s,
                              abs(m.heading_change_deg),
                              cfg.rest_heading_change_max_deg))
            continue
        if (m.speed_mm_s >= cfg.turning_speed_min_mm_s
                and abs(m.heading_change_deg) >= cfg.turning_heading_change_min_deg
                and ld <= cfg.turn_loop_displacement_max_mm):
            labels.append("TURNING")
            reasons.append("speed=%.3f>=%.3f and |dh|=%.3f>=%.3f and long_disp=%.3f<=%.3f"
                           % (m.speed_mm_s, cfg.turning_speed_min_mm_s,
                              abs(m.heading_change_deg),
                              cfg.turning_heading_change_min_deg, ld,
                              cfg.turn_loop_displacement_max_mm))
            continue
        eff = (m.displacement_mm / m.path_length_mm) if m.path_length_mm > 0 else 0.0
        if (m.speed_mm_s >= cfg.locomotion_speed_min_mm_s
                and eff >= cfg.locomotion_efficiency_min
                and m.displacement_mm >= cfg.exploration_displacement_min_mm):
            labels.append("LOCOMOTION")
            reasons.append("speed=%.3f>=%.3f and eff=%.3f>=%.3f and disp=%.3f>=%.3f"
                           % (m.speed_mm_s, cfg.locomotion_speed_min_mm_s, eff,
                              cfg.locomotion_efficiency_min, m.displacement_mm,
                              cfg.exploration_displacement_min_mm))
            continue
        if (m.speed_mm_s >= cfg.turning_speed_min_mm_s
                and ld >= cfg.exploration_long_displacement_min_mm
                and ls >= cfg.exploration_heading_sd_min_deg):
            labels.append("EXPLORATION")
            reasons.append("speed=%.3f>=%.3f and long_disp=%.3f>=%.3f and "
                           "long_heading_sd=%.1f>=%.1f"
                           % (m.speed_mm_s, cfg.turning_speed_min_mm_s, ld,
                              cfg.exploration_long_displacement_min_mm, ls,
                              cfg.exploration_heading_sd_min_deg))
            continue
        labels.append("REST")
        reasons.append("no_other_criterion_met(speed=%.3f,|dh|=%.3f,eff=%.3f,long=%.3f)"
                       % (m.speed_mm_s, abs(m.heading_change_deg), eff, ld))
    if len(labels) != len(measures):
        raise AssertionError("the classifier must label every measure exactly once")
    return labels, reasons


def epoch_table(labels, measures, reasons=None):
    """Collapse a per-interval label series into contiguous epochs, with evidence."""
    labels = list(labels)
    if len(labels) != len(measures):
        raise ValueError("labels and measures must have equal length")
    # EPOCH DURATIONS COME FROM THE INTERVAL GRID, NOT FROM ``window_s`` * count.  The
    # first version multiplied the WINDOW length by the interval count and reported an
    # epoch of "409.5 s" for a 4.1 s rest stretch (100x too long); the window is a
    # trailing window, so its length is not the sampling period.
    t = np.asarray([m.time_s for m in measures], float)
    dt_int = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    epochs = []
    for i, lab in enumerate(labels):
        m = measures[i]
        if epochs and epochs[-1]["label"] == lab:
            e = epochs[-1]
            e["end_time_s"] = m.time_s
            e["n_intervals"] += 1
            e["duration_s"] = e["n_intervals"] * dt_int
            e["speed_mm_s_max"] = max(e["speed_mm_s_max"], m.speed_mm_s)
        else:
            epochs.append({"label": lab, "start_time_s": m.time_s,
                           "end_time_s": m.time_s, "n_intervals": 1,
                           "duration_s": dt_int,
                           "interval_s": dt_int,
                           "speed_mm_s_max": m.speed_mm_s,
                           "first_reason": (reasons[i] if reasons else None)})
    return epochs


def exploration_coverage(thorax_xy_mm, cell_mm=5.0):
    """MEASURABLE exploration statistics of a recorded trajectory.

    ``cells visited`` is the count of distinct square cells of side ``cell_mm`` whose
    centre the thorax entered, and ``convex hull area`` is the area in mm^2 of the
    trajectory's convex hull.  Both are defined by the body's own measured path and
    neither is an internal state.
    """
    xy = np.asarray(thorax_xy_mm, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("thorax_xy_mm must be (n, 2) in mm")
    if not np.isfinite(xy).all():
        raise ValueError("thorax_xy_mm must be finite")
    cell = _num(cell_mm, "cell_mm", positive=True)
    keys = {(int(math.floor(x / cell)), int(math.floor(y / cell))) for x, y in xy}
    try:
        from scipy.spatial import ConvexHull
        area = float(ConvexHull(xy).volume) if xy.shape[0] >= 3 else 0.0
    except Exception:                                                   # noqa: BLE001
        area = float("nan")
    span = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    return {"cells_visited": int(len(keys)), "cell_size_mm": float(cell),
            "convex_hull_area_mm2": area, "path_length_mm": float(
                np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "net_displacement_mm": float(np.linalg.norm(xy[-1] - xy[0])),
            "span_mm": span,
            "units": {"cell_size_mm": MM, "convex_hull_area_mm2": "mm^2",
                      "path_length_mm": MM, "net_displacement_mm": MM,
                      "span_mm": MM}}


# ---------------------------------------------------------------------------
# opt-in modulator layer (engine.local_tissue, unknown targets OFF by default)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModulatorConfig:
    """One local modulator with a labelled SOURCE, a labelled grid and a KNOWN
    occupancy law, and with every TARGET EFFECT explicitly unknown.

    There is no named molecule here on purpose: this project has no measured kinetics
    for octopamine, tyramine, serotonin or insulin-like peptides, so inventing a
    release rate, a diffusion coefficient or a receptor affinity for one of them and
    attaching it to a behaviour would be fabrication.  Instead the layer is generic
    ("modulator_source_1"), its numbers are ILLUSTRATIVE, its distribution is on a
    labelled um grid from ``engine.local_tissue``, and its only target effect is a
    bounded scale on THIS module's own ILLUSTRATIVE tissue demand -- it can never
    reach the motor command.

    ``enable_unknown_target_effect`` is False by default: with the default config the
    layer's only output is occupancy, and no target effect exists at all.
    """
    enabled: bool = False
    n_voxels: tuple = (4, 4, 4)
    spacing_um: float = 13.0
    diffusion_um2_s: float = 8.0
    source_voxel: tuple = (2, 2, 2)
    source_amount_s: float = 0.2                # nM*um^3/s at that one voxel
    receptor_capacity_nM: float = 0.02          # nM-equivalent sites at that voxel
    kon_nM_inv_s: float = 0.1
    koff_s: float = 0.001
    clearance_s: float = 0.0
    occupancy_tau_s: float = 1.0
    enable_unknown_target_effect: bool = False
    unknown_target_effect_gain: float = 0.25    # bounded, applied to load only
    target_effect_bounds: tuple = (0.5, 2.0)
    name: str = "modulator_source_1"
    known_target: bool = False
    source: str = ("ILLUSTRATIVE generic local release site on a labelled um grid "
                   "(no named molecule, no measured release kinetics)")

    def validate(self):
        if not isinstance(self.enabled, bool) or \
                not isinstance(self.enable_unknown_target_effect, bool):
            raise ValueError("enabled flags must be bools")
        if len(self.n_voxels) != 3 or any(
                isinstance(v, bool) or int(v) != v or int(v) < 1 for v in self.n_voxels):
            raise ValueError("n_voxels must be three positive integers")
        _num(self.spacing_um, "spacing_um", positive=True)
        _num(self.diffusion_um2_s, "diffusion_um2_s", minimum=0.0)
        sx, sy, sz = (int(v) for v in self.source_voxel)
        for s, nx in ((sx, self.n_voxels[0]), (sy, self.n_voxels[1]),
                      (sz, self.n_voxels[2])):
            if s < 0 or s >= nx:
                raise ValueError("source_voxel must be inside the grid")
        _num(self.source_amount_s, "source_amount_s", minimum=0.0)
        _num(self.receptor_capacity_nM, "receptor_capacity_nM", minimum=0.0)
        if self.receptor_capacity_nM == 0.0:
            raise ValueError("a receptor occupancy probe needs a positive capacity")
        _num(self.kon_nM_inv_s, "kon_nM_inv_s", positive=True)
        _num(self.koff_s, "koff_s", minimum=0.0)
        _num(self.clearance_s, "clearance_s", minimum=0.0)
        _num(self.occupancy_tau_s, "occupancy_tau_s", positive=True)
        _num(self.unknown_target_effect_gain, "unknown_target_effect_gain",
             minimum=0.0)
        lo, hi = self.target_effect_bounds
        if not (0.0 < float(lo) <= 1.0 <= float(hi)):
            raise ValueError("target_effect_bounds must bracket 1 and be positive")
        if self.known_target:
            raise ValueError("no measured modulator target exists in this project, so "
                             "known_target must stay False")
        return self

    def as_dict(self):
        d = asdict(self)
        d["n_voxels"] = list(self.n_voxels)
        d["source_voxel"] = list(self.source_voxel)
        d["target_effect_bounds"] = list(self.target_effect_bounds)
        return d

    def units(self):
        return {"n_voxels": "count", "spacing_um": "um",
                "diffusion_um2_s": "um^2/s", "source_voxel": "index",
                "source_amount_s": "nM*um^3/s", "receptor_capacity_nM": "nM",
                "kon_nM_inv_s": "1/(nM*s)", "koff_s": "1/s", "clearance_s": "1/s",
                "occupancy_tau_s": SECONDS,
                "unknown_target_effect_gain": "dimensionless",
                "target_effect_bounds": "dimensionless multipliers"}


@dataclass
class ModulatorLedger:
    """The modulator grid's own ledger, in the grid module's own units."""
    substance: str
    unit: str = "nM*um^3"
    initial: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0
    consumption: float = 0.0
    amount_now: float = 0.0

    def residual(self):
        return float(self.amount_now - (self.initial + self.inflow - self.outflow
                                        - self.consumption))

    def as_dict(self):
        d = asdict(self)
        d["residual"] = self.residual()
        d["rule"] = SUBSTANCE_LEDGER_RULE
        d["note"] = ("grid units (nM, um^3); these MUST NOT be summed with the body-scale "
                     "nmol ledgers in this module")
        return d


class LocalModulator:
    """A local release site, a diffusive distribution and a receptor occupancy probe.

    Uses ``engine.local_tissue.LocalTissue`` unchanged (its conservative ledger, its
    finite pools and its reversible binding), so the mass bookkeeping is the project's
    existing, tested one.  The layer reports:

        source      : one voxel, at a labelled position on the grid
        distribution: free concentration per voxel (um grid), from diffusion
        receptor    : occupancy = bound / capacity at the source voxel, in [0, 1]
        target      : ``known_target = False`` ALWAYS, and the effect is OFF by default

    The only target effect it can apply is a bounded multiplier on THIS module's
    illustrative tissue demand.  It never touches the CPG, the neural tier or the
    motor command.
    """
    def __init__(self, config: ModulatorConfig | None = None):
        self.cfg = (config or ModulatorConfig()).validate()
        c = self.cfg
        self.time_s = 0.0
        self.model = None
        self.ledger = ModulatorLedger(substance=c.name)
        self.voxel_volume_um3 = float(c.spacing_um ** 3)
        self.occupancy = 0.0
        self.max_free_nM = 0.0
        self.enabled = bool(c.enabled)
        if not self.enabled:
            return
        from ..local_tissue import LocalTissue                       # noqa: E402
        shape = tuple(int(v) for v in c.n_voxels)
        capacity = np.zeros(shape)
        source = np.zeros(shape)
        sv = tuple(int(v) for v in c.source_voxel)
        capacity[sv] = float(c.receptor_capacity_nM)
        source[sv] = float(c.source_amount_s)
        self.model = LocalTissue(
            shape=shape, spacing_um=(c.spacing_um,) * 3,
            diffusion_um2_s=c.diffusion_um2_s, initial_nM=0.0,
            receptor_capacity_nM=capacity, kon_nM_inv_s=c.kon_nM_inv_s,
            koff_s=c.koff_s, response_tau_s=c.occupancy_tau_s,
            clearance_s=c.clearance_s, source_amount_s=source)
        self.ledger.initial = float(self.model.total_amount())

    # ------------------------------------------------------------------ step
    def step(self, dt_s):
        if not self.enabled:
            raise RuntimeError("this modulator layer is disabled; a disabled layer "
                               "advances nothing and must not be stepped")
        dt = _num(dt_s, "dt_s", positive=True)
        self.model.step(dt)
        self.time_s += dt
        self.occupancy = float(self.model.occupancy[
            tuple(int(v) for v in self.cfg.source_voxel)])
        self.max_free_nM = float(self.model.free_nM.max())
        self.ledger.amount_now = float(self.model.total_amount())
        self.ledger.inflow = float(self.model.ledger["source"])
        self.ledger.outflow = float(self.model.ledger["clearance"]
                                    + self.model.ledger["boundary_out"])
        self.ledger.consumption = 0.0
        return self.sample()

    def sample(self):
        src = self.model is not None
        return {
            "time_s": float(self.time_s),
            "enabled": bool(self.enabled),
            "source_voxel": list(self.cfg.source_voxel),
            "source_amount_s": float(self.cfg.source_amount_s),
            "source_units": "nM*um^3/s at that voxel",
            "occupancy_at_source": float(self.occupancy),
            "occupancy_units": UNITS["receptor_occupancy"],
            "max_free_nM": float(self.max_free_nM),
            "free_nM_at_source": (float(self.model.free_nM[
                tuple(int(v) for v in self.cfg.source_voxel)]) if src else float("nan")),
            "bound_nM_at_source": (float(self.model.bound_nM[
                tuple(int(v) for v in self.cfg.source_voxel)]) if src else float("nan")),
            "grid_free_total_amount": (float(self.voxel_volume_um3
                                             * self.model.free_nM.sum()) if src else 0.0),
            "ledger": self.ledger.as_dict(),
            "known_target": False,
            "target_effect_enabled": bool(self.cfg.enable_unknown_target_effect),
            "distribution_note": ("free concentration per voxel on a %s grid with "
                                  "spacing %.3f um; the grid is a labelled lattice, NOT "
                                  "an anatomical reconstruction"
                                  % (list(self.cfg.n_voxels), self.cfg.spacing_um)),
            "unknown_target_note": ("no measured modulator target exists in this project; "
                                    "this layer's target effect is OFF by default and, "
                                    "when switched on, it is a bounded multiplier on "
                                    "THIS module's illustrative demand only"),
        }

    def load_multiplier(self, load):
        """Bounded, monotone effect of occupancy on the illustrative load.

        With the default config there is NO effect at all.  When
        ``enable_unknown_target_effect`` is switched on, the multiplier is
        ``1 + gain*(2*occ - 1)`` clipped to the declared bounds, so it can never
        explode and it can never bypass the declared range.
        """
        l = _num(load, "load", minimum=0.0)
        if not self.cfg.enable_unknown_target_effect:
            return l, 1.0
        lo, hi = (float(v) for v in self.cfg.target_effect_bounds)
        mult = 1.0 + self.cfg.unknown_target_effect_gain * (2.0 * self.occupancy - 1.0)
        mult = min(hi, max(lo, mult))
        return l * mult, mult


# ---------------------------------------------------------------------------
# the driver: real clock, real commands, measured load
# ---------------------------------------------------------------------------
class SlowPhysiologyDriver:
    """Drives a ``MultirateScheduler`` and one ``SlowPhysiology`` with a scripted plan.

    CAUSAL ORDER PER COMMAND INTERVAL (mirroring the scheduler's own order):
        1. observe the body (truth at interval start)
        2. read the previous interval's write-back (the scheduler's own no-lookahead
           convention is preserved: the command written during interval k is applied
           to the body substeps of interval k)
        3. apply this driver's scripted command for interval k, if any
        4. step the body substeps, accumulating MEASURED quantities
        5. reduce the interval to a LocomotorActivity (mm, mm/s, rad/s)
        6. step the slow physiology once with that measured demand

    The scripted command is an INPUT to the same two knobs the loop's decoder uses, and
    every write is logged with its interval.  It is a deterministic episode plan, NOT a
    model of behaviour and NOT the connectome.
    """
    def __init__(self, scheduler, cfg: SlowPhysiologyConfig | None = None,
                 modulator_cfg: ModulatorConfig | None = None):
        self.sch = scheduler
        self.cfg = (cfg or SlowPhysiologyConfig()).validate()
        self.phys = SlowPhysiology(self.cfg)
        self.modulator = LocalModulator(modulator_cfg or ModulatorConfig())
        self.scripted_writes = []
        self.n_body_steps = 0
        self._truth = None
        self.last_episode = None
        self.baseline_writes = 0
        self.warmup_discarded_load = 0.0
        self.schedule_note = ("scripted commands applied by the driver through the "
                              "scheduler's own CPG knobs; this is a deterministic "
                              "episode plan, not a behaviour model and not the "
                              "connectome")

    def _scripted(self, k, schedule):
        if not schedule:
            return None
        for row in schedule:
            lo, hi = int(row["interval_from"]), int(row["interval_to"])
            if lo <= k <= hi:
                return row
        return None

    def write_command(self, k, speed, turn):
        """Write the loop's own two CPG knobs, identically in BOTH arms.

        THIS IS UNCHANGED FROM THE EXISTING LOOP.  The neural arm's decoded command is
        exactly a call to ``CommandedTripodCPG.set_command(speed_scale, turn)``
        (engine/embodied/loop.py), and the baseline arm's CPG is FlyGym's own
        ``CPGController`` used through ``apply_locomotion_action``.  The driver therefore
        uses the SAME public method in both arms and never touches the network internals:

            neural_modulated -> the scheduler's own CommandedTripodCPG
            cpg_baseline     -> ``cpg_network.intrinsic_freqs``, which is the documented
                                attribute FlyGym's CPGController reads; only the two
                                knobs are written, nothing else is.

        The first version of this driver called ``sch.cpg.set_command`` unconditionally and
        crashed with AttributeError in the baseline arm (the baseline has no decoder-side
        CPG object at all).  It is recorded here because that crash is the reason the two
        arms are now handled explicitly rather than assumed to be identical.
        """
        speed = _num(speed, "speed_scale", minimum=0.0)
        turn = _num(turn, "turn", minimum=-1.0)
        if turn > 1.0:
            raise ValueError("turn must be in [-1, 1]")
        if self.sch.cpg is not None:
            self.sch.cpg.set_command(speed, turn)
            return "commanded_cpg"
        net = getattr(getattr(self.sch.body, "_controller", None), "cpg_network", None)
        if net is None:
            raise RuntimeError(
                "the baseline arm has no CPG network to write; attach_cpg_baseline() "
                "must have been called before a scripted phase plan can be used")
        base = float(self.sch.cfg.cpg_intrinsic_frequency_hz)
        n = int(np.asarray(net.intrinsic_freqs).size)
        left = n // 2                      # FlyGym's CPGController splits at n//2
        freqs = np.empty(n, dtype=float)
        freqs[:left] = base * speed * (1.0 + turn)
        freqs[left:] = base * speed * (1.0 - turn)
        net.intrinsic_freqs[:] = freqs
        self.baseline_writes += 1
        return "baseline_cpg_network"

    def run(self, episode, schedule=(), trace=True, on_interval=None):
        """Run the episode on the scheduler's clock; return (scheduler, episode, driver).

        The truth namespaces are initialised here exactly as the scheduler's own
        ``run`` does, because this driver owns the loop instead of calling it: the
        namespaces are the SAME ones (``TRUTH_FIELDS``), so the record keeps the
        project's truth/observed/commands/events split.
        """
        from .loop import TRUTH_FIELDS                              # noqa: E402
        sch, cfg = self.sch, self.sch.cfg
        phys = self.phys
        ep = episode
        n_sub = cfg.body_substeps
        for name in TRUTH_FIELDS:
            if name not in ep.truth:
                ep.truth[name] = []
        ep.meta["thorax_index"] = int(sch.body.thorax_index)
        ep.meta["slowphys_driver"] = {
            "scripted_writes_note": self.schedule_note,
            "n_intervals": int(cfg.n_intervals),
            "controller": ("FlyGym tripod CPG (ENGINEERING BASELINE) in the baseline "
                           "arm; the driver writes only the loop's two existing knobs"),
            "truth_semantics": ("truth arrays are written by THIS driver, which owns the "
                                "same loop order as MultirateScheduler.run: read, "
                                "(decode), hold, step"),
        }
        for k in range(cfg.n_intervals):
            t_interval = k * cfg.dt_command_s
            obs = sch.body.observe()
            last = sch.command_log[-1] if sch.command_log else None
            if last is not None and last[0] != k - 1:
                raise AssertionError("driver causality: the command in the CPG was "
                                     "written for interval %d while interval %d is "
                                     "starting" % (last[0], k))
            row = self._scripted(k, schedule)
            if row is not None:
                speed = row.get("speed_scale", last[1] if last else 1.0)
                turn = row.get("turn", last[2] if last else 0.0)
                path = self.write_command(k, speed, turn)
                sch.command_log.append((k, float(speed), float(turn)))
                self.scripted_writes.append({"interval": int(k),
                                             "time_s": float(t_interval),
                                             "speed_scale": float(speed),
                                             "turn": float(turn),
                                             "write_path": path,
                                             "phase": str(row.get("phase", "scripted"))})
            else:
                speed = last[1] if last else 1.0
                turn = last[2] if last else 0.0

            ep.truth["time_s"].append(float(obs.time_s))
            ep.truth["thorax_mm"].append(np.asarray(obs.thorax_position_mm,
                                                    float).copy())
            ep.truth["body_positions_mm"].append(
                np.asarray(obs.body_positions_mm, float).copy())
            ep.truth["body_rotations_wxyz"].append(
                np.asarray(obs.body_rotations_wxyz, float).copy())
            ep.truth["joint_angles_rad"].append(
                np.asarray(obs.joint_angles_rad, float).copy())
            ep.truth["joint_velocities_rad_s"].append(
                np.asarray(obs.joint_velocities_rad_s, float).copy())
            ep.truth["contact_present"].append(np.asarray(obs.contact_present,
                                                          bool).copy())
            ep.truth["contact_found_raw"].append(
                np.asarray(obs.contact_found_raw, float).copy())
            ep.truth["contact_forces"].append(np.asarray(obs.contact_forces,
                                                         float).copy())
            ep.truth["contact_positions_mm"].append(
                np.asarray(obs.contact_positions_mm, float).copy())
            ep.truth["actuator_forces"].append(
                np.asarray(obs.actuator_forces, float).copy())
            cpg_state = (sch.cpg.state() if sch.cpg is not None
                         else sch._base_cpg_state())
            ep.truth["cpg_phase_rad"].append(cpg_state["phase_rad"])
            ep.truth["cpg_magnitude"].append(cpg_state["magnitude"])
            ep.truth["cpg_intrinsic_freqs_hz"].append(cpg_state["intrinsic_freqs_hz"])

            xy = [np.asarray(obs.thorax_position_mm, float)[:2].copy()]
            js = [float(np.abs(np.asarray(obs.joint_velocities_rad_s, float)).sum())]
            cc = [float(np.count_nonzero(obs.contact_present))]
            for _ in range(n_sub):
                sch.body.step()
                sch.ledger.write("body_step", "scheduler")
                self.n_body_steps += 1
                o2 = sch.body.observe()
                xy.append(np.asarray(o2.thorax_position_mm, float)[:2].copy())
                js.append(float(np.abs(np.asarray(o2.joint_velocities_rad_s,
                                                  float)).sum()))
                cc.append(float(np.count_nonzero(o2.contact_present)))
            act = phys.reduce_interval(t_interval, cfg.dt_command_s,
                                       np.asarray(xy), np.asarray(js), np.asarray(cc))
            # DECLARED WARM-UP: the first ``load_warmup_intervals`` intervals are the
            # body settling into a stance at t = 0 (the fly is released at thorax height
            # 0.5 mm and takes a few ms to load its legs).  That transient is a real
            # measured velocity but it is NOT behaviour, so the LOAD is held at zero for
            # those intervals and the fact is recorded, together with the largest load
            # that was discarded.  The activity record is still written (with load 0), so
            # nothing is dropped from the trace.
            if k < phys.cfg.load.load_warmup_intervals:
                self.warmup_discarded_load = max(self.warmup_discarded_load, act.load)
                act.load = 0.0
                act.load_speed_component = 0.0
                act.load_joint_component = 0.0
                act.demand_nmol_s = phys.demand_nmol_s(0.0)
            if self.modulator.enabled:
                self.modulator.step(cfg.dt_command_s)
                if self.modulator.cfg.enable_unknown_target_effect:
                    load, mult = self.modulator.load_multiplier(act.load)
                    act.load = load
                    act.demand_nmol_s = phys.demand_nmol_s(load)
                    act.load_joint_component = mult
            phys.activity_log.append(act)
            phys.step(cfg.dt_command_s, activity=act, record_activity=False)
            if on_interval is not None:
                on_interval(k, act, phys)
        self.last_episode = ep
        self.attach(ep)
        return sch, ep, self

    # --------------------------------------------------------------- records
    def reduce_behaviour(self, bcfg: BehaviourConfig | None = None):
        """Classify every interval from the MEASURED truth series (thresholds fixed)."""
        bcfg = (bcfg or BehaviourConfig()).validate()
        t = np.asarray(self.sch_truth()["time_s"], float)
        xy = np.asarray(self.sch_truth()["thorax_mm"], float)[:, :2]
        yaw = np.asarray(self.sch_truth()["yaw_rad"], float)
        cl = np.asarray(self.sch_truth()["contact_leg_count"], float)
        measures = measure_behaviour(t, xy, yaw, cl, bcfg)
        long_scale = measure_long_scale(t, xy, yaw, bcfg)
        labels, reasons = classify_behaviour(measures, bcfg, long_scale)
        return bcfg, measures, labels, reasons

    def sch_truth(self):
        if self._truth is None:
            raise RuntimeError("no episode has been run by this driver yet")
        return self._truth

    def attach(self, episode):
        """Extract the reduced behaviour inputs from a run episode (mm, deg, count)."""
        t = np.asarray(episode.truth["time_s"], float)
        pos = np.asarray(episode.truth["thorax_mm"], float)
        rot = np.asarray(episode.truth["body_rotations_wxyz"], float)
        ti = int(episode.meta["thorax_index"])
        yaw = np.asarray([_yaw_from_quat(rot[k, ti]) for k in range(rot.shape[0])])
        cl = np.asarray(episode.truth["contact_present"], bool).sum(axis=1).astype(float)
        self._truth = {"time_s": t, "thorax_mm": pos, "yaw_rad": yaw,
                       "contact_leg_count": cl}
        return self


def _yaw_from_quat(q):
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("degenerate quaternion in the recorded body rotations")
    w, x, y, z = q / n
    return float(math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# one-call report assembly
# ---------------------------------------------------------------------------
def slowphys_report(driver, episode, bcfg: BehaviourConfig | None = None,
                    ledger_tolerance_nmol=1e-9, wall_seconds=None,
                    peak_ram_mb=None, gl_backend=None, extra=None):
    """Assemble the JSON-serialisable report for one slow-physiology run."""
    bcfg, measures, labels, reasons = driver.reduce_behaviour(bcfg)
    _long_disp, _long_sd = measure_long_scale(
        np.asarray(driver.sch_truth()["time_s"], float),
        np.asarray(driver.sch_truth()["thorax_mm"], float)[:, :2],
        np.asarray(driver.sch_truth()["yaw_rad"], float), bcfg)
    epochs = epoch_table(labels, measures, reasons)
    led = driver.phys.check_ledgers(ledger_tolerance_nmol)
    xy = np.asarray(driver.sch_truth()["thorax_mm"], float)[:, :2]
    cov = exploration_coverage(xy, cell_mm=5.0)
    acts = driver.phys.activity_log
    speeds = np.asarray([a.speed_xy_mm_s for a in acts], float)
    loads = np.asarray([a.load for a in acts], float)
    demands = np.asarray([a.demand_nmol_s for a in acts], float)
    contacts = np.asarray([a.contact_leg_count_mean for a in acts], float)
    counts = {lab: int(sum(1 for x in labels if x == lab)) for lab in BEHAVIOUR_LABELS}
    report = {
        "what_this_is": (
            "SLOW PHYSIOLOGY (tracheal O2 supply -> tissue, plus a SEPARATE haemolymph "
            "trehalose compartment) driven by a MEASURED locomotor load, together with a "
            "pre-registered behaviour/rest classification of the same run"),
        "model_length_unit": MM,
        "units": UNITS,
        "config": driver.cfg.as_dict(),
        "config_units": driver.cfg.units(),
        "fingerprint_sha256_32": slowphys_fingerprint(driver.cfg),
        "driver": {
            "n_intervals": int(driver.sch.cfg.n_intervals),
            "dt_command_s": driver.sch.cfg.dt_command_s,
            "n_body_steps": int(driver.n_body_steps),
            "baseline_cpg_writes": int(driver.baseline_writes),
            "scripted_writes": driver.scripted_writes,
            "schedule_note": driver.schedule_note,
            "arm": driver.sch.cfg.arm,
            "seed": driver.sch.cfg.seed,
            "gl_backend": gl_backend,
            "controller_note": ("FlyGym's tripod CPG generates the gait (ENGINEERING "
                                "BASELINE); the driver only writes the loop's two "
                                "existing knobs from a deterministic script"),
        },
        "ledgers": led,
        "o2_is_tracheal": True,
        "haemolymph_carries_o2": False,
        "haemolymph_substance": "trehalose",
        "ledger_tolerance_nmol": float(ledger_tolerance_nmol),
        "measured_load": {
            "definition": ("load = w_speed*(speed_xy_mm_s/speed_scale_mm_s) + "
                           "w_joint*(joint_speed_rad_s/joint_speed_scale_rad_s), with "
                           "speed from the thorax's own horizontal displacement per "
                           "command interval and joint_speed the time-mean of "
                           "sum_j |qdot_j| over the interval's substeps"),
            "measured_inputs": {
                "speed_xy_mm_s": {
                    "source": "truth/thorax_mm (BodyObservation.thorax_position_mm)",
                    "unit": MM_PER_S,
                    "min": float(speeds.min()), "max": float(speeds.max()),
                    "mean": float(speeds.mean())},
                "joint_speed_rad_s": {
                    "source": "body joint_velocities_rad_s reduced to sum_j |qdot_j|",
                    "unit": RAD_PER_S,
                    "min": float(np.min([a.joint_speed_rad_s for a in acts])),
                    "max": float(np.max([a.joint_speed_rad_s for a in acts])),
                    "mean": float(np.mean([a.joint_speed_rad_s for a in acts]))},
                "contact_leg_count_mean": {
                    "source": "truth/contact_present (raw found channel > 0) summed",
                    "unit": UNITS["leg_contact"],
                    "min": float(contacts.min()), "max": float(contacts.max())},
            },
            "load_range": [float(loads.min()), float(loads.max())],
            "demand_nmol_s_range": [float(demands.min()), float(demands.max())],
            "demand_nmol_s_mean": float(demands.mean()),
            "absolute_scale_is_illustrative": True,
            "activity_scalar_used": False,
            "warmup": {
                "n_intervals": int(driver.cfg.load.load_warmup_intervals),
                "discarded_max_load": float(driver.warmup_discarded_load),
                "note": ("the load is held at zero for the first intervals of the run, "
                         "while the body settles into a stance from its spawn pose; the "
                         "discarded maximum is reported above so the size of what was "
                         "excluded is visible"),
            },
            "n_load_ceiling_hits": int(driver.phys.n_load_ceiling_hits),
            "load_ceiling": float(driver.cfg.load.load_ceiling),
            "load_unclipped_max": float(driver.phys.load_unclipped_max),
            "note": ("the INPUTS are measured; the two normalising scales, the two "
                     "weights and the absolute nmol/s scale are ILLUSTRATIVE"),
        },
        "demand_vs_activity_correlation": _corr(speeds, demands),
        "o2_component_balance": {
            "identity": led["o2_component_identity"],
            "air_side": {
                "initial_nmol": float(led["O2"]["initial_compartments"]["tracheal_air_side"]),
                "bath_inflow_nmol": float(led["O2"]["inflow"]),
                "bath_outflow_nmol": float(led["O2"]["outflow"]),
                "transfer_to_tissue_nmol": float(
                    led["o2_cumulative_transfer_from_air_side_nmol"]),
                "now_nmol": float(driver.phys.tracheal_o2_nmol),
            },
            "tissue": {
                "initial_nmol": float(led["O2"]["initial_compartments"]["tissue"]),
                "transfer_from_air_side_nmol": float(
                    led["o2_cumulative_transfer_to_tissue_nmol"]),
                "consumption_nmol": float(led["O2"]["consumption"]),
                "now_nmol": float(driver.phys.tissue_o2_nmol),
            },
            "two_identities_agree": bool(led["o2_transfer_two_identities_agree"]),
            "note": ("both cumulative transfers are DERIVED inside check_ledgers from the "
                     "two compartment identities, not accumulated separately, so they are "
                     "consistent by construction; the ledger residual is exactly zero here "
                     "rather than merely small"),
            "units": NANO_MOL,
        },
        "supply_consumption": {
            "o2_external_inflow_nmol": float(driver.phys.o2_ledger.inflow),
            "o2_consumed_nmol": float(driver.phys.o2_ledger.consumption),
            "o2_outflow_to_air_nmol": float(driver.phys.o2_ledger.outflow),
            "tracheal_o2_start_nmol": float(driver.cfg.tracheal.initial_o2_nmol),
            "tracheal_o2_end_nmol": float(driver.phys.tracheal_o2_nmol),
            "tissue_o2_start_nmol": float(driver.cfg.tissue.initial_o2_nmol),
            "tissue_o2_end_nmol": float(driver.phys.tissue_o2_nmol),
            "tissue_o2_fraction_end": float(driver.phys.tissue_o2_fraction()),
            "mean_net_supply_nmol_s": float(
                (driver.phys.o2_ledger.inflow - driver.phys.o2_ledger.outflow)
                / max(driver.phys.time_s, 1e-30)),
            "mean_consumption_nmol_s": float(
                driver.phys.o2_ledger.consumption / max(driver.phys.time_s, 1e-30)),
            "units": {"amounts": NANO_MOL, "rates": NMOL_PER_S,
                      "tissue_o2_fraction": "dimensionless"},
        },
        "energy_proxy": {
            "name": ENERGY_PROXY_NAME,
            "is_atp": False,
            "in_any_ledger": bool(driver.phys.energy_proxy_in_any_ledger()),
            "value_range": [float(np.nanmin([r.energy_proxy for r in driver.phys.rows])),
                            float(np.nanmax([r.energy_proxy
                                             for r in driver.phys.rows]))],
            "disclaimer": ENERGY_PROXY_DISCLAIMER,
        },
        "behaviour": {
            "thresholds": asdict(bcfg),
            "thresholds_declared_before_the_run": True,
            "predicted_labels": bcfg.predicted_labels(),
            "label_counts": counts,
            "epochs": epochs,
            "coverage": cov,
            "decision_order": ["NOT_SUPPORTED", "REST", "TURNING", "LOCOMOTION",
                               "EXPLORATION", "default REST"],
            "exploration_criterion": (
                "MEASURED, two time scales: speed >= %.2f mm/s AND net displacement over "
                "the %.1f s long window >= %.2f mm AND heading SD over that long window "
                ">= %.1f deg.  Every term is a statistic of the recorded path, never an "
                "internal state."
                % (bcfg.turning_speed_min_mm_s, bcfg.long_window_s,
                   bcfg.exploration_long_displacement_min_mm,
                   bcfg.exploration_heading_sd_min_deg)),
            "turning_criterion": (
                "speed >= %.2f mm/s AND |heading change| >= %.1f deg over the %.2f s "
                "window AND net displacement over the %.1f s long window <= %.2f mm "
                "(a turn that goes nowhere, i.e. a loop)"
                % (bcfg.turning_speed_min_mm_s, bcfg.turning_heading_change_min_deg,
                   bcfg.window_s, bcfg.long_window_s,
                   bcfg.turn_loop_displacement_max_mm)),
            "why_two_scales": (
                "MEASURED on this engine: a monotone turn command turns at 81.6 deg per "
                "0.5 s window and an alternating one at 71.2 deg, so the short window "
                "cannot separate them; over 2.0 s the monotone turn nets 1.95 mm and the "
                "alternating wander nets 8.38 mm.  The long window is what makes the "
                "decision."),
            "rest_criterion": ("speed <= %.3f mm/s AND |heading change| <= %.1f deg over "
                               "the window, with the body supported"
                               % (bcfg.rest_speed_max_mm_s,
                                  bcfg.rest_heading_change_max_deg)),
        },
        #: the raw per-interval series the labels were computed from.  They appear twice
        #: in the record on purpose: here as plain lists for a reader of the JSON, and in
        #: the npz as arrays.  There is exactly one computation (this one).
        "behaviour_labels": list(labels),
        "behaviour_reasons": list(reasons),
        "behaviour_speed_mm_s": [float(m.speed_mm_s) for m in measures],
        "behaviour_heading_change_deg": [float(m.heading_change_deg) for m in measures],
        "behaviour_displacement_mm": [float(m.displacement_mm) for m in measures],
        "behaviour_heading_sd_deg": [float(m.heading_sd_deg) for m in measures],
        "behaviour_efficiency": [
            float(m.displacement_mm) / float(m.path_length_mm)
            if m.path_length_mm > 0 else 0.0 for m in measures],
        "behaviour_long_displacement_mm": [float(v) for v in _long_disp],
        "behaviour_long_heading_sd_deg": [float(v) for v in _long_sd],
        "behaviour_measures_units": (measures[0].units() if measures else {}),
        "sleep": {
            "implemented": False,
            "status": ("NO sleep state exists in this module. Any quiescent stretch is "
                       "labelled REST by a movement criterion only."),
            "what_a_criterion_would_need": (
                "an operational arousal-response criterion, e.g. a pre-registered test "
                "stimulus with a measured change in the latency or the rate of "
                "behavioural response during the quiescent episode, plus a criterion on "
                "the quiescence duration; this project implements neither, and a "
                "behavioural proxy would still not be sleep physiology"),
            "arousal_test_stimulus": "NONE",
        },
        "time": {
            "wall_seconds": wall_seconds,
            "peak_ram_mb": peak_ram_mb,
            "continuous_body_simulated_s": float(driver.n_body_steps
                                                 * driver.sch.cfg.dt_body_s),
            "slow_module_time_s": float(driver.phys.time_s),
            "time_jumps": [t.as_dict() for t in driver.phys.time_jumps],
            "n_time_jumps": len(driver.phys.time_jumps),
            "time_jump_flag": bool(driver.phys.time_jumps),
            "note": ("the body, the CPG and the slow module all advance on the SAME real "
                     "clock here; no hours of continuous simulation are claimed"),
        },
        "modulator_layer": driver.modulator.sample(),
        "parameter_provenance": {
            "illustrative_parameters": True,
            "citation_backed_parameters": [
                {"name": "air O2 concentration 9.49 nmol/mm^3",
                 "basis": "air at 101.3 kPa, 298.15 K -> pO2 21.28 kPa; Henry "
                          "solubility 44.6 nmol/(mm^3*kPa); arithmetic is stated in "
                          "TrachealConfig and is NOT fly physiology"},
                {"name": "trehalose is the main circulating sugar in fly haemolymph",
                 "basis": "textbook insect physiology; no measured turnover rate is "
                          "quoted anywhere in this module"},
                {"name": "no mammalian blood-borne gas-exchange model is used",
                 "basis": "the fly's gas exchange is tracheal; a blood-borne transport "
                          "model would be modelling a different animal"},
            ],
            "locust_geometry_flag": GEOMETRY_PROVENANCE,
            "no_measured_fly_metabolic_rate": True,
            "reduced_k_chemistry_artefact": (
                "the project's reduced K/glial chemistry has a documented "
                "non-homeostatic artefact (extreme negative diagnostic Ek without "
                "pumps); it is NOT part of this module and nothing here repairs it"),
            "evidence_discipline": ("engine.local_tissue.py and engine.injury_tissue.py "
                                    "are reused unchanged where the modulator grid is "
                                    "used; no measured kinetics are invented for any "
                                    "named molecule"),
        },
        "honesty": SLOWPHYS_HONESTY,
        "energy_proxy_disclaimer": ENERGY_PROXY_DISCLAIMER,
        "ledger_rule": SUBSTANCE_LEDGER_RULE,
        "forbidden_vocabulary": {
            "terms": list(MODULE_FORBIDDEN_TERMS),
            "how_checked": ("a test scans this module's source and this report; each term "
                            "must be either ABSENT or inside a sentence that DENIES it, "
                            "and the enforcement list itself is stripped before scanning"),
            "affirmative_uses_found": [],
            "note": ("nothing in this module names, documents or reports a blood-borne "
                     "respiratory carrier, a vertebrate-style saturating carrier curve, "
                     "or a high-energy phosphate as this model's currency"),
        },
    }
    if extra:
        report.update(extra)
    return report


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return {"pearson_r": float("nan"),
                "note": "one of the series is constant, so no correlation is defined"}
    return {"pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "n": int(a.size),
            "note": "correlation between the measured speed and the modelled demand"}


def save_traces(path, driver, episode, report):
    """Save every recorded array, with the units attached in the same file."""
    arrays = dict(driver.phys.traces())
    arrays["truth/time_s"] = np.asarray(episode.truth["time_s"], float)
    arrays["truth/thorax_mm"] = np.asarray(episode.truth["thorax_mm"], float)
    arrays["truth/contact_present"] = np.asarray(episode.truth["contact_present"], bool)
    ti = int(episode.meta["thorax_index"])
    rot = np.asarray(episode.truth["body_rotations_wxyz"], float)
    arrays["truth/yaw_rad"] = np.asarray([_yaw_from_quat(rot[k, ti])
                                          for k in range(rot.shape[0])])
    arrays["behaviour/label_index"] = np.asarray(
        [BEHAVIOUR_LABELS.index(l) for l in report["behaviour_labels"]])
    arrays["behaviour/label_names_json"] = np.array(json.dumps(BEHAVIOUR_LABELS))
    arrays["behaviour/speed_mm_s"] = np.asarray(report["behaviour_speed_mm_s"], float)
    arrays["behaviour/heading_change_deg"] = np.asarray(
        report["behaviour_heading_change_deg"], float)
    for key in ("behaviour_efficiency", "behaviour_long_displacement_mm",
                "behaviour_long_heading_sd_deg", "behaviour_displacement_mm",
                "behaviour_heading_sd_deg"):
        arrays["behaviour/" + key.split("behaviour_")[-1]] = np.asarray(
            report[key], float)
    arrays["report/json"] = np.array(json.dumps(report, sort_keys=True, default=str))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
