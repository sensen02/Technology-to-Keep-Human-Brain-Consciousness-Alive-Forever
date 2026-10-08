"""engine.embodied.adapters -- the neural <-> body adapters for the first closed loop.

WHAT THIS FILE IS
-----------------
The two halves of this project finally meet here:

  * a PHYSICAL adult-fly body in a scene (``engine.embodied.body_backend``, FlyGym
    2.1.0 / NeuroMechFly v2 on MuJoCo 3.9.0), and
  * a CONNECTOME-DERIVED conductance point tier (``engine.neural_cond``) built from
    a bounded, deterministic SELECTION of the cached BANC connectome
    (``engine.neck_cut_data``).

This module contains every piece of glue and NOTHING else: the tier selection and
its coverage accounting, the receptor bank that turns a body state into a neural
current, the command decoder that turns descending spike counts into the two
numbers the CPG is allowed to see, the commanded CPG wrapper, the ownership ledger
that makes "who wrote that joint target" auditable, and the honesty block that is
restated on the figure and in the report.

THE ARMS (what is and is not the connectome)
--------------------------------------------
``cpg_baseline``   : FlyGym's own demo tripod CPG, completely unchanged.  This is an
                     ENGINEERING BASELINE, not a brain model, and it reproduces
                     milestone 1.
``neural_modulated``: the SAME tripod CPG still generates the gait.  The neural
                     tier's descending-neuron firing modulates exactly two things:
                     (i) the CPG intrinsic frequency and (ii) a left/right intrinsic
                     frequency asymmetry.  Nothing else is touched.

The rationale for this deliberately conservative first step: having a descending
command set SPEED and TURN while a local pattern generator produces the rhythm is
the smallest thing that can be called a closed loop.  The connectome does NOT
generate the gait here and no result in this project may be described that way.

UNITS, STATED ONCE AND USED EVERYWHERE
--------------------------------------
body:      length mm, time s, angle rad, contact force in the BACKEND's own unit
           (the channel's absolute calibration is UNRESOLVED -- see below).
neural:    time ms inside the tier, voltage mV, conductance uS, current nA,
           capacitance nF (this is the unit system of ``engine.neural_cond``).
receptor:  the receptor models in ``engine.receptors`` return
           ``gain_mV * normalised_state`` with the state in [0, 1]; the module
           labels that number "mV".  It is a normalised EQUIVALENT DRIVE, not a
           membrane potential and not a current.  The chain used here is explicit:
               physical stimulus -> normalised drive D in [0, gain_mV]
               -> I_nA = current_nA_per_drive * D      (ILLUSTRATIVE conversion)
               -> injected as the tier's external current.
           Both factors are ILLUSTRATIVE and are recorded in the episode metadata.

CONTACT-FORCE CALIBRATION IS UNRESOLVED (recorded, not hidden)
--------------------------------------------------------------
``BodyObservation.contact_forces`` is documented by the backend as "N".  The model
is a millimetre model (``gravity = -9810 mm/s^2``), whose consistent force unit is
kg*mm/s^2 = mN, and the measured per-leg magnitudes (~27 per leg, ~164 summed over
six legs) do NOT sum to the body weight (0.0010243 kg * 9.81 m/s^2 = 0.01005 N =
10.05 mN).  So the channel's ABSOLUTE scale does not balance against the body's own
mass and this module therefore uses only its RELATIVE pattern, against a single
pre-registered reference value that is recorded in every episode's metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os

import numpy as np

from ..neural_cond import ConductanceNetwork, ConductanceParams
from ..neck_cut_data import (
    TouchTierConfig, build_cut_set, build_edges, derive_crossing_set,
    load_banc, local_mask, mechanosensory_partition, presynaptic_groups,
    select_touch_tier,
)
from ..receptors import MechanoReceptor

__all__ = [
    "LEGS", "LEG_NERVE", "HONESTY", "UNITS", "BANC_RELATIVE_PATH",
    "TierConfig", "DecodeConfig", "ReceptorConfig", "Command",
    "OwnershipError", "OwnershipLedger",
    "build_neural_tier", "NeuralTier", "MechanoReceptorBank", "CommandDecoder",
    "CommandedTripodCPG", "model_fingerprint",
]

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_INDEX = {leg: i for i, leg in enumerate(LEGS)}
#: the dataset's own nerve vocabulary, matched against ``ann_nerve``.  The mapping
#: leg -> nerve is a string match on the annotation column, not an anatomical claim
#: about this MuJoCo body.
LEG_NERVE = {
    "lf": "left_prothoracic_leg_nerve",
    "lm": "left_mesothoracic_leg_nerve",
    "lh": "left_metathoracic_leg_nerve",
    "rf": "right_prothoracic_leg_nerve",
    "rm": "right_mesothoracic_leg_nerve",
    "rh": "right_metathoracic_leg_nerve",
}
#: left = the first three legs of ``flygym.anatomy.LEGS``; the split is by ANNOTATED
#: NERVE SIDE, never by an assumption about the model's index order.
LEFT_LEGS = ("lf", "lm", "lh")
RIGHT_LEGS = ("rf", "rm", "rh")

BANC_RELATIVE_PATH = "data/flywire/banc_connectome.npz"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNITS = {
    "body_length": "mm",
    "body_time": "s",
    "body_angle": "rad",
    "body_contact_force": ("backend's own force channel (labelled N by the backend; "
                           "absolute scale UNRESOLVED, used only relatively)"),
    "neural_time": "ms (inside the tier)",
    "neural_voltage": "mV",
    "neural_conductance": "uS",
    "neural_current": "nA",
    "receptor_drive": ("normalised equivalent drive, module-labelled mV: "
                       "gain_mV * open_probability in [0, gain_mV]"),
    "receptor_to_current": "nA per unit of drive (ILLUSTRATIVE)",
    "command_speed_scale": "dimensionless multiplier of the CPG intrinsic frequency",
    "command_turn": "dimensionless left/right intrinsic-frequency asymmetry in [-1, 1]",
    "firing_rate": "Hz",
}

HONESTY = [
    "FlyGym's tripod CPG is an ENGINEERING BASELINE, not a brain model, and in the "
    "neural_modulated arm it STILL GENERATES THE GAIT. The connectome does not "
    "generate the gait in this round and no result here may be described that way.",
    "The map from descending firing rate to (intrinsic frequency scale, left/right "
    "asymmetry) is ASSUMED, not measured. Its two gains and its rate scale are "
    "pre-registered constants, not fitted values.",
    "Receptor kinetics (MechanoReceptor: two-state Boltzmann channel + first-order "
    "approach) and every receptor parameter are ILLUSTRATIVE. This project has no "
    "measured Drosophila receptor kinetics.",
    "The neural tier is a bounded SELECTION of a dataset's ANNOTATION, not a brain. "
    "All cached edges are UNSIGNED: nt_pair is -1 on every row, so synapse counts "
    "are bandwidth, never excitation. Conductance sign comes only from the pure "
    "ann_nt_verified presynaptic label, and every unknown/mixed/histamine row is "
    "excluded and counted.",
    "The contact-force channel's ABSOLUTE scale is unresolved (measured per-leg "
    "magnitudes do not balance the body weight in the model's own unit system); only "
    "its relative pattern is used, against a recorded pre-registered reference.",
    "Sensory feedback is mechanosensory ONLY (leg contact force + leg joint angle). "
    "NO VISION is implemented: no retina, no optics and no visual stimulus exists in "
    "this pipeline, so no visual claim of any kind is made.",
    "Walking performance is NOT evidence about experience, perception or "
    "consciousness, and no such claim is made anywhere in this project.",
    "Model units are MILLIMETRES (gravity -9810 mm/s^2). Distances are not metres and "
    "no value is silently rescaled.",
    "Failures and negative results are left visible in the report rather than tuned "
    "away: gains were fixed before the runs and were not changed afterwards.",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(float(value)):
        raise ValueError("%s must be finite" % name)
    return float(value)


@dataclass(frozen=True)
class TierConfig:
    """The bounded BANC subnetwork: reused selection rule, bounded twice.

    ``max_neurons`` and ``partner_min_synapses`` are passed straight into
    ``engine.neck_cut_data.select_touch_tier``, whose documented deterministic rule
    is: seeds = mechanosensory neurons (class + innervating nerve), then partners =
    every neuron receiving at least ``partner_min_synapses`` synapses from the seeds,
    ranked by (-inflow, global index) and truncated to fit the cap.

    ``growth_iterations`` applies THAT SAME RULE again with the previous tier's kept
    neurons as the seed set.  One application (the module's own documented tier) does
    NOT reach the annotated descending group over modelled edges: measured at cap
    12000, iteration 1 gives 196 descending neurons of which 0 receive a single
    modelled in-tier edge.  Two applications reach it.  The iteration is an
    explicitly documented extension of the module's rule, applied through the
    module's own function; it is still a SELECTION, and its coverage is reported.
    """

    max_neurons: int = 12000
    partner_min_synapses: float = 10.0
    growth_iterations: int = 2
    #: seed set = body-route mechanosensory neurons whose ANNOTATED nerve is a leg
    #: nerve.  Vision/antennal (head-route) mechanosensory seeds are EXCLUDED because
    #: this loop has no head sensation at all; the exclusion is counted and reported.
    use_head_route_seeds: bool = False
    #: ADD THE VISUAL PATHWAY TO THE SEED SET.  Off by default so the historical
    #: tier stays BIT-IDENTICAL.  Measured with the flag OFF: the tier is 12,000
    #: neurons of which only 11 are visual-machinery neurons, because the growth
    #: rule starts from LEG-NERVE mechanosensory seeds and the visual system barely
    #: touches that route.  The dataset holds 79,538 visual-machinery neurons
    #: (51.7% of all 153,962), so 11 was a consequence of the SELECTION RULE, not
    #: of missing data.
    #: Turning this ON changes the tier definition (neuron set, growth history and
    #: fingerprint), which invalidates comparison against every previously recorded
    #: tier-level result.  That is why it is opt-in, and why the coverage report
    #: records which mode produced a given tier.
    include_visual_route: bool = False
    #: Deterministic cap on visual seed neurons.  No RNG is used: a uniform stride
    #: over the sorted dataset indices is taken, so the same cache always yields
    #: the same seeds.  79,538 candidates exist and the module bound on max_neurons
    #: is 20,000, so seeding all of them is not possible.
    visual_route_seed_limit: int = 4000
    #: Which annotation super-classes constitute the visual pathway.
    visual_classes: tuple = ("visual_projection", "visual_centrifugal",
                             "optic_lobe_intrinsic")
    dt_ms: float = 0.5
    background_mean_nA: float = 0.0
    background_sd_nA: float = 0.004
    background_seed: int = 0
    data_path: str = BANC_RELATIVE_PATH

    def validate(self):
        if isinstance(self.max_neurons, bool) or not isinstance(self.max_neurons, int):
            raise ValueError("max_neurons must be an integer")
        if not 1 <= self.max_neurons <= 20000:
            raise ValueError("max_neurons must lie in [1, 20000] (module bound)")
        _finite(self.partner_min_synapses, "partner_min_synapses")
        if self.partner_min_synapses < 0:
            raise ValueError("partner_min_synapses must be nonnegative")
        if isinstance(self.growth_iterations, bool) or not isinstance(self.growth_iterations, int):
            raise ValueError("growth_iterations must be an integer")
        if not 1 <= self.growth_iterations <= 4:
            raise ValueError("growth_iterations must lie in [1, 4]")
        _finite(self.dt_ms, "dt_ms")
        if not 0 < self.dt_ms <= 1.0:
            raise ValueError("neural dt must lie in (0, 1] ms")
        # ConductanceNetwork requires the synaptic delay to be a whole number of
        # steps; the default delay is 2 ms, so 0.5 ms gives exactly 4 steps.  The
        # check is done here so a bad clock fails with a clear message rather than
        # inside the tier.
        steps = ConductanceParams().delay_ms / self.dt_ms
        if steps < 1 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-9):
            raise ValueError("the 2 ms synaptic delay is not a whole number of neural "
                             "steps at dt_ms=%r" % (self.dt_ms,))
        _finite(self.background_mean_nA, "background_mean_nA")
        _finite(self.background_sd_nA, "background_sd_nA")
        if self.background_sd_nA < 0:
            raise ValueError("background_sd_nA must be nonnegative")
        if type(self.background_seed) is not int or self.background_seed < 0:
            raise ValueError("background_seed must be a nonnegative integer")
        if not isinstance(self.data_path, str) or not self.data_path:
            raise ValueError("data_path must be a nonempty string")
        if not isinstance(self.include_visual_route, bool):
            raise ValueError("include_visual_route must be a bool")
        if isinstance(self.visual_route_seed_limit, bool) or \
                not isinstance(self.visual_route_seed_limit, int):
            raise ValueError("visual_route_seed_limit must be an integer")
        if not 1 <= self.visual_route_seed_limit <= 20000:
            raise ValueError("visual_route_seed_limit must lie in [1, 20000] "
                             "(module bound)")
        if not isinstance(self.visual_classes, (tuple, list)) or \
                not self.visual_classes or \
                not all(isinstance(s, str) and s.strip() for s in self.visual_classes):
            raise ValueError("visual_classes must be a non-empty sequence of "
                             "non-empty strings")
        return self


@dataclass(frozen=True)
class ReceptorConfig:
    """ILLUSTRATIVE mechanosensory transduction parameters.

    Every number here is a HYPOTHESIS.  ``receptor_gain_mV`` and
    ``receptor_tau_ms`` are the receptor model's own knobs; ``reference_force`` and
    ``reference_angle_rad`` are the saturation scales used to turn a physical
    stimulus into the model's normalised strain; ``current_nA_per_drive`` converts
    the receptor's returned drive into the tier's external current.
    """

    receptor_gain_mV: float = 10.0
    receptor_tau_ms: float = 20.0
    kT: float = 1.0
    k_gate: float = 6.0
    dG: float = 4.0
    #: measured maximum per-leg contact-force magnitude in a pre-registered 60 ms
    #: baseline walk at seed 0 was 48.55 in the backend's own force unit; 50.0 is
    #: that scale rounded up, fixed before the loop runs.  Recorded, not tuned.
    reference_force: float = 50.0
    reference_angle_rad: float = 0.5
    contact_weight: float = 1.0
    proprio_weight: float = 0.5
    current_nA_per_drive: float = 0.05

    def validate(self):
        for name in ("receptor_gain_mV", "receptor_tau_ms", "kT", "k_gate", "dG",
                     "reference_force", "reference_angle_rad", "contact_weight",
                     "proprio_weight", "current_nA_per_drive"):
            _finite(getattr(self, name), name)
        if self.receptor_tau_ms <= 0:
            raise ValueError("receptor_tau_ms must be positive")
        if self.reference_force <= 0 or self.reference_angle_rad <= 0:
            raise ValueError("reference scales must be positive")
        if self.kT <= 0:
            raise ValueError("kT must be positive")
        if self.current_nA_per_drive < 0:
            raise ValueError("current_nA_per_drive must be nonnegative")
        if self.contact_weight < 0 or self.proprio_weight < 0:
            raise ValueError("channel weights must be nonnegative")
        return self

    def provenance(self):
        return {k: {"value": v, "status": "illustrative",
                    "scope": "uncalibrated; not a Drosophila receptor measurement"}
                for k, v in asdict(self).items()}


@dataclass(frozen=True)
class DecodeConfig:
    """The ASSUMED map from descending firing rate to the two CPG knobs.

    speed_scale = clip(1 + k_speed * tanh(d_rate_total / rate_scale_hz), s_min, s_max)
    turn        = k_turn * tanh((d_rate_left - d_rate_right) / rate_scale_hz)

    where d_rate_* is the FILTERED readout rate minus the pre-registered reference
    rate measured by the calibration procedure (see ``CommandDecoder.calibrate``).

    The form and both gains are ASSUMED, not measured; they were fixed before the
    loop was run, and the causality test reports the behavioural numbers they
    produce rather than re-tuning them.
    """

    k_speed: float = 0.5
    k_turn: float = 0.4
    rate_scale_hz: float = 20.0
    s_min: float = 0.5
    s_max: float = 1.5
    tau_command_ms: float = 50.0

    def validate(self):
        for name in ("k_speed", "k_turn", "rate_scale_hz", "s_min", "s_max",
                     "tau_command_ms"):
            _finite(getattr(self, name), name)
        if self.rate_scale_hz <= 0 or self.tau_command_ms <= 0:
            raise ValueError("rate scale and command time constant must be positive")
        if not 0 < self.s_min <= 1.0 <= self.s_max:
            raise ValueError("s_min <= 1 <= s_max is required")
        if self.k_speed < 0 or self.k_turn < 0:
            raise ValueError("decode gains must be nonnegative")
        return self


# ---------------------------------------------------------------------------
# ownership
# ---------------------------------------------------------------------------
class OwnershipError(RuntimeError):
    """Raised when a tier writes a resource it does not own."""


class OwnershipLedger:
    """Audit log of who wrote which resource, and how often.

    The point is to make "the neural tier never writes joints" a CHECKED property
    instead of a comment.  The allowed writer for every resource is declared up
    front; any other writer raises, and the per-resource counts are recorded so a
    test can assert exactly one joint-target writer per outer step.
    """

    ALLOWED = {
        "cpg_intrinsic_freqs": ("command_adapter",),
        # The third command dimension, and it is registered here for the same reason as the
        # other two: "only the command adapter writes the CPG's posture" has to be a CHECKED
        # property, not a comment.  MEASURED: writing an unregistered resource raises
        # OwnershipError, which is exactly what happened the first time this was added.
        "cpg_posture_bias": ("command_adapter",),
        "body_step": ("scheduler",),
        "neural_state": ("neural",),
        "receptor_state": ("receptor_bank",),
    }

    def __init__(self):
        self.counts = {k: {} for k in self.ALLOWED}
        self.writes = 0

    def write(self, resource, writer):
        if resource not in self.ALLOWED:
            raise OwnershipError("unknown resource %r" % (resource,))
        if writer not in self.ALLOWED[resource]:
            raise OwnershipError(
                "writer %r is not allowed to write %r (allowed: %r)"
                % (writer, resource, self.ALLOWED[resource]))
        self.counts[resource][writer] = self.counts[resource].get(writer, 0) + 1
        self.writes += 1
        return self

    def total(self, resource):
        return int(sum(self.counts.get(resource, {}).values()))

    def writers(self, resource):
        return tuple(sorted(self.counts.get(resource, {}).keys()))

    def as_dict(self):
        return {"allowed": {k: list(v) for k, v in self.ALLOWED.items()},
                "counts": {k: dict(v) for k, v in self.counts.items()},
                "total_writes": int(self.writes)}


# ---------------------------------------------------------------------------
# the bounded BANC subnetwork: reuse of the module's own selection rule
# ---------------------------------------------------------------------------
def _digest(*arrays, extra=""):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    h.update(str(extra).encode())
    return h.hexdigest()[:32]


def build_neural_tier(config: TierConfig | None = None, data_path: str | None = None):
    """Select the bounded tier and build its signed conductance edges.

    Returns a dict with the tier, the edges, every named group projected into the
    tier with the module's CORRECT mask semantics (``in_tier_only``: out-of-tier ids
    are excluded, never wrapped onto the last row), the per-leg sensory groups, the
    descending readout group, and an explicit coverage report.
    """
    c = (config or TierConfig()).validate()
    path = data_path or os.path.join(REPO_ROOT, c.data_path)
    if not os.path.isfile(path):
        raise FileNotFoundError("BANC cache not found: %s" % path)
    a = load_banc(path)
    n = int(a["root_ids"].size)
    mech = mechanosensory_partition(a)
    nerve = np.asarray(a["ann_nerve"])
    super_class = np.asarray(a["ann_super_class"])

    is_leg_nerve = np.array([("leg_nerve" in str(s)) for s in nerve])
    seeds_body_all = mech["body_route"]
    seeds = seeds_body_all & is_leg_nerve
    head_seeds = mech["head_route"] if c.use_head_route_seeds else np.zeros(n, bool)
    # OPTIONAL: seed the visual pathway as well.  Off by default so the historical
    # tier is unchanged.  See TierConfig.include_visual_route for the measured
    # reason this exists (11 visual neurons of 12,000 under the leg-only rule).
    vis_all = np.isin(super_class, list(c.visual_classes))
    vis_seeds = np.zeros(n, bool)
    if c.include_visual_route:
        idx = np.flatnonzero(vis_all)
        if idx.size > c.visual_route_seed_limit:
            # DETERMINISTIC uniform stride -- no RNG, so the same cache always
            # produces the same seed set.
            sel = np.linspace(0, idx.size - 1,
                              c.visual_route_seed_limit).round().astype(np.int64)
            idx = idx[np.unique(sel)]
        vis_seeds[idx] = True
    if int(seeds.sum()) == 0:
        raise ValueError("no leg-nerve body-route mechanosensory seed neurons found")
    if int(seeds.sum()) + int(head_seeds.sum()) + int(vis_seeds.sum()) > c.max_neurons:
        raise ValueError(
            "the seed set alone exceeds max_neurons: leg %d + head %d + visual %d > "
            "%d.  Raise TierConfig.max_neurons (module bound 20000) or lower "
            "TierConfig.visual_route_seed_limit."
            % (int(seeds.sum()), int(head_seeds.sum()), int(vis_seeds.sum()),
               c.max_neurons))

    crossing = derive_crossing_set(a)
    cut = build_cut_set(a, crossing)
    groups_global = presynaptic_groups(a, mech, cut)
    tcfg = TouchTierConfig(max_neurons=c.max_neurons,
                           partner_min_synapses=c.partner_min_synapses)
    seed_mask = seeds | head_seeds | vis_seeds
    history = []
    tier = None
    for it in range(c.growth_iterations):
        mech_in = {"mechanosensory": mech["mechanosensory"],
                   "body_route": seed_mask,
                   "head_route": head_seeds}
        tier = select_touch_tier(a, mech_in, cut, tcfg)
        history.append({"iteration": it + 1, "seed_neurons": int(seed_mask.sum()),
                        "tier_neurons": int(tier["n"]),
                        "cap_binding": bool(tier["counts"]["partners_truncated_by_cap"])})
        seed_mask = np.asarray(tier["keep_mask"])
    edges = build_edges(a, tier, cut, "annotation", "intact", tcfg)

    local_of_global = np.asarray(tier["local_of_global"])
    nn = int(tier["n"])
    gi = np.asarray(tier["global_index"])

    def _local(global_mask):
        return local_mask(tier, global_mask)

    boundary = {}
    for name, mask in groups_global["route_overlapping"].items():
        m = _local(mask)
        boundary[name] = {
            "local_indices": np.flatnonzero(m).astype(np.int64),
            "in_tier": int(m.sum()),
            "in_dataset": int(np.asarray(mask).sum()),
            "fraction_of_dataset_group_in_tier": (float(m.sum() / np.asarray(mask).sum())
                                                  if np.asarray(mask).sum() else None),
        }
    # per-leg sensory groups: mechanosensory neurons whose ANNOTATED nerve is this
    # leg's nerve.  All six are expected to be nonempty; a missing one is an error,
    # not a silent zero.
    sensory = {}
    for leg in LEGS:
        want = LEG_NERVE[leg]
        m = np.array([str(s) == want for s in nerve]) & mech["mechanosensory"]
        loc = np.flatnonzero(_local(m))
        if loc.size == 0:
            raise ValueError("no mechanosensory neurons for leg %s (nerve %s) inside "
                             "the tier" % (leg, want))
        sensory[leg] = loc.astype(np.int64)

    # descending readout: annotated descending neurons inside the tier that RECEIVE at
    # least one MODELLED edge from another in-tier neuron.  Measured at this
    # configuration: 196 descending neurons are inside the tier; under the correct
    # reading 195 of them receive at least one modelled in-tier edge and only 1 does
    # not, so readout_local is 195.
    #
    # DEFECT FIXED HERE.  This used to compute
    #     W_any[:, desc_local].getnnz(axis=0)
    # which is the COLUMN count.  The edge convention in this project is
    #     csr_matrix((w, (loc[post], loc[pre])))     (neck_cut_data.build_edges)
    #     self.ge += self.We @ arrivals              (neural_cond)
    # i.e. ROW = postsynaptic TARGET (incoming) and COLUMN = presynaptic SOURCE
    # (outgoing).  So the old expression measured OUT-degree while the variable name,
    # this comment and the coverage key all claimed IN-degree -- and the two sets
    # differ by 176 of 196 neurons (19 selected by out-degree, 195 by in-degree).
    # The FUNCTIONALLY correct quantity is the in-degree, because the stated
    # exclusion rationale is "cannot be DRIVEN under this sign rule": a descending
    # neuron with no modelled incoming edge has no tier input at all and cannot act
    # as a readout.  The convention is pinned by a self-test check so it cannot
    # silently flip again.
    desc_local = boundary["crossing_descending"]["local_indices"]
    W_any = (edges["We"] + edges["Wi"]).tocsr()
    # rows of desc_local -> incoming (target) degree; see the convention note above
    in_degree = np.asarray(W_any[desc_local, :].getnnz(axis=1)).ravel() if desc_local.size \
        else np.zeros(0, dtype=int)
    keep_readout = in_degree >= 1
    readout_local = desc_local[keep_readout]

    # left/right bias of each readout neuron, from the DATASET's own unsigned synapse
    # counts onto left-nerve vs right-nerve MOTOR neurons.  This is a proxy: the
    # descending annotation carries no side (its ann_nerve is empty for 1314 of 1327
    # cells), so the side of a descending cell is inferred from where its output
    # goes.  Ties are kept but reported; they contribute to neither side.
    pre, post, syn = a["pre"], a["post"], a["syn"]
    is_motor = super_class == "motor"
    is_left_nerve = np.array([("left" in str(s)) for s in nerve])
    is_right_nerve = np.array([("right" in str(s)) for s in nerve])
    left_motor = np.flatnonzero(is_motor & is_left_nerve & ~is_right_nerve)
    right_motor = np.flatnonzero(is_motor & is_right_nerve & ~is_left_nerve)
    readout_global = gi[readout_local]
    to_left = np.isin(post, left_motor)
    to_right = np.isin(post, right_motor)
    syn_left = np.bincount(pre[to_left], weights=syn[to_left].astype(float),
                           minlength=n)[readout_global]
    syn_right = np.bincount(pre[to_right], weights=syn[to_right].astype(float),
                            minlength=n)[readout_global]
    bias = syn_left - syn_right
    left_readout = readout_local[bias > 0]
    right_readout = readout_local[bias < 0]
    tie_readout = readout_local[bias == 0]
    if left_readout.size == 0 or right_readout.size == 0:
        raise ValueError("the descending readout has no cells on one side; the "
                         "left/right channel would be undefined (left=%d right=%d)"
                         % (left_readout.size, right_readout.size))

    coverage = {
        "selection_rule": tier["selection_rule"],
        "growth_iterations": c.growth_iterations,
        "growth_history": history,
        "is_a_selection_not_a_brain": True,
        "dataset": {"path": str(path), "neurons_in_dataset": n,
                    "annotated_rows_in_dataset": int(pre.size),
                    "source": ("cached BANC snapshot: a dataset ANNOTATION, "
                               "conditional on this snapshot")},
        "tier": {"neurons": nn, "max_neurons_config": int(c.max_neurons),
                 "cap_binding_at_final_iteration": bool(history[-1]["cap_binding"]),
                 "annotated_rows_inside_tier": int(np.asarray(tier["inside_rows"]).sum()),
                 "share_of_dataset_rows_inside_tier":
                     float(np.asarray(tier["inside_rows"]).sum() / pre.size),
                 "tier_root_id_digest_sha256_32": tier["digest"]["selected_root_ids_sha256_32"],
                 "index_map_is_a_bijection": True},
        "modality_boundaries": {
            name: {"in_tier": v["in_tier"], "in_dataset": v["in_dataset"],
                   "fraction": v["fraction_of_dataset_group_in_tier"]}
            for name, v in boundary.items()},
        "sensory_groups_per_leg": {
            leg: {"nerve": LEG_NERVE[leg], "in_tier": int(sensory[leg].size)}
            for leg in LEGS},
        "descending_readout": {
            "descending_in_tier": int(desc_local.size),
            "descending_in_dataset": boundary["crossing_descending"]["in_dataset"],
            "in_tier_with_at_least_one_modelled_incoming_edge": int(readout_local.size),
            "in_tier_with_zero_modelled_incoming_edges": int(desc_local.size
                                                             - readout_local.size),
            "readout_left_biased": int(left_readout.size),
            "readout_right_biased": int(right_readout.size),
            "readout_no_bias_tie": int(tie_readout.size),
            "left_right_bias_proxy": ("sign of (synapses from the cell onto left-nerve "
                                      "motor neurons) minus (onto right-nerve motor "
                                      "neurons), from the dataset's UNSIGNED counts; the "
                                      "descending annotation itself carries no side"),
        },
        "edges": {
            "exc_nnz": int(edges["nnz_exc"]), "inh_nnz": int(edges["nnz_inh"]),
            "exc_uS_total": float(edges["sum_exc_uS"]),
            "inh_uS_total": float(edges["sum_inh_uS"]),
            "excluded_edge_accounting": edges["excluded"],
            "unsigned_warning": ("nt_pair is -1 on every cached row, so every edge "
                                 "count is a synapse count (bandwidth), never "
                                 "excitation; sign comes only from the pure "
                                 "ann_nt_verified label"),
        },
        "visual_absence": {
            "photo_receptor_used": False,
            "visual_projection_neurons_in_tier": int(np.isin(
                super_class[gi],
                list(c.visual_classes)).sum()),
            "visual_machinery_neurons_in_dataset": int(np.isin(
                super_class,
                list(c.visual_classes)).sum()),
            "note": ("NO PHOTOTRANSDUCTION IS GENERATED BY THIS MODULE. No retina, no "
                     "optics and no light level are produced anywhere in this tier "
                     "builder, so engine.receptors.PhotoReceptor stays UNUSED here "
                     "rather than being fed a fabricated light level. Only leg contact "
                     "force and leg joint angle are transduced INTO the tier by this "
                     "module; a visual drive, if any, is supplied by the caller."
                     + ("  NOTE: include_visual_route=True, so the visual pathway IS "
                        "part of the seed set and the tier therefore contains far more "
                        "visual neurons than the leg-only rule produced. That makes the "
                        "visual pathway CONNECTABLE; it still does not make the tier a "
                        "visual system, and no photoreceptor is modelled."
                        if c.include_visual_route else
                        "  NOTE: include_visual_route=False (the historical rule). The "
                        "measured consequence is that only a handful of visual neurons "
                        "survives into the tier even though the dataset is 51.7% visual "
                        "machinery -- a property of the SELECTION RULE, not of the data.")),
        },
        "visual_route": {
            "include_visual_route": bool(c.include_visual_route),
            "visual_classes": [str(s) for s in c.visual_classes],
            "visual_route_seed_limit": int(c.visual_route_seed_limit),
            "visual_candidates_in_dataset": int(vis_all.sum()),
            "visual_seeds_used": int(vis_seeds.sum()),
            "visual_projection_neurons_in_tier": int(np.isin(super_class[gi],
                                                             list(c.visual_classes)).sum()),
            "seed_rule": ("off: legs only (historical).  on: legs plus a DETERMINISTIC "
                          "uniform stride over the sorted visual-machinery indices; no "
                          "RNG, so the seed set is reproducible from the cache alone."),
            "comparability_warning": ("turning include_visual_route on changes the tier "
                                      "definition (neuron set, growth history, "
                                      "fingerprint), so results from the two modes are "
                                      "NOT directly comparable; every recorded tier-level "
                                      "result predating this switch was produced with it "
                                      "OFF."),
        },
        "nt_pair_all_minus_one": bool(np.all(np.asarray(a["nt_pair"]) == -1)),
    }

    return {
        "tier": tier, "edges": edges, "data": a, "cut": cut,
        "groups_global": groups_global, "n": nn,
        "global_index": gi, "local_of_global": local_of_global,
        "boundary": boundary, "sensory": sensory,
        "descending_local": desc_local, "readout_local": readout_local,
        "readout_left": left_readout, "readout_right": right_readout,
        "readout_tie": tie_readout, "readout_bias": bias,
        "coverage": coverage,
        "fingerprint": _digest(gi, edges["We"].indptr, edges["We"].indices,
                               edges["We"].data, edges["Wi"].indptr, edges["Wi"].indices,
                               edges["Wi"].data,
                               extra=json.dumps(coverage["tier"], sort_keys=True)),
    }


class NeuralTier:
    """The conductance point tier, advanced only by its owner (the scheduler)."""

    name = "neural"

    def __init__(self, built, config: TierConfig, ledger: OwnershipLedger):
        self.built = built
        self.cfg = config.validate()
        self.ledger = ledger
        self.n = int(built["n"])
        self.net = ConductanceNetwork(self.n, ConductanceParams(),
                                      self.cfg.dt_ms).set_connectivity(
            built["edges"]["We"], built["edges"]["Wi"])
        self.rng = np.random.default_rng(self.cfg.background_seed)
        self.background_nA = self.rng.normal(self.cfg.background_mean_nA,
                                             self.cfg.background_sd_nA, self.n)
        self.readout_local = np.asarray(built["readout_local"])
        self.left = np.asarray(built["readout_left"])
        self.right = np.asarray(built["readout_right"])
        self.sensory = {k: np.asarray(v) for k, v in built["sensory"].items()}
        self.steps = 0

    def reset(self):
        """Return to the exact rest state, with the same background draw."""
        self.net = ConductanceNetwork(self.n, ConductanceParams(),
                                      self.cfg.dt_ms).set_connectivity(
            self.built["edges"]["We"], self.built["edges"]["Wi"])
        self.rng = np.random.default_rng(self.cfg.background_seed)
        self.background_nA = self.rng.normal(self.cfg.background_mean_nA,
                                             self.cfg.background_sd_nA, self.n)
        self.steps = 0
        return self

    def advance(self, n_substeps, sensory_current_nA):
        """Advance the tier by ``n_substeps``; return the interval's spike matrix.

        ``sensory_current_nA`` is the EXTERNAL current for this interval, computed
        from the body state read at the START of the interval and held constant
        across the interval's substeps.  Spikes are returned at interval END.
        """
        n_sub = int(n_substeps)
        if n_sub < 1:
            raise ValueError("n_substeps must be >= 1")
        cur = np.asarray(sensory_current_nA, float)
        if cur.shape != (self.n,):
            raise ValueError("sensory current must have one entry per tier neuron")
        if not np.all(np.isfinite(cur)):
            raise ValueError("sensory current must be finite")
        spikes = np.zeros((n_sub, self.n), dtype=bool)
        drive = self.background_nA + cur
        for k in range(n_sub):
            spikes[k] = self.net.step(drive)
        self.steps += n_sub
        self.ledger.write("neural_state", "neural")
        return spikes

    def group_rates_hz(self, spikes):
        """Per-interval rates (Hz) of the readout groups, in spikes/neuron/second."""
        sp = np.asarray(spikes)
        n_sub, dt_ms = sp.shape[0], self.cfg.dt_ms
        window_s = n_sub * dt_ms / 1000.0
        total = sp[:, self.readout_local].sum()
        left = sp[:, self.left].sum()
        right = sp[:, self.right].sum()
        return {
            "total_hz": float(total) / (self.readout_local.size * window_s),
            "left_hz": float(left) / (self.left.size * window_s),
            "right_hz": float(right) / (self.right.size * window_s),
        }


# ---------------------------------------------------------------------------
# sensation: body state -> receptor -> current
# ---------------------------------------------------------------------------
class MechanoReceptorBank:
    """Leg contact force + leg joint angle -> MechanoReceptor -> external current.

    Twelve receptor instances (one contact and one proprioceptive per leg), each
    driven by a normalised strain, each returning an equivalent drive that is
    converted to the tier's external current.  Every index mapping is injected, not
    guessed: the receptor of leg X writes only to the tier neurons whose ANNOTATED
    nerve is leg X's nerve.
    """

    name = "receptor_bank"

    def __init__(self, sensory_groups, leg_angle_rows, n_neurons, config: ReceptorConfig,
                 dt_ms, ledger: OwnershipLedger, seed=0):
        self.cfg = config.validate()
        self.dt_ms = float(dt_ms)
        if self.dt_ms <= 0:
            raise ValueError("receptor timestep must be positive")
        self.n = int(n_neurons)
        self.ledger = ledger
        self.groups = {leg: np.asarray(sensory_groups[leg], dtype=np.int64)
                       for leg in LEGS}
        for leg, idx in self.groups.items():
            if idx.ndim != 1 or idx.size == 0 or idx.min() < 0 or idx.max() >= self.n:
                raise ValueError("sensory group for leg %s is not a valid index set" % leg)
        self.rows = {leg: np.asarray(leg_angle_rows[leg], dtype=np.int64)
                     for leg in LEGS}
        for leg, rows in self.rows.items():
            # these index the JOINT-ANGLE vector (66 hinge angles), not the 42 DOFs
            if rows.ndim != 1 or rows.size == 0 or rows.min() < 0 or rows.max() >= 66:
                raise ValueError("joint-angle rows for leg %s are not valid indices into "
                                 "the joint-angle vector" % leg)
        self.contact = {}
        self.proprio = {}
        for i, leg in enumerate(LEGS):
            kw = dict(kT=self.cfg.kT, k_gate=self.cfg.k_gate, dG=self.cfg.dG,
                      tau_ms=self.cfg.receptor_tau_ms)
            self.contact[leg] = MechanoReceptor(1, gain_mV=self.cfg.receptor_gain_mV,
                                                dt_ms=self.dt_ms, seed=seed + i, **kw)
            self.proprio[leg] = MechanoReceptor(1, gain_mV=self.cfg.receptor_gain_mV,
                                                dt_ms=self.dt_ms, seed=100 + seed + i, **kw)
        self.reference_angles = None

    # -- reference pose -----------------------------------------------------
    def set_reference_pose(self, joint_angles_rad):
        q = np.asarray(joint_angles_rad, float)
        if q.ndim != 1 or q.size < 66:
            raise ValueError("the reference pose must be the full joint-angle vector "
                             "returned by the body (>= 66 hinge angles)")
        self.reference_angles = q.copy()
        return self

    def reset(self):
        for i, leg in enumerate(LEGS):
            self.contact[leg].state = np.zeros(1)
            self.proprio[leg].state = np.zeros(1)
        return self

    # -- transduction -------------------------------------------------------
    def stimuli(self, obs):
        """Normalised strains per leg (contact, proprioceptive) from one body state."""
        if self.reference_angles is None:
            raise RuntimeError("set_reference_pose() must be called before transduction")
        forces = np.asarray(obs.contact_forces, float)
        if forces.shape != (6, 3):
            raise ValueError("contact forces must be (6,3)")
        angles = np.asarray(obs.joint_angles_rad, float)
        if angles.ndim != 1 or angles.size < 66:
            raise ValueError("expected the full joint-angle vector (>= 66 hinge angles), "
                             "got shape %r" % (angles.shape,))
        force_strain = np.zeros(6)
        angle_strain = np.zeros(6)
        for leg in LEGS:
            i = LEG_INDEX[leg]
            mag = float(np.linalg.norm(forces[i]))
            force_strain[i] = min(max(mag / self.cfg.reference_force, 0.0), 1.0)
            rows = self.rows[leg]
            dev = np.abs(angles[rows] - self.reference_angles[rows]).mean()
            angle_strain[i] = min(max(dev / self.cfg.reference_angle_rad, 0.0), 1.0)
        return force_strain, angle_strain

    def transduce(self, obs):
        """One interval of transduction.  Returns (current_nA, drive dict)."""
        force_strain, angle_strain = self.stimuli(obs)
        drive_contact = np.zeros(6)
        drive_proprio = np.zeros(6)
        for leg in LEGS:
            i = LEG_INDEX[leg]
            drive_contact[i] = float(self.contact[leg].step(np.array([force_strain[i]]),
                                                            dt_ms=self.dt_ms)[0])
            drive_proprio[i] = float(self.proprio[leg].step(np.array([angle_strain[i]]),
                                                            dt_ms=self.dt_ms)[0])
        per_leg_nA = self.cfg.current_nA_per_drive * (
            self.cfg.contact_weight * drive_contact
            + self.cfg.proprio_weight * drive_proprio)
        current = np.zeros(self.n)
        for leg in LEGS:
            current[self.groups[leg]] += per_leg_nA[LEG_INDEX[leg]]
        if not np.all(np.isfinite(current)):
            raise ValueError("transduction produced non-finite current")
        self.ledger.write("receptor_state", "receptor_bank")
        return current, {
            "force_strain": force_strain,
            "angle_strain": angle_strain,
            "drive_contact": drive_contact,
            "drive_proprio": drive_proprio,
            "current_nA_per_leg": per_leg_nA,
        }


# ---------------------------------------------------------------------------
# decoding: descending spikes -> the two CPG knobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Command:
    """One decoded command.  ``source_interval`` makes the causal order auditable."""

    source_interval: int
    speed_scale: float
    turn: float
    rate_total_hz: float
    rate_left_hz: float
    rate_right_hz: float
    delta_total_hz: float
    delta_left_hz: float
    delta_right_hz: float
    forced: bool = False

    def clip_report(self, cfg: DecodeConfig):
        return {"speed_scale_clipped": bool(self.speed_scale <= cfg.s_min
                                            or self.speed_scale >= cfg.s_max)}


NEUTRAL_RATE = {"total_hz": 0.0, "left_hz": 0.0, "right_hz": 0.0}


class CommandDecoder:
    """Spike counts -> (speed scale, turn), with a pre-registered reference.

    The reference rates are measured by ``calibrate`` using a FIXED procedure: run
    the tier for ``calibration_ms`` with the body at its spawn state, hold the
    sensation constant, and take the mean rates over the whole window.  The filter
    is then INITIALISED at the reference, so forcing the rate to its reference value
    leaves the filter exactly at the reference and the command exactly neutral.
    """

    def __init__(self, cfg: DecodeConfig, dt_ms_command, dt_ms_neural, ledger,
                 tier: NeuralTier | None = None):
        self.cfg = cfg.validate()
        # TWO clocks, kept apart on purpose: the interval length governs the rate
        # window, and the COMMAND interval governs the filter.  Conflating them (an
        # earlier version of this file did) divides every rate by the ratio of the
        # two clocks, which is a silent factor-of-ten error in the command.
        self.dt_ms = float(dt_ms_neural)
        self.dt_ms_command = float(dt_ms_command)
        for name, v in (("dt_ms_neural", self.dt_ms),
                        ("dt_ms_command", self.dt_ms_command)):
            if not math.isfinite(v) or v <= 0:
                raise ValueError("%s must be positive and finite" % name)
        self.alpha = self.dt_ms_command / self.cfg.tau_command_ms
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("the command filter would be unstable at this timestep")
        self.ledger = ledger
        self.rate0 = None
        self.filt = None
        self.forced = None
        self.calibration = None
        self._read = self._left = self._right = None
        if tier is not None:
            self.bind_readout(tier)

    # -- calibration --------------------------------------------------------
    def calibrate(self, tier: NeuralTier, calibration_spikes, n_intervals,
                  settled_from_step=0):
        """Fix the reference rates from the SETTLED part of a fixed calibration run.

        The calibration window must be long enough that the tier's own startup
        transient is excluded: a conductance network released from rest fires a
        synchronised burst, and averaging over that burst inflates the reference and
        silently rescales every command.  The module's own scenario design uses the
        same idea (a phase window plus a settled measurement window), so the
        reference here is the rate over the SETTLED window only, and the full-window
        rate is recorded next to it so the difference is visible.
        """
        n = int(n_intervals)
        if n < 1:
            raise ValueError("calibration needs at least one interval")
        sp = np.asarray(calibration_spikes)
        if sp.ndim != 2 or sp.shape[1] != tier.n:
            raise ValueError("calibration spikes must be (n_substeps, n_neurons)")
        settle = int(settled_from_step)
        if settle < 0 or settle >= sp.shape[0]:
            raise ValueError("the settled calibration window is empty")
        settled = sp[settle:]
        rates = tier.group_rates_hz(settled)
        self.rate0 = {k: float(v) for k, v in rates.items()}
        self.filt = dict(self.rate0)
        self.calibration = {
            "procedure": ("tier advanced with the body's spawn-state sensation held "
                          "constant; the reference rate is the mean over the SETTLED "
                          "window only, excluding the tier's startup transient"),
            "n_intervals": n,
            "n_neural_steps": int(sp.shape[0]),
            "settled_from_neural_step": settle,
            "reference_rates_hz": dict(self.rate0),
            "whole_window_rates_hz": {k: float(v)
                                      for k, v in tier.group_rates_hz(sp).items()},
            "readout_neurons": int(tier.readout_local.size),
        }
        return self.calibration

    # -- decoding -----------------------------------------------------------
    def decode(self, spikes, source_interval, forced_rate=None):
        if self.rate0 is None:
            raise ValueError("calibrate() must be called before decode()")
        if isinstance(source_interval, bool) or not isinstance(source_interval, int):
            raise ValueError("source_interval must be an integer")
        forced = forced_rate is not None
        if forced:
            if set(forced_rate) != {"total_hz", "left_hz", "right_hz"}:
                raise ValueError("forced rate must give total_hz, left_hz, right_hz")
            raw = {k: float(v) for k, v in forced_rate.items()}
            for v in raw.values():
                if not math.isfinite(v) or v < 0:
                    raise ValueError("forced rates must be finite and nonnegative")
        else:
            raw = {k: float(v) for k, v in self.rate_hz(spikes).items()}
        for k in self.filt:
            self.filt[k] += self.alpha * (raw[k] - self.filt[k])
        d = {k: self.filt[k] - self.rate0[k] for k in self.filt}
        s = 1.0 + self.cfg.k_speed * math.tanh(d["total_hz"] / self.cfg.rate_scale_hz)
        s = min(max(s, self.cfg.s_min), self.cfg.s_max)
        a = self.cfg.k_turn * math.tanh(
            (d["left_hz"] - d["right_hz"]) / self.cfg.rate_scale_hz)
        return Command(source_interval=source_interval, speed_scale=float(s),
                       turn=float(a), rate_total_hz=self.filt["total_hz"],
                       rate_left_hz=self.filt["left_hz"],
                       rate_right_hz=self.filt["right_hz"],
                       delta_total_hz=d["total_hz"], delta_left_hz=d["left_hz"],
                       delta_right_hz=d["right_hz"], forced=forced)

    def rate_hz(self, spikes):
        if self._read is None:
            raise ValueError("bind_readout() must be called before decoding")
        sp = np.asarray(spikes)
        if sp.ndim != 2:
            raise ValueError("spikes must be a 2-D interval matrix")
        window_s = sp.shape[0] * self.dt_ms / 1000.0
        return {"total_hz": float(sp[:, self._read].sum()) / (self._read.size * window_s),
                "left_hz": float(sp[:, self._left].sum()) / (self._left.size * window_s),
                "right_hz": float(sp[:, self._right].sum()) / (self._right.size * window_s)}

    def bind_readout(self, tier: NeuralTier):
        self._read = np.asarray(tier.readout_local)
        self._left = np.asarray(tier.left)
        self._right = np.asarray(tier.right)
        return self

    def neutral(self):
        """The command used before any interval has completed: exactly neutral."""
        r = self.rate0 if self.rate0 is not None else NEUTRAL_RATE
        d = {k: 0.0 for k in r}
        return Command(source_interval=-1, speed_scale=1.0, turn=0.0,
                       rate_total_hz=float(r.get("total_hz", 0.0)),
                       rate_left_hz=float(r.get("left_hz", 0.0)),
                       rate_right_hz=float(r.get("right_hz", 0.0)),
                       delta_total_hz=d.get("total_hz", 0.0),
                       delta_left_hz=d.get("left_hz", 0.0),
                       delta_right_hz=d.get("right_hz", 0.0))


# ---------------------------------------------------------------------------
# the commanded CPG
# ---------------------------------------------------------------------------
class CommandedTripodCPG:
    """FlyGym's tripod CPG, with exactly two commanded knobs.

    Constructed EXACTLY like ``BodyBackend.attach_cpg_baseline`` (same factory, same
    seed, same timestep, same base frequency, same preprogrammed steps, same DOF
    order) so that a neutral command reproduces the baseline bit for bit.  The only
    difference from the baseline is that ``set_command`` writes
    ``intrinsic_freqs`` once per command interval:

        freq[left legs]  = base * speed_scale * (1 + turn)
        freq[right legs] = base * speed_scale * (1 - turn)

    With speed_scale == 1.0 and turn == 0.0 these assignments are exact float
    identities, which is what the zero-modulation test checks.
    """

    is_cpg_baseline = False
    name = "commanded_cpg"

    def __init__(self, timestep_s, base_frequency_hz, seed, dof_order, ledger):
        from flygym_demo.complex_terrain.cpg_controller import (
            CPGController, make_tripod_cpg_network)
        from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
        _finite(base_frequency_hz, "base_frequency_hz")
        if base_frequency_hz <= 0:
            raise ValueError("base_frequency_hz must be positive")
        self.ledger = ledger
        # KEPT, because the posture bias resolves its target joints BY NAME against this order
        # rather than by a hard-coded index.  MEASURED: without this the first run raised
        # AttributeError -- the CPG controller was handed dof_order and then forgot it.
        self.dof_order = dof_order
        self.net = make_tripod_cpg_network(timestep=timestep_s,
                                           intrinsic_frequency=base_frequency_hz,
                                           seed=seed)
        self.ctl = CPGController(self.net, PreprogrammedSteps(),
                                 output_dof_order=dof_order)
        self.base = np.ones(6) * float(base_frequency_hz)
        self.n_writes = 0
        self.last_command = (1.0, 0.0)
        # The factory above already sets intrinsic_freqs == base exactly, which is
        # what set_command(1.0, 0.0) would write, so NO write happens here: the
        # ownership ledger then shows exactly one command write per outer step.
        return

    def set_command(self, speed_scale, turn):
        if not (math.isfinite(speed_scale) and math.isfinite(turn)):
            raise ValueError("command values must be finite")
        if speed_scale <= 0:
            raise ValueError("speed scale must be positive")
        if abs(turn) > 1.0:
            raise ValueError("turn must lie in [-1, 1]")
        f = np.empty(6)
        f[:3] = self.base[:3] * (speed_scale * (1.0 + turn))
        f[3:] = self.base[3:] * (speed_scale * (1.0 - turn))
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            raise ValueError("commanded intrinsic frequencies must be positive")
        self.net.intrinsic_freqs = f
        self.n_writes += 1
        self.ledger.write("cpg_intrinsic_freqs", "command_adapter")
        self.last_command = (float(speed_scale), float(turn))
        return self

    #: WHICH JOINTS CARRY THE POSTURE PUSH, chosen by NAME, not by index: the femur-tibia
    #: ("knee") pitch of each leg.  Extending the knee is what straightens a leg against the
    #: ground; the SIGN of the extension is not assumed here, it is measured (see
    #: tools_posture_sign.py) because getting it backwards would push the fly INTO its own back.
    # "_tibia-pitch", NOT "-tibia-pitch": MEASURED, the names are "lf_trochanterfemur-lf_tibia-
    # pitch", so the character before "tibia" is an UNDERSCORE.  My first pattern used a hyphen
    # and matched nothing -- and the failure was silent until the code path ran.
    POSTURE_JOINT_SUBSTR = ("_tibia-pitch",)

    def posture_rows(self):
        """DOF rows whose joint name matches the posture target, resolved by NAME."""
        rows = []
        for i, nm in dof_names(self.dof_order):
            if any(sub in nm for sub in self.POSTURE_JOINT_SUBSTR):
                rows.append(i)
        if not rows:
            raise RuntimeError(
                "no DOF matched the posture joint pattern %r; the DOF names present are %r"
                % (self.POSTURE_JOINT_SUBSTR, [nm for _i, nm in dof_names(self.dof_order)]))
        self._posture_rows = rows
        return rows

    def set_posture_bias(self, bias_rad):
        """A UNIFORM knee offset, added to the CPG's own action.  THIS IS THE THIRD COMMAND.

        The baseline and every previous episode have exactly two knobs (speed, turn); a
        righting behaviour needs a dimension that says "push the legs out", and there was none.
        A bias of exactly 0.0 leaves the action untouched, so default behaviour is unchanged.
        """
        b = float(bias_rad)
        if not math.isfinite(b):
            raise ValueError("posture bias must be finite")
        self.posture_bias_rad = b
        self.ledger.write("cpg_posture_bias", "command_adapter")
        return self

    def step(self):
        act = self.ctl.step()
        b = float(getattr(self, "posture_bias_rad", 0.0))
        if b == 0.0:
            return act                      # exact identity: no array is touched
        rows = getattr(self, "_posture_rows", None) or self.posture_rows()
        ja = np.asarray(act.joint_angles, dtype=float).copy()
        if ja.ndim != 1 or rows and max(rows) >= ja.size:
            raise RuntimeError("posture rows %r do not index the action of size %d"
                               % (rows, ja.size))
        ja[rows] = ja[rows] + b
        return type(act)(joint_angles=ja, adhesion_onoff=act.adhesion_onoff)

    def state(self):
        return {"phase_rad": np.asarray(self.net.curr_phases, float).copy(),
                "magnitude": np.asarray(self.net.curr_magnitudes, float).copy(),
                "intrinsic_freqs_hz": np.asarray(self.net.intrinsic_freqs, float).copy()}


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def model_fingerprint(tier_fingerprint, tier_config, receptor_config, decode_config,
                      loop_config_dict):
    payload = json.dumps({"tier": asdict(tier_config),
                          "receptor": asdict(receptor_config),
                          "decode": asdict(decode_config),
                          "loop": loop_config_dict}, sort_keys=True, default=str)
    return _digest(np.frombuffer(payload.encode(), dtype=np.uint8),
                   extra=str(tier_fingerprint))


def dof_names(dof_order):
    """The model joint names the DOF order refers to, as "<parent>-<child>-<axis>"."""
    names = []
    for i, dof in enumerate(dof_order):
        parent = str(getattr(getattr(dof, "parent", None), "name", dof.parent))
        child = str(getattr(getattr(dof, "child", None), "name", dof.child))
        axis = str(getattr(dof.axis, "value", dof.axis)).lower()
        names.append((i, "%s-%s-%s" % (parent, child, axis)))
    return names


def joint_name_index_map(model):
    """Map a compiled model's joint NAME (without the fly prefix) to its index."""
    out = {}
    for j in range(int(model.njnt)):
        name = str(model.joint(j).name)
        key = name.split("/", 1)[1] if "/" in name else name
        out[key] = j
    return out


def leg_rows_from_dof_order(dof_order, model=None):
    """Map the 42 actuated DOFs onto the six legs, and (with a model) onto ANGLES.

    Returns ``{"dof_rows": {leg: (7,) DOF indices in the CPG's own order},
    "angle_rows": {leg: (7,) indices into the 66-vector returned by
    get_joint_angles()}``.

    MEASURED, and a correction of an earlier wrong assumption in this file: the 42
    actuated locomotion DOFs are NOT the first 42 entries of the joint-angle vector.
    The compiled model has 67 joints (joint 0 is the free root joint) and
    ``get_joint_angles`` returns the 66 hinge angles in JOINT order, which includes the
    passive tarsal joints that the locomotion DOF order does not actuate.  Model joint
    index 8 is ``lf_tarsus1-lf_tarsus2-pitch`` while DOF 7 is ``c_thorax-lm_coxa-yaw``,
    so indexing with a DOF number reads the WRONG leg's joint.  The angle map is
    therefore resolved joint by joint by NAME against the compiled model, and a DOF with
    no matching joint name is an error rather than a silent offset.
    """
    rows = {leg: [] for leg in LEGS}
    for i, dof in enumerate(dof_order):
        child = getattr(dof, "child", None)
        name = str(getattr(child, "name", child))
        hit = [leg for leg in LEGS if name.startswith(leg + "_")]
        if len(hit) != 1:
            raise ValueError("DOF %d (%s) does not belong to exactly one leg: %r"
                             % (i, name, hit))
        rows[hit[0]].append(i)
    for leg in LEGS:
        if len(rows[leg]) != 7:
            raise ValueError("leg %s has %d DOFs, expected 7" % (leg, len(rows[leg])))
    if sum(len(v) for v in rows.values()) != 42:
        raise ValueError("leg DOFs do not total 42")
    dof_rows = {leg: np.asarray(v, dtype=np.int64) for leg, v in rows.items()}
    if model is None:
        return {"dof_rows": dof_rows, "angle_rows": None,
                "note": ("no compiled model given: the angle map is unresolved, and the "
                         "DOF numbers are NOT valid joint-angle indices")}
    jmap = joint_name_index_map(model)
    angle_of_dof = np.full(len(dof_order), -1, dtype=np.int64)
    missing = []
    for i, want in dof_names(dof_order):
        if want not in jmap:
            missing.append(want)
            continue
        # joint 0 is the free root joint and carries 7 qpos entries; the hinge angles
        # returned by get_joint_angles are joints 1..n-1 in order, so angle k is joint
        # k+1.
        angle_of_dof[i] = jmap[want] - 1
    if missing or np.any(angle_of_dof < 0):
        raise ValueError("these DOFs have no matching joint in the compiled model: %r"
                         % (missing[:5],))
    angle_rows = {leg: angle_of_dof[dof_rows[leg]] for leg in LEGS}
    return {"dof_rows": dof_rows, "angle_rows": angle_rows,
            "angle_of_dof": angle_of_dof,
            "note": ("angle indices resolved by joint NAME against the compiled model; "
                     "they are not the DOF numbers")}


def sensor_matrix_row_map(sensory_groups, joint_rows):
    """The (receptor -> neurons) and (receptor -> DOFs) maps, checked for overlap.

    Two legs must never share a tier neuron or a DOF row; a violation would silently
    double-drive a neuron or read another leg's joint, so it is an error here.
    """
    seen = {}
    for leg, idx in sensory_groups.items():
        for i in np.asarray(idx).tolist():
            if i in seen:
                raise ValueError("tier neuron %d is claimed by legs %s and %s"
                                 % (i, seen[i], leg))
            seen[i] = leg
    seen_rows = {}
    for leg, rr in joint_rows.items():
        for i in np.asarray(rr).tolist():
            if i in seen_rows:
                raise ValueError("DOF row %d is claimed by legs %s and %s"
                                 % (i, seen_rows[i], leg))
            seen_rows[i] = leg
    return {"neurons_claimed": len(seen), "dof_rows_claimed": len(seen_rows)}
