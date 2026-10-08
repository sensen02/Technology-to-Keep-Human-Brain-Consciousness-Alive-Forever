"""engine.embodied.plasticity -- a SWITCHABLE plasticity layer for the neural tier.

WHAT THIS FILE IS, AND WHAT IT IS NOT
-------------------------------------
This module adds four INDEPENDENTLY SWITCHABLE mechanisms on top of the existing
conductance point tier (``engine.neural_cond.ConductanceNetwork`` via
``engine.embodied.adapters.NeuralTier``).  Everything here is OFF by default, and with
all four mechanisms off ``PlasticNeuralTier.advance`` calls the parent's own
``NeuralTier.advance`` verbatim -- not a re-implementation, the actual parent method --
so the frozen arm is bit-identical to the pre-existing un-plastic tier by construction
and that identity is also tested.

Nothing in ``engine/embodied/loop.py``, ``engine/embodied/adapters.py``,
``engine/neural_cond.py`` or ``engine/neural_hist.py`` is modified.  The layer is
*attached* to a scheduler through :func:`attach_plasticity`, which swaps the scheduler's
tier object for a subclass instance and rebinds the command decoder to it.

    1. SHORT-TERM PLASTICITY (STP, ms-s)     finite presynaptic release resource with
                                             depletion, recovery and OPTIONAL
                                             facilitation, applied per presynaptic
                                             neuron to the conductance increment its
                                             spikes deliver.
    2. HOMEOSTASIS (model ms-s)              a slow population-activity target driving a
                                             BOUNDED gain on synaptic scaling and/or
                                             intrinsic excitability.
    3. LONG-TERM WEIGHT CHANGE               a bounded three-factor rule: an eligibility
                                             trace multiplied by an observable
                                             task-feedback proxy.
    4. BOUNDED STRUCTURAL REWIRING           additions/removals inside an explicitly
                                             declared candidate set, with caps, full
                                             logging, and a hard lesion barrier.

RELEASE SEMANTICS (stated once, and kept consistent with the parent's convention)
---------------------------------------------------------------------------------
The parent tier's convention is: **spikes are returned at interval END, external events
are delivered at interval START**, and a returned spike reaches its targets
``delay_steps + 1`` substeps later through the parent's delay ring.

This layer does not change that convention and does not use the parent's
``exc_events_uS``/``inh_events_uS`` channels at all, so no spike-vs-event ambiguity is
introduced.  It only rescales the CONDUCTANCE INCREMENT that an already-scheduled
delivered spike produces.  The scaling is per PRESYNAPTIC neuron, because the release
machinery is presynaptic.

* At the instant a neuron crosses threshold (END of substep ``k``) the layer computes
  the release fraction from that neuron's resource ``x`` and utilisation ``u`` AT THAT
  INSTANT and stores it in a second delay ring that is advanced in lockstep with the
  parent's own ring index.  When the spike is delivered at the START of substep
  ``k + delay_steps + 1`` the ring supplies exactly that fraction.
* Re-reading ``x`` at arrival time instead would OVERESTIMATE release, because the
  resource recovers by ``exp((delay_steps+1) * dt / tau_rec_ms)`` in the meantime.
  With the default delay (2 ms) and ``tau_rec = 200 ms`` that is a 1.3% error; the ring
  removes it.  A test asserts the lockstep (our ring index equals the parent's).

UNITS AND WHERE THE NUMBERS COME FROM
--------------------------------------
Tier units are the parent's: time ms, voltage mV, conductance uS, current nA.  Every
parameter in this file is ILLUSTRATIVE (see ``plasticity_honesty()``).  The resource
``x`` and the release fraction are DIMENSIONLESS numbers in [0, 1] that multiply a
conductance whose absolute scale is itself UNMEASURED, so the layer multiplies an
unmeasured baseline rather than calibrating it.

NO COORDINATES, NO DISTANCES
-----------------------------
The tier is a point tier: it carries no positions, no lengths and no geometry.
``RewiringConfig`` therefore has NO distance/coordinate/proximity field at all (a test
asserts that, so a distance constraint is structurally impossible rather than merely
unused), and the rewiring rule never reads a coordinate.  A distance-dependent rule
would require real measured coordinates; none exist here and none are fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
import json
import math

import numpy as np
from scipy import sparse

from .adapters import LEGS, NeuralTier, OwnershipError

__all__ = [
    "STPConfig", "HomeostasisConfig", "LongTermConfig", "RewiringConfig",
    "PlasticityConfig", "LesionBarrier", "PlasticNeuralTier", "ABLATION_ARMS",
    "TRIPOD_A", "TRIPOD_B", "tripod_alternation_reward", "attach_plasticity",
    "plasticity_honesty", "PLASTICITY_PARAMETER_PROVENANCE",
]

#: the tripod gait groups, matching ``engine.embodied.adapters.LEGS`` order
#: ("lf","lm","lh","rf","rm","rh").  Used ONLY by the engineering feedback proxy.
TRIPOD_A = ("lf", "rm", "lh")
TRIPOD_B = ("rf", "lm", "rh")


# ---------------------------------------------------------------------------
# the engineering feedback proxy
# ---------------------------------------------------------------------------
def tripod_alternation_reward(drive_per_leg, gain=2.0):
    """ENGINEERING REWARD PROXY in [-1, 1] from the observed per-leg drives.

    THIS IS NOT A NEUROMODULATOR AND MUST NEVER BE CALLED ONE.  It is a declared,
    deterministic, dimensionless function of quantities the plastic layer can observe
    (the transduced per-leg drives it receives each interval).  It answers the
    engineering question "does the observed leg loading alternate between the two
    tripod groups?", which is the CPG loop's own gait objective:

        s = mean(drive on lf, rm, lh) - mean(drive on rf, lm, rh)      in [-1, 1]
        F = clip(gain * |s| - 1, -1, +1)

    Strong alternation (|s| = 1) gives +1; perfectly synchronised loading (|s| = 0)
    gives -1.  ``gain`` is ILLUSTRATIVE.  This project has NO dopaminergic neuron,
    receptor, release model or measurement, and no result here may be described as a
    dopamine concentration, a dopamine signal, or a reward prediction error.
    """
    d = np.asarray(drive_per_leg, float)
    if d.shape != (6,):
        raise ValueError("drive_per_leg must have one entry per leg (6)")
    if not np.all(np.isfinite(d)):
        raise ValueError("drive_per_leg must be finite")
    idx = {leg: i for i, leg in enumerate(LEGS)}
    a = float(np.mean([d[idx[l]] for l in TRIPOD_A]))
    b = float(np.mean([d[idx[l]] for l in TRIPOD_B]))
    return float(np.clip(float(gain) * abs(a - b) - 1.0, -1.0, 1.0))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _finite(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(float(v)):
        raise ValueError("%s must be finite" % name)
    return float(v)


def _positive(v, name):
    if _finite(v, name) <= 0.0:
        raise ValueError("%s must be positive" % name)
    return float(v)


def _fraction(v, name, lo=0.0, hi=1.0):
    f = _finite(v, name)
    if not (lo <= f <= hi):
        raise ValueError("%s must lie in [%g, %g]" % (name, lo, hi))
    return f


def _flag(v, name):
    if not isinstance(v, bool):
        raise ValueError("%s must be a bool (the switch is explicit, not truthy)" % name)
    return bool(v)


def _nonneg_int(v, name, maximum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("%s must be an integer" % name)
    if v < 0:
        raise ValueError("%s must be nonnegative" % name)
    if maximum is not None and v > maximum:
        raise ValueError("%s must be <= %d" % (name, maximum))
    return int(v)


@dataclass(frozen=True)
class STPConfig:
    """Finite presynaptic release resource (Tsodyks-Markram form), ILLUSTRATIVE.

    State per tier neuron, both in [0, 1]:

        x   release resource, x = 1 at rest (fully recovered)
        u   running utilisation, u = 0 at rest when facilitation is ON and u == U
            when facilitation is OFF

    Per substep, in this exact order (see ``PlasticNeuralTier._stp_step``):

        1. facilitation decay        u <- u * exp(-dt/tau_fac_ms)          [ON only]
        2. release at the spikes EMITTED at the END of this substep:
               u <- u + U*(1-u)                                          [ON only]
               release <- u * x                    (dimensionless, in [0, 1])
               x <- x - release
        3. recovery of every neuron over dt:
               x <- 1 - (1 - x) * exp(-dt/tau_rec_ms)

    With ``facilitation = False`` (default) ``u == U`` is a constant and step 2 reduces
    to ``release = U*x``, so the recovery-only case has the exact closed form

        x(t) = 1 - (1 - x0) * exp(-t / tau_rec_ms)          (t measured from the spike)

    which a test checks numerically (to a stated tolerance, because the iterative
    product ``exp(-dt/tau)**n`` and the direct ``exp(-n*dt/tau)`` differ in the last
    bits; the measured deviation is reported instead of being rounded to zero).
    """
    enabled: bool = False
    utilization_U: float = 0.4
    tau_rec_ms: float = 200.0
    tau_fac_ms: float = 50.0
    facilitation: bool = False

    def validate(self):
        _flag(self.enabled, "stp.enabled")
        _fraction(self.utilization_U, "stp.utilization_U")
        _positive(self.tau_rec_ms, "stp.tau_rec_ms")
        _positive(self.tau_fac_ms, "stp.tau_fac_ms")
        _flag(self.facilitation, "stp.facilitation")
        return self


@dataclass(frozen=True)
class HomeostasisConfig:
    """Slow activity target -> BOUNDED gain, ILLUSTRATIVE.

    ``target_rate_hz`` is a CONFIGURATION PARAMETER.  It is NOT a claim about normal
    adult-fly firing rates: this project has no measured adult-fly central firing-rate
    distribution (its own evidence ledger says the weight scale cannot be calibrated
    against one).  ``declared_target_range_hz`` is simply the range of this parameters
    that the sensitivity check in the report explores; it is a software range, not a
    biological one.

    Two bounded gains come out of ONE slow scalar ``z``:

        g_syn = gain_min + (gain_max - gain_min) * sigmoid(z)      synaptic scaling
        g_exc = the SAME value                                     intrinsic excitability

    so ``gain_min``/``gain_max`` are hard bounds: no run can ever exceed them, whatever
    the drive.  ``mode`` selects which one is applied ("synaptic", "intrinsic" or
    "both").

    Integration, per substep, in model time (ILLUSTRATIVE constants):

        r_inst <- spikes.sum() * 1000 / dt_ms / n_neurons            [Hz]
        r_filt <- r_filt + (dt/tau_rate_ms)   * (r_inst - r_filt)
        z      <- clip(z + (dt/tau_homeo_ms) * (target - r_filt)/rate_scale_hz, +/-z_limit)

    ``tau_homeo_ms`` is slow RELATIVE TO THE TIER'S OWN constants (membrane
    C/leak = 20 ms, adaptation 100 ms); the absolute millisecond scale is not measured.

    WHERE THE GAIN IS APPLIED, AND WHY THAT MATTERS
    -----------------------------------------------
    ``mode`` selects which bounded gain is applied:
      * "intrinsic": ``g_exc`` multiplies the TRANSDUCED EXTERNAL CURRENT only;
      * "synaptic":  ``g_syn`` multiplies the recurrent conductance increment each
                     delivered spike produces;
      * "both":      both.

    The DEFAULT is "intrinsic", and that default was CHOSEN BY MEASUREMENT, not by taste.
    The round's requirement is that homeostasis must not be able to hide a wrong input
    (a broken input must still change the population response).  MEASURED, and reported in
    full rather than removed: with ``mode="synaptic"`` the tier's own bistable regime is
    excited by the scaling itself, and at ZERO transduced drive the arm
    ``homeostasis_only`` settles at 46.19 Hz -- i.e. synaptic-scaling homeostasis
    MANUFACTURES a response where there is no sensation at all, which is exactly the
    failure the requirement forbids.  "intrinsic" is structurally safe instead of
    empirically lucky: the tier's fixed per-neuron background noise is never scaled by
    ``g_exc`` (``gain_applies_to_background`` is False by default), so with zero transduced
    input the tier is driven by nothing that scales and no gain can invent a response.
    ``synaptic`` and ``both`` remain available, and the violating measurement is recorded
    in the report.

    ``gain_applies_to_background`` exists so the boundary of that guarantee is measured
    rather than asserted: with it True the fixed background IS scaled and the report
    records how much activity homeostasis can fabricate from noise alone.
    """
    enabled: bool = False
    mode: str = "intrinsic"
    target_rate_hz: float = 40.0
    declared_target_range_hz: tuple = (5.0, 200.0)
    tau_homeo_ms: float = 500.0
    tau_rate_ms: float = 100.0
    rate_scale_hz: float = 20.0
    gain_min: float = 0.25
    gain_max: float = 4.0
    z_limit: float = 8.0
    gain_applies_to_background: bool = False

    MODES = ("synaptic", "intrinsic", "both")

    def validate(self):
        _flag(self.enabled, "homeostasis.enabled")
        if self.mode not in self.MODES:
            raise ValueError("homeostasis.mode must be one of %r" % (self.MODES,))
        _positive(self.target_rate_hz, "homeostasis.target_rate_hz")
        rng = tuple(self.declared_target_range_hz)
        if len(rng) != 2:
            raise ValueError("declared_target_range_hz must be two numbers")
        lo, hi = _positive(rng[0], "declared_target_range_hz[0]"), \
            _positive(rng[1], "declared_target_range_hz[1]")
        if lo >= hi:
            raise ValueError("declared_target_range_hz must be increasing")
        if not (lo <= self.target_rate_hz <= hi):
            raise ValueError("target_rate_hz must lie inside the declared range")
        for nm in ("tau_homeo_ms", "tau_rate_ms", "rate_scale_hz", "z_limit"):
            _positive(getattr(self, nm), "homeostasis." + nm)
        _positive(self.gain_min, "homeostasis.gain_min")
        _positive(self.gain_max, "homeostasis.gain_max")
        if self.gain_min > self.gain_max:
            raise ValueError("gain_min must not exceed gain_max")
        _flag(self.gain_applies_to_background, "homeostasis.gain_applies_to_background")
        return self

    def gain_of(self, z):
        """Bounded gain for a scalar integrator state z (in [-z_limit, +z_limit])."""
        zc = min(max(float(z), -self.z_limit), self.z_limit)
        s = 1.0 / (1.0 + math.exp(-zc))
        return float(self.gain_min + (self.gain_max - self.gain_min) * s)


@dataclass(frozen=True)
class LongTermConfig:
    """Bounded three-factor weight change: eligibility x observable feedback.

    Factor 1+2 (eligibility, per modelled edge (pre -> post)):

        e <- e * exp(-dt/tau_elig_ms) + phi_pre[pre] * phi_post[post] * (dt/1000)

    where phi are exponentially decaying spike traces of the two cells.  This is an
    activity/timing-dependent kernel with a SOFT temporal preference: a pre spike
    contributes for ``tau_pre_ms`` and a post spike for ``tau_post_ms``, so with
    ``tau_pre_ms > tau_post_ms`` a pre-before-post pairing gives the larger accumulated
    eligibility (tested).  It is NOT a hard +/- STDP window and no such window is
    claimed; ``tau_pre_ms``/``tau_post_ms`` are ILLUSTRATIVE.

    Factor 3 (feedback): ``F`` from :func:`tripod_alternation_reward` or an externally
    supplied value, in [-1, 1].  ENGINEERING REWARD PROXY, never a neuromodulator.

    Bounded update, and the bound is per edge relative to that edge's own baseline:

        dw   <- dw + eta * F * e * (dt/1000)
        W    <- W0 * clip(1 + dw, w_min_frac, w_max_frac)

    so no edge can leave ``[w_min_frac * W0, w_max_frac * W0]`` however long the run or
    however large the feedback.  ``eta`` is ILLUSTRATIVE.
    """
    enabled: bool = False
    eta: float = 0.01
    tau_elig_ms: float = 1000.0
    tau_pre_ms: float = 40.0
    tau_post_ms: float = 15.0
    w_min_frac: float = 0.5
    w_max_frac: float = 2.0

    def validate(self):
        _flag(self.enabled, "long_term.enabled")
        _finite(self.eta, "long_term.eta")
        if self.eta < 0:
            raise ValueError("long_term.eta must be nonnegative")
        for nm in ("tau_elig_ms", "tau_pre_ms", "tau_post_ms"):
            _positive(getattr(self, nm), "long_term." + nm)
        if self.tau_pre_ms <= self.tau_post_ms:
            raise ValueError("long_term requires tau_pre_ms > tau_post_ms, otherwise the "
                             "trace-product eligibility has no pre-before-post "
                             "preference at all")
        _finite(self.w_min_frac, "long_term.w_min_frac")
        _finite(self.w_max_frac, "long_term.w_max_frac")
        if not 0.0 <= self.w_min_frac <= 1.0 <= self.w_max_frac:
            raise ValueError("require 0 <= w_min_frac <= 1 <= w_max_frac")
        return self


@dataclass(frozen=True)
class RewiringConfig:
    """Bounded structural rewiring inside an explicitly declared candidate set.

    NOTE WHAT IS ABSENT ON PURPOSE: there is no distance, length, radius, proximity or
    coordinate field anywhere in this dataclass, and a test asserts that.  The tier has
    no coordinates, so a distance constraint is structurally impossible here rather
    than merely unused, and no distance is ever fabricated.

    The candidate set is declared at construction and logged:

      * pool = ordered pairs (pre, post) with pre != post, both inside the tier, pre in
        the declared SOURCE pool and post in the declared TARGET pool, and the pair not
        already a modelled edge.  The source/target pools are the union of the six
        per-leg mechanosensory groups and the descending readout group.
      * the candidate set is a seed-fixed uniform sample of that pool, of size
        ``candidate_set_size``, drawn ONCE and recorded verbatim in the change log's
        header, so "inside an explicitly declared candidate set" is checkable.

    Caps (all hard):

        max_added_edges      total additions over the life of the layer
        max_removed_edges    total removals over the life of the layer
        max_edges_total      ceiling on nnz(We) + nnz(Wi)
        per_edge_weight_cap_uS, min_edge_weight_uS

    Rules, evaluated every ``interval_ms`` of model time:

        ADD     when F > feedback_add_threshold:  up to ``k_per_event`` candidates that
                are NOT already modelled edges, ranked by the declared pair score
                ``phi_pre[pre] * phi_post[post]`` and gated by
                ``pair_score_add_threshold``, are added at
                min(w_new_uS, per_edge_weight_cap_uS).
        REMOVE  when F < feedback_remove_threshold:  up to ``k_per_event`` candidates
                that ARE already modelled edges AND whose conductance is at or below
                ``per_edge_weight_cap_uS`` are removed.  A rule that cannot act is not
                allowed to do nothing silently: if no candidate satisfies the cap, the
                attempt is logged as ``remove_refused_weight_above_cap`` with the number
                of candidate edges seen and the conductance of the closest one.

    The declared pool deliberately INCLUDES pairs that are already modelled edges, so
    that both the ADD rule and the REMOVE rule are reachable; a candidate list holding
    only non-edges would make removal impossible and the study would report a rule it
    never exercised.

    Every accepted change AND every refused change is logged with the model time, the
    rule, the parent state (feedback, gains, homeostatic z, pair score) and the old/new
    edge.
    """
    enabled: bool = False
    seed: int = 0
    candidate_set_size: int = 512
    source_pool: str = "sensory_plus_readout"
    target_pool: str = "sensory_plus_readout"
    interval_ms: float = 250.0
    k_per_event: int = 4
    max_added_edges: int = 64
    max_removed_edges: int = 64
    max_edges_total: int = 400000
    per_edge_weight_cap_uS: float = 0.001
    min_edge_weight_uS: float = 1e-6
    w_new_uS: float = 0.0005
    feedback_add_threshold: float = 0.25
    feedback_remove_threshold: float = -0.75
    pair_score_add_threshold: float = 0.05
    tau_score_ms: float = 1000.0

    def validate(self):
        _flag(self.enabled, "rewiring.enabled")
        _nonneg_int(self.seed, "rewiring.seed")
        _nonneg_int(self.candidate_set_size, "rewiring.candidate_set_size", 100000)
        if self.candidate_set_size < 1:
            raise ValueError("candidate_set_size must be at least 1")
        for nm in ("source_pool", "target_pool"):
            if getattr(self, nm) not in ("sensory_plus_readout", "sensory", "readout"):
                raise ValueError("rewiring.%s must be one of 'sensory_plus_readout', "
                                 "'sensory', 'readout'" % nm)
        _positive(self.interval_ms, "rewiring.interval_ms")
        for nm in ("k_per_event", "max_added_edges", "max_removed_edges",
                   "max_edges_total"):
            _nonneg_int(getattr(self, nm), "rewiring." + nm)
        if self.k_per_event < 1:
            raise ValueError("k_per_event must be at least 1")
        for nm in ("per_edge_weight_cap_uS", "min_edge_weight_uS", "w_new_uS"):
            _positive(getattr(self, nm), "rewiring." + nm)
        if self.w_new_uS > self.per_edge_weight_cap_uS:
            raise ValueError("w_new_uS must not exceed per_edge_weight_cap_uS")
        if self.min_edge_weight_uS > self.per_edge_weight_cap_uS:
            raise ValueError("min_edge_weight_uS must not exceed per_edge_weight_cap_uS")
        if not (-1.0 <= self.feedback_remove_threshold
                < self.feedback_add_threshold <= 1.0):
            raise ValueError("feedback thresholds must lie in [-1, 1] and be ordered")
        _finite(self.pair_score_add_threshold, "rewiring.pair_score_add_threshold")
        if self.pair_score_add_threshold < 0:
            raise ValueError("rewiring.pair_score_add_threshold must be nonnegative")
        _positive(self.tau_score_ms, "rewiring.tau_score_ms")
        return self


@dataclass(frozen=True)
class PlasticityConfig:
    """The four switches.  ALL OFF BY DEFAULT; the frozen arm is exactly this."""
    stp: STPConfig = field(default_factory=STPConfig)
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    long_term: LongTermConfig = field(default_factory=LongTermConfig)
    rewiring: RewiringConfig = field(default_factory=RewiringConfig)

    def validate(self):
        self.stp.validate()
        self.homeostasis.validate()
        self.long_term.validate()
        self.rewiring.validate()
        return self

    def enabled_names(self):
        return tuple(n for n in ("stp", "homeostasis", "long_term", "rewiring")
                     if getattr(self, n).enabled)

    def any_enabled(self):
        return bool(self.enabled_names())

    def as_dict(self):
        d = {k: asdict(getattr(self, k)) for k in
             ("stp", "homeostasis", "long_term", "rewiring")}
        d["enabled"] = list(self.enabled_names())
        return d


#: The PRE-REGISTERED ablation arms.  They differ ONLY in which mechanisms are enabled;
#: every other parameter is identical, and they share the same seeds and the same
#: initial tier state.
ABLATION_ARMS = {
    "frozen": PlasticityConfig(),
    "stp_only": PlasticityConfig(stp=STPConfig(enabled=True)),
    "homeostasis_only": PlasticityConfig(homeostasis=HomeostasisConfig(enabled=True)),
    "long_term_only": PlasticityConfig(long_term=LongTermConfig(enabled=True)),
    "stp_homeostasis": PlasticityConfig(stp=STPConfig(enabled=True),
                                        homeostasis=HomeostasisConfig(enabled=True)),
    "all": PlasticityConfig(stp=STPConfig(enabled=True),
                            homeostasis=HomeostasisConfig(enabled=True),
                            long_term=LongTermConfig(enabled=True),
                            rewiring=RewiringConfig(enabled=True)),
}


# ---------------------------------------------------------------------------
# the lesion barrier
# ---------------------------------------------------------------------------
class LesionBarrier:
    """Edges removed for a DECLARED reason, which no plasticity rule may restore.

    The barrier is a set of ordered pairs ``(pre, post)`` with a declared reason and
    the model time of the declared removal, plus the conductance that was removed (for
    the record only -- the edge is gone).  Two independent guarantees:

      * :meth:`apply_to` physically deletes every barred pair from a working
        connectivity pair, so the lesion is applied BEFORE any plasticity runs;
      * :meth:`forbids` is consulted by every rewiring ADD rule, so a barred pair can
        never come back.

    A refusal is not an exception and not a silent skip: it is logged as an event with
    the rule name, the reason and the time, so "the rule tried and was refused" is
    visible in the record.
    """
    def __init__(self, name="undeclared"):
        self.name = str(name)
        self.reasons = {}          # (pre, post) -> {"reason": str, "time_ms": float}
        self.removed_weight = {}   # (pre, post) -> float (exc uS; inh recorded as negative)
        self.refusals = []
        self.removal_log = []

    def __len__(self):
        return len(self.reasons)

    def add(self, pairs, reason, time_ms=0.0, weight_uS=0.0, channel="unknown"):
        if not isinstance(reason, str) or not reason:
            raise ValueError("a declared lesion needs a nonempty declared reason")
        t = _finite(time_ms, "time_ms")
        w = _finite(weight_uS, "weight_uS")
        pairs = list(pairs)
        for p in pairs:
            if len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                                  for v in p):
                raise ValueError("a barred edge is an integer pair (pre, post)")
            key = (int(p[0]), int(p[1]))
            if key not in self.reasons:
                self.reasons[key] = {"reason": str(reason), "time_ms": t}
                self.removed_weight[key] = float(w)
                self.removal_log.append({"time_ms": t, "rule": "declared_lesion",
                                         "reason": str(reason), "pre": key[0],
                                         "post": key[1], "channel": str(channel),
                                         "old_weight_uS": float(w), "new_weight_uS": 0.0})
        return self

    def forbids(self, pre, post):
        return (int(pre), int(post)) in self.reasons

    def reason_of(self, pre, post):
        rec = self.reasons.get((int(pre), int(post)))
        return None if rec is None else rec["reason"]

    def log_refusal(self, time_ms, rule, pre, post, parent_state=None):
        rec = {"time_ms": float(time_ms), "rule": str(rule), "refused_by":
               "lesion_barrier:%s" % self.name,
               "reason": self.reason_of(pre, post), "pre": int(pre), "post": int(post),
               "parent_state": dict(parent_state or {})}
        self.refusals.append(rec)
        return rec

    def apply_to(self, We, Wi):
        """Physically remove every barred pair from a (post, pre) CSR pair."""
        out = []
        n = We.shape[0]
        for W in (We, Wi):
            W = sparse.csr_matrix(W, dtype=float, copy=True)
            coo = W.tocoo()
            keep = np.ones(coo.nnz, bool)
            if self.reasons:
                keys = set(self.reasons)
                for k, (r, c) in enumerate(zip(coo.row.tolist(), coo.col.tolist())):
                    if (c, r) in keys:      # stored (post, pre) == (row, col)
                        keep[k] = False
            W2 = sparse.csr_matrix((coo.data[keep],
                                    (coo.row[keep], coo.col[keep])), shape=(n, n))
            W2.sum_duplicates()
            out.append(W2)
        return out[0], out[1]

    def as_dict(self):
        by_reason = {}
        for rec in self.reasons.values():
            by_reason[rec["reason"]] = by_reason.get(rec["reason"], 0) + 1
        return {"name": self.name, "n_barred_edges": len(self.reasons),
                "barred_edges_by_reason": by_reason,
                "n_refused_attempts": len(self.refusals),
                "refusals": self.refusals[:50],
                "refusals_truncated_at": 50,
                "removal_log_first10": self.removal_log[:10],
                "note": ("the barrier is applied BEFORE any plasticity runs and is "
                         "consulted by every rewiring ADD rule; a barred edge can be "
                         "neither present nor restored")}


# ---------------------------------------------------------------------------
# the plastic tier
# ---------------------------------------------------------------------------
class PlasticNeuralTier(NeuralTier):
    """``NeuralTier`` + four switchable plasticity mechanisms (all off by default).

    With every switch off, :meth:`advance` calls ``NeuralTier.advance`` itself, so the
    frozen arm is bit-identical to the pre-existing un-plastic tier by construction.

    ``reset()`` split, declared explicitly:
        cleared -> fast state (STP resource/utilisation, homeostatic integrator and
                   rate filter, eligibility traces and spike traces)
        kept    -> slow state (long-term weight multipliers, rewired edges, the lesion
                   barrier), because the slow state is the entire point of the layer
    """

    #: name used in the ownership ledger; the ledger's allowed writer stays "neural"
    name = "neural"

    def __init__(self, built, tier_config, ledger, plasticity=None, lesion=None):
        super().__init__(built, tier_config, ledger)
        self.plast = (plasticity or PlasticityConfig()).validate()
        self.lesion = lesion or LesionBarrier("no_lesion")
        self._We0 = sparse.csr_matrix(built["edges"]["We"], dtype=float, copy=True)
        self._Wi0 = sparse.csr_matrix(built["edges"]["Wi"], dtype=float, copy=True)
        # the lesion is applied BEFORE anything plastic runs
        self._We0, self._Wi0 = self.lesion.apply_to(self._We0, self._Wi0)
        self._nnz_at_lesion = int(self._We0.nnz + self._Wi0.nnz)
        self._build_structure()
        self._init_slow_state()
        self._init_fast_state()
        self._init_rewiring()
        self._external_feedback = None
        self.plasticity_events = []
        self.history = {k: [] for k in
                        ("t_ms", "pop_rate_hz", "readout_rate_hz", "g_syn", "g_exc",
                         "homeo_z", "stp_mean_x", "stp_mean_u", "lt_mean_frac",
                         "lt_max_abs_frac", "feedback", "n_edges")}
        self.n_rewire_added = 0
        self.n_rewire_removed = 0
        self.n_stp_clip = 0
        self.n_homeo_saturated = 0
        self._pending_refusals = 0

    # ---------------------------------------------------------------- structure
    def _build_structure(self):
        """Cache per-nnz row (post) indices, keys and the working connectivity arrays."""
        n = self.n
        self._We_col = self._We0.indices          # = presynaptic index (csr column)
        self._Wi_col = self._Wi0.indices
        self._We_row = np.repeat(np.arange(n), np.diff(self._We0.indptr))
        self._Wi_row = np.repeat(np.arange(n), np.diff(self._Wi0.indptr))
        self._We_key = self._We_row.astype(np.int64) * n + self._We_col
        self._Wi_key = self._Wi_row.astype(np.int64) * n + self._Wi_col

    @staticmethod
    def _realign(old_key, old_val, new_key, default):
        """Carry per-edge values across a structural change, keyed by (post, pre)."""
        out = np.full(new_key.size, float(default))
        if old_key.size == 0 or new_key.size == 0:
            return out
        order = np.argsort(old_key, kind="stable")
        sk = old_key[order]
        sv = old_val[order]
        pos = np.clip(np.searchsorted(sk, new_key), 0, sk.size - 1)
        hit = sk[pos] == new_key
        out[hit] = sv[pos[hit]]
        return out

    def _set_matrices(self, new_we, new_wi):
        """Install new connectivity matrices, carrying long-term state across."""
        n = self.n
        new_we = sparse.csr_matrix(new_we, dtype=float)
        new_wi = sparse.csr_matrix(new_wi, dtype=float)
        new_we.sum_duplicates()
        new_wi.sum_duplicates()
        new_we.eliminate_zeros()
        new_wi.eliminate_zeros()
        r = np.repeat(np.arange(n), np.diff(new_we.indptr))
        c = new_we.indices
        key_e = r.astype(np.int64) * n + c
        r2 = np.repeat(np.arange(n), np.diff(new_wi.indptr))
        c2 = new_wi.indices
        key_i = r2.astype(np.int64) * n + c2
        self._lt_e = self._realign(self._We_key, self._lt_e, key_e, 1.0)
        self._elig_e = self._realign(self._We_key, self._elig_e, key_e, 0.0)
        self._lt_i = self._realign(self._Wi_key, self._lt_i, key_i, 1.0)
        self._elig_i = self._realign(self._Wi_key, self._elig_i, key_i, 0.0)
        self._We0, self._Wi0 = new_we, new_wi
        self._build_structure()

    def _init_slow_state(self):
        """Long-term multipliers and eligibility, one entry per modelled edge."""
        self._lt_e = np.ones(self._We0.nnz)
        self._lt_i = np.ones(self._Wi0.nnz)
        self._elig_e = np.zeros(self._We0.nnz)
        self._elig_i = np.zeros(self._Wi0.nnz)

    def _init_fast_state(self):
        n = self.n
        U = self.plast.stp.utilization_U
        self._x = np.ones(n)
        self._u = np.zeros(n) if self.plast.stp.facilitation else np.full(n, U)
        self._phi_pre = np.zeros(n)
        self._phi_post = np.zeros(n)
        self._last_spiked = np.zeros(n, bool)
        self._homeo_z = 0.0
        self._homeo_rate = 0.0
        self._g_syn = 1.0
        self._g_exc = 1.0
        L = self.net.delay_steps + 1
        self._ring_rel = np.ones((L, n))
        self._ri = 0
        # the rewiring clock is fast scheduling state: the network's own t_ms restarts
        # at zero on reset(), so the schedule must restart with it
        self._rewire_next_ms = 0.0

    def _init_rewiring(self):
        cfg = self.plast.rewiring
        self._cand_pairs = None
        self._cand_header = None
        self._cand_pre = np.zeros(0, np.int64)
        self._cand_post = np.zeros(0, np.int64)
        self._cand_score_trace = np.zeros(0)
        self._cand_barred_first = np.zeros(0, bool)
        self._cand_existing = np.zeros(0, bool)
        self._rewire_next_ms = 0.0
        if not cfg.enabled:
            return
        sens = np.concatenate([np.asarray(v) for v in self.sensory.values()])
        read = np.asarray(self.readout_local)
        pools = {"sensory": np.unique(sens), "readout": np.unique(read),
                 "sensory_plus_readout": np.unique(np.concatenate([sens, read]))}
        src = pools[cfg.source_pool]
        dst = pools[cfg.target_pool]
        existing = set()
        for W in (self._We0, self._Wi0):
            coo = W.tocoo()
            existing.update(zip(coo.col.tolist(), coo.row.tolist()))
        src_set, dst_set = set(src.tolist()), set(dst.tolist())
        # THREE DECLARED BUCKETS, each with its own budget, so that every rule the study
        # reports is actually REACHABLE and the lesion barrier is actually EXERCISED
        # rather than given a pool that avoids it:
        #   (a) barrier-protected pairs inside the declared pools -> the ADD rule is
        #       tested against the exact edges it must never restore;
        #   (b) pairs that ARE currently modelled edges inside the pools -> the REMOVE
        #       rule is reachable at all;
        #   (c) a seed-fixed uniform sample of the remaining declared pool.
        budget = int(cfg.candidate_set_size)
        share = max(1, budget // 4)
        barred_inside = sorted(p for p in self.lesion.reasons
                               if p[0] in src_set and p[1] in dst_set)
        barred_take = barred_inside[:share]
        existing_inside = sorted(p for p in existing if p[0] in src_set and p[1] in dst_set)
        rng = np.random.default_rng(cfg.seed)
        if existing_inside:
            pick = rng.choice(len(existing_inside),
                              size=min(share, len(existing_inside)), replace=False)
            existing_take = [existing_inside[int(i)] for i in np.sort(pick)]
        else:
            existing_take = []
        pairs = list(barred_take) + list(existing_take)
        seen = set(pairs)
        attempts = 0
        max_attempts = budget * 400 + 10000
        while len(pairs) < budget and attempts < max_attempts:
            attempts += 1
            pre = int(src[rng.integers(src.size)])
            post = int(dst[rng.integers(dst.size)])
            if pre == post:
                continue
            key = (pre, post)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        if len(pairs) < budget:
            raise RuntimeError("could not draw the declared candidate set: only %d of %d "
                               "pairs found in %d attempts"
                               % (len(pairs), budget, attempts))
        self._cand_pairs = pairs
        #: ordering used by the ADD rule: barrier-protected pairs FIRST (so a barred
        #: pair is proposed and refused at the head of every rewiring event, by design),
        #: then everything else by accumulated pair score, descending.
        self._cand_barred_first = np.array([p in set(barred_take) for p in pairs], bool)
        self._cand_existing = np.array([p in existing for p in pairs], bool)
        self._cand_pre = np.array([p[0] for p in pairs], np.int64)
        self._cand_post = np.array([p[1] for p in pairs], np.int64)
        self._cand_score_trace = np.zeros(len(pairs))
        self._cand_header = {
            "declared_candidate_set_size": budget,
            "pairs_drawn": len(pairs),
            "seed": int(cfg.seed),
            "source_pool": cfg.source_pool, "target_pool": cfg.target_pool,
            "source_pool_size": int(src.size), "target_pool_size": int(dst.size),
            "declared_pool_size": int(src.size) * int(dst.size),
            "draw_attempts": int(attempts),
            "bucket_barrier_protected": len(barred_take),
            "bucket_barrier_protected_available": len(barred_inside),
            "bucket_already_modelled_edges": len(existing_take),
            "bucket_already_modelled_edges_available_in_pool": len(existing_inside),
            "bucket_uniform_draw": int(len(pairs) - len(barred_take) - len(existing_take)),
            "buckets_reason": ("three declared buckets so that every rule the study "
                               "reports is reachable: (a) barrier-protected pairs so the "
                               "ADD rule is tested against the exact edges it must never "
                               "restore, (b) pairs that are already modelled edges so "
                               "REMOVE is reachable, (c) a uniform draw for the rest"),
            "add_rule_order": ("candidate list order: barrier-protected pairs first, "
                               "then the rest by the accumulated pair score "
                               "descending; a barrier-protected pair is therefore "
                               "PROPOSED at the head of every rewiring event and "
                               "refused by the barrier, instead of relying on a lucky "
                               "draw to test the barrier at all"),
            "distance_constraint": ("NONE, and none is possible: the tier carries no "
                                    "coordinates, so no distance is computed, implied or "
                                    "fabricated"),
            "first_10_pairs": [[int(p[0]), int(p[1])] for p in pairs[:10]],
        }

    # ------------------------------------------------------------------- reset
    def reset(self):
        super().reset()
        self.net.set_connectivity(self._We0, self._Wi0)
        self._init_fast_state()
        return self

    # ---------------------------------------------------------------- helpers
    def _edge_index(self, matrix_sel, pre, post):
        """Index of edge (pre, post) inside the We/Wi value arrays, or None."""
        W = self._We0 if matrix_sel == "e" else self._Wi0
        for j in range(W.indptr[post], W.indptr[post + 1]):
            if W.indices[j] == pre:
                return int(j)
        return None

    def _per_leg_drive(self, current_nA):
        d = np.zeros(6)
        for leg in LEGS:
            g = self.sensory[leg]
            d[LEGS.index(leg)] = float(np.mean(current_nA[g])) if g.size else 0.0
        return d

    def set_feedback(self, value):
        """Supply an external task-feedback proxy; overrides the built-in one."""
        if value is None:
            self._external_feedback = None
            return self
        v = _finite(value, "feedback")
        if not (-1.0 <= v <= 1.0):
            raise ValueError("an external feedback proxy must lie in [-1, 1]")
        self._external_feedback = float(v)
        return self

    # ------------------------------------------------------------ the four rules
    def _stp_step(self, spiked, dt_ms):
        """One substep of the release-resource rule.  Returns the release fractions."""
        cfg = self.plast.stp
        rel = np.ones(self.n)
        if not cfg.enabled:
            return rel
        if cfg.facilitation:
            self._u *= math.exp(-dt_ms / cfg.tau_fac_ms)
        if spiked.any():
            idx = np.flatnonzero(spiked)
            u = self._u[idx]
            if cfg.facilitation:
                u = u + cfg.utilization_U * (1.0 - u)
                self._u[idx] = u
            r = u * self._x[idx]
            self._x[idx] = self._x[idx] - r
            rel[idx] = r
        rec = math.exp(-dt_ms / cfg.tau_rec_ms)
        self._x = 1.0 - (1.0 - self._x) * rec
        # the bounds are asserted, not enforced: the clip counter must stay at zero
        bad = int(np.count_nonzero((self._x < 0.0) | (self._x > 1.0)))
        self.n_stp_clip += bad
        self._x = np.clip(self._x, 0.0, 1.0)
        return rel

    def _homeostasis_step(self, spiked, dt_ms):
        """Slow BOUNDED-gain homeostasis.  Returns (g_syn, g_exc) for the NEXT substep."""
        cfg = self.plast.homeostasis
        if not cfg.enabled:
            self._g_syn = self._g_exc = 1.0
            return 1.0, 1.0
        r_inst = float(spiked.sum()) * 1000.0 / dt_ms / self.n
        self._homeo_rate += (dt_ms / cfg.tau_rate_ms) * (r_inst - self._homeo_rate)
        dz = (dt_ms / cfg.tau_homeo_ms) * \
            ((cfg.target_rate_hz - self._homeo_rate) / cfg.rate_scale_hz)
        z_new = float(np.clip(self._homeo_z + dz, -cfg.z_limit, cfg.z_limit))
        if abs(self._homeo_z + dz) > cfg.z_limit:
            self.n_homeo_saturated += 1
        self._homeo_z = z_new
        g = cfg.gain_of(self._homeo_z)
        self._g_syn = g if cfg.mode in ("synaptic", "both") else 1.0
        self._g_exc = g if cfg.mode in ("intrinsic", "both") else 1.0
        return self._g_syn, self._g_exc

    def _eligibility_step(self, spiked, dt_ms):
        """Spike traces (always needed) and per-edge eligibility (long-term rule).

        The spike traces are updated whenever the long-term rule OR the rewiring rule is
        on, because the rewiring rule's declared pair score is built from them.
        """
        cfg = self.plast.long_term
        rw = self.plast.rewiring
        if not (cfg.enabled or rw.enabled):
            return
        self._phi_pre *= math.exp(-dt_ms / cfg.tau_pre_ms)
        self._phi_post *= math.exp(-dt_ms / cfg.tau_post_ms)
        if spiked.any():
            self._phi_pre[spiked] += 1.0
            self._phi_post[spiked] += 1.0
        if rw.enabled:
            self._cand_score_trace *= math.exp(-dt_ms / rw.tau_score_ms)
            self._cand_score_trace += (dt_ms / 1000.0) \
                * self._phi_pre[self._cand_pre] * self._phi_post[self._cand_post]
        if not cfg.enabled:
            return
        dec = math.exp(-dt_ms / cfg.tau_elig_ms)
        s_dt = dt_ms / 1000.0
        self._elig_e *= dec
        self._elig_e += s_dt * self._phi_pre[self._We_col] * self._phi_post[self._We_row]
        self._elig_i *= dec
        self._elig_i += s_dt * self._phi_pre[self._Wi_col] * self._phi_post[self._Wi_row]

    def _long_term_step(self, F, dt_ms):
        """Bounded three-factor update.  Returns (mean_frac, max_abs_frac)."""
        cfg = self.plast.long_term
        if not cfg.enabled:
            return 0.0, 0.0
        step = cfg.eta * float(F) * (dt_ms / 1000.0)
        lo = cfg.w_min_frac - 1.0
        hi = cfg.w_max_frac - 1.0
        self._lt_e = 1.0 + np.clip((self._lt_e - 1.0) + step * self._elig_e, lo, hi)
        self._lt_i = 1.0 + np.clip((self._lt_i - 1.0) + step * self._elig_i, lo, hi)
        de = self._lt_e - 1.0
        di = self._lt_i - 1.0
        n = de.size + di.size
        if n == 0:
            return 0.0, 0.0
        mean = float(de.sum() + di.sum()) / n
        mx = max(float(np.abs(de).max()) if de.size else 0.0,
                 float(np.abs(di).max()) if di.size else 0.0)
        # the at-bound diagnostic is a REPORT, not part of the dynamics: sampling it
        # every 20 substeps keeps the hot path free of three extra 291k-element passes
        self._lt_diag_counter = getattr(self, "_lt_diag_counter", 0) + 1
        if self._lt_diag_counter >= 20:
            self._lt_diag_counter = 0
            near_e = np.minimum(np.abs(de - lo), np.abs(de - hi))
            near_i = np.minimum(np.abs(di - lo), np.abs(di - hi))
            at = int(np.count_nonzero(near_e <= 1e-12)
                     + np.count_nonzero(near_i <= 1e-12))
            self._lt_at_bound_max = max(getattr(self, "_lt_at_bound_max", 0.0), at / n)
        return mean, mx

    def _rewiring_step(self, t_ms, F):
        cfg = self.plast.rewiring
        if not cfg.enabled or t_ms + 1e-9 < self._rewire_next_ms:
            return
        self._rewire_next_ms = t_ms + cfg.interval_ms
        changed = False
        parent_state = {"feedback": float(F), "g_syn": float(self._g_syn),
                        "g_exc": float(self._g_exc), "homeo_z": float(self._homeo_z),
                        "pop_rate_hz": float(self._homeo_rate)}
        order = np.lexsort((-self._cand_score_trace, ~self._cand_barred_first))
        added = 0
        if F > cfg.feedback_add_threshold \
                and self.n_rewire_added < cfg.max_added_edges:
            for k in order:
                if added >= cfg.k_per_event:
                    break
                pre, post = self._cand_pairs[int(k)]
                if self.lesion.forbids(pre, post):
                    # checked BEFORE the score gate: a barred pair is refused by the
                    # barrier whatever its co-activity score happens to be
                    rec = self.lesion.log_refusal(
                        t_ms, "rewiring_add", pre, post,
                        dict(parent_state,
                             pair_score=float(self._cand_score_trace[int(k)])))
                    self.plasticity_events.append(
                        dict(rec, old_edge=None, new_edge=None,
                             outcome="refused_by_lesion_barrier"))
                    continue
                if self._has_edge(pre, post):
                    continue                  # already a modelled edge: nothing to add
                if self._cand_score_trace[int(k)] < cfg.pair_score_add_threshold:
                    self.plasticity_events.append(
                        {"time_ms": float(t_ms), "rule": "add_refused_low_pair_score",
                         "parent_state": dict(
                             parent_state,
                             pair_score=float(self._cand_score_trace[int(k)])),
                         "old_edge": None, "new_edge": None, "pre": int(pre),
                         "post": int(post), "outcome": "refused_low_pair_score"})
                    continue
                if self.n_edges() >= cfg.max_edges_total:
                    self.plasticity_events.append(
                        {"time_ms": float(t_ms), "rule": "add_refused_edge_cap",
                         "parent_state": dict(parent_state), "old_edge": None,
                         "new_edge": None, "pre": int(pre), "post": int(post),
                         "outcome": "refused_by_edge_cap"})
                    continue
                w = min(cfg.w_new_uS, cfg.per_edge_weight_cap_uS)
                self._add_edge(pre, post, w,
                               parent_state=dict(
                                   parent_state,
                                   pair_score=float(self._cand_score_trace[int(k)])))
                added += 1
                changed = True
        if F < cfg.feedback_remove_threshold \
                and self.n_rewire_removed < cfg.max_removed_edges:
            removed = 0
            pool = sorted({p for p in self._cand_pairs if self._has_edge(*p)},
                          key=lambda p: (self._edge_weight(*p), p))
            weakest = None
            n_above_cap = 0
            for pre, post in pool:
                if removed >= cfg.k_per_event:
                    break
                if self.lesion.forbids(pre, post):
                    continue                      # cannot even be present; never touch
                w = self._edge_weight(pre, post)
                if w > cfg.per_edge_weight_cap_uS:
                    n_above_cap += 1
                    if weakest is None or w < weakest[2]:
                        weakest = (int(pre), int(post), float(w))
                    continue
                self._remove_edge(pre, post, parent_state=parent_state)
                removed += 1
                changed = True
            if removed == 0 and pool:
                # a rule that cannot act is NOT allowed to do nothing silently: it
                # records that it ran, that no candidate satisfied the declared cap, and
                # how far away the closest candidate was
                self.plasticity_events.append({
                    "time_ms": float(t_ms), "rule": "remove_refused_weight_above_cap",
                    "parent_state": dict(parent_state, n_candidate_edges=len(pool),
                                         n_above_cap=n_above_cap),
                    "old_edge": (None if weakest is None else
                                 {"pre": weakest[0], "post": weakest[1],
                                  "weight_uS": weakest[2]}),
                    "new_edge": None, "outcome": "refused_weight_above_cap",
                    "per_edge_weight_cap_uS": float(cfg.per_edge_weight_cap_uS)})
        if changed:
            self._refresh_after_structural_change()

    def _cand_score(self, pair):
        """The DECLARED pair score for a rewiring candidate: how co-active are the two
        cells right now.  Defined for pairs that are NOT edges, which is exactly what a
        structural ADD rule needs, and it reads no coordinate because none exists.

        (The per-EDGE eligibility used by the long-term rule is a different object and
        cannot be used here: it is zero for every edge that does not exist yet.)
        """
        pre, post = pair
        return float(self._phi_pre[pre] * self._phi_post[post])

    def _has_edge(self, pre, post):
        return self._edge_weight(pre, post) > 0.0 or self._edge_index("e", pre,
                                                                      post) is not None \
            or self._edge_index("i", pre, post) is not None

    def _edge_weight(self, pre, post):
        j = self._edge_index("e", pre, post)
        if j is not None:
            return float(self._We0.data[j])
        j = self._edge_index("i", pre, post)
        if j is not None:
            return float(self._Wi0.data[j])
        return 0.0

    def n_edges(self):
        return int(self._We0.nnz + self._Wi0.nnz)

    def _add_edge(self, pre, post, w, parent_state=None):
        cfg = self.plast.rewiring
        if self.lesion.forbids(pre, post):
            raise AssertionError("the lesion barrier was bypassed: (%d,%d)" % (pre, post))
        if not (cfg.min_edge_weight_uS <= w <= cfg.per_edge_weight_cap_uS):
            raise AssertionError("new edge weight outside the declared caps")
        old_nnz = self._We0.nnz
        added = sparse.csr_matrix((np.array([w]), (np.array([post]), np.array([pre]))),
                                  shape=(self.n, self.n))
        new = sparse.csr_matrix(self._We0 + added)
        if new.nnz <= old_nnz:
            raise AssertionError("the added edge did not appear (duplicate?)")
        self._set_matrices(new, self._Wi0)
        self.n_rewire_added += 1
        self.plasticity_events.append({
            "time_ms": float(self.net.t_ms), "rule": "add_edge",
            "parent_state": dict(parent_state or {}),
            "old_edge": None,
            "new_edge": {"pre": int(pre), "post": int(post), "channel": "exc",
                         "weight_uS": float(w),
                         "distance_constraint": "NONE (the tier has no coordinates)"},
            "outcome": "accepted", "n_edges": self.n_edges(),
            "cumulative_added": self.n_rewire_added})
        return True

    def _remove_edge(self, pre, post, parent_state=None):
        for sel, W in (("e", self._We0), ("i", self._Wi0)):
            j = self._edge_index(sel, pre, post)
            if j is None:
                continue
            w = float(W.data[j])
            coo = W.tocoo()
            keep = np.ones(coo.nnz, bool)
            keep[j] = False
            new = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                                    shape=(self.n, self.n))
            if sel == "e":
                self._set_matrices(new, self._Wi0)
            else:
                self._set_matrices(self._We0, new)
            self.n_rewire_removed += 1
            self.plasticity_events.append({
                "time_ms": float(self.net.t_ms), "rule": "remove_edge",
                "parent_state": dict(parent_state or {}),
                "old_edge": {"pre": int(pre), "post": int(post), "channel": sel,
                             "weight_uS": w},
                "new_edge": None, "outcome": "accepted", "n_edges": self.n_edges(),
                "cumulative_removed": self.n_rewire_removed})
            return True
        return False

    def _refresh_after_structural_change(self):
        self.net.set_connectivity(self._We0, self._Wi0)

    # ------------------------------------------------------------------ advance
    def advance(self, n_substeps, sensory_current_nA):
        """Advance the tier; return the interval's spike matrix (spikes at END).

        With EVERY mechanism off this delegates to ``NeuralTier.advance`` -- the parent
        method object itself -- so the frozen arm is bit-identical to the un-plastic
        tier, and a test checks that identity against an independent plain tier.
        """
        if not self.plast.any_enabled():
            return NeuralTier.advance(self, n_substeps, sensory_current_nA)

        n_sub = int(n_substeps)
        if n_sub < 1:
            raise ValueError("n_substeps must be >= 1")
        cur = np.asarray(sensory_current_nA, float)
        if cur.shape != (self.n,):
            raise ValueError("sensory current must have one entry per tier neuron")
        if not np.all(np.isfinite(cur)):
            raise ValueError("sensory current must be finite")

        cfg = self.plast
        dt_ms = self.cfg.dt_ms
        F = self._external_feedback
        if F is None:
            F = tripod_alternation_reward(self._per_leg_drive(cur))
        spikes = np.zeros((n_sub, self.n), dtype=bool)
        need_matrix = (cfg.stp.enabled
                       or (cfg.homeostasis.enabled
                           and cfg.homeostasis.mode in ("synaptic", "both"))
                       or cfg.long_term.enabled
                       or cfg.rewiring.enabled)
        bg = self.background_nA
        for k in range(n_sub):
            # (a) the factors that multiply the DELIVERED spikes of this substep
            if need_matrix:
                rel_a = self._ring_rel[self._ri] if cfg.stp.enabled \
                    else np.ones(self.n)
                fac = rel_a * self._g_syn
                self.net.We = sparse.csr_matrix(
                    (self._We0.data * self._lt_e * fac[self._We_col],
                     self._We_col, self._We0.indptr), shape=self._We0.shape)
                self.net.Wi = sparse.csr_matrix(
                    (self._Wi0.data * self._lt_i * fac[self._Wi_col],
                     self._Wi_col, self._Wi0.indptr), shape=self._Wi0.shape)
            # (b) intrinsic excitability: the TRANSDUCED current only
            drive = bg + cur
            if cfg.homeostasis.enabled and self._g_exc != 1.0:
                drive = (bg + self._g_exc * cur) if not \
                    cfg.homeostasis.gain_applies_to_background \
                    else (self._g_exc * (bg + cur))
            # (c) the parent's own backward-Euler step (unchanged numerics)
            spiked = self.net.step(drive)
            spikes[k] = spiked
            self._last_spiked = spiked
            # (d) the four rules, in this order
            rel = self._stp_step(spiked, dt_ms)
            self._ring_rel[self._ri] = rel
            self._ri = (self._ri + 1) % self._ring_rel.shape[0]
            self._homeostasis_step(spiked, dt_ms)
            self._eligibility_step(spiked, dt_ms)
            mean_dw, max_dw = self._long_term_step(F, dt_ms)
            self._rewiring_step(self.net.t_ms, F)
            self._record(F, mean_dw, max_dw)
        self.steps += n_sub
        self.ledger.write("neural_state", "neural")
        return spikes

    def _record(self, F, mean_dw, max_dw):
        h = self.history
        h["t_ms"].append(float(self.net.t_ms))
        h["pop_rate_hz"].append(float(self._last_pop_rate()))
        h["readout_rate_hz"].append(float(self._last_readout_rate()))
        h["g_syn"].append(float(self._g_syn))
        h["g_exc"].append(float(self._g_exc))
        h["homeo_z"].append(float(self._homeo_z))
        h["stp_mean_x"].append(float(self._x.mean()) if self.plast.stp.enabled else 1.0)
        h["stp_mean_u"].append(float(self._u.mean()) if self.plast.stp.enabled else 0.0)
        h["lt_mean_frac"].append(float(mean_dw))
        h["lt_max_abs_frac"].append(float(max_dw))
        h["feedback"].append(float(F))
        h["n_edges"].append(int(self.n_edges()))

    def _last_pop_rate(self):
        # spikes of the substep just taken, as an instantaneous population rate (Hz)
        sp = self._last_spiked
        return float(sp.sum()) * 1000.0 / self.cfg.dt_ms / self.n

    def _last_readout_rate(self):
        sp = self._last_spiked
        return float(sp[self.readout_local].sum()) * 1000.0 / self.cfg.dt_ms \
            / max(self.readout_local.size, 1)

    # ------------------------------------------------------------------ report
    def plasticity_report(self):
        cfg = self.plast
        by_rule = {}
        for e in self.plasticity_events:
            by_rule[e.get("rule", "?")] = by_rule.get(e.get("rule", "?"), 0) + 1
        accepted = [e for e in self.plasticity_events
                    if e.get("rule") in ("add_edge", "remove_edge")]
        refused = [e for e in self.plasticity_events
                   if str(e.get("outcome", "")).startswith("refused")]
        scores = self._cand_score_trace
        return {
            "config": cfg.as_dict(),
            "enabled": list(cfg.enabled_names()),
            "n_tier_neurons": int(self.n),
            "n_edges_at_start_after_lesion": int(self._nnz_at_lesion),
            "n_edges_now": int(self.n_edges()),
            "rewiring": {
                "candidate_set_header": self._cand_header,
                "added": int(self.n_rewire_added),
                "removed": int(self.n_rewire_removed),
                "max_added_edges": int(cfg.rewiring.max_added_edges),
                "max_removed_edges": int(cfg.rewiring.max_removed_edges),
                "max_edges_total": int(cfg.rewiring.max_edges_total),
                "per_edge_weight_cap_uS": float(cfg.rewiring.per_edge_weight_cap_uS),
                "edge_cap_respected": bool(self.n_edges() <= cfg.rewiring.max_edges_total),
                "add_cap_respected": bool(self.n_rewire_added
                                          <= cfg.rewiring.max_added_edges),
                "remove_cap_respected": bool(self.n_rewire_removed
                                             <= cfg.rewiring.max_removed_edges),
                "events_by_rule": dict(by_rule),
                "n_refused_total": len(refused),
                "n_refused_by_lesion_barrier": int(
                    by_rule.get("rewiring_add", 0)),
                "accepted_changes_logged": len(accepted),
                "accepted_changes_first10": self.plasticity_events and [
                    e for e in self.plasticity_events
                    if e.get("rule") in ("add_edge", "remove_edge")][:10],
                "refusals_sample_first5": refused[:5],
                "candidate_pair_score_max": (float(scores.max()) if scores.size else 0.0),
                "candidate_pair_score_median": (float(np.median(scores))
                                                if scores.size else 0.0),
                "distance_constraint": ("NONE. The tier carries no coordinates; no "
                                        "distance is computed, implied or fabricated, "
                                        "and RewiringConfig has no distance field."),
            },
            "lesion_barrier": self.lesion.as_dict(),
            "counters": {
                "stp_bound_violations_clipped": int(self.n_stp_clip),
                "homeostasis_integrator_saturations": int(self.n_homeo_saturated),
                "long_term_fraction_of_edges_at_a_bound_max": float(
                    getattr(self, "_lt_at_bound_max", 0.0)),
            },
            "n_events_total": len(self.plasticity_events),
        }


# ---------------------------------------------------------------------------
# attachment
# ---------------------------------------------------------------------------
def attach_plasticity(scheduler, plasticity=None, lesion=None):
    """Swap a built scheduler's tier for a :class:`PlasticNeuralTier` (same tier).

    Nothing in ``loop.py`` is edited: the scheduler already exposes the built tier it
    created (``scheduler.built``), so the plastic tier is constructed from that exact
    object (same selected neurons, same edges, same fingerprints) and the command
    decoder is rebound to the new tier's readout.
    """
    if scheduler.tier is None:
        raise ValueError("this scheduler has no neural tier (the cpg_baseline arm); "
                         "plasticity is defined on the neural tier only")
    cfg = (plasticity or PlasticityConfig()).validate()
    tier = PlasticNeuralTier(scheduler.built, scheduler.cfg.tier, scheduler.ledger,
                             plasticity=cfg, lesion=lesion)
    tier._external_feedback = None
    scheduler.tier = tier
    if scheduler.decoder is not None:
        scheduler.decoder.bind_readout(tier)
    return tier


# ---------------------------------------------------------------------------
# honesty
# ---------------------------------------------------------------------------
def plasticity_honesty():
    """The honesty block for this round, restated on the figure and in the report."""
    return list(PLASTICITY_HONESTY)


PLASTICITY_HONESTY = [
    "EVERY plasticity parameter in engine/embodied/plasticity.py is ILLUSTRATIVE: the "
    "release utilisation U, tau_rec, tau_fac, the homeostatic target and its gains, the "
    "eligibility time constants, eta and every rewiring cap are uncalibrated choices, "
    "not measurements.",
    "This project has NO measured adult-fly central synaptic plasticity timescale "
    "(tau_p). No run here, and no number derived from one, can say how long a trained "
    "mapping stays valid, whether it persists, or whether it would decay. The long-term "
    "mechanism below is a bounded engineering integrator, nothing more.",
    "The connectome weights this plasticity acts on are UNMEASURED, and every cached "
    "edge is UNSIGNED: nt_pair is -1 on every row of the cached BANC snapshot and is "
    "never read for sign. Sign comes only from the pure ann_nt_verified presynaptic "
    "label. A caveat on the parent project's own description: the FAFB cache is a "
    ">=5-synapse strong-connection graph, but the BANC cache used by THIS tier is NOT "
    "(measured minimum 3.0 synapses and 50.3% of rows below 5), so the baseline is "
    "unmeasured either way. Plasticity therefore starts from an unmeasured baseline and "
    "AMPLIFIES that uncertainty rather than reducing it: it cannot make the weights "
    "mean more than they did.",
    "The feedback signal of the three-factor rule is an ENGINEERING REWARD PROXY, a "
    "declared deterministic function of observable per-leg drives. It is NOT a "
    "neuromodulator measurement. This project contains no dopaminergic neuron, no "
    "dopamine receptor, no release model and no dopamine measurement, so no result may "
    "be called a dopamine concentration, a dopamine signal, or a reward prediction "
    "error.",
    "Structural rewiring here has NO DISTANCE CONSTRAINT, because the tier has no "
    "coordinates. No distance is computed, implied or fabricated, and no spatial or "
    "wiring-length claim of any kind is made.",
    "A lesion barrier prevents declared cut edges from being restored, but it is a "
    "software invariant, not a model of injury repair, regeneration or glial scarring.",
    "Homeostasis is a bounded global scalar gain. It is not a biological homeostatic "
    "system, and its target rate is a SOFTWARE PARAMETER with no claim about normal "
    "adult-fly firing rates.",
    "No consciousness, viability, survival, health or behaviour-quality claim is made "
    "anywhere in this round. Walking speed and heading are engineering readouts of a "
    "MuJoCo millimetre model driven by FlyGym's demo tripod CPG.",
    "Failures and negative results are reported as found. Predictions were written down "
    "before the numbers and none was rescued by re-tuning a gain.",
]

PLASTICITY_PARAMETER_PROVENANCE = {
    "status": "ILLUSTRATIVE throughout; no parameter in this module is a measurement",
    "measured_absent": ["adult-fly central synaptic plasticity timescale tau_p",
                        "adult-fly firing-rate distribution for any cell class",
                        "release probability / resource recovery at any fly central "
                        "synapse",
                        "sign and weight of any cached connectome edge",
                        "any neuromodulator concentration"],
    "bound_not_measured": ["homeostatic gain bounds gain_min/gain_max: hard software "
                           "bounds, chosen to span two decades, not derived from data",
                           "long-term per-edge bounds w_min_frac/w_max_frac",
                           "rewiring caps max_added_edges/max_removed_edges/"
                           "max_edges_total/per_edge_weight_cap_uS"],
}
