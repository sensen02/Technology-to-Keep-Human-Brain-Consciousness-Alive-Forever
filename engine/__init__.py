"""engine -- the project's simulation engine core.

This package does not simulate anything by itself.  It is the scaffolding that
makes the project's existing simulation modules safe to compose:

    units     explicit physical units; mismatches raise instead of silently
              producing a plausible wrong number
    params    parameter registry with ENFORCED provenance (a `measured`
              parameter without a citation cannot be created) and species/stage
              mixing checks
    layers    the L0..L6 layered stack, strictly feed-forward, with per-layer
              time steps; reading forwards or reaching back to ground truth is
              a hard error
    profile   organism profiles carrying the project's verified NEGATIVE
              constraints (Drosophila has no vasculature; Hydra neither; stages
              must not be mixed) as machine-enforced guards
    rules     a declarative behaviour-rule grammar with compile-time
              port/unit validation and auto-generated documentation
    neural    spiking LIF network for the fly nervous system, ready for
              the FlyWire connectome (signed synapses by neurotransmitter)
    checks    the verification harness (analytic / conservation / determinism /
              convergence / invariant / order) with an explicit statement of
              what a pass does NOT mean
    record    self-describing run records with the required caveats injected

The engine's job is to make the project's own hard-won rules impossible to
forget, not to add another model.
"""

from .units import Quantity, Q, Dims, UnitError, unit_of, convert
from .params import (Param, ParamRegistry, ProvenanceError,
                     MEASURED, DERIVED, ILLUSTRATIVE, ASSUMED)
from .layers import (Layer, Pipeline, Port, FeedforwardViolation, WiringError,
                     ConstantSource, Integrator, RatioMonitor)
from .profile import (OrganismProfile, ProfileViolation, PROFILES,
                      DROSOPHILA_NOTUM, DROSOPHILA_CNS, HYDRA, MAMMALIAN_TISSUE)
from .rules import Rule, RuleSet, RuleError
from .neural import (LIFNetwork, NetworkParams, lemplev_ziv,
                     binarize_population, build_synthetic_network, nt_sign)
from .checks import Check, CheckResult, CheckSuite, ChecksFailed
from .record import RunRecord

__all__ = [
    "Quantity", "Q", "Dims", "UnitError", "unit_of", "convert",
    "Param", "ParamRegistry", "ProvenanceError",
    "MEASURED", "DERIVED", "ILLUSTRATIVE", "ASSUMED",
    "Layer", "Pipeline", "Port", "FeedforwardViolation", "WiringError",
    "ConstantSource", "Integrator", "RatioMonitor",
    "OrganismProfile", "ProfileViolation", "PROFILES",
    "DROSOPHILA_NOTUM", "DROSOPHILA_CNS", "HYDRA", "MAMMALIAN_TISSUE",
    "Rule", "RuleSet", "RuleError",
    "LIFNetwork", "NetworkParams", "lemplev_ziv", "binarize_population",
    "build_synthetic_network", "nt_sign",
    "Check", "CheckResult", "CheckSuite", "ChecksFailed",
    "RunRecord",
]

__version__ = "0.1.0"
