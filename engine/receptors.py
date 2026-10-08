"""engine.receptors -- sensory transduction: environment -> receptor current.

WHY THIS EXISTS
---------------
The whole-brain LIF run showed the real connectome is silent without input, and
the sensorimotor interface exposed the ports but injected current DIRECTLY into
sensory neurons.  That is not transduction: it skips the receptor.  A virtual
environment should produce a physical stimulus (light intensity, tissue strain,
odorant concentration) and the RECEPTOR should turn it into a current.

BANC makes this attachable, because it annotates the actual receptor neurons:
photoreceptor_neuron (1,835), olfactory_receptor_neuron (2,967),
bristle_neuron (5,485), chordotonal_organ_neuron (1,945),
campaniform_sensillum_neuron (196), taste bristle/peg gustatory neurons,
hygrosensory_receptor_neuron (90), thermosensory_receptor_neuron (31).

MODELS
------
PhotoReceptor    Naka-Rushton response with light adaptation.  Steady state
                 R = Rmax I^n / (I^n + I50(I)^n) with a slowly adapting I50.
MechanoReceptor  two-state channel: P_open = 1 / (1 + exp(-(k x - dG)/kT))
ChemoReceptor    Langmuir binding, cross-checked against hormone.py
ThermoReceptor   saturating sigmoid around a preferred temperature
HygroReceptor    saturating sigmoid in humidity, dry- and moist-sensitive

Each has an exact steady state, so each is checked analytically rather than
just run.

UNITS -- READ THIS BEFORE FEEDING A RECEPTOR INTO A NEURAL TIER
---------------------------------------------------------------
The receptor models return an EQUIVALENT DRIVE, not a physical current:

    DRIVE_UNIT = "mV, i.e. the R_m * I drive that the LIF tier consumes"

``engine.neural.LIFNetwork.step(i_ext=...)`` works in exactly this convention, so
a receptor driving a LIF network needs NO conversion.  The CONDUCTANCE tiers do
NOT: ``neural_cond.ConductanceNetwork.step(current_nA=...)`` and
``neural_hist.HistamineNetwork.step(current_nA=...)`` take NANOAMPERES.  Feeding a
receptor drive straight into ``current_nA`` is therefore a silent unit error of
unknown size (the project has no measured Drosophila transduction gain, so there
is no defensible default factor).  ``ReceptorBank.current_nA`` exists to make that
conversion EXPLICIT and to REFUSE to guess: it requires a caller-supplied factor
with its provenance recorded.

HONEST LIMITS
-------------
* These are minimal transduction models.  Real phototransduction is a G-protein
  cascade with quantum bumps; real mechanotransduction involves NOMPC and
  other channels.  None of that is modelled.
* Every parameter here is ILLUSTRATIVE.  This project has no measured
  Drosophila receptor kinetics or sensitivities.
* Transduction produces a current.  It does not produce a percept, and nothing
  here claims the simulated fly sees, smells or feels anything.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PhotoReceptor", "MechanoReceptor", "ChemoReceptor",
           "ThermoReceptor", "HygroReceptor", "ReceptorBank",
           "DRIVE_UNIT", "ConversionRefused", "to_nA"]

#: Unit of every receptor model's output.  NOT nanoamperes.  See module docstring.
DRIVE_UNIT = "mV (R_m*I equivalent drive; the unit engine.neural.LIFNetwork consumes)"

#: Provenance text that every nA conversion factor must carry.
NO_MEASURED_TRANSDUCTION_GAIN = (
    "this project has no measured Drosophila transduction gain, so no default "
    "conversion factor exists; the caller must state where its factor comes from")


class ConversionRefused(ValueError):
    """Raised when an mV-equivalent receptor drive would be used as nanoamperes."""


def to_nA(drive_mV, nA_per_mV, provenance=None):
    """Convert a receptor DRIVE (mV-equivalent) into nanoamperes, explicitly.

    Parameters
    ----------
    drive_mV : array-like
        Output of a receptor model or ``ReceptorBank.current``.
    nA_per_mV : float
        The conversion factor.  There is NO default: the project has no measured
        transduction gain, so a default would be an invented number.
    provenance : str
        Where the factor comes from.  Required and must be non-trivial, so that
        a converted result can never be reported without its origin.
    """
    f = float(nA_per_mV)
    if not np.isfinite(f) or f <= 0:
        raise ConversionRefused(
            "nA_per_mV must be a positive finite number; got %r. " % (nA_per_mV,)
            + NO_MEASURED_TRANSDUCTION_GAIN)
    if not provenance or len(str(provenance).strip()) < 8:
        raise ConversionRefused(
            "a non-trivial provenance string is required for the conversion factor; "
            + NO_MEASURED_TRANSDUCTION_GAIN)
    a = np.asarray(drive_mV, float)
    if not np.isfinite(a).all():
        raise ConversionRefused("drive contains non-finite values")
    return a * f


class _Receptor:
    """Common interface: stimulus (arbitrary units) -> current in mV."""

    modality = "generic"

    def __init__(self, n_receptors, gain_mV=10.0, dt_ms=1.0, seed=0):
        self.n = int(n_receptors)
        self.gain = float(gain_mV)
        self.dt = float(dt_ms)
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.n)

    def steady_state(self, stimulus):
        """Closed-form steady-state response (normalised 0..1)."""
        raise NotImplementedError

    def step(self, stimulus, dt_ms=None):
        """Advance the transduction state; return the current in mV."""
        raise NotImplementedError

    def output_current(self):
        return self.gain * self.state


class PhotoReceptor(_Receptor):
    """Naka-Rushton with slow light adaptation.

        R(I)  = I^n / (I^n + I50_a^n)          instantaneous
        I50_a = I50 * (1 + a * A)              adaptation raises the half-point
        dA/dt = (I/(I + I50) - A) / tau_a      adaptation state chases the light
    Steady state: A = I/(I+I50), so I50_a = I50 (1 + a I/(I+I50)) and
        R_ss = I^n / (I^n + I50_a^n).
    """

    modality = "vision"
    DEFAULT = dict(n_exp=1.0, i50=0.5, adapt_gain=4.0, tau_a_ms=200.0)

    def __init__(self, n_receptors, gain_mV=10.0, dt_ms=1.0, seed=0, **kw):
        super().__init__(n_receptors, gain_mV, dt_ms, seed)
        self.p = dict(self.DEFAULT)
        self.p.update(kw)
        self.A = np.zeros(self.n)

    def _i50_adapted(self, I):
        p = self.p
        return p["i50"] * (1.0 + p["adapt_gain"] * self.A)

    def instantaneous(self, I):
        p = self.p
        I = np.maximum(np.asarray(I, float), 0.0)
        i50 = self._i50_adapted(I)
        In = I ** p["n_exp"]
        return In / (In + i50 ** p["n_exp"])

    def steady_state(self, I):
        p = self.p
        I = np.maximum(np.asarray(I, float), 0.0)
        A = I / (I + p["i50"])
        i50 = p["i50"] * (1.0 + p["adapt_gain"] * A)
        In = I ** p["n_exp"]
        return In / (In + i50 ** p["n_exp"])

    def step(self, stimulus, dt_ms=None):
        p = self.p
        dt = self.dt if dt_ms is None else float(dt_ms)
        I = np.maximum(np.asarray(stimulus, float), 0.0)
        target = I / (I + p["i50"])
        self.A += (dt / p["tau_a_ms"]) * (target - self.A)
        self.state = self.instantaneous(I)
        return self.output_current()


class MechanoReceptor(_Receptor):
    """Two-state mechanotransduction channel.

        P_open = 1 / (1 + exp(-(k x - dG) / kT))
    where x is the local strain and dG the resting free-energy difference.  The
    steady state is the Boltzmann function itself, so the model is exact by
    construction and can be checked against it.
    """

    modality = "mechanosensory"
    DEFAULT = dict(kT=1.0, k_gate=6.0, dG=4.0, tau_ms=5.0)

    def __init__(self, n_receptors, gain_mV=10.0, dt_ms=1.0, seed=0, **kw):
        super().__init__(n_receptors, gain_mV, dt_ms, seed)
        self.p = dict(self.DEFAULT)
        self.p.update(kw)

    def steady_state(self, strain):
        p = self.p
        x = np.asarray(strain, float)
        z = (p["k_gate"] * x - p["dG"]) / p["kT"]
        return 1.0 / (1.0 + np.exp(-z))

    def step(self, stimulus, dt_ms=None):
        p = self.p
        dt = self.dt if dt_ms is None else float(dt_ms)
        tgt = self.steady_state(stimulus)
        self.state += (dt / p["tau_ms"]) * (tgt - self.state)
        return self.output_current()


class ChemoReceptor(_Receptor):
    """Langmuir binding of an odorant, with a first-order approach to it.

        theta = c / (c + Kd)
    Cross-checked against `hormone.equilibrium_occupancy`, which solves the
    same equilibrium -- two independent implementations agreeing is a real
    check, even though both are simple.
    """

    modality = "olfactory"
    DEFAULT = dict(kd=1.0, tau_ms=50.0, hill=1.0)

    def __init__(self, n_receptors, gain_mV=10.0, dt_ms=1.0, seed=0, **kw):
        super().__init__(n_receptors, gain_mV, dt_ms, seed)
        self.p = dict(self.DEFAULT)
        self.p.update(kw)

    def steady_state(self, conc):
        p = self.p
        c = np.maximum(np.asarray(conc, float), 0.0)
        ch = c ** p["hill"]
        return ch / (p["kd"] ** p["hill"] + ch)

    def step(self, stimulus, dt_ms=None):
        p = self.p
        dt = self.dt if dt_ms is None else float(dt_ms)
        tgt = self.steady_state(stimulus)
        self.state += (dt / p["tau_ms"]) * (tgt - self.state)
        return self.output_current()


class ThermoReceptor(_Receptor):
    """Saturating sigmoid around a preferred temperature (normalised)."""

    modality = "thermosensory"
    DEFAULT = dict(t_pref=25.0, width=8.0, tau_ms=100.0)

    def __init__(self, n_receptors, gain_mV=10.0, dt_ms=1.0, seed=0, **kw):
        super().__init__(n_receptors, gain_mV, dt_ms, seed)
        self.p = dict(self.DEFAULT)
        self.p.update(kw)

    def steady_state(self, temp_C):
        p = self.p
        z = (np.asarray(temp_C, float) - p["t_pref"]) / p["width"]
        return 1.0 / (1.0 + np.exp(-z))

    def step(self, stimulus, dt_ms=None):
        p = self.p
        dt = self.dt if dt_ms is None else float(dt_ms)
        tgt = self.steady_state(stimulus)
        self.state += (dt / p["tau_ms"]) * (tgt - self.state)
        return self.output_current()


class HygroReceptor(ThermoReceptor):
    """Humidity transduction; `t_pref` is the half-point humidity instead."""

    modality = "hygrosensory"
    DEFAULT = dict(t_pref=0.5, width=0.15, tau_ms=100.0)


class ReceptorBank:
    """Attach receptor populations to real sensory-neuron groups.

    The bank maps a modality name to (receptor model, neuron indices).  It
    converts a physical stimulus into a per-neuron current array for the whole
    connectome, which is what the LIF network consumes.
    """

    def __init__(self, n_neurons, dt_ms=1.0, seed=0):
        self.n_neurons = int(n_neurons)
        self.dt = float(dt_ms)
        self.entries = {}
        self.seed = seed

    def attach(self, modality, indices, model_cls, gain_mV=10.0, **model_kw):
        idx = np.asarray(indices, dtype=np.int64)
        model = model_cls(idx.size, gain_mV=gain_mV, dt_ms=self.dt,
                          seed=self.seed, **model_kw)
        self.entries[modality] = {"indices": idx, "model": model}
        return self

    def current(self, stimuli, dt_ms=None):
        """`stimuli` maps modality -> stimulus (scalar, or per-receptor array).

        Returns a per-neuron array in the DRIVE_UNIT (mV-equivalent), which is
        the convention ``engine.neural.LIFNetwork`` consumes.  It is NOT
        nanoamperes: for the conductance tiers use ``current_nA`` and supply a
        conversion factor with provenance.  Modalities not present in ``stimuli``
        receive zero.
        """
        i_ext = np.zeros(self.n_neurons)
        for m, ent in self.entries.items():
            if m not in stimuli:
                continue
            s = stimuli[m]
            arr = np.asarray(s, float)
            if arr.ndim == 0:
                arr = np.full(ent["indices"].size, float(arr))
            else:
                arr = np.broadcast_to(arr, (ent["indices"].size,))
            i_ext[ent["indices"]] = ent["model"].step(arr, dt_ms=dt_ms)
        return i_ext

    def current_nA(self, stimuli, nA_per_mV, provenance=None, dt_ms=None):
        """Like ``current`` but returns NANOAMPERES for the conductance tiers.

        The factor and its provenance are REQUIRED (see ``to_nA``): this project
        has no measured transduction gain, so converting silently would invent a
        number.  Use this whenever the target is ``neural_cond`` or
        ``neural_hist`` rather than the LIF tier.
        """
        return to_nA(self.current(stimuli, dt_ms=dt_ms), nA_per_mV, provenance)

    def describe(self):
        rows = ["ReceptorBank:"]
        for m, e in self.entries.items():
            rows.append(f"  {m:16s} {e['indices'].size:>6,} 个受体神经元  "
                        f"模型 {type(e['model']).__name__}")
        return "\n".join(rows)
