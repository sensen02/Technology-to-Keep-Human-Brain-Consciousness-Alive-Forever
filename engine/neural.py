"""engine.neural -- spiking neural layer for the fly nervous system.

WHY A SEPARATE MODULE
---------------------
The nervous system is the one part of this project that was at 0% and is also
the part the user actually wants.  It has to satisfy the same engine rules as
everything else: explicit units, enforced provenance, feed-forward wiring, and
checks that can fail.

WHAT IS MODELLED
----------------
Leaky integrate-and-fire (LIF) neurons with an absolute refractory period, a
fixed synaptic delay, and signed synapses:

    tau_m dV/dt = -(V - V_rest) + R_m I(t)
    I(t) = I_ext + sum_{pre} w(pre,post) * s_pre(t - d)

Sign convention follows Drosophila physiology, which the FlyWire dataset
annotates per neuron: acetylcholine is the main EXCITATORY transmitter and
GABA / glutamate are INHIBITORY.  So the weight of an edge is
``synapse_count * sign(presynaptic neuron)``, and a connectome with
``nt_type`` annotations determines the sign of every edge -- that part is data,
not a modelling choice.

DESIGNED FOR THE REAL CONNECTOME
--------------------------------
The network is stored as a signed sparse matrix in CSR order, and the update is
one sparse matrix-vector product per step.  For the FlyWire FAFB snapshot
(139,255 neurons, 3,732,460 connections) that is a 3.7M-nonzero matvec, a few
milliseconds, so a second of simulated whole-brain activity costs on the order
of seconds of CPU.  The whole fly brain is therefore *computationally* cheap;
the real constraint is data access, not compute.

UNITS
-----
Voltages are mV, time ms, currents in the arbitrary "R_m * I" units the LIF
reduction works in.  Every parameter is registered with its provenance; the
project has no verified Drosophila neuron-parameter measurements, so most are
ILLUSTRATIVE and say so.

HONEST LIMITS
-------------
* A LIF neuron is not a neuron.  It has no dendrite, no channels, no
  adaptation, no neuromodulation.  Spikes are all-or-none but shape and
  propagation are not modelled.
* "Structure is not function": the connectome fixes WHO talks to WHOM, not the
  synaptic weights, the release probability, or the physiology.  Getting a
  connectome does not give you a working brain model, and this module does not
  claim otherwise.
* Complexity readouts (Lempel-Ziv, synchrony) are DESCRIPTIVE MEASURES.  None of
  them measures experience, and passing them is not evidence of anything about
  consciousness.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import sparse as _sp
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False

__all__ = ["LIFNetwork", "NetworkParams", "lemplev_ziv", "build_synthetic_network"]

# Drosophila neurotransmitter -> sign.  ACh is the main excitatory transmitter;
# GABA and glutamate are inhibitory.  This is the annotation scheme FlyWire uses.
EXCITATORY_NT = {"ACH", "ACh", "acetylcholine", "EXC", "exc", "glu_exc"}
INHIBITORY_NT = {"GABA", "GLUT", "Glu", "glutamate", "INH", "inh"}


def nt_sign(nt_type):
    """+1 for excitatory, -1 for inhibitory, 0 if unknown."""
    if nt_type is None:
        return 0
    t = str(nt_type).strip()
    if t in EXCITATORY_NT:
        return 1
    if t in INHIBITORY_NT:
        return -1
    up = t.upper()
    if up in ("ACH", "EXC", "EXCITATORY"):
        return 1
    if up in ("GABA", "GLUT", "GLU", "INH", "INHIBITORY"):
        return -1
    return 0


class NetworkParams:
    """Neuron parameters, each with a provenance tag.

    There are no verified Drosophila single-neuron LIF parameters in this
    project's records (the records only note that innexin gap junctions have NO
    measured single-channel conductance for the fly).  Everything here is
    therefore ILLUSTRATIVE unless stated otherwise, and it is labelled so that a
    reader cannot mistake it for measured physiology.
    """

    DEFAULTS = {
        "tau_m_ms":       (20.0,  "illustrative"),
        "v_rest_mV":      (-65.0, "illustrative"),
        "v_th_mV":        (-50.0, "illustrative"),
        "v_reset_mV":     (-65.0, "illustrative"),
        "t_ref_ms":       (2.0,   "illustrative"),
        "syn_delay_ms":   (2.0,   "illustrative; FlyWire does not provide delays"),
        "w_scale_mV":     (0.5,   "illustrative; per-synapse weight scale"),
        "noise_mV":       (1.0,   "illustrative; membrane noise sd"),
    }

    def __init__(self, **over):
        self.p = dict(self.DEFAULTS)
        for k, v in over.items():
            if k not in self.p:
                raise KeyError(f"unknown neuron parameter {k!r}")
            self.p[k] = (float(v), "override")

    def __getitem__(self, k):
        return self.p[k][0]

    def provenance(self, k):
        return self.p[k][1]

    def to_dict(self):
        return {k: v[0] for k, v in self.p.items()}

    def provenance_dict(self):
        return {k: v[1] for k, v in self.p.items()}


try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    _HAVE_NUMBA = False


def _lz76_py(s):
    """Reference LZ76 (Kaspar & Schuster).  O(n^2) on sparse input in CPython."""
    n = s.size
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while l + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k_max = 1
                k = 1
            else:
                k = 1
    return c


if _HAVE_NUMBA:
    _lz76_core = _njit(cache=True)(_lz76_py)
else:                                                # pragma: no cover
    _lz76_core = _lz76_py


def lemplev_ziv(binary_string, max_len=None):
    """Lempel-Ziv complexity (LZ76) of a binary sequence, normalised by length.

    A DESCRIPTIVE measure of how non-repetitive a spike raster is; used in the
    perturbational-complexity literature.  It is **not** a test of
    consciousness and must never be reported as one.

    WHY THIS IS JIT-COMPILED
    ------------------------
    The straightforward transcription of LZ76 is O(n^2) in CPython on sparse
    input -- which is exactly what a spike raster is.  Measured: 60,000
    elements of 1%-density data took **43.5 seconds**.  With the whole-brain
    raster (139,255 neurons x 1000 steps = 1.4e8 elements) it would never
    finish, and because the call sits inside `run(record=True)` it hung the
    engine at 100% CPU with flat memory rather than raising.  The neural
    self-test caught it.  The same algorithm under numba is ~1000x faster.

    `max_len` optionally truncates the input (deterministically, from the
    start) for enormous rasters; when it truncates, it says so in the return
    value's companion attribute via `lemplev_ziv_info`.
    """
    s = np.ascontiguousarray(np.asarray(binary_string).astype(np.uint8).ravel())
    if max_len is not None and s.size > max_len:
        s = s[:max_len]
    n = s.size
    if n == 0:
        return 0.0
    return float(_lz76_core(s)) / float(n)


def binarize_population(spikes, bin_steps=10, threshold=None):
    """Binarise a spike raster into a 1-D population-activity string.

    THIS is the object a complexity measure should be applied to.  Computing LZ
    over the raw (neuron x time) raster is both far more expensive and less
    meaningful: the perturbational-complexity literature works on a binned
    POPULATION response, not on individual spike trains.

    spikes     : (n_steps, n_neurons) boolean array
    bin_steps  : time steps per bin
    threshold  : a bin is 1 when its spike count is >= threshold
                 (default 1, i.e. "any activity in this bin")
    """
    s = np.asarray(spikes)
    n_steps = s.shape[0]
    bs = max(1, int(bin_steps))
    nb = max(1, n_steps // bs)
    trimmed = s[:nb * bs].reshape(nb, bs, -1).sum(axis=(1, 2))
    thr = 1 if threshold is None else threshold
    return (trimmed >= thr).astype(np.uint8)


def lemplev_ziv_info(binary_string, max_len=None):
    s = np.asarray(binary_string).ravel()
    used = s.size if max_len is None else min(s.size, max_len)
    return {"lz": lemplev_ziv(s, max_len=max_len), "n_used": int(used),
            "n_total": int(s.size), "truncated": bool(used < s.size)}


class LIFNetwork:
    """A vectorised LIF network with signed synapses and a fixed delay."""

    def __init__(self, n_neurons, params=None, dt_ms=1.0, seed=0):
        self.n = int(n_neurons)
        self.p = params or NetworkParams()
        self.dt = float(dt_ms)
        self.rng = np.random.default_rng(seed)
        self.t_ms = 0.0
        self.spike_count = np.zeros(self.n, dtype=np.int64)

        self.v = np.full(self.n, self.p["v_rest_mV"])
        self.refractory = np.zeros(self.n, dtype=np.int64)   # steps remaining
        self.delay_steps = max(1, int(round(self.p["syn_delay_ms"] / self.dt)))
        self._ring = np.zeros((self.delay_steps, self.n))
        self._ring_i = 0

        self.W = None            # CSR, shape (post, pre), signed
        self.n_synapses = 0
        self.exc_frac = None
        self.input_rate_hz = None

    # ------------------------------------------------------------------
    def set_connectivity(self, W):
        """Set the signed sparse weight matrix, shape (n_post, n_pre)."""
        if W.shape != (self.n, self.n):
            raise ValueError(
                f"weight matrix must be ({self.n},{self.n}), got {W.shape}")
        self.W = W.tocsr() if _HAVE_SCIPY and _sp.issparse(W) else W
        self.n_synapses = int(self.W.nnz)
        return self

    def set_connectivity_from_edges(self, pre, post, syn_count, nt_of_presynaptic):
        """Build signed weights from a FlyWire-style edge list.

        pre, post        : int arrays of presynaptic / postsynaptic neuron index
        syn_count        : number of synapses for that pair
        nt_of_presynaptic: array of neurotransmitter labels per neuron index
        """
        if not _HAVE_SCIPY:
            raise RuntimeError("scipy is required for sparse connectivity")
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        cnt = np.asarray(syn_count, dtype=np.float64)
        signs = np.array([nt_sign(x) for x in np.asarray(nt_of_presynaptic)],
                         dtype=np.float64)
        unknown = int(np.sum(signs == 0))
        signs[signs == 0] = 1.0     # unknown -> treat as excitatory, and report it
        w = cnt * signs[pre] * self.p["w_scale_mV"]
        W = _sp.csr_matrix((w, (post, pre)), shape=(self.n, self.n))
        self.set_connectivity(W)
        self.exc_frac = float(np.mean(np.asarray(nt_of_presynaptic, dtype=object)
                                      != "GABA")) if len(nt_of_presynaptic) else None
        return {"n_edges": int(len(pre)), "n_unknown_nt": unknown,
                "nnz": int(W.nnz)}

    # ------------------------------------------------------------------
    def step(self, i_ext=None, gain=None):
        """Advance one dt.  Returns the boolean spike vector.

        `gain` is a per-neuron multiplicative factor on the synaptic input
        (and on the external drive), used to model metabolic limitation: it is
        the port through which the oxygen/ATP layer acts on the nervous system.
        """
        p = self.p
        tau, vr, vth, vres = (p["tau_m_ms"], p["v_rest_mV"],
                              p["v_th_mV"], p["v_reset_mV"])
        alpha = self.dt / tau

        # synaptic input arriving now (delayed).  Always an array so that the
        # vectorised update below is valid for a single isolated neuron too.
        if self.W is not None:
            i_syn = self.W.dot(self._ring[self._ring_i])
        else:
            i_syn = np.zeros(self.n)
        if gain is not None:
            i_syn = i_syn * np.asarray(gain, dtype=float)
        if i_ext is not None:
            drive = np.asarray(i_ext, dtype=float)
            if gain is not None:
                drive = drive * np.asarray(gain, dtype=float)
            i_syn = i_syn + drive

        noise = p["noise_mV"] * self.rng.standard_normal(self.n) * np.sqrt(alpha)

        active = self.refractory <= 0
        self.v[active] += alpha * (-(self.v[active] - vr) + i_syn[active]) + noise[active]
        self.v[~active] = vres
        self.refractory = np.maximum(self.refractory - 1, 0)

        spiked = (self.v >= vth) & active
        if spiked.any():
            self.v[spiked] = vres
            self.refractory[spiked] = max(1, int(round(p["t_ref_ms"] / self.dt)))
            self.spike_count[spiked] += 1

        # Delay line: the slot we just READ is the one written `delay_steps`
        # steps ago, so we overwrite it with the current spikes and advance.
        # (Reading and writing in this order is what makes the delay exactly
        # `delay_steps`; getting it backwards gives delay_steps - 1.)
        self._ring[self._ring_i] = spiked.astype(float)
        self._ring_i = (self._ring_i + 1) % self.delay_steps

        self.t_ms += self.dt
        return spiked

    # ------------------------------------------------------------------
    def run(self, duration_ms, i_ext=None, gain=None, record=False,
            lz_mode="population", lz_bin_steps=10, lz_max_len=200000):
        """Run and return a summary dict; optionally the full spike raster.

        `lz_mode` selects what the Lempel-Ziv complexity is computed on:
          "population" (default) -- the binarised population-activity string.
              This is standard practice for this measure and it is cheap: the
              string has one entry per time bin, not per neuron per time step.
          "raster"               -- the raw flattened neuron x time raster.
              Our LZ76 is O(n^2) on sparse input, so this is guarded by
              `lz_max_len`; the input is truncated from the start and the
              truncation is reported in the output.
        """
        n_steps = max(1, int(round(duration_ms / self.dt)))
        spikes = np.zeros((n_steps, self.n), dtype=bool) if record else None
        rates = np.zeros(n_steps)
        lfp = np.zeros(n_steps)
        for k in range(n_steps):
            s = self.step(i_ext=i_ext, gain=gain)
            if record:
                spikes[k] = s
            rates[k] = s.mean() / (self.dt / 1000.0)     # Hz
            lfp[k] = float(self.v.mean() - self.p["v_rest_mV"])
        out = {"times_ms": np.arange(1, n_steps + 1) * self.dt,
               "rate_hz": rates, "lfp_proxy_mV": lfp,
               "mean_rate_hz": float(rates.mean()),
               "total_spikes": int(self.spike_count.sum())}
        if record:
            out["spikes"] = spikes
            if lz_mode == "population":
                pop = binarize_population(spikes, bin_steps=lz_bin_steps)
                out["lz_complexity"] = lemplev_ziv(pop)
                out["lz_on"] = "population_binned"
                out["lz_n"] = int(pop.size)
            else:
                info = lemplev_ziv_info(spikes.reshape(-1).astype(np.uint8),
                                        max_len=lz_max_len)
                out["lz_complexity"] = info["lz"]
                out["lz_on"] = "raster"
                out["lz_n"] = info["n_used"]
                out["lz_truncated"] = info["truncated"]
            # synchrony: variance/mean of the per-step spike count (Fano factor)
            counts = spikes.sum(axis=1)
            out["fano"] = (float(counts.var() / counts.mean())
                           if counts.mean() > 0 else 0.0)
        return out

    # ------------------------------------------------------------------
    def f_i_curve(self, currents, duration_ms=2000.0):
        """Steady-state rate vs constant injected current (for the analytic test)."""
        out = []
        for I in currents:
            net = LIFNetwork(self.n, self.p, self.dt, seed=int(self.rng.integers(1 << 30)))
            net.W = None      # isolated neuron: no synaptic drive
            net.v[:] = self.p["v_reset_mV"]
            net.refractory[:] = 0
            r = net.run(duration_ms, i_ext=I)
            out.append(r["mean_rate_hz"])
        return np.array(out)


def analytic_lif_rate(I, tau_m_ms, v_rest, v_th, v_reset, t_ref_ms):
    """Closed-form steady-state rate of a noise-free LIF under constant drive.

    With V_inf = v_rest + I, the time to rise from v_reset to v_th is
        t = tau_m * ln((v_reset - V_inf) / (v_th - V_inf))
    and the rate is 1 / (t_ref + t).  Below threshold the rate is 0.
    Returns nan when the drive cannot reach threshold.
    """
    v_inf = v_rest + I
    num = v_reset - v_inf
    den = v_th - v_inf
    if den == 0 or num / den <= 0:
        return float("nan")
    t = tau_m_ms * np.log(num / den)
    if t <= 0:
        return float("nan")
    return 1000.0 / (t_ref_ms + t)


def build_synthetic_network(n, n_edges, exc_frac=0.6, seed=0, w_scale=0.5,
                            syn_per_edge=6, params=None, nt_unknown=0.0):
    """A FlyWire-SHAPED synthetic network, for testing the machinery before the
    real connectome is available.  Its topology is random, so it is not a fly
    and must never be described as one."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, size=n_edges)
    post = rng.integers(0, n, size=n_edges)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    cnt = rng.integers(1, syn_per_edge * 2 + 1, size=pre.size).astype(float)
    nt = np.where(rng.random(n) < exc_frac, "ACH", "GABA").astype(object)
    if nt_unknown > 0:
        m = rng.random(n) < nt_unknown
        nt[m] = None
    return pre, post, cnt, nt
