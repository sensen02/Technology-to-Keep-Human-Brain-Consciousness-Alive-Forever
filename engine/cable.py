"""Passive multicompartment cable approximation (not an active neuron backend).

Geometry is in um; state voltage mV, time ms, capacitance nF, conductance uS,
resistance ohm and applied current nA. Positive applied current is inward.
Backward Euler is stable for this passive linear system, but large timesteps
still lose temporal accuracy. Explicit voltage-dependent synaptic currents do
not inherit that unconditional stability guarantee.

Each nonroot row represents the segment ending at that SWC node. Membrane is
the lateral surface of its linearly tapered frustum (no end caps); axial paths
join segment centers through their shared node. A root is a junction with an
explicit 1e-6 um virtual cylindrical length, not a reconstructed soma. This
small regularization preserves legacy cylinder discretization. Branch junctions
use the parent segment's distal half for each daughter; this is an approximate
center-to-center tree discretization, not a full junction finite-volume mesh.

The default passive parameters were fitted to DNp01/DNp03 (PMC11071487), not
validated for every cell. No active channels, vesicle release or neuromodulation
are implemented. NEURON is an independent reference implementation; no general
scalability comparison is established here. Neurotransmitter identity alone does
not determine current sign: reversal potential and membrane voltage do.
"""
from __future__ import annotations

import re
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

__all__ = ["Morphology", "CableNeuron", "Synapse", "DN_PASSIVE", "swc_to_morphology"]
DN_PASSIVE = {
    "Cm_uF_cm2": (0.7, "measured/fitted: DNp01 & DNp03, PMC11071487"),
    "g_leak_S_cm2": (4.35e-4, "measured/fitted: DNp01 & DNp03, PMC11071487"),
    "Ra_ohm_cm": (212.0, "measured/fitted: DNp01 & DNp03, PMC11071487"),
    "E_leak_mV": (-66.63, "measured/fitted: DNp01 & DNp03, PMC11071487"),
}


def _number(value, name, minimum=None, strict=False):
    value = float(value)
    if not np.isfinite(value) or (minimum is not None and
            (value <= minimum if strict else value < minimum)):
        raise ValueError(f"invalid {name}: {value}")
    return value


def _indices(values, n, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= n):
        raise ValueError(f"{name} must be a one-dimensional sequence of node indices")
    return a.astype(int)


class Morphology:
    """One connected rooted tree, in micrometers; row order is never changed.

    Diameters must be finite and strictly positive; nonroot edges must have
    nonzero length. Arrays are copied and read-only to prevent stale matrices.
    """
    def __init__(self, x, y, z, d, parent):
        arrays = [np.array(a, dtype=float, copy=True) for a in (x, y, z, d)]
        if any(a.ndim != 1 for a in arrays) or not arrays[0].size:
            raise ValueError("geometry must contain nonempty one-dimensional arrays")
        self.n = arrays[0].size
        if any(a.size != self.n or not np.isfinite(a).all() for a in arrays):
            raise ValueError("geometry arrays must have equal size and finite values")
        if np.any(arrays[3] <= 0):
            raise ValueError("diameters must be strictly positive (no radius clamping)")
        p = np.asarray(parent)
        if p.shape != (self.n,) or p.dtype.kind not in "iu":
            raise ValueError("parent must contain integer indices")
        if np.any(p < -1) or np.any(p >= self.n) or np.count_nonzero(p == -1) != 1:
            raise ValueError("parent must describe exactly one root and valid indices")
        self.x, self.y, self.z, self.d = arrays
        self.parent = p.astype(int, copy=True)
        # Linear-time cycle check, independent of input row ordering.
        state = np.zeros(self.n, dtype=np.uint8)
        for start in range(self.n):
            if state[start]:
                continue
            trail, node = [], start
            while node != -1 and state[node] == 0:
                state[node] = 1
                trail.append(node)
                node = int(self.parent[node])
            if node != -1 and state[node] == 1:
                raise ValueError("parent topology contains a cycle")
            state[trail] = 2
        if np.any(self.lengths_um()[self.parent >= 0] <= 0):
            raise ValueError("nonroot edges must have positive length")
        for a in (*arrays, self.parent):
            a.flags.writeable = False

    def lengths_um(self):
        lengths = np.zeros(self.n)
        i = np.flatnonzero(self.parent >= 0)
        p = self.parent[i]
        lengths[i] = np.sqrt((self.x[i] - self.x[p]) ** 2 +
                             (self.y[i] - self.y[p]) ** 2 +
                             (self.z[i] - self.z[p]) ** 2)
        return lengths

    def children(self):
        ch = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(i)
        return ch

    @classmethod
    def cylinder(cls, length_um, diameter_um, nseg=50):
        length_um = _number(length_um, "length_um", 0, True)
        diameter_um = _number(diameter_um, "diameter_um", 0, True)
        if isinstance(nseg, (bool, np.bool_)) or not isinstance(nseg, (int, np.integer)) or nseg < 1:
            raise ValueError("nseg must be a positive integer")
        x = np.linspace(0., length_um, nseg + 1)
        return cls(x, np.zeros_like(x), np.zeros_like(x),
                   np.full(nseg + 1, diameter_um), np.arange(-1, nseg))


def swc_to_morphology(path, keep=None, *, units=None):
    """Read strict seven-column SWC; return (morphology, IDs in file row order).

    Coordinate AND radius units must be declared explicitly ('um' or 'nm') or
    in a recognized units header (including navis Meta '1 nanometer'). Conflicting
    declarations and unlabeled input without explicit units are refused.
    Missing parents, forests, cycles, duplicate IDs and invalid radii are errors.
    Reduction is disabled until a topology-preserving reducer with ID mapping
    exists. Comments after seven columns are accepted.
    """
    if units not in (None, "um", "nm"):
        raise ValueError("units must be 'um' or 'nm'")
    declared = set()
    ids, xyz, radii, parents = [], [], [], []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if line.lstrip().startswith("#") and re.search(r"\bunits\b", line, re.I):
                match = re.search(r'''\bunits["']?\s*[:=]\s*["']?(?:1\s+)?(nanometers?|nanometres?|nm|micrometers?|micrometres?|microns?|um|µm)\b''', line, re.I)
                if not match:
                    raise ValueError(f"unrecognized SWC units declaration on line {lineno}")
                declared.add("nm" if match[1].lower().startswith(("nano", "nm")) else "um")
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) != 7:
                raise ValueError(f"SWC line {lineno}: expected seven columns")
            try:
                nid, _kind, parent = int(fields[0]), int(fields[1]), int(fields[6])
                pos = tuple(float(v) for v in fields[2:5])
                radius = float(fields[5])
            except ValueError as exc:
                raise ValueError(f"invalid SWC line {lineno}") from exc
            if nid < 0:
                raise ValueError(f"SWC line {lineno}: node IDs must be nonnegative")
            ids.append(nid)
            xyz.append(pos)
            radii.append(radius)
            parents.append(parent)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("SWC must contain nodes with unique IDs")
    idx = {nid: i for i, nid in enumerate(ids)}
    missing = set(parents) - set(ids) - {-1}
    if missing:
        raise ValueError(f"SWC references missing parents: {sorted(missing)}")
    if len(declared) > 1 or (units is not None and declared and units not in declared):
        raise ValueError("conflicting SWC units declarations")
    if units is None:
        if not declared:
            raise ValueError("SWC units are undeclared; pass units='um' or units='nm'")
        units = declared.pop()
    scale = 1.0 if units == "um" else 0.001
    xyz = np.asarray(xyz) * scale
    m = Morphology(*xyz.T, 2 * np.asarray(radii) * scale,
                   np.array([idx[p] if p != -1 else -1 for p in parents]))
    if keep is not None:
        if isinstance(keep, bool) or not isinstance(keep, (int, np.integer)) or keep < 1:
            raise ValueError("keep must be a positive integer")
        if keep < m.n:
            raise NotImplementedError("unsafe SWC reduction disabled; topology-preserving reducer required")
    return m, ids


class CableNeuron:
    """Passive implicit cable: um, ms, mV, nF, uS, nA (positive inward).

    siz_index is retained only as an injection-site convenience; it does not
    enable a spike initiation mechanism. seed is retained for compatibility.
    Call _build_matrix after deliberate g_mem/g_ax edits. dt changes invalidate
    the cached factorization automatically. Morphology is immutable.
    """
    def __init__(self, morph, Cm_uF_cm2=None, g_leak_S_cm2=None,
                 Ra_ohm_cm=None, E_leak_mV=None, siz_index=None,
                 active=False, dt_ms=0.025, seed=0):
        if active:
            raise NotImplementedError("active cable backend is not implemented; use active=False")
        self.m = morph
        def param(value, key, minimum=None, strict=False):
            return _number(DN_PASSIVE[key][0] if value is None else value, key, minimum, strict)
        self.Cm = param(Cm_uF_cm2, "Cm_uF_cm2", 0, True)
        self.g_leak = param(g_leak_S_cm2, "g_leak_S_cm2", 0)
        self.Ra = param(Ra_ohm_cm, "Ra_ohm_cm", 0, True)
        self.E_leak = param(E_leak_mV, "E_leak_mV")
        self.dt = _number(dt_ms, "dt_ms", 0, True)
        self.t, self.active = 0.0, False
        self.siz = int(np.argmax(morph.d)) if siz_index is None else int(_indices([siz_index], morph.n, "siz_index")[0])
        L = morph.lengths_um()
        L[morph.parent < 0] = 1e-6  # explicit virtual root junction
        distal = morph.d / 2
        proximal = distal.copy()
        nonroot = np.flatnonzero(morph.parent >= 0)
        proximal[nonroot] = distal[morph.parent[nonroot]]
        middle = (proximal + distal) / 2
        self.area_um2 = np.pi * (proximal + distal) * np.hypot(L, distal - proximal)
        self.area_cm2 = self.area_um2 * 1e-8
        self.C = self.Cm * self.area_cm2 * 1000  # nF; nF/ms = uS
        self.g_mem = self.g_leak * self.area_cm2 * 1e6
        self.g_ax = np.zeros(morph.n)
        # Exact integral of Ra/(pi*r(x)^2) along each linearly tapered half.
        # L/r^2 in um^-1 -> cm^-1 multiplies by 1e4.
        p = morph.parent[nonroot]
        resistance = self.Ra * 1e4 / np.pi * (
            L[nonroot] / (2 * proximal[nonroot] * middle[nonroot]) +
            L[p] / (2 * middle[p] * distal[p]))
        self.g_ax[nonroot] = 1e6 / resistance
        self.g_end = np.zeros(morph.n)
        self.E_end = np.full(morph.n, self.E_leak)
        self._build_matrix()
        self.v = np.full(morph.n, self.E_leak)
        self.rng = np.random.default_rng(seed)

    def _build_matrix(self):
        n = self.m.n
        i = np.flatnonzero(self.m.parent >= 0)
        p, g = self.m.parent[i], self.g_ax[i]
        diagonal = self.g_mem + self.g_end
        diagonal = diagonal.copy()
        np.add.at(diagonal, i, g)
        np.add.at(diagonal, p, g)
        # O(N) triplets: never allocate a dense N-by-N intermediate.
        rows = np.concatenate((np.arange(n), i, p))
        cols = np.concatenate((np.arange(n), p, i))
        self.G = sparse.coo_matrix((np.concatenate((diagonal, -g, -g)),
                                    (rows, cols)), shape=(n, n)).tocsr()
        self.G.eliminate_zeros()
        self._A = self._lu = None
        self._factor_dt = None

    def cut_edges(self, child_indices):
        """Disconnect parent axial edges by CHILD ROW INDEX, not SWC ID.

        This is irreversible on this instance, idempotent, and does not change
        row mapping, geometry or membrane. New ends are sealed unless a separate
        set_end_leak call adds a phenomenological conductance at either endpoint.
        """
        indices = _indices(child_indices, self.m.n, "child_indices")
        if np.any(self.m.parent[indices] < 0):
            raise ValueError("cannot cut a root's nonexistent parent edge")
        self.g_ax[indices] = 0
        self._build_matrix()

    def set_end_leak(self, node_indices, g_uS, E_rev_mV):
        """Set (not increment) distinct end conductance/reversal at chosen rows.

        Caller selects exposed endpoints; no membrane density or area is inferred.
        Zero conductance removes the leak. Does not modify ordinary membrane leak.
        """
        indices = _indices(node_indices, self.m.n, "node_indices")
        g = _number(g_uS, "g_uS", 0)
        reversal = _number(E_rev_mV, "E_rev_mV")
        self.g_end[indices], self.E_end[indices] = g, reversal
        self._build_matrix()

    def _lhs(self):
        dt = _number(self.dt, "dt_ms", 0, True)
        if self._lu is None or self._factor_dt != dt:
            self._lu = splu((sparse.diags(self.C / dt) + self.G).tocsc())
            self._factor_dt = dt
        return self._lu

    def step(self, i_syn=None, i_inject=None):
        """Advance dt; currents are nA, POSITIVE INWARD, scalar or (n,) array.

        Use Synapse.inward_current(v), or -Synapse.current(v). These are explicit
        current samples, not conductances incorporated implicitly into the solve.
        """
        lu = self._lhs()
        rhs = self.C / self.dt * self.v + self.g_mem * self.E_leak + self.g_end * self.E_end
        for current in (i_syn, i_inject):
            if current is not None:
                current = np.asarray(current, float)
                if current.shape not in ((), (self.m.n,)) or not np.isfinite(current).all():
                    raise ValueError("current must be finite scalar or per-compartment (n,) array")
                rhs = rhs + current
        self.v = lu.solve(rhs)
        self.t += self.dt
        return self.v

    def run(self, duration_ms, i_inject=None):
        """Return step-end voltages; duration rounds to nearest whole timestep."""
        duration_ms = _number(duration_ms, "duration_ms", 0)
        dt = _number(self.dt, "dt_ms", 0, True)
        V = np.zeros((int(round(duration_ms / dt)), self.m.n))
        for k in range(len(V)):
            V[k] = self.step(i_inject=i_inject)
        return V

    def analytic(self):
        """Uniform-cylinder theory only; not a tapered/branched exact solution."""
        if not np.all(self.m.d == self.m.d[0]) or any(len(c) > 1 for c in self.m.children()):
            raise ValueError("analytic cable quantities require an unbranched uniform diameter")
        if self.g_leak <= 0:
            raise ValueError("analytic cable quantities require positive leak")
        d_cm = self.m.d[0] * 1e-4
        Rm = 1.0 / self.g_leak
        return {"tau_m_ms": Rm * self.Cm * 1e-3,
                "lambda_um": np.sqrt(Rm * d_cm / (4 * self.Ra)) * 1e4,
                "Rm_ohm_cm2": Rm, "d_um": self.m.d[0]}


class Synapse:
    """Exponential conductance, uS; reversal mV; currents nA.

    current/outward_current use g*(V-E); inward_current is its negative for
    CableNeuron.step. Receptor reversal and voltage, not NT label alone, set sign.
    Events are a nonnegative integer count applied AFTER exact exponential decay.
    """
    def __init__(self, g_max_uS=0.001, E_rev_mV=0.0, tau_ms=2.0):
        self.g_max = _number(g_max_uS, "g_max_uS", 0)
        self.E_rev = _number(E_rev_mV, "E_rev_mV")
        self.tau = _number(tau_ms, "tau_ms", 0, True)
        self.g = 0.0

    def step(self, spikes_or_events, dt_ms):
        dt = _number(dt_ms, "dt_ms", 0)
        if not isinstance(spikes_or_events, (bool, int, np.bool_, np.integer)) or spikes_or_events < 0:
            raise ValueError("spikes_or_events must be a nonnegative integer count or bool")
        self.g = self.g * np.exp(-dt / self.tau) + int(spikes_or_events) * self.g_max
        return self.g

    def outward_current(self, v_mV):
        v = np.asarray(v_mV, float)
        if not np.isfinite(v).all():
            raise ValueError("voltage must be finite")
        return self.g * (v - self.E_rev)

    def current(self, v_mV):
        """Backward-compatible alias of outward_current (positive OUTWARD)."""
        return self.outward_current(v_mV)

    def inward_current(self, v_mV):
        return -self.outward_current(v_mV)
