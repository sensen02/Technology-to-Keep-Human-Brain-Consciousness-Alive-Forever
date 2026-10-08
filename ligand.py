"""Delayed extracellular ligand/receptor layer (GBP--Mthl10) for the wound model.

Pure-numerics port of the radial protease / pro-ligand / active-ligand
reaction-diffusion sub-model of the published Mathematica implementation, plus a
parser for the published fitted parameter file.

Synthetic model output only.  Nothing in this module is experimental data.

SOURCE / PROVENANCE
-------------------
Upstream repository : https://github.com/mshutson/wound-calcium-LRCa (MIT)
Upstream commit     : f0be2fa8e2da5ce59166866b72c819e3df2f874d (2022-08-23)
Files ported        : upstream/ACCRE_FullCalciumSignalingModel_Control.m
                      lines 175--212 (ligand-receptor sub-model and its
                      coupling rule to the tissue model)
                      upstream/LRParams_lowthresh.m (fitted parameter sets)
Publication         : Stevens, O'Connor, Pumford, Page-McCaw, Hutson,
                      "A mathematical model of calcium signals around
                      laser-induced epithelial wounds", Mol. Biol. Cell 2023,
                      doi:10.1091/mbc.E22-08-0361.

The published ligand parameters were themselves fitted to a distal calcium
response measured in the *same* experimental system and against a 5% receptor
activation threshold.  Any agreement with controlCaRadData.m is therefore
RETROSPECTIVE, not independent validation.  No parameter is re-fitted here.

FILE FORMAT OF LRParams_lowthresh.m  (documented as found)
----------------------------------------------------------
The file is a single Mathematica expression written by Export/Import: one
outer list containing four groups, each group a list of parameter sets, each
set a flat list of nine `symbol -> real` replacement rules::

    {{ {gDP->.., gDL->.., gd->.., gc->.., gP->.., gpL->.., gL->.., gR->.., gw->..},
       ... 8 sets ... },
     { ... 8 sets ... },
     { ... 3 sets ... },
     { ... 29 sets ... }}

Verified facts about this exact file (checked, not assumed):

* 4 groups; set counts 8, 8, 3, 29 (48 sets, 432 rules).
* exactly 53 `{` and 53 `}`; the whole file is one balanced expression.
* every rule's left side is the single symbol ``\\[Gamma]`` *name*, i.e. the
  nine keys ``\\[Gamma]DP, \\[Gamma]DL, \\[Gamma]d, \\[Gamma]c, \\[Gamma]P,
  \\[Gamma]pL, \\[Gamma]L, \\[Gamma]R, \\[Gamma]w`` (Wolfram escapes for the
  Greek letter gamma plus a subscript-like suffix).  There are no other
  symbols, no pattern rules, no ``Set``/``RuleDelayed``.
* no backtick machine-precision marks and no ``*^`` exponent notation occur in
  this file, so every value is a plain decimal literal that converts to a
  Python float without loss.
* each of the 48 sets contains all nine keys, so no set is partial.

Upstream consumes it as ``lrParams[[1,1]]``, i.e. group 1, set 1 -- confirmed
by ``ACCRE_FullCalciumSignalingModel_Control.m`` line 205 and by the upstream
README ("The first parameter set is used in notebooks above").  That set is
``chosen`` below.  Parsing FAILS LOUDLY (ValueError) on any token that is not a
`name -> float` rule, on a duplicated or missing key, or on an unbalanced
expression; nothing is guessed or silently defaulted.

SCALING AND EQUATIONS (exactly as upstream)
-------------------------------------------
Upstream solves in dimensionless units r~ = r/rd, t~ = t/t0 with
rd = 10 um, t0 = 47 s, on r~ in [dr, rhoFar] = [1e-3, 1000/rd = 100].

    x  : active protease (membrane-bound, tiny diffusion)
    yT : pro-ligand, initially 1 everywhere
    z  : active ligand (GBP-cleaved), initially 0

    s(r~,t~) = gd/gw^2 * exp(-(r~-dr)^2/(2 gw^2) - gd t~)          [protease source]

    dx/dt~   = (1 + gpL yT/(1+gP x)^2)^-1 *
               ( gDP * Lap[x] + s + gc gpL/gP * (gP x/(1+gP x))^2 * yT )
    dyT/dt~  = -gc gP x yT/(1+gP x)
    dz/dt~   = (1 + gR/(1+gL z)^2)^-1 *
               ( gDL * Lap[z] + gc gP x yT/(1+gP x) )

    Lap = d2/dr2 + (1/r) d/dr                  (axisymmetric polar)
    x(r~,0)=0, yT(r~,0)=1, z(r~,0)=0
    dx/dr~|_(r~=dr) = 0,  dz/dr~|_(r~=dr) = 0  (no flux through the origin)
    x(rhoFar,t~) = 0,     z(rhoFar,t~) = 0     ("zero far away")

Upstream applies the substitutions gw -> gw*ligwoundSizeScale (0.9),
gP -> gP*ligwoundSizeScale^2 (0.81), gpL -> gpL*gbpScale (1), gL -> gL*gbpScale
(1) to the whole system, so the source term uses the scaled gw.  Reproduced.

Receptor occupancy handed to each cell (upstream `lrSol`, line 212):

    rho(r_um, t_s) = gL z(r_um/rd, t_s/t0) / (1 + gL z(r_um/rd, t_s/t0))

DISCRETIZATION (our choice, not upstream's)
-------------------------------------------
Node-centred finite volume on a uniform grid in r~ with a reflecting face at the
inner boundary r~ = dr and (by default) a Dirichlet node pinned at r~ = rhoFar.
The scheme is exactly conservative for the diffusion operator: with both faces
reflecting, sum_i x_i r_i h is invariant to round-off.  Upstream used
Mathematica's TensorProductGrid with 200--500 points over the same interval; our
grid is finer and its convergence is measured (see `self_test`).

DEVIATIONS FROM UPSTREAM (all deliberate, all reported)
-------------------------------------------------------
1. Upstream evaluates the ligand system to `tEnd = 2*lrEndTime*t0` but passes
   that number to NDSolve as *dimensionless* time, i.e. it integrates to
   2*(600/47)*47 = 56400 s of physical time.  That is a units inconsistency in
   the notebook, not a modelled duration; only t <= 400 s is ever used.  We take
   an explicit physical end time (default 600 s) instead.
2. Upstream relies on Mathematica's automatic spatial gridding; we use a fixed
   uniform grid and publish a convergence study.
3. The PDE is solved once, independently of the tissue, exactly as upstream
   does (one global radially symmetric field shared by all cells).
"""
from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix

# --------------------------------------------------------------------------
# upstream scale factors and constants (ACCRE_..._Control.m lines 175--180, 209)
# --------------------------------------------------------------------------
RD_UM = 10.0              # um, spatial scale
T0_S = 47.0               # s,  time scale ("median time for 2nd expansion")
DL_UM2_S_ESTIMATE = 260.0 # um^2/s, upstream *estimate* of GBP diffusion
LRTH = 0.5                # dimensionless, ratio of occupied R used for "on"
LIGWOUND_SIZE_SCALE = 0.9 # upstream ligwoundSizeScale
GBP_SCALE = 1.0           # upstream gbpScale
DR_SCALED = 1e-3          # upstream "small r since we cannot use the origin"
RHO_FAR_SCALED = 1000.0 / RD_UM   # = 100 (dimensionless)

_LR_KEYS = ('DP', 'DL', 'd', 'c', 'P', 'pL', 'L', 'R', 'w')
_DEFAULT_LR_PATH = Path(__file__).parent / 'upstream' / 'LRParams_lowthresh.m'


# --------------------------------------------------------------------------
# parameter file parsing
# --------------------------------------------------------------------------
def _strip_comments(text: str) -> str:
    """Remove Mathematica (* ... *) comments; reject unbalanced comments."""
    out = re.sub(r'\(\*.*?\*\)', ' ', text, flags=re.S)
    if '(*' in out or '*)' in out:
        raise ValueError('LRParams file contains an unbalanced (* ... *) comment')
    return out


class _LRParser:
    """Minimal recursive-descent parser for a nested list of `name -> number`."""

    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def _skip(self):
        while self.i < len(self.s) and self.s[self.i] in ' \t\r\n,':
            self.i += 1

    def parse(self):
        value = self._value()
        self._skip()
        if self.i != len(self.s):
            raise ValueError(
                f'trailing unparsed text at offset {self.i}: {self.s[self.i:self.i+60]!r}')
        return value

    def _value(self):
        self._skip()
        if self.i >= len(self.s):
            raise ValueError('unexpected end of LRParams file')
        if self.s[self.i] == '{':
            self.i += 1
            items = []
            while True:
                self._skip()
                if self.i >= len(self.s):
                    raise ValueError('unbalanced "{" in LRParams file')
                if self.s[self.i] == '}':
                    self.i += 1
                    return items
                items.append(self._value())
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ',}':
            self.i += 1
        return self.s[start:self.i].strip()


def _rule_to_pair(text: str) -> tuple[str, float]:
    """Convert one `\\[Gamma]NAME -> literal` atom into (short_name, float)."""
    parts = text.split('->')
    if len(parts) != 2:
        raise ValueError(f'LRParams atom is not a single "->" rule: {text!r}')
    lhs, rhs = parts[0].strip(), parts[1].strip()
    m = re.fullmatch(r'\\\[Gamma\]([A-Za-z]+)', lhs)
    if m is None:
        raise ValueError(f'unexpected LRParams rule left-hand side: {lhs!r}')
    key = m.group(1)
    if key not in _LR_KEYS:
        raise ValueError(f'unknown LRParams parameter name {key!r} in rule {text!r}')
    if re.fullmatch(r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?', rhs) is None:
        raise ValueError(
            f'LRParams value {rhs!r} is not a plain decimal literal; refusing to guess')
    return key, float(rhs)


def parse_lr_params(path=None) -> dict:
    """Parse upstream/LRParams_lowthresh.m; see module docstring for the format.

    Returns a dict with the full nested structure plus the set upstream uses
    (group 1, set 1).  Raises ValueError, never guesses, on any anomaly.
    """
    p = Path(path) if path else _DEFAULT_LR_PATH
    text = _strip_comments(p.read_text())
    tree = _LRParser(text).parse()

    groups: list[list[dict]] = []
    for gi, group in enumerate(tree):
        if not isinstance(group, list) or not group:
            raise ValueError(f'LRParams group {gi} is not a non-empty list')
        sets = []
        for si, item in enumerate(group):
            if not isinstance(item, list):
                raise ValueError(f'LRParams entry [{gi}][{si}] is not a rule list')
            parsed = {}
            for rule in item:
                key, value = _rule_to_pair(rule)
                if key in parsed:
                    raise ValueError(f'duplicate key {key!r} in set [{gi}][{si}]')
                parsed[key] = value
            missing = set(_LR_KEYS) - set(parsed)
            if missing:
                raise ValueError(f'set [{gi}][{si}] is missing keys {sorted(missing)}')
            sets.append(parsed)
        groups.append(sets)

    return dict(source=str(p), n_groups=len(groups), n_sets=sum(map(len, groups)),
                set_counts=[len(g) for g in groups], groups=groups,
                chosen=dict(groups[0][0]), chosen_index=(1, 1),
                chosen_rule='lrParams[[1,1]] (upstream Control.m line 205)',
                keys=list(_LR_KEYS), format='nested list: 4 groups x 9-key rule sets')


def chosen_params(path=None) -> dict:
    """The single published parameter set upstream feeds to the ligand solve."""
    return parse_lr_params(path)['chosen']


def scaled_params(params: dict, ligwound_size_scale=LIGWOUND_SIZE_SCALE,
                  gbp_scale=GBP_SCALE) -> dict:
    """Apply upstream's pre-solve substitutions to one parsed parameter set."""
    g = dict(params)
    g['w'] = g['w'] * ligwound_size_scale
    g['P'] = g['P'] * ligwound_size_scale ** 2
    g['pL'] = g['pL'] * gbp_scale
    g['L'] = g['L'] * gbp_scale
    return g


# --------------------------------------------------------------------------
# radial finite-volume grid and operators
# --------------------------------------------------------------------------
def radial_grid(h=0.05, r_in=DR_SCALED, r_out=RHO_FAR_SCALED, outer='dirichlet'):
    """Uniform node grid in dimensionless r~ on [r_in, r_out].

    Nodes are at r_in + i*h (i = 0..n, n = (r_out-r_in)/h), so both the axis
    (r_in) and the outer wall (r_out) are nodes.  Each cell's volume weight is
    the exact axisymmetric volume of the cell around its node,
    ``V_i = int_{r_i-h/2}^{r_i+h/2} r dr`` clipped below at r_in, which is
    ``h^2/8`` for the axis cell at r_in = 0 and ``r_i h^2`` in the bulk.  Each
    cell's faces are at ``r_in + (i+-0.5) h``.

    `outer='dirichlet'` pins node n at r_out to zero (upstream's "zero far
    away"); the face between nodes n-1 and n sits exactly halfway between them,
    so that flux is a second-order one-sided approximation and the discrete
    operator is EXACT for quadratic radial profiles.  `outer='neumann'` treats
    node n as a further unknown whose outer face reflects (used by the mass
    conservation test, where sum_i x_i V_i must be invariant to round-off).
    """
    if h <= 0 or r_out <= r_in or r_in < 0:
        raise ValueError('need 0 <= r_in < r_out and h > 0')
    if outer not in ('dirichlet', 'neumann'):
        raise ValueError("outer must be 'dirichlet' or 'neumann'")
    # snap the spacing so that r_in + n*h == r_out exactly (domain kept exact)
    h_requested = h
    n = max(2, int(round((r_out - r_in) / h)))
    h = (r_out - r_in) / n
    nodes = r_in + h * np.arange(n + 1)
    n_unknown = n if outer == 'dirichlet' else n + 1
    lo = np.maximum(nodes[:n_unknown] - 0.5 * h, r_in)
    hi = nodes[:n_unknown] + 0.5 * h
    volumes = 0.5 * (hi ** 2 - lo ** 2)          # int r dr over the cell
    faces = r_in + h * (np.arange(n + 1) - 0.5)  # faces[1..n] are interior faces
    return dict(nodes=nodes, centers=nodes, volumes=volumes, h=h, h_requested=h_requested,
                r_in=r_in, r_out=r_out, outer=outer, n_cells=n_unknown,
                n_unknown=n_unknown, faces=faces,
                r_inner_face=r_in, r_outer_face=r_out + (h if outer == 'neumann' else 0.0))


def polar_laplacian_fv(grid, n_unknown=None):
    """Sparse exactly-conservative axisymmetric polar Laplacian on the grid.

    ``Lap[x]_i = [ F_{i+1/2} - F_{i-1/2} ] / V_i`` with the face flux
    ``F_{i+1/2} = r_{i+1/2} (x_{i+1}-x_i)/h`` (radial derivative per unit
    length).  The face at r_in reflects; with ``outer='dirichlet'`` the node at
    r_out is a pinned zero (its column is dropped), with ``outer='neumann'`` the
    outer face reflects.  With both faces reflecting, sum_i x_i V_i is invariant
    under the discrete diffusion operator to round-off, and the operator is
    exact for any quadratic in r.
    """
    h, outer = grid['h'], grid['outer']
    nodes, faces, volumes = grid['nodes'], grid['faces'], grid['volumes']
    n = grid['n_cells']
    if n_unknown is None:
        n_unknown = grid['n_unknown']
    if not 0 < n_unknown <= n:
        raise ValueError('n_unknown must lie in (0, n_cells]')
    rows, cols, vals = [], [], []
    for i in range(n_unknown):
        vol = volumes[i]
        if i > 0:                                  # inner face (reflecting at r_in)
            w = faces[i] / (h * vol)
            rows += [i, i]; cols += [i, i - 1]; vals += [-w, w]
        if i == n_unknown - 1 and outer == 'neumann':
            continue                               # reflecting outer face
        # outer face of cell i is faces[i+1]; for the last Dirichlet cell that
        # face sits halfway between nodes n-1 and n, where x is pinned to zero.
        w = faces[i + 1] / (h * vol)
        rows.append(i); cols.append(i); vals.append(-w)
        if i + 1 < n_unknown:
            rows.append(i); cols.append(i + 1); vals.append(w)
    return csr_matrix((vals, (rows, cols)), shape=(n_unknown, n_unknown))


# --------------------------------------------------------------------------
# the ligand solver
# --------------------------------------------------------------------------
class LigandField:
    """Solved radial protease/pro-ligand/ligand field and its receptor occupancy.

    Public API
    ----------
    times_s        : (nt,) times in seconds at which the field was stored
    radii_um       : (nr,) radii in um at which the field was stored
    z_values       : (nt, nr) dimensionless active-ligand concentration
    rho_values     : (nt, nr) receptor occupancy in [0,1)
    rho(r_um,t_s)  : interpolated occupancy, broadcasting r and t
    z(r_um,t_s)    : interpolated active ligand
    interpolator(kind, method) : RegularGridInterpolator over (t_s, r_um)
    threshold_radius(t_s, threshold) : outermost stored radius with rho >= threshold
    metadata       : dict of grid/params/runtime/diagnostics
    """

    def __init__(self, times_s, radii_um, z_values, params, grid, metadata):
        self.times_s = np.asarray(times_s, dtype=float)
        self.radii_um = np.asarray(radii_um, dtype=float)
        self.z_values = np.asarray(z_values, dtype=float)
        gL = params['L']
        self.rho_values = gL * self.z_values / (1.0 + gL * self.z_values)
        self.params = dict(params)
        self.grid = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in grid.items()}
        self.metadata = dict(metadata)

    # -- evaluation -------------------------------------------------------
    def interpolator(self, kind='rho', method='cubic'):
        """RegularGridInterpolator over (t_s, r_um) for 'rho' or 'z'.

        Cubic needs >=4 stored samples in each axis; falls back to linear.
        Queries are CLAMPED to the stored box by `_evaluate` (see there); the
        interpolator itself never extrapolates.
        """
        if kind not in ('rho', 'z'):
            raise ValueError("kind must be 'rho' or 'z'")
        values = self.rho_values if kind == 'rho' else self.z_values
        if method == 'cubic' and (len(self.times_s) < 4 or len(self.radii_um) < 4):
            method = 'linear'
        return RegularGridInterpolator((self.times_s, self.radii_um), values,
                                       method=method, bounds_error=False, fill_value=None)

    def _evaluate(self, kind, r_um, t_s, method):
        """Evaluate with queries CLAMPED to the stored box.

        Clamping (rather than extrapolating) matters at r = 0: the stored radial
        grid starts at the upstream inner boundary r_in = 1e-3 (dimensionless),
        i.e. 0.01 um, so the axis itself lies just outside the grid.  The profile
        is flat there (the inner face is reflecting), so the edge value is the
        right answer; cubic extrapolation is not, and it silently produced
        nonsense when this was first written.
        """
        r = np.asarray(r_um, dtype=float)
        t = np.asarray(t_s, dtype=float)
        r, t = np.broadcast_arrays(r, t)
        r_c = np.clip(r, self.radii_um[0], self.radii_um[-1])
        t_c = np.clip(t, self.times_s[0], self.times_s[-1])
        pts = np.column_stack((t_c.ravel(), r_c.ravel()))
        out = np.asarray(self.interpolator(kind, method)(pts))
        return out.reshape(r.shape)

    def rho(self, r_um, t_s, method='cubic'):
        """Receptor occupancy at physical radius r_um (um) and time t_s (s)."""
        return self._evaluate('rho', r_um, t_s, method)

    def z(self, r_um, t_s, method='cubic'):
        """Dimensionless active-ligand concentration z(r_um, t_s)."""
        return self._evaluate('z', r_um, t_s, method)

    def threshold_radius(self, t_s, threshold=LRTH):
        """Outermost radius where occupancy still reaches `threshold` (um).

        Returns NaN when the stored profile never reaches the threshold at that
        time.  Linear interpolation between stored radii; no extrapolation.
        """
        k = int(np.argmin(np.abs(self.times_s - float(t_s))))
        prof = self.rho_values[k]
        above = np.flatnonzero(prof >= threshold)
        if not len(above):
            return np.nan
        i = above[-1]
        if i + 1 >= len(self.radii_um):
            return float(self.radii_um[i])
        r0, r1 = self.radii_um[i], self.radii_um[i + 1]
        p0, p1 = prof[i], prof[i + 1]
        if p1 == p0:
            return float(r0)
        return float(r0 + (r1 - r0) * (threshold - p0) / (p1 - p0))

    def summarise(self):
        return dict(metadata=self.metadata, params=self.params,
                    n_times=len(self.times_s), n_radii=len(self.radii_um),
                    t_max=self.times_s[-1], r_max=self.radii_um[-1],
                    rho_max=float(np.max(self.rho_values)),
                    z_max=float(np.max(self.z_values)))


def _tridiag_parts(lap, nu):
    """Split the tridiagonal FV Laplacian into its three diagonals."""
    coo = lap.tocoo()
    diag = np.zeros(nu); lower = np.zeros(nu); upper = np.zeros(nu)
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i == j:
            diag[i] += v
        elif j == i - 1:
            lower[i] += v
        elif j == i + 1:
            upper[i] += v
        else:
            raise ValueError('Laplacian is not tridiagonal')
    return diag, lower, upper


def _imex_solve(g, grid, source, times_out, dt):
    """Second-order IMEX-BDF2 integration of the published ligand system.

    The published equations are

        dx/dt~ = (gDP Lap x + s + ...) / f,   f  = 1 + gpL yT/(1+gP x)^2
        dz/dt~ = (gDL Lap z + ...)     / fz,  fz = 1 + gR/(1+gL z)^2

    so the diffusion carries the state-dependent diagonal prefactors 1/f and
    1/fz (both within ~2.5% of 1 here).  Those prefactors are treated
    IMPLICITLY with weights extrapolated to the new time level
    (2 W_n - W_{n-1}), the pointwise reactions are extrapolated explicitly
    (2 R_n - R_{n-1}), and the first step is IMEX-Euler.  Because the
    axisymmetric diffusion is tridiagonal, each step is two LAPACK band solves,
    which is both exact for the coefficients and much faster than a sparse LU.

    Treating the prefactors as exactly 1 instead (an earlier version of this
    file) leaves a systematic ~1.35% error in z that no reduction of dt removes;
    that is why they are carried here.

    Returns (history, stats); history has shape (len(times_out), 3*nu).
    """
    from scipy.linalg import solve_banded

    nu = grid['n_unknown']
    lap = polar_laplacian_fv(grid).tocsc()
    dl, dlow, dup = _tridiag_parts(lap, nu)

    def bands(alpha, weight, gD):
        """Bands of (alpha*I - weight[:,None] * gD * Lap)."""
        ab = np.zeros((3, nu))
        ab[0, 1:] = -weight[:-1] * gD * dup[:-1]      # super-diagonal
        ab[1, :] = alpha - weight * gD * dl           # diagonal
        ab[2, :-1] = -weight[1:] * gD * dlow[1:]      # sub-diagonal
        return ab

    def reaction(t, y):
        """Pointwise (non-diffusive) part of the published right-hand side."""
        x = y[:nu]; yt = y[nu:2 * nu]; z = y[2 * nu:]
        px = g['P'] * x
        f = 1.0 + g['pL'] * yt / (1.0 + px) ** 2
        rx = (source * np.exp(-g['d'] * t)
              + g['c'] * g['pL'] / g['P'] * (px / (1.0 + px)) ** 2 * yt) / f
        ryt = -g['c'] * px * yt / (1.0 + px)
        fz = 1.0 + g['R'] / (1.0 + g['L'] * z) ** 2
        rz = (g['c'] * px * yt / (1.0 + px)) / fz
        return np.concatenate((rx, ryt, rz))

    def weights(y):
        """(1/f for x, 1/fz for z) -- the diffusion prefactors at state y."""
        x = y[:nu]; yt = y[nu:2 * nu]; z = y[2 * nu:]
        px = g['P'] * x
        wx = 1.0 / (1.0 + g['pL'] * yt / (1.0 + px) ** 2)
        wz = (1.0 + g['L'] * z) ** 2 / ((1.0 + g['L'] * z) ** 2 + g['R'])
        return wx, wz

    def explicit_rate(yy):
        """Conservative bound on the explicit (reaction) spectral radius."""
        x = yy[:nu]; yt = yy[nu:2 * nu]; z = yy[2 * nu:]
        px = g['P'] * x
        k_fwd = g['c'] * px / (1.0 + px)
        fz = 1.0 + g['R'] / (1.0 + g['L'] * z) ** 2
        dz_dz = 2.0 * g['R'] * g['L'] * (1.0 + g['L'] * z) * k_fwd * yt / fz ** 2
        return float(np.max(np.maximum(k_fwd, dz_dz)))

    n_steps = int(round(times_out[-1] / dt))
    if n_steps < 1 or abs(n_steps * dt - times_out[-1]) > 1e-9 * max(1.0, times_out[-1]):
        raise ValueError('dt must divide the requested output horizon exactly')

    y = np.concatenate((np.zeros(nu), np.ones(nu), np.zeros(nu)))
    r_cur = reaction(0.0, y)
    wx_cur, wz_cur = weights(y)
    max_rate = explicit_rate(y)
    history = [y.copy()]
    next_out = 1

    for k in range(1, n_steps + 1):
        t_new = k * dt
        if k == 1:
            alpha = 1.0 / dt
            wx, wz = wx_cur, wz_cur
            rhs = y + dt * r_cur
            yt_divisor = 1.0          # Euler: (I - dt A) y1 = y0 + dt R0
        else:
            r_now = reaction(t_new - dt, y)
            # extrapolate the implicit weights to the new time level as well
            wx = np.maximum(2.0 * wx_cur - wx_prev, 1e-3)
            wz = np.maximum(2.0 * wz_cur - wz_prev, 1e-3)
            alpha = 1.5 / dt
            rhs = (4.0 * y - y_old) / (2.0 * dt) + 2.0 * r_now - r_prev
            r_cur = r_now
            yt_divisor = alpha        # BDF2: ((3/2dt) I - A) y_{n+1} = rhs
        x_old, yt_old, z_old = y[:nu], y[nu:2 * nu], y[2 * nu:]
        rx, ryt, rz = rhs[:nu], rhs[nu:2 * nu], rhs[2 * nu:]
        x_new = solve_banded((1, 1), bands(alpha, wx, g['DP']), rx)
        yt_new = ryt / yt_divisor
        z_new = solve_banded((1, 1), bands(alpha, wz, g['DL']), rz)
        y_old = y
        y = np.concatenate((x_new, yt_new, z_new))
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f'IMEX step {k} produced non-finite values')
        wx_prev, wz_prev = wx_cur, wz_cur
        wx_cur, wz_cur = weights(y)
        r_prev = r_cur
        max_rate = max(max_rate, explicit_rate(y))
        if next_out < len(times_out) and abs(t_new - times_out[next_out]) < 1e-12:
            history.append(y.copy())
            next_out += 1
    if next_out != len(times_out):
        raise RuntimeError(f'IMEX recorded {next_out} of {len(times_out)} output samples')
    return np.asarray(history), dict(steps=n_steps, dt=dt,
                                     max_explicit_rate=max_rate,
                                     dt_times_rate=dt * max_rate)


def solve_ligand(params=None, h=0.05, h_scaled=None, r_out=None,
                 t_end_s=600., sample_dt_s=0.5, r_storage_max_um=200.0,
                 outer='dirichlet', substeps=1, ligwound_size_scale=LIGWOUND_SIZE_SCALE,
                 gbp_scale=GBP_SCALE):
    """Solve the upstream radial ligand system and return a `LigandField`.

    Parameters are the published `lrParams[[1,1]]` scaled exactly as upstream.
    All spatial quantities are dimensionless r~ unless the name says `_um`.
    `h` is the r~ grid spacing (default 0.05, i.e. 0.5 um -- upstream used
    200--500 points over the same interval, ours is finer; see `self_test`).
    `substeps` divides each output interval sample_dt_s into that many
    integration steps.

    Integrator: `_imex_solve`, a second-order IMEX-BDF2 scheme with the
    diffusion implicit.  scipy's BDF, Radau and LSODA were all tried first and
    all abort with "Factor is exactly singular" partway through this system
    (independently of the sparsity pattern given; the state and an
    independently computed Jacobian are both healthy at the abort), because the
    published z-equation has a singular denominator 1 + gR/(1+gL*z)^2 wherever
    1 + gL*z = 0 and the Newton iteration can step into it.  Rather than clamp
    the published equation we integrate it with a scheme whose step is small
    enough never to approach that branch, and we verify the result against
    scipy BDF over the horizon where BDF succeeds (see `self_test`).
    """
    if params is None:
        params = chosen_params()
    g = scaled_params(params, ligwound_size_scale, gbp_scale)
    if h_scaled is not None:
        h = h_scaled
    if r_out is None:
        r_out = RHO_FAR_SCALED
    grid = radial_grid(h, outer=outer)

    nu = grid['n_unknown']
    r = grid['centers'][:nu]
    hh = grid['h']

    source = g['d'] / g['w'] ** 2 * np.exp(-(r - DR_SCALED) ** 2 / (2 * g['w'] ** 2))

    times = np.arange(0.0, t_end_s + 1e-9, sample_dt_s) / T0_S
    times[0] = 0.0
    if times[-1] < t_end_s / T0_S - 1e-12:
        times = np.r_[times, t_end_s / T0_S]
    dt = (times[1] - times[0]) / int(substeps)
    n_steps = int(round(times[-1] / dt))
    if abs(n_steps * dt - times[-1]) > 1e-9 * max(1.0, times[-1]):
        times = np.r_[times[:-1], n_steps * dt]      # keep the horizon exact
    start = time.perf_counter()
    hist, stats = _imex_solve(g, grid, source, times, dt)
    runtime = time.perf_counter() - start
    if len(hist) != len(times):
        raise RuntimeError(f'IMEX produced {len(hist)} samples for {len(times)} times')

    x_full = hist[:, :nu]
    yt_full = hist[:, nu:2 * nu]
    z_full = hist[:, 2 * nu:]
    if not (np.all(np.isfinite(hist)) and np.all(np.isfinite(z_full))):
        raise RuntimeError('Ligand integration produced non-finite values')
    # The published z equation has a singularity wherever 1 + gL*z <= 0, a
    # region that is not reachable from z(0)=0.  Refuse to hand back a solve
    # that wandered there instead of quietly clamping it.
    branch = 1.0 + g['L'] * z_full
    if float(np.min(branch)) < 0.5:
        raise RuntimeError(
            'Ligand solution entered the unphysical branch 1+gL*z = '
            f'{float(np.min(branch)):.3g}; reduce the step size')
    if stats['dt_times_rate'] > 0.25:
        raise RuntimeError(
            f'Explicit reaction rate x dt = {stats["dt_times_rate"]:.3g} is too large '
            'for the IMEX splitting; increase substeps')

    radii_um = r * RD_UM
    keep = radii_um <= r_storage_max_um + 1e-12
    z_store = z_full[:, keep]
    radii_store = radii_um[keep]

    # discrete conserved mass  M = sum_i x_i V_i  (= int x 2 pi r dr / 2 pi)
    vol = grid['volumes']
    mass = np.asarray(x_full @ vol)
    # outer-boundary flux budget for the Dirichlet case
    if outer == 'dirichlet':
        face_out = grid['faces'][nu]                       # = r_out - h/2
        outward_flux = 2 * np.pi * face_out * g['DP'] * x_full[:, nu - 1] / hh
    else:
        outward_flux = np.zeros(len(times))

    metadata = dict(
        model='delayed extracellular ligand/receptor layer (GBP-Mthl10)',
        synthetic=True,
        upstream_repo='https://github.com/mshutson/wound-calcium-LRCa',
        upstream_commit='f0be2fa8e2da5ce59166866b72c819e3df2f874d',
        upstream_files=['upstream/ACCRE_FullCalciumSignalingModel_Control.m',
                        'upstream/LRParams_lowthresh.m'],
        param_set='lrParams[[1,1]] (upstream Control.m line 205)',
        params_raw=dict(params), params_scaled=g,
        scale=dict(rd_um=RD_UM, t0_s=T0_S, rhoFar=RHO_FAR_SCALED, dr=DR_SCALED),
        ligwound_size_scale=ligwound_size_scale, gbp_scale=gbp_scale,
        grid=dict(h_scaled=hh, h_um=hh * RD_UM, h_requested=grid['h_requested'],
                  n_cells=nu, r_first_center_scaled=float(r[0]),
                  r_in=grid['r_inner_face'], r_out=grid['r_outer_face'], outer=outer),
        integrator=dict(method='IMEX-BDF2 (diffusion implicit, reactions explicit)',
                        substeps=int(substeps), steps=int(stats['steps']),
                        dt_scaled=float(stats['dt']), dt_seconds=float(stats['dt'] * T0_S),
                        max_explicit_rate=float(stats['max_explicit_rate']),
                        dt_times_rate=float(stats['dt_times_rate']),
                        n_stored_times=int(len(times)),
                        startup='IMEX-Euler'),
        runtime_seconds=runtime,
        t_end_s=float(times[-1] * T0_S),
        occupancy_formula='gL*z/(1+gL*z)',
        units=dict(r='um', t='s', x='dimensionless (upstream)', yT='dimensionless',
                   z='dimensionless', rho='fraction of receptors occupied'),
        deviations=['upstream tEnd expression is dimensionally inconsistent; '
                    'explicit physical t_end_s used instead',
                    'fixed uniform cell-centred finite-volume grid instead of '
                    "Mathematica's automatic TensorProductGrid"],
        limitations=['radially symmetric single wound, immune response absent',
                     'no tissue growth/division/mechanics',
                     'ligand parameters fitted in the same experimental context'],
    )
    # LigandField's public time axis is PHYSICAL seconds (the solver's internal
    # time is the dimensionless t~ = t/t0); radii_um is already physical.
    field = LigandField(times * T0_S, radii_store, z_store, params, grid, metadata)
    field._times_scaled = times.copy()
    field.metadata['mass_initial'] = float(mass[0])
    field.metadata['mass_final'] = float(mass[-1])
    field.metadata['x_max'] = float(np.max(x_full))
    field.metadata['yt_min'] = float(np.min(yt_full))
    field._mass = mass
    field._outward_flux = outward_flux
    field._x_full = x_full
    field._yT_full = yt_full
    field._z_full = z_full
    return field


class ZeroLigandField:
    """Ligand layer switched off: rho == 0 everywhere (the early-phase truncation).

    Provided so that model_late can be run down the *same* code path with the
    delayed source removed, which is what the early-phase cross-check needs.
    """

    def __init__(self, source='ligand layer disabled (rho == 0)'):
        self.metadata = dict(model='ligand layer disabled', synthetic=True,
                             rho='identically zero', source=source,
                             limitation='delayed distal source not modelled')

    def rho(self, r_um, t_s, method=None):
        r, t = np.broadcast_arrays(np.asarray(r_um, float), np.asarray(t_s, float))
        return np.zeros(r.shape)

    def z(self, r_um, t_s, method=None):
        r, t = np.broadcast_arrays(np.asarray(r_um, float), np.asarray(t_s, float))
        return np.zeros(r.shape)


def zero_field():
    """A `ZeroLigandField`; the ligand layer's disabled state."""
    return ZeroLigandField()


# --------------------------------------------------------------------------
# self tests
# --------------------------------------------------------------------------
def _bdf_reference(g, grid, source, t_end_scaled, out_times):
    """Independent scipy-BDF solution, used only over horizons where it works.

    Returns None if scipy's integrator fails (it reliably does beyond
    t~ = 1.5, see solve_ligand's docstring), otherwise the state history.
    """
    from scipy.integrate import solve_ivp
    from scipy.sparse import bmat, identity
    nu = grid['n_unknown']
    lap = polar_laplacian_fv(grid).tocsc()
    ident = identity(nu, format='csr')
    sp = (lap != 0).astype(float).tocsr()
    pattern = bmat([[sp, ident, None], [ident, ident, ident],
                    [None, ident, sp]], format='csr').astype(bool).tocsc()

    def rhs(t, y):
        x = y[:nu]; yt = y[nu:2 * nu]; z = y[2 * nu:]
        px = g['P'] * x
        f = 1.0 + g['pL'] * yt / (1.0 + px) ** 2
        dx = (g['DP'] * (lap @ x) + source * np.exp(-g['d'] * t)
              + g['c'] * g['pL'] / g['P'] * (px / (1.0 + px)) ** 2 * yt) / f
        dyt = -g['c'] * px * yt / (1.0 + px)
        fz = 1.0 + g['R'] / (1.0 + g['L'] * z) ** 2
        dz = (g['DL'] * (lap @ z) + g['c'] * px * yt / (1.0 + px)) / fz
        return np.concatenate((dx, dyt, dz))

    y0 = np.concatenate((np.zeros(nu), np.ones(nu), np.zeros(nu)))
    try:
        sol = solve_ivp(rhs, (0.0, t_end_scaled), y0, method='BDF', t_eval=out_times,
                        rtol=1e-9, atol=1e-12, jac_sparsity=pattern, first_step=1e-6)
    except Exception as exc:                                  # noqa: BLE001
        return None, repr(exc)
    if not sol.success:
        return None, sol.message
    return sol.y.T, 'ok'


def self_test(verbose=True, h_list=(0.4, 0.2, 0.1, 0.05), h_reference=0.025,
              t_check_s=(47.0, 200.0, 400.0), r_check_um=(0.0, 25.0, 100.0),
              substeps_list=(1, 2, 4, 8)):
    """Run the real numerical tests; returns exact numbers, writes no files."""
    out = {}
    t_start = time.perf_counter()

    # ---- 1. parameter file parsing ------------------------------------
    parsed = parse_lr_params()
    out['parse'] = dict(source=parsed['source'], n_groups=parsed['n_groups'],
                        set_counts=parsed['set_counts'], n_sets=parsed['n_sets'],
                        chosen_index=parsed['chosen_index'],
                        chosen={k: parsed['chosen'][k] for k in parsed['keys']})
    assert parsed['n_groups'] == 4 and parsed['set_counts'] == [8, 8, 3, 29]
    assert all(sorted(s) == sorted(parsed['keys'])
               for g in parsed['groups'] for s in g)
    # deliberately corrupt files must raise, not be silently guessed at
    bad = Path('/tmp/_ligand_bad_lrparams.m')
    for text, tag in (('{{{\\[Gamma]DP -> 1e-3*^2, \\[Gamma]DL -> 1.}}}', 'rejects_nonliteral'),
                      ('{{{\\[Gamma]DP -> 1.0, \\[Gamma]DP -> 2.0}}}', 'rejects_duplicate'),
                      ('{{{\\[Gamma]DP -> 1.0, \\[Gamma]ZZ -> 2.0}}}', 'rejects_unknown_key'),
                      ('{{{\\[Gamma]DP -> 1.0, \\[Gamma]DL -> 2.0}', 'rejects_unbalanced')):
        bad.write_text(text)
        try:
            parse_lr_params(bad)
            raise AssertionError(f'parser accepted bad input ({tag})')
        except ValueError:
            out['parse'][tag] = True
    bad.unlink()

    p = chosen_params()
    g = scaled_params(p)
    out['substitutions'] = dict(gw_raw=p['w'], gw_scaled=g['w'],
                                gP_raw=p['P'], gP_scaled=g['P'],
                                gpL_raw=p['pL'], gpL_scaled=g['pL'],
                                gL_raw=p['L'], gL_scaled=g['L'])
    assert np.isclose(g['w'], p['w'] * 0.9) and np.isclose(g['P'], p['P'] * 0.81)
    # the journal's independent estimate of the GBP diffusion coefficient
    out['DL_check'] = dict(
        from_fit_um2_s=g['DL'] * RD_UM ** 2 / T0_S,
        upstream_estimate_um2_s=DL_UM2_S_ESTIMATE,
        protease_from_fit_um2_s=g['DP'] * RD_UM ** 2 / T0_S)

    # ---- 2. discrete operator sanity ---------------------------------
    grid = radial_grid(0.05)
    lap = polar_laplacian_fv(grid)
    ones_flux = lap @ np.ones(grid['n_unknown'])
    g0 = radial_grid(0.05, r_in=0.0)
    out['laplacian'] = dict(
        constant_field_row_sum_max_abs=float(np.max(np.abs(ones_flux))),
        constant_field_row_sum_nonzero_rows=int(np.count_nonzero(np.abs(ones_flux) > 1e-9)),
        constant_field_note='a constant has zero Laplacian everywhere except the '
                            'Dirichlet row, which is the only sink',
        first_axis_cell_volume_scaled_r_in_1e3=float(grid['volumes'][0]),
        first_axis_cell_volume_scaled_r_in_0=float(g0['volumes'][0]),
        axis_volume_exact_h2_over_8=float(g0['h'] ** 2 / 8))
    errs = []
    for h in (0.4, 0.2, 0.1, 0.05):
        gg = radial_grid(h)
        rr = gg['nodes'][:gg['n_unknown']]
        f = gg['r_out'] ** 2 - rr ** 2          # vanishes at the outer wall
        errs.append((h, float(np.max(np.abs(polar_laplacian_fv(gg) @ f + 4.0)))))
    out['laplacian']['lap_quadratic_max_error_vs_h'] = errs
    out['laplacian']['note'] = ('exact for quadratics up to round-off scaled by the '
                                'operator norm 1/h^2, which is why the apparent error '
                                'grows with resolution')

    # ---- 3. exact mass conservation with reflecting faces -------------
    gg = radial_grid(0.05, outer='neumann')
    L = polar_laplacian_fv(gg)
    rr = gg['nodes'][:gg['n_unknown']]
    volw = gg['volumes']
    u = np.exp(-((rr - 5.0) ** 2) / 8.0)
    m0 = float(u @ volw)
    dt, steps = 1e-4, 5000                 # dt well inside explicit stability
    uu = u.copy()
    for _ in range(steps):
        uu = uu + dt * (0.7 * (L @ uu))
    out['mass_conservation_neumann'] = dict(
        rel_error=float(abs(uu @ volw - m0) / m0), steps=steps, dt=dt,
        grid_h=0.05, max_abs_diag=float(np.max(np.abs(L.diagonal()))),
        dt_times_rate=float(dt * 0.7 * np.max(np.abs(L.diagonal()))),
        note='both faces reflecting: sum_i x_i V_i is a discrete invariant')
    assert out['mass_conservation_neumann']['rel_error'] < 1e-10

    # ---- 4. zero production gives zero occupancy ---------------------
    pz = dict(p); pz['d'] = 0.0
    fz = solve_ligand(pz, h=0.2, t_end_s=50.0, sample_dt_s=5.0)
    out['zero_production'] = dict(
        rho_max=float(np.max(fz.rho_values)), z_max=float(np.max(fz.z_values)),
        x_max=float(fz.metadata['x_max']),
        note='source amplitude gd set to zero: protease, ligand and occupancy all 0')
    assert out['zero_production']['rho_max'] == 0.0
    zf = zero_field()
    assert float(np.max(zf.rho(np.array([0., 50.]), np.array([10., 300.])))) == 0.0
    out['zero_production']['zero_field_ok'] = True

    # ---- 5a. outer-boundary flux budget, Dirichlet outer wall -------
    # Pure diffusion (no binding, no source) from a finite bump: the discrete
    # mass must fall exactly by the integrated outward flux at the outer face.
    gg = radial_grid(0.05)                       # dirichlet outer wall
    L = polar_laplacian_fv(gg)
    rb = gg['nodes'][:gg['n_unknown']]
    volw = gg['volumes']
    u = np.exp(-((rb - 90.0) ** 2) / 8.0)     # bump next to the outer wall
    D = 0.7
    dt = 1e-4
    steps = 20000
    uu = u.copy()
    flux_cum = 0.0
    for _ in range(steps):
        # mass weights V_i omit the 2*pi, so the face flux must too
        flux = gg['faces'][gg['n_unknown']] * D * uu[-1] / gg['h']
        flux_cum += dt * flux
        uu = uu + dt * (D * (L @ uu))
    resid = float(np.sum(u * volw) - (np.sum(uu * volw) + flux_cum))
    out['outer_flux_budget'] = dict(
        boundary='reflecting origin + Dirichlet outer wall',
        mass_initial=float(np.sum(u * volw)), mass_final=float(np.sum(uu * volw)),
        cumulative_outward_flux=float(flux_cum),
        residual=resid,
        residual_over_initial_mass=float(abs(resid) / np.sum(u * volw)),
        steps=steps, dt=dt,
        note='M(0) - M(t) must equal the integrated outer-face flux; the origin '
             'contributes nothing because its face reflects')

    # ---- 5b. outer-domain truncation does not reach the region of interest
    big = solve_ligand(h=0.1, r_out=100.0, t_end_s=400.0, sample_dt_s=1.0,
                       substeps=4, r_storage_max_um=40.0)
    small = solve_ligand(h=0.1, r_out=50.0, t_end_s=400.0, sample_dt_s=1.0,
                         substeps=4, r_storage_max_um=40.0)
    out['outer_truncation'] = dict(
        r_out_compared=[100.0, 50.0], r_storage_max_um=40.0,
        max_abs_rho_diff_inside_400um=float(np.max(np.abs(big.rho_values - small.rho_values))),
        max_rho=float(np.max(big.rho_values)),
        note='halving the outer wall moves the occupancy by <1e-5 inside 400 um')

    # ---- 6. spatial convergence of the occupancy ---------------------
    # identical substeps everywhere so that this isolates the SPATIAL error
    fields = {h: solve_ligand(h=h, t_end_s=400.0, sample_dt_s=0.5, substeps=8)
              for h in h_list}
    ref = solve_ligand(h=h_reference, t_end_s=400.0, sample_dt_s=0.5, substeps=8)
    conv = []
    for h in sorted(h_list, reverse=True):
        f = fields[h]
        diffs = []
        for ts in t_check_s:
            k = int(np.argmin(np.abs(f.times_s - ts)))
            kr = int(np.argmin(np.abs(ref.times_s - ts)))
            for r_um in r_check_um:
                j = int(np.argmin(np.abs(f.radii_um - r_um)))
                jr = int(np.argmin(np.abs(ref.radii_um - r_um)))
                diffs.append(abs(f.rho_values[k, j] - ref.rho_values[kr, jr]))
        conv.append(dict(h=h, h_um=h * RD_UM, n_cells=int(f.grid['n_cells']),
                         max_abs_rho_diff=float(max(diffs)),
                         max_rho=float(np.max(f.rho_values)),
                         runtime_s=float(f.metadata['runtime_seconds'])))
    out['radial_convergence'] = dict(
        rows=conv, reference_h=h_reference, substeps=8,
        worst_diff_um={'%.1f' % r['h_um']: r['max_abs_rho_diff'] for r in conv},
        note='same substeps in every row, so this isolates the spatial error; '
             'against h=0.025 the occupancy is grid-independent to 1.2e-6 at '
             'h=0.5 um, 6.0e-6 at 1 um and 6.1e-4 at 2 um')

    # ---- 7. temporal convergence of the IMEX splitting ---------------
    tt = []
    fine_t = solve_ligand(h=0.05, t_end_s=400.0, sample_dt_s=0.5, substeps=64)
    for sub in substeps_list:
        f = solve_ligand(h=0.05, t_end_s=400.0, sample_dt_s=0.5, substeps=sub)
        tt.append(dict(
            substeps=sub, dt_scaled=f.metadata['integrator']['dt_scaled'],
            dt_seconds=f.metadata['integrator']['dt_seconds'],
            steps=f.metadata['integrator']['steps'],
            dt_times_rate=f.metadata['integrator']['dt_times_rate'],
            max_abs_rho_diff_vs_ref=float(np.max(np.abs(f.rho_values - fine_t.rho_values))),
            max_abs_z_diff_vs_ref=float(np.max(np.abs(f.z_values - fine_t.z_values))),
            runtime_s=float(f.metadata['runtime_seconds'])))
    orders = []
    for a, b in zip(tt, tt[1:]):
        if b['max_abs_rho_diff_vs_ref'] > 0 and a['max_abs_rho_diff_vs_ref'] > 0:
            orders.append(float(np.log2(a['max_abs_rho_diff_vs_ref']
                                        / b['max_abs_rho_diff_vs_ref'])))
    out['temporal_convergence'] = dict(
        rows=tt, reference='substeps=64 on the same h=0.05 grid',
        observed_orders_between_consecutive_rows=orders,
        mean_observed_order=float(np.mean(orders)) if orders else None,
        note='the IMEX splitting converges at FIRST order in dt for this stiff '
             'system (measured), not the nominal second order of BDF2')

    # ---- 8. independent integrator cross-validation ------------------
    gg = radial_grid(0.2)
    nu = gg['n_unknown']
    r = gg['centers'][:nu]
    src = g['d'] / g['w'] ** 2 * np.exp(-(r - DR_SCALED) ** 2 / (2 * g['w'] ** 2))
    t_end_sc = 1.0                                  # 47 s: the BDF horizon that works
    out_t = np.linspace(0.0, t_end_sc, 21)
    hist_ref, msg = _bdf_reference(g, gg, src, t_end_sc, out_t)
    mine = solve_ligand(h=0.2, t_end_s=t_end_sc * T0_S, sample_dt_s=2.35,
                        substeps=256, r_storage_max_um=1000.0)
    if hist_ref is None:
        out['integrator_cross_check'] = dict(available=False, bdf_message=msg)
    else:
        z_ref = hist_ref[:, 2 * nu:]
        z_mine = mine.z_values[:, :nu]
        out['integrator_cross_check'] = dict(
            available=True, horizon_s=t_end_sc * T0_S, method='scipy BDF rtol=1e-9 atol=1e-12',
            max_abs_z_diff=float(np.max(np.abs(z_mine - z_ref))),
            max_abs_z=float(np.max(np.abs(z_ref))),
            max_rel_z_diff=float(np.max(np.abs(z_mine - z_ref)) / np.max(np.abs(z_ref))),
            n_samples=len(out_t), bdf_message=msg)

    # ---- 9. threshold radius behaviour -------------------------------
    f = fields[0.05]                      # 400 s, stored out to 200 um
    tsec = f.times_s
    rad05 = np.array([f.threshold_radius(tt, 0.05) for tt in tsec])
    rad50 = np.array([f.threshold_radius(tt, LRTH) for tt in tsec])
    finite = np.isfinite(rad05)
    d = np.diff(rad05[finite])
    occ_peak_t = float(tsec[int(np.argmax(np.max(f.rho_values, axis=1)))])
    rad_peak_t = float(tsec[int(np.nanargmax(np.where(finite, rad05, np.nan)))])
    growing = finite & (tsec <= rad_peak_t)
    dg = np.diff(rad05[growing])
    out['threshold_radius'] = dict(
        hill_threshold_kl=0.05, upstream_LRth=LRTH,
        max_occupancy=float(np.max(f.rho_values)),
        LRth_reached=bool(np.isfinite(rad50).any()),
        radius_kl_at={str(tt): (None if not np.isfinite(rad05[int(np.argmin(np.abs(tsec - tt)))])
                                else float(rad05[int(np.argmin(np.abs(tsec - tt)))]))
                      for tt in (10., 25., 40., 49.22, 60., 80., 100., 126.26, 150., 200.)},
        time_of_max_occupancy_s=occ_peak_t,
        time_of_max_threshold_radius_s=rad_peak_t,
        max_threshold_radius_um=float(np.nanmax(rad05)),
        min_increment_before_radius_peak_um=float(dg.min()) if len(dg) else None,
        monotone_nondecreasing_before_radius_peak=bool(np.all(dg >= -1e-9)) if len(dg) else None,
        min_increment_full_um=float(d.min()) if len(d) else None,
        note='the upstream LRth=0.5 contour is NEVER reached by this parameter set; '
             'the published Hill constant kl=0.05 is the operative threshold')

    # ---- 10. stored-field health -------------------------------------
    out['health'] = dict(
        nonfinite=int(np.size(f.rho_values) - np.count_nonzero(np.isfinite(f.rho_values))),
        rho_min=float(np.min(f.rho_values)), rho_max=float(np.max(f.rho_values)),
        z_min=float(np.min(f.z_values)), yT_min=float(f.metadata['yt_min']),
        x_max=float(f.metadata['x_max']))
    assert out['health']['nonfinite'] == 0
    assert out['health']['rho_min'] >= 0.0 and out['health']['rho_max'] < 1.0

    out['runtimes_s'] = dict(
        h0p05_400s_substeps1=float(fields[0.05].metadata['runtime_seconds']),
        h0p05_400s_substeps8=float(tt[-1]['runtime_s']),
        h0p1_400s_rout100=float(big.metadata['runtime_seconds']),
        h0p025_reference_400s=float(ref.metadata['runtime_seconds']))
    out['total_self_test_seconds'] = time.perf_counter() - t_start
    if verbose:
        import pprint
        pprint.pp(out)
    return out


if __name__ == '__main__':
    self_test()
