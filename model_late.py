"""Late-phase tissue model: microtear damage PLUS the delayed GBP--Mthl10 source.

Synthetic model output only.  Nothing here is experimental data.

This module is the late-time companion to `model.py`.  It keeps that file's
published parameter set, hexagonal geometry, gap-junction operator, spreadsheet
microtear damage profile and ablation handling unchanged, and re-enables the
term that `model.py` deliberately truncates to zero:

    rh_i = alpha_i * (c_i/(Kc+c_i)) * (rho_i^nl/(kl^nl + rho_i^nl) + alpha0)

with `rho_i` the receptor occupancy at cell i's own distance from the wound
centre, taken from the radial ligand/receptor solution in `ligand.py`.  This is
upstream's `lrSol[r,t]` coupling (see ACCRE_FullCalciumSignalingModel_Control.m
lines 40--41 and 212, and `{r -> distances[[i]]}` at line 355).

PROVENANCE
----------
Upstream repository : https://github.com/mshutson/wound-calcium-LRCa (MIT)
Upstream commit     : f0be2fa8e2da5ce59166866b72c819e3df2f874d (2022-08-23)
Publication         : Stevens, O'Connor, Pumford, Page-McCaw, Hutson, "A
                      mathematical model of calcium signals around laser-induced
                      epithelial wounds", Mol. Biol. Cell 2023,
                      doi:10.1091/mbc.E22-08-0361.

The published parameter set (both the calcium parameters in `model.PARAMETERS`
and the ligand parameters in `LRParams_lowthresh.m`) was fitted in this same
experimental system.  Comparison against `upstream/controlCaRadData.m` is
therefore RETROSPECTIVE, never independent validation.  Nothing here is fitted
to that trace; every number is used exactly as published.

WHAT IS NOT MODELLED
--------------------
No tissue growth, no cell division, no cell death beyond the imposed ablations,
no mechanics/tissue motion/wound contraction, no immune or haemolymph response,
no ATP/energy metabolism, no receptor trafficking or desensitisation beyond the
published gR term, no feedback of calcium onto ligand release, no extracellular
geometry (3D or advective transport), no cell-shape change, no junction
remodelling, no anaesthetic/temperature effects.  The domain is a fixed
sealed hexagonal patch, so late waves are bounded by the patch edge.

BOUNDARY / DOMAIN
-----------------
`hex_geometry` produces an n_side x n_side parallelogram patch with an
inscribed circle of radius ~ (n_side-1)*spacing/2.  `measurement.measure` only
returns radii whose annuli are complete, so the usable radius is roughly that
inscribed radius.  Randomness is coordinate-keyed exactly as in `model.py`, so
cells shared between nested grids of size n_side and n_side+4 receive identical
draws.
"""
from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import ligand as ligand_layer
from model import (PARAMETERS, _sparsity, damage_profile, fit_microtear,
                   gap_operator, hex_geometry)

__all__ = ['run', 'cross_check_early', 'radius_series', 'early_pattern',
           'experimental_control_radius', 'self_test', 'LIGAND_DEFAULTS']

_EXPERIMENTAL_PATH = Path(__file__).parent / 'upstream' / 'controlCaRadData.m'


def experimental_control_radius(path=None):
    """EXPERIMENTAL control wound trace: (time_s, half-height radius_um).

    Parsed from upstream/controlCaRadData.m, which is a plain Mathematica list of
    `{time, radius}` pairs plus a one-line generator comment.  This is the ONLY
    experimental series available and it is a single control wound, so it is a
    retrospective reference, not a validation set.  Returned separately from
    every simulated quantity and never mixed into model output.
    """
    p = Path(path) if path else _EXPERIMENTAL_PATH
    text = p.read_text()
    body = text[text.index('{{'):]
    nums = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?',
                                         body.replace('`', ''))]
    if len(nums) % 2:
        raise ValueError('controlCaRadData.m did not parse into time/radius pairs')
    arr = np.asarray(nums, dtype=float).reshape(-1, 2)
    if np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError('controlCaRadData.m time column is not increasing')
    return dict(t_s=arr[:, 0], radius_um=arr[:, 1], source=str(p),
                n_points=len(arr), experimental=True,
                note='single control wound trace; retrospective reference only')

# Ligand solve settings used by `ligand='full'`.  h is dimensionless (units of
# rd = 10 um); substeps divides each 0.5 s output interval of the ligand solve.
LIGAND_DEFAULTS = dict(h=0.05, t_end_s=600., sample_dt_s=0.5, substeps=16,
                       r_storage_max_um=200.0)
_FIELD_CACHE: dict = {}


def _ligand_field(spec, ligand_kwargs=None):
    """Return a ligand field object: a solved `LigandField` or the zero field.

    `spec` is 'full' (default), 'off'/'none'/False (rho == 0 everywhere, the
    early-phase truncation), or an already constructed field object.
    """
    if spec in (None, 'off', 'none', False):
        return ligand_layer.zero_field()
    if not isinstance(spec, str):
        return spec                       # already a field-like object
    if spec != 'full':
        raise ValueError("ligand must be 'full', 'off', False, or a field object")
    kw = dict(LIGAND_DEFAULTS)
    if ligand_kwargs:
        kw.update(ligand_kwargs)
    key = tuple(sorted((k, v) for k, v in kw.items()))
    if key not in _FIELD_CACHE:
        _FIELD_CACHE[key] = ligand_layer.solve_ligand(**kw)
    return _FIELD_CACHE[key]


class _RhoSource:
    """Per-cell receptor occupancy as a smooth function of time.

    The field is sampled once on the cell radii (a (n_times, n_cells) table) and
    then evaluated with a cubic spline in time, so the tissue right-hand side
    stays C2 in t and the stiff solver never sees an interpolant kink.
    """

    def __init__(self, field, radius, method='cubic'):
        self.field = field
        self.zero = isinstance(field, ligand_layer.ZeroLigandField)
        self.radius = np.asarray(radius, dtype=float)
        if self.zero:
            self.times = None
            self.spline = None
            self.table_max = 0.0
            return
        t = np.asarray(field.times_s, dtype=float)
        tab = np.asarray(field.rho(self.radius[None, :], t[:, None], method='linear'),
                         dtype=float)
        self.times = t
        self.table_max = float(np.max(np.abs(tab)))
        order = np.argsort(t)
        self.spline = CubicSpline(t[order], tab[order], axis=0, extrapolate=False)
        self.n_samples = len(t)

    def __call__(self, t):
        if self.zero:
            return np.zeros(self.radius.shape)
        tt = float(t)
        if tt < self.times[0] - 1e-9 or tt > self.times[-1] + 1e-9:
            raise ValueError(
                f'ligand field covers t in [{self.times[0]}, {self.times[-1]}] s '
                f'but the tissue model asked for t = {tt}')
        v = self.spline(np.clip(tt, self.times[0], self.times[-1]))
        return np.nan_to_num(v, nan=0.0)


def _rhs_factory(p, operator, alpha, bx, damage, ablated, rho_of_t):
    """Tissue right-hand side with the delayed receptor-occupancy source.

    Identical to `model._rhs_factory` except that the IP3 production rate uses
    the cell's own receptor occupancy.  With rho == 0 the two are bitwise
    identical, which is what `cross_check_early` verifies.
    """
    n = len(alpha)
    nl = p['nl']
    kl_nl = p['kl'] ** nl

    def rhs(t, y):
        c, er, ip3, h = y.reshape(4, n)
        rho = rho_of_t(t)
        rho_nl = rho ** nl
        # rho == 0 gives exactly 0.0 here, so rh reduces bitwise to
        # alpha*c/(Kc+c)*alpha0, i.e. the truncation used by model.py.
        rh = alpha*c/(p['Kc']+c)*(rho_nl/(kl_nl+rho_nl) + p['alpha0'])
        zeta = p['d2']*(ip3+p['d1'])/(ip3+p['d3'])
        dh = p['a2']*(zeta-(zeta+c)*h)
        m = ip3/(p['d1']+ip3)*c/(p['d5']+c)
        j_ipr = p['etaIPR']*m**3*h**3*(er-c)
        j_serca = p['etaSERCA']*(c**p['nSERCA']-p['kSERCA1']**p['nSERCA']*p['kSERCA2']**p['nSERCA']*er**p['nSERCA'])/(p['kSERCA1']**p['nSERCA']+c**p['nSERCA'])
        j_er = j_ipr+p['eta_lkER']*(er-c)-j_serca
        heal = np.exp(-t/p['tau_heal'])*(-np.expm1(-t/p['tau_d'])) if t >= 0 else 0.
        tear = p['r_microtear']*damage*(1-c/p['cExt'])*heal
        pmca = p['rPMCA']*c**p['nPMCA']/(p['kPMCA']**p['nPMCA']+c**p['nPMCA'])
        ncx = c**p['nNCX']/(p['kNCX']**p['nNCX']+c**p['nNCX'])
        soc = p['rSOC']*p['kSOC']**p['nSOC']/(p['kSOC']**p['nSOC']+er**p['nSOC'])
        pm = p['etaNCX']*(tear+p['rlkPM']*(1-c/p['cExt'])+soc-pmca-ncx)
        beta = 1/(1+p['Ke']*p['Be']/(p['Ke']+c)**2 + p['nx']*p['Kx']**p['nx']*bx*c**(p['nx']-1)/(p['Kx']**p['nx']+c**p['nx'])**2)
        dc = beta*(j_er+pm+p['etaGJc']*(operator@c))
        dc[ablated] = 0.
        return np.concatenate((dc, -p['epsilon']*j_er,
                               rh-p['kdeg']*ip3+p['etaGJIP3']*(operator@ip3), dh))
    return rhs


def run(n_side=41, seed=2025, gj_scale=1., wound=True, rtol=1e-6, t_end=300.,
        *, atol=1e-9, method='BDF', sample_dt=0.5, times=None,
        equilibration_time=300., heterogeneity=True, parameters=None,
        spreadsheet=None, plc_scale=1., damage_override=None,
        ablated_override=None, ligand='full', ligand_kwargs=None):
    """Integrate the wounded tissue WITH the delayed distal source.

    Same arguments, geometry, seed convention and return keys as `model.run`,
    plus `rho` (per-cell occupancy at the stored times) and the ligand
    diagnostics.  `ligand='off'` reproduces `model.run` exactly.
    Any t_end is allowed (model.py caps at 25 s); the ligand field must cover it.
    """
    start = time.perf_counter()
    if t_end <= 0:
        raise ValueError('t_end must be positive')
    if gj_scale < 0 or sample_dt <= 0 or equilibration_time <= 0:
        raise ValueError('gj_scale must be >=0; time intervals must be positive')
    if method not in ('BDF', 'Radau'):
        raise ValueError('Use sparse stiff BDF or Radau')
    p = PARAMETERS.copy()
    if parameters:
        unknown = set(parameters)-set(p)
        if unknown:
            raise ValueError(f'Unknown parameters: {unknown}')
        p.update(parameters)
    geo = hex_geometry(n_side, area=p['reference_area'])
    n = len(geo['areas'])
    radius = np.linalg.norm(geo['positions'], axis=1)
    ablated = (radius < p['ablated_radius']) & bool(wound)
    if ablated_override is not None:
        ablated = np.asarray(ablated_override, dtype=bool).copy()
        if ablated.shape != (n,):
            raise ValueError(f'ablated_override must have shape ({n},)')

    # coordinate-keyed random streams (identical mechanism to model.run)
    side = np.sqrt(2*p['reference_area']/(3*np.sqrt(3)))
    keys = np.rint(geo['positions']/[np.sqrt(3)*side/4, 1.5*side/2]).astype(np.int64)
    if heterogeneity:
        z = np.array([np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(x) & 0xffffffff, int(y) & 0xffffffff])).normal(size=3)
                      for x, y in keys])
        gj = np.exp(-.4**2/2+.4*z[:, 0])
        plc = np.exp(-.5*np.log1p(.2)+np.sqrt(np.log1p(.2))*z[:, 1])
        gcamp = np.exp(-.1**2/2+.1*z[:, 2])
    else:
        gj, plc, gcamp = np.ones((3, n))
    if plc_scale < 0:
        raise ValueError('plc_scale must be nonnegative')
    alpha, bx = p['alpha']*plc*plc_scale, p['Bx']*gcamp
    op_kw = dict(cell_multipliers=gj, gj_scale=gj_scale,
                 connect_ablated=p['connect_ablated'], reference_area=p['reference_area'])
    op_eq = gap_operator(geo['areas'], geo['edges'], geo['edge_lengths'], **op_kw)
    pattern = _sparsity(op_eq)
    empty = np.zeros(n, dtype=bool)

    field = _ligand_field(ligand, ligand_kwargs)
    rho_src = _RhoSource(field, radius)
    zero_rho = np.zeros(n)

    # Equilibration uses rho == 0, exactly as upstream ("deqsTissueEQ = deqns /.
    # {rMuT->0., lr->0.}"), so the pre-wound baseline is identical to model.py.
    eq_rhs = _rhs_factory(p, op_eq, alpha, bx, np.zeros(n), empty,
                          lambda t: zero_rho)
    initial = np.repeat([.1, 200., .01, .67], n)
    eq_start = time.perf_counter()
    eq = solve_ivp(eq_rhs, (0, equilibration_time), initial, method=method,
                   rtol=rtol, atol=atol, jac_sparsity=pattern, t_eval=[equilibration_time])
    if not eq.success:
        raise RuntimeError('Equilibration failed: ' + eq.message)
    eq_seconds = time.perf_counter()-eq_start
    baseline = eq.y[:, -1].reshape(4, n).copy()
    y0 = baseline.copy()
    y0[0, ablated] = p['cExt']
    if damage_override is None:
        damage = damage_profile(radius, p['wound_radius'], spreadsheet) if wound else np.zeros(n)
    else:
        damage = np.asarray(damage_override, dtype=float).copy()
        if damage.shape != (n,):
            raise ValueError(f'damage_override must have shape ({n},)')
        if np.any(damage < 0) or not np.all(np.isfinite(damage)):
            raise ValueError('damage_override must be finite and nonnegative')
    op = gap_operator(geo['areas'], geo['edges'], geo['edge_lengths'],
                      ablated=ablated, **op_kw)
    rhs = _rhs_factory(p, op, alpha, bx, damage, ablated, rho_src)
    if times is None:
        times = np.unique(np.r_[np.arange(0, t_end, sample_dt), t_end])
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > t_end:
        raise ValueError('times must be increasing and within [0,t_end]')
    if not rho_src.zero and times[-1] > rho_src.times[-1] + 1e-9:
        raise ValueError(
            f'ligand field ends at {rho_src.times[-1]:.3g} s but t_end = {times[-1]:.3g} s')
    solve_start = time.perf_counter()
    sol = solve_ivp(rhs, (0, t_end), y0.ravel(), method=method, rtol=rtol, atol=atol,
                    jac_sparsity=pattern, t_eval=times, first_step=min(1e-5, t_end))
    if not sol.success:
        raise RuntimeError('Wound integration failed: ' + sol.message)
    solve_seconds = time.perf_counter()-solve_start
    values = sol.y.T.reshape(len(times), 4, n)
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Nonfinite solution')
    rho_t = np.array([rho_src(t) for t in times])
    metadata = dict(
        model='microtear + delayed ligand/receptor model (late phase)', synthetic=True,
        upstream='upstream/ACCRE_FullCalciumSignalingModel_Control.m',
        upstream_commit='f0be2fa8e2da5ce59166866b72c819e3df2f874d',
        n_cells=n, seed=seed, gj_scale=gj_scale, plc_scale=plc_scale, wound=bool(wound),
        parameters=p, method=method, rtol=rtol, atol=atol,
        equilibration_time=equilibration_time,
        equilibration_max_abs_rhs=float(np.max(np.abs(eq_rhs(equilibration_time, eq.y[:, -1])))),
        equilibration_seconds=eq_seconds, integration_seconds=solve_seconds,
        total_seconds=time.perf_counter()-start, nfev=sol.nfev, njev=sol.njev, nlu=sol.nlu,
        damage_fit=fit_microtear(spreadsheet) if (wound and damage_override is None) else None,
        damage_source='external mechanics override (synthetic, uncalibrated)' if damage_override is not None else ('published erfc/spreadsheet damage profile (prescribed)' if wound else 'none'),
        heterogeneity=heterogeneity, boundary='sealed/no-flux',
        ligand=('disabled (rho == 0, model.py truncation)' if rho_src.zero else
                field.metadata.get('model', 'ligand field')),
        ligand_metadata=(None if rho_src.zero else field.metadata),
        rho_max=float(np.max(rho_t)), rho_at_t0_max=float(np.max(rho_t[0])),
        units=dict(times='s', c='uM', er='uM per ER volume', ip3='uM',
                   h='dimensionless', rho='fraction of receptors occupied',
                   positions='um', areas='um^2'),
        limitations=['fixed regular geometry, no mechanics',
                     'finite sealed domain; test larger n_side',
                     'ablated c is an imposed 1000 uM reservoir',
                     'no division, growth, immune response or junction remodelling',
                     'published parameters fitted in the same experimental context'],
    )
    return dict(times=sol.t, t=sol.t, c=values[:, 0], er=values[:, 1],
                ip3=values[:, 2], h=values[:, 3], rho=rho_t,
                baseline_c=baseline[0].copy(), gcamp=gcamp, **geo,
                metadata=metadata, baseline=baseline, ablated=ablated,
                damage=damage, gj_multipliers=gj, plc_multipliers=plc,
                gcamp_multipliers=gcamp, radius=radius, ligand_field=field)


# --------------------------------------------------------------------------
# comparison helpers (no plots, no files)
# --------------------------------------------------------------------------
def cross_check_early(n_side=20, seed=2025, t_end=25., rtol=1e-6, sample_dt=.1,
                      **kwargs):
    """Compare model_late (ligand off, and ligand on) with model.run.

    Returns max and RMS differences in c (uM), plus the same for ip3, over the
    early window.  The 'off' comparison must be essentially exact; the 'on'
    comparison measures how much the delayed source contributes before t_end.
    """
    import model as early_model
    ref = early_model.run(n_side=n_side, seed=seed, t_end=t_end, rtol=rtol,
                          sample_dt=sample_dt, **kwargs)
    out = dict(n_side=n_side, seed=seed, t_end=t_end, n_cells=ref['metadata']['n_cells'],
               max_radius_um=float(np.max(np.linalg.norm(ref['positions'], axis=1))))
    for tag, spec in (('ligand_off', 'off'), ('ligand_on', 'full')):
        got = run(n_side=n_side, seed=seed, t_end=t_end, rtol=rtol,
                  sample_dt=sample_dt, ligand=spec, **kwargs)
        assert np.array_equal(got['ablated'], ref['ablated'])
        assert np.allclose(got['positions'], ref['positions'])
        d = got['c']-ref['c']
        di = got['ip3']-ref['ip3']
        out[tag] = dict(
            c_max_abs=float(np.max(np.abs(d))), c_rms=float(np.sqrt(np.mean(d**2))),
            c_ref_scale=float(np.max(np.abs(ref['c']-ref['baseline'][0]))),
            ip3_max_abs=float(np.max(np.abs(di))),
            ip3_rms=float(np.sqrt(np.mean(di**2))),
            baseline_c_max_abs=float(np.max(np.abs(got['baseline'][0]-ref['baseline'][0]))),
            bitwise_identical=bool(np.array_equal(got['c'], ref['c'])),
            runtime_seconds=got['metadata']['total_seconds'])
        # normalised difference, in units of the model's own early signal
        scale = max(out[tag]['c_ref_scale'], 1e-30)
        out[tag]['c_max_rel_to_early_signal'] = out[tag]['c_max_abs']/scale
    return out


def radius_series(result, measurement_module=None):
    """Measured annular half-height radius for a run, using measurement.measure."""
    if measurement_module is None:
        import measurement as measurement_module
    return measurement_module.measure(result['c'], result['baseline_c'],
                                      result['polygons'], result['gcamp'],
                                      result['ablated'])


def early_pattern(result, measured, t_early=30., t_trough=(30., 70.),
                  t_late=(70., None)):
    """Describe the (a) first expansion (b) recession (c) second expansion.

    Reads the MEASURED radius series, ignoring censored samples, and reports the
    peaks and trough with their times plus whether the late peak exceeds the
    early one.  Purely descriptive; no fitting.
    """
    t = np.asarray(result['times'], dtype=float)
    r = np.asarray(measured['radius'], dtype=float)
    good = np.isfinite(r)
    out = dict(n_samples=int(len(t)), n_usable=int(good.sum()),
               censored_after_first=int(np.sum(~good)))
    if not good.any():
        return dict(**out, error='every sample censored')
    te = t <= t_early
    e = te & good
    if e.any():
        i = np.argmax(r[e])
        out['early_peak_radius_um'] = float(r[e][i])
        out['early_peak_time_s'] = float(t[e][i])
    lo, hi = t_trough
    m = (t >= lo) & (t < (hi if hi is not None else np.inf)) & good
    if m.any():
        j = np.argmin(r[m])
        out['trough_radius_um'] = float(r[m][j])
        out['trough_time_s'] = float(t[m][j])
    lo = t_late[0]
    m = (t >= lo) & good
    if m.any():
        k = np.argmax(r[m])
        out['late_peak_radius_um'] = float(r[m][k])
        out['late_peak_time_s'] = float(t[m][k])
    if 'trough_radius_um' in out and 'early_peak_radius_um' in out:
        out['shows_recession'] = bool(out['trough_radius_um'] < out['early_peak_radius_um'] - 0.5)
    if 'late_peak_radius_um' in out and 'trough_radius_um' in out:
        out['shows_second_expansion'] = bool(
            out['late_peak_radius_um'] > out['trough_radius_um'] + 0.5)
    out['radius_at'] = {str(x): float(r[int(np.argmin(np.abs(t-x)))])
                        for x in (19.26, 25.68, 49.22, 126.26)
                        if np.isfinite(r[int(np.argmin(np.abs(t-x)))])}
    return out


# --------------------------------------------------------------------------
# self tests
# --------------------------------------------------------------------------
def self_test(verbose=True, late_n_side=45, boundary_n_sides=(37, 41, 45),
              t_end_late=300., seed=2025):
    """Run the actual required tests. Returns numbers; writes no files."""
    out = {}
    t_start = time.perf_counter()

    # ---- 1. ligand layer: resolution, mass and sanity -----------------
    out['ligand'] = ligand_layer.self_test(verbose=False)

    # ---- 2. early-phase reproduction of model.run ---------------------
    out['cross_check_early'] = cross_check_early(n_side=20, seed=seed, t_end=25.)
    assert out['cross_check_early']['ligand_off']['c_max_abs'] < 1e-12, \
        'ligand-off run does not reproduce model.run'

    # ---- 3. full late run at a domain big enough for the late radius ---
    late = run(n_side=late_n_side, seed=seed, t_end=t_end_late, sample_dt=0.5,
               ligand='full')
    measured = radius_series(late)
    out['late_run'] = dict(
        n_side=late_n_side, n_cells=late['metadata']['n_cells'],
        cell_area_um2=float(late['areas'][0]),
        max_cell_radius_um=float(np.max(late['radius'])),
        max_measurable_radius_um=float(measured['max_complete_radius']),
        total_seconds=late['metadata']['total_seconds'],
        integration_seconds=late['metadata']['integration_seconds'],
        equilibration_seconds=late['metadata']['equilibration_seconds'],
        ligand_seconds=late['metadata']['ligand_metadata']['runtime_seconds'],
        ligand_grid=late['metadata']['ligand_metadata']['grid'],
        nfev=late['metadata']['nfev'], njev=late['metadata']['njev'],
        nlu=late['metadata']['nlu'], rho_max=late['metadata']['rho_max'])
    tp = np.asarray(late['times'])
    for label, tq in (('t_49.22', 49.22), ('t_126.26', 126.26), ('t_0', 0.0)):
        k = int(np.argmin(np.abs(tp-tq)))
        out['late_run'][label] = dict(
            t_actual=float(tp[k]), radius_um=(None if measured['censored'][k] else float(measured['radius'][k])),
            censored=bool(measured['censored'][k]),
            rho_max_over_cells=float(np.max(late['rho'][k])),
            hill_term_max=float(np.max(late['rho'][k]**PARAMETERS['nl']
                                      / (PARAMETERS['kl']**PARAMETERS['nl']
                                         + late['rho'][k]**PARAMETERS['nl']))))
    out['late_pattern'] = early_pattern(late, measured)
    exp = experimental_control_radius()
    out['experimental_reference'] = dict(
        source=exp['source'], n_points=exp['n_points'], experimental=True,
        note=exp['note'],
        radius_at={str(q): float(exp['radius_um'][int(np.argmin(np.abs(exp['t_s']-q)))])
                   for q in (19.26, 25.68, 49.22, 126.26, 300.)})
    out['comparison_vs_experiment'] = {}
    for q in (19.26, 25.68, 49.22, 126.26):
        k = int(np.argmin(np.abs(tp-q)))
        obs = float(exp['radius_um'][int(np.argmin(np.abs(exp['t_s']-q)))])
        if measured['censored'][k]:
            out['comparison_vs_experiment'][str(q)] = dict(
                t_model=float(tp[k]), censored=True, observed_um=obs,
                note='model radius censored at this time; no agreement claimed')
        else:
            mod = float(measured['radius'][k])
            out['comparison_vs_experiment'][str(q)] = dict(
                t_model=float(tp[k]), censored=False, observed_um=obs,
                model_um=mod, difference_um=mod-obs)
    out['comparison_vs_experiment']['note'] = (
        'descriptive differences only.  The published parameters were fitted in '
        'this same experimental system, so agreement/disagreement is '
        'RETROSPECTIVE.  No parameter was adjusted to improve any of these numbers.')
    out['late_health'] = dict(
        nonfinite=int(np.size(late['c'])-np.count_nonzero(np.isfinite(late['c']))),
        c_min=float(np.min(late['c'])), c_max=float(np.max(late['c'])),
        ip3_min=float(np.min(late['ip3'])), er_min=float(np.min(late['er'])),
        h_min=float(np.min(late['h'])), h_max=float(np.max(late['h'])),
        negative_c_cells=int(np.count_nonzero(late['c'] < -1e-8)),
        negative_ip3_cells=int(np.count_nonzero(late['ip3'] < -1e-8)),
        ablated_c_ok=bool(np.allclose(late['c'][:, late['ablated']], 1000.)),
        censored_times=[float(tp[k]) for k in np.flatnonzero(measured['censored'])][:20],
        n_censored=int(np.sum(measured['censored'])))

    # ---- 4. finite-boundary check on nested grids ---------------------
    dom = []
    for ns in boundary_n_sides:
        res = run(n_side=ns, seed=seed, t_end=t_end_late, sample_dt=1.0, ligand='full')
        meas = radius_series(res)
        kk = int(np.argmin(np.abs(res['times']-300.)))
        peak = np.nanmax(meas['radius']) if np.isfinite(meas['radius']).any() else np.nan
        dom.append(dict(n_side=ns, n_cells=res['metadata']['n_cells'],
                        max_cell_radius_um=float(np.max(res['radius'])),
                        max_measurable_radius_um=float(meas['max_complete_radius']),
                        radius_at_300s=(None if meas['censored'][kk] else float(meas['radius'][kk])),
                        radius_at_49s=(None if meas['censored'][int(np.argmin(np.abs(res['times']-49.22)))]
                                       else float(meas['radius'][int(np.argmin(np.abs(res['times']-49.22)))])),
                        radius_at_126s=(None if meas['censored'][int(np.argmin(np.abs(res['times']-126.26)))]
                                        else float(meas['radius'][int(np.argmin(np.abs(res['times']-126.26)))])),
                        max_measured_radius_um=(None if not np.isfinite(peak) else float(peak)),
                        n_censored=int(np.sum(meas['censored'])),
                        seconds=res['metadata']['total_seconds']))
    out['boundary'] = dom
    out['boundary_differences'] = {}
    for a in dom:
        for b in dom:
            if a['n_side'] < b['n_side']:
                for tag, key in (('t=49.22s', 'radius_at_49s'),
                                 ('t=126.26s', 'radius_at_126s'),
                                 ('t=300s', 'radius_at_300s')):
                    va, vb = a[key], b[key]
                    if va is not None and vb is not None:
                        out['boundary_differences'][f'{a["n_side"]}vs{b["n_side"]}@{tag}'] = float(vb-va)
    # central-cell identity between nested grids (the coordinate-keyed mechanism)
    a = run(n_side=boundary_n_sides[0], seed=seed, t_end=1., sample_dt=1., ligand='off')
    b = run(n_side=boundary_n_sides[1], seed=seed, t_end=1., sample_dt=1., ligand='off')
    # model.py quantises coordinates into an INTEGER lattice key before seeding,
    # which is what makes shared cells match across nested grids; comparing raw
    # coordinates does NOT work because the patch is re-centred by its own mean.
    side_um = np.sqrt(2*PARAMETERS['reference_area']/(3*np.sqrt(3)))
    def keys_of(res):
        return np.rint(res['positions']/[np.sqrt(3)*side_um/4, 1.5*side_um/2]).astype(np.int64)
    ka, kb = keys_of(a), keys_of(b)
    map_a = {tuple(k): i for i, k in enumerate(ka)}
    ib = np.array([i for i, k in enumerate(kb) if tuple(k) in map_a], dtype=int)
    ia = np.array([map_a[tuple(kb[i])] for i in ib], dtype=int)
    shared = list(ia)
    het = dict(shared_cells=int(len(shared)),
               mechanism='coordinate-keyed random streams (identical to model.run)')
    if len(shared):
        het['shared_gj_multiplier_max_abs_diff'] = float(
            np.max(np.abs(a['gj_multipliers'][ia]-b['gj_multipliers'][ib])))
        het['shared_plc_multiplier_max_abs_diff'] = float(
            np.max(np.abs(a['plc_multipliers'][ia]-b['plc_multipliers'][ib])))
        het['shared_gcamp_multiplier_max_abs_diff'] = float(
            np.max(np.abs(a['gcamp_multipliers'][ia]-b['gcamp_multipliers'][ib])))
        het['nested_sides'] = [int(boundary_n_sides[0]), int(boundary_n_sides[1])]
    out['boundary_heterogeneity'] = het

    out['total_self_test_seconds'] = time.perf_counter()-t_start
    if verbose:
        import pprint
        pprint.pp(out)
    return out


if __name__ == '__main__':
    self_test()
