"""Early microtear-only calcium wound model (0--25 s); not experimental data.

Reduced Python port of ACCRE_FullCalciumSignalingModel_Control.m from
mshutson/wound-calcium-LRCa, described by Aaron C. Stevens, James T. O'Connor,
Andrew D. Pumford, Andrea Page-McCaw and M. Shane Hutson, 'A mathematical
model of calcium signals around laser-induced epithelial wounds'.
Dependencies: Python >=3.10, numpy, scipy. No Mathematica/openpyxl required.
Run this file for numerical smoke tests; importing it produces no files.

Fixed regular hexagons replace the original 4826-cell Voronoi mesh. No
mechanics, tissue motion, GBP/protease/ligand field, or late expansion is
modelled. Concentrations are micromolar, time seconds, geometry micrometres.
Ablated cells are upstream's imposed extracellular calcium reservoirs, NOT
viable cells: exclude them from calcium-response statistics. Outer edges
are sealed (no-flux), so the small default patch needs boundary checks.

Ported portions copyright (c) 2022 mshutson, MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.spatial import cKDTree
from scipy.special import erf

PARAMETERS = dict(
    kdeg=1.25, tau_d=0.001, nPMCA=2., cExt=1000., nSOC=3.8, kSOC=187.,
    epsilon=1/.185, nSERCA=2., d1=.13, d2=1.05, d3=.943, d5=.0823,
    a2=.2, Be=150., Ke=10., Kx=.167, nx=2.96,
    r_microtear=41.14497773461118, tau_heal=5.25, rPMCA=.12, kPMCA=.1,
    rSOC=.15000000000000002, etaNCX=10., nNCX=2.5, kNCX=1.6,
    rlkPM=.00025, alpha=.9, alpha0=.1, kl=.05, nl=5., Kc=.4,
    etaIPR=4., etaSERCA=5., kSERCA1=.1, kSERCA2=0., eta_lkER=.01,
    Bx=5., etaGJIP3=4., etaGJc=2., connect_ablated=.025,
    reference_area=43., wound_radius=51.25, ablated_radius=24.5,
)


def hex_geometry(n_side=20, area=43.):
    """Return centred fixed pointy-top hexagons and undirected shared edges.

    n_side**2 cells, area in um², edge length in um. Boundary cells retain
    their full physical area; absent exterior neighbours imply zero flux.
    """
    if int(n_side) != n_side or n_side < 2 or area <= 0:
        raise ValueError('n_side must be an integer >=2 and area positive')
    n_side = int(n_side)
    side = np.sqrt(2*area/(3*np.sqrt(3)))
    spacing = np.sqrt(3)*side
    row, col = np.indices((n_side, n_side))
    pos = np.column_stack(((col + .5*(row % 2)).ravel()*spacing,
                           row.ravel()*1.5*side))
    pos -= pos.mean(axis=0)
    angle = np.arange(6)*np.pi/3 + np.pi/6
    polygons = pos[:, None, :] + side*np.column_stack((np.cos(angle), np.sin(angle)))
    pairs = cKDTree(pos).query_pairs(spacing*(1+1e-8), output_type='ndarray')
    return dict(positions=pos, polygons=polygons,
                areas=np.full(n_side*n_side, area), edges=pairs,
                edge_lengths=np.full(len(pairs), side))


@lru_cache(maxsize=4)
def fit_microtear(spreadsheet=None):
    """Reproduce upstream lines 154--164, including row-10 fit and cutoff.

    Spreadsheet values are rescaled to [0,1]; fit a*erf(-(x-x0)/chi)+b
    with chi,a>0. First sample <.01 after row 9 defines cavitation pixels.
    Values are NOT clipped: the upstream unconstrained offset is retained.
    """
    path = Path(spreadsheet) if spreadsheet else Path(__file__).parent / 'upstream' / 'microtearDistribution_Andrew_dataPoints.xlsx'
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    data = np.array([[float(c.find('s:v', ns).text) for c in r.findall('s:c', ns)[:2]]
                     for r in root.findall('.//s:row', ns)], dtype=float)
    x, y = data.T
    y = (y-y.min())/np.ptp(y)
    def form(x, chi, x0, a, b):
        return a*erf(-(x-x0)/chi)+b
    pars, _ = curve_fit(form, x[9:], y[9:], p0=(10,46,.5,.5),
                        bounds=([1e-12,-np.inf,1e-12,-np.inf], [np.inf]*4),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    cutoff = x[9+np.flatnonzero(y[9:] < .01)[0]]
    return dict(zip(('chi','x0','a','b'), map(float, pars)),
                cutoff_pixels=float(cutoff), source=str(path))


def damage_profile(radius, wound_radius=51.25, spreadsheet=None):
    """Spreadsheet-fitted microtear spatial density, zero beyond cavitation."""
    f = fit_microtear(spreadsheet)
    r = np.asarray(radius)
    density = f['a']*erf(-(r*f['cutoff_pixels']/wound_radius-f['x0'])/f['chi'])+f['b']
    return np.where(r <= wound_radius, density, 0.)


def gap_operator(areas, edges, edge_lengths, cell_multipliers=None,
                 ablated=None, gj_scale=1., connect_ablated=.025,
                 reference_area=43.):
    """Conservative GJ operator before species eta and calcium buffering.

    K_ij = min(g_i,g_j)*edge_length*reference_area/(pi*diameter).
    J_i = sum_j K_ij*(u_j-u_i)/area_i; hence sum(area_i*J_i)=0.
    Diameter is 2*sqrt(mean(area)/pi), matching upstream mesh scaling.
    Calcium conservation applies to total rapid-buffered cytosolic calcium,
    not free c after multiplication by beta; ablation is an imposed source.
    """
    areas = np.asarray(areas)
    n = len(areas)
    i, j = np.asarray(edges).T
    mult = np.ones(n) if cell_multipliers is None else np.asarray(cell_multipliers)
    weight = gj_scale*reference_area/(2*np.sqrt(np.pi*areas.mean()))*edge_lengths*np.minimum(mult[i], mult[j])
    if ablated is not None:
        weight = weight*np.where(ablated[i] | ablated[j], connect_ablated, 1.)
    k = coo_matrix((np.r_[weight, weight], (np.r_[i,j], np.r_[j,i])), shape=(n,n)).tocsr()
    return (diags(1/areas) @ (k-diags(np.asarray(k.sum(axis=1)).ravel()))).tocsr()


def _rhs_factory(p, operator, alpha, bx, damage, ablated):
    n = len(alpha)
    def rhs(t, y):
        c, er, ip3, h = y.reshape(4, n)
        # No positivity clipping or saturation changes to the published ODE.
        rh = alpha*c/(p['Kc']+c)*p['alpha0']  # receptor occupancy lr=0
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


def _sparsity(operator):
    n = operator.shape[0]
    # A conservative superset of each four-species local reaction block.
    local = kron(np.ones((4,4)), eye(n), format='csr')
    coupling = kron(diags([1.,0.,1.,0.]), abs(operator), format='csr')
    return (local+coupling).astype(bool).tocsc()


def run(n_side=20, seed=2025, gj_scale=1., wound=True, rtol=1e-6,
        t_end=25., *, atol=1e-9, method='BDF', sample_dt=.1,
        times=None, equilibration_time=300., heterogeneity=True,
        parameters=None, spreadsheet=None, plc_scale=1., damage_override=None,
        ablated_override=None):
    """Integrate equilibrated early microtear-only tissue and return a dict.

    c, er, ip3, h have shape (number_of_times, n_side**2); geometry is fixed.
    metadata includes parameters, equilibration residual, solver statistics,
    fit, and runtime. `baseline` is the pre-ablation equilibrated state (4,N).
    `ablated` is a Boolean cell mask. `gj_scale=0` eliminates BOTH GJ species.
    Randomness is coordinate-keyed: the same seed matches central cells on
    nested grids differing by multiples of four (e.g. n_side=20 versus 28).
    heterogeneity=False gives a uniform control. plc_scale multiplies alpha
    throughout equilibration and wound simulation (not a post-wound switch).
    Default heterogeneity matches actual upstream code: GJ lognormal sigma
    .4, GCaMP sigma .1, PLC multiplier variance .2, all mean one. The upstream
    randExtParams applies PLC variation even though plcRand=False.
    Equilibration is the upstream 300 s tissue solve from (.1,200,.01,.67),
    not an assertion of exact steady state; residual is explicitly returned.
    """
    start = time.perf_counter()
    if not 0 < t_end <= 25:
        raise ValueError('Early microtear-only model requires 0 < t_end <= 25 s')
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
        # Mechanics-computed ablated set; replaces the radial prescription.
        ablated = np.asarray(ablated_override, dtype=bool).copy()
        if ablated.shape != (n,):
            raise ValueError(f'ablated_override must have shape ({n},)')
    # Coordinate-derived random streams preserve central-cell draws on nested
    # grids whose n_side differs by a multiple of four (e.g. 20 versus 28).
    side = np.sqrt(2*p['reference_area']/(3*np.sqrt(3)))
    keys = np.rint(geo['positions']/[np.sqrt(3)*side/4, 1.5*side/2]).astype(np.int64)
    if heterogeneity:
        z = np.array([np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(x) & 0xffffffff, int(y) & 0xffffffff])).normal(size=3)
                      for x,y in keys])
        gj = np.exp(-.4**2/2+.4*z[:,0])
        plc = np.exp(-.5*np.log1p(.2)+np.sqrt(np.log1p(.2))*z[:,1])
        gcamp = np.exp(-.1**2/2+.1*z[:,2])
    else:
        gj, plc, gcamp = np.ones((3,n))
    if plc_scale < 0:
        raise ValueError('plc_scale must be nonnegative')
    alpha, bx = p['alpha']*plc*plc_scale, p['Bx']*gcamp
    op_kw = dict(cell_multipliers=gj, gj_scale=gj_scale,
                 connect_ablated=p['connect_ablated'], reference_area=p['reference_area'])
    op_eq = gap_operator(geo['areas'],geo['edges'],geo['edge_lengths'], **op_kw)
    pattern = _sparsity(op_eq)
    empty = np.zeros(n, dtype=bool)
    eq_rhs = _rhs_factory(p,op_eq,alpha,bx,np.zeros(n),empty)
    initial = np.repeat([.1,200.,.01,.67],n)
    eq_start = time.perf_counter()
    eq = solve_ivp(eq_rhs,(0,equilibration_time),initial,method=method,
                   rtol=rtol,atol=atol,jac_sparsity=pattern,t_eval=[equilibration_time])
    if not eq.success:
        raise RuntimeError('Equilibration failed: '+eq.message)
    eq_seconds = time.perf_counter()-eq_start
    baseline = eq.y[:,-1].reshape(4,n).copy()
    y0 = baseline.copy()
    y0[0,ablated] = p['cExt']
    if damage_override is None:
        damage = damage_profile(radius,p['wound_radius'],spreadsheet) if wound else np.zeros(n)
    else:
        # Synthetic microtear field from an external mechanics module.
        damage = np.asarray(damage_override, dtype=float).copy()
        if damage.shape != (n,):
            raise ValueError(f'damage_override must have shape ({n},)')
        if np.any(damage < 0) or not np.all(np.isfinite(damage)):
            raise ValueError('damage_override must be finite and nonnegative')
    op = gap_operator(geo['areas'],geo['edges'],geo['edge_lengths'], ablated=ablated, **op_kw)
    rhs = _rhs_factory(p,op,alpha,bx,damage,ablated)
    if times is None:
        times = np.unique(np.r_[np.arange(0,t_end,sample_dt),t_end])
    times = np.asarray(times,dtype=float)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times)<=0) or times[0]<0 or times[-1]>t_end:
        raise ValueError('times must be increasing and within [0,t_end]')
    solve_start = time.perf_counter()
    sol = solve_ivp(rhs,(0,t_end),y0.ravel(),method=method,rtol=rtol,atol=atol,
                    jac_sparsity=pattern,t_eval=times,first_step=min(1e-5,t_end))
    if not sol.success:
        raise RuntimeError('Wound integration failed: '+sol.message)
    solve_seconds = time.perf_counter()-solve_start
    values = sol.y.T.reshape(len(times),4,n)
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Nonfinite solution')
    metadata = dict(model='early microtear-only model', synthetic=True,
        upstream='upstream/ACCRE_FullCalciumSignalingModel_Control.m',
        n_cells=n, seed=seed, gj_scale=gj_scale, plc_scale=plc_scale, wound=bool(wound),
        parameters=p, method=method, rtol=rtol, atol=atol,
        equilibration_time=equilibration_time,
        equilibration_max_abs_rhs=float(np.max(np.abs(eq_rhs(equilibration_time,eq.y[:,-1])))),
        equilibration_seconds=eq_seconds, integration_seconds=solve_seconds,
        total_seconds=time.perf_counter()-start, nfev=sol.nfev, njev=sol.njev, nlu=sol.nlu,
        damage_fit=fit_microtear(spreadsheet) if (wound and damage_override is None) else None,
        damage_source='external mechanics override (synthetic, uncalibrated)' if damage_override is not None else ('published erfc/spreadsheet damage profile (prescribed)' if wound else 'none'),
        heterogeneity=heterogeneity, boundary='sealed/no-flux',
        units=dict(times='s',c='uM',er='uM per ER volume',ip3='uM',h='dimensionless',positions='um',areas='um^2'),
        limitations=['fixed regular geometry, no mechanics','no GBP/ligand late expansion',
                     'finite small domain; test larger n_side','ablated c is imposed 1000 uM reservoir'])
    return dict(times=sol.t,t=sol.t,c=values[:,0],er=values[:,1],ip3=values[:,2],h=values[:,3],
                baseline_c=baseline[0].copy(),gcamp=gcamp,
                **geo, metadata=metadata, baseline=baseline, ablated=ablated,
                damage=damage, gj_multipliers=gj, plc_multipliers=plc,
                gcamp_multipliers=gcamp)


def self_test():
    """Actual numerical tests, no plots or output files; returns diagnostics."""
    geo = hex_geometry()
    poly = geo['polygons']
    area = .5*np.abs(np.sum(poly[:,:,0]*np.roll(poly[:,:,1],-1,axis=1)-poly[:,:,1]*np.roll(poly[:,:,0],-1,axis=1),axis=1))
    assert np.allclose(area,43)
    rng = np.random.default_rng(7)
    irregular_areas = rng.uniform(20,70,400)
    op = gap_operator(irregular_areas,geo['edges'],geo['edge_lengths'])
    flux = op@rng.normal(size=400)
    assert abs(irregular_areas@flux)<1e-10
    assert np.max(abs(op@np.ones(400)))<1e-14
    result = run()
    assert result['c'].shape == (251,400)
    assert np.all(result['c'][:,result['ablated']]==1000)
    for key in ('c','er','ip3','h'):
        assert np.min(result[key])>=-1e-8
    assert np.max(result['h'])<=1+1e-8
    unwounded = run(wound=False)
    drift = float(np.max(abs(unwounded['c']-unwounded['baseline'][0])))
    assert drift<1e-4
    return dict(conservation_error=float(irregular_areas@flux),
                unwounded_c_drift=drift, wounded=result['metadata'],
                unwounded_seconds=unwounded['metadata']['total_seconds'])


if __name__ == '__main__':
    import pprint
    pprint.pp(self_test())
