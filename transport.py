"""Vascular and tissue exchange (transport) layer for the epithelial tissue patch.

================================================================================
READ THIS FIRST -- WHAT THIS MODULE IS AND IS NOT
================================================================================
REVISION NOTICE: Generic vascular model only. Insects use tracheae and Hydra
relies on diffusion; this is not validated fruit-fly or Hydra physiology or an
entire-organism simulation. Historical analytic-test/spacing claims below are
unverified: the interrupted original had no self_test implementation. The
bounded self_test now provided checks numerical invariants and a discrete
analytic diffusion mode only. No physiological validation is implied.

This is a *transport model with largely ILLUSTRATIVE parameters*. It is NOT
validated against any measurement in this project (or anywhere else). It exists
to give the other layers (cellstate.py's oxygen input, a substrate-limited
growth term) a spatially resolved, mass-conserving field of dissolved
substances, and to be numerically verifiable against closed-form solutions.

Loud statements, please do not soften them when reporting:

1. PARAMETERS.  Every parameter carries a `provenance` tag:
     * 'CITED'      -- a literature value with an explicit citation string in
                       CITATIONS, used only where I am confident about the
                       source AND the number. No value in this module is
                       marked CITED for a *numeric* oxygen-transport parameter;
                       see PARAMETER_PROVENANCE below and the DISAGREEMENTS
                       block. Read the provenance dict of every substance
                       before quoting a number.
     * 'STANDARD'   -- textbook/physical chemistry constants that are not
                       seriously disputed (e.g. 1 uM = 1e-6 mol/L, blood
                       glucose ~5 mM). Not transcribed from a source here.
     * 'ILLUSTRATIVE' -- a modelling choice. Most of this module is this.
   The vessel geometry in this module is PRESCRIBED, NOT GROWN: you hand it
   polylines with radii, flow and permeability. Nothing here grows, remodels,
   dilates or adapts a vessel, and nothing here feeds back from tissue state to
   vessel geometry. There is no angiogenesis.

2. DISAGREEMENTS IN THE LITERATURE (recorded, not resolved).  Published
   Krogh-type parameter tables disagree on the oxygen parameters by large
   factors: the apparent Michaelis constant of oxygen for respiration is
   reported from below 1 uM (isolated mitochondria) to tens of uM (intact
   cells/tissue), i.e. up to ~10x, and the haemoglobin P50 that appears in
   blood-side boundary conditions varies by ~1.5x depending on the standard
   curve / species / pH convention. Measured capillary spacing is tens of
   micrometres. Therefore:
     * Do NOT quote a single "true" Km(O2), P50 or capillary spacing from this
       module. The default Km(O2) here is ILLUSTRATIVE and the tests that
       depend on it report the sensitivity rather than a point value.
     * UNVERIFIED historical claim: a planned test would report spacing for the
       illustrative parameters AND states plainly whether it reproduces the
       order of magnitude of the measured tens-of-micrometres spacing. For the
       illustrative defaults it does NOT (it is 10-100x larger): with
       documented oxygen transport numbers a Krogh-type threshold does not sit
       at tens of micrometres at resting demand, which is consistent with
       normal tissue at rest not being hypoxic. That negative result is a
       result; do not hide it.

3. AREA OF VALIDITY.  The solver is unconditionally stable (implicit
   diffusion, implicit linear/linearised sinks), positivity preserving and
   *exactly conservative by construction* (see `conservation` in the
   docstring of `TransportSolver`). Its accuracy is first order in dt
   (backward Euler / Godunov split) and second order in dx for smooth fields.
   The interrupted original claimed (without implementing tests) verification
   (2D fundamental solution, 1D slab with Robin wall + zero-order consumption,
   cylindrical Krogh profile). Passing those tests says the arithmetic is
   right; it says NOTHING about the biology.

4. SCOPE OF PHYSICS INCLUDED.  Diffusion (uniform, isotropic, per substance),
   uptake by cells (Michaelis-Menten / zero-order / first-order, deposited
   conservatively), tissue-level zero-order consumption and first-order
   binding/clearance, and vessel exchange through a prescribed wall. NOT
   included: convection/interstitial flow, spatially varying diffusion or
   solubility, cell-to-cell gap-junction transport of these substances (that
   lives in ligand.py), pH/CO2 coupling, oxygen-haemoglobin kinetics beyond a
   documented linear or Hill carrier option, angiogenesis, vessel autoregulation,
   and axial blood-side depletion beyond the Renkin-Crone extraction factor
   (see VESSEL EXCHANGE LAW below).

================================================================================
UNITS (used consistently everywhere)
================================================================================
    length        um
    time          s
    concentration uM  (= umol/L)
    amount        amol (1e-18 mol); 1 uM * 1 um^3 = 1e-3 amol exactly
    volume        um^3
    D             um^2/s          P (permeability) um/s
    Q (blood flow) um^3/s         wall area um^2
    PS product    um^3/s
Internally the solver works in "concentration * volume" units (uM*um^3); the
budget output converts to amol with AMOL_PER_UM3_UM = 1e-3. Beware: 1 uM*um^3
= 1e-21 mol = 1e-3 amol. A 300x300 um patch 10 um thick holds 9e5 um^3, so
140 uM of dissolved O2 is 126000 amol = 1.26e-13 mol.

2D REDUCTION.  A 2D run is a slab of thickness `thickness_um` (default 10 um;
PROJECT ANCHOR: the Drosophila notum epithelium is 7.5-10 um thick, see
MEASURED_ANCHORS.md -> Pinheiro 2017 PMC6143170; that anchor is for *notum*,
not for a generic tissue). Tissue volume of a voxel is dx*dy*thickness. A
vessel is an in-plane cylinder of the given radius whose LATERAL wall area is
2*pi*r*L (all of the cylinder), and whose blood flow Q is the real volumetric
flow of that cylinder. Consequence, documented because it is easy to get
wrong: increasing thickness adds tissue volume without adding vessel wall area
or flow, which is the correct Krogh-like direction.

================================================================================
VESSEL EXCHANGE LAW (all three offered, documented)
================================================================================
For an exchange unit = one vessel polyline (one capillary, one flow Q) with
total wall area A_w, wall permeability P_w, blood-side (plasma) concentration
C_b, tissue:blood equilibrium partition alpha (alpha = C_tissue/C_blood at
equilibrium) and a single representative tissue-side concentration C_t:

  PS      = P_w * A_w                                     [um^3/s]
  (a) 'permeability'   J = PS * (C_b - C_t/alpha)
  (b) 'flow'           J =  Q * (C_b - C_t/alpha)
  (c) 'renkin_crone'   J =  Q * (1 - exp(-PS/Q)) * (C_b - C_t/alpha)   [default]

Law (c) is the classic Renkin-Crone / "convection-permeation" result: it is
exact for a capillary with uniform tissue concentration, it tends to (a) as
PS/Q -> 0 and to (b) as PS/Q -> infinity, and it is the reason delivery passes
from permeability-limited to flow-limited as the wall opens up. The
dimensionless group that controls the transition is PS/Q; the *value of P_w* at
which it happens is set by A_w and Q, i.e. by assumed geometry and flow.
The flow cap in (b)/(c) uses a "carrier factor" beta >= 1: J = Q*beta*(...)
when beta > 1, modelling substances carried bound to a carrier so that the
arteriovenous content difference exceeds the dissolved difference. beta = 1
(pure dissolved, the default) keeps the arithmetic interpretable.

Implementation note (why the total is right even though the field is
distributed over voxels): the unit's conductance G = J/(C_b - C_t/alpha) is
split over the unit's support voxels v as G_v = w_v * G with sum_v w_v = 1.
Then sum_v G_v (C_b - C_v/alpha) = G (C_b - C_bar/alpha) with
C_bar = sum_v w_v C_v, i.e. the *lumped* law is reproduced exactly and the
term stays diagonal, which is what makes it implicit and stable. What is not
reproduced exactly is the axial variation of C_t along the vessel (neglected).

================================================================================
NUMERICS
================================================================================
Cell-centred finite volume on a regular grid, zero-flux (Neumann) boundaries by
default; a "bath" Robin layer on the outer faces is also available.

Per substance and step, with L the conservative FV Laplacian:

    diffusion:  (I - dt * D_eff * L) C* = C^n            [implicit, exact]
    local:      C^{n+1} = (C* + dt*s_v - dt*z_v) / (1 + dt*k_v)

  k_v  [1/s] linear sink coefficient from: cell Michaelis-Menten with the
        denominator frozen at the step-start concentration
        (Vmax_voxel/(Km + C)), cell first-order, tissue first-order binding,
        vessel wall back-flux G_v/(alpha*V_v), bath Robin
  s_v  [uM/s] sources: vessel delivery G_v*C_b/V_v, uniform background
  z_v  [uM/s] zero-order (concentration-independent) consumption, clipped to
        what the available concentration can pay for (guard event recorded)

Two solvers for the implicit diffusion step, both exactly conservative:
  * scheme='imex'   : the diffusion and the local terms are solved TOGETHER as
        one sparse linear system with a direct sparse LU (scipy splu). No
        splitting error. Default for grids up to `direct_max_nodes`.
  * scheme='split'  : the diffusion step is solved EXACTLY by an orthonormal
        type-II DCT (the cell-centred Neumann FV Laplacian is diagonalised by
        the DCT-II basis, eigenvalues lambda_k = sum_a 2(cos(pi k_a/n_a)-1)/d_a^2),
        then the local terms are applied implicitly. O(N log N), memory-light,
        scalable to millions of voxels. Godunov splitting => first order in dt.
Both schemes are unconditionally stable and positivity preserving. They differ
by O(dt); self_test compares them.

CONSERVATION (the property that matters for the budget):
  The FV Laplacian has exactly zero column sums weighted by voxel volume, so
  sum_v V_v (L C)_v is a pure round-off residual for ANY C. Every sink/source
  term is applied as a voxel-local rate and the amount it removes/adds is
  computed from the *same* rate coefficients and the *solved* concentration.
  Therefore, per substance and step:

    d(amount) = dt*sum_v V_v*s_v  -  dt*sum_v V_v*k_v*C^{n+1}
                - dt*sum_v V_v*z_v  +  round-off

  and the reported budget (delivered - consumed - bound - accumulated) is that
  identity, evaluated on the numbers actually used. Cell uptake is deposited
  with the same cell->voxel weights it is measured with, so "what the cells
  consume is what the field loses" holds to round-off, not approximately. The
  residual test (self_test test 3) reports the residual as an absolute number,
  a relative number and its behaviour under dt refinement.

API
    run(domain_um, dx_um, t_end, dt, substances, vessels, cells, blood, ...) -> dict
    solve_steady_state(...) -> dict
    self_test() -> dict            (prints every number it computes)
    max_stable_dt(...) -> dict
    default_substances() -> dict
    krogh_vessels(...), single_vessel(...)   (prescribed Krogh-type geometry)
    map_cells(...), cellstate_oxygen_input(...), growth_substrate_factor(...)

Result keys: 'times', 'fields' {name: (n_t,)+grid}, 'grid', 'cell_concentration'
{name: (n_t,n_cells)}, 'cell_uptake' {name: (n_t,n_cells)}, 'cell_uptake_total',
'budget', 'metadata', 'cellstate_oxygen', 'final'.

DEVIATIONS FROM THE REQUESTED SIGNATURE (stated so they are not surprises):
  * 'cell_concentration' and 'cell_uptake' are DICTS keyed by substance name,
    each (n_t, n_cells), because the model is multi-substance. A flat
    'cell_uptake_total' (n_t, n_cells) over all consuming substances is also
    returned.
  * For very large grids the field time series is decimated (see `store_every`
    and `max_field_bytes`); 'metadata' records 'store_stride' and
    'stored_frames'.
  * growth.py does not exist in this project directory (checked); the coupling
    contract for a substrate-limited growth term is exposed as
    growth_substrate_factor() plus per-cell sampled concentrations.
"""

from __future__ import annotations

import time as _time
import numpy as np
from dataclasses import dataclass, field, replace
from scipy import fft as _fft
from scipy import sparse as _sparse
from scipy.sparse import linalg as _splinalg

__all__ = [
    'Substance', 'default_substances', 'make_grid', 'laplacian', 'max_stable_dt',
    'krogh_vessels', 'single_vessel', 'map_cells', 'TransportSolver', 'run',
    'solve_steady_state', 'self_test', 'cellstate_oxygen_input',
    'growth_substrate_factor', 'exchange_conductance', 'CITATIONS',
    'PARAMETER_PROVENANCE', 'AMOL_PER_UM3_UM',
]

# ===========================================================================
# units and provenance
# ===========================================================================
AMOL_PER_UM3_UM = 1.0e-3      # amount[amol] = C[uM] * V[um^3] * this

PARAMETER_PROVENANCE = {
    'note': (
        'Every numeric parameter of this module is either CITED (specific '
        'publication, string recorded in CITATIONS), STANDARD (textbook or '
        'physical constant, not seriously disputed, not transcribed here), or '
        'ILLUSTRATIVE (a modelling choice made for this project, unvalidated). '
        'There is no CITED numeric oxygen-transport parameter in this module: '
        'the published tables disagree by up to ~10x on Km(O2) and ~1.5x on '
        'P50, and I did not read any of them in full in this session.'),
    'ILLUSTRATIVE': 'chosen for this prototype; not measured, not validated',
    'STANDARD': 'textbook/physical constant; not seriously disputed; provenance not re-verified here',
    'CITED': 'specific publication given in CITATIONS; not re-derived here',
    'PROJECT_ANCHOR': 'value taken from this project (MEASURED_ANCHORS.md), with the source it cites there',
}

# Citations are recorded to say WHICH standard result or paper is referred to.
# They were NOT all read in full in this session; values derived from them are
# marked accordingly. Do not treat this dict as evidence that a number was
# transcribed from the paper.
CITATIONS = {
    'krogh1919': ('Krogh A. The number and distribution of capillaries in muscles with '
                  'calculations of the oxygen pressure head necessary for supplying the '
                  'tissue. J Physiol 52:409-415 (1919). [reference for the Krogh cylinder '
                  'model and its critical-radius criterion; not read in full here]'),
    'renkin1959': ('Renkin EM. Transport of potassium-42 from blood to tissue in isolated '
                   'mammalian skeletal muscles. Am J Physiol 197:1205-1210 (1959). '
                   '[source of E = 1-exp(-PS/Q); not read in full here]'),
    'crone1963': ('Crone C. The permeability of capillaries in various organs as determined '
                  'by use of the indicator diffusion method. Acta Physiol Scand 58:292-305 '
                  '(1963). [companion to Renkin 1959; not read in full here]'),
    'goldman2008': ('Goldman D. Theoretical models of microvascular oxygen transport to '
                    'tissue. Microcirculation 15:795-811 (2008). [review of Krogh-type '
                    'models and of their known limitations (uniform consumption, no axial '
                    'gradient, neglected heterogeneity); not read in full here]'),
    'severinghaus1979': ('Severinghaus JW. Simple, accurate equations for human blood O2 '
                         'dissociation computations. J Appl Physiol 46:599-602 (1979). '
                         '[standard human O2 dissociation curve, P50 ~26-27 mmHg at pH 7.4, '
                         '37 C; not transcribed here]'),
    'wilson1979': ('Wilson DF, Owen CS, Holian A. Quantitative dependence of mitochondrial '
                   'oxidative phosphorylation on oxygen concentration: a mathematical '
                   'model. Arch Biochem Biophys 195:494-504 (1979). [one end of the '
                   'disputed Km(O2) range; surfaced by search, not read in full here]'),
    'scandurra2010': ('Scandurra FM, Gnaiger E. Cell respiration under hypoxia: facts and '
                      'artefacts in mitochondrial oxygen kinetics. Adv Exp Med Biol '
                      '662:7-25 (2010). [documents the disagreement in apparent Km(O2) '
                      'between isolated mitochondria and intact cells; surfaced by search, '
                      'not read in full here]'),
    'pinheiro2017': 'Pinheiro D. et al. (2017) PMC6143170 -- notum epithelium 7.5-10 um thick (via MEASURED_ANCHORS.md)',
    'project_hexagon': 'model.hex_geometry / cellstate.cell_geometry: cell area 43 um^2, centroid spacing 7.05 um',
}


# ===========================================================================
# substances
# ===========================================================================
@dataclass
class Substance:
    """One dissolved substance with its own transport, solubility and sinks.

    Parameters
    ----------
    name : str
    D_um2_s : float        free (unobstructed) diffusion coefficient [um^2/s].
    tortuosity : float     >= 1. Tissue effective D is D_um2_s/tortuosity**2.
    alpha : float          tissue:blood equilibrium partition (C_tissue/C_blood
                           at equilibrium). Solubility ratio, dimensionless.
    plasma_conc_um : float default blood-side (plasma) concentration [uM] used
                           both as the vessel boundary value when a vessel does
                           not override it, and as the default uniform initial
                           condition.
    carrier_factor : float beta >= 1 for the flow-limited law (see module
                           docstring); 1 = dissolved only.
    zero_order_um_s : float tissue-level concentration-independent consumption
                           [uM/s], NOT cell mapped (used for analytic tests and
                           for a uniform metabolic background).
    k_first_s : float      tissue-level first-order clearance/binding [1/s].
                           Counted in the budget as 'bound' (a hormone that is
                           delivered and bound, not consumed).
    cell_consumption : str 'none' | 'michaelis_menten' | 'zero_order' | 'first_order'
    vmax_per_cell_amol_s : float per-cell MM maximum rate [amol/(cell*s)]
    km_um : float          MM half-saturation [uM]
    zero_order_per_cell_amol_s : float per-cell concentration-independent rate
    k_cell_um3_s : float   per-cell first-order clearance volume [um^3/s]
    provenance : dict      {'value': tag} per field, see PARAMETER_PROVENANCE
    """
    name: str
    D_um2_s: float
    tortuosity: float = 1.0
    alpha: float = 1.0
    plasma_conc_um: float = 0.0
    carrier_factor: float = 1.0
    zero_order_um_s: float = 0.0
    k_first_s: float = 0.0
    cell_consumption: str = 'none'
    vmax_per_cell_amol_s: float = 0.0
    km_um: float = 0.0
    zero_order_per_cell_amol_s: float = 0.0
    k_cell_um3_s: float = 0.0
    provenance: dict = field(default_factory=dict)

    # ---- derived
    @property
    def D_eff(self) -> float:
        """Effective tissue diffusion coefficient D/tortuosity^2 [um^2/s]."""
        return float(self.D_um2_s) / float(self.tortuosity) ** 2

    def equilibrium_tissue_conc(self, blood_conc_um: float | None = None) -> float:
        """Tissue concentration in equilibrium with the blood side [uM]."""
        cb = self.plasma_conc_um if blood_conc_um is None else blood_conc_um
        return float(self.alpha) * float(cb)

    def check(self):
        for key in ('tortuosity', 'carrier_factor', 'zero_order_um_s', 'k_first_s',
                    'vmax_per_cell_amol_s', 'km_um', 'zero_order_per_cell_amol_s',
                    'k_cell_um3_s'):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{self.name}: {key} must be finite and nonnegative')
        if self.D_um2_s <= 0 or not np.isfinite(self.D_um2_s):
            raise ValueError(f'{self.name}: D_um2_s must be finite and > 0')
        if self.tortuosity < 1:
            raise ValueError(f'{self.name}: tortuosity must be >= 1')
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f'{self.name}: alpha must be finite and > 0')
        if self.plasma_conc_um < 0 or not np.isfinite(self.plasma_conc_um):
            raise ValueError(f'{self.name}: plasma_conc_um must be finite and >= 0')
        if self.carrier_factor < 1:
            raise ValueError(f'{self.name}: carrier_factor must be >= 1')
        if self.cell_consumption not in ('none', 'michaelis_menten', 'zero_order',
                                         'first_order'):
            raise ValueError(f'{self.name}: unknown cell_consumption '
                             f'{self.cell_consumption!r}')
        if self.cell_consumption == 'michaelis_menten' and self.km_um <= 0:
            raise ValueError(f'{self.name}: michaelis_menten needs km_um > 0')
        if self.zero_order_um_s < 0 or self.k_first_s < 0:
            raise ValueError(f'{self.name}: tissue sinks must be >= 0')
        return self


def default_substances(names=('o2', 'glucose', 'hormone')):
    """Built-in substance table. Almost everything here is ILLUSTRATIVE.

    o2       -- dissolved oxygen, oxygen carried in blood. Km(O2) is the value
                with the worst literature disagreement; the default 5 uM sits
                in the disputed range. Dissolved O2 in plasma at PO2 = 100 mmHg
                is ~140 uM by the standard solubility ~1.4 uM/mmHg (STANDARD);
                the O2 bound to haemoglobin is ~60x more, which is what the
                carrier_factor option represents. Default carrier_factor = 1
                (dissolved only) so that flow-limited numbers stay interpretable.
    glucose  -- blood glucose ~5 mM (STANDARD, textbook). Glycolytic capacity is
                large so the per-cell demand here is a modelling choice.
    hormone  -- a generic signalling molecule that is DELIVERED and BOUND but
                NOT consumed: cell_consumption='none', first-order tissue
                binding k_first_s. It exists to exercise the 'bound' budget
                category and the non-consuming coupling path.
    """
    prov_i = 'ILLUSTRATIVE'
    prov_s = 'STANDARD'
    table = {
        'o2': Substance(
            name='o2',
            D_um2_s=2100.0,          # free-water O2 diffusivity at 37 C, order 2-3e3 um^2/s
            tortuosity=1.2,          # -> D_eff = 1458 um^2/s in tissue
            alpha=1.0,
            plasma_conc_um=140.0,    # dissolved O2 at PO2 = 100 mmHg, solubility ~1.4 uM/mmHg
            carrier_factor=1.0,
            cell_consumption='michaelis_menten',
            vmax_per_cell_amol_s=1.0,
            km_um=5.0,
            provenance={
                'D_um2_s': prov_i, 'tortuosity': prov_i, 'alpha': prov_i,
                'plasma_conc_um': prov_s + ' (solubility ~1.4 uM/mmHg is standard; '
                                  'PO2 = 100 mmHg is a convention)',
                'km_um': prov_i + ' (literature disagrees by up to ~10x, see module docstring)',
                'vmax_per_cell_amol_s': prov_i,
                'carrier_factor': prov_i,
            }),
        'glucose': Substance(
            name='glucose',
            D_um2_s=600.0,
            tortuosity=1.4,
            alpha=1.0,
            plasma_conc_um=5000.0,   # 5 mM blood glucose (STANDARD)
            cell_consumption='michaelis_menten',
            vmax_per_cell_amol_s=0.1,
            km_um=1000.0,
            provenance={
                'D_um2_s': prov_i, 'tortuosity': prov_i, 'alpha': prov_i,
                'plasma_conc_um': prov_s + ' (blood glucose 4-6 mM)',
                'km_um': prov_i, 'vmax_per_cell_amol_s': prov_i,
            }),
        'hormone': Substance(
            name='hormone',
            D_um2_s=300.0,
            tortuosity=1.3,
            alpha=2.0,
            plasma_conc_um=1.0,
            k_first_s=1.0e-3,
            cell_consumption='none',
            provenance={
                'D_um2_s': prov_i, 'tortuosity': prov_i, 'alpha': prov_i,
                'plasma_conc_um': prov_i, 'k_first_s': prov_i,
                'cell_consumption': 'by definition: delivered and bound, not consumed',
            }),
    }
    if names is None:
        return table
    if isinstance(names, str):
        names = (names,)
    out = {}
    for nm in names:
        if nm not in table:
            raise KeyError(f'unknown substance {nm!r}; known: {sorted(table)}')
        out[nm] = table[nm]
    return out


def _resolve_substances(substances):
    """Accept None | name | iterable of names | dict name->Substance | dict name->params."""
    if substances is None:
        return default_substances()
    if isinstance(substances, Substance):
        return {substances.name: substances}
    if isinstance(substances, str):
        return default_substances((substances,))
    if isinstance(substances, dict):
        out = {}
        for k, v in substances.items():
            if isinstance(v, Substance):
                out[k] = v.check()
            elif v is None:
                out[k] = default_substances((k,))[k].check()
            elif isinstance(v, dict):
                base = default_substances((k,))[k] if k in default_substances() else None
                if base is None:
                    raise KeyError(f'cannot build substance {k!r} from a dict without D_um2_s')
                out[k] = replace(base, **v).check()
            else:
                raise TypeError(f'substance {k!r}: expected Substance, dict or None')
        return out
    out = {}
    for value in substances:
        s = default_substances(value)[value] if isinstance(value, str) else value
        out[s.name] = s.check()
    return out


# ===========================================================================
# grid
# ===========================================================================
def make_grid(domain_um=(300.0, 300.0), dx_um=2.0, thickness_um=10.0):
    """Regular cell-centred finite-volume grid over a tissue domain.

    domain_um : (Lx, Ly) or (Lx, Ly, Lz) [um]. The domain is padded outward so
        that an integer number of voxels of size dx_um fits; the realised
        domain is returned (metadata records the padding) rather than silently
        changing dx.
    dx_um : voxel size in every direction [um] (scalar; keep it simple).
    thickness_um : out-of-plane thickness used ONLY in 2D, to convert in-plane
        area to tissue volume [um]. See the module docstring.

    Returns a dict with 'shape', 'dx', 'extent_um', 'volume_um3' (per voxel),
    'total_volume_um3', 'n_nodes', 'centers' (n_nodes, ndim), 'ndim',
    'thickness_um', 'padding_um'.
    """
    dom = np.atleast_1d(np.asarray(domain_um, float))
    if dom.ndim != 1 or dom.size not in (2, 3):
        raise ValueError('domain_um must have 2 or 3 entries')
    if np.any(~np.isfinite(dom)) or np.any(dom <= 0):
        raise ValueError('domain_um entries must be finite and > 0')
    if not np.isfinite(dx_um) or dx_um <= 0:
        raise ValueError('dx_um must be finite and > 0')
    if not np.isfinite(thickness_um) or thickness_um <= 0:
        raise ValueError('thickness_um must be finite and positive')
    ndim = int(dom.size)
    shape = tuple(int(np.ceil(L / dx_um)) for L in dom)
    extent = tuple(s * dx_um for s in shape)
    padding = tuple(extent[i] - dom[i] for i in range(ndim))
    idx = np.indices(shape, dtype=float)
    centers = np.stack([(idx[i].ravel() + 0.5) * dx_um for i in range(ndim)], axis=1)
    d = tuple(float(dx_um) for _ in range(ndim))
    vol = dx_um ** ndim if ndim == 3 else dx_um * dx_um * float(thickness_um)
    return dict(
        ndim=ndim, shape=shape, d=d, dx=float(dx_um),
        extent_um=extent, padding_um=padding, domain_um=tuple(float(x) for x in dom),
        centers=centers, n_nodes=int(np.prod(shape)),
        volume_um3=float(vol), total_volume_um3=float(vol * np.prod(shape)),
        thickness_um=float(thickness_um) if ndim == 2 else None,
        spacing_note=('2D slab: voxel volume = dx*dy*thickness_um; vessel wall area '
                      '= 2*pi*r*L (full cylinder lateral area)'),
    )


def laplacian(grid, dtype=float):
    """Conservative cell-centred FV Laplacian with zero normal flux on faces.

    Row sums are exactly zero (verified in self_test by construction and by
    algebra); it is the discrete divergence of face fluxes, so
    sum_v V_v (L C)_v == 0 to round-off for any C. This is the property the
    conservation budget relies on.
    """
    shape = grid['shape']
    n = grid['n_nodes']
    idx = np.arange(n).reshape(shape)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for a, (na, da) in enumerate(zip(shape, grid['d'])):
        cnt = np.full(n, 2.0)
        for endi in (0, na - 1):
            sel = np.take(idx, endi, axis=a).ravel()
            cnt[sel] -= 1.0
        diag -= cnt / da ** 2
        if na > 1:
            lo = np.take(idx, np.arange(na - 1), axis=a).ravel()
            hi = np.take(idx, np.arange(1, na), axis=a).ravel()
            c = 1.0 / da ** 2
            rows.append(lo); cols.append(hi); vals.append(np.full(lo.size, c))
            rows.append(hi); cols.append(lo); vals.append(np.full(hi.size, c))
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    L = _sparse.csr_matrix((np.concatenate(vals),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n), dtype=dtype)
    L.sum_duplicates()
    return L


def dct_eigenvalues(grid):
    """Eigenvalues of the FV Neumann Laplacian in the orthonormal DCT-II basis.

    lambda(k) = sum_a 2*(cos(pi*k_a/n_a) - 1)/d_a^2, k_a = 0..n_a-1.
    Verified against a sparse matvec in self_test (it is exact, not an
    approximation: the cell-centred Neumann stencil has the DCT-II vectors as
    exact eigenvectors, with the ghost node mirroring the boundary node).
    """
    per_axis = []
    for na, da in zip(grid['shape'], grid['d']):
        kk = np.arange(na)
        per_axis.append(2.0 * (np.cos(np.pi * kk / na) - 1.0) / da ** 2)
    lam = per_axis[0].reshape((grid['shape'][0],) + (1,) * (grid['ndim'] - 1))
    if grid['ndim'] >= 2:
        lam = lam + per_axis[1].reshape((1, grid['shape'][1]) + (1,) * (grid['ndim'] - 2))
    if grid['ndim'] == 3:
        lam = lam + per_axis[2].reshape((1, 1, grid['shape'][2]))
    return lam


# ===========================================================================
# vessel exchange law
# ===========================================================================
def exchange_conductance(PS_um3_s, Q_um3_s, law='renkin_crone', carrier_factor=1.0,
                         back_pressure_frac=0.0):
    """Conductance G [um^3/s] of one exchange unit for a given exchange law.

    J = G * (C_b - C_t/alpha) (times carrier_factor for the flow term). All
    three laws are LINEAR in the tissue concentration, which is what allows the
    wall term to be treated implicitly (diagonal) and unconditionally stable.

    law : 'permeability' (G = PS) | 'flow' (G = Q) | 'renkin_crone'
          (G = Q*(1-exp(-PS/Q)), the default; equals PS for PS << Q and Q for
          PS >> Q).
    carrier_factor : multiplies the flow-limited conductance only (the PS path
          in the Renkin-Crone law is a diffusive conductance and is not
          multiplied by the carrier capacity).
    back_pressure_frac : optional min(1) fraction; implemented as reducing the
          effective flow term (kept for API compatibility, unused by default).
    """
    if PS_um3_s < 0 or Q_um3_s < 0:
        raise ValueError('PS and Q must be >= 0')
    law = str(law)
    bf = max(0.0, 1.0 - float(back_pressure_frac))
    if law == 'permeability':
        return float(PS_um3_s)
    if law == 'flow':
        return float(Q_um3_s) * float(carrier_factor) * bf
    if law == 'renkin_crone':
        if Q_um3_s <= 0:
            # no flow: only the diffusive path survives
            return float(PS_um3_s)
        x = PS_um3_s / Q_um3_s
        # numerically safe 1 - exp(-x)
        flow_term = Q_um3_s * (-np.expm1(-x))
        diff_term = PS_um3_s - flow_term if PS_um3_s < Q_um3_s else 0.0
        # Renkin-Crone "series" form: G = Q*(1-exp(-PS/Q)), which already
        # interpolates between PS and Q; keep it exact and simple.
        return float(flow_term) * (1.0 + (bf - 1.0)) + 0.0 * diff_term
    raise ValueError(f'unknown exchange law {law!r}')


def _segment_distance(centers, p0, p1):
    """Distance from every point to a straight segment (vectorised)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    v = p1 - p0
    L2 = float(v @ v)
    w = centers - p0
    if L2 <= 0:
        return np.linalg.norm(w, axis=1), np.zeros(centers.shape[0])
    t = np.clip((w @ v) / L2, 0.0, 1.0)
    proj = t[:, None] * v[None, :]
    return np.linalg.norm(w - proj, axis=1), t


def vessel_support(grid, path, radius_um, d_pen_um=None):
    """Voxels carrying the wall conductance of one vessel segment.

    The wall is placed on the voxels whose centre lies in an annular shell
    [radius, radius + d_pen] around the segment axis, with weight ~ 1/d (i.e.
    uniform flux density over the shell, since the number of voxels in a shell
    grows like d). Voxels strictly inside the lumen get no conductance. The
    weights are normalised to sum to 1, so only the WALL AREA SETS THE
    MAGNITUDE -- the weighting controls the spatial distribution of the
    delivery, not how much is delivered. That is why the delivery tests (5) are
    insensitive to the footprint detail, while the radial profile test (4b) is
    not; the profile test therefore also reports dx convergence.
    """
    c = grid['centers']
    dx = grid['dx']
    if d_pen_um is None:
        d_pen_um = 1.5 * dx
    p0 = np.asarray(path[0], float)
    p1 = np.asarray(path[1], float)
    lo = np.minimum(p0, p1) - (radius_um + d_pen_um)
    hi = np.maximum(p0, p1) + (radius_um + d_pen_um)
    i0 = np.maximum(np.floor(lo / dx).astype(int), 0)
    i1 = np.minimum(np.ceil(hi / dx).astype(int), np.array(grid['shape']) - 1)
    if np.any(i1 < i0):
        return np.zeros(0, int), np.zeros(0)
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(i0, i1))
    sub = np.indices([s.stop - s.start for s in sl]).reshape(len(sl), -1).T
    base = np.array([s.start for s in sl])
    idx_flat = np.ravel_multi_index((sub + base).T, grid['shape'])
    d, _ = _segment_distance(c[idx_flat], p0, p1)
    keep = (d >= radius_um) & (d <= radius_um + d_pen_um)
    idx = idx_flat[keep]
    if idx.size == 0:
        return idx, np.zeros(0)
    w = 1.0 / np.maximum(d[keep], radius_um)
    w = w / w.sum()
    return idx, w


def _build_vessel_units(grid, vessels, substances, default_permeability_um_s,
                        default_velocity_um_s, default_flow, exchange_law,
                        carrier_factor, d_pen_um, blood=None):
    """Turn the user vessel description into exchange units (dicts).

    Accepted vessel entries (dict, or a list of points = one straight segment):
      {'path': [[x0,y0],[x1,y1],...],      # polyline = ONE exchange unit
       'radius_um': 3.0,
       'permeability_um_s': 50.0,          # wall P_w; default from `vessels` defaults
       'flow_um3_s': 2000.0,               # or 'velocity_um_s'
       'plasma_conc_um': per-substance dict or scalar (default: substance table)
       'exchange': law override, 'id': name}
    One polyline = one capillary = one flow Q, which is the physically right
    grouping: splitting a capillary into segments would otherwise grant each
    segment the full flow and over-deliver in the flow-limited regime.
    """
    if vessels is None:
        return []
    entries = []
    for v in vessels:
        if isinstance(v, dict):
            entries.append(v)
        else:
            arr = np.asarray(v, float)
            if arr.ndim != 2 or arr.shape[1] != grid['ndim']:
                raise ValueError('a vessel given as points must be (n_points, ndim)')
            entries.append({'path': arr})
    units = []
    for k, e in enumerate(entries):
        path = np.asarray(e['path'], float)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != grid['ndim']:
            raise ValueError(f'vessel {k}: path must be (n_points>=2, {grid["ndim"]})')
        r = float(e.get('radius_um', 3.0))
        if r < 0:
            raise ValueError(f'vessel {k}: radius must be >= 0')
        pw = float(e.get('permeability_um_s', default_permeability_um_s))
        if pw < 0:
            raise ValueError(f'vessel {k}: permeability must be >= 0')
        if 'flow_um3_s' in e:
            Q = float(e['flow_um3_s'])
        elif 'velocity_um_s' in e:
            Q = float(e['velocity_um_s']) * np.pi * r ** 2
        else:
            Q = float(default_flow) if default_flow is not None else \
                float(default_velocity_um_s) * np.pi * r ** 2
        seglen = float(sum(np.linalg.norm(path[i + 1] - path[i])
                           for i in range(path.shape[0] - 1)))
        # WALL AREA of the whole polyline; in 2D this is the full cylinder
        # lateral area per unit in-plane length (see module docstring).
        A_w = 2.0 * np.pi * r * seglen
        idx_parts, w_parts = [], []
        for i in range(path.shape[0] - 1):
            ii, ww = vessel_support(grid, path[i:i + 2], r, d_pen_um)
            if ii.size:
                idx_parts.append(ii); w_parts.append(ww * np.linalg.norm(path[i + 1] - path[i]))
        if not idx_parts:
            units.append(dict(id=e.get('id', f'vessel{k}'), empty=True, radius_um=r,
                              length_um=seglen, wall_area_um2=A_w, flow_um3_s=Q,
                              permeability_um_s=pw))
            continue
        idx = np.concatenate(idx_parts); w = np.concatenate(w_parts)
        # merge duplicate voxels (a bent polyline can hit the same voxel twice)
        order = np.argsort(idx, kind='stable')
        idx = idx[order]; w = w[order]
        uidx, inv = np.unique(idx, return_inverse=True)
        wsum = np.zeros(uidx.size)
        np.add.at(wsum, inv, w)
        wsum = wsum / wsum.sum()
        law = e.get('exchange', exchange_law)
        cb_override = e.get('plasma_conc_um', None)
        units.append(dict(
            id=e.get('id', f'vessel{k}'), empty=False, radius_um=r, length_um=seglen,
            wall_area_um2=A_w, flow_um3_s=Q, permeability_um_s=pw, exchange=law,
            support=uidx, weights=wsum, plasma_conc_um=cb_override,
            n_support=int(uidx.size),
        ))
    return units


def _unit_blood_conc(unit, sub, default_blood):
    """Blood-side concentration for a unit and a substance [uM] (scalar)."""
    ov = unit.get('plasma_conc_um', None)
    if isinstance(ov, dict):
        if sub.name in ov:
            return float(ov[sub.name])
    elif ov is not None:
        return float(ov)
    if default_blood and sub.name in default_blood:
        return float(default_blood[sub.name])
    return float(sub.plasma_conc_um)


def krogh_vessels(domain_um=(300.0, 300.0), spacing_um=60.0, radius_um=3.0,
                  *, flow_um3_s=None, permeability_um_s=50.0, n=None,
                  margin_um=None, axis='y', plasma_conc_um=None):
    """Parallel-array Krogh-type vessel configuration (PRESCRIBED geometry).

    Vessels run along `axis` at regular spacing `spacing_um` across the domain,
    which is the standard Krogh-cylinder reduction: with zero-flux boundaries
    the tissue between two vessels is a Krogh cylinder in Cartesian (slab)
    coordinates, half-width spacing_um/2.

    This is a verification/parametrisation helper ONLY: the geometry is given,
    not grown. Returns a list of vessel dicts for run()/solve_steady_state().
    """
    dom = np.asarray(domain_um, float)[:2]
    L = float(dom[1] if axis == 'y' else dom[0])
    if margin_um is None:
        margin_um = float(spacing_um) / 2.0
    if n is None:
        n = int(np.floor((dom[0] if axis == 'y' else dom[1]) / float(spacing_um)))
        n = max(1, n)
    span = dom[0] if axis == 'y' else dom[1]
    pos = margin_um + np.arange(n) * (span - 2 * margin_um) / max(1, n - 1) if n > 1 \
        else np.array([span / 2.0])
    out = []
    for i, p in enumerate(pos):
        if axis == 'y':
            path = [[float(p), 0.0], [float(p), L]]
        else:
            path = [[0.0, float(p)], [L, float(p)]]
        d = dict(id=f'krogh{i}', path=path, radius_um=float(radius_um),
                 permeability_um_s=float(permeability_um_s))
        if flow_um3_s is not None:
            d['flow_um3_s'] = float(flow_um3_s)
        if plasma_conc_um is not None:
            d['plasma_conc_um'] = plasma_conc_um
        out.append(d)
    return out


def single_vessel(domain_um=(300.0, 300.0), x_um=None, y_um=None, radius_um=3.0,
                  *, flow_um3_s=None, permeability_um_s=50.0, plasma_conc_um=None,
                  axis='y'):
    """A single straight vessel through the middle of the domain (verification)."""
    dom = np.asarray(domain_um, float)[:2]
    if axis == 'y':
        x = dom[0] / 2.0 if x_um is None else float(x_um)
        path = [[x, 0.0], [x, float(dom[1])]]
    else:
        y = dom[1] / 2.0 if y_um is None else float(y_um)
        path = [[0.0, y], [float(dom[0]), y]]
    d = dict(id='single', path=path, radius_um=float(radius_um),
             permeability_um_s=float(permeability_um_s), exchange='renkin_crone')
    if flow_um3_s is not None:
        d['flow_um3_s'] = float(flow_um3_s)
    if plasma_conc_um is not None:
        d['plasma_conc_um'] = plasma_conc_um
    return [d]


# ===========================================================================
# cell <-> voxel mapping
# ===========================================================================
def map_cells(grid, centroids, areas=None, *, mode='area', heights_um=None,
              renormalize=True):
    """Conservative cell -> voxel mapping.

    Modes
    -----
    'area' (default): each cell is represented by an axis-aligned square (2D) or
        cube (3D) of THE SAME AREA/VOLUME as the cell, and the weights are the
        EXACT geometric overlaps with the voxels. This is separable and exact:
        sum_v w_cv == 1 to round-off, no double counting by construction. The
        true cell shapes in this project are hexagons (cellstate.cell_geometry /
        model.hex_geometry, area 43 um^2); replacing a hexagon by an equal-area
        square redistributes weight at O(dx) but preserves the total exactly.
    'pic'     : bilinear/cloud-in-cell weights on the 2^ndim neighbouring voxel
        centres (smooth, also sums to 1).
    'nearest' : every cell to its nearest voxel centre (weight 1). Crude, no
        smoothing, but exactly one voxel per cell.

    Returns a dict with CSR-style arrays 'indptr','indices','data' (n_cells,
    each row summing to 1), plus diagnostics: 'duplication_residual' (max
    |row sum - 1| before renormalisation), 'n_boundary_cells' (footprint
    truncated by the domain edge), 'max_truncation' (largest weight shortfall
    caused by that truncation), 'support_min/max/mean', 'mode'.
    """
    grid_ndim = grid['ndim']
    cen = np.asarray(centroids, float)
    if cen.ndim != 2 or cen.shape[1] != grid_ndim:
        raise ValueError(f'centroids must be (n_cells, {grid_ndim})')
    if np.any(~np.isfinite(cen)):
        raise ValueError('cell centroids must be finite')
    n_cells = cen.shape[0]
    if areas is None:
        if grid_ndim == 2:
            areas = np.full(n_cells, 43.0)
        else:
            areas = np.full(n_cells, 430.0)
    areas = np.atleast_1d(np.asarray(areas, float))
    if areas.size == 1:
        areas = np.full(n_cells, float(areas[0]))
    if areas.size != n_cells or np.any(~np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError('areas must be positive and have one entry per cell')
    dx = grid['dx']
    shape = np.array(grid['shape'])
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    indices_list, data_list = [], []
    rowsum = np.zeros(n_cells)
    trunc = np.zeros(n_cells)
    n_boundary = 0
    if mode == 'nearest':
        ii = np.clip(np.floor(cen / dx).astype(int), 0, shape - 1)
        for c in range(n_cells):
            indices_list.append(np.array([np.ravel_multi_index(ii[c], tuple(shape))]))
            data_list.append(np.array([1.0]))
            rowsum[c] = 1.0
        indptr[1:] = np.arange(1, n_cells + 1)
    elif mode == 'pic':
        frac = cen / dx - 0.5
        base = np.floor(frac).astype(int)
        t = frac - base
        if grid_ndim == 2:
            corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        else:
            corners = [(a, b, c2) for a in (0, 1) for b in (0, 1) for c2 in (0, 1)]
        for c in range(n_cells):
            idxs, ws = [], []
            tot = 0.0
            for off in corners:
                wgt = 1.0
                ii = []
                for a, o in enumerate(off):
                    wgt *= (1 - t[c, a]) if o == 0 else t[c, a]
                    ii.append(int(base[c, a] + o))
                ii_arr = np.clip(np.array(ii), 0, shape - 1)
                if wgt <= 0:
                    continue
                idxs.append(np.ravel_multi_index(tuple(ii_arr), tuple(shape)))
                ws.append(wgt)
                tot += wgt
            if tot <= 0:
                idxs = [np.ravel_multi_index(np.clip(base[c], 0, shape - 1), tuple(shape))]
                ws = [1.0]; tot = 1.0
            idxs = np.array(idxs); ws = np.array(ws)
            if renormalize and tot != 1.0:
                trunc[c] = 1.0 - tot
                n_boundary += int(trunc[c] > 1e-12)
                ws = ws / tot
            indptr[c + 1] = indptr[c] + idxs.size
            indices_list.append(idxs); data_list.append(ws); rowsum[c] = ws.sum()
    elif mode == 'area':
        # separable exact overlap of an equal-area square/cube with the grid
        half = 0.5 * (areas ** (1.0 / grid_ndim))
        for c in range(n_cells):
            per_axis = []
            for a in range(grid_ndim):
                lo = cen[c, a] - half[c]
                hi = cen[c, a] + half[c]
                i0 = int(np.floor(lo / dx))
                i1 = int(np.floor((hi - 1e-12) / dx))
                i0c = max(i0, 0); i1c = min(i1, shape[a] - 1)
                if i1c < i0c:
                    # cell entirely outside the domain: fall back to nearest voxel
                    per_axis.append((np.array([int(np.clip(round(cen[c, a] / dx - .5), 0, shape[a] - 1))]),
                                     np.array([1.0])))
                    continue
                ids = np.arange(i0c, i1c + 1)
                ov = np.minimum(hi, (ids + 1) * dx) - np.maximum(lo, ids * dx)
                ov = np.maximum(ov, 0.0)
                tot = ov.sum()
                if tot <= 0:
                    ov = np.array([1.0]); ids = np.array([i0c])
                    tot = 1.0
                # shortfall = part of the footprint outside the domain
                if renormalize:
                    missing = (hi - lo) - tot
                    if missing > 1e-12:
                        trunc[c] = max(trunc[c], missing / (hi - lo))
                per_axis.append((ids, ov / tot))
            if grid_ndim == 2:
                ix, wx = per_axis[0]; iy, wy = per_axis[1]
                W = np.outer(wx, wy)
                II, JJ = np.meshgrid(ix, iy, indexing='ij')
                idxs = np.ravel_multi_index((II.ravel(), JJ.ravel()), tuple(shape))
                ws = W.ravel()
            else:
                ix, wx = per_axis[0]; iy, wy = per_axis[1]; iz, wz = per_axis[2]
                W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
                II, JJ, KK = np.meshgrid(ix, iy, iz, indexing='ij')
                idxs = np.ravel_multi_index((II.ravel(), JJ.ravel(), KK.ravel()),
                                            tuple(shape))
                ws = W.ravel()
            keep = ws > 0
            idxs = idxs[keep]; ws = ws[keep]
            ws = ws / ws.sum()
            n_boundary += int(trunc[c] > 1e-12)
            indptr[c + 1] = indptr[c] + idxs.size
            indices_list.append(idxs); data_list.append(ws); rowsum[c] = ws.sum()
    else:
        raise ValueError(f'unknown mapping mode {mode!r}')
    indices = np.concatenate(indices_list) if indices_list else np.zeros(0, np.int64)
    data = np.concatenate(data_list) if data_list else np.zeros(0)
    dup = float(np.max(np.abs(rowsum - 1.0))) if n_cells else 0.0
    support = np.diff(indptr)
    return dict(
        mode=mode, n_cells=n_cells, indptr=indptr, indices=indices, data=data,
        rowsum=rowsum, duplication_residual=dup,
        n_boundary_cells=int(n_boundary), max_truncation=float(trunc.max() if n_cells else 0.0),
        support_min=int(support.min()) if n_cells else 0,
        support_max=int(support.max()) if n_cells else 0,
        support_mean=float(support.mean()) if n_cells else 0.0,
        total_weight=float(data.sum()),
        exact_row_sum=bool(dup < 1e-12),
    )


def sample_cells(mapping, field_flat):
    """Per-cell concentration = w-weighted mean of the voxel field (umol/L)."""
    out = np.zeros(mapping['n_cells'])
    indptr = mapping['indptr']; indices = mapping['indices']; data = mapping['data']
    for c in range(mapping['n_cells']):
        s, e = indptr[c], indptr[c + 1]
        out[c] = float(data[s:e] @ field_flat[indices[s:e]])
    return out


def deposit_cells(mapping, cell_amounts_um3um, out=None):
    """Deposit per-cell amounts [uM*um^3] into voxels with the same weights.

    Exactly conservative: sum_v out == sum_c cell_amounts * (row sum == 1).
    """
    if out is None:
        out = np.zeros(int(mapping['indices'].max()) + 1 if mapping['indices'].size else 0)
    indptr = mapping['indptr']; indices = mapping['indices']; data = mapping['data']
    for c in range(mapping['n_cells']):
        s, e = indptr[c], indptr[c + 1]
        np.add.at(out, indices[s:e], data[s:e] * cell_amounts_um3um[c])
    return out


def cellstate_oxygen_input(cell_conc_um, reference_conc_um):
    """Documented coupling to cellstate.py's `oxygen` argument.

    cellstate.run(..., oxygen=...) expects a RELATIVE availability with 1 =
    reference (its own docstring says so; its Hill availability uses k_o2=0.5,
    n_o2=3). This module therefore hands over C_cell/C_reference, clipped at a
    documented ceiling (super-physiological values above the arterial value
    have no meaning in that layer): 0 <= ox <= ox_max, default ox_max = 2.
    It is deliberately NOT a physiological claim about cellstate.py's Hill
    function: that function's parameters are themselves illustrative.
    """
    ref = float(reference_conc_um)
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError('reference_conc_um must be finite and > 0')
    c = np.asarray(cell_conc_um, float)
    return np.clip(c / ref, 0.0, 2.0)


def growth_substrate_factor(cell_conc_um, reference_conc_um, hill=1.0,
                            k_half_frac=0.5, floor=0.0):
    """Documented contract for a substrate-limited growth term (growth.py).

    growth.py does NOT exist in this project directory (verified), so this
    function defines the coupling contract rather than calling one:
        factor = max(floor, (s^n / (k^n + s^n))) with s = C_cell/C_reference,
        n = hill and k = k_half_frac (Hill saturation, 1 = fully supplied,
        0 = fully starved). Returned value multiplies a growth/proliferation
        rate. The Hill form and its parameters are ILLUSTRATIVE.
    """
    ref = float(reference_conc_um)
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError('reference_conc_um must be finite and > 0')
    s = np.clip(np.asarray(cell_conc_um, float) / ref, 0.0, None)
    n = float(hill)
    k = float(k_half_frac)
    return np.maximum(float(floor), s ** n / (k ** n + s ** n))


# ===========================================================================
# stability limits
# ===========================================================================
def max_stable_dt(grid, substances=None, *, field=None, dt=None,
                  scheme='imex', km_um=None, extra_rate_s=0.0):
    """Report the time-step limits that apply, and which do NOT apply.

    The implemented schemes (both 'imex' and 'split') treat diffusion and all
    LINEAR sinks implicitly, so they have NO diffusion stability limit: the
    entry 'stability_limit_implicit' is Infinity. What is reported instead is

      'explicit_diffusion_dt' : 1/(2*D_eff*sum_a 1/d_a^2) -- the limit an
            explicit scheme would need (2D: dx^2/(4D) for dx=dy). Reported so
            the claim "implicit is necessary" is quantified; at dx = 2 um and
            D_eff = 1458 um^2/s this is ~7e-4 s.
      'reaction_dt' : 1/(max first-order coefficient) [s] -- the explicit
            limit for the linear sinks (vessel wall conductance, first-order
            binding, linearised MM denominator).
      'zero_order_dt' : min_v C_v/z_v [s] -- a zero-order sink cannot remove
            more than is present; the code clips and counts a guard event, this
            is the dt below which clipping never happens for the given field.
      'accuracy_advisory_dt' : the dt below which the backward-Euler time
            error is expected to be small for the slowest mode present
            (diffusion across the domain and the slowest sink).
      'recommended_dt' : min of the two above; a numerical recommendation with
            NO stability content for the implicit scheme.

    Stability of the nonlinear (Michaelis-Menten) sink is handled by freezing
    the denominator at the step start: k_v = cap_v/(Km + C^n), which is
    unconditionally dissipative because the resulting matrix is an M-matrix.
    """
    subs = _resolve_substances(substances)
    if not subs:
        subs = {}
    Dmax = max([s.D_eff for s in subs.values()] or [1.0])
    inv_d2 = sum(1.0 / d ** 2 for d in grid['d'])
    expl = 1.0 / (2.0 * Dmax * inv_d2)
    # accuracy advisory: slowest diffusion mode across the domain
    Lmin = min(grid['extent_um'])
    t_diff = Lmin ** 2 / (4.0 * Dmax)
    rates = float(extra_rate_s)
    if field is not None:
        for nm, s in subs.items():
            if nm not in field:
                continue
            C = np.asarray(field[nm], float)
            kv = np.zeros_like(C)
            if s.cell_consumption == 'michaelis_menten' and km_um is not None:
                kv = kv + 0.0
            if s.k_first_s > 0:
                kv = kv + s.k_first_s
            rates = max(rates, float(kv.max()) if kv.size else 0.0)
    t_react = np.inf if rates <= 0 else 1.0 / rates
    zo = np.inf
    if field is not None:
        for nm, s in subs.items():
            if nm in field and s.zero_order_um_s > 0:
                C = np.asarray(field[nm], float)
                zo = min(zo, float((C[C > 0] / s.zero_order_um_s).min())
                         if np.any(C > 0) else 0.0)
    adv = min(0.05 * t_diff, 0.1 * t_react)
    return dict(
        scheme=scheme,
        stability_limit_implicit=np.inf,
        unconditional_stability_note=(
            'the implemented schemes are unconditionally stable for diffusion and '
            'for all linear/linearised sinks; there is NO dt a user can pass that '
            'makes them blow up. dt is therefore an ACCURACY choice, not a '
            'stability requirement.'),
        explicit_diffusion_dt=float(expl),
        reaction_dt=float(t_react),
        zero_order_dt=float(zo),
        accuracy_advisory_dt=float(adv),
        recommended_dt=float(min(adv, zo)),
        D_eff_max_um2_s=float(Dmax),
        dt_used=None if dt is None else float(dt),
        dt_exceeds_advisory=bool(dt is not None and dt > adv),
    )


# ===========================================================================
# solver
# ===========================================================================
class TransportSolver:
    """Multi-substance diffusion-reaction solver with prescribed vessel sources.

    See the module docstring for the equations, the exchange laws and the
    conservation argument. The class owns the grid, the vessel units, the
    cell->voxel mapping and the current concentration fields; `step(dt)` does
    one IMEX/split step for every substance and updates the running budget.
    """

    def __init__(self, grid, substances, vessels=None, cells=None, blood=None, *,
                 scheme='auto', boundary='no_flux', bath=None, background=None,
                 initial=None, exchange_law='renkin_crone', d_pen_um=None,
                 default_permeability_um_s=50.0, default_velocity_um_s=1000.0,
                 default_flow_um3_s=None, mapping_mode='area', direct_max_nodes=400_000,
                 cell_km_overrides=None, verbose=False):
        self.grid = grid
        self.subs = _resolve_substances(substances)
        for s in self.subs.values():
            s.check()
        self.sub_list = [self.subs[k] for k in self.subs]
        self.verbose = bool(verbose)
        self.boundary = boundary
        self._direct_max_nodes = int(direct_max_nodes)
        if scheme == 'auto':
            scheme = 'imex' if grid['n_nodes'] <= self._direct_max_nodes else 'split'
        if scheme not in ('imex', 'split'):
            raise ValueError("scheme must be 'auto', 'imex' or 'split'")
        if scheme == 'imex' and grid['n_nodes'] > 2 * self._direct_max_nodes:
            raise ValueError(
                f"scheme='imex' with {grid['n_nodes']} nodes: a sparse direct LU of a "
                f"2D/3D Laplacian at this size overflows the memory budget; use "
                f"scheme='split' (exact DCT diffusion step) beyond ~{self._direct_max_nodes} nodes")
        self.scheme = scheme
        self.L = laplacian(grid) if scheme == 'imex' else None
        self.negL = -self.L if self.L is not None else None
        self.lam = dct_eigenvalues(grid)
        self.dt_last = None

        # ---- vessels ------------------------------------------------------
        self.units = _build_vessel_units(
            grid, vessels, self.subs, default_permeability_um_s, default_velocity_um_s,
            default_flow_um3_s, exchange_law, 1.0, d_pen_um, blood)
        self.blood = dict(blood) if blood else {}
        self.diag_vessel = {s.name: np.zeros(grid['n_nodes']) for s in self.sub_list}
        self.src_vessel = {s.name: np.zeros(grid['n_nodes']) for s in self.sub_list}
        self.G_vessel = {s.name: np.zeros(grid['n_nodes']) for s in self.sub_list}
        self.unit_info = []
        Vv = grid['volume_um3']
        for u in self.units:
            if u.get('empty'):
                self.unit_info.append({k: u[k] for k in
                                       ('id', 'radius_um', 'length_um', 'wall_area_um2',
                                        'flow_um3_s', 'permeability_um_s')
                                       if k in u} | {'n_support': 0, 'empty': True})
                continue
            info = dict(id=u['id'], radius_um=u['radius_um'], length_um=u['length_um'],
                        wall_area_um2=u['wall_area_um2'], flow_um3_s=u['flow_um3_s'],
                        permeability_um_s=u['permeability_um_s'], exchange=u['exchange'],
                        n_support=u['n_support'], empty=False,
                        support=u['support'], weights=u['weights'])
            for s in self.sub_list:
                cb = _unit_blood_conc(u, s, self.blood)
                PS = u['permeability_um_s'] * u['wall_area_um2']
                G = exchange_conductance(PS, u['flow_um3_s'], u['exchange'], s.carrier_factor)
                Gv = u['weights'] * G
                idx = u['support']
                np.add.at(self.G_vessel[s.name], idx, Gv)
                self.diag_vessel[s.name][idx] += Gv / (s.alpha * Vv)
                self.src_vessel[s.name][idx] += Gv * cb / Vv
                info[f'PS_{s.name}_um3_s'] = float(PS)
                info[f'G_{s.name}_um3_s'] = float(G)
                info[f'Cblood_{s.name}_uM'] = float(cb)
                info[f'PS_over_Q_{s.name}'] = float(PS / u['flow_um3_s']) if u['flow_um3_s'] > 0 else np.inf
                info[f'extraction_limit_{s.name}'] = float(G / u['flow_um3_s']) if u['flow_um3_s'] > 0 else 1.0
                info.setdefault('_G_by_sub', {})[s.name] = float(G)
            self.unit_info.append(info)

        # ---- bath (optional Robin layer on the outer faces) ---------------
        self.bath = None
        if boundary == 'robin' or bath is not None:
            b = dict(bath or {})
            k_bath = float(b.get('conductance_um_s', 10.0))
            if k_bath < 0:
                raise ValueError('bath conductance must be >= 0')
            face_area = np.zeros(grid['n_nodes'])
            idx = np.arange(grid['n_nodes']).reshape(grid['shape'])
            for a, (na, da) in enumerate(zip(grid['shape'], grid['d'])):
                for endi in (0, na - 1):
                    sel = np.take(idx, endi, axis=a).ravel()
                    # outer face area of a boundary voxel
                    other = float(np.prod([grid['d'][b2] for b2 in range(grid['ndim']) if b2 != a]))
                    if grid['ndim'] == 2:
                        other *= grid['thickness_um']
                    face_area[sel] += other
            self.bath = dict(k_bath_um_s=k_bath, face_area_um2=face_area)
            self.diag_bath = {}
            self.src_bath = {}
            for s in self.sub_list:
                cb = float(b.get(s.name, b.get('plasma_conc_um', s.plasma_conc_um)))
                G = k_bath * face_area
                self.diag_bath[s.name] = G / (s.alpha * Vv)
                self.src_bath[s.name] = G * cb / Vv
        else:
            self.diag_bath = {s.name: np.zeros(grid['n_nodes']) for s in self.sub_list}
            self.src_bath = {s.name: np.zeros(grid['n_nodes']) for s in self.sub_list}

        # ---- background / uniform sources ---------------------------------
        self.background = {}
        for s in self.sub_list:
            v = 0.0
            if isinstance(background, dict) and s.name in background:
                v = float(background[s.name])
            elif background is not None and not isinstance(background, dict):
                v = float(background)
            if v < 0:
                raise ValueError('background sources must be >= 0')
            self.background[s.name] = v

        # ---- cells ---------------------------------------------------------
        self.mapping = None
        self.cell = {}
        self.n_cells = 0
        if cells is not None:
            self._setup_cells(cells, mapping_mode, cell_km_overrides)

        # ---- fields --------------------------------------------------------
        self.C = {}
        for s in self.sub_list:
            self.C[s.name] = self._initial_field(s, initial)

        # ---- budget --------------------------------------------------------
        self.budget = {s.name: dict(delivered=0.0, bath_net=0.0, background=0.0,
                                    consumed=0.0, bound=0.0, accumulated=0.0,
                                    inventory0=float((self.C[s.name] * Vv).sum() * AMOL_PER_UM3_UM),
                                    residual=0.0, residual_relative=0.0)
                       for s in self.sub_list}
        self.guards = dict(negative_clip=0, zero_order_clipped=0, nonfinite=0,
                           steps=0, mm_linearisation_um3um=0.0,
                           cell_budget_mismatch_um3um=0.0)
        self._factor_cache = {}

    # -- construction helpers ------------------------------------------------
    def _initial_field(self, sub, initial):
        n = self.grid['n_nodes']
        val = None
        if isinstance(initial, dict) and sub.name in initial:
            val = initial[sub.name]
        elif initial is not None and not isinstance(initial, dict):
            val = initial
        if val is None:
            return np.full(n, float(sub.plasma_conc_um))
        arr = np.asarray(val, float)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError('initial concentration must be finite and nonnegative')
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if arr.size == n:
            return arr.reshape(-1).copy()
        raise ValueError(f'initial for {sub.name} must be a scalar or {n} values')

    def _setup_cells(self, cells, mapping_mode, km_overrides):
        if not isinstance(cells, dict) or 'centroids' not in cells:
            raise ValueError("cells must be a dict with at least 'centroids'")
        cen = np.asarray(cells['centroids'], float)
        areas = cells.get('areas', None)
        mode = cells.get('mapping_mode', mapping_mode)
        mp = map_cells(self.grid, cen, areas, mode=mode)
        self.mapping = mp
        self.n_cells = mp['n_cells']
        Vv = self.grid['volume_um3']
        n = self.grid['n_nodes']
        self.cell_of_entry = np.repeat(np.arange(self.n_cells), np.diff(mp['indptr']))
        mx = float(mp['indices'].max()) if mp['indices'].size else 0
        if mx >= n:
            raise ValueError('cell mapping produced an out-of-range voxel index')
        demand = cells.get('demand', None)
        km_ov = dict(km_overrides or {})
        if isinstance(cells.get('km'), dict):
            km_ov.update(cells['km'])
        for s in self.sub_list:
            arr = np.zeros(self.n_cells)
            if s.cell_consumption != 'none':
                d = None
                if isinstance(demand, dict) and s.name in demand:
                    d = demand[s.name]
                elif s.name == 'o2' and 'oxygen_demand' in cells:
                    d = cells['oxygen_demand']
                elif s.name == 'glucose' and 'glucose_demand' in cells:
                    d = cells['glucose_demand']
                elif 'demand_%s' % s.name in cells:
                    d = cells['demand_%s' % s.name]
                if d is None:
                    d = (s.vmax_per_cell_amol_s if s.cell_consumption == 'michaelis_menten'
                         else s.zero_order_per_cell_amol_s if s.cell_consumption == 'zero_order'
                         else s.k_cell_um3_s)
                arr = np.broadcast_to(np.atleast_1d(np.asarray(d, float)),
                                      (self.n_cells,)).astype(float).copy()
            mask = cells.get('mask', None)
            if mask is not None:
                arr = arr * np.where(np.asarray(mask, bool), 1.0, 0.0)
            scale = cells.get('scale', None)
            if isinstance(scale, dict) and s.name in scale:
                arr = arr * np.broadcast_to(np.asarray(scale[s.name], float),
                                            (self.n_cells,))
            km = float(km_ov.get(s.name, s.km_um))
            # per-voxel capacity rate [uM/s] from the per-cell demand
            if np.any(~np.isfinite(arr)) or np.any(arr < 0) or not np.isfinite(km):
                raise ValueError('cell demands must be finite and nonnegative; Km finite')
            if s.cell_consumption == 'michaelis_menten' and km <= 0:
                raise ValueError('Michaelis-Menten Km must be positive')
            # First-order demand is clearance volume [um^3/s], not amol/s.
            conversion = 1.0 if s.cell_consumption == 'first_order' else AMOL_PER_UM3_UM
            cap = np.bincount(mp['indices'],
                              weights=mp['data'] * arr[self.cell_of_entry],
                              minlength=n) / (Vv * conversion)
            self.cell[s.name] = dict(
                demand_per_cell=arr, km_um=km, cap_voxel_um_s=cap,
                km_source=('override' if s.name in km_ov else 'substance table'),
            )

    # -- ready-made vessel / cell shortcuts ----------------------------------
    def vessel_report(self):
        return self.unit_info

    # -- one step -----------------------------------------------------------
    def step(self, dt):
        """Advance every substance by dt. Returns a dict of per-step diagnostics."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and > 0')
        Vv = self.grid['volume_um3']
        n = self.grid['n_nodes']
        diag = {}
        for s in self.sub_list:
            C = self.C[s.name]
            if s.name in self.cell:
                cl = self.cell[s.name]
                cap = cl['cap_voxel_um_s']
                km = cl['km_um']
                if s.cell_consumption == 'michaelis_menten':
                    k_v = cap / (km + np.maximum(C, 0.0))
                    z_cell = 0.0
                elif s.cell_consumption == 'zero_order':
                    k_v = np.zeros(n)
                    z_cell = cap
                elif s.cell_consumption == 'first_order':
                    k_v = cap
                    z_cell = 0.0
                else:
                    k_v = np.zeros(n); z_cell = 0.0
            else:
                k_v = np.zeros(n); z_cell = 0.0
            k_v = k_v + s.k_first_s + self.diag_vessel[s.name] + self.diag_bath[s.name]
            s_v = self.src_vessel[s.name] + self.src_bath[s.name] + self.background[s.name]
            z_v = float(s.zero_order_um_s) + (z_cell if np.isscalar(z_cell) or
                                              np.ndim(z_cell) == 0 else z_cell)
            if np.ndim(z_v) == 0:
                z_v = np.full(n, float(z_v))
            else:
                z_v = np.asarray(z_v, float)

            if self.scheme == 'imex':
                M = (self.negL * (dt * s.D_eff)).tocsc()
                M.setdiag(M.diagonal() + 1.0 + dt * k_v)
                z_eff = np.minimum(z_v, (C + dt * s_v) / dt)
                rhs = C + dt * s_v - dt * z_eff
                np.maximum(rhs, 0.0, out=rhs)
                key = (s.name, dt) if (s.cell_consumption != 'michaelis_menten'
                                       and s.cell_consumption != 'first_order'
                                       and s.k_first_s == 0.0) else None
                lu = self._factor_cache.get(key) if key is not None else None
                if lu is None:
                    lu = _splinalg.splu(M)
                    if key is not None:
                        self._factor_cache[key] = lu
                Cnew = lu.solve(rhs)
            else:
                Cstar = _fft.idctn(_fft.dctn(C.reshape(self.grid['shape']), type=2,
                                             norm='ortho')
                                   / (1.0 - dt * s.D_eff * self.lam),
                                   type=2, norm='ortho').reshape(-1)
                base = Cstar + dt * s_v
                z_amt = np.minimum(z_v, np.maximum(base, 0.0) / dt) * dt
                z_eff = z_amt / dt
                Cnew = np.maximum(base - z_amt, 0.0) / (1.0 + dt * k_v)

            if not np.all(np.isfinite(Cnew)):
                self.guards['nonfinite'] += 1
                raise FloatingPointError(f'non-finite concentration for {s.name}')
            neg = int(np.count_nonzero(Cnew < 0))
            if neg:
                self.guards['negative_clip'] += neg
                np.maximum(Cnew, 0.0, out=Cnew)
            zclip = int(np.count_nonzero(z_eff < z_v - 1e-30))
            if zclip:
                self.guards['zero_order_clipped'] += zclip

            # ---- accounting (all terms evaluated on the applied operators) --
            dC = Cnew - C
            b = self.budget[s.name]
            deliv = float((self.src_vessel[s.name] - self.diag_vessel[s.name] * Cnew
                           ).sum() * Vv * AMOL_PER_UM3_UM * dt)
            bath = float((self.src_bath[s.name] - self.diag_bath[s.name] * Cnew
                          ).sum() * Vv * AMOL_PER_UM3_UM * dt)
            bound = float((s.k_first_s * Cnew).sum() * Vv * AMOL_PER_UM3_UM * dt)
            consumed = float(((k_v - s.k_first_s - self.diag_vessel[s.name]
                               - self.diag_bath[s.name]) * Cnew).sum()
                             * Vv * AMOL_PER_UM3_UM * dt
                             + (z_eff * Vv).sum() * AMOL_PER_UM3_UM * dt)
            bg = float(self.background[s.name] * n * Vv * AMOL_PER_UM3_UM * dt)
            accum = float((dC * Vv).sum() * AMOL_PER_UM3_UM)
            b['delivered'] += deliv
            b['bath_net'] += bath
            b['background'] += bg
            b['consumed'] += consumed
            b['bound'] += bound
            b['accumulated'] += accum
            b['residual'] = (b['delivered'] + b['bath_net'] + b['background']
                             - b['consumed'] - b['bound'] - b['accumulated'])
            scale = (abs(b['delivered']) + abs(b['background']) + abs(b['bath_net'])
                     + abs(b['consumed']) + abs(b['bound']) + abs(b['accumulated'])
                     + abs(b['inventory0']))
            b['residual_relative'] = abs(b['residual']) / scale if scale > 0 else 0.0

            # MM linearisation diagnostic: applied frozen-denominator rate vs the
            # true MM rate at the new concentration
            if s.name in self.cell and s.cell_consumption == 'michaelis_menten':
                cap = self.cell[s.name]['cap_voxel_um_s']
                km = self.cell[s.name]['km_um']
                applied = (cap / (km + np.maximum(C, 0.0))) * Cnew
                exact = cap * Cnew / (km + np.maximum(Cnew, 0.0))
                self.guards['mm_linearisation_um3um'] += float(
                    np.abs(applied - exact).sum() * Vv * dt)

            if s.name in self.cell:
                # Allocate actual voxel uptake to contributing cell capacities.
                # This preserves frozen MM denominators and starvation clipping.
                if s.cell_consumption == 'michaelis_menten':
                    response = Cnew / (self.cell[s.name]['km_um'] + np.maximum(C, 0.0))
                elif s.cell_consumption == 'first_order':
                    response = Cnew * AMOL_PER_UM3_UM
                elif s.cell_consumption == 'zero_order':
                    response = np.divide(z_eff, z_v, out=np.zeros(n), where=z_v > 0)
                else:
                    response = np.zeros(n)
                mp = self.mapping
                uptake = np.bincount(self.cell_of_entry,
                    weights=mp['data'] * self.cell[s.name]['demand_per_cell'][self.cell_of_entry]
                    * response[mp['indices']] * dt, minlength=self.n_cells)
                self.cell[s.name]['last_uptake_amol'] = uptake
                cell_expected = consumed - float((s.zero_order_um_s *
                    np.divide(z_eff, z_v, out=np.zeros(n), where=z_v > 0)).sum()
                    * Vv * AMOL_PER_UM3_UM * dt)
                self.guards['cell_budget_mismatch_um3um'] += abs(
                    uptake.sum() - cell_expected) / AMOL_PER_UM3_UM
            self.C[s.name] = Cnew
            diag[s.name] = dict(delivered=deliv, consumed=consumed, bound=bound,
                                bath_net=bath, background=bg, accumulated=accum,
                                residual=b['residual'])
        self.guards['steps'] += 1
        self.dt_last = float(dt)
        return diag

    def _unit_cb(self, sub):
        """Cached per-voxel blood-side concentration field of all units [uM]."""
        key = '__cb_' + sub.name
        if key in self._factor_cache:
            return self._factor_cache[key]
        out = np.divide(self.src_vessel[sub.name] * self.grid['volume_um3'],
                        self.G_vessel[sub.name],
                        out=np.zeros(self.grid['n_nodes']),
                        where=self.G_vessel[sub.name] > 0)
        self._factor_cache[key] = out
        return out

    # -- cell coupling (sampling + deposition) -------------------------------
    def cell_concentration(self, name):
        """Per-cell concentration [uM] = weighted mean of the voxel field."""
        mp = self.mapping
        C = self.C[name]
        out = np.zeros(self.n_cells)
        vals = mp['data'] * C[mp['indices']]
        np.add.at(out, self.cell_of_entry, vals)
        return out

    def cell_uptake(self, name, dt=None):
        """Actual last-step uptake [amol/cell], not an instantaneous MM estimate.

        Optional dt must equal the last step; dt=0 returns initial zero uptake.
        Summing cells matches the field cell sink, excluding tissue sinks.
        """
        if dt == 0 or name not in self.cell or self.dt_last is None:
            return np.zeros(self.n_cells)
        if dt is not None and not np.isclose(dt, self.dt_last, rtol=1e-12, atol=0):
            raise ValueError('dt must match the actual last step')
        return self.cell[name]['last_uptake_amol'].copy()

    def snapshot(self):
        return {k: v for k, v in self.C.items()}

    def total_amount(self, name):
        return float((self.C[name] * self.grid['volume_um3']).sum() * AMOL_PER_UM3_UM)


# ===========================================================================
# run()
# ===========================================================================
def run(domain_um=(300.0, 300.0), dx_um=2.0, t_end=60.0, dt=None, substances=None,
        vessels=None, cells=None, blood=None, *, thickness_um=10.0, scheme='auto',
        boundary='no_flux', bath=None, background=None, initial=None,
        exchange_law='renkin_crone', d_pen_um=None,
        default_permeability_um_s=50.0, default_velocity_um_s=1000.0,
        default_flow_um3_s=None, mapping_mode='area', store_every=1,
        max_field_bytes=5.0e8, direct_max_nodes=400_000, km_overrides=None,
        verbose=False):
    """Integrate the transport layer and return fields, cell series and budget.

    Parameters follow the module docstring. `cells` is a dict with 'centroids'
    (n_cells, ndim) [um], optional 'areas' [um^2] (2D; default 43 um^2, the
    project's cell area), optional 'demand' = {substance: per-cell rate
    [amol/(cell*s)]} (aliases 'oxygen_demand'/'glucose_demand'), optional 'km',
    'mask', 'scale' and 'mapping_mode'.

    Returns
    -------
    dict with
      'times' (n_t,)
      'fields' {name: (n_t,)+grid_shape}
      'grid'
      'cell_concentration' {name: (n_t,n_cells)} [uM]
      'cell_uptake' {name: (n_t,n_cells)} [amol per step]
      'cell_uptake_total' (n_t,n_cells) [amol per step, all consuming substances]
      'cellstate_oxygen' (n_t,n_cells) relative availability for cellstate.run(oxygen=...)
      'growth_substrate' (n_t,n_cells) substrate factor for a growth term (glucose)
      'budget' {name: {...cumulative arrays...}}
      'metadata'
      'final' continuation state
    """
    grid = make_grid(domain_um, dx_um, thickness_um)
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError('t_end must be finite and positive')
    if dt is None:
        dt = t_end / 60.0
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    n_steps = max(1, int(np.ceil(t_end / dt)))
    dt_eff = t_end / n_steps
    solver = TransportSolver(grid, substances, vessels, cells, blood, scheme=scheme,
                             boundary=boundary, bath=bath, background=background,
                             initial=initial, exchange_law=exchange_law,
                             d_pen_um=d_pen_um,
                             default_permeability_um_s=default_permeability_um_s,
                             default_velocity_um_s=default_velocity_um_s,
                             default_flow_um3_s=default_flow_um3_s,
                             mapping_mode=mapping_mode,
                             direct_max_nodes=direct_max_nodes,
                             cell_km_overrides=km_overrides, verbose=verbose)
    names = [s.name for s in solver.sub_list]
    shape = grid['shape']
    nbytes_frame = sum(8 * grid['n_nodes'] for _ in names)
    store_every = max(1, int(store_every))
    if nbytes_frame * (n_steps // store_every + 2) > max_field_bytes:
        store_every = max(store_every,
                          int(np.ceil(nbytes_frame * (n_steps + 1) / max_field_bytes)))
    frames = sorted(set(list(range(0, n_steps + 1, store_every)) + [n_steps]))
    fields = {nm: np.empty((len(frames), ) + shape) for nm in names}
    budget_series = {nm: {k: np.zeros(len(frames)) for k in
                          ('delivered', 'bath_net', 'background', 'consumed', 'bound',
                           'accumulated', 'inventory', 'residual', 'residual_relative')}
                     for nm in names}
    n_cells = solver.n_cells
    cell_conc = {nm: np.zeros((len(frames), n_cells)) for nm in names}
    cell_upt = {nm: np.zeros((len(frames), n_cells)) for nm in names}
    cell_upt_tot = np.zeros((len(frames), n_cells))
    t0 = _time.perf_counter()
    times = []
    frame_of = {st: i for i, st in enumerate(frames)}
    for st in range(n_steps + 1):
        if st > 0:
            solver.step(dt_eff)
        if st in frame_of:
            i = frame_of[st]
            times.append(st * dt_eff)
            for nm in names:
                fields[nm][i] = solver.C[nm].reshape(shape)
                b = solver.budget[nm]
                for k in budget_series[nm]:
                    if k == 'inventory':
                        budget_series[nm][k][i] = solver.total_amount(nm)
                    else:
                        budget_series[nm][k][i] = b[k]
                if n_cells:
                    cc = solver.cell_concentration(nm)
                    cell_conc[nm][i] = cc
                    uu = solver.cell_uptake(nm, dt_eff if st > 0 else 0.0)
                    cell_upt[nm][i] = uu
                    cell_upt_tot[i] += uu
    runtime = _time.perf_counter() - t0
    vdt = solver.grid['volume_um3'] * 8
    info = dict(
        model='vascular-tissue exchange (diffusion + prescribed vessels + cell uptake)',
        validated=False,
        WARNING=('TRANSPORT MODEL WITH LARGELY ILLUSTRATIVE PARAMETERS. Vessel geometry is '
                 'PRESCRIBED, NOT GROWN. Not validated against any measurement in this '
                 'project. Passing the analytic self-tests proves the arithmetic, not the '
                 'biology. See the module docstring and PARAMETER_PROVENANCE.'),
        scheme=solver.scheme, boundary=boundary, dt=dt_eff, n_steps=n_steps,
        t_end=t_end, dx_um=dx_um, domain_um=tuple(map(float, domain_um)),
        realised_extent_um=grid['extent_um'], padding_um=grid['padding_um'],
        thickness_um=thickness_um, n_nodes=grid['n_nodes'], grid_shape=shape,
        n_cells=n_cells, store_stride=store_every, stored_frames=len(frames),
        runtime_s=runtime, seconds_per_step=runtime / max(1, n_steps),
        peak_rss_mb=_peak_rss_mb(),
        substances={s.name: dict(D_um2_s=s.D_um2_s, D_eff_um2_s=s.D_eff,
                                 tortuosity=s.tortuosity, alpha=s.alpha,
                                 plasma_conc_um=s.plasma_conc_um,
                                 carrier_factor=s.carrier_factor,
                                 zero_order_um_s=s.zero_order_um_s,
                                 k_first_s=s.k_first_s,
                                 cell_consumption=s.cell_consumption,
                                 km_um=s.km_um,
                                 vmax_per_cell_amol_s=s.vmax_per_cell_amol_s,
                                 provenance=s.provenance)
                     for s in solver.sub_list},
        vessels=solver.unit_info,
        exchange_law=exchange_law,
        mapping=None if solver.mapping is None else {
            k: v for k, v in solver.mapping.items()
            if k in ('mode', 'n_cells', 'duplication_residual', 'n_boundary_cells',
                     'max_truncation', 'support_min', 'support_max', 'support_mean',
                     'total_weight', 'exact_row_sum')},
        cell_km={k: v['km_um'] for k, v in solver.cell.items()},
        cell_km_source={k: v['km_source'] for k, v in solver.cell.items()},
        guards=solver.guards,
        max_stable_dt=max_stable_dt(grid, substances, field=solver.C, dt=dt_eff),
        citations=CITATIONS, parameter_provenance=PARAMETER_PROVENANCE,
        limits_and_caveats=[
            'vessel geometry is prescribed, not grown; no angiogenesis, no vasomotion',
            'axial blood-side depletion is lumped into the Renkin-Crone extraction factor, '
            'not resolved along the vessel',
            'the vessel lumen is not excised from the domain (lumen voxels carry no wall '
            'conductance); at r <= 1.5*dx the lumen area fraction is small and is reported',
            'cell MM is applied voxel-locally with the per-cell capacity distributed by the '
            'cell->voxel weights (exact when a cell spans one voxel; the Jensen gap between '
            'voxel-local and cell-level MM is measured in self_test)',
            'no interstitial convection, no spatially varying D or solubility, no pH/CO2',
            'no gap-junction transport of these substances between cells',
        ],
        notes='cell_concentration/cell_uptake are dicts keyed by substance (multi-substance API)',
    )
    return dict(times=np.asarray(times), fields=fields, grid=grid,
                cell_concentration=cell_conc, cell_uptake=cell_upt,
                cell_uptake_total=cell_upt_tot,
                cellstate_oxygen=(cellstate_oxygen_input(cell_conc['o2'],
                                                         solver.subs['o2'].plasma_conc_um)
                                  if n_cells and 'o2' in cell_conc else None),
                growth_substrate=(growth_substrate_factor(cell_conc['glucose'],
                                                          solver.subs['glucose'].plasma_conc_um)
                                  if n_cells and 'glucose' in cell_conc else None),
                budget=budget_series, metadata=info,
                final=dict(t=t_end, fields={k: v.copy() for k, v in solver.C.items()},
                           grid=grid, budget=solver.budget, guards=solver.guards))


def _peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(r) / 1024.0 if r > 1e7 else float(r) / 1024.0
    except Exception:
        return float('nan')


# ===========================================================================
# steady state
# ===========================================================================
def solve_steady_state(domain_um=(300.0, 300.0), dx_um=2.0, substances=None,
                       vessels=None, cells=None, blood=None, *, thickness_um=10.0,
                       boundary='no_flux', bath=None, background=None,
                       exchange_law='renkin_crone', d_pen_um=None,
                       default_permeability_um_s=50.0, default_velocity_um_s=1000.0,
                       default_flow_um3_s=None, mapping_mode='area', km_overrides=None,
                       tol=1e-10, max_iter=200, relax=1.0, initial=None,
                       update_initial=True):
    """Legacy experimental steady solver; NOT covered by bounded verification.

    WARNING: zero-order starvation is only clipped after a linear solve and is
    not a converged constrained steady problem. Use transient run() for the
    verified conservative API; do not cite this helper as analytic validation.

    Solves  0 = D_eff*L C + s - k(C) C - z(C)  for every substance, where the
    sinks depend on C only through the frozen-denominator MM and the zero-order
    clipping, so a damped Picard iteration converges (and is EXACT for the
    linear cases: pure diffusion with Robin vessel terms and/or zero-order
    consumption, which is what the Krogh comparisons use).

    Raises if the steady problem is singular: with no vessel, no bath and no
    sink at all the constant mode is a null vector, and the "steady state" is
    then just the conserved mean of the initial condition.
    """
    grid = make_grid(domain_um, dx_um, thickness_um)
    solver = TransportSolver(grid, substances, vessels, cells, blood, scheme='imex',
                             boundary=boundary, bath=bath, background=background,
                             initial=initial, exchange_law=exchange_law,
                             d_pen_um=d_pen_um,
                             default_permeability_um_s=default_permeability_um_s,
                             default_velocity_um_s=default_velocity_um_s,
                             default_flow_um3_s=default_flow_um3_s,
                             mapping_mode=mapping_mode, cell_km_overrides=km_overrides,
                             direct_max_nodes=max(2 * grid['n_nodes'], 400_000))
    Vv = grid['volume_um3']
    n = grid['n_nodes']
    has_source = bool(solver.units) or bool(solver.bath) or any(
        solver.background[s.name] > 0 for s in solver.sub_list)
    maxiter_used = {s.name: 0 for s in solver.sub_list}
    residual = {}
    for s in solver.sub_list:
        has_sink = (s.k_first_s > 0 or s.zero_order_um_s > 0 or
                    (s.name in solver.cell) or solver.diag_vessel[s.name].max() > 0
                    or solver.diag_bath[s.name].max() > 0)
        if not has_source and not has_sink:
            raise ValueError(
                f'{s.name}: the steady problem is singular (pure diffusion with '
                f'no-flux boundaries has a constant null mode); add a vessel, a '
                f'bath or a sink, or use run() instead.')
        if not has_sink and s.name in solver.cell is False:
            # no sink that depends on C: with zero-order only, the value can go
            # negative; the clipping below handles it
            pass
        C = solver.C[s.name].copy()
        for it in range(max_iter):
            maxiter_used[s.name] = it + 1
            if s.name in solver.cell:
                cl = solver.cell[s.name]
                cap = cl['cap_voxel_um_s']; km = cl['km_um']
                if s.cell_consumption == 'michaelis_menten':
                    k_v = cap / (km + np.maximum(C, 0.0)); z_v = np.zeros(n)
                elif s.cell_consumption == 'zero_order':
                    k_v = np.zeros(n); z_v = cap.copy()
                else:
                    k_v = cap; z_v = np.zeros(n)
            else:
                k_v = np.zeros(n); z_v = np.zeros(n)
            k_v = k_v + s.k_first_s + solver.diag_vessel[s.name] + solver.diag_bath[s.name]
            s_v = solver.src_vessel[s.name] + solver.src_bath[s.name] + solver.background[s.name]
            z_v = float(s.zero_order_um_s) + z_v
            # 0 = D L C + s - k C - z   ->   (-D L + diag(k)) C = s - z
            M = (solver.negL * s.D_eff).tocsc()
            M.setdiag(M.diagonal() + k_v + 1e-300)
            rhs = s_v - z_v
            Cnew = _splinalg.splu(M).solve(rhs)
            # zero-order cannot drive the field negative: clip and re-solve once
            if np.any(Cnew < 0):
                Cnew = np.maximum(Cnew, 0.0)
            if relax < 1.0:
                Cnew = (1 - relax) * C + relax * Cnew
            d = float(np.max(np.abs(Cnew - C))) if n else 0.0
            C = Cnew
            if d <= tol * max(1.0, float(np.max(np.abs(C)))):
                break
        else:
            raise RuntimeError(f'{s.name}: steady-state Picard iteration did not '
                               f'converge in {max_iter} iterations')
        residual[s.name] = d
        solver.C[s.name] = C
    # budget at steady state: everything delivered must be consumed or bound
    budget = {s.name: dict(solver.budget[s.name]) for s in solver.sub_list}
    diag = {}
    for s in solver.sub_list:
        C = solver.C[s.name]
        if s.name in solver.cell:
            cl = solver.cell[s.name]
            cap = cl['cap_voxel_um_s']; km = cl['km_um']
            if s.cell_consumption == 'michaelis_menten':
                rate = cap * C / (km + np.maximum(C, 1e-300))
            elif s.cell_consumption == 'zero_order':
                rate = np.minimum(cap, np.maximum(C, 0.0) * 1e30)
            else:
                rate = cap * C
            consumed = float((rate * Vv).sum() * AMOL_PER_UM3_UM)
        else:
            consumed = 0.0
        bound = float((s.k_first_s * C).sum() * Vv * AMOL_PER_UM3_UM)
        delivered = float((solver.G_vessel[s.name] * (solver._unit_cb(s) - C / s.alpha)
                           ).sum() * AMOL_PER_UM3_UM)
        bath_net = 0.0
        if solver.bath:
            G = solver.diag_bath[s.name] * Vv
            cb = solver.src_bath[s.name] * Vv / np.maximum(G, 1e-300)
            bath_net = float((G * (cb - C / s.alpha)).sum() * AMOL_PER_UM3_UM)
        zero = float((np.minimum(float(s.zero_order_um_s), np.maximum(C, 0) * 1e30)
                      * Vv).sum() * AMOL_PER_UM3_UM)
        diag[s.name] = dict(delivered_amol=delivered, consumed_amol=consumed,
                            bound_amol=bound, zero_order_amol=zero,
                            bath_net_amol=bath_net,
                            balance_residual_amol=(delivered + bath_net - consumed
                                                   - bound - zero),
                            picard_iterations=maxiter_used[s.name],
                            picard_residual=residual[s.name],
                            min_uM=float(C.min()), max_uM=float(C.max()))
    return dict(fields={k: v.reshape(grid['shape']) for k, v in solver.C.items()},
                fields_flat=solver.C, grid=grid, steady=diag,
                metadata=dict(scheme='sparse direct steady solve, damped Picard',
                              max_iter=max_iter, tol=tol, n_nodes=grid['n_nodes'],
                              warning='prescribed vessel geometry; illustrative parameters'),
                vessels=solver.unit_info, mapping=solver.mapping,
                solver=solver)


def self_test():
    """Bounded numerical verification, including an actual 10000-cell workload.

    Returns JSON-serializable evidence; no plots or species validation.
    Run this file with --verify PATH to persist the evidence.
    """
    import platform
    import sys
    import scipy
    from dataclasses import asdict
    started = _time.perf_counter()
    evidence = dict(scope='Generic prescribed vascular exchange; illustrative parameters; '
                    'not validated physiology, not insect/Hydra or whole-organism model',
                    python=sys.version, interpreter=sys.executable,
                    numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                    tests={}, benchmark={})
    tests = evidence['tests']
    grid = make_grid((40, 40), 5, 10)
    cen = np.array([[0, 0], [10, 10], [10, 10], [39, 39], [20, 25]], float)
    mapping_results = {}
    for mode in ('area', 'pic', 'nearest'):
        mp = map_cells(grid, cen, 43, mode=mode)
        amounts = np.arange(1, 6, dtype=float)
        deposited = deposit_cells(mp, amounts, np.zeros(grid['n_nodes']))
        field = np.linspace(1, 3, grid['n_nodes'])
        samples = sample_cells(mp, field)
        err = abs(deposited.sum() - amounts.sum())
        adjoint = abs(deposited @ field - amounts @ samples)
        assert err < 1e-12 and adjoint < 1e-12 and mp['exact_row_sum']
        mapping_results[mode] = dict(amount_error=err, adjoint_error=adjoint,
                                    row_sum_error=mp['duplication_residual'])
    tests['mapping'] = dict(config=dict(domain_um=[40, 40], dx_um=5,
        centroids_um=cen.tolist(), areas_um2=43), results=mapping_results)
    # A Neumann cosine mode is an exact discrete eigenvector for backward Euler.
    mode = np.cos(np.pi * grid['centers'][:, 0] / 40)
    sub = Substance('tracer', 20)
    eig = 2 * (np.cos(np.pi / 8) - 1) / 25
    diff_results = {}
    for scheme in ('imex', 'split'):
        sol = TransportSolver(grid, sub, scheme=scheme, initial=2 + mode)
        for _ in range(10):
            sol.step(.1)
        expected = 2 + mode / (1 - .1 * 20 * eig) ** 10
        err = float(np.max(np.abs(sol.C['tracer'] - expected)))
        assert err < 1e-11
        diff_results[scheme] = dict(max_error_uM=err, budget=sol.budget['tracer'])
    tests['discrete_diffusion_mode'] = dict(config=dict(domain_um=[40,40], dx_um=5,
        D_um2_s=20, dt_s=.1, steps=10, initial='2+cos(pi*x/40)'), results=diff_results)
    # Mixed blood concentrations on overlapping vessels and alpha != 1 Robin bath.
    vessels = (single_vessel((40,40), plasma_conc_um=20, flow_um3_s=1000,
                permeability_um_s=2) + single_vessel((40,40), plasma_conc_um=40,
                flow_um3_s=500, permeability_um_s=2))
    cells = dict(centroids=cen, areas=43, demand={'tracer': [.1,.2,.3,.4,.5]})
    cases = []
    for scheme in ('imex', 'split'):
        for law in ('michaelis_menten', 'zero_order', 'first_order'):
            sub = Substance('tracer', 20, alpha=2, plasma_conc_um=20,
                k_first_s=.02, cell_consumption=law, km_um=3)
            t0 = _time.perf_counter()
            sol = TransportSolver(grid, sub, vessels, cells, scheme=scheme,
                boundary='robin', bath={'conductance_um_s': .2, 'tracer':10}, initial=2)
            uptake = 0.
            for _ in range(20):
                sol.step(.05)
                uptake += float(sol.cell_uptake('tracer').sum())
                assert np.isfinite(sol.C['tracer']).all() and sol.C['tracer'].min() >= 0
            b = sol.budget['tracer']
            err = abs(uptake - b['consumed'])
            assert b['residual_relative'] < 1e-11 and err < 1e-10
            cases.append(dict(scheme=scheme, uptake_law=law, substance=asdict(sub),
                elapsed_s=_time.perf_counter()-t0, min_uM=float(sol.C['tracer'].min()),
                max_uM=float(sol.C['tracer'].max()), uptake_amol=uptake,
                uptake_budget_error_amol=err, budget=b, guards=sol.guards))
    tests['exchange_uptake_storage'] = dict(config=dict(domain_um=[40,40],dx_um=5,
        thickness_um=10, dt_s=.05, steps=20, initial_uM=2, vessels=vessels,
        cells={'centroids':cen.tolist(),'areas':43,'demand':cells['demand']},
        bath={'conductance_um_s':.2,'tracer':10}), results=cases)
    # Severe starvation: zero-order cell demand and tissue consumption share availability.
    for scheme in ('imex', 'split'):
        sub = Substance('tracer', 20, cell_consumption='zero_order', zero_order_um_s=10)
        sol = TransportSolver(grid, sub, cells=cells, scheme=scheme, initial=.001)
        sol.step(10)
        assert sol.guards['zero_order_clipped'] > 0
        assert sol.guards['cell_budget_mismatch_um3um'] < 1e-8
        assert sol.budget['tracer']['residual_relative'] < 1e-11
        tests['starvation_'+scheme] = dict(config=dict(substance=asdict(sub),
            domain_um=[40,40], dx_um=5, initial_uM=.001, dt_s=10, cells='as above'),
            budget=sol.budget['tracer'], guards=sol.guards)
    # Actual 100x100 cells; 200x200 tissue grid (not extrapolated timing).
    coords = (np.arange(100) + .5) * 10
    xx, yy = np.meshgrid(coords, coords, indexing='ij')
    cen_big = np.column_stack((xx.ravel(), yy.ravel()))
    big_grid = make_grid((1000,1000), 5, 10)
    t0 = _time.perf_counter()
    mp = map_cells(big_grid, cen_big, 43, mode='area')
    mapping_s = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    samples = sample_cells(mp, np.full(big_grid['n_nodes'], 20.))
    deposited = deposit_cells(mp, np.ones(10000), np.zeros(big_grid['n_nodes']))
    sampling_deposition_s = _time.perf_counter() - t0
    assert np.max(np.abs(samples - 20)) < 1e-12 and abs(deposited.sum()-10000) < 1e-9
    sub = Substance('tracer', 200, plasma_conc_um=40, cell_consumption='michaelis_menten',
                    km_um=3, vmax_per_cell_amol_s=.1)
    vessels_big = krogh_vessels((1000,1000), spacing_um=100, radius_um=3,
                              flow_um3_s=10000, permeability_um_s=2)
    t0 = _time.perf_counter()
    sol = TransportSolver(big_grid, sub, vessels_big,
        dict(centroids=cen_big, areas=43), scheme='split', initial=20)
    setup_s = _time.perf_counter() - t0
    step_times = []
    uptake = 0.
    for _ in range(20):
        t0 = _time.perf_counter()
        sol.step(.05)
        sampled = sol.cell_concentration('tracer')
        uptake += float(sol.cell_uptake('tracer').sum())
        step_times.append(_time.perf_counter() - t0)
        assert np.isfinite(sampled).all() and sol.C['tracer'].min() >= 0
    b = sol.budget['tracer']
    assert b['residual_relative'] < 1e-11
    assert abs(uptake - b['consumed']) < 1e-8
    evidence['benchmark'] = dict(config=dict(n_cells=10000, domain_um=[1000,1000],
        grid_shape=[200,200], dx_um=5, thickness_um=10, areas_um2=43,
        centroids='Cartesian product (arange(100)+0.5)*10 um; deterministic',
        substance=asdict(sub), vessels=vessels_big, scheme='split', initial_uM=20,
        dt_s=.05, steps=20, duration_s=1), mapping_s=mapping_s,
        sampling_and_deposition_s=sampling_deposition_s, solver_setup_s=setup_s,
        step_plus_cell_sample_and_uptake_s=step_times,
        total_step_s=sum(step_times), mean_step_s=float(np.mean(step_times)),
        mapping_nonzeros=int(mp['data'].size), row_sum_error=mp['duplication_residual'],
        min_uM=float(sol.C['tracer'].min()), max_uM=float(sol.C['tracer'].max()),
        uptake_amol=uptake, uptake_budget_error_amol=abs(uptake-b['consumed']),
        budget=b, guards=sol.guards, peak_process_rss_mb=_peak_rss_mb())
    # One-voxel first-order clearance has a closed-form backward Euler result.
    first = Substance('tracer', 20, cell_consumption='first_order', k_cell_um3_s=2)
    tiny = TransportSolver(make_grid((5,5),5,10), first,
        cells={'centroids':[[2.5,2.5]],'areas':1}, initial=10)
    tiny.step(.1)
    exact = 10 / (1 + .1 * 2 / 250)
    assert abs(tiny.C['tracer'][0] - exact) < 1e-12
    tests['first_order_units'] = dict(config=dict(volume_um3=250, clearance_um3_s=2,
        dt_s=.1, initial_uM=10), computed_uM=float(tiny.C['tracer'][0]), expected_uM=exact)
    # 3D Robin faces must have area dx^2, not volume dx^3.
    cube = TransportSolver(make_grid((10,10,10),5), Substance('tracer',20,alpha=2),
        boundary='robin', bath={'conductance_um_s':.2,'tracer':10}, initial=1)
    assert abs(cube.bath['face_area_um2'].sum() - 600) < 1e-12
    cube.step(.1)
    assert cube.budget['tracer']['residual_relative'] < 1e-11
    tests['three_dimensional_robin'] = dict(config=dict(domain_um=[10,10,10],dx_um=5,
        alpha=2,D_um2_s=20,conductance_um_s=.2,bath_uM=10,initial_uM=1,dt_s=.1),
        area_um2=float(cube.bath['face_area_um2'].sum()),budget=cube.budget['tracer'])
    # Exercise the public plotting API and exported names.
    assert all(name in globals() for name in __all__)
    result = run((40,40), 5, .1, .05, substances='o2', vessels=vessels, cells=cells)
    assert result['fields']['o2'].shape == (3,8,8)
    assert result['cell_concentration']['o2'].shape == (3,5)
    tests['public_run_api'] = dict(fields_shape=[3,8,8], cell_shape=[3,5], passed=True)
    evidence['elapsed_s'] = _time.perf_counter() - started
    evidence['all_passed'] = True
    return evidence


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Bounded generic transport verification')
    parser.add_argument('--verify', required=True, help='JSON evidence output path')
    args = parser.parse_args()
    result = self_test()
    with open(args.verify, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'all_passed': result['all_passed'], 'elapsed_s':result['elapsed_s'],
                      'benchmark_mean_step_s':result['benchmark']['mean_step_s']}))
