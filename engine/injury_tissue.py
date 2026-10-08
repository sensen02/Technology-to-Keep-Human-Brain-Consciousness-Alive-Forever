"""Illustrative injury reduction, NOT measured fly injury/viability physiology.

K chemistry: finite intracellular/extracellular volumes [um^3], concentrations
[mM], amounts [mM*um^3 = 1e-18 mol], time [s]. Membrane permeability P [um/s]
times area [um^2] gives exchange G [um^3/s]. Flux is G*(Ki-Ke), not a channel
current: this is externally clamped, electroneutral reduced salt chemistry.
Unmodelled counterions and the external electrical clamp maintain neutrality;
no charge/energy ledger, ATP, pumps, electrodiffusion or survival model exists.
A reversible, finite-capacity glial sequestration pool is a phenomenological
binding sink, NOT a measured glial transporter. Without pumps/homeostasis it
can almost deplete a closed extracellular pool: extreme negative diagnostic Ek
is a model-insufficiency artifact, NOT physiological glial protection or resting
voltage. All defaults are illustrative.

Backward Euler exchange is conservative and positive; bounded implicit binding
conserves extracellular+buffer amount. Lie splitting is first order. The only
external K mass exchange is a prescribed reservoir with separate in/out ledger.
Nernst voltage is diagnostic: R [J/(mol K)] * T [K] / F [C/mol] is volts, z=+1,
and the same-unit mM ratio is dimensionless. Zero concentrations are legal mass
states but Nernst explicitly rejects them (no hidden concentration floor).

Optional ligand wrapper reconstructs LocalTissue through its public constructor
when barrier edges change, transferring free/bound/finite-pool state and the
massless response, while retaining completed-segment ledgers separately. No
private matrix mutation or modifications to local_tissue.py are needed. Ligand
[nM*um^3] and potassium [mM*um^3] ledgers MUST NOT be summed without conversion.
"""
from dataclasses import dataclass, replace
import copy
import math
import numpy as np
from .local_tissue import LocalTissue

R_J_MOL_K = 8.31446261815324
F_C_MOL = 96485.33212
AMOUNT_MM_UM3_TO_MOL = 1e-18


def _number(value, name, positive=False):
    x = float(value)
    if not math.isfinite(x) or x < 0 or (positive and x == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return x


def nernst_k_mV(inside_mM, outside_mM, temperature_K=298.15):
    """Ideal activity approximation, monovalent K; no mM-to-M factor in ratio."""
    ki = _number(inside_mM, 'inside_mM', True)
    ke = _number(outside_mM, 'outside_mM', True)
    t = _number(temperature_K, 'temperature_K', True)
    return 1000 * R_J_MOL_K * t / F_C_MOL * (math.log(ke) - math.log(ki))


@dataclass(frozen=True)
class PotassiumConfig:
    intracellular_volume_um3: float = 1000.
    extracellular_volume_um3: float = 200.
    initial_inside_mM: float = 120.
    initial_outside_mM: float = 3.
    membrane_area_um2: float = 100.
    permeability_um_s: float = 0.
    buffer_capacity_amount: float = 4000.  # mM*um^3, not concentration
    initial_buffer_amount: float = 0.
    buffer_on_mM_inv_s: float = .02
    buffer_off_s: float = .01
    reservoir_mM: float = 3.
    reservoir_conductance_um3_s: float = 0.
    temperature_K: float = 298.15


class InjuryPotassium:
    """Two finite free pools + finite bound pool, with explicit reservoir ledger.

    Do not mutate state or config externally. apply_injury changes rates only;
    chemical membrane permeability and cable conductance are independent inputs.
    """
    def __init__(self, config=None):
        self.config = config or PotassiumConfig()
        c = self.config
        for name, value in vars(c).items():
            _number(value, name, name in ('intracellular_volume_um3',
                      'extracellular_volume_um3', 'temperature_K'))
        if c.initial_buffer_amount > c.buffer_capacity_amount:
            raise ValueError('initial buffer exceeds capacity')
        self.inside_mM = float(c.initial_inside_mM)
        self.outside_mM = float(c.initial_outside_mM)
        self.buffer_amount = float(c.initial_buffer_amount)
        self.permeability_um_s = float(c.permeability_um_s)
        self.reservoir_conductance_um3_s = float(c.reservoir_conductance_um3_s)
        self.time_s = 0.
        self.ledger = dict(reservoir_in=0., reservoir_out=0.)
        self.initial_amount = self.total_amount()
        self.events = []
        self.last_membrane_out_amount = 0.

    def total_amount(self):
        c = self.config
        return (c.intracellular_volume_um3*self.inside_mM +
                c.extracellular_volume_um3*self.outside_mM + self.buffer_amount)

    def mass_balance_error(self):
        return self.total_amount() - (self.initial_amount + self.ledger['reservoir_in']
                                     - self.ledger['reservoir_out'])

    @property
    def ek_mV(self):
        return nernst_k_mV(self.inside_mM, self.outside_mM, self.config.temperature_K)

    def apply_injury(self, permeability_um_s, reservoir_conductance_um3_s=None):
        p = _number(permeability_um_s, 'permeability_um_s')
        g = (self.reservoir_conductance_um3_s if reservoir_conductance_um3_s is None
             else _number(reservoir_conductance_um3_s, 'reservoir_conductance_um3_s'))
        before = self.total_amount()
        self.permeability_um_s, self.reservoir_conductance_um3_s = p, g
        self.events.append(dict(time_s=self.time_s, permeability_um_s=p,
                                reservoir_conductance_um3_s=g,
                                mass_jump=self.total_amount()-before))

    def step(self, dt_s):
        dt = _number(dt_s, 'dt_s')
        if dt == 0:
            return self.mass_balance_error()
        c = self.config
        vi, ve = c.intracellular_volume_um3, c.extracellular_volume_um3
        g = c.membrane_area_um2*self.permeability_um_s
        h = self.reservoir_conductance_um3_s
        matrix = np.array([[vi+dt*g, -dt*g], [-dt*g, ve+dt*(g+h)]])
        rhs = np.array([vi*self.inside_mM, ve*self.outside_mM+dt*h*c.reservoir_mM])
        ki, ke = np.linalg.solve(matrix, rhs)
        if not np.isfinite([ki, ke]).all() or min(ki, ke) < 0:
            raise FloatingPointError('invalid exchange solve; rescale inputs')
        total = ve*ke + self.buffer_amount
        lo, hi = 0., min(total, c.buffer_capacity_amount)
        # f(B) = B-Bold-dt*(kon*Ke*(capacity-B)-koff*B).
        for _ in range(80):
            b = lo + (hi-lo)*.5
            f = (b-self.buffer_amount-dt*(c.buffer_on_mM_inv_s*(total-b)/ve*
                 (c.buffer_capacity_amount-b)-c.buffer_off_s*b))
            if not math.isfinite(f):
                raise FloatingPointError('buffer solve overflow; rescale inputs')
            if f < 0:
                lo = b
            else:
                hi = b
        b = lo + (hi-lo)*.5
        transfer = dt*h*(c.reservoir_mM-ke)
        self.ledger['reservoir_in'] += max(transfer, 0.)
        self.ledger['reservoir_out'] += max(-transfer, 0.)
        self.last_membrane_out_amount = dt*g*(ki-ke)
        self.inside_mM, self.outside_mM = float(ki), float((total-b)/ve)
        self.buffer_amount = b
        self.time_s += dt
        return self.mass_balance_error()


class InjuryLigand:
    """Piecewise LocalTissue, using fresh public construction at an injury event.

    Owns the constructor configuration; supports finite pools and baths. The
    current LocalTissue ledger is segment-local; this wrapper's ledger and clock
    span ALL segments. Only wrapper.step should be used for consistent timing.
    """
    def __init__(self, **local_tissue_config):
        self._config = copy.deepcopy(local_tissue_config)
        self.model = LocalTissue(**self._config)
        self.initial_amount = self.model.total_amount()
        self._completed_ledger = dict.fromkeys(self.model.ledger, 0.)
        self.time_s = 0.
        self.events = []

    @property
    def ledger(self):
        return {k: v+self.model.ledger[k] for k, v in self._completed_ledger.items()}

    def total_amount(self):
        return self.model.total_amount()

    def mass_balance_error(self):
        l = self.ledger
        return self.total_amount() - (self.initial_amount+l['source']+
                                     l['boundary_in']-l['boundary_out']-l['clearance'])

    def switch_barrier(self, barrier_edges):
        """Replace full barrier map; unlisted edges revert to D*A/dx, no mass jump."""
        old = self.model
        config = copy.deepcopy(self._config)
        config.update(initial_nM=old.free_nM.copy(), initial_bound_nM=old.bound_nM.copy(),
                      barrier_edges=copy.deepcopy(barrier_edges))
        config['finite_pools'] = tuple(replace(pool, initial_nM=float(value)) for pool, value
            in zip(config.get('finite_pools', ()), old.pool_nM))
        new = LocalTissue(**config)  # invalid event fails BEFORE touching old state
        # LocalTissue has no initial-response constructor argument. Public response
        # is massless, so copy it solely into the freshly constructed replacement.
        new.response[:] = old.response
        jump = new.total_amount()-old.total_amount()
        if abs(jump) > 1e-12*max(1., old.total_amount()):
            raise FloatingPointError('barrier reconstruction changed ligand amount')
        ledger = self.ledger
        self._completed_ledger = ledger
        self._config, self.model = config, new
        self.events.append(dict(time_s=self.time_s, mass_jump=jump,
                                barrier_edges=repr(barrier_edges)))

    def step(self, dt_s):
        dt = _number(dt_s, 'dt_s')
        self.model.step(dt)
        self.time_s += dt
        return self.mass_balance_error()
