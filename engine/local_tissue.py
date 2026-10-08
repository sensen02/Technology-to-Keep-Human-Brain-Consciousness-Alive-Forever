"""README — illustrative conservative local tissue / neuromodulation prototype.

Scope
-----
Small rectangular 3-D finite-volume grid; no anatomical reconstruction, validated
neck-cut physiology, measured receptor pathways, or organism survival claim.
Every default is an arbitrary numerical demonstration, NOT a measured parameter.
This module does not change eye/brain connectivity or add a body/spinal network.
NumPy and SciPy are required. Import directly from engine.local_tissue.

Units and state
---------------
Time: s; length: um; voxel volume: um^3; free ligand concentration: nM.
Receptor capacity and bound ligand: nM-equivalent binding sites per voxel volume
(one ligand per site). Amount is nM*um^3 throughout. 1 nM*um^3 = 1e-24 mol
because 1 um^3 = 1e-15 litre. Amounts need not be integer molecules: this is a
continuum approximation, not a stochastic low-copy-number model.
D: um^2/s; internal edge conductance D*face_area/centre_distance: um^3/s.
A barrier can REPLACE any nearest-neighbour edge conductance (including zero);
permeability P [um/s] across that face gives conductance P*face_area [um^3/s].
No-flux external faces are default. No hidden Dirichlet concentrations.
kon: 1/(nM*s); koff, free clearance: 1/s; source: nM*um^3/s per voxel.
Response: dimensionless [0,1], relaxing to local occupancy with time constant s;
it is a phenomenological slow signal, NOT a named/measured molecular pathway.
Heterogeneous receptor capacity is supported; capacity zero means bound ligand,
occupancy and receptor-driven response are exactly zero.

Reservoirs and ledger
---------------------
The grid always holds a finite ligand amount. FinitePool adds another finite,
well-mixed compartment with specified volume and initial concentration, coupled
to grid voxels by conductances. Pool depletion/refilling is solved simultaneously
with grid diffusion; pool transfers are internal and conserve total amount.
PrescribedBath is a DISTINCT infinite external reservoir with a prescribed
concentration and finite exchange conductance. It is not a finite pool or a
clamped tissue concentration. Its signed exchange is recorded as gross inflow
and outflow. It can supply ligand indefinitely, explicitly, never silently.
Sources and clearance act on grid FREE ligand only, not bound ligand or pools.
Cumulative ledger, all in nM*um^3:
  amount_now = initial + source + boundary_in - boundary_out - clearance.
amount_now includes free grid + bound grid + finite pools. mass_balance_error()
reports residual (floating-point roundoff, not a hidden correction). Do not
mutate public state arrays or reservoir definitions while running; changing
state externally invalidates this ledger. Construct a new model for resets.

Numerics
--------
Conservative sparse backward Euler diffusion/exchange/source/clearance uses
an M-matrix (volume-weighted), with unconditional mathematical positivity for
finite nonnegative inputs. No post-hoc clipping or unreported mass repair.
Reversible binding dB/dt=kon*C*(R-B)-koff*B is then solved by 80 bisections of
the bounded backward-Euler root on [0,min(C+B,R)], preserving local C+B.
Slow response uses an exact exponential relaxation to the post-binding target.
This Lie split is first order overall: stable does not mean temporally accurate;
perform timestep and mesh convergence for each application. Binding and response
within a step are not an exact solution of their fully coupled time course.
Extreme scales causing nonfinite/negative linear solutions raise, not hide errors.

Example (all illustrative)
--------------------------
>>> tissue = LocalTissue((3, 2, 2), initial_nM=1., receptor_capacity_nM=2.)
>>> tissue.step(0.1)
>>> abs(tissue.mass_balance_error()) < 1e-10
True

Optional support
----------------
NonPhysiologicalSupportProxy is separate and opt-in. Its dimensionless resource
is neither oxygen concentration nor ATP/energy in physical units. It only tracks
an explicitly artificial bounded availability scalar with supply and demand
rates; no metabolic stoichiometry, physiological calibration, or survival rule
is implied. Physical oxygen/energy modelling requires independently supplied
parameters, units, reactions, and validation, not renamed proxy variables.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

AMOUNT_NM_UM3_TO_MOL = 1e-24


def _nonnegative(value, name):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return a


def _field(value, shape, name):
    return np.broadcast_to(_nonnegative(value, name), shape).copy()


@dataclass(frozen=True)
class FinitePool:
    """Finite well-mixed pool: volume um^3, initial nM, grid conductance um^3/s."""
    volume_um3: float
    initial_nM: float
    conductance_um3_s: object


@dataclass(frozen=True)
class PrescribedBath:
    """Infinite bath: fixed nM, exchange conductance to each voxel in um^3/s."""
    concentration_nM: float
    conductance_um3_s: object


class LocalTissue:
    """Finite ligand grid with reversible receptors and a mass-flow ledger.

    barrier_edges maps pairs of flattened (C-order) neighbouring voxel indices
    to replacement conductance in um^3/s. Unlisted edges retain D*A/dx.
    Grid-shaped fields or scalars are accepted for all grid parameters.
    """

    def __init__(self, shape=(3, 3, 3), spacing_um=(1., 1., 1.),
                 diffusion_um2_s=1., initial_nM=0., receptor_capacity_nM=0.,
                 initial_bound_nM=0., kon_nM_inv_s=0.1, koff_s=0.1,
                 response_tau_s=10., clearance_s=0., source_amount_s=0.,
                 barrier_edges: Mapping = None, finite_pools=(), baths=()):
        if len(shape) != 3 or any(int(x) != x or x < 1 for x in shape):
            raise ValueError("shape must contain three positive integers")
        self.shape = tuple(int(x) for x in shape)
        self.n = int(np.prod(self.shape))
        spacing = _field(spacing_um, (3,), "spacing_um")
        if np.any(spacing == 0):
            raise ValueError("spacing must be positive")
        self.voxel_volume_um3 = float(np.prod(spacing))
        d = float(_nonnegative(diffusion_um2_s, "diffusion"))
        self.free_nM = _field(initial_nM, self.shape, "initial_nM")
        self.capacity_nM = _field(receptor_capacity_nM, self.shape, "capacity")
        self.bound_nM = _field(initial_bound_nM, self.shape, "bound")
        if np.any(self.bound_nM > self.capacity_nM):
            raise ValueError("initial bound exceeds receptor capacity")
        self.kon = _field(kon_nM_inv_s, self.shape, "kon")
        self.koff = _field(koff_s, self.shape, "koff")
        self.tau = _field(response_tau_s, self.shape, "response_tau_s")
        if np.any(self.tau == 0):
            raise ValueError("response_tau_s must be positive")
        self.clearance = _field(clearance_s, self.shape, "clearance")
        self.source = _field(source_amount_s, self.shape, "source")
        self.response = np.zeros(self.shape)
        edges = {}
        for index in np.ndindex(self.shape):
            i = np.ravel_multi_index(index, self.shape)
            for axis in range(3):
                neighbour = list(index)
                neighbour[axis] += 1
                if neighbour[axis] < self.shape[axis]:
                    j = np.ravel_multi_index(tuple(neighbour), self.shape)
                    edges[(i, j)] = d * self.voxel_volume_um3 / spacing[axis]**2
        for pair, conductance in (barrier_edges or {}).items():
            if len(pair) != 2 or any(int(i) != i for i in pair):
                raise ValueError("barrier endpoints must be two integer indices")
            key = tuple(sorted(pair))
            if key not in edges:
                raise ValueError("barrier edge must join grid neighbours")
            edges[key] = float(_nonnegative(conductance, "barrier conductance"))
        self.pool_nM = []
        volumes = [self.voxel_volume_um3] * self.n
        self._pool_links = []
        for k, pool in enumerate(finite_pools):
            volume = float(_nonnegative(pool.volume_um3, "pool volume"))
            if volume == 0:
                raise ValueError("pool volume must be positive")
            volumes.append(volume)
            self.pool_nM.append(float(_nonnegative(pool.initial_nM, "pool initial")))
            g = _field(pool.conductance_um3_s, self.shape, "pool conductance").ravel()
            self._pool_links.append(g)
            for i in np.flatnonzero(g):
                edges[(int(i), self.n + k)] = g[i]
        self.pool_nM = np.array(self.pool_nM, dtype=float)
        self.volumes = np.array(volumes)
        size = len(volumes)
        row, col, data = [], [], []
        for (i, j), g in edges.items():
            row.extend((i, j, i, j))
            col.extend((i, j, j, i))
            data.extend((g, g, -g, -g))
        self._laplacian = coo_matrix((data, (row, col)), shape=(size, size)).tocsc()
        self._baths = []
        self._bath_g = np.zeros(size)
        self._bath_gc = np.zeros(size)
        for bath in baths:
            c = float(_nonnegative(bath.concentration_nM, "bath concentration"))
            g = _field(bath.conductance_um3_s, self.shape, "bath conductance").ravel()
            self._baths.append((c, g))
            self._bath_g[:self.n] += g
            self._bath_gc[:self.n] += g*c
        self.ledger = dict(source=0., boundary_in=0., boundary_out=0., clearance=0.)
        self.time_s = 0.
        self.last_pool_to_grid_amount = np.zeros(len(self.pool_nM))
        self.initial_amount = self.total_amount()

    @property
    def occupancy(self):
        return np.divide(self.bound_nM, self.capacity_nM,
                         out=np.zeros(self.shape), where=self.capacity_nM > 0)

    def total_amount(self):
        """Grid free+bound and finite pools, in nM*um^3 (excludes infinite baths)."""
        return float(self.voxel_volume_um3 * np.sum(self.free_nM + self.bound_nM)
                     + self.volumes[self.n:] @ self.pool_nM)

    def mass_balance_error(self):
        l = self.ledger
        return self.total_amount() - (self.initial_amount + l['source']
                + l['boundary_in'] - l['boundary_out'] - l['clearance'])

    def step(self, dt_s):
        """Advance one positive-duration split step; return mass residual in nM*um^3."""
        dt = float(_nonnegative(dt_s, "dt_s"))
        if dt == 0:
            return self.mass_balance_error()
        old = np.concatenate((self.free_nM.ravel(), self.pool_nM))
        source = np.zeros_like(old)
        source[:self.n] = self.source.ravel()
        sink = np.zeros_like(old)
        sink[:self.n] = self.voxel_volume_um3 * self.clearance.ravel()
        matrix = diags(self.volumes + dt*(sink + self._bath_g)) + dt*self._laplacian
        rhs = self.volumes*old + dt*(source + self._bath_gc)
        transport = spsolve(matrix.tocsc(), rhs)
        if not np.all(np.isfinite(transport)) or np.any(transport < 0):
            raise FloatingPointError("nonfinite/negative transport; rescale inputs")
        free = transport[:self.n].reshape(self.shape)
        total = free + self.bound_nM
        lo = np.zeros(self.shape)
        hi = np.minimum(total, self.capacity_nM)
        # Residual is strictly increasing on this physical interval.
        for _ in range(80):
            mid = lo + (hi-lo)*0.5
            residual = (mid-self.bound_nM + dt*self.koff*mid
                        - dt*self.kon*(total-mid)*(self.capacity_nM-mid))
            if not np.all(np.isfinite(residual)):
                raise FloatingPointError("binding overflow; rescale inputs")
            lo = np.where(residual < 0, mid, lo)
            hi = np.where(residual >= 0, mid, hi)
        new_bound = lo + (hi-lo)*0.5
        self.ledger['source'] += dt*float(source.sum())
        self.ledger['clearance'] += dt*float(sink @ transport)
        for c, g in self._baths:
            flow = dt*g*(c-transport[:self.n])
            self.ledger['boundary_in'] += float(np.maximum(flow, 0).sum())
            self.ledger['boundary_out'] += float(np.maximum(-flow, 0).sum())
        self.last_pool_to_grid_amount = np.array([
            dt*float(g @ (transport[self.n+k]-transport[:self.n]))
            for k, g in enumerate(self._pool_links)])
        self.free_nM = total-new_bound
        self.bound_nM = new_bound
        self.pool_nM = transport[self.n:].copy()
        self.response += -np.expm1(-dt/self.tau)*(self.occupancy-self.response)
        self.response[self.capacity_nM == 0] = 0.
        self.time_s += dt
        return self.mass_balance_error()


class NonPhysiologicalSupportProxy:
    """Opt-in dimensionless availability, NOT oxygen/ATP or physiological energy.

    dx/dt = supply_s*(1-x)-demand_s*x, with exact positivity-preserving update.
    supply_s and demand_s are artificial rates [1/s]. No ligand coupling, no
    survival threshold, and no claim of chemical mass/energy conservation.
    """
    def __init__(self, shape=(3, 3, 3), initial=1.):
        self.availability = _field(initial, shape, "initial availability")
        if np.any(self.availability > 1):
            raise ValueError("availability must be in [0,1]")

    def step(self, dt_s, supply_s=0., demand_s=0.):
        dt = float(_nonnegative(dt_s, "dt_s"))
        supply = _field(supply_s, self.availability.shape, "supply")
        demand = _field(demand_s, self.availability.shape, "demand")
        rate = supply+demand
        target = np.divide(supply, rate, out=np.zeros_like(rate), where=rate > 0)
        self.availability += -np.expm1(-rate*dt)*(target-self.availability)
        return self.availability.copy()
