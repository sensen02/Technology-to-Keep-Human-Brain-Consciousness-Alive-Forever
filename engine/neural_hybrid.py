"""Fixed-owner bounded hybrid coordinator, NOT a whole-brain implementation.

A single external directed graph owns ALL recurrence (including within tiers).
Each spike is quantized to a shared step END and each edge emits exactly one
START event at emission+delay. No dynamic tier transfer, plasticity, distributed
execution, arbitrary receptor kinetics, or inferred transmitter sign.
"""
from dataclasses import dataclass
from types import MappingProxyType
import math
import numpy as np
from .neural_cond import ConductanceNetwork, ConductanceParams
from .neural_active import ActiveRuntime


@dataclass(frozen=True)
class ConductanceEdge:
    pre: object
    post: object
    receptor: str
    increment_uS: float
    delay_ms: float


@dataclass(frozen=True)
class Delivery:
    event_id: int
    edge_index: int
    pre: object
    post: object
    receptor: str
    increment_uS: float
    emitted_ms: float
    arrival_ms: float


class HybridNetwork:
    """Construct with explicit global IDs, point IDs and detailed ID->spec dict.

    IDs must partition the global universe exactly once. At most 8 detailed
    cables and 10000 global IDs; illustrative single-process scope. Two named
    receptor classes ('exc', 'inh') have caller-supplied ConductanceParams E/tau;
    these names do NOT assert actual current sign (g*(E-V) determines sign).
    Detailed incoming events target proximal segment; outgoing spikes use distal
    segment. This imposed geometry/site mapping is NOT a reconstructed synapse.

    Use step(point_current_nA=scalar/vector in point_ids order). For detailed
    drive/field/probes use network.active.cables[gid]. Events due at current
    time are delivered before both tiers advance. Returned set is emitted gids.
    Read deliveries/emissions for timestamp audit; bounded max_event_records
    stops execution BEFORE overflow (logs never silently drop events).
    """
    def __init__(self, global_ids, point_ids, detailed_specs, edges=(),
                 params=None, dt_ms=0.025, max_event_records=100000):
        ids, point = tuple(global_ids), tuple(point_ids)
        detailed = dict(detailed_specs)
        if (not ids or len(ids) > 10000 or len(set(ids)) != len(ids) or
                len(set(point)) != len(point) or set(point) & set(detailed) or
                set(point) | set(detailed) != set(ids)):
            raise ValueError('IDs must be a unique exhaustive disjoint partition')
        self.p = params or ConductanceParams()
        self.p.validate()
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError('positive finite dt required')
        self.dt = float(dt_ms)
        if not isinstance(max_event_records, int) or max_event_records < 1:
            raise ValueError('positive integer log budget required')
        self.max_event_records = max_event_records
        self.global_ids, self.point_ids = ids, point
        point_set = set(point)
        self.ownership = MappingProxyType({gid: 'point' if gid in point_set else 'detailed' for gid in ids})
        self.point_index = {gid: i for i, gid in enumerate(point)}
        self.edges = tuple(edges)
        self._outgoing = {gid: [] for gid in ids}
        self._delay_steps = []
        for i, edge in enumerate(self.edges):
            if edge.pre not in self.ownership or edge.post not in self.ownership:
                raise ValueError('edge refers to unowned global ID')
            if edge.receptor not in ('exc', 'inh'):
                raise ValueError('only explicit exc/inh receptor classes supported')
            if not math.isfinite(edge.increment_uS) or edge.increment_uS < 0:
                raise ValueError('finite nonnegative increments required')
            d = edge.delay_ms / self.dt
            if not math.isfinite(d) or d < 1 or not math.isclose(d, round(d), rel_tol=0, abs_tol=1e-9):
                raise ValueError('edge delay must be positive integer steps')
            self._delay_steps.append(int(round(d)))
            self._outgoing[edge.pre].append(i)
        self.point = ConductanceNetwork(len(point), self.p, self.dt) if point else None
        self.active = None
        try:
            if detailed:
                self.active = ActiveRuntime(self.dt, {'exc': (self.p.exc_reversal_mV, self.p.exc_tau_ms),
                                                     'inh': (self.p.inh_reversal_mV, self.p.inh_tau_ms)})
                for gid, spec in detailed.items():
                    self.active.add_cable(gid, spec)
                self.active.reset()
        except Exception:
            if self.active is not None:
                self.active.close()
            raise
        self.steps, self._serial = 0, 0
        self._pending = {}
        self.deliveries, self.emissions = [], []
        self.closed = False

    @property
    def t_ms(self):
        return self.steps * self.dt

    def step(self, point_current_nA=0.0):
        if self.closed:
            raise RuntimeError('hybrid network is closed')
        if self.point is not None:
            if self.point.We.nnz or self.point.Wi.nnz:
                raise RuntimeError('local recurrence forbidden: hybrid graph owns every edge')
            if abs(self.point.t_ms - self.t_ms) > 1e-9:
                raise RuntimeError('point tier advanced outside coordinator')
            # Validate before consuming events; caller errors must not lose them.
            self.point._vector(point_current_nA, 'point current')
        if self.active is not None and abs(self.active.t_ms - self.t_ms) > 1e-9:
            raise RuntimeError('detailed tier advanced outside coordinator')
        if self._serial + len(self.edges) > self.max_event_records or len(self.emissions) + len(self.global_ids) > self.max_event_records:
            raise RuntimeError('bounded audit log budget reached; use a shorter run')
        ge, gi = np.zeros(len(self.point_ids)), np.zeros(len(self.point_ids))
        for event in self._pending.pop(self.steps, ()):
            if self.ownership[event.post] == 'point':
                target = ge if event.receptor == 'exc' else gi
                target[self.point_index[event.post]] += event.increment_uS
            else:
                self.active.cables[event.post].add_conductance_uS(event.receptor, event.increment_uS)
            self.deliveries.append(event)
        emitted = set()
        if self.point is not None:
            spikes = self.point.step(point_current_nA, ge, gi)
            emitted.update(gid for gid, spike in zip(self.point_ids, spikes) if spike)
        if self.active is not None:
            emitted.update(self.active.step())
        self.steps += 1
        # Global ID order makes event identity deterministic across heterogeneous IDs.
        for gid in self.global_ids:
            if gid not in emitted:
                continue
            self.emissions.append((self.t_ms, gid))
            for i in self._outgoing[gid]:
                edge = self.edges[i]
                arrival_step = self.steps + self._delay_steps[i]
                event = Delivery(self._serial, i, gid, edge.post, edge.receptor,
                                 edge.increment_uS, self.t_ms, arrival_step * self.dt)
                self._serial += 1
                self._pending.setdefault(arrival_step, []).append(event)
        return emitted

    def close(self):
        if not self.closed and self.active is not None:
            self.active.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
