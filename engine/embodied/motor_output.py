"""Auditable spike -> muscle translation; declared hypothesis, not biological calibration.

No pose, desired motion, CPG, bias, or polarity inference enters this interface.
A causal exponential population-rate filter drives nonnegative controls. MuJoCo
owns activation dynamics. Shared anatomical pools remain shared, not duplicated
neurons within a pool. Missing tier cells are reported, never substituted.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


def build_motor_manifest(pool_result, tier_root_ids, actuator_names):
    roots = [int(r) for r in tier_root_ids]
    names = list(actuator_names)
    if len(set(roots)) != len(roots) or len(set(names)) != len(names):
        raise ValueError('tier roots and actuator names must be unique')
    local = {r: i for i, r in enumerate(roots)}
    measured = pool_result['measured']
    records = measured['resolved']
    unresolved = measured['unresolved']
    source = [r['actuator'] for r in records + unresolved]
    if len(set(source)) != len(source) or set(source) != set(names):
        raise ValueError('source actuator identities must exactly match compiled body')
    by_name = {r['actuator']: r for r in records}
    missing = {r['actuator']: r for r in unresolved}
    root_identity = {}
    rows = []
    for ai, name in enumerate(names):
        r = by_name.get(name)
        if r is None:
            source = missing[name]
            # Preserve identities explicitly present in the source, even when the
            # motor pool is unresolved. Never infer an anatomical connection from
            # the actuator name, and never create control indices for this row.
            identity = {key: source[key] for key in ('leg', 'side', 'muscle_target')
                        if key in source}
            rows.append(dict(actuator=name, actuator_index=ai, local_indices=[],
                             pool_root_ids=[], tier_root_ids=[], status='unresolved',
                             source_annotation=source, **identity))
            continue
        pool = sorted(set(int(v) for v in r['pool_root_ids']))
        identity = (r['leg'], r['side'], r['muscle_target'])
        for root in pool:
            if root in root_identity and root_identity[root] != identity:
                raise ValueError(f'conflicting muscle/leg/side identities for root {root}')
            root_identity[root] = identity
        present = [root for root in pool if root in local]
        rows.append(dict(actuator=name, actuator_index=ai, leg=r['leg'], side=r['side'],
                         muscle_target=r['muscle_target'], pool_root_ids=pool,
                         tier_root_ids=present, local_indices=[local[root] for root in present],
                         missing_tier_roots=[root for root in pool if root not in local],
                         duplicates_removed=len(r['pool_root_ids'])-len(pool),
                         status=('absent_from_tier' if not present else
                                 'partial' if len(present) < len(pool) else 'complete'),
                         source_annotation=r))
    shared = {}
    for row in rows:
        if row['pool_root_ids']:
            shared.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator'])
    return dict(n_neurons=len(roots), n_actuators=len(names), rows=rows,
                usable_actuators=sum(bool(r['local_indices']) for r in rows),
                fully_covered_actuators=sum(r['status'] == 'complete' for r in rows),
                shared_pools=[dict(pool_root_ids=list(k), actuators=v)
                              for k, v in shared.items() if len(v) > 1],
                provenance='Existing declared aliases + peripheral_target_type/body_part_effector/side; NOT confirmed one-to-one anatomy',
                normalization='Mean over PRESENT tier cells; partial pools explicitly flagged',
                shared_pool_caveat='Identical pool command to multiple muscle bundles is an unvalidated recruitment hypothesis')


@dataclass(frozen=True)
class MotorOutputConfig:
    filter_tau_s: float = 0.005
    rate_for_full_control_hz: float = 100.0
    max_control: float = 1.0

    def validate(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError('positive finite translation parameters required')
        if self.max_control > 1:
            raise ValueError('max control exceeds normalized muscle range')
        return self


class MotorOutput:
    def __init__(self, manifest, config=None):
        self.manifest = manifest
        self.config = (config or MotorOutputConfig()).validate()
        for name in ('n_neurons', 'n_actuators'):
            value = manifest[name]
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer, not bool')
        self.n = int(manifest['n_neurons'])
        self.nu = int(manifest['n_actuators'])
        if len(manifest['rows']) != self.nu:
            raise ValueError('manifest actuator count mismatch')
        self.groups = []
        for ai, row in enumerate(manifest['rows']):
            order = row['actuator_index']
            if (isinstance(order, (bool, np.bool_))
                    or not isinstance(order, (int, np.integer)) or order != ai):
                raise ValueError('invalid actuator ordering')
            # Preserve mixed-list bool/float identities before any integer cast.
            values = np.asarray(row['local_indices'], dtype=object)
            if values.ndim != 1 or any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer)) for i in values.flat
            ):
                raise ValueError('neuron indices must be a 1D integer index set')
            if (any(i < 0 or i >= self.n for i in values)
                    or len(set(values)) != values.size):
                raise ValueError('invalid neuron indices')
            # Own the validated indices: caller array mutation cannot reroute pools.
            self.groups.append(np.array(values, dtype=int, copy=True))
        self.reset()

    def reset(self):
        self.rates = np.zeros(self.nu)
        self.elapsed_s = 0.0
        self.interval = -1

    def consume(self, spikes, *, spike_dt_s, start_s, end_s, connected=True):
        """Consume an already completed interval; returned control is valid AT end_s.

        Caller must never apply this command before end_s. A cut clears translator
        memory, not neural state or physical muscle activation.
        """
        sp = np.asarray(spikes)
        if sp.ndim != 2 or sp.shape[1] != self.n or sp.shape[0] < 1:
            raise ValueError('expected nonempty time-by-neuron spike matrix')
        if not np.all((sp == 0) | (sp == 1)):
            raise ValueError('binary finite spikes required')
        if not all(math.isfinite(v) for v in (spike_dt_s, start_s, end_s)) or spike_dt_s <= 0:
            raise ValueError('invalid timing')
        if not math.isclose(start_s, self.elapsed_s, abs_tol=1e-12, rel_tol=0):
            raise ValueError('noncontiguous or replayed source interval')
        if not math.isclose(end_s-start_s, sp.shape[0]*spike_dt_s, abs_tol=1e-12, rel_tol=0):
            raise ValueError('interval duration mismatch')
        decay = math.exp(-spike_dt_s / self.config.filter_tau_s)
        counts = np.array([int(sp[:, idx].sum()) if idx.size else 0 for idx in self.groups])
        if connected:
            for sample in sp:
                instantaneous = np.array([sample[idx].mean()/spike_dt_s if idx.size else 0.0
                                          for idx in self.groups])
                self.rates = decay*self.rates + (1-decay)*instantaneous
        else:
            self.rates[:] = 0
        ctrl = np.clip(self.rates / self.config.rate_for_full_control_hz,
                       0, self.config.max_control)
        self.elapsed_s = float(end_s)
        self.interval += 1
        return ctrl, dict(source_interval=self.interval, source_start_s=float(start_s),
                          source_end_s=float(end_s), valid_from_s=float(end_s),
                          pool_spike_counts=counts.tolist(), rates_hz=self.rates.tolist(),
                          connected=bool(connected), saturated=(ctrl >= self.config.max_control).tolist(),
                          note='Nonzero filtered commands may retain previous spikes; no independent bias')
