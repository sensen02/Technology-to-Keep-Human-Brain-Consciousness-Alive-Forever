"""BANC local, identity-preserving illustrative spiking conductance dynamics.

No body/target/time/reference input; currents in nA only on declared ports.
ACh excitation and GABA/glutamate inhibition are illustrative receptor assumptions,
not receptor measurements. Unknown/mixed labels have no active synapses.
"""
from pathlib import Path
import csv
import importlib.util
import json
import sys
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE = PROJECT / 'experiments/flybody-retarget/20261107T182940Z/neural-interface'
# Load the existing implementation directly, independent of cwd and engine package side effects.
_ENGINE_NAME = '_banc_local_existing_neural_cond'
if _ENGINE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_ENGINE_NAME, PROJECT / 'engine/neural_cond.py')
    _engine = importlib.util.module_from_spec(_spec)
    sys.modules[_ENGINE_NAME] = _engine
    _spec.loader.exec_module(_engine)
ConductanceNetwork = sys.modules[_ENGINE_NAME].ConductanceNetwork
ConductanceParams = sys.modules[_ENGINE_NAME].ConductanceParams


class LocalCircuit:
    """20,000 real IDs, strict original edges; reset starts at rest without drives.

    total_conductance_uS is restricted to the preregistered engineering candidates.
    None loads CHOSEN_CONFIG.json when available, otherwise provisional 0.02.
    dt_ms admits only the declared nominal/refinement steps 0.5/0.25 ms.
    net is exposed for scientific inspection; direct net mutation bypasses API guards.
    """
    def __init__(self, *, total_conductance_uS=None, dt_ms=0.5):
        if total_conductance_uS is None:
            chosen = HERE / 'CHOSEN_CONFIG.json'
            total_conductance_uS = json.loads(chosen.read_text())['total_conductance_uS'] if chosen.exists() else 0.02
        if total_conductance_uS not in (0.005, 0.02, 0.08):
            raise ValueError('conductance must be a preregistered value: .005, .02, .08 uS')
        if dt_ms not in (0.5, 0.25):
            raise ValueError('dt_ms must be 0.5 or 0.25')
        self.total_conductance_uS = float(total_conductance_uS)
        self.dt_ms = float(dt_ms)
        self.filter_tau_ms = 20.0
        with np.load(SOURCE / 'LOCAL_MODULE_CANDIDATE.npz', allow_pickle=False) as z:
            self.root_ids = z['root_ids'].copy()
            self.cache_indices = z['cache_indices'].copy()
            pre, post, syn, nt = (z[k] for k in ('pre_local', 'post_local', 'synapse_counts', 'source_verified_nt'))
            eligible = np.isin(nt, ['acetylcholine', 'gaba', 'glutamate'])
            if not np.array_equal(eligible, z['pure_verified_label_mask']):
                raise ValueError('strict label mask disagrees with exact pure source labels')
            if len(self.root_ids) != 20000 or len(pre) != 705068 or eligible.sum() != 435126:
                raise ValueError('candidate counts changed')
            self.strict_source_edge_rows = z['source_edge_rows'][eligible].copy()
            if not np.array_equal(self.strict_source_edge_rows, z['strict_source_edge_rows']):
                raise ValueError('strict source rows changed')
            pre, post = pre[eligible], post[eligible]
            raw = syn[eligible].astype(np.float64)
            if not np.isfinite(raw).all() or np.any(raw <= 0):
                raise ValueError('invalid source synapse counts')
            incoming = np.bincount(post, weights=raw, minlength=20000)
            # Across both receptor types: sum incoming increments = selected total uS.
            weight = self.total_conductance_uS * raw / incoming[post]
            exc = nt[eligible] == 'acetylcholine'
            self._We = sparse.csr_matrix((weight[exc], (post[exc], pre[exc])), shape=(20000, 20000))
            self._Wi = sparse.csr_matrix((weight[~exc], (post[~exc], pre[~exc])), shape=(20000, 20000))
            self.incoming_raw_synapse_counts = incoming
        if np.unique(self.root_ids).size != 20000 or np.unique(self.cache_indices).size != 20000:
            raise ValueError('duplicate identity')
        with np.load(SOURCE / 'GROUP_CACHE_INDICES.npz', allow_pickle=False) as g:
            local = lambda key: np.flatnonzero(np.isin(self.cache_indices, g[key]))
            self.source_sensory_local = local('leg_sensory_nerve_union')
            self.descending_local = local('descending_boundary')
            self.ascending_local = local('ascending_boundary')
        review = json.loads((SOURCE / 'SENSORY_CHANNEL_REVIEW.json').read_text())
        excluded = [int(r['cache_index_0based']) for r in review['ambiguous_records']]
        self.excluded_ambiguous_sensory_local = self.source_sensory_local[np.isin(self.cache_indices[self.source_sensory_local], excluded)]
        self.allowed_sensory_local = np.setdiff1d(self.source_sensory_local, self.excluded_ambiguous_sensory_local)
        with (SOURCE / 'LOCAL_MOTOR_READOUT_PORTS.csv').open(newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['readout_eligible_single_leg'] == 'True']
        self.mn_local = np.array([int(r['local_index_0based']) for r in rows], dtype=np.int64)
        self.mn_leg = np.array([r['nerve_matched_legs'] for r in rows])
        if len(rows) != 323 or not np.array_equal(self.root_ids[self.mn_local], [int(r['root_id_string']) for r in rows]):
            raise ValueError('motor CSV identities/order changed')
        self._sensory_mask = np.zeros(20000, bool)
        self._descending_mask = np.zeros(20000, bool)
        self._sensory_mask[self.allowed_sensory_local] = True
        self._descending_mask[self.descending_local] = True
        if np.any(self._sensory_mask & self._descending_mask) or np.any((self._sensory_mask | self._descending_mask)[self.mn_local]):
            raise ValueError('input roles overlap or directly include motor readout cells')
        for a in (self.root_ids, self.cache_indices, self.mn_local, self.mn_leg,
                  self.allowed_sensory_local, self.descending_local, self.ascending_local,
                  self.source_sensory_local, self.excluded_ambiguous_sensory_local):
            a.flags.writeable = False
        self.reset()

    def reset(self):
        """Reset all voltage, conductance, delays, spikes and filters, retaining config."""
        self.net = ConductanceNetwork(20000, ConductanceParams(), dt_ms=self.dt_ms,
                                      voltage_method='backward_euler')
        self.net.set_connectivity(self._We, self._Wi)
        self.filtered_rates_hz = np.zeros(20000, dtype=np.float64)
        return self

    def advance(self, sensory_current_nA, descending_current_nA, duration_ms=5.):
        """Hold currents for duration; return 323 filtered MN Hz in eligible CSV order.

        Spike filter: r <- exp(-dt/20ms)*r + (1000/20ms)*spike.
        This is a causal exponential point-process filter, sampled at step endpoints.
        """
        s = np.asarray(sensory_current_nA, dtype=np.float64)
        d = np.asarray(descending_current_nA, dtype=np.float64)
        for a, mask, name in ((s, self._sensory_mask, 'sensory'), (d, self._descending_mask, 'descending')):
            if a.shape != (20000,) or not np.isfinite(a).all():
                raise ValueError(name + ' current must be a finite (20000,) vector')
            if np.any(a[~mask] != 0):
                raise ValueError(name + ' current outside allowed input identities')
        ratio = float(duration_ms) / self.dt_ms
        if not np.isfinite(ratio) or ratio <= 0 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError('duration_ms must be a positive integer number of neural steps')
        current = s + d
        if not np.isfinite(current).all():
            raise ValueError('summed current is not finite')
        decay = np.exp(-self.dt_ms / self.filter_tau_ms)
        for _ in range(int(round(ratio))):
            spikes = self.net.step(current_nA=current)
            self.filtered_rates_hz *= decay
            self.filtered_rates_hz += (1000. / self.filter_tau_ms) * spikes
            if not all(np.isfinite(a).all() for a in (self.net.v, self.net.ge, self.net.gi, self.net.adaptation, self.filtered_rates_hz)):
                raise FloatingPointError('nonfinite neural state')
        return self.filtered_rates_hz[self.mn_local].copy()
