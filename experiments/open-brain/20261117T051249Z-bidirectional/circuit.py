"""Tick-level same-donor BANC interface using the frozen open brain's LIF kernel."""
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
import torch

HERE = Path(__file__).resolve().parent
FOUNDATION = HERE.parent / '20261117T045901Z-foundation'
sys.path.insert(0, str(FOUNDATION / 'upstream/code'))
import run_pytorch as eon

DT_MS = eon.DT
TRANSMISSION_TICKS = 20


class Circuit:
    """Simulate selected BANC ports, mapped receptor inputs and first-hop receivers."""

    def __init__(self, catalog, batch=1):
        self.ports = catalog['ports']
        self.port_roots = list(self.ports)
        local, neck = [], []
        roots = set(self.port_roots)
        for root, port in self.ports.items():
            for receptor, count in port['incoming_receptors']:
                roots.add(receptor)
                local.append((receptor, root, count))
            for target, count in port['outgoing']:
                roots.add(target)
                neck.append((root, target, count * port['sign']))
        self.roots = sorted(roots)
        self.index = {root: idx for idx, root in enumerate(self.roots)}
        self.port_index = {root: idx for idx, root in enumerate(self.port_roots)}
        self.port_columns = [self.index[root] for root in self.port_roots]
        self.receptor_roots = {receptor for receptor, _, _ in local}
        self.receptor_roots.update(root for root, p in self.ports.items() if p['direction'] == 'SA')
        self.brain_roots = {root for root, p in self.ports.items() if p['direction'] == 'DN'}
        self.allowed_drive_roots = self.brain_roots | self.receptor_roots
        self.allowed_drive_columns = {self.index[root] for root in self.allowed_drive_roots}
        self.brain_columns = {self.index[root] for root in self.brain_roots}
        self.dn_ports = [self.port_index[root] for root in self.port_roots if root in self.brain_roots]
        self.local = self._matrix(local, self.index)
        self.neck = self._matrix(neck, self.port_index)
        self._cells = eon.AlphaLIF(batch, len(roots), DT_MS, eon.MODEL_PARAMS)
        self._state = self._cells.state_init()
        self._pending = np.zeros((TRANSMISSION_TICKS, batch, len(self.ports)), dtype=bool)
        self._tick = 0
        self.batch = batch

    def _matrix(self, edges, sources):
        pres, posts, values = zip(*[(sources[pre], self.index[post], count) for pre, post, count in edges], strict=True)
        return csr_matrix((np.asarray(values, dtype=np.float32), (posts, pres)), shape=(len(self.roots), len(sources)))

    def validate_inputs(self, roots, side):
        if side not in ('brain', 'body'):
            raise ValueError('side must be brain or body')
        allowed = self.brain_roots if side == 'brain' else self.receptor_roots
        if not set(roots) <= allowed:
            raise ValueError(f'{side} input is outside the verified stimulus ports')
        return [self.index[root] for root in roots]

    @torch.inference_mode()
    def step(self, drive, cut, eon_output=None):
        """Advance one step. External Eon events can enter DN axons only."""
        if drive.shape != (self.batch, len(self.roots)) or drive.dtype != np.float32:
            raise ValueError('drive must be indexed by the bound receptor or brain cells')
        if cut.shape != (len(self.ports),) or cut.dtype != np.bool_:
            raise ValueError('cut must name bound axons')
        driven = set(np.flatnonzero(np.any(drive != 0, axis=0)))
        if not driven <= self.allowed_drive_columns:
            raise ValueError('external drive into a downstream target would bypass the neck')
        if eon_output is not None:
            if eon_output.shape != self._pending.shape[1:] or eon_output.dtype != np.bool_:
                raise ValueError('invalid Eon output shape or type')
            cols = np.flatnonzero(np.any(eon_output, axis=0))
            if any(self.port_roots[index] not in self.brain_roots for index in cols):
                raise ValueError('Eon output is limited to reviewed DN axons')
            if driven & self.brain_columns:
                raise ValueError('select the same-donor brain or the Eon source for this tick')
        self._pending[:, :, cut] = False
        ring_index = self._tick % TRANSMISSION_TICKS
        arriving = self._pending[ring_index].copy()
        spikes = self._state[2].numpy()
        gain = eon.MODEL_PARAMS['wScale']
        volley = (self.local @ spikes.T).T + (self.neck @ arriving.T.astype(np.float32)).T
        state = self._cells(torch.from_numpy(volley * gain), torch.from_numpy(drive), *self._state)
        self._state = state
        source_events = state[2].numpy()[:, self.port_columns].astype(bool)
        if eon_output is not None:
            source_events[:, self.dn_ports] = eon_output[:, self.dn_ports]
        self._pending[ring_index] = source_events
        self._pending[ring_index, :, cut] = False
        self._tick += 1
        return source_events, arriving, state[2].numpy().astype(bool), state[3].numpy()
