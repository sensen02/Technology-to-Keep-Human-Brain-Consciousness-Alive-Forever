"""Explicit tonic current into identity-checked neurons, NOT a gait generator.

No default targets or stimulation. A caller must supply a sourced identity manifest
and the actual tier identity arrays. Metadata matching is not biological validation.
The interface never reads pose, commands muscles, adds edges, or steps a simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numbers
import numpy as np


def _real(value, name):
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _ids(values):
    out = []
    for value in values:
        # Floating point IDs silently lose precision for connectome root IDs.
        if isinstance(value, (bool, np.bool_, float, np.floating)):
            raise ValueError('root IDs must be positive integers or decimal strings, never floats')
        text = str(value)
        if not text.isdecimal() or int(text) <= 0 or str(int(text)) != text:
            raise ValueError('root IDs must be canonical positive decimal identities')
        out.append(text)
    if not out or len(set(out)) != len(out):
        raise ValueError('root IDs must be nonempty and unique')
    return tuple(out)


@dataclass(frozen=True)
class SourcedNeuronTarget:
    root_id: str
    cell_type: str
    super_class: str
    identity_source: str
    functional_source: str


class TonicNeuronCommand:
    """Half-open [onset_s, offset_s) current; input/output are auditable vectors.

    Identity arrays must describe exactly the local-index order used by net.step.
    Missing targets fail closed: never substitute the last/nearest/another neuron.
    Dataset name and SHA256 pin the annotation identity, not physiological truth.
    """

    def __init__(self, *, tier_root_ids, tier_cell_types, tier_super_classes,
                 dataset, dataset_sha256, manifest_dataset, manifest_sha256,
                 targets, amplitude_nA, onset_s, offset_s):
        if not isinstance(dataset, str) or not dataset.strip() or dataset != manifest_dataset:
            raise ValueError('target manifest dataset must match actual dataset')
        if (not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64
                or any(c not in '0123456789abcdef' for c in dataset_sha256)
                or dataset_sha256 != manifest_sha256):
            raise ValueError('target manifest SHA256 must match actual dataset')
        ids = _ids(tier_root_ids)
        types, classes = tuple(tier_cell_types), tuple(tier_super_classes)
        if len(types) != len(ids) or len(classes) != len(ids):
            raise ValueError('tier annotation arrays must align with root IDs')
        targets = tuple(targets)
        if not targets or any(not isinstance(t, SourcedNeuronTarget) for t in targets):
            raise ValueError('explicit sourced targets required')
        target_ids = _ids(t.root_id for t in targets)
        by_id = {v: i for i, v in enumerate(ids)}
        local = []
        records = []
        for target, rid in zip(targets, target_ids):
            for field in ('cell_type', 'super_class', 'identity_source', 'functional_source'):
                value = getattr(target, field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'target {rid}: missing {field}')
            if rid not in by_id:
                raise ValueError(f'target {rid} absent from tier; no fallback allowed')
            i = by_id[rid]
            if types[i] != target.cell_type or classes[i] != target.super_class:
                raise ValueError(f'target {rid}: annotation identity mismatch')
            local.append(i)
            records.append(dict(root_id=rid, local_index=i, cell_type=target.cell_type,
                                super_class=target.super_class,
                                identity_source=target.identity_source,
                                functional_source=target.functional_source))
        amplitude = _real(amplitude_nA, 'amplitude_nA')
        onset, offset = _real(onset_s, 'onset_s'), _real(offset_s, 'offset_s')
        if amplitude < 0 or onset < 0 or offset <= onset:
            raise ValueError('require amplitude >= 0 and 0 <= onset < offset')
        self._n = len(ids)
        self._local = np.asarray(local, dtype=np.int64)
        self._amplitude, self._onset, self._offset = amplitude, onset, offset
        self._metadata = dict(dataset=dataset, dataset_sha256=dataset_sha256,
                              tier_identity_sha256=hashlib.sha256(
                                  json.dumps(list(zip(ids, types, classes)), separators=(',', ':')).encode()
                              ).hexdigest(), targets=records, amplitude_nA=amplitude,
                              onset_s=onset, offset_s=offset, interval='[onset_s, offset_s)',
                              kind='tonic neuron current, not gait or muscle controller',
                              biological_calibration='NOT_ESTABLISHED')

    def current(self, time_s):
        t = _real(time_s, 'time_s')
        if t < 0:
            raise ValueError('time_s must be nonnegative')
        out = np.zeros(self._n, dtype=np.float64)
        if self._onset <= t < self._offset:
            out[self._local] = self._amplitude
        return out

    def compose(self, sensory_current_nA, time_s):
        sensory = np.asarray(sensory_current_nA)
        if sensory.dtype != np.float64 or sensory.shape != (self._n,) or not np.isfinite(sensory).all():
            raise ValueError('sensory current must be a finite float64 tier vector')
        command = self.current(time_s)
        total = sensory.copy()
        # Only selected cells are touched, preserving all other bytes incl. -0.
        if command[self._local].any():
            with np.errstate(over='ignore', invalid='ignore'):
                total[self._local] += command[self._local]
        if not np.isfinite(total).all():
            raise ValueError('total current overflow; no clipping applied')
        return dict(sensory_current_nA=sensory.copy(), command_current_nA=command,
                    total_current_nA=total)

    def describe(self):
        return json.loads(json.dumps(self._metadata))
