"""Cached replay dataset and provenance manifest; clocks are never synchronized."""
import copy
import hashlib
from pathlib import Path
import threading
from .artifacts import strict_loads, digest
from .runs import Conflict


class DatasetRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.signature = None
        self.parse_count = 0

    def _refresh(self):
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if signature == self.signature:
            return
        raw = self.path.read_bytes()
        text = raw.decode('utf-8').strip()
        prefix = 'window.VIEWER_DATA='
        if not text.startswith(prefix):
            raise ValueError('unsupported dataset wrapper')
        data = strict_loads(text[len(prefix):].rstrip().removesuffix(';'))
        if not isinstance(data, dict):
            raise ValueError('dataset must be object')
        source_hash = hashlib.sha256(raw).hexdigest()
        datasets = []
        for name in ('body', 'neural', 'electrodes', 'calcium', 'bench'):
            value = data.get(name)
            available = isinstance(value, dict) and bool(value)
            value = value if available else {}
            times = value.get('times_s', [])
            kind = {'body':'simulation', 'neural':'illustration', 'electrodes':'illustration', 'calcium':'simulation', 'bench':'measurement'}[name]
            datasets.append({'dataset_id': name, 'source_id': 'viewer/data.js#' + name,
                'run_id': 'snapshot-' + source_hash[:16], 'source_hash': digest(value),
                'origin': value.get('provenance', 'legacy replay snapshot; provenance unspecified'),
                'kind': kind, 'available': available, 'availability': 'available' if available else 'missing',
                'units': {'position': 'mm' if name == 'body' else ('um' if name in ('neural','electrodes','calcium') else None),
                          'value': value.get('units'), 'time': 's' if times else None},
                'coordinate_frame': value.get('coordinate_frame'),
                'clock': {'clock_id': name + '-independent', 'unit': 's', 'times_s': times,
                          'frame_count': len(times), 'synchronized': False,
                          'origin_s': times[0] if times else None},
                'data_url': '/api/datasets/' + name + '?snapshot=' + source_hash, 'data_key': name})
        self.data = data
        self.manifest_data = {'schema_version': 'workbench-manifest-1',
            'dataset_id': 'viewer-replay', 'run_id': 'snapshot-' + source_hash[:16],
            'source_hash': source_hash, 'source_id': 'viewer/data.js',
            'mode': 'simulation replay; no live acquisition',
            'clock_policy': 'independent source seconds; normalized playback is not synchronization',
            'meta': copy.deepcopy(data.get('meta', {})),
            'datasets': datasets}
        self.signature = signature
        self.parse_count += 1

    def section(self, name, snapshot=None):
        if name not in ('body', 'neural', 'electrodes', 'calcium', 'bench'):
            raise KeyError('unknown dataset')
        with self.lock:
            self._refresh()
            if snapshot is not None and snapshot != self.manifest_data['source_hash']:
                raise Conflict('dataset snapshot changed; reload manifest and all sections')
            if name not in self.data:
                raise KeyError('dataset unavailable')
            return copy.deepcopy(self.data[name])

    def snapshot(self):
        with self.lock:
            self._refresh()
            return copy.deepcopy(self.data)

    def manifest(self):
        with self.lock:
            self._refresh()
            return copy.deepcopy(self.manifest_data)
