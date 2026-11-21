"""Bind the current frozen cervical circuit and its exact registered SWC paths."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from ..provenance import file_digest


@dataclass(frozen=True)
class NeckPath:
    root: str
    cell_type: str
    side: str
    direction: str
    points_m: np.ndarray
    speed_m_s: float
    outgoing: tuple


def load_fixed_neck(project):
    entry = Path(project) / 'experiments/open-brain/NECK_DYNAMICS_CURRENT.json'
    manifest = json.loads(entry.read_text())
    fixed = Path(manifest['implementation_root'])
    identity = json.loads((fixed / 'IDENTITY.json').read_text())['sha256']
    chosen = {Path(k).name: v for k, v in identity.items()
              if Path(k).name in ('fixed_neck.py', 'PARAMETERS.json', 'CATALOG.json')}
    catalog_path = next(Path(k) for k in identity if Path(k).name == 'CATALOG.json')
    parameter_path = Path(manifest['parameters'])
    for path in (fixed / 'fixed_neck.py', parameter_path, catalog_path):
        if file_digest(path) != chosen[path.name]:
            raise ValueError(f'frozen neck source hash mismatch: {path}')
    catalog = json.loads(catalog_path.read_text())
    parameters = json.loads(parameter_path.read_text())
    if (catalog['ports'].keys() != parameters['ports'].keys()
            or len(catalog['ports']) != manifest['source_neck_ports']
            or catalog['source'] != 'BANC v888 official simple v2'):
        raise ValueError('fixed neck identities and geometry differ')

    @lru_cache(maxsize=1)
    def read_tree(path, expected):
        if file_digest(path) != expected:
            raise ValueError(f'frozen SWC hash mismatch: {path}')
        rows = np.loadtxt(path, comments='#')
        return {int(row[0]): (int(row[6]), row[2:5]) for row in rows}

    neck = []
    for root, port in catalog['ports'].items():
        physical = parameters['ports'][root]
        path = Path(physical['swc'])
        if (port['object']['filename'] != path.name or port['object']['sha256'] != physical['sha256']
                or port['geometry']['crossing_children'] != physical['crossing_children']):
            raise ValueError(f'geometry provenance differs for root {root}')
        tree = read_tree(path, physical['sha256'])
        a, b = physical['crossing_children']

        def ancestors(start, end):
            result = []
            while start in tree:
                result.append(start)
                if start == end:
                    return result
                start = tree[start][0]
                if len(result) > len(tree):
                    break
            return None

        chain = ancestors(b, a) or ancestors(a, b)
        if chain is None:
            raise ValueError(f'no registered cervical chain for {root}')
        front, rear = port['geometry']['front_nm'], port['geometry']['rear_nm']
        low, high = (rear, front) if chain[0] == b else (front, rear)
        points = np.vstack([low, *[tree[node][1] for node in chain[1:]], high])
        if chain[0] == b:
            points = points[::-1]
        if port['direction'] != 'DN':
            points = points[::-1]
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum() / 1000
        if not np.isclose(length, physical['length_um'], rtol=0, atol=1e-8):
            raise ValueError(f'path length mismatch for {root}')
        points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
        neck.append(NeckPath(root, port['cell_type'], port['side'], port['direction'], points * 1e-9,
                             physical['velocity_m_s'], tuple(tuple(edge) for edge in port['outgoing'])))
    return neck, catalog, parameters, dict(current_manifest=str(entry), catalog=str(catalog_path),
                                           parameters=str(parameter_path), runtime=str(fixed / 'fixed_neck.py'),
                                           sha256=chosen)
