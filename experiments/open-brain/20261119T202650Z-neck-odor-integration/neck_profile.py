"""Apply one static OmniSim lesion before the first spike of a P1 ground trial."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
CATALOG = BASE / '20261117T051249Z-bidirectional/CATALOG.json'
FIXED = BASE / '20261117T164207Z-fixed-neck'


def configure(model, profile, mode):
    if model._tick != 0 or mode not in ('intact', 'shaft', 'target_only'):
        raise ValueError('static profile requires a fresh circuit and a known mode')
    data = json.loads(profile.read_text())
    for source in (CATALOG, FIXED / 'PARAMETERS.json', FIXED / 'fixed_neck.py'):
        if hashlib.sha256(source.read_bytes()).hexdigest() != data['source']['sha256'][source.name]:
            raise ValueError(f'OmniSim source mismatch for {source}')
    ports = json.loads(CATALOG.read_text())['ports']
    routes = {row['root']: row for row in data['routes']}
    if len(data['routes']) != len(ports) or routes.keys() != ports.keys():
        raise ValueError('profile root set differs from the frozen catalog')
    for root, row in routes.items():
        if any(row[k] != ports[root][k] for k in ('direction', 'cell_type', 'side')):
            raise ValueError(f'inconsistent anatomical identity for {root}')
        gain, delay = row['gain'], row['delay_s']
        if type(row['available']) is not bool or not (0 <= gain <= 1):
            raise ValueError(f'invalid availability or gain for {root}')
        if row['available']:
            if delay is None or not math.isfinite(delay) or delay <= 0:
                raise ValueError(f'invalid transit delay for {root}')
        elif gain != 0 or delay is not None:
            raise ValueError(f'inconsistent blocked route for {root}')
    target = data['selected']['injury']['root']
    if target not in model.port_index:
        raise ValueError('selected route is absent from the behavioral circuit')
    cut = np.zeros(len(model.port_roots), dtype=bool)
    gains = np.ones(len(cut))
    for i, root in enumerate(model.port_roots):
        if mode == 'intact' or (mode == 'target_only' and root != target):
            continue
        row = routes[root]
        cut[i] = not row['available']
        gains[i] = row['gain']
        if not cut[i]:
            model.neck_delay[i] = math.ceil(row['delay_s'] * 1000 / model.dt_ms - 1e-12)
    if (gains != 1).any():
        model.neck = [m.multiply(gains[None, :]).tocsr() for m in model.neck]
        front = model.front_sources
        front.feedback.matrices = [m[front.neck_cells] for m in model.neck]
    if len(model._pending) != model.neck_delay.max() + 1:
        model._pending = np.zeros((int(model.neck_delay.max())+1, model.batch, len(cut)), dtype=bool)
    if cut.any():
        original_step = model.step

        def step(drive, runtime_cut, eon_output=None):
            return original_step(drive, runtime_cut | cut, eon_output=eon_output)

        model.step = step
    receipt = dict(mode=mode, profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                   target=target, dt_ms=model.dt_ms, omitted_roots=sorted(routes.keys()-model.port_index.keys()),
                   applied=[dict(root=r, available=not bool(cut[i]), gain=float(gains[i]),
                                 delay_ticks=int(model.neck_delay[i]) if not cut[i] else None)
                            for i, r in enumerate(model.port_roots)])
    return cut, receipt
