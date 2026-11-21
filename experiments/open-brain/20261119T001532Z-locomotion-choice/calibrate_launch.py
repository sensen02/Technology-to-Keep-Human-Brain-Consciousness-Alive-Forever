"""Test whether Eon can produce launch bursts from upstream input alone."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from cervical import FlightPaths, THRESHOLD

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261118T231627Z-open-fruit-loop'))
from fastbrain import FastBrain, eon, load_connectome

DT = .0001
STEADY = 1800
N = 3000


def candidates(weights, dn, roots_by_index):
    rows = weights[dn[:2]].toarray()
    pos = rows > 0
    common = np.argsort(-np.minimum(*rows))
    common = common[np.all(pos[:, common], axis=0)]
    pair = []
    own = set(dn)
    for i in range(2):
        indices = np.argsort(-rows[i])
        ranked = [int(j) for j in indices if rows[i, j] > 0 and j not in own
                  and rows[1 - i, j] < .15 * rows[i, j]]
        pair.append(ranked[:8])
    ranks = {
        'bilateral2': list(common[:2]),
        'bilateral6': list(common[:6]),
        'bilateral12': list(common[:12]),
        'homolateral8': pair[0][:4] + pair[1][:4],
        'mixed8': list(common[:4]) + pair[0][:2] + pair[1][:2],
    }
    out = {}
    for key, selection in ranks.items():
        assert not own.intersection(selection)
        out[key] = ([int(roots_by_index[int(i)]) for i in selection], selection)
    return out


@torch.inference_mode()
def run(out):
    torch.set_num_threads(1)
    template = FlightPaths()
    index, _ = eon.get_hash_tables(eon.path_comp)
    inv = {i: root for root, i in index.items()}
    weights = load_connectome(index)
    flight = [index[root] for root in template.eon_roots]
    outgoing = weights.T.tocsr()
    choices = candidates(weights, flight, inv)
    scans = []
    for name, (roots, indices) in choices.items():
        for rate in (45., 100., 180., 250.):
            for seed in (241,):
                brain = FastBrain(outgoing)
                drive = np.array(indices)
                active = np.random.default_rng(seed).random((N, len(drive)), dtype=np.float32) < rate * DT
                active[:250] = False
                active[STEADY:] = False
                events = np.zeros((N,4), dtype=bool)
                for tick, inputs in enumerate(active):
                    brain.step(drive[inputs])
                    events[tick] = brain.spikes[flight]
                neck = FlightPaths()
                _, levels = neck.integrate(events)
                both = np.min(levels[:, :2], axis=1)
                k = np.flatnonzero(both >= THRESHOLD)
                row = dict(name=name, roots=roots, rate=rate, seed=seed, synaptic_weights=[
                           weights[flight[0], indices].toarray()[0].tolist(),
                           weights[flight[1], indices].toarray()[0].tolist()],
                           endogenous_spikes=events.sum(axis=0).tolist(),
                           first_bilateral_threshold_s=float(k[0] * DT) if len(k) else None,
                           peak=float(both.max()))
                scans.append(row)
                print(name, rate, row['endogenous_spikes'], row['first_bilateral_threshold_s'],
                      round(row['peak'],3), flush=True)
    out.write_text(json.dumps(scans, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
