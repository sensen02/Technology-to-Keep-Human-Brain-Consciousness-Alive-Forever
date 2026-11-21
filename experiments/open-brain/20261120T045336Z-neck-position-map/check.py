"""Learn preferred direction by dominant thoracic contact. Test once on intact held-out data."""
from pathlib import Path
import json
import sys
import numpy as np

BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261120T043500Z-neck-recording')
sys.path.insert(0, str(BASE))
from run import PROJECT, load_fixed_neck, contact_layout, lead_vectors, voltages
from decode import peaks


def events(voltage):
    times = peaks(voltage)
    sites = np.abs(voltage[times]).argmax(axis=1)
    return times, sites


def main():
    paths, catalog, _, _ = load_fixed_neck(PROJECT)
    counts = np.zeros((8, 2), dtype=int)
    for seed in (10901, 11311):
        path = BASE/'recordings'/f'{seed}_intact_reference.npz'
        with np.load(path) as ref:
            leads = lead_vectors({p.root: p for p in paths}, ref['roots'], contact_layout(catalog))
            noise = np.random.default_rng(seed).normal(0., .08, (len(ref['entry']), 16))
            for i, direction in enumerate(('DN', 'AN')):
                active = ref['direction'] == direction
                uv = voltages(ref['entry'][:, active], ref['arrival'][:, active], leads[active], noise)
                _, sites = events(uv[:, 8:])
                counts[:, i] += np.bincount(sites, minlength=8)
    # A site requires at least 20 calibration peaks and 90% directional purity.
    total = counts.sum(axis=1)
    confident = (total >= 20) & (counts.max(axis=1) >= .9*total)
    labels = np.where(confident, counts.argmax(axis=1), -1)
    with np.load(BASE/'recordings/11701_intact_observation.npz') as obs:
        times, sites = events(obs['thorax_uv'])
        prediction = labels[sites]
    with np.load(BASE/'recordings/11701_intact_reference.npz') as ref:
        dn = ref['arrival'][:, ref['direction']=='DN'].sum(axis=1)
        an = ref['entry'][:, ref['direction']=='AN'].sum(axis=1)
    result = {}
    for i, (target, other) in enumerate(((dn, an), (an, dn))):
        selected = times[prediction == i]
        # One peak can claim at most one event. Both directions at same tick are ambiguous.
        valid = (target[selected] > 0) & (other[selected] == 0)
        result[('down','up')[i]] = dict(correct=int(valid.sum()), called=len(selected),
                                        target_events=int(target.sum()),
                                        mixed_ticks=int(np.sum((target[selected]>0)&(other[selected]>0))))
    return dict(calibration_counts_down_up=counts.tolist(), site_labels=labels.tolist(),
                unknown_peaks=int((prediction == -1).sum()), test=result)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
