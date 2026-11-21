"""Compare frozen two-bank direction decoding with idealized outside-bank spike block.

This is a software counterfactual, not a drug/dose or tissue diffusion model.
The source generator uses known anatomical source side; neither decoder sees it.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = PROJECT / 'experiments/open-brain'
RECORD = BASE / '20261120T043500Z-neck-recording'
SORT = BASE / '20261120T051324Z-neck-spike-sorting'
sys.path[:0] = [str(SORT), str(RECORD)]
from sorting import detect
from evaluate import reference
from run import contact_layout, lead_vectors, load_fixed_neck, voltages

SEED = 11701
REPEATS = ((11702, 11703), (11704, 11705), (11706, 11707))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(pred, truth):
    actual = Counter(truth)
    calls = Counter(pred)
    hit = actual & calls
    out = {}
    for label in ('down', 'up'):
        n = sum(v for k, v in actual.items() if k[2] == label)
        c = sum(v for k, v in calls.items() if k[2] == label)
        h = sum(v for k, v in hit.items() if k[2] == label)
        out[label] = dict(correct=h, called=c, actual=n,
                          precision=h / c if c else None,
                          recall=h / n if n else None,
                          f1=2*h/(c+n) if c+n else None)
    n, c, h = (sum(row[key] for row in out.values()) for key in ('actual', 'called', 'correct'))
    out['pooled'] = dict(correct=h, called=c, actual=n,
                         precision=h / c if c else None,
                         recall=h / n if n else None,
                         f1=2*h/(c+n) if c+n else None)
    return out


def labeled(pred, arm):
    if arm == 'algorithm':
        return [(int(h), int(t), 'down' if h < t else 'up') for h, t, _ in pred]
    return [(int(h), int(t), arm) for h, t, _ in pred]


def main():
    HERE.mkdir(exist_ok=True, parents=True)
    out = HERE / 'RESULT.json'
    if out.exists():
        raise FileExistsError(out)
    paths, catalog, params, identity = load_fixed_neck(PROJECT)
    paths = {p.root: p for p in paths}
    with np.load(RECORD / f'recordings/{SEED}_intact_reference.npz', allow_pickle=False) as z:
        roots = z['roots'].astype(str)
        directions = z['direction'].astype(str)
        depart, arrive = z['entry'], z['arrival']
    assert len(roots) == 19 and list(directions) == ['DN'] * 17 + ['AN'] * 2
    with np.load(RECORD / f'recordings/{SEED}_intact_observation.npz', allow_pickle=False) as z:
        observed = np.column_stack((z['head_uv'], z['thorax_uv']))
    leads = lead_vectors(paths, roots, contact_layout(catalog))
    noise = np.random.default_rng(SEED).normal(0., .08, observed.shape)
    healthy = voltages(depart, arrive, leads, noise)
    np.testing.assert_array_equal(healthy, observed, err_msg='source voltage did not reproduce archive')
    # A head-side gate outside both banks blocks head-origin DN before recording;
    # chest-origin AN has already crossed both banks before that gate. Reverse for
    # a chest-side gate. This source-side mask is the idealized intervention model.
    dn = directions == 'DN'
    an = directions == 'AN'
    def blocked(keep, local_noise):
        return voltages(depart * keep, arrive * keep, leads, local_noise)
    with np.load(SORT / 'holdout/templates.npz', allow_pickle=False) as z:
        bank, lags, scale = (z[key] for key in ('bank', 'signed_lag', 'scale'))
    def sort(voltage):
        return detect(voltage, bank, lags, scale, multichannel=True, subtract=True)
    algorithm_pred = sort(observed)
    with np.load(SORT / 'holdout/predictions.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(algorithm_pred, z['intact_combined'])
    # Only after every voltage-only prediction do we load event identity for scoring.
    # The anatomical side labels above are part of the intervention generator only.
    predictions = {'algorithm': algorithm_pred}
    cases = [('same_noise', noise, noise)]
    cases += [(f'independent_noise_{i}',
               np.random.default_rng(a).normal(0., .08, observed.shape),
               np.random.default_rng(b).normal(0., .08, observed.shape))
              for i, (a, b) in enumerate(REPEATS, 1)]
    diagnostics = {}
    for name, head_noise, chest_noise in cases:
        head_block = blocked(an, head_noise)
        chest_block = blocked(dn, chest_noise)
        # Difference is measured against intact voltage, then the same frozen
        # sorter locates both endpoints. Direction is assigned by blocked side.
        removed_by_head = observed - head_block
        removed_by_chest = observed - chest_block
        down_pred = sort(removed_by_head)
        up_pred = sort(removed_by_chest)
        predictions[name + '_down'] = down_pred
        predictions[name + '_up'] = up_pred
        diagnostics[name] = {
            'down_calls': len(down_pred), 'up_calls': len(up_pred),
            'head_residual_max_uV': float(np.max(abs(removed_by_head))),
            'chest_residual_max_uV': float(np.max(abs(removed_by_chest))),
        }
    np.savez_compressed(HERE / 'PREDICTIONS.npz', **predictions)
    truth = [(h, t, 'down' if h < t else 'up') for h, t, _ in reference(SEED, 'intact')]
    scores = {'algorithm': metrics(labeled(algorithm_pred, 'algorithm'), truth)}
    for name, _, _ in cases:
        pred = labeled(predictions[name + '_down'], 'down') + labeled(predictions[name + '_up'], 'up')
        scores[name] = metrics(pred, truth)
    assert scores['algorithm']['down']['correct'] == 309
    assert scores['algorithm']['up']['correct'] == 204
    result = dict(
        scope='idealized outside-bank local spike block; same held neural events and aligned trials',
        not_modeled=['TTX concentration', 'diffusion', 'channel occupancy', 'tissue damage',
                     'response changes after blockade', 'independent biological trials'],
        seed=SEED, sample_ms=.1, roots=19, channels=16,
        block_geometry='head-side and chest-side regions outside both recorded planes',
        readout='frozen multichannel two-bank template sorter; block side supplies direction label',
        independent_noise_seeds=REPEATS,
        source_sha256={str(p): sha(p) for p in (
            RECORD / f'recordings/{SEED}_intact_reference.npz',
            RECORD / f'recordings/{SEED}_intact_observation.npz',
            SORT / 'holdout/templates.npz',
            SORT / 'holdout/predictions.npz')},
        generator_reproduces_archived_voltage=True,
        algorithm_prediction_exact_replay=True,
        diagnostics=diagnostics, scores=scores)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
