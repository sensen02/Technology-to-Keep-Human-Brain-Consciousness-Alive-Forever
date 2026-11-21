"""Fit a regularized brain-side stretch readout from fixed-dose Eon activity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def examples(folder):
    with np.load(folder / 'TRACE.npz', allow_pickle=False) as trace:
        rates, labels = trace['features'][14:], trace['crop_before'][14:]
    columns = json.loads((folder / 'BINDING.json').read_text())['feature_roots']
    return rates.reshape(-1, rates.shape[-1]), labels.ravel(), columns


def fit(doses, output):
    datasets = [examples(folder) for folder in doses]
    roots = datasets[0][2]
    assert all(columns == roots for _, _, columns in datasets)
    x = np.concatenate([a for a, _, _ in datasets])
    y = np.concatenate([b for _, b, _ in datasets])
    mean_x, mean_y = x.mean(axis=0), y.mean()
    scale = np.maximum(1., x.std(axis=0))
    z = (x - mean_x) / scale
    alpha = 16.
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(len(roots)), z.T @ (y - mean_y)) / scale
    bias = mean_y - mean_x @ beta
    metrics = {}
    for folder, (features, target, _) in zip(doses, datasets, strict=True):
        prediction = features @ beta + bias
        metrics[folder.name] = dict(
            mean_absolute_error=float(np.mean(abs(prediction - target))),
            mean_prediction_per_dose=prediction.reshape(-1, 4).mean(axis=0).tolist(),
            source_sha256=hashlib.sha256((folder / 'TRACE.npz').read_bytes()).hexdigest())
    model = dict(feature_roots=roots, coefficients=beta.tolist(), intercept=float(bias),
                 training_samples=len(y), regularization=alpha, fit='one closed-form ridge solve',
                 eon_weight_updates=0, metrics=metrics)
    with output.open('x') as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('doses', type=Path, nargs='+')
    options = p.parse_args()
    fit(options.doses, options.output)
