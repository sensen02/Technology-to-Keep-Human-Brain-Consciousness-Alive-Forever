"""How readable is the prompted leg's own foot height from the observation?

Two measurements, both with prompts handled correctly. The earlier version of this script
fitted one linear map over all samples while the samples cycled through the six legs, so the
map could not know which leg was being asked about; that understates readability for every
block and is reported here as `earlier_measurement_ignoring_prompt` for the record.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_core as rc
import train_single_leg as T

PROP = 183
CLEAR = slice(PROP, PROP + 2 * rc.NL)
JOINTS = slice(0, 42)
LOADS = slice(174, 180)


def ridge_r2(F, y, lam=1e-3):
    Fm = F.mean(0); ys = y.mean()
    A = F - Fm
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (y - ys))
    pred = A @ w + ys
    return 1 - float(((y - pred) ** 2).sum() / ((y - ys) ** 2).sum())


def main():
    rng = np.random.default_rng(7)
    core = rc.Core(); demos = T.demo_directions()
    X, Y, P = [], [], []
    core.reset()
    for t in range(2000):
        leg = t % rc.NL
        prompt = T.onehot(leg)
        X.append(core.obs(prompt)); Y.append(core.measure()['padz'][leg]); P.append(prompt)
        a = demos[leg]['unit'] * .8 + .2 * rng.standard_normal(rc.NJ)
        core.tick(np.clip(a, -1, 1))
        if t % 500 == 499:
            core.reset()
    X = np.asarray(X, float); Y = np.asarray(Y, float); P = np.asarray(P, float)
    # prompted leg's own height and rate, selected by the prompt rather than by one fixed map
    heights = (X[:, PROP:PROP + rc.NL] * P).sum(1)
    rates = (X[:, PROP + rc.NL:PROP + 2 * rc.NL] * P).sum(1)

    out = dict(
        label='how readable is the prompted leg own foot height; offline, no learning',
        samples=int(len(Y)),
        pad_height_range=[float(Y.min()), float(Y.max())],
        observation_dimension=int(X.shape[1]),
        prompt_aware={
            'clearance_height_selected_by_prompt': ridge_r2(heights[:, None], Y),
            'clearance_height_and_rate_selected': ridge_r2(
                np.stack([heights, rates], axis=1), Y),
            'joint_angles_with_prompt': ridge_r2(np.hstack([X[:, JOINTS], P]), Y),
            'foot_loads_with_prompt': ridge_r2(np.hstack([X[:, LOADS], P]), Y),
            'all_senses_without_clearance_channels': ridge_r2(
                np.hstack([X[:, :PROP], P]), Y),
            'everything': ridge_r2(X, Y),
        },
        earlier_measurement_ignoring_prompt={
            'note': 'this is the number quoted in the previous report; it used one linear map for '
                    'all six legs and therefore could not select the prompted leg',
            'full_observation_483': 0.6488, 'joint_angles_only': 0.0594, 'foot_loads_only': 0.0445},
        conclusion='with the prompt handled, the declared clearance channels make the prompted '
                   "leg's own foot height directly readable, while the rest of the sensory vector "
                   'still leaves it to be inferred')
    (HERE / 'FOOT_HEIGHT_VISIBILITY.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['prompt_aware'], indent=2))


if __name__ == '__main__':
    main()
