"""Verify population, checkpoint identity, and the paired holdout evaluations."""
import collections
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from practice import (HERE, PARENT, PRIOR, RUN, ObservedBody, digest, trials, write, torch)


def read(path):
    return json.loads(path.read_text())


def main():
    training = read(RUN / 'training.json')
    assert training['updates'] == 28800
    assert training['physical_episodes'] == 2800
    assert training['checkpoints'][-1]['step'] == 28800
    assert training['checkpoints'][-1]['training_episodes'] == 2800
    assert training['weights_sha256'] == digest(RUN / 'weights.pt')
    assert training['parent_sha256'] == digest(PARENT / 'weights.pt')
    model = ObservedBody(RUN / 'weights.pt')
    parent = ObservedBody(PARENT / 'weights.pt')
    assert all(torch.equal(a, b) for a, b in zip(model.raw.feedback.parameters(), parent.raw.feedback.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(model.raw.motion.parameters(), parent.raw.motion.parameters()))
    test = trials(3000, 3020)
    fresh = trials(20, 2792)
    paths = [PRIOR / 'episodes' / f'terrain{i:02d}' for i in range(20)] + test + fresh
    assert len({digest(d/'terrain.npz') for d in paths}) == len(paths)
    histogram = collections.Counter()
    receipt_travel = []
    for d in fresh:
        row = read(d/'receipt.json')
        assert row['split'] == 'train' and row['seed'] in (10901, 11311)
        assert row['steps'] == 20000 and row['supported_foot_max_surface_um'] > 20
        histogram[(row['seed'], row['mode'])] += 1
        receipt_travel.append(row['travel_mm'])
    for d in test:
        row = read(d/'receipt.json')
        assert row['split'] == 'test' and row['seed'] in (11701,11903)
    results = read(RUN/'evaluation.json')
    assert collections.Counter(r['split'] for r in results) == {'flat':7, 'old_terrain':4, 'new_terrain':20}
    means = read(RUN/'summary.json')
    figure, axes = plt.subplots(ncols=2, figsize=(9, 3.5), layout='constrained')
    splits = ('flat','old_terrain','new_terrain')
    keys = ('error_50ms_mm','estimated_tail_rms_mm_s')
    for ax, key in zip(axes, keys):
        for k, (arm, label, color) in enumerate([('baseline','226-needle parent','#64748b'),('prior_practice','28-episode practice','#bc7732'),('practice100','2,800-episode practice','#228a74')]):
            ax.bar([i+(k-1)*.24 for i in range(3)], [means[s][arm][key] for s in splits], .22,
                   color=color, label=label)
        ax.set_xticks(range(3), ('Flat\n7 trials','Prior hills\n4 trials','New hills\n20 trials'))
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('50 ms marker error (mm)')
    axes[1].set_ylabel('Final 200 ms speed RMS (mm/s)')
    axes[0].legend(loc='upper left', fontsize=8)
    figure.savefig(RUN/'comparison.png', dpi=150)
    data = dict(new_train_episodes=len(fresh), reused_train_episodes=28, new_terrain_test=len(test),
                native_train_steps=len(fresh)*20000, observed_training_frames=2800*400, updates=28800,
                archived_command_counts={f'{seed}_{mode}': n for (seed,mode), n in sorted(histogram.items())},
                new_train_distance_range_mm=[min(receipt_travel),max(receipt_travel)],
                frozen_feedback_verified=True, weights_sha256=training['weights_sha256'])
    write(RUN/'audit.json', data)
    print(json.dumps(data, indent=2))


if __name__=='__main__':
    main()
