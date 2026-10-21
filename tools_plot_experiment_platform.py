"""Plot a completed P1 evidence bundle; does not simulate or change the bundle."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np

ARMS = ('normal', 'normal_repeat', 'replay_baseline', 'feedback_push',
        'replay_push', 'sensory_cut', 'motor_cut', 'output_silence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output == bundle or bundle in output.parents or output.exists():
        parser.error('output must be a new file outside the bundle')
    result = json.loads((bundle / 'result.json').read_text(encoding='utf-8'))
    traces = bundle / 'traces.npz'
    if hashlib.sha256(traces.read_bytes()).hexdigest() != result['traces_sha256']:
        raise ValueError('trace hash mismatch')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    with np.load(traces, allow_pickle=False) as arrays:
        for arm in ARMS:
            times = arrays[arm + '_interval_end_s'] * 1000
            records = result['episodes'][arm]['records']
            heights = [row['height_model'] / result['episodes'][arm]['initial_height_model']
                       for row in records]
            axes[0].plot(times, heights, label=arm)
            axes[1].plot(times, np.max(arrays[arm + '_controls'], axis=1))
            axes[2].plot(arrays[arm + '_interval_start_s'] * 1000,
                         np.sum(arrays[arm + '_sensory_current_nA'], axis=1))
        axes[0].axhline(.9, color='black', linestyle='--', label='0.9 initial height')
    axes[0].set_ylabel('Thorax height / initial height')
    axes[1].set_ylabel('Max applied muscle command')
    axes[2].set_ylabel('Actual sensory current sum (nA)')
    axes[2].set_xlabel('Simulation time (ms); input sampled at interval start')
    axes[0].legend(fontsize=8, ncol=3)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle('P1 eight-arm short baseline — not standing or biological validation')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing an earlier delivered figure.
    with output.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=150)
    plt.close(fig)
    print(str(output))


if __name__ == '__main__':
    main()
