"""Read-only s78 recovery preflight. Never imports or advances a neural model.

A reservation without a committed state is charged, never reclaimed or replayed.
A ready verdict is necessary, not sufficient: model/config identities and a
single-writer lock must also be verified by a future execution entry point.
"""
import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ('factual', 'fixed_initial_input')
ARRAYS = ('_ring', 'v', 'ge', 'gi', 'adaptation',
          'refractory_until_ms', 'spike_count')


def inspect_recovery(directory, budget=800, ticks_per_arm=400):
    directory = Path(directory)
    rows = [json.loads(line) for line in
            (directory / 'CALLS.jsonl').read_text().splitlines()]
    if budget != 2 * ticks_per_arm or len(rows) > budget:
        raise ValueError('invalid budget or journal over budget')
    for i, row in enumerate(rows):
        expected = dict(arm=ARMS[i // ticks_per_arm],
                        tick=i % ticks_per_arm, total=i + 1)
        if row != expected:
            raise ValueError('noncontiguous or invalid reservation journal')
    if not rows:
        raise ValueError('empty journal: use a separately registered fresh entry')
    arm = rows[-1]['arm']
    path = directory / (arm + '_RECOVERY.npz')
    result = dict(arm=arm, reserved_calls=len(rows), budget=budget,
                  remaining_unreserved_calls=budget-len(rows),
                  neural_calls_executed_by_preflight=0,
                  physics_calls_executed_by_preflight=0,
                  checkpoint_path=str(path), safe_to_resume=False)
    if not path.exists():
        return dict(result, verdict='STOP_NO_CHECKPOINT')
    with np.load(path, allow_pickle=False) as z:
        required = {'current', 'spikes', 'pool_activity', 'features',
                    'raw_command', 'command', 'FIR_history', 'FIR_elapsed_s',
                    'FIR_interval', 'completed_tick', 'total_calls', 'arm',
                    'weights', 'actuator_pool', 'neural_scalars_json'}
        required.update('neural_' + k for k in ARRAYS)
        if not required.issubset(z.files):
            raise ValueError('incomplete state checkpoint')
        scalar = json.loads(str(z['neural_scalars_json'].item()))
        for key in ('voltage_method', 'n', 'dt', 'delay_steps', '_ring_i',
                    't_ms', 'steps'):
            if key not in scalar:
                raise ValueError('missing neural scalar: ' + key)
        tick = int(z['completed_tick'].item())
        committed = int(z['total_calls'].item())
        arm_offset = ARMS.index(arm) * ticks_per_arm
        if (str(z['arm'].item()) != arm or not 0 <= tick < ticks_per_arm
                or committed != arm_offset + tick + 1
                or committed > len(rows)):
            raise ValueError('checkpoint/journal identity mismatch')
        n, count = scalar['n'], tick + 1
        if (not isinstance(n, int) or n <= 0 or scalar['dt'] <= 0
                or not 0 <= scalar['_ring_i'] <= scalar['delay_steps']
                or scalar['t_ms'] != scalar['steps'] * scalar['dt']):
            raise ValueError('invalid neural clock or ring index')
        for key in ARRAYS:
            expected_shape = ((scalar['delay_steps'] + 1, n)
                              if key == '_ring' else (n,))
            if z['neural_' + key].shape != expected_shape:
                raise ValueError('invalid neural array shape: ' + key)
        for key in ('current', 'spikes', 'pool_activity', 'features',
                    'raw_command', 'command'):
            if z[key].shape[0] != count:
                raise ValueError('inconsistent accumulated prefix: ' + key)
        if z['current'].shape != (count, n) or z['spikes'].shape != (count, n):
            raise ValueError('invalid input/spike shape')
        for key in z.files:
            value = z[key]
            if value.dtype.kind in 'fci' and not np.isfinite(value).all():
                raise ValueError('nonfinite checkpoint: ' + key)
    gap = len(rows) - committed
    return dict(result, checkpoint_calls=committed,
                checkpoint_completed_tick=tick, uncommitted_reservations=gap,
                safe_to_resume=gap == 0,
                verdict=('STOP_UNCOMMITTED_RESERVATIONS' if gap else
                         'STATE_ALIGNED_REQUIRES_IDENTITY_AND_SINGLE_WRITER_CHECK'),
                missing_state_must_not_be_replayed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_recovery(args.directory), indent=2))


if __name__ == '__main__':
    main()
