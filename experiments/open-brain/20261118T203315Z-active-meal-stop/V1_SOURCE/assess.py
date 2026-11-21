"""Recompute the route and active-stop criteria from the saved live traces."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261118T195801Z-feeding-interoception'))
from meal import CONTROL, DELAY, DT, RESTART_LEVEL, RESTORE_TICK, STOP_LEVEL, FeedingBody, FeedingState, Food


def read(path):
    return json.loads(path.read_text())


def first_active_stop(command, inhibited, gate, room, cumulative, start=0):
    """Find a motor-drive-preserving, voluntary off window with accessible food."""
    candidates = np.flatnonzero(command[:-1] & ~command[1:]) + 1
    for tick in candidates:
        if tick < start or cumulative[tick] < .04 or tick + 10 > len(command):
            continue
        window = slice(tick, tick + 10)
        if (not command[window].any() and inhibited[window].all()
                and gate[window].mean() >= .5 and room[window].all()):
            return int(tick)
    return None


def assess(folder):
    metadata = read(folder / 'RESULT.json')
    fit = read(Path(metadata['fit']))
    with np.load(folder / 'TRACE.npz', allow_pickle=False) as file:
        trace = {k: file[k] for k in file.files}
    names = metadata['names']
    assert names == ['intact', 'cut', 'restored', 'low_fill']
    restored_at = RESTORE_TICK + DELAY
    expected = np.zeros_like(trace['stn_arrived'])
    expected[DELAY:, (0, 3)] = trace['stn_source'][:-DELAY, (0, 3)]
    expected[restored_at:, 2] = trace['stn_source'][RESTORE_TICK:-DELAY, 2]
    assert np.array_equal(trace['stn_arrived'], expected)
    # Until the first restored input the cut and restored brains/bodies share
    # identical histories. The same is true until the first intact arrival.
    for lane, first in ((0, np.flatnonzero(expected[:, 0].any(axis=1))[0]), (2, restored_at)):
        for key in ('mn9_spikes', 'oral_spikes', 'population', 'stn_source'):
            assert np.array_equal(trace[key][:first, lane], trace[key][:first, 1])
        for key in ('crop_before', 'command', 'ingested'):
            end = first // CONTROL
            assert np.array_equal(trace[key][:end, lane], trace[key][:end, 1])
    scores = trace['features'] @ np.array(fit['coefficients']) + fit['intercept']
    assert np.allclose(trace['predicted_crop'], scores, rtol=0., atol=1e-12)
    inhibited = np.zeros(len(names), dtype=bool)
    states = [FeedingState(Food(2., 2.), crop=Food(level, 0.)) for level in metadata['initial_crop']]
    body = FeedingBody()
    for i, score in enumerate(scores):
        inhibited = np.where(inhibited, score > RESTART_LEVEL, score >= STOP_LEVEL)
        assert np.array_equal(trace['inhibited'][i], inhibited)
        step = CONTROL * DT
        for lane, state in enumerate(states):
            end = (i + 1) * CONTROL
            motor = trace['mn9_spikes'][max(0, end - 700):end, lane].sum()
            feed = not inhibited[lane] and motor >= 2
            assert motor == trace['mn9_count'][i, lane] and feed == trace['command'][i, lane]
            assert abs(state.crop.volume - trace['crop_before'][i, lane]) < 1e-12
            receipt = body.advance(state, step, feed=bool(feed), contact=bool(trace['contact'][end-1]))
            assert receipt.ingested.volume == trace['ingested'][i, lane]
            assert receipt.can_ingest == trace['could_ingest'][i, lane]
            assert receipt.state.patch.volume == trace['food_remaining'][i, lane]
            states[lane] = receipt.state
    metrics = {}
    step = CONTROL * DT
    for j, name in enumerate(names):
        command = trace['command'][:, j]
        inhibited = trace['inhibited'][:, j]
        motor = trace['mn9_count'][:, j] >= 2
        room = trace['could_ingest'][:, j]
        amounts = trace['ingested'][:, j]
        cumulative = np.r_[0., np.cumsum(amounts)]
        start = (restored_at + CONTROL - 1) // CONTROL - 1 if j == 2 else 0
        stop = first_active_stop(command, inhibited, motor, room, cumulative, start)
        switches = np.count_nonzero(command[1:] != command[:-1])
        metrics[name] = dict(
            total_ingested=float(cumulative[-1]), switches=int(switches),
            stop_ms=(stop + 1) * step * 1000 if stop is not None else None,
            crop_at_stop=float(trace['crop_before'][stop, j]) if stop is not None else None,
            food_remaining_at_stop=float(trace['food_remaining'][stop, j]) if stop is not None else None,
            end_command_fraction=float(np.mean(command[-20:] & room[-20:])),
            min_prediction=float(scores[:, j].min()), max_prediction=float(scores[:, j].max()),
            mean_prediction_mid=float(np.mean(scores[25:60, j])),
        )
    cut = metrics['cut']
    gates = dict(
        intact_stop=metrics['intact']['stop_ms'] is not None,
        restore_stop=metrics['restored']['stop_ms'] is not None,
        cut_continues=cut['stop_ms'] is None and cut['end_command_fraction'] >= .8,
        low_continues=(metrics['low_fill']['stop_ms'] is None
                       and metrics['low_fill']['end_command_fraction'] >= .8),
        reduced_intake=all(cut['total_ingested'] - metrics[name]['total_ingested'] >= .02
                           for name in ('intact', 'restored')),
    )
    return dict(seed=metadata['seed'], result=metrics, gates=gates, passed=all(gates.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('trials', type=Path, nargs='+')
    args = parser.parse_args()
    results = [assess(trial) for trial in args.trials]
    with args.output.open('x') as stream:
        json.dump(dict(passed=all(r['passed'] for r in results), trials=results), stream, indent=2)
        stream.write('\n')
    print(json.dumps(results, ensure_ascii=False))
