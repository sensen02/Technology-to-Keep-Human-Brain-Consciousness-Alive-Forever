"""Serial matched MuJoCo tests and independent audit of saved native traces."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from terrain_spec import CLOSED, HERE, REWARD

BRAIN_SHA = '71904e7a9765e5b0fb58afbe9c0ce27efa591f84997b51f0996f58a6eafbeaf3'
BASELINE = REWARD/'POLICY.npz'
RUN_ENV = HERE.parent/'20261119T012530Z-survival-orchard/run_env.sh'


def read(path):
    return json.loads(path.read_text())


def write(path, contents):
    path.with_suffix('.tmp').write_text(json.dumps(contents, indent=2, allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entered(trace, gate, i):
    inside = np.flatnonzero((trace[:, 1] >= gate['near']) & (np.arange(len(trace)) >= i))
    return int(inside[0]) if len(inside) else None


def crossing(trace, gate, start, elevation, fall):
    """Check a supported exit and lateral margin during the sampled crossing."""
    near, far, edge, side = (gate[k] for k in ('near', 'far', 'edge', 'side'))
    first = entered(trace, gate, start)
    fact = dict(entry_s=None, entry_margin_mm=None, crossing_s=None,
                minimum_cross_margin_mm=None, valid=False, exit=None)
    if first is None:
        return fact
    sample = trace[first]
    fact.update(entry_s=float(sample[0]), entry_margin_mm=float(10*(side*sample[2]-edge)))
    departures = np.flatnonzero((trace[1:, 1] >= far) & (trace[:-1, 1] < far)) + 1
    for i in (int(i) for i in departures if i >= first and i > start):
        if fall is not None and trace[i, 0] >= fall:
            break
        earlier = np.flatnonzero(trace[:i, 1] < near)
        if not len(earlier):
            continue
        last_before = int(earlier[-1])
        prev, at = trace[last_before:last_before+2]
        ratio = (near-prev[1])/(at[1]-prev[1])
        y_entry = prev[2]+ratio*(at[2]-prev[2])
        prev, at = trace[i-1:i+1]
        ratio = (far-prev[1])/(at[1]-prev[1])
        y_exit = prev[2]+ratio*(at[2]-prev[2])
        center = np.r_[y_entry, trace[last_before+1:i, 2], y_exit]
        margin = float((side*center-edge).min())
        window = trace[i:min(i+4,len(trace))]
        valid = bool(margin >= 0. and np.any(
            (window[:, 4] >= 3) & (window[:, 6] > .85) &
            (window[:, 3] - elevation(window[:, 1]) > .07)))
        if fact['crossing_s'] is None or valid:
            fact.update(crossing_s=float(at[0]), minimum_cross_margin_mm=10*margin,
                        valid=valid, exit=i)
        if valid:
            break
    return fact


def inspect(folder, plan, case):
    record = read(folder/'RESULT.json')
    for key in ('seed', 'challenge', 'side'):
        assert record[key] == case[key]
    assert record['neural_seed'] == plan['neural_seed'] and record['native_extra_force'] == 0.
    assert record['max_duration_s'] == case['duration']
    # The original plan's .1 ms field is the Eon/BANC sampling interval.
    # FlyBody MuJoCo integrates twice per neural sample.
    neck_samples = round(record['duration_s']/plan['native_step_seconds'])
    assert record['native_steps'] == 2*neck_samples
    with np.load(folder/'NECK.npz', allow_pickle=False) as spikes:
        emitted, arrived, delays = (spikes[k] for k in ('emitted', 'arrived', 'delays'))
        assert len(delays) == 17 and emitted.shape == arrived.shape == (neck_samples, 17)
        assert spikes['weights'].item() == record['model_sha'] == BRAIN_SHA
        assert all((not arrived[:d, i].any() and np.array_equal(emitted[:-d, i], arrived[d:, i]))
                   if d else np.array_equal(emitted[:, i], arrived[:, i])
                   for i, d in enumerate(delays))
        ascending, descending = int(spikes['ascending'].sum()), int(arrived.sum())
    with np.load(folder/'BODY.npz', allow_pickle=False) as body:
        poses, trace, meal, obs, commands = (body[k] for k in
                                            ('poses', 'trace', 'meal', 'observation', 'commands'))
        assert all(np.isfinite(a).all() for a in (poses,trace,meal,obs,commands))
        assert trace.shape[0] == poses.shape[0] and np.all(np.diff(trace[:, 0]) > 0)
        assert abs(meal[:, 4].sum()-record['stock_loss']) < 1e-10
        assert abs(meal[-1, 1]-record['intake']) < 1e-10
        assert np.allclose(np.cumsum(meal[:, 4]), meal[:, 1], rtol=0., atol=1e-10)
    tail = trace[trace[:, 0] >= record['duration_s']-plan['standing_window_seconds']]
    speed = np.linalg.norm(tail[:, 7:9], axis=1)
    assert len(tail) >= 27
    stood = bool(np.all((tail[:, 4] >= 4) & (tail[:, 6] > .94) & (speed <= .08)))
    ate = bool(meal[-1, 1] >= plan['minimum_meal_volume'])
    assert stand_and_meal_match(record, stood, ate)
    geometry = record['geometry']
    gates = geometry.get('gates')
    if gates is None:
        gates = [geometry] if geometry['side'] else []
    profile = np.asarray(geometry.get('height_profile', ((-2.,0.), (5.,0.))))
    limits = []
    index = 0
    for gate in gates:
        fact = crossing(trace, gate, index, lambda x: np.interp(x,profile[:,0],profile[:,1]),
                        record['first_fall_s'])
        index = fact['exit']+1 if fact['valid'] else len(trace)
        limits.append(fact)
    return record | dict(
        gates=limits, all_supported=all(item['valid'] for item in limits),
        validated_meal=record['meal_pass'] and all(item['valid'] for item in limits),
        ascending_events=ascending, descending_events=descending, delays_ms=(delays*.1).tolist(),
        tail_speed_mm_s=float(10*speed.max()), tail_feet=int(tail[:,4].min()),
        visual_hazard_peak=float(obs[:,5:11].max()))


def stand_and_meal_match(record, stood, ate):
    return (record['ended_standing'] == stood and record['intake_target_pass'] == ate
            and record['meal_pass'] == (ate and stood and record['first_fall_s'] is None))


def summary(rows):
    groups = defaultdict(list)
    for r in rows.values():
        groups[r['challenge']].append(r)
    return {key: dict(n=len(cases), meals=sum(r['validated_meal'] for r in cases),
                      fell=sum(r['first_fall_s'] is not None for r in cases),
                      crossed=sum(r['all_supported'] for r in cases),
                      ate=sum(r['intake_target_pass'] for r in cases),
                      intake=sum(r['intake'] for r in cases),
                      native_steps=sum(r['native_steps'] for r in cases))
            for key,cases in groups.items()}


def run(root, policy):
    plan = read(HERE/'physical_plan.json')
    root.mkdir(exist_ok=True)
    digest = hash_file(policy)
    identity = root/'IDENTITY.json'
    if not identity.exists():
        write(identity, dict(policy=str(policy), sha256=digest, plan_sha256=hash_file(HERE/'physical_plan.json')))
    assert read(identity)['sha256'] == digest
    assert read(identity)['plan_sha256'] == hash_file(HERE/'physical_plan.json')
    rows = {}
    for case in plan['cases']:
        name = case['name']
        folder = root/name
        if not (folder/'RESULT.json').is_file():
            given = [str(RUN_ENV), str(HERE/'live_compound.py'), str(folder), '--policy', str(policy),
                     '--challenge', case['challenge'], '--seed', str(case['seed']),
                     '--side', str(case['side']), '--duration', str(case['duration'])]
            print('start', name, flush=True)
            with (root/f'{name}.log').open('x') as trace:
                subprocess.run(given, stdout=trace, stderr=subprocess.STDOUT, check=True)
        rows[name] = inspect(folder, plan, case)
        write(root/'AUDIT.json', dict(n=len(rows), expected=len(plan['cases']), by_stage=summary(rows), cases=rows))
        latest = rows[name]
        print(name, 'fall', latest['first_fall_s'], 'supported',
              latest['all_supported'], 'full meal', latest['validated_meal'],
              'intake', round(latest['intake'],4), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--policy', type=Path, default=BASELINE)
    arguments = parser.parse_args()
    run(arguments.root, arguments.policy)
