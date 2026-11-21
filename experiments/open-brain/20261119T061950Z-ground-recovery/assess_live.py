"""Verify the matched fruit-edge and continuous recovery trials."""
import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / '20261119T052620Z-hunger-foraging/VERIFIED/FADE'


def read(folder):
    return json.loads((folder / 'RESULT.json').read_text()), np.load(folder / 'TRACE.npy')


def neck(folder):
    with np.load(folder / 'GROUND_DN.npz') as data:
        emitted, arrived = data['emitted'], data['arrived']
        delays = data['delay_ticks']
        exact = all(not arrived[:d, i].any()
                    and np.array_equal(emitted[:-d, i], arrived[d:, i])
                    for i, d in enumerate(delays))
        return dict(aligned=exact, channels=len(delays), emitted=int(emitted.sum()),
                    arrived=int(arrived.sum()), delay_ms=(delays / 10.).tolist())


def assess_version(baseline, folder, control_is_bypassed):
    result, trace = read(folder)
    onset = next((float(t[0]) for t in trace if t[6]), None)
    guard = np.load(folder / 'EDGE_GUARD.npy')
    recovery_log = folder / 'RECOVERY_EVENTS.json'
    events = json.loads(recovery_log.read_text()) if recovery_log.is_file() else []
    first_limit = next((float(t[0]) for t in guard if t[7] < t[6]), None)
    baseline_fall = float(baseline[baseline[:, 6] > 0][0, 0])
    cutoff = baseline_fall if control_is_bypassed else first_limit
    prefix = len(baseline) if cutoff is None else np.searchsorted(baseline[:, 0], cutoff, side='left')
    match = bool(np.array_equal(baseline[:prefix], trace[:prefix]))
    inputs = np.array_equal(np.load(BASELINE / 'INITIAL.npy'), np.load(folder / 'INITIAL.npy'))
    timing = neck(folder)
    inclusive = trace[:, 0] >= trace[-1, 0] - .20
    tail = trace[inclusive]
    speed = float(10 * np.linalg.norm(np.diff(tail[:, 1:3], axis=0), axis=1).max() /
                  np.diff(tail[:, 0]).min())
    stood = [r['time_s'] for r in events if r['event'] == 'stood']
    post = {}
    if stood:
        later = trace[trace[:, 0] >= stood[0]]
        post = dict(since_first_stood_s=float(later[-1, 0] - later[0, 0]),
                    renewed_nonfoot_samples=int(np.count_nonzero(later[:, 6])),
                    max_travel_mm=float(10 * np.linalg.norm(later[:, 1:3]-later[0, 1:3], axis=1).max()),
                    end_travel_mm=float(10 * np.linalg.norm(later[-1, 1:3]-later[0, 1:3])))
    timing_ok = inputs and match and timing['aligned'] and result['max_external'] == 0.
    did_stand = onset == baseline_fall and bool(stood) and stood[0] > onset
    resumed = bool(post and post['renewed_nonfoot_samples'] == 0
                   and post['end_travel_mm'] >= 1. and post['since_first_stood_s'] >= .50)
    prevention = onset is None and result['intake']['ground_far'] >= .01 and not events
    passed = timing_ok and (did_stand if control_is_bypassed else prevention)
    return dict(passed=bool(passed), exact_initial=bool(inputs), exact_prefix=match,
                prefix_native_steps=int(prefix), recovery_probe=control_is_bypassed,
                first_guard_s=first_limit, first_nonfoot_s=onset, first_stood_s=stood[0] if stood else None,
                neck=timing, last_200ms_peak_linear_speed_mm_s=speed, post_stand=post,
                resume_without_fall=resumed,
                events=events, result=result)


def main(args):
    reference, baseline = read(BASELINE)
    audit = {'baseline': {'result': reference,
                          'first_nonfoot_s': float(baseline[baseline[:, 6] > 0][0, 0])}}
    if args.continuous:
        audit['continuous'] = assess_version(baseline, HERE / args.continuous, True)
    if args.guarded:
        audit['guarded'] = assess_version(baseline, HERE / args.guarded, False)
    (HERE / args.output).write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    return all(v['passed'] for k, v in audit.items() if k != 'baseline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guarded')
    parser.add_argument('--continuous')
    parser.add_argument('--output', default='LIVE_AUDIT.json')
    raise SystemExit(0 if main(parser.parse_args()) else 1)
