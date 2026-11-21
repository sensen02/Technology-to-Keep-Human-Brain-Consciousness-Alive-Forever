"""Frozen Eon, a learned postsynaptic readout, and a one-command feeding body."""
import argparse
import gzip
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '20261118T195801Z-feeding-interoception'
ROOT = HERE.parents[2]
sys.path[:0] = [str(PREVIOUS), str(ROOT)]
from probe import (
    ARCHIVE, DT, DELAY, MN9, ContextBrain, FrozenCPUBrain, FeedingBody, FeedingState,
    Food, bind, eon, load_connectome, read, sha, sources, verify, verify_upstream, write,
)


CONTROL = 100  # 10 ms body and motor clock.
LOOKBACK = 900
MOTOR_LOOKBACK = 700
STRETCH_RATE_HZ = 80.
SUGAR_START = 250
RESTORE_TICK = 6200
DOSE_TICKS = 4300
CLOSED_TICKS = 12000
STOP_LEVEL = .63
RESTART_LEVEL = .53
DOSES = (0., .25, .55, .80)
MOUTH = PREVIOUS / 'MOUTH/TRACE.npz'


def network():
    verify_upstream()
    index, _ = eon.get_hash_tables(eon.path_comp)
    weights = load_connectome(index)
    group_map, group_audit, evidence = bind(index)
    group = group_map['set_4']
    stn = np.array([index[root] for root in group])
    sugar_config = eon.get_experiment('sugar')
    sugar = np.array([index[root] for root in sugar_config['neu_exc']])
    metadata = ARCHIVE / 'data/flywire/fafb_classification.csv.gz'
    with gzip.open(metadata, 'rt') as stream:
        annotations = {int(row['root_id']): row for row in csv.DictReader(stream)}
    roots = {i: int(root) for root, i in index.items()}
    signed = weights[:, stn].tocsr()
    magnitude = np.asarray(abs(signed).sum(axis=1)).ravel()
    choices = sorted(np.flatnonzero(magnitude), key=lambda i: (-magnitude[i], int(i)))
    central = [i for i in choices if annotations.get(roots[i], {}).get('super_class') == 'central'][:24]
    oral = [i for i in choices if annotations.get(roots[i], {}).get('class') == 'brain_motor_neuron'
            and annotations[roots[i]]['nerve'] == 'PhN' and magnitude[i] >= 5.]
    assert len(central) == 24 and oral
    feature_roots = np.array(central)
    motor = [index[MN9], index[720575940618238523]]
    assert not set(feature_roots) & (set(stn) | set(sugar) | set(motor))
    binding = dict(
        route='body StN / PhN / brain SEZ', set_4=group_audit['set_4'],
        feature_roots=[str(roots[i]) for i in central], bilateral_mn9=[str(roots[i]) for i in motor],
        oral_postsynaptic_motor=[
            dict(flywire_root=str(roots[i]), unsigned_eon_weight=float(magnitude[i]),
                 input_signed_weights=signed[i].toarray().ravel().tolist())
            for i in oral],
        molecular_piezo_identity='unresolved',
        delay_ms=DELAY * DT * 1000., delay_basis='engineering assumption',
        readout_uses_arriving_source_directly=False)
    return (weights, sugar, sugar_config['stim_rate'], stn, feature_roots, motor, oral, binding,
            [*evidence, MOUTH, eon.path_comp, eon.path_con, __file__,
             ROOT/'engine/embodied/feeding.py', PREVIOUS/'BANC_STN_AUDIT.json'])


def contact_at_start(ticks):
    with np.load(MOUTH, allow_pickle=False) as mouth:
        at_end = mouth['contact'][:ticks // CONTROL, 0]
    # The archived mouth point is at the droplet center at t=0. Consume each
    # measured sample only after its acquisition time.
    return np.repeat(np.r_[True, at_end[:-1]], CONTROL)


def window_counts(cumulative, tick, period):
    return cumulative[tick + 1] - cumulative[max(0, tick + 1 - period)]


@torch.inference_mode()
def run(output, seed, fit_path=None):
    torch.set_num_threads(1)
    (weights, sugar, sugar_rate, stn, features, mn9, oral, binding, files) = network()
    if fit_path is None:
        ticks, names, initial = DOSE_TICKS, ('empty', 'quarter', 'middle', 'high'), DOSES
        fit = None
    else:
        ticks = CLOSED_TICKS
        names = ('intact', 'cut', 'restored', 'low_fill')
        initial = (.54, .54, .54, .10)
        fit = read(fit_path)
        if fit['feature_roots'] != binding['feature_roots']:
            raise ValueError('readout root order differs')
        files.append(fit_path)
    batch = len(names)
    output.mkdir()
    pins = sources(files)
    binding['source_sha256'] = pins
    write(output / 'BINDING.json', binding)
    brain = ContextBrain(FrozenCPUBrain(batch, weights))
    rng = np.random.default_rng(seed)
    taste = rng.random((ticks, len(sugar))) < sugar_rate * DT
    contact = contact_at_start(ticks)
    assert len(contact) == ticks and contact.all()
    taste[:SUGAR_START] = False
    taste[~contact] = False
    stretch_random = rng.random((ticks, len(stn)))
    body = FeedingBody()
    states = [FeedingState(Food(2., 2.), crop=Food(level, 0.)) for level in initial]
    crop = np.array(initial)
    command = np.zeros(batch, dtype=bool)
    inhibited = np.zeros(batch, dtype=bool)
    arrivals = np.zeros((ticks, batch, len(stn)), dtype=bool)
    emissions = np.zeros_like(arrivals)
    spikes = np.zeros((ticks + 1, batch, len(features)), dtype=np.int32)
    motor = np.zeros((ticks + 1, batch), dtype=np.int32)
    oral_spikes = np.zeros((ticks, batch, len(oral)), dtype=bool)
    observed_mn9 = np.zeros((ticks, batch, len(mn9)), dtype=bool)
    population = np.zeros((ticks, batch), dtype=np.int32)
    history = []
    for tick in range(ticks):
        # Lesion acts at the receiver. Source emissions continue and cannot be
        # used by the readout. Restored traffic originates after reconnection.
        emissions[tick] = stretch_random[tick] < (crop[:, None] * STRETCH_RATE_HZ * DT)
        if tick >= DELAY:
            arrivals[tick] = emissions[tick - DELAY]
            if fit is not None:
                arrivals[tick, 1] = False
                if tick - DELAY < RESTORE_TICK:
                    arrivals[tick, 2] = False
        common = sugar[taste[tick]].tolist()
        brain.step([common + stn[events].tolist() for events in arrivals[tick]])
        activity = brain.spikes.numpy()
        population[tick] = activity.sum(axis=1)
        spikes[tick + 1] = spikes[tick] + activity[:, features].astype(np.int32)
        observed_mn9[tick] = activity[:, mn9] > 0
        motor[tick + 1] = motor[tick] + observed_mn9[tick].sum(axis=1)
        oral_spikes[tick] = activity[:, oral] > 0
        if (tick + 1) % CONTROL:
            continue
        counts = window_counts(spikes, tick, LOOKBACK)
        mn9_count = window_counts(motor, tick, MOTOR_LOOKBACK)
        if fit is None:
            prediction = np.zeros(batch)
        else:
            prediction = counts @ np.array(fit['coefficients']) + fit['intercept']
            inhibited = np.where(inhibited, prediction > RESTART_LEVEL, prediction >= STOP_LEVEL)
            command = (mn9_count >= 2) & ~inhibited
        was_crop = crop.copy()
        receipts = [body.advance(state, CONTROL * DT, feed=bool(command[i]),
                                 contact=bool(contact[tick])) for i, state in enumerate(states)]
        states = [receipt.state for receipt in receipts]
        crop = np.array([receipt.crop_stretch.value for receipt in receipts])
        history.append((counts, mn9_count, prediction, command.copy(), inhibited.copy(), was_crop,
                        [receipt.ingested.volume for receipt in receipts],
                        [receipt.can_ingest for receipt in receipts],
                        [state.patch.volume for state in states]))
        if (tick + 1) % 2000 == 0:
            print(seed, tick + 1, 'crop', np.round(crop, 3).tolist(),
                  'command', command.astype(int).tolist(), flush=True)
    if fit is not None:
        assert not arrivals[:, 1].any() and not arrivals[:RESTORE_TICK + DELAY, 2].any()
        assert np.array_equal(arrivals[RESTORE_TICK + DELAY:, 2], emissions[RESTORE_TICK:-DELAY, 2])
    trace = output / 'TRACE.npz'
    fields = ('features', 'mn9_count', 'predicted_crop', 'command', 'inhibited',
              'crop_before', 'ingested', 'could_ingest', 'food_remaining')
    np.savez_compressed(trace, **{k: np.asarray(v) for k, v in zip(fields, zip(*history), strict=True)},
                        stn_source=emissions, stn_arrived=arrivals, mn9_spikes=observed_mn9,
                        oral_spikes=oral_spikes, population=population, contact=contact, sugar=taste)
    verify(pins)
    write(output / 'RESULT.json', dict(seed=seed, names=names, initial_crop=initial, ticks=ticks,
          lane_ticks=batch * ticks, dt_s=DT, body_clock_ticks=CONTROL,
          restore_tick=RESTORE_TICK if fit is not None else None, fit=str(fit_path) if fit_path else None,
          mouth_source='prior independently replayed MuJoCo contact trace, causal replay',
          native_physics_steps=0, eon_weight_updates=0, trace_sha256=sha(trace)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('seed', type=int)
    parser.add_argument('--fit', type=Path)
    args = parser.parse_args()
    run(args.output, args.seed, args.fit)
