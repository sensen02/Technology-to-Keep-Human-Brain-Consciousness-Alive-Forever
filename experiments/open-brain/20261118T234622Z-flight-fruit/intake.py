"""Replay real mouth reach into frozen Eon and integrate measured food transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OPEN_FRUIT = HERE.parent / '20261118T231627Z-open-fruit-loop'
sys.path.insert(0, str(OPEN_FRUIT))
sys.path.insert(0, str(HERE.parents[2]))
from fastbrain import FastBrain, eon, load_connectome
from engine.embodied.feeding import FeedingBody, FeedingState, Food

STEP = .0001
MN9 = (720575940660219265, 720575940618238523)
RATE_HZ = 200.
WINDOW = 700
TASTE_TICKS = 50


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def run(body_path, out, seed, blocked):
    torch.set_num_threads(1)
    receipt = json.loads((body_path / 'RESULT.json').read_text())
    assert sha(body_path / 'TRACE.npy') == receipt['files']['TRACE.npy']
    parity = json.loads((OPEN_FRUIT / 'EON_PARITY.json').read_text())
    assert parity['status'] == 'BIT_EXACT_STOCK_EON_PARITY'
    raw = np.load(body_path / 'TRACE.npy', allow_pickle=False)
    tick_stride = round(STEP / receipt['native_dt_s'])
    tick_count = len(raw) // tick_stride
    samples = raw[tick_stride - 1:tick_count * tick_stride:tick_stride]
    contact = samples[:, receipt['columns'].index('taste_contact')].astype(bool)
    anatomy, _ = eon.get_hash_tables(eon.path_comp)
    tasters = np.array([anatomy[root] for root in eon.get_experiment('sugar')['neu_exc']])
    motor = [anatomy[root] for root in MN9]
    brain = FastBrain(load_connectome(anatomy).T.tocsr())
    rng = np.random.default_rng(seed)
    masks = rng.random((tick_count, len(tasters)), dtype=np.float32) < RATE_HZ * STEP
    feed_body = FeedingBody()
    initial = Food(1.8, 1.62)
    state = FeedingState(initial)
    output = np.zeros((tick_count, 10))
    delivered = absorbed = excreted = total_use = 0.
    window_count = 0
    taste = False
    for tick, touching in enumerate(contact):
        if tick % TASTE_TICKS == 0:
            taste = touching and not blocked and state.patch.volume > 0.
        brain.step(tasters[masks[tick] & taste])
        pulses = int(brain.spikes[motor].sum())
        window_count += pulses
        if tick >= WINDOW:
            window_count -= int(output[tick - WINDOW, 3])
        feed = window_count >= 2
        h = feed_body.advance(state, STEP, feed=feed, contact=bool(touching), blocked=blocked)
        state = h.state
        delivered += h.ingested.volume
        absorbed += h.absorbed
        excreted += h.excreted.nutrient
        total_use += h.metabolized
        output[tick] = (samples[tick, 0], touching, taste, pulses, feed, delivered,
                        state.patch.volume, state.crop.volume, absorbed, state.reserve)
    nutrient_accounting = (state.patch.nutrient + state.crop.nutrient + state.midgut.nutrient
                           + absorbed + excreted)
    assert abs(nutrient_accounting - initial.nutrient) < 1e-9
    assert abs(state.reserve - (.2 + absorbed - total_use)) < 1e-9
    assert abs(initial.volume - state.patch.volume - delivered) < 1e-9
    assert not np.any(np.diff(output[:, 5], prepend=0.)[~contact] > 0.)
    hits = output[output[:, 5] > 0, 0]
    out.mkdir()
    np.save(out / 'MEAL.npy', output, allow_pickle=False)
    facts = dict(body_path=str(body_path.resolve()), trace_sha256=receipt['files']['TRACE.npy'],
                 meal_sha256=sha(out / 'MEAL.npy'), blocked=blocked, seed=seed,
                 columns=['time_s', 'physical_mouth_contact', 'sugar_signal', 'mn9_spikes',
                          'feed', 'cumulative_intake', 'on_fruit', 'crop', 'absorbed_nutrient',
                          'reserve'], taste_mn9_roots=MN9, mn9_spikes=int(output[:, 3].sum()),
                 first_intake_s=float(hits[0]) if len(hits) else None,
                 intake=delivered, crop_volume=state.crop.volume, absorbed_nutrient=absorbed,
                 fruit_remaining=state.patch.volume, nutrient_balance_error=nutrient_accounting - initial.nutrient)
    (out / 'RESULT.json').write_text(json.dumps(facts, indent=2, allow_nan=False) + '\n')
    print(facts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('body', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=483)
    parser.add_argument('--blocked', action='store_true')
    args = parser.parse_args()
    run(args.body, args.output, args.seed, args.blocked)
