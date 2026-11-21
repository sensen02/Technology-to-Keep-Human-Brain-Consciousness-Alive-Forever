"""Resumeable motion training on the expanding physical corpus, with fixed evaluation sets."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '20261121T024505Z-mild-terrain-practice'
sys.path.insert(0, str(PRIOR))
from train_practice import PARENT, ObservedBody, corpus, digest, flat, floor_model, tensor, write

torch.set_num_threads(1)
SCHEDULE = ((96, 900), (240, 2800), (520, 6000), (900, 10500),
            (1400, 15300), (2000, 20800), (2800, 28800))
RUN = HERE / 'run'
CHECKPOINT = RUN / 'resume.pt'
HORIZON = 14
BASE_SEED = 51603


def trials(first, last):
    return [HERE / 'episodes' / f'terrain{j:02d}' for j in range(first, last)]


def observations(directory, bank, split):
    receipt = json.loads((directory / 'receipt.json').read_text())
    if receipt['split'] != split:
        raise ValueError(f'Unexpected split for {directory}')
    key = f"{receipt['seed']}_{receipt['mode']}"
    with np.load(directory / 'external.npz') as archive:
        pos = tensor(archive['markers_mm'])
        force = tensor(archive['force_uN'])
    touch = torch.stack((torch.cat((force[:1], force[:-1])), force), dim=1)
    return directory.name, pos, touch, bank[key][3]


def ready(required):
    while not all((trial / 'receipt.json').is_file() for trial in required):
        fail = HERE / 'collection.exit'
        if fail.is_file() and fail.read_text().strip() != '0':
            raise RuntimeError('Physical collection failed. See logs/terrain-*.log.')
        time.sleep(5)


def atomic_checkpoint(content, path):
    temp = path.with_suffix('.tmp')
    torch.save(content, temp)
    temp.replace(path)


def fit():
    RUN.mkdir(exist_ok=True)
    torch.manual_seed(BASE_SEED)
    net = ObservedBody(PARENT / 'weights.pt')
    net.raw.feedback.requires_grad_(False)
    floor = floor_model()
    opt = torch.optim.AdamW(net.raw.motion.parameters(), lr=.00016, weight_decay=.0005)
    base, rough, _ = corpus('train')
    bank = {r[0]: r for r in base}
    entries = base + rough
    pos = torch.empty((2800, 400, 21, 3), dtype=torch.float32)
    touch = torch.empty((2800, 400, 2, 6), dtype=torch.float32)
    command_ids = torch.empty(2800, dtype=torch.long)
    key_to_command = {k: i for i, k in enumerate(bank)}
    command_bank = torch.stack([r[3] for r in base])

    def load_entry(i, row, key):
        pos[i].copy_(row[1])
        touch[i].copy_(row[2])
        command_ids[i] = key_to_command[key]

    for i, row in enumerate(entries):
        if i < len(base):
            key = row[0]
        else:
            info = json.loads((PRIOR / 'episodes' / row[0] / 'receipt.json').read_text())
            key = f"{info['seed']}_{info['mode']}"
        load_entry(i, row, key)
    loaded = len(entries)
    assert loaded == 28 and len(bank) == 12
    step = 0
    start_loss = 0.
    count = 0
    reports = []
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
        net.raw.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['rng'])
        step = state['step']
        reports = state['reports']
        print('resumed at', step, flush=True)
    offsets = torch.arange(HORIZON + 2)
    for total, target in SCHEDULE:
        if step >= target:
            continue
        required = trials(20, 20 + (total - 28))
        print('waiting for', total, 'training episodes at update', step, flush=True)
        ready(required)
        for i, dest in enumerate(required[loaded - 28:], start=loaded):
            row = observations(dest, bank, 'train')
            info = json.loads((dest / 'receipt.json').read_text())
            load_entry(i, row, f"{info['seed']}_{info['mode']}")
        loaded = total
        print('training on', loaded, 'episodes through update', target, flush=True)
        while step < target:
            opt.param_groups[0]['lr'] = .000016 + .000144 * (1 + math.cos(math.pi*step/28800))*.5
            lanes = torch.randint(12, loaded, (64,))
            lanes[:8] = torch.randint(12, (8,))
            times = torch.randint(1, 400 - HORIZON - 1, (64, 1)) + offsets
            p = pos[lanes[:, None], times]
            f = touch[lanes[:, None], times]
            dn = command_bank[command_ids[lanes, None], times]
            prev, now = p[:, 0], p[:, 1]
            last = floor(prev, prev)
            use_est = torch.rand((len(lanes), 1, 1)) < .5
            loss = 0.
            for k in range(1, HORIZON + 1):
                est = floor(prev, now)
                loads = torch.where(use_est, torch.stack((last, est), dim=1), f[:, k])
                estimate, _ = net(prev, now, loads, dn[:, k])
                loss += torch.nn.functional.smooth_l1_loss((estimate-p[:, k+1])/.035,
                                                           torch.zeros_like(estimate))
                last = est
                prev, now = now, estimate
            loss = loss / HORIZON
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.raw.motion.parameters(), 2.)
            opt.step()
            start_loss += loss.item()
            count += 1
            step += 1
            if step % 300 == 0 or step == target:
                avg = start_loss / count
                reports.append(dict(step=step, training_episodes=loaded, mean_loss=avg))
                atomic_checkpoint(dict(step=step, model=net.raw.state_dict(), optimizer=opt.state_dict(),
                                       rng=torch.random.get_rng_state(), reports=reports), CHECKPOINT)
                write(RUN / 'progress.json', reports[-1])
                print('update', step, 'episodes', loaded, 'loss', round(avg, 5), flush=True)
                start_loss = 0.
                count = 0
    atomic_checkpoint(net.raw.state_dict(), RUN / 'weights.pt')
    (RUN / 'skeleton.json').write_bytes((PARENT / 'skeleton.json').read_bytes())
    write(RUN / 'training.json', dict(physical_episodes=2800, updates=step,
          same_prior_physics=28, training_schedule=SCHEDULE, checkpoints=reports,
          weights_sha256=digest(RUN / 'weights.pt'), parent_sha256=digest(PARENT / 'weights.pt')))


@torch.inference_mode()
def assess(candidate=RUN):
    plane, previous_terrains, _ = corpus('test')
    test = trials(3000, 3020)
    ready(test)
    bank = {r[0]: r for r in plane}
    terrain = [observations(d, bank, 'test') for d in test]
    rows = plane + previous_terrains + terrain
    pos, pressure, dn = [torch.stack([r[i] for r in rows]) for i in (1, 2, 3)]
    floor = floor_model()
    record = []
    for i, (label, _, _, _) in enumerate(rows):
        record.append({'trial': label, 'split': 'new_terrain' if i >= 11 else 'old_terrain' if i >= 7 else 'flat',
                       'teacher_net_mm': float((pos[i,-1,0,:2] - pos[i,0,0,:2]).norm())})
    for name, source in [('baseline', PARENT), ('prior_practice', PRIOR / 'expanded'), ('practice100', candidate)]:
        m = ObservedBody(source / 'weights.pt').eval()
        for horizon in (1, 10, 40):
            starts = torch.arange(1, 400-horizon, horizon)
            for lo in range(0, len(rows), 6):
                sl = slice(lo, lo+6)
                p, f, d = pos[sl], pressure[sl], dn[sl]
                a, b = p[:,starts-1].flatten(0,1), p[:,starts].flatten(0,1)
                for j in range(horizon):
                    step, _ = m(a, b, f[:,starts+j].flatten(0,1), d[:,starts+j].flatten(0,1))
                    a, b = b, step
                error = (b.reshape_as(p[:, starts+horizon])-p[:,starts+horizon]).norm(dim=-1).mean((1,2))
                for i, err in enumerate(error, start=lo):
                    record[i].setdefault(name, {})[f'error_{5*horizon}ms_mm'] = float(err)
        for mode in ('observed','estimated'):
            seq = [pos[:,0], pos[:,1]]
            last = floor(seq[0], seq[0])
            for t in range(1,399):
                est = floor(*seq[-2:])
                force = pressure[:,t] if mode=='observed' else torch.stack((last,est),dim=1)
                step, _ = m(*seq[-2:], force, dn[:,t])
                seq.append(step)
                last = est
            path = torch.stack(seq, dim=1)[:,:,0]
            assert torch.isfinite(path).all()
            for i, p in enumerate(path):
                row = record[i][name]
                row[f'{mode}_net_mm'] = float((p[-1,:2]-p[0,:2]).norm())
                row[f'{mode}_endpoint_mm'] = float((p[-1]-pos[i,-1,0]).norm())
                row[f'{mode}_tail_rms_mm_s'] = float((p[-41:,:2].diff(dim=0).square().sum(-1).mean()).sqrt()/.005)
    write(candidate / 'evaluation.json', record)
    grouped = {}
    for split in ('flat','old_terrain','new_terrain'):
        members = [row for row in record if row['split']==split]
        grouped[split] = dict(n=len(members), teacher_net_mm=float(np.mean([r['teacher_net_mm'] for r in members])))
        for name in ('baseline','prior_practice','practice100'):
            grouped[split][name] = {k: float(np.mean([r[name][k] for r in members])) for k in members[0][name]}
    write(candidate / 'summary.json', grouped)
    print(json.dumps(grouped, indent=2), flush=True)


if __name__ == '__main__':
    if '--assess-only' not in sys.argv:
        fit()
    assess()
