"""Independently verify a SAC snapshot, export parity and fixed holdout scores."""
import argparse
from copy import deepcopy
import json

import numpy as np
import torch

from train_timed import HERE, START, HELDOUT, assess, digest, learn, MirrorActor, new_env, sha
from policy_runtime import actor_mean


def record(path):
    return json.loads(path.read_text())


def check(root):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    learn.Actor = MirrorActor
    result = record(root/'FINAL.json')
    result_rows = [result['initial']] + [json.loads(line) for line in (root/'CHECKPOINTS.jsonl').read_text().splitlines()]
    for left, right in zip(result_rows, result_rows[1:]):
        assert left['step'] < right['step'] and left['elapsed'] < right['elapsed']
        assert left['updates'] < right['updates']
    selected = result['selected']
    steps = {row['step'] for row in result_rows}
    assert len(steps) == len(result_rows) and result['terminal'] == result_rows[-1]
    assert selected['step'] in steps
    from pathlib import Path
    assert all(sha(Path(row['checkpoint'])) == row['sha256'] for row in result_rows)
    initial = torch.load(START, map_location='cpu', weights_only=True)
    checkpoint = torch.load(selected['checkpoint'], map_location='cpu', weights_only=True)
    updated = torch.load(result['terminal']['checkpoint'], map_location='cpu', weights_only=True)
    selected_step = selected['step']
    if selected_step > 0:
        assert checkpoint['step'] == selected_step and checkpoint['updates'] == selected['updates']
    assert updated['step'] == result['terminal']['step'] and updated['updates'] == result['terminal']['updates']
    assert bool(updated['actor_optimizer']['state']) and bool(updated['critic_optimizer']['state'])
    assert bool(updated['alpha_optimizer']['state'])
    assert any(not torch.equal(updated['actor'][key], value) for key,value in initial['actor'].items())
    assert checkpoint['identity']['brain_sha256'] == initial['identity']['brain_sha256']
    assert result['elapsed'] >= record(root/'PLAN.json')['requested_seconds']
    agent = learn.SAC(51661)
    agent.actor.load_state_dict(checkpoint['actor'])
    rng = np.random.default_rng(53347)
    sense = rng.random((1536,14), dtype=np.float32)
    angles = rng.uniform(-1.6,1.6,len(sense))
    sense[:,1],sense[:,2] = np.sin(angles),np.cos(angles)
    sense[:,0] = (sense[:,0] > .25)
    with np.load(root/'POLICY.npz',allow_pickle=False) as archive, torch.no_grad():
        assert json.loads(archive['identity'].item())['brain_sha256'] == checkpoint['identity']['brain_sha256']
        tensor = agent.actor(torch.from_numpy(sense),sample=False)[0].numpy()
        numpy = actor_mean(archive,sense)
        error = float(np.abs(tensor-numpy).max())
        # Stress inputs include combinations never emitted by the live sensor.
        # Confirm exact parameters and float64 parity before bounding float32 error.
        for layer in (0,2,4):
            linear = agent.actor.net[layer]
            assert np.array_equal(linear.weight.detach().numpy(), archive[f'w{layer}'])
            assert np.array_equal(linear.bias.detach().numpy(), archive[f'b{layer}'])
        precise = {key:(archive[key].astype(np.float64) if key.startswith(('w','b')) else archive[key])
                   for key in archive.files}
        widened = sense.astype(np.float64)
        double = deepcopy(agent.actor).double()(torch.from_numpy(widened),sample=False)[0].numpy()
        precise_gap = float(np.abs(double-actor_mean(precise,widened)).max())
        assert precise_gap < 1e-10 and error < 2e-5
        physical_paths = sorted(HERE.glob('LIVE_BASELINE/*/BODY.npz')) + sorted(HERE.glob('VISIBLE_BASELINE/*/BODY.npz'))
        assert len(physical_paths) == 7
        live = []
        for path in physical_paths:
            with np.load(path,allow_pickle=False) as physical:
                live.append(physical['observation'][:,1:15].astype(np.float32))
        actual_inputs = np.concatenate(live)
        th = agent.actor(torch.from_numpy(actual_inputs),sample=False)[0].numpy()
        live_gap = float(np.abs(th-actor_mean(archive,actual_inputs)).max())
        assert live_gap < 5e-6
        flipped = sense[:,archive['mirror']].copy()
        flipped[:,1] *= -1.
        mirror_gap = float(np.abs(actor_mean(archive,flipped)-numpy*[1.,-1.]).max())
        assert mirror_gap < 5e-6
    independent = digest(assess(agent,new_env(),HELDOUT))
    assert independent == result['heldout_after']
    payload = dict(identity=checkpoint['identity'], selected_step=selected_step,
                   checkpoint_sha256=selected['sha256'], export_sha256=sha(root/'POLICY.npz'),
                   checkpoints=len(result_rows)-1, independent_holdout=independent,
                   max_float32_stress_error=error, max_float64_error=precise_gap,
                   live_input_export_error=live_gap, live_input_count=len(actual_inputs),
                   mirror_error=mirror_gap)
    (root/'RELOAD_AUDIT.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(payload,allow_nan=False))


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, default=HERE/'BUDGET_1800', nargs='?')
    check(parser.parse_args().root)
