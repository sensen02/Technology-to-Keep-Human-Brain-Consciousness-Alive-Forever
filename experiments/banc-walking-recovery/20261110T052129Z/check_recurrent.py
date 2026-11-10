"""Synthetic full-episode recurrent PPO correctness checks; no physical training."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import random
import tempfile
import numpy as np
import torch
from models import ActorCritic, latent_log_prob, gaussian_kl
from ppo import gae, save_checkpoint, load_checkpoint
from recurrent import RecurrentPPOConfig, RecurrentPPOTrainer, recurrent_loss, replay_mean


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def model_new():
    return ActorCritic('gru238', critic_obs_dim=255)


@torch.no_grad()
def episode_new(model, n, truncated=False):
    # Collect through the exact public inference interface, one observation at a
    # time; compare this independent collection path with sequence replay later.
    obs = torch.randn(n, 238, dtype=torch.float64) * .2
    cv = torch.randn(n, 255, dtype=torch.float64) * .2
    state = model.actor.initial_state(1)
    collected = []
    for t in range(n):
        output = model.act(obs[t], cv[t], state=state)
        state = output['state']
        collected.append(output)
    e = dict(obs=obs, critic_obs=cv, starts_at_reset=True)
    for key in ('latent', 'mean', 'logstd'):
        e[key] = torch.cat([row[key] for row in collected])
    e['log_prob'] = torch.cat([row['log_prob'] for row in collected])
    e['value'] = torch.cat([row['value'] for row in collected])
    e['reward'] = torch.linspace(.1, .4, n, dtype=torch.float64)
    e['terminated'] = torch.zeros(n, dtype=torch.bool)
    e['truncated'] = torch.zeros(n, dtype=torch.bool)
    e['truncated' if truncated else 'terminated'][-1] = True
    # Final true successor value differs from current value, tests bootstrap use.
    e['next_value'] = torch.cat([e['value'][1:], torch.tensor([.8], dtype=torch.float64)])
    return e


def finite_differences():
    maximum, coordinates = 0., 0
    for lengths in ((2, 3), (3, 5)):
        for clipping in (False, True):
            for std in (-.5, -3.2, .5):
                model = model_new()
                with torch.no_grad():
                    model.raw_logstd.fill_(std)
                config = RecurrentPPOConfig(entropy_coef=.007, normalize_advantage=False)
                trainer = RecurrentPPOTrainer(model, config)
                episodes = trainer.tensor_episodes([episode_new(model, n) for n in lengths])
                for e in episodes:
                    e['advantages'] = torch.linspace(-1, 1, len(e['obs']), dtype=torch.float64)
                    if clipping:
                        ratios = torch.tensor([.7, 1.4, 1.05, .85, 1.35][:len(e['obs'])], dtype=torch.float64)
                        e['log_prob'] -= ratios.log()
                loss, _ = recurrent_loss(model, episodes, config)
                loss.backward()
                for name, parameter in model.named_parameters():
                    size = parameter.numel()
                    if name == 'raw_logstd':
                        indices = [1, 17, 41]
                    elif 'gru.weight' in name or 'gru.bias' in name:
                        # PyTorch GRU packs reset/update/new gates along dim0;
                        # test a coordinate in each gate rather than only one.
                        per_gate = size // 3
                        indices = [k * per_gate + per_gate // 2 for k in range(3)]
                    else:
                        indices = [size // 2]
                    flat = parameter.reshape(-1)
                    for index in indices:
                        analytic = float(parameter.grad.reshape(-1)[index])
                        original = float(flat[index])
                        epsilon = 1e-6
                        with torch.no_grad():
                            flat[index] = original + epsilon
                            plus = float(recurrent_loss(model, episodes, config)[0])
                            flat[index] = original - epsilon
                            minus = float(recurrent_loss(model, episodes, config)[0])
                            flat[index] = original
                        numerical = (plus - minus) / (2 * epsilon)
                        error = abs(numerical - analytic)
                        assert error < 4e-7 + 2e-5 * abs(numerical), (lengths, clipping, std, name, index, analytic, numerical)
                        maximum = max(maximum, error)
                        coordinates += 1
    return dict(coordinates=coordinates, max_absolute_error=maximum,
                regimes='unequal episode lengths, clipped/unclipped, std interior/outside both bounds, all GRU gates')


def replay_boundaries_weighting():
    model = model_new()
    trainer = RecurrentPPOTrainer(model, RecurrentPPOConfig(normalize_advantage=False))
    raw = [episode_new(model, 2), episode_new(model, 5, truncated=True)]
    episodes = trainer.tensor_episodes(raw)
    errors = trainer.check_behavior_replay(episodes)
    assert max(errors.values()) < 1e-12
    for e in episodes:
        np_adv, np_ret = gae(*(e[k].numpy() for k in ('reward', 'value', 'next_value', 'terminated', 'truncated')),
                             trainer.config.gamma, trainer.config.gae_lambda)
        np.testing.assert_array_equal(e['advantages'].numpy(), np_adv)
        np.testing.assert_array_equal(e['returns'].numpy(), np_ret)
        assert replay_mean(model, e).shape == (len(e['obs']), 42)
    combined, _ = recurrent_loss(model, episodes, trainer.config)
    singles = [recurrent_loss(model, [e], trainer.config)[0] for e in episodes]
    torch.testing.assert_close(combined, (singles[0] * 2 + singles[1] * 5) / 7, rtol=1e-14, atol=1e-14)
    # Uniform episode sampling needs length scaling, including entropy, so its
    # expected minibatch gradient is the full sample-weighted gradient.
    stochastic_expected = (singles[0] * (2 * 2 / 7) + singles[1] * (2 * 5 / 7)) / 2
    full_grads = torch.autograd.grad(combined, tuple(model.parameters()), retain_graph=True)
    sampled_grads = torch.autograd.grad(stochastic_expected, tuple(model.parameters()))
    for a, b in zip(full_grads, sampled_grads):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)
    # A recurrent trace continued through reset must differ: replay starts from
    # true zero and equals the original independently collected second episode.
    joined = {'obs': torch.cat([e['obs'] for e in episodes])}
    joined_mean = replay_mean(model, joined)
    reset_mean = replay_mean(model, episodes[1])
    assert (joined_mean[2:] - reset_mean).abs().max() > 1e-8
    # Full KL uses sample weighting and fresh reset hidden states.
    with torch.no_grad():
        model.actor.head.bias[0] += .02
    expected = sum(gaussian_kl(e['mean'], e['logstd'], replay_mean(model, e), model.logstd()).sum()
                   for e in episodes) / 7
    assert abs(trainer.rollout_kl(episodes) - float(expected.detach())) < 1e-15
    before = trainer.snapshot()
    try:
        trainer.update(raw)
    except ValueError as error:
        assert 'stale policy' in str(error)
    else:
        raise AssertionError('stale behavior accepted')
    assert same(before, trainer.snapshot())
    invalids = []
    missing_reset = deepcopy(raw)
    missing_reset[0].pop('starts_at_reset')
    invalids.append(missing_reset)
    partial = deepcopy(raw)
    partial[0]['terminated'][-1] = False
    invalids.append(partial)
    internal_reset = deepcopy(raw)
    internal_reset[1]['truncated'][1] = True
    invalids.append(internal_reset)
    wrong_logp = deepcopy(raw)
    wrong_logp[1]['log_prob'][0] += .01
    # Restore sampling weights before checking logprob mismatch alone.
    with torch.no_grad():
        model.actor.head.bias[0] -= .02
    for invalid in invalids:
        try:
            trainer.update(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid recurrent episode accepted')
    try:
        trainer.update(wrong_logp)
    except ValueError as error:
        assert 'log_prob' in str(error)
    else:
        raise AssertionError('wrong historical logprob accepted')
    return dict(replay_errors=errors, checks='reset boundaries, sample weighting, manual GAE, true full-rollout KL, stale/partial/logprob rejection')


def rollback_retry():
    model = model_new()
    config = RecurrentPPOConfig(epochs=1, episodes_per_minibatch=2, normalize_advantage=False)
    trainer = RecurrentPPOTrainer(model, config)
    initial = trainer.update([episode_new(model, 3), episode_new(model, 5)])
    assert initial['accepted_steps'] == 1
    assert trainer.actor_optimizer.state and trainer.critic_optimizer.state
    for optimizer in (trainer.actor_optimizer, trainer.critic_optimizer):
        optimizer.param_groups[0]['lr'] = 1.
    trainer.config = replace(config, target_kl=0., backtracks=2)
    episodes = [episode_new(model, 3), episode_new(model, 5)]
    before = trainer.snapshot()
    rejected = trainer.update(episodes)
    assert rejected['stopped'] and rejected['steps'][0]['attempts'] == 3
    assert same(before, trainer.snapshot()), 'warmed model/Adam/counters/LR rejection mismatch'
    trainer.config = replace(config, target_kl=.002, backtracks=20)
    permutation_rng = torch.get_rng_state()
    accepted = trainer.update(episodes)
    attempts = accepted['steps'][0]['attempts']
    assert accepted['steps'][0]['accepted'] and attempts > 1
    expected = trainer.snapshot()
    trainer.restore(before)
    for optimizer in (trainer.actor_optimizer, trainer.critic_optimizer):
        optimizer.param_groups[0]['lr'] *= .5 ** (attempts - 1)
    torch.set_rng_state(permutation_rng)
    direct = trainer.update(episodes)
    assert direct['steps'][0]['attempts'] == 1
    assert same(expected, trainer.snapshot()), 'retry differs from one accepted direct Adam step'
    return dict(rejected_attempts=3, accepted_attempts=attempts,
                state='exact actor+critic+both warmed Adam states+LR+step/update counters')


def checkpoint_restore(directory):
    config = RecurrentPPOConfig(epochs=1, episodes_per_minibatch=2)
    trainer = RecurrentPPOTrainer(model_new(), config)
    trainer.update([episode_new(trainer.model, 2), episode_new(trainer.model, 4)])
    state = trainer.snapshot()
    path = directory / 'recurrent.pt'
    identity = dict(policy='gru238-full-episode-v1', physics='synthetic-correctness-only')
    save_checkpoint(path, trainer, identity, extra={'normalization': np.arange(255)})
    expected_rng = (np.random.random(3), random.random(), torch.rand(3))
    # Build a new model+optimizer instance, as an independent evaluator would.
    restored = RecurrentPPOTrainer(model_new(), config)
    extra = load_checkpoint(path, restored, identity)
    assert same(state, restored.snapshot())
    assert same(expected_rng, (np.random.random(3), random.random(), torch.rand(3)))
    assert np.array_equal(extra['normalization'], np.arange(255))
    episodes = [episode_new(trainer.model, 3), episode_new(trainer.model, 5)]
    inputs = restored.tensor_episodes(episodes)
    errors = restored.check_behavior_replay(inputs)
    assert max(errors.values()) < 1e-12
    permutation_rng = torch.get_rng_state()
    result_a = trainer.update(episodes)
    torch.set_rng_state(permutation_rng)
    result_b = restored.update(episodes)
    assert same(result_a, result_b) and same(trainer.snapshot(), restored.snapshot())
    try:
        load_checkpoint(path, restored, {'mismatch': True})
    except ValueError:
        pass
    else:
        raise AssertionError('identity mismatch accepted')
    return 'new-instance checkpoint restores model/Adam/counters/RNG; next replay/update exactly equivalent'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parent / 'RECURRENT_CHECKS.json')
    args = parser.parse_args()
    args.report.write_text(json.dumps(dict(all_passed=False, status='RUNNING_OR_FAILED')) + '\n', encoding='utf-8')
    torch.set_num_threads(1)
    torch.manual_seed(20261110)
    np.random.seed(20261111)
    random.seed(20261112)
    results = {}
    with tempfile.TemporaryDirectory(prefix='recurrent-check-', dir=Path(__file__).resolve().parent) as temp:
        for name, function in [('finite_differences', finite_differences),
                               ('reset_replay_weighting', replay_boundaries_weighting),
                               ('rollback_retry', rollback_retry),
                               ('checkpoint_restore', lambda: checkpoint_restore(Path(temp)))]:
            results[name] = function()
            print(json.dumps({name: results[name]}), flush=True)
    here = Path(__file__).resolve().parent
    hashes = {str(here / name): hashlib.sha256((here / name).read_bytes()).hexdigest()
              for name in ('models.py', 'ppo.py', 'recurrent.py', 'check_recurrent.py')}
    report = dict(all_passed=True, checks=results, source_hashes=hashes, torch=torch.__version__,
        limitations=['Full-episode BPTT only; partial episodes and burn-in unsupported',
                     'starts_at_reset=True is required collector provenance, not independently provable from observations',
                     'Synthetic correctness only; physical recurrent rollout integration/behavior not validated'])
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', report=str(args.report.resolve()), source_hashes=hashes)), flush=True)


if __name__ == '__main__':
    main()
