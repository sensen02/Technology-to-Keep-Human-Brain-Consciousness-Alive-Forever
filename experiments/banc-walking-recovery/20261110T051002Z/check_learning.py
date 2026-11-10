"""CPU correctness only: no physics rollout, installation, or model training.

Run with this experiment's env/bin/python. Optional --legacy-npz imports a real
historical weight file read-only in addition to the synthetic import regression.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import json
import hashlib
import numpy as np
import torch
from models import ActorCritic, latent_log_prob, action_log_prob, gaussian_kl
from ppo import PPOConfig, PPOTrainer, RolloutBuffer, gae, ppo_loss, save_checkpoint, load_checkpoint


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def make_batch(model, n):
    obs = torch.randn(n, model.obs_dim, dtype=torch.float64) * 0.2
    cv = torch.randn(n, model.critic_obs_dim, dtype=torch.float64) * 0.2
    with torch.no_grad():
        mean = model.actor(obs)
        ls = model.logstd().expand_as(mean).clone()
        latent = mean + torch.randn_like(mean) * ls.exp()
        lp = latent_log_prob(latent, mean, ls)
    return dict(obs=obs, critic_obs=cv, latent=latent, log_prob=lp,
                mean=mean, logstd=ls, advantages=torch.linspace(-1, 1, n, dtype=torch.float64),
                returns=torch.linspace(0.2, 1.3, n, dtype=torch.float64))


def finite_differences():
    checks, max_error = 0, 0.0
    # Clipped case mixes active/inactive branches and both signs of advantage.
    for n in (3, 8):
        for regime in ('unclipped', 'clipped'):
            for std in (-0.5, -3.2, 0.5, -3 + 1e-3, 0.3 - 1e-3):
                model = ActorCritic('ff238', initial_logstd=std)
                batch = make_batch(model, n)
                if regime == 'clipped':
                    desired = torch.tensor(([0.7, 1.4, 1.05, 0.8, 1.3, 0.97, 1.4, 0.7])[:n], dtype=torch.float64)
                    batch['log_prob'] -= desired.log()
                config = PPOConfig(normalize_advantage=False, entropy_coef=0.007)
                loss, _ = ppo_loss(model, batch, config)
                loss.backward()
                # At least one coordinate in every actor/critic tensor plus several logstd axes.
                for name, parameter in model.named_parameters():
                    flat = parameter.reshape(-1)
                    coordinates = (0, 17, 41) if name == 'raw_logstd' else (flat.numel() // 2,)
                    for index in coordinates:
                        analytic = float(parameter.grad.reshape(-1)[index])
                        original = float(flat[index])
                        epsilon = 1e-6
                        with torch.no_grad():
                            flat[index] = original + epsilon
                            plus = float(ppo_loss(model, batch, config)[0])
                            flat[index] = original - epsilon
                            minus = float(ppo_loss(model, batch, config)[0])
                            flat[index] = original
                        numeric = (plus - minus) / (2 * epsilon)
                        error = abs(numeric - analytic)
                        assert error < 3e-7 + 2e-5 * abs(numeric), (n, regime, std, name, index, numeric, analytic)
                        max_error = max(max_error, error)
                        checks += 1
    # Exact clamp boundaries are nondifferentiable; verify stated inward/outward
    # directional derivatives of the actual loss and PyTorch's inward convention.
    for bound, inward in ((-3.0, 1), (0.3, -1)):
        model = ActorCritic('ff238', initial_logstd=bound)
        batch = make_batch(model, 5)
        config = PPOConfig(entropy_coef=0.01)
        loss = ppo_loss(model, batch, config)[0]
        loss.backward()
        derivative = float(model.raw_logstd.grad[2])
        base = float(loss.detach())
        with torch.no_grad():
            model.raw_logstd[2] = bound + inward * 1e-7
            inside = float(ppo_loss(model, batch, config)[0])
            model.raw_logstd[2] = bound - inward * 1e-7
            outside = float(ppo_loss(model, batch, config)[0])
        assert abs((inside - base) / (inward * 1e-7) - derivative) < 2e-5
        assert outside == base
    return dict(coordinates=checks, max_absolute_error=max_error, boundary='one-sided clamp kink checks')


def gae_checks():
    # Middle timeout cuts the trace but bootstraps its true final state (value 4).
    adv, ret = gae([1, 2, 3], [0.5, 0.6, 0.7], [0.6, 4, 99],
                   [False, False, True], [False, True, False], 0.9, 0.8)
    expected = np.array([1 + .9 * .6 - .5 + .9 * .8 * (2 + .9 * 4 - .6),
                         2 + .9 * 4 - .6, 3 - .7])
    np.testing.assert_allclose(adv, expected, atol=1e-14)
    np.testing.assert_allclose(ret, expected + [.5, .6, .7], atol=1e-14)
    # Collector cutoff uses explicit successor bootstrap, no guessed V(s_t).
    a, _ = gae([1], [2], [5], [False], [False], .9, .8)
    np.testing.assert_allclose(a, [3.5])
    # Terminal followed by a reset transition must never leak reset rewards.
    a, _ = gae([1, 100], [2, 3], [99, 4], [True, False], [False, True], .9, .8)
    np.testing.assert_allclose(a, [-1, 100 + .9 * 4 - 3])
    return 'manual terminal, timeout, reset boundary, collector cutoff passed'


def legacy_compare(path):
    model = ActorCritic('ff495').import_legacy_npz(path)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(7, 495))
    with np.load(path, allow_pickle=False) as p:
        mean = np.tanh(np.tanh(x @ p['w0'] + p['b0']) @ p['w1'] + p['b1']) @ p['w2'] + p['b2']
        value = (np.tanh(np.tanh(x @ p['vc0'] + p['vbc0']) @ p['vc1'] + p['vbc1']) @ p['vc'] + p['vbc'])[:, 0]
        for i in range(3):
            assert np.array_equal(model.actor.layers[i].weight.detach().numpy(), p[f'w{i}'].T)
            assert np.array_equal(model.actor.layers[i].bias.detach().numpy(), p[f'b{i}'])
        assert np.array_equal(model.raw_logstd.detach().numpy(), p['logstd'])
        native_std = np.exp(np.clip(p['logstd'], -3, .3))
        stored_dtype = str(p['logstd'].dtype)
        std = np.exp(np.clip(p['logstd'].astype(np.float64), -3, .3))
    with torch.no_grad():
        tm = model.actor(torch.tensor(x)).numpy()
        tv = model.value(torch.tensor(x)).numpy()
        ts = model.logstd().exp().numpy()
    np.testing.assert_allclose(tm, mean, atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(tv, value, atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(ts, std, atol=1e-15, rtol=1e-15)
    return dict(mean_max_error=float(np.max(np.abs(tm - mean))), value_max_error=float(np.max(np.abs(tv - value))),
                logstd_storage_dtype=stored_dtype, std_float64_reference_max_error=float(np.max(np.abs(ts - std))),
                std_native_storage_math_max_difference=float(np.max(np.abs(ts - native_std))),
                std_min=float(ts.min()), std_max=float(ts.max()))


def synthetic_npz(directory):
    rng = np.random.default_rng(33)
    p = {}
    for key, shape in dict(w0=(495,128), b0=(128,), w1=(128,128), b1=(128,),
        w2=(128,42), b2=(42,), vc0=(495,128), vbc0=(128,), vc1=(128,128),
        vbc1=(128,), vc=(128,1), vbc=(1,)).items():
        p[key] = rng.normal(size=shape) * .05
    p['logstd'] = np.linspace(-3.2, .5, 42)
    path = directory / 'synthetic.npz'
    np.savez(path, **p)
    return legacy_compare(path)


def rollback_check():
    model = ActorCritic('ff238')
    config = PPOConfig(epochs=1, minibatch=7, normalize_advantage=False)
    trainer = PPOTrainer(model, config)
    first = trainer.update(make_batch(model, 7))
    assert first['accepted_steps'] == 1
    # Both Adams already contain nonzero first/second moments and step counters.
    trainer.config = replace(config, actor_lr=1., critic_lr=1., target_kl=0., backtracks=2)
    for opt in (trainer.actor_optimizer, trainer.critic_optimizer):
        opt.param_groups[0]['lr'] = 1.
    snapshot = trainer.snapshot()
    result = trainer.update(make_batch(model, 7))
    assert result['stopped'] and result['steps'][0]['attempts'] == 3
    assert same(snapshot, trainer.snapshot()), 'rejection changed weights/Adam/time/LR/counters'
    # Successful retries must use exactly the same Adam step, not count failures.
    trainer.config = replace(config, target_kl=0.002, backtracks=20)
    batch = make_batch(model, 7)
    before = trainer.snapshot()
    permutation_rng = torch.get_rng_state()
    result = trainer.update(batch)
    assert result['steps'][0]['accepted'] and result['steps'][0]['attempts'] > 1
    accepted = trainer.snapshot()
    attempts = result['steps'][0]['attempts']
    trainer.restore(before)
    for opt in (trainer.actor_optimizer, trainer.critic_optimizer):
        opt.param_groups[0]['lr'] *= .5 ** (attempts - 1)
    torch.set_rng_state(permutation_rng)
    result_direct = trainer.update(batch)
    assert result_direct['steps'][0]['attempts'] == 1
    assert same(accepted, trainer.snapshot()), 'retry differs from one direct accepted Adam step'
    return dict(rejection='exact model+both Adam+step+lr+counters', accepted_retry_attempts=attempts)


def checkpoint_check(directory):
    trainer = PPOTrainer(ActorCritic('ff238'), PPOConfig(epochs=1, minibatch=4))
    trainer.update(make_batch(trainer.model, 4))
    path = directory / 'checkpoint.pt'
    identity = dict(xml_sha256='synthetic-test-only', observation='238-v1')
    state = trainer.snapshot()
    save_checkpoint(path, trainer, identity, extra={'normalization': np.arange(3)})
    expected_rng = (np.random.random(3), random.random(), torch.rand(3))
    trainer.update(make_batch(trainer.model, 4))
    extra = load_checkpoint(path, trainer, identity)
    assert same(state, trainer.snapshot())
    assert same(expected_rng, (np.random.random(3), random.random(), torch.rand(3)))
    assert np.array_equal(extra['normalization'], np.arange(3))
    try:
        load_checkpoint(path, trainer, {'wrong': True})
    except ValueError:
        pass
    else:
        raise AssertionError('identity mismatch not rejected')
    assert same(state, trainer.snapshot())
    assert not list(directory.glob('*.tmp'))
    return 'atomic save/load, exact model/Adam/counters/RNG, identity rejection passed'


def gru_check():
    model = ActorCritic('gru238')
    x = torch.randn(5, 2, 238, dtype=torch.float64)
    initial = model.actor.initial_state(2)
    resets = torch.zeros(5, 2, dtype=torch.bool)
    resets[3, 0] = True
    means, state = model.actor.sequence(x, initial, resets)
    manual_state = initial
    manual = []
    for t in range(5):
        mean, manual_state = model.actor.step(x[t], manual_state, resets[t])
        manual.append(mean)
    assert torch.equal(means, torch.stack(manual)) and torch.equal(state, manual_state)
    means[2:].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.actor.parameters())
    try:
        PPOTrainer(model)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('unsupported recurrent PPO silently accepted')
    return 'explicit GRU sequence/step/reset/backprop passed; recurrent PPO intentionally rejected'


def buffer_and_density_check():
    model = ActorCritic('ff495')
    buffer = RolloutBuffer()
    for t in range(3):
        obs = torch.randn(495, dtype=torch.float64)
        out = model.act(obs)
        buffer.add(obs, obs, out['latent'], out['log_prob'], out['mean'], out['logstd'],
                   out['value'], 1., t == 2, False, .4)
    b = buffer.finish(.9, .8)
    assert b['obs'].shape == (3, 495) and b['latent'].shape == (3, 42)
    result = PPOTrainer(model, PPOConfig(epochs=1, minibatch=3)).update(b)
    assert result['accepted_steps'] == 1
    z = torch.tensor([[0., .4, -1.]], dtype=torch.float64)
    mean, ls = z * 0, z * 0 - .5
    expected = latent_log_prob(z, mean, ls) - torch.log(1 - z.tanh().square()).sum(-1)
    torch.testing.assert_close(action_log_prob(z, mean, ls), expected)
    assert torch.isfinite(action_log_prob(z * 1000, mean, ls)).all()
    torch.testing.assert_close(gaussian_kl(mean, ls, mean, ls), torch.zeros(1, dtype=torch.float64))
    return 'buffer singleton shapes, FF495 PPO, latent/squashed density passed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-npz', type=Path)
    parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parent / 'LEARNING_CHECKS.json')
    args = parser.parse_args()
    args.report.write_text(json.dumps(dict(all_passed=False, status='RUNNING_OR_FAILED')) + '\n', encoding='utf-8')
    torch.set_num_threads(1)
    torch.manual_seed(427)
    np.random.seed(428)
    random.seed(429)
    results = {}
    with tempfile.TemporaryDirectory(prefix='learning-check-', dir=Path(__file__).resolve().parent) as directory:
        directory = Path(directory)
        for name, fn in [('finite_differences', finite_differences), ('gae', gae_checks),
                         ('legacy_synthetic', lambda: synthetic_npz(directory)),
                         ('rollback', rollback_check), ('checkpoint', lambda: checkpoint_check(directory)),
                         ('gru', gru_check), ('buffer_density', buffer_and_density_check)]:
            results[name] = fn()
            print(json.dumps({name: results[name]}), flush=True)
        if args.legacy_npz:
            results['legacy_original'] = legacy_compare(args.legacy_npz)
            print(json.dumps({'legacy_original': results['legacy_original']}), flush=True)
    here = Path(__file__).resolve().parent
    hashes = {str(here / name): hashlib.sha256((here / name).read_bytes()).hexdigest()
              for name in ('models.py', 'ppo.py', 'check_learning.py')}
    report = dict(all_passed=True, status='PASS', checks=results, torch=torch.__version__,
                  source_hashes=hashes, legacy_npz=str(args.legacy_npz.resolve()) if args.legacy_npz else None,
                  legacy_npz_sha256=hashlib.sha256(args.legacy_npz.read_bytes()).hexdigest() if args.legacy_npz else None,
                  limitations=['GRU PPO not integrated; explicit sequence/inference API only',
                               'Hard logstd bound endpoints are clamp kinks; one-sided derivatives tested',
                               'No physics or behavioral learning claims from these checks'])
    # The initial false report prevents a stale PASS after failure. Callers must
    # additionally require a zero exit code and matching source hashes.
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checks': list(results), 'torch': torch.__version__,
                      'report': str(args.report.resolve()), 'source_hashes': hashes}), flush=True)


if __name__ == '__main__':
    main()
