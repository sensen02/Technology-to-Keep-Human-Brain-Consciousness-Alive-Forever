"""Corrected-learning FF495 baseline. Separate diagnostic from final acceptance."""
from pathlib import Path
import argparse
from dataclasses import asdict
import json
import math
import random
import time
import numpy as np
import torch
from models import ActorCritic
from ppo import PPOConfig, PPOTrainer, RolloutBuffer, save_checkpoint, load_checkpoint
from physics import Core, NativeRecorder, summarise, W, GATES
from identity import HERE, LEGACY, atomic_json, capture, sha256, verify


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def record_row(core, measurement):
    return (float(core.d.time), core.d.subtree_com[core.th].copy(),
            measurement['loads'].copy(), float(measurement['nonfoot']),
            measurement['speed'].copy())


@torch.no_grad()
def evaluate(model, ticks=2000, native_dir=None):
    start = time.monotonic()
    recorder = None if native_dir is None else NativeRecorder(native_dir)
    core = Core(recorder=recorder)
    rec = []
    try:
        for _ in range(ticks):
            out = model.act(core.obs(), deterministic=True)
            m = core.tick(out['action'][0].cpu().numpy())
            rec.append(record_row(core, m))
    finally:
        native = None if recorder is None else recorder.close()
    result = summarise(rec, ticks * 80, initial_heading=core.initial_heading, native_summary=native)
    result['wall_s'] = time.monotonic() - start
    result['actual_ticks'] = ticks
    result['evaluation_scope'] = 'derived engineering diagnostic; no prewarm handover; not original acceptance'
    return result


def legacy_reward(core, measurement, previous_x):
    # Exact current legacy reward proxy, including its known shortcomings. Kept only
    # as the first corrected-learning baseline, NOT as an acceptance definition.
    x = float(core.d.subtree_com[core.th, 0])
    vx = (x - previous_x) / .001
    support = float(measurement['loads'].sum() / W)
    nonfoot = float(measurement['nonfoot'])
    height = float(core.d.subtree_com[core.th, 2])
    allow = max(.5 * abs(vx), .02)
    slip = float(np.maximum(0, measurement['speed'] - allow).sum())
    fall = nonfoot > GATES['nonfoot_force_max_W'] or height < GATES['com_height_min'] - .05
    components = dict(forward=6 * float(np.clip(vx, -.5, .5)),
                      support=1.0 if support >= .9 else -2.0,
                      nonfoot=-2 * nonfoot, slip=-slip,
                      com=-4 * max(0., 1.2 - height), alive=.1,
                      terminal=-20.0 if fall else 0.)
    return sum(components.values()), bool(fall), x, components


def baseline_config():
    return PPOConfig(gamma=.5 ** (.001 / .5), gae_lambda=.95, clip=.2,
                     actor_lr=1e-4, critic_lr=1e-4, epochs=4, minibatch=64,
                     max_grad_norm=.5, target_kl=.03, backtracks=6)


def model_and_trainer(config=None):
    model = ActorCritic('ff495', dtype=torch.float64)
    model.import_legacy_npz(LEGACY)
    return model, PPOTrainer(model, config or baseline_config())


def require_checks():
    for filename in ('LEARNING_CHECKS.json', 'PHYSICS_CHECKS.json'):
        path = HERE / filename
        report = json.loads(path.read_text())
        # Check runners set explicit top-level success after all assertions.
        if not (report.get('all_passed') is True or report.get('passed') is True or report.get('success') is True):
            raise RuntimeError('required check report not passing: ' + str(path))
        for p, expected in report.get('source_hashes', {}).items():
            if sha256(p) != expected:
                raise RuntimeError('checks stale for source: ' + p)


def make_run(out, config, label):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core = Core()
    identity = capture(core, dict(label=label, **config))
    atomic_json(out / 'IDENTITY.json', identity)
    return out, identity


def verify_legacy(args):
    model, trainer = model_and_trainer()
    out, identity = make_run(args.out, dict(ticks=args.ticks, seed=args.seed), 'saved legacy independent replay')
    eval_result = evaluate(model, args.ticks, out / 'native' if args.native else None)
    expected = json.loads((LEGACY.parent / 'WALK_BC_LOG.json').read_text())['best_eval']
    keys = ('forward_signed_native_per_s', 'foot_support_mean', 'foot_support_min',
            'nonfoot_force_max_W', 'com_height_min', 'slip_ratio_max')
    deltas = {k: eval_result[k] - expected[k] for k in keys}
    # Only the original 2000-tick window can be directly compared.
    matching = None if args.ticks != 2000 else all(abs(v) < 2e-5 for v in deltas.values())
    result = dict(legacy_model_sha256=sha256(LEGACY), evaluation=eval_result,
                  original_log_deltas=deltas, original_window_matches=matching)
    if matching is False:
        atomic_json(out / 'EVALUATION.json', result)
        raise AssertionError('legacy independent replay changed beyond declared tolerance')
    saved = save_checkpoint(out / 'initial.pt', trainer, identity,
                            extra=dict(source_npz=str(LEGACY), source_sha256=sha256(LEGACY)))
    result['checkpoint_sha256'] = sha256(out / 'initial.pt')
    verify(identity)
    atomic_json(out / 'EVALUATION.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def train(args):
    require_checks()
    config = baseline_config()
    model, trainer = model_and_trainer(config)
    settings = dict(seed=args.seed, updates=args.updates, episode_ticks=args.ticks,
                    rollout_ticks=args.rollout_ticks, ppo=asdict(config),
                    initialization='legacy saved best update9, not BC',
                    reward='legacy current source proxy (not gate equivalent)',
                    algorithm_changes='autograd gradients; separate actor/critic clipping and Adam; transactional full KL; no performance attribution to any single fix')
    out, identity = make_run(args.out, settings, 'corrected FF495 short diagnostic')
    log = dict(config=settings, identity_sha256=identity['identity_sha256'], updates=[],
               status='running', engineering_gates=GATES, original_acceptance='not evaluated')
    save_checkpoint(out / 'initial.pt', trainer, identity, extra=dict(update=-1))
    atomic_json(out / 'TRAINING.json', log)
    start = time.monotonic()
    core = Core()
    try:
        for update in range(args.updates):
            buffer = RolloutBuffer()
            counts, lengths, returns = {}, [], []
            components_sum = {}
            while len(buffer.rows) < args.rollout_ticks:
                core.reset()
                previous_x = float(core.d.subtree_com[core.th, 0])
                total = 0.
                for k in range(args.ticks):
                    obs = core.obs()
                    sample = model.act(obs)
                    measurement = core.tick(sample['action'][0].cpu().numpy())
                    reward, fall, previous_x, components = legacy_reward(core, measurement, previous_x)
                    # This baseline explicitly defines the 2s task as finite horizon.
                    deadline = k + 1 == args.ticks
                    terminal = fall or deadline
                    with torch.no_grad():
                        next_value = 0. if terminal else float(model.value(torch.as_tensor(core.obs()[None], dtype=torch.float64))[0])
                    buffer.add(obs, obs, sample['latent'], sample['log_prob'], sample['mean'],
                               sample['logstd'], sample['value'], reward, terminal, False, next_value)
                    total += reward
                    for name, value in components.items():
                        components_sum[name] = components_sum.get(name, 0.) + value
                    if terminal:
                        lengths.append(k + 1)
                        returns.append(total)
                        reason = 'fall' if fall else 'deadline'
                        counts[reason] = counts.get(reason, 0) + 1
                        break
            batch = buffer.finish(config.gamma, config.gae_lambda)
            optimization = trainer.update(batch)
            checkpoint = out / 'checkpoints' / f'update_{update:04d}.pt'
            save_checkpoint(checkpoint, trainer, identity, extra=dict(update=update,
                environment_steps=len(buffer.rows), finite_horizon=True))
            evaluation = evaluate(model, args.ticks)
            row = dict(update=update, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                       decisions=len(buffer.rows), episode_lengths=lengths, episode_returns=returns,
                       termination_counts=counts, mean_components={k: v/len(buffer.rows) for k,v in components_sum.items()},
                       optimization=optimization, evaluation=evaluation)
            log['updates'].append(row)
            verify(identity)
            atomic_json(out / 'TRAINING.json', log)
            print(json.dumps(dict(update=update, decisions=len(buffer.rows),
                  accepted=sum(s['accepted'] for s in optimization['steps']), kl=optimization['kl'],
                  evaluation=evaluation), ensure_ascii=False), flush=True)
        save_checkpoint(out / 'final.pt', trainer, identity, extra=dict(update=args.updates-1))
        log.update(status='completed', wall_s=time.monotonic()-start)
    except BaseException as exc:
        log.update(status='failed', error=repr(exc), wall_s=time.monotonic()-start)
        atomic_json(out / 'TRAINING.json', log)
        raise
    verify(identity)
    atomic_json(out / 'TRAINING.json', log)


def replay(args):
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = payload['identity']
    verify(identity)
    model = ActorCritic('ff495', dtype=torch.float64)
    trainer = PPOTrainer(model, PPOConfig(**payload['config']))
    load_checkpoint(checkpoint, trainer, identity)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  evaluation=evaluate(model, args.ticks, out/'native' if args.native else None))
    verify(identity)
    atomic_json(out/'EVALUATION.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['verify', 'train', 'replay'])
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--ticks', type=int, default=2000)
    p.add_argument('--rollout-ticks', type=int, default=8000)
    p.add_argument('--updates', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--native', action='store_true')
    args = p.parse_args()
    if args.ticks < 2 or args.updates < 1 or args.rollout_ticks < 1:
        p.error('positive budgets and at least two evaluation ticks required')
    seed_all(args.seed)
    {'verify': verify_legacy, 'train': train, 'replay': replay}[args.mode](args)


if __name__ == '__main__':
    main()
