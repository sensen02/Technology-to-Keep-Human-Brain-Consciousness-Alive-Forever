"""Sequence distillation source: declared learned FF teacher, not walking acceptance."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from run_baseline import model_and_trainer, seed_all, record_row
from physics import Core, summarise
from identity import atomic_json, capture, verify, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    seed_all(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = dict(teacher='legacy learned FF best update9; fails slip gate', seed=args.seed,
                  episodes=args.episodes, ticks=args.ticks, decision_ms=5, target_speed=.10,
                  perturbation='episode0 exact reset; later episodes rootXY velocity sd0.01, actuated velocity sd0.02rad/s; initial velocities only',
                  target='deterministic teacher rate action sampled/held every5ms; no teacher in deployment',
                  validation='hold out final trajectory; no duplicated exact trajectories')
    teacher, _ = model_and_trainer()
    core = Core()
    identity = capture(core, config)
    atomic_json(out/'IDENTITY.json', identity)
    rng = np.random.default_rng(args.seed)
    arrays = {}
    evaluations = []
    start = time.monotonic()
    for episode in range(args.episodes):
        core.reset()
        if episode:
            qpos = core.d.qpos.copy()
            qvel = core.d.qvel.copy()
            qvel[:2] += rng.normal(0, .01, 2)
            qvel[core.a_dof] += rng.normal(0, .02, 42)
            core.reset_from(qpos, qvel)
        previous = np.zeros(42)
        observations, targets, critic, rec = [], [], [], []
        for k in range(0, args.ticks, 5):
            observations.append(core.obs238(previous,.10))
            critic.append(np.r_[observations[-1], core.critic_extras(), (args.ticks-k)*.001])
            action = teacher.act(core.obs(), deterministic=True)['action'][0].numpy()
            targets.append(action.copy())
            for _ in range(min(5,args.ticks-k)):
                m = core.tick(action)
                rec.append(record_row(core,m))
            previous = action.copy()
        arrays[f'obs_{episode}'] = np.asarray(observations)
        arrays[f'action_{episode}'] = np.asarray(targets)
        arrays[f'critic_obs_{episode}'] = np.asarray(critic)
        arrays[f'qpos_final_{episode}'] = core.d.qpos.copy()
        evaluation = summarise(rec,args.ticks*80,initial_heading=core.initial_heading)
        evaluations.append(evaluation)
        print(json.dumps(dict(episode=episode,forward=evaluation['forward_signed_native_per_s'],
                    com=evaluation['com_height_min'],nonfoot=evaluation['nonfoot_force_max_native'],
                    slip=evaluation['slip_ratio_max'])),flush=True)
    np.savez_compressed(out/'sequences.npz',**arrays)
    verify(identity)
    atomic_json(out/'DATASET.json',dict(config=config, identity_sha256=identity['identity_sha256'],
            dataset_sha256=sha256(out/'sequences.npz'), independent_episodes=args.episodes,
            train_episodes=list(range(args.episodes-1)),validation_episodes=[args.episodes-1],
            evaluations=evaluations,wall_s=time.monotonic()-start,
            conclusion='teacher behavior measured here; imitation is initialization only'))


if __name__=='__main__':
    main()
