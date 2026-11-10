"""Full-episode recurrent PPO. No stored hidden states or burn-in approximation.

Every input episode starts at a physical reset and includes its final transition.
A collector cutoff inside a physical episode must retain its complete reset prefix;
this initial API rejects partial episodes rather than guessing hidden state.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
import torch
from models import latent_log_prob, gaussian_kl
from ppo import PPOConfig, gae


@dataclass(frozen=True)
class RecurrentPPOConfig(PPOConfig):
    episodes_per_minibatch: int = 1
    replay_tolerance: float = 1e-8


def replay_mean(model, episode):
    """Differentiable full BPTT from actual reset; never reuses behavior hidden."""
    means, _ = model.actor.sequence(episode['obs'].unsqueeze(1), model.actor.initial_state(1))
    return means[:, 0]


def recurrent_loss(model, episodes, config):
    """Actual PPO objective, weighted by tick count across all supplied episodes."""
    count = sum(len(e['obs']) for e in episodes)
    if not count:
        raise ValueError('empty recurrent minibatch')
    actor_sum = model.raw_logstd.new_zeros(())
    critic_sum = model.raw_logstd.new_zeros(())
    clip_sum = model.raw_logstd.new_zeros(())
    ratio_sum = model.raw_logstd.new_zeros(())
    logstd = model.logstd()
    entropy = (logstd + .5 * math.log(2 * math.pi * math.e)).sum()
    for e in episodes:
        mean = replay_mean(model, e)
        logp = latent_log_prob(e['latent'], mean, logstd)
        ratio = (logp - e['log_prob']).exp()
        objective = torch.minimum(ratio * e['advantages'],
            ratio.clamp(1 - config.clip, 1 + config.clip) * e['advantages'])
        actor_sum = actor_sum - objective.sum()
        critic_sum = critic_sum + (model.value(e['critic_obs']) - e['returns']).square().sum()
        clip_sum = clip_sum + ((ratio - 1).abs() > config.clip).to(ratio.dtype).sum()
        ratio_sum = ratio_sum + ratio.sum()
    actor = actor_sum / count - config.entropy_coef * entropy
    critic = config.value_coef * critic_sum / count
    return actor + critic, dict(actor_loss=actor, critic_loss=critic, entropy=entropy,
        clip_fraction=clip_sum / count, ratio_mean=ratio_sum / count, samples=count)


class RecurrentPPOTrainer:
    """Both Adam optimizers and networks are transactional under reset-replay KL.

    Epochs shuffle whole episodes only. The objective inside each minibatch uses
    total ticks as denominator. Minibatch gradients are additionally scaled by
    (rollout episodes / minibatch episodes) * (minibatch ticks / rollout ticks),
    giving the rollout sample-weighted gradient in expectation under uniform
    episode sampling, rather than weighting short and long episodes equally.
    Choose episodes_per_minibatch >= episode count for an exact full-rollout
    gradient per epoch. Full KL always weights ticks, irrespective of batching.
    """
    def __init__(self, model, config=None):
        if model.architecture != 'gru238':
            raise ValueError('recurrent PPO requires gru238')
        self.model = model
        self.config = config or RecurrentPPOConfig()
        c = self.config
        if (c.epochs < 1 or c.episodes_per_minibatch < 1 or c.backtracks < 0
                or c.target_kl < 0 or c.replay_tolerance <= 0 or c.max_grad_norm <= 0):
            raise ValueError('invalid recurrent PPO configuration')
        self.actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=c.actor_lr)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=c.critic_lr)
        self.accepted_steps = 0
        self.updates = 0

    def snapshot(self):
        return deepcopy(dict(model=self.model.state_dict(), actor=self.actor_optimizer.state_dict(),
            critic=self.critic_optimizer.state_dict(), accepted_steps=self.accepted_steps, updates=self.updates))

    def restore(self, snapshot):
        self.model.load_state_dict(snapshot['model'])
        self.actor_optimizer.load_state_dict(deepcopy(snapshot['actor']))
        self.critic_optimizer.load_state_dict(deepcopy(snapshot['critic']))
        self.accepted_steps = snapshot['accepted_steps']
        self.updates = snapshot['updates']

    def tensor_episodes(self, episodes):
        if not episodes:
            raise ValueError('empty recurrent rollout')
        result = []
        parameter = self.model.raw_logstd
        vector_keys = dict(obs=238, critic_obs=self.model.critic_obs_dim, latent=42, mean=42, logstd=42)
        scalar_keys = ('log_prob', 'value', 'reward', 'terminated', 'truncated', 'next_value')
        for episode in episodes:
            if episode.get('starts_at_reset') is not True:
                raise ValueError('each episode requires starts_at_reset=True provenance')
            e = {key: torch.as_tensor(episode[key], device=parameter.device, dtype=parameter.dtype).detach().clone()
                 for key in (*vector_keys, *scalar_keys)}
            n = len(e['obs'])
            if n < 1:
                raise ValueError('empty episode')
            for key, width in vector_keys.items():
                if e[key].shape != (n, width):
                    raise ValueError(f'invalid episode {key} shape: {e[key].shape}')
            for key in scalar_keys:
                if e[key].shape != (n,):
                    raise ValueError(f'invalid episode {key} shape: {e[key].shape}')
            if not all(torch.isfinite(v).all() for v in e.values()):
                raise ValueError('nonfinite episode')
            for key in ('terminated', 'truncated'):
                if not ((e[key] == 0) | (e[key] == 1)).all():
                    raise ValueError('termination flags must be boolean')
            boundary = e['terminated'].bool() | e['truncated'].bool()
            if not boundary[-1] or boundary[:-1].any():
                raise ValueError('split at physical episode boundaries; partial episodes unsupported')
            if 'advantages' in episode or 'returns' in episode:
                if 'advantages' not in episode or 'returns' not in episode:
                    raise ValueError('provide both advantages and returns')
                for key in ('advantages', 'returns'):
                    e[key] = torch.as_tensor(episode[key], device=parameter.device,
                                            dtype=parameter.dtype).detach().clone()
                    if e[key].shape != (n,) or not torch.isfinite(e[key]).all():
                        raise ValueError(f'invalid {key}')
            else:
                adv, ret = gae(*(e[key].cpu().numpy() for key in
                    ('reward', 'value', 'next_value', 'terminated', 'truncated')),
                    self.config.gamma, self.config.gae_lambda)
                e['advantages'] = torch.as_tensor(adv, dtype=parameter.dtype, device=parameter.device)
                e['returns'] = torch.as_tensor(ret, dtype=parameter.dtype, device=parameter.device)
            result.append(e)
        return result

    @torch.no_grad()
    def rollout_kl(self, episodes):
        numerator = self.model.raw_logstd.new_zeros(())
        count = 0
        for e in episodes:
            numerator += gaussian_kl(e['mean'], e['logstd'], replay_mean(self.model, e), self.model.logstd()).sum()
            count += len(e['obs'])
        if not count:
            raise ValueError('empty recurrent rollout')
        return float(numerator / count)

    @torch.no_grad()
    def check_behavior_replay(self, episodes):
        """Require current weights to be the sampling weights before any update.

        Compare means/std and sampled latent logprob, not just a permissive KL
        threshold. This catches reset/hidden-history errors and stale rollouts.
        """
        errors = dict(mean=0.0, logstd=0.0, log_prob=0.0)
        for e in episodes:
            means = replay_mean(self.model, e)
            ls = self.model.logstd().expand_as(means)
            logp = latent_log_prob(e['latent'], means, ls)
            for key, current in (('mean', means), ('logstd', ls), ('log_prob', logp)):
                errors[key] = max(errors[key], float((e[key] - current).abs().max()))
        if any(error > self.config.replay_tolerance for error in errors.values()):
            raise ValueError(f'stale policy or invalid reset/history: behavior replay errors {errors}')
        return errors

    def update(self, episodes):
        c = self.config
        episodes = self.tensor_episodes(episodes)
        replay_errors = self.check_behavior_replay(episodes)
        initial_kl = self.rollout_kl(episodes)
        if not math.isfinite(initial_kl) or initial_kl > c.target_kl:
            raise ValueError('initial recurrent KL exceeds budget')
        if c.normalize_advantage:
            all_adv = torch.cat([e['advantages'] for e in episodes])
            mean, scale = all_adv.mean(), all_adv.std(unbiased=False) + 1e-8
            for e in episodes:
                e['advantages'] = (e['advantages'] - mean) / scale
        total_samples = sum(len(e['obs']) for e in episodes)
        rows = []
        stopped = False
        for epoch in range(c.epochs):
            order = torch.randperm(len(episodes)).tolist()
            for start in range(0, len(order), c.episodes_per_minibatch):
                minibatch = [episodes[i] for i in order[start:start + c.episodes_per_minibatch]]
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss, metrics = recurrent_loss(self.model, minibatch, c)
                sample_weight_scale = (len(episodes) / len(minibatch)) * (metrics['samples'] / total_samples)
                loss = loss * sample_weight_scale
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite recurrent loss')
                loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(), c.max_grad_norm)
                critic_grad = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.max_grad_norm)
                if not torch.isfinite(actor_grad) or not torch.isfinite(critic_grad):
                    raise FloatingPointError('nonfinite recurrent gradient')
                snapshot = self.snapshot()
                accepted = False
                for attempt in range(c.backtracks + 1):
                    self.restore(snapshot)
                    for optimizer in (self.actor_optimizer, self.critic_optimizer):
                        for group in optimizer.param_groups:
                            group['lr'] *= .5 ** attempt
                        optimizer.step()
                    candidate_kl = self.rollout_kl(episodes)
                    if math.isfinite(candidate_kl) and candidate_kl <= c.target_kl:
                        accepted = True
                        self.accepted_steps += 1
                        break
                if not accepted:
                    self.restore(snapshot)
                    stopped = True
                row = {key: float(value.detach()) if torch.is_tensor(value) else value
                       for key, value in metrics.items()}
                row.update(epoch=epoch, episodes=len(minibatch), sample_weight_scale=sample_weight_scale,
                    accepted=accepted, attempts=attempt + 1,
                    candidate_kl=candidate_kl, kl=self.rollout_kl(episodes),
                    actor_grad_before=float(actor_grad), critic_grad_before=float(critic_grad),
                    actor_grad_after=min(float(actor_grad), c.max_grad_norm),
                    critic_grad_after=min(float(critic_grad), c.max_grad_norm))
                rows.append(row)
                if stopped:
                    break
            if stopped:
                break
        if any(row['accepted'] for row in rows):
            self.updates += 1
        with torch.no_grad():
            returns = torch.cat([e['returns'] for e in episodes])
            predicted = torch.cat([self.model.value(e['critic_obs']) for e in episodes])
            variance = returns.var(unbiased=False)
            ev = float(1 - (returns - predicted).var(unbiased=False) / variance) if variance > 0 else None
            saturation = torch.cat([e['latent'] for e in episodes]).tanh().abs() > .99
        kl = self.rollout_kl(episodes)
        return dict(steps=rows, stopped=stopped, kl=kl, kl_per_dim=kl / 42,
            explained_variance=ev, std_min=float(self.model.logstd().detach().exp().min()),
            std_max=float(self.model.logstd().detach().exp().max()),
            saturation_fraction=float(saturation.to(returns.dtype).mean()),
            actor_lr=self.actor_optimizer.param_groups[0]['lr'],
            critic_lr=self.critic_optimizer.param_groups[0]['lr'], accepted_steps=self.accepted_steps,
            updates=self.updates, replay_errors=replay_errors, episodes=len(episodes),
            samples=sum(len(e['obs']) for e in episodes), bptt='full_episode_from_reset')
