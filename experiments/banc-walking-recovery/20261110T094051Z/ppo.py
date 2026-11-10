"""Feedforward PPO with transactional KL acceptance and complete reproducible state."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
import math
import os
import random
import tempfile
import numpy as np
import torch
from models import latent_log_prob, gaussian_kl


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = math.exp(-0.001 / 1.0)
    gae_lambda: float = math.exp(-0.001 / 0.2)
    clip: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    epochs: int = 4
    minibatch: int = 64
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    backtracks: int = 6
    normalize_advantage: bool = True


def gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Explicit successor values are from final observations, never reset observations.

    A true terminal zeroes bootstrap; truncation bootstraps but cuts the trace.
    A rollout boundary also cuts the trace by initializing carry to zero.
    Finite task deadlines should be marked terminated if the task defines zero value.
    """
    r, v, nv = [np.asarray(a, dtype=np.float64) for a in (rewards, values, next_values)]
    term, trunc = [np.asarray(a, dtype=bool) for a in (terminated, truncated)]
    if r.ndim != 1 or any(a.shape != r.shape for a in (v, nv, term, trunc)):
        raise ValueError('GAE expects matching 1D arrays')
    if not all(np.isfinite(a).all() for a in (r, v, nv)):
        raise ValueError('nonfinite GAE input')
    adv = np.empty_like(r)
    carry = 0.0
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + gamma * (0 if term[t] else nv[t]) - v[t]
        carry = delta + gamma * gae_lambda * (not (term[t] or trunc[t])) * carry
        adv[t] = carry
    return adv, adv + v


class RolloutBuffer:
    """One environment, chronological transitions. End episodes with explicit flags."""
    keys = ('obs', 'critic_obs', 'latent', 'log_prob', 'mean', 'logstd', 'value',
            'reward', 'terminated', 'truncated', 'next_value')

    def __init__(self):
        self.rows = []

    def add(self, obs, critic_obs, latent, log_prob, mean, logstd, value,
            reward, terminated, truncated, next_value):
        row = dict(zip(self.keys, (obs, critic_obs, latent, log_prob, mean, logstd,
                                  value, reward, terminated, truncated, next_value)))
        def detached(a):
            if torch.is_tensor(a):
                return a.detach().cpu().numpy().copy()
            return np.array(a, copy=True)
        self.rows.append({k: detached(v) for k, v in row.items()})

    def finish(self, gamma, gae_lambda):
        if not self.rows:
            raise ValueError('empty rollout')
        batch = {k: np.stack([row[k] for row in self.rows]) for k in self.keys}
        for k in ('log_prob', 'value', 'reward', 'terminated', 'truncated', 'next_value'):
            batch[k] = batch[k].reshape(len(self.rows))
        for k in ('obs', 'critic_obs', 'latent', 'mean', 'logstd'):
            batch[k] = batch[k].reshape(len(self.rows), -1)
        batch['advantages'], batch['returns'] = gae(batch['reward'], batch['value'],
            batch['next_value'], batch['terminated'], batch['truncated'], gamma, gae_lambda)
        return batch


def ppo_loss(model, batch, config):
    """Actual scalar training loss; no additional averaging in any gradient path."""
    mean = model.actor(batch['obs'])
    logstd = model.logstd()
    logp = latent_log_prob(batch['latent'], mean, logstd)
    ratio = torch.exp(logp - batch['log_prob'])
    surrogate = torch.minimum(ratio * batch['advantages'],
                              ratio.clamp(1 - config.clip, 1 + config.clip) * batch['advantages'])
    entropy = (logstd + 0.5 * math.log(2 * math.pi * math.e)).sum()
    actor = -surrogate.mean() - config.entropy_coef * entropy
    value = model.value(batch['critic_obs'])
    critic = config.value_coef * (value - batch['returns']).square().mean()
    return actor + critic, dict(actor_loss=actor, critic_loss=critic, entropy=entropy,
        clip_fraction=((ratio - 1).abs() > config.clip).to(mean.dtype).mean(),
        ratio_mean=ratio.mean())


class PPOTrainer:
    """Both actor AND critic (including Adam counters) roll back on rejection.

    A retry uses the same samples and gradients, halves both learning rates, and
    advances no counters. Accepted backtracked rates persist. Exhaustion restores
    the original model and optimizers exactly, including the original rates.
    """
    def __init__(self, model, config=None):
        if model.architecture == 'gru238':
            raise NotImplementedError('recurrent rollout/burn-in/reset-replay KL not integrated')
        self.model = model
        self.config = config or PPOConfig()
        c = self.config
        if c.minibatch < 1 or c.epochs < 1 or c.backtracks < 0 or c.target_kl < 0:
            raise ValueError('invalid PPO configuration')
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

    def tensor_batch(self, batch):
        p = self.model.raw_logstd
        return {k: torch.as_tensor(v, dtype=p.dtype, device=p.device).detach() for k, v in batch.items()}

    @torch.no_grad()
    def rollout_kl(self, batch):
        return gaussian_kl(batch['mean'], batch['logstd'],
            self.model.actor(batch['obs']), self.model.logstd()).mean().item()

    def update(self, batch):
        c = self.config
        b = self.tensor_batch(batch)
        n = len(b['obs'])
        if not n or any(len(v) != n for v in b.values()):
            raise ValueError('empty or inconsistent batch')
        if not all(torch.isfinite(v).all() for v in b.values()):
            raise ValueError('nonfinite batch')
        initial_kl = self.rollout_kl(b)
        if not math.isfinite(initial_kl) or initial_kl > c.target_kl:
            raise ValueError('initial rollout KL exceeds budget')
        if c.normalize_advantage:
            a = b['advantages']
            b['advantages'] = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
        rows = []
        stopped = False
        for epoch in range(c.epochs):
            permutation = torch.randperm(n, device=b['obs'].device)
            for start in range(0, n, c.minibatch):
                indices = permutation[start:start + c.minibatch]
                mb = {k: v[indices] for k, v in b.items()}
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss, metrics = ppo_loss(self.model, mb, c)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite PPO loss')
                loss.backward()
                an = torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(), c.max_grad_norm)
                vn = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), c.max_grad_norm)
                if not torch.isfinite(an) or not torch.isfinite(vn):
                    raise FloatingPointError('nonfinite gradient')
                snap = self.snapshot()
                accepted = False
                for attempt in range(c.backtracks + 1):
                    self.restore(snap)
                    for opt in (self.actor_optimizer, self.critic_optimizer):
                        for group in opt.param_groups:
                            group['lr'] *= 0.5 ** attempt
                        opt.step()
                    candidate_kl = self.rollout_kl(b)
                    if math.isfinite(candidate_kl) and candidate_kl <= c.target_kl:
                        accepted = True
                        self.accepted_steps += 1
                        break
                if not accepted:
                    self.restore(snap)
                    stopped = True
                row = {k: float(v.detach()) for k, v in metrics.items()}
                row.update(epoch=epoch, accepted=accepted, attempts=attempt + 1,
                    candidate_kl=candidate_kl, kl=self.rollout_kl(b),
                    actor_grad_before=float(an), critic_grad_before=float(vn),
                    actor_grad_after=min(float(an), c.max_grad_norm),
                    critic_grad_after=min(float(vn), c.max_grad_norm))
                rows.append(row)
                if stopped:
                    break
            if stopped:
                break
        # Update counter means successful optimization calls, not attempted calls.
        if any(r['accepted'] for r in rows):
            self.updates += 1
        with torch.no_grad():
            predicted = self.model.value(b['critic_obs'])
            variance = b['returns'].var(unbiased=False)
            ev = float(1 - (b['returns'] - predicted).var(unbiased=False) / variance) if variance > 0 else None
        return dict(steps=rows, stopped=stopped, kl=self.rollout_kl(b), kl_per_dim=self.rollout_kl(b) / 42,
            explained_variance=ev, std_min=float(self.model.logstd().detach().exp().min()),
            std_max=float(self.model.logstd().detach().exp().max()),
            saturation_fraction=float((b['latent'].tanh().abs() > 0.99).to(b['obs'].dtype).mean()),
            actor_lr=self.actor_optimizer.param_groups[0]['lr'],
            critic_lr=self.critic_optimizer.param_groups[0]['lr'],
            accepted_steps=self.accepted_steps, updates=self.updates)


def save_checkpoint(path, trainer, identity, extra=None):
    """Atomic replace on the destination filesystem; full state, no unsafe partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, model_identity=trainer.model.identity(),
        config=asdict(trainer.config), identity=deepcopy(identity), state=trainer.snapshot(),
        rng=dict(numpy=np.random.get_state(), python=random.getstate(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        extra=deepcopy(extra))
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, trainer, identity, restore_rng=True):
    """Load trusted local checkpoints only (contains Python/numpy RNG objects)."""
    payload = torch.load(path, map_location=trainer.model.raw_logstd.device, weights_only=False)
    if payload['format_version'] != 1 or payload['identity'] != identity:
        raise ValueError('checkpoint input/configuration identity mismatch')
    if payload['model_identity'] != trainer.model.identity() or payload['config'] != asdict(trainer.config):
        raise ValueError('checkpoint architecture or PPO configuration mismatch')
    trainer.restore(payload['state'])
    if restore_rng:
        np.random.set_state(payload['rng']['numpy'])
        random.setstate(payload['rng']['python'])
        torch.set_rng_state(payload['rng']['torch'].cpu())
        if payload['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(payload['rng']['cuda'])
    return payload['extra']
