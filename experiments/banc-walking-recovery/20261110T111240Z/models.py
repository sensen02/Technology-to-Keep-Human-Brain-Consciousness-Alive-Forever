"""Declared observation-only actors. No clock, phase, or implicit recurrent state."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn


def latent_log_prob(latent, mean, logstd):
    """Joint density of stored pre-tanh samples; PPO ratios use this contract."""
    return (-0.5 * ((latent - mean) * torch.exp(-logstd)).square()
            - logstd - 0.5 * math.log(2 * math.pi)).sum(-1)


def action_log_prob(latent, mean, logstd):
    # Stable log(1-tanh(z)^2), including saturated floating-point tanh.
    jac = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
    return latent_log_prob(latent, mean, logstd) - jac.sum(-1)


def gaussian_kl(old_mean, old_logstd, mean, logstd):
    """KL(old || current), one joint 42-dimensional value per observation."""
    delta = logstd - old_logstd
    return (delta + 0.5 * torch.expm1(-2 * delta)
            + 0.5 * ((old_mean - mean) * torch.exp(-logstd)).square()).sum(-1)


class MLP(nn.Module):
    def __init__(self, inputs, width, outputs):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, width), nn.Linear(width, width),
                                     nn.Linear(width, outputs)])

    def forward(self, x):
        return self.layers[2](torch.tanh(self.layers[1](torch.tanh(self.layers[0](x)))))


class NeuralFeedbackActorFF(MLP):
    """Replacement ANN prior, not a biological connectome or external controller.

    The residual sees physical features only. Each normalized target POSITION
    x[126:168] = Tposition/H = (T-q0)/H (relative to native reference q0)
    drives its output through a trainable inhibitory
    neural connection: latent = physical_MLP - softplus(raw_gain)*Tposition/H.
    At fixed target, q has only the learned MLP gradient, no explicit +gain path.
    The old error branch coupled inhibitory target drive to a positive local q
    gain (about +9.6/rad in the baseline); this position prior removes that path.
    Softplus constrains its sign, not its upper magnitude or physical stability.
    """
    MASK_RANGES = ((42, 84), (126, 168), (195, 237))
    TRANSFORM = 'physical_mlp_plus_trainable_inhibitory_target_v2'

    def __init__(self, halfspan=None):
        super().__init__(238, 128, 42)
        h = torch.as_tensor(np.ones(42) if halfspan is None else halfspan,
                            dtype=torch.float64).clone()
        if h.shape != (42,) or not torch.isfinite(h).all() or (h < .1).any():
            raise ValueError('actor_halfspan requires 42 finite native halfspans >= .1')
        self.register_buffer('initial_halfspan', h)
        mask = torch.ones(238, dtype=torch.float64)
        for start, stop in self.MASK_RANGES:
            mask[start:stop] = 0
        self.register_buffer('physical_mask', mask)
        g = 10 * h
        self.raw_gain = nn.Parameter(g + torch.log(-torch.expm1(-g)))

    def gain(self):
        return torch.nn.functional.softplus(self.raw_gain)

    def forward(self, x):
        return super().forward(x * self.physical_mask) - self.gain() * x[..., 126:168]

    def transform_identity(self):
        return dict(actor_input_transform=self.TRANSFORM,
                    actor_mask_ranges=[list(r) for r in self.MASK_RANGES],
                    inhibitory_gain_parameterization='softplus(raw_gain); all 42 trainable; no upper bound',
                    gain_init_version='native_actuator_halfspan_x10_v1',
                    gain_init_H=self.initial_halfspan.detach().cpu().tolist(),
                    gain_init_tau_s=.05, gain_init_rate_rad_s=2.,
                    initialization_prior='trainable inhibitory -Tposition/H target-position prior; not anatomical',
                    inhibitory_input_range=[126, 168],
                    inhibitory_input='normalized target POSITION Tposition/H; excludes target error',
                    residual_prior='-softplus(raw_gain)*Tposition/H; fixed-T q derivative only from physical MLP; no explicit +gain',
                    residual='238->128 tanh->128 tanh->42; masked channels have zero drive',
                    all_actor_weights_trainable=True)


class GRUActor(nn.Module):
    """Explicit (T,B,D) training API and (B,D) inference API; hidden never cached."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(238, 128)
        self.gru = nn.GRUCell(128, 128)
        self.hidden = nn.Linear(128, 128)
        self.head = nn.Linear(128, 42)

    def initial_state(self, batch_size):
        return self.encoder.weight.new_zeros((batch_size, 128))

    def step(self, obs, state, reset=None):
        if reset is not None:
            state = state * (~reset.bool()).to(state.dtype).unsqueeze(-1)
        state = self.gru(torch.tanh(self.encoder(obs)), state)
        return self.head(torch.tanh(self.hidden(state))), state

    def sequence(self, obs, state, reset=None, valid=None):
        """Reset before tick; padding leaves hidden unchanged. Caller masks loss/burn-in."""
        if obs.ndim != 3:
            raise ValueError('sequence observations require (T,B,238)')
        outputs = []
        for t in range(obs.shape[0]):
            mean, candidate = self.step(obs[t], state, None if reset is None else reset[t])
            if valid is not None:
                state = torch.where(valid[t, :, None].bool(), candidate, state)
            else:
                state = candidate
            outputs.append(mean)
        return torch.stack(outputs), state

    def forward(self, *args, **kwargs):
        raise RuntimeError('GRU requires explicit step(obs,state) or sequence(obs,state)')


class ActorCritic(nn.Module):
    def __init__(self, architecture='ff495', critic_obs_dim=None, dtype=torch.float64,
                 initial_logstd=-0.5, logstd_bounds=(-3.0, 0.3), actor_halfspan=None):
        super().__init__()
        if architecture not in ('ff495', 'ff238', 'gru238'):
            raise ValueError(architecture)
        self.architecture = architecture
        self.obs_dim = 495 if architecture == 'ff495' else 238
        self.critic_obs_dim = self.obs_dim if critic_obs_dim is None else critic_obs_dim
        self.logstd_bounds = tuple(map(float, logstd_bounds))
        if not self.logstd_bounds[0] < self.logstd_bounds[1]:
            raise ValueError('invalid logstd bounds')
        self.actor = (GRUActor() if architecture == 'gru238' else
                      NeuralFeedbackActorFF(actor_halfspan) if architecture == 'ff238' else
                      MLP(self.obs_dim, 128, 42))
        self.critic = MLP(self.critic_obs_dim, 128 if architecture == 'ff495' else 256, 1)
        self.raw_logstd = nn.Parameter(torch.full((42,), float(initial_logstd), dtype=dtype))
        self.to(dtype=dtype)
        head = self.actor.head if architecture == 'gru238' else self.actor.layers[-1]
        with torch.no_grad():
            head.weight.mul_(0.01)
            head.bias.zero_()

    def logstd(self):
        """Hard bounds preserve legacy coordinates. Autograd: zero outside, one inside.

        Exact endpoints are clamp kinks (PyTorch chooses derivative one); callers
        should not expect symmetric finite differences at a nondifferentiable kink.
        """
        return self.raw_logstd.clamp(*self.logstd_bounds)

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.raw_logstd]

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, critic_obs=None, deterministic=False, state=None, reset=None):
        p = self.raw_logstd
        obs = torch.as_tensor(obs, dtype=p.dtype, device=p.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        new_state = None
        if self.architecture == 'gru238':
            if state is None:
                raise ValueError('pass explicit GRU state (initial_state at reset)')
            mean, new_state = self.actor.step(obs, state, reset)
        else:
            mean = self.actor(obs)
        ls = self.logstd().expand_as(mean)
        latent = mean if deterministic else mean + ls.exp() * torch.randn_like(mean)
        cv = obs if critic_obs is None else torch.as_tensor(critic_obs, dtype=p.dtype, device=p.device)
        if cv.ndim == 1:
            cv = cv.unsqueeze(0)
        return dict(action=latent.tanh(), latent=latent, log_prob=latent_log_prob(latent, mean, ls),
                    mean=mean, logstd=ls, value=self.value(cv), state=new_state)

    def identity(self):
        identity = dict(architecture=self.architecture, obs_dim=self.obs_dim,
                        critic_obs_dim=self.critic_obs_dim, logstd_bounds=self.logstd_bounds,
                        dtype=str(self.raw_logstd.dtype))
        if self.architecture == 'ff238':
            identity.update(self.actor.transform_identity())
        return identity

    def import_legacy_npz(self, path):
        """Import original row-major numpy matrices by transpose, including critic."""
        if self.architecture != 'ff495' or self.critic_obs_dim != 495:
            raise ValueError('legacy import requires ff495 and 495-dimensional critic')
        with np.load(Path(path), allow_pickle=False) as data, torch.no_grad():
            pairs = [('w0', self.actor.layers[0].weight, True), ('b0', self.actor.layers[0].bias, False),
                     ('w1', self.actor.layers[1].weight, True), ('b1', self.actor.layers[1].bias, False),
                     ('w2', self.actor.layers[2].weight, True), ('b2', self.actor.layers[2].bias, False),
                     ('vc0', self.critic.layers[0].weight, True), ('vbc0', self.critic.layers[0].bias, False),
                     ('vc1', self.critic.layers[1].weight, True), ('vbc1', self.critic.layers[1].bias, False),
                     ('vc', self.critic.layers[2].weight, True), ('vbc', self.critic.layers[2].bias, False),
                     ('logstd', self.raw_logstd, False)]
            # Validate every field before mutating any parameter.
            values = []
            for key, parameter, transpose in pairs:
                a = np.asarray(data[key])
                a = a.T if transpose else a
                if a.shape != tuple(parameter.shape) or not np.isfinite(a).all():
                    raise ValueError(f'invalid legacy field {key}: {a.shape}')
                values.append(torch.as_tensor(a.copy(), dtype=parameter.dtype, device=parameter.device))
            for (_, parameter, _), value in zip(pairs, values):
                parameter.copy_(value)
        return self
