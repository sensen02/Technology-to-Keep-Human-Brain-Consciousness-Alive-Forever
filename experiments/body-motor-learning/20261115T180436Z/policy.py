"""Artificial body-side motor policy. The independent command inputs are bench-only."""
import torch
from torch import nn

# Actuator and pad orders are frozen independent of the six sensory load slots.
LEGS = ("LF", "LH", "LM", "RF", "RH", "RM")
LOAD_LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
JOINT_GROUPS = (0, 42, 84, 126, 195)
FORMAT = "SIX_LEG_COORDINATED_GRU_NO_PREVIOUS_ACTION_V2"


class BodyPolicy(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        joint = torch.arange(42).reshape(6, 7)
        local = [joint + start for start in JOINT_GROUPS]
        local += [torch.tensor([168 + LOAD_LEGS.index(l) for l in LEGS])[:, None]]
        local += [torch.arange(start, start + 6)[:, None] for start in (183, 189)]
        self.register_buffer("leg_indices", torch.cat(local, dim=-1), persistent=False)
        self.register_buffer("leg_identity", torch.eye(6), persistent=False)
        self.encoder = nn.Sequential(nn.Linear(44, 48), nn.Tanh(), nn.Linear(48, 32), nn.Tanh())
        self.leg_memory = nn.GRU(32, 24)
        self.coordinator = nn.GRU(6 * 24 + 9 + 2, 96)
        self.action = nn.Sequential(nn.Linear(24 + 96 + 2 + 6, 64), nn.Tanh(), nn.Linear(64, 7))
        self.target_gain = nn.Parameter(torch.full((6, 7), 3.0))

    def sequence(self, x, command, state=None):
        """Inputs [time, batch, 238] and [time, batch, 2], with explicit recurrent state."""
        t, b, _ = x.shape
        normalized = ((x - self.mean) / self.scale).clamp(-8.0, 8.0)
        # Remove the recorded previous action as a predictor, with identical parameter shapes.
        normalized = torch.cat((normalized[..., :195], torch.zeros_like(normalized[..., 195:237]),
                                normalized[..., 237:]), dim=-1)
        identity = self.leg_identity.expand(t, b, 6, 6)
        legs = self.encoder(torch.cat((normalized[..., self.leg_indices], identity), -1))
        local, next_local = self.leg_memory(legs.reshape(t, b * 6, 32), None if state is None else state[0])
        context, next_context = self.coordinator(
            torch.cat((local.reshape(t, b, 144), normalized[..., 174:183], command), -1),
            None if state is None else state[1],
        )
        shared = context[:, :, None, :].expand(t, b, 6, 96)
        commands = command[:, :, None, :].expand(t, b, 6, 2)
        local = local.reshape(t, b, 6, 24)
        raw = self.action(torch.cat((local, shared, commands, identity), -1))
        # Learned inhibitory feedback from the actuator's current target offset.
        raw = raw - torch.nn.functional.softplus(self.target_gain) * x[..., 126:168].reshape(t, b, 6, 7)
        return raw.tanh().reshape(t, b, 42), (next_local, next_context)

    def step(self, x, command, state=None):
        action, state = self.sequence(x[None, None, :], command[None, None, :], state)
        return action[0, 0], state
