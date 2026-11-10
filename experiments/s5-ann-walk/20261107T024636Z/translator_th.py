"""Translator with a short own-target-state history, resolving the phase ambiguity.

Geometric finding that forces this design: the gait loop projected into 42-D actuator
target space is nearly self-intersecting. For 100% of phases there exists a non-adjacent
phase whose joint-space point is close (median distance 0.118) while the required joint
velocity differs by a median 2.88 rad/s (reference mean speed 0.58 rad/s). A memoryless
map from the current target state therefore cannot determine what the gait is doing next,
which is exactly how every memoryless variant failed (full ANN, DAgger, state-only, brain-only).

Adding K previous target states (position + direction of travel along the loop) removes the
ambiguity while keeping the causal contract: at policy tick k only states up to k are used.

Still a single learned network reading the 12,000-cell brain and its own state history;
no phase clock, no reference trajectory, no PD law at deployment.
"""
import json

import numpy as np
import torch
import torch.nn as nn

METADATA_TH = dict(version='walk-translator-th-v1', brain_input=('all12000', 4, 5.0),
                   target_history_frames=20, target_history_span_ms=95.0,
                   output='normalized_joint_rate', output_scale=30.0,
                   notes='causal; history padded by repeating the oldest frame')


class TranslatorTH(nn.Module):
    def __init__(self, hist_frames=20):
        super().__init__()
        self.K = hist_frames
        self.projection = nn.Linear(12000, 32)
        self.temporal = nn.Conv1d(32, 32, kernel_size=4)
        self.hidden = nn.Linear(32 + self.K * 42, 96)
        self.output = nn.Linear(96, 42)

    def forward(self, history, target_history):
        h = torch.tanh(self.projection(history)).transpose(1, 2)
        h = torch.tanh(self.temporal(h)).squeeze(-1)
        z = torch.tanh(self.hidden(torch.cat([h, target_history.reshape(len(h), -1)], -1)))
        return torch.tanh(self.output(z))

    def export(self, path):
        arrays = {n: v.detach().numpy().astype(np.float32) for n, v in self.named_parameters()}
        arrays['metadata_json'] = np.array(json.dumps(METADATA_TH))
        np.savez(path, **arrays)


class NumpyTranslatorTH:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.metadata = json.loads(str(z['metadata_json'].item()))
            self.K = int(self.metadata['target_history_frames'])
            self.w = {k: z[k].copy() for k in z.files if k != 'metadata_json'}

    def __call__(self, history, target_history):
        history = np.asarray(history, np.float32)
        target_history = np.asarray(target_history, np.float32)
        w = self.w
        proj = np.tanh(history @ w['projection.weight'].T + w['projection.bias'])
        conv = np.tanh(np.einsum('btc,oct->bo', proj, w['temporal.weight']) + w['temporal.bias'])
        feat = np.concatenate([conv, target_history.reshape(len(conv), -1)], -1)
        hidden = np.tanh(feat @ w['hidden.weight'].T + w['hidden.bias'])
        return np.tanh(hidden @ w['output.weight'].T + w['output.bias'])

    forward = __call__


class THWrapper:
    """Adapts a history-taking translator to the runtime's (brain_history, current_state) call.

    The runtime calls model(history[None], state[None]) and indexes [0], so this returns the
    batched (1, 42) output, exactly like NumpyFullBrainTranslator does.
    """

    def __init__(self, model, K=20):
        self.model = model
        self.K = K
        self.buf = None
        self.zero_brain = False

    def __call__(self, history, target_state):
        state = np.asarray(target_state, np.float32)
        flat = state.reshape(-1, state.shape[-1])
        if self.buf is None:
            self.buf = [flat.copy() for _ in range(self.K)]
        else:
            self.buf = self.buf[1:] + [flat.copy()]
        th = np.concatenate(self.buf, axis=1).reshape(len(flat), self.K, -1)
        h = np.zeros_like(history) if self.zero_brain else history
        return self.model(h, th)


def parity_check(model, numpy_model, K=20):
    rng = np.random.default_rng(7)
    h = rng.normal(0.2, 0.3, (5, 4, 12000)).astype(np.float32)
    t = rng.normal(0, 0.2, (5, K, 42)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(h), torch.from_numpy(t)).numpy()
    return float(np.max(np.abs(numpy_model(h, t) - expected)))
