"""Causal ANN interface for the existing 12,000-cell runtime (not the full bank).

Inputs are already normalized: history is four oldest-to-newest snapshots of
20 ms exponentially filtered firing rates divided by 100 Hz, sampled every
5 ms. target_state is actuator target minus neutral target, in radians, with
no additional scaling; it is not measured body pose. Output is normalized
joint velocity command in [-1, 1]; the caller owns dt/vmax scaling, target
integration, joint limits, and finite torque servo. No simulation or training
runs on import.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn


METADATA = {
    'format_version': 1,
    'model': 'FullBrainTranslator',
    'runtime_cells': 12000,
    'full_connectome_bank': False,
    'history_shape': [4, 12000],
    'history_order': 'oldest_to_newest',
    'history_filter_tau_s': 0.020,
    'history_rate_divisor_hz': 100.0,
    'history_sample_interval_s': 0.005,
    'history_note': 'four snapshots; first-to-last separation 15ms',
    'target_state_size': 42,
    'target_state_definition': 'actuator_target_minus_neutral_target',
    'target_state_units': 'radian',
    'target_state_divisor': 1.0,
    'input_normalization': 'caller_supplies_normalized_inputs',
    'output_definition': 'normalized_joint_velocity_command',
    'output_range': [-1.0, 1.0],
    'output_scaling': 'caller_integrates_dt_times_vmax_times_action',
    'architecture': 'shared_linear12000_32;conv32_32_kernel4;tanh;concat_target42;linear74_64;tanh;linear64_42;tanh',
    'projection_activation': 'none',
    'convolution': 'valid_cross_correlation_no_padding_stride1',
    'bias': True,
    'dtype': 'float32',
}


def _check_shapes(history, target_state):
    if history.ndim != 3 or tuple(history.shape[1:]) != (4, 12000):
        raise ValueError('history must have shape (batch, 4, 12000)')
    if target_state.ndim != 2 or tuple(target_state.shape) != (history.shape[0], 42):
        raise ValueError('target_state must have shape (batch, 42)')


class FullBrainTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12000, 32, bias=True)
        self.temporal = nn.Conv1d(32, 32, kernel_size=4, bias=True)
        self.hidden = nn.Linear(32 + 42, 64, bias=True)
        self.output = nn.Linear(64, 42, bias=True)

    def forward(self, history, target_state):
        _check_shapes(history, target_state)
        projected = self.projection(history).transpose(1, 2)
        temporal = torch.tanh(self.temporal(projected).squeeze(-1))
        hidden = torch.tanh(self.hidden(torch.cat((temporal, target_state), dim=-1)))
        return torch.tanh(self.output(hidden))

    def export(self, path):
        """Write all weights, biases and the complete input/output contract."""
        arrays = {name: value.detach().cpu().numpy().astype(np.float32, copy=True)
                  for name, value in self.state_dict().items()}
        arrays['metadata_json'] = np.asarray(json.dumps(METADATA, sort_keys=True))
        # File handle preserves the exact requested filename (no implicit .npz).
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, **arrays)

    export_npz = export

    def restore_export(self, path):
        exported = NumpyFullBrainTranslator(path)
        self.load_state_dict({name: torch.from_numpy(value.copy())
                              for name, value in exported.weights.items()}, strict=True)
        return self


class NumpyFullBrainTranslator:
    """Batched NumPy inference with the exact same causal input contract."""
    def __init__(self, path):
        shapes = {
            'projection.weight': (32, 12000), 'projection.bias': (32,),
            'temporal.weight': (32, 32, 4), 'temporal.bias': (32,),
            'hidden.weight': (64, 74), 'hidden.bias': (64,),
            'output.weight': (42, 64), 'output.bias': (42,),
        }
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(shapes) | {'metadata_json'}:
                raise ValueError('Unexpected or missing model fields')
            self.metadata = json.loads(str(archive['metadata_json'].item()))
            if self.metadata != METADATA:
                raise ValueError('Unsupported model or normalization metadata')
            self.weights = {}
            for name, shape in shapes.items():
                value = archive[name]
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError('Invalid parameter: ' + name)
                self.weights[name] = value.copy()

    def __call__(self, history, target_state):
        history = np.asarray(history, dtype=np.float32)
        target_state = np.asarray(target_state, dtype=np.float32)
        _check_shapes(history, target_state)
        if not np.isfinite(history).all() or not np.isfinite(target_state).all():
            raise ValueError('Inputs must be finite')
        w = self.weights
        projected = history @ w['projection.weight'].T + w['projection.bias']
        # Torch Conv1d is cross-correlation, with weights [out, in, time].
        temporal = np.tanh(np.einsum('btc,oct->bo', projected,
                                    w['temporal.weight'], optimize=True)
                           + w['temporal.bias'])
        hidden = np.tanh(np.concatenate((temporal, target_state), axis=-1)
                         @ w['hidden.weight'].T + w['hidden.bias'])
        return np.tanh(hidden @ w['output.weight'].T + w['output.bias'])

    forward = __call__


NumpyTranslator = NumpyFullBrainTranslator


def parity_check():
    """Random CPU parity and export round-trip; no physics or optimization."""
    import tempfile
    torch.set_num_threads(1)
    torch.manual_seed(20261107)
    rng = np.random.default_rng(20261107)
    model = FullBrainTranslator().eval()
    max_error = 0.0
    with tempfile.TemporaryDirectory(prefix='ann-translator-parity-') as folder:
        path = Path(folder) / 'model.npz'
        model.export(path)
        numpy_model = NumpyFullBrainTranslator(path)
        restored = FullBrainTranslator().restore_export(path).eval()
        for name, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name]), name
        for batch in (1, 3, 32):
            history = rng.normal(0.2, 0.3, (batch, 4, 12000)).astype(np.float32)
            targets = rng.normal(0.0, 0.2, (batch, 42)).astype(np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(history), torch.from_numpy(targets)).numpy()
            actual = numpy_model(history, targets)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            assert actual.shape == (batch, 42) and np.max(np.abs(actual)) <= 1.0
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return {'passed': True, 'batches': [1, 3, 32],
            'max_absolute_error': max_error,
            'parameters': sum(p.numel() for p in model.parameters()),
            'export_roundtrip_exact': True, 'physics_runs': 0, 'training_steps': 0}


if __name__ == '__main__':
    print(json.dumps(parity_check(), sort_keys=True))
