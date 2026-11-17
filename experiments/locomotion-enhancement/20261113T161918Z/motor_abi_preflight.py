"""Parent opt-in NN-only ABI probe; stdlib only until explicit execution flag.
No sensory/brain model, Core, simulator, optimizer, or physical integration.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
MOTOR_DIR = PROJECT / 'experiments/food-approach/20261111T161007Z/combined_turning'
MOTOR = MOTOR_DIR / 'COMBINED.pt'
MOTOR_SHA = '789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
TRIAL = PROJECT / 'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/trials/V_empty'
TRACE = TRIAL / 'neural_decisions.npz'
FORBIDDEN = ('mujoco', 'physics', 'student_wave', 'flygym')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def header(path, member):
    """Read NPY metadata through ZIP stream, never deserialize array here."""
    with zipfile.ZipFile(path) as archive, archive.open(member + '.npy') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError('NPY magic')
        version = tuple(stream.read(2))
        if version == (1, 0):
            size = struct.unpack('<H', stream.read(2))[0]
        elif version in ((2, 0), (3, 0)):
            size = struct.unpack('<I', stream.read(4))[0]
        else:
            raise ValueError('Unsupported NPY version')
        if size > 65536:
            raise ValueError('Unexpected header size')
        value = ast.literal_eval(stream.read(size).decode('utf8' if version == (3, 0) else 'latin1').strip())
    return dict(version=list(version), shape=list(value['shape']), descr=value['descr'], fortran_order=value['fortran_order'])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def execute(out):
    if any(name in sys.modules for name in FORBIDDEN):
        raise ValueError('Fresh process without physical backend required')
    if 'model' in sys.modules or 'common' in sys.modules:
        raise ValueError('Cold model/common namespace required')
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('CPU thread environment must be 1')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    prereg = json.loads((ROOT / 'MOTOR_ABI_PREREG.json').read_text())
    if prereg['steps_per_valid_command'] != 32 or prereg['motor_sha256'] != MOTOR_SHA:
        raise ValueError('Prereg identity')
    h = header(TRACE, 'X238')
    if h['shape'] != [3080, 238] or h['fortran_order'] or h['descr'] != '<f8':
        raise ValueError('Actual V_empty X238 header mismatch')
    result_source = json.loads((TRIAL / 'RESULT.json').read_text())
    if result_source['condition'] != 'V_empty' or result_source['checkpoint_sha256'] != MOTOR_SHA or result_source['neural_records'] != 3080:
        raise ValueError('Actual latest V_empty identity')
    if result_source['neural_decisions']['path'] != str(TRACE) or result_source['neural_decisions']['rows'] != 3080:
        raise ValueError('Actual top-level neural_decisions metadata schema')
    if sha(TRACE) != result_source['neural_decisions']['sha256'] or sha(MOTOR) != MOTOR_SHA:
        raise ValueError('Source file digest mismatch')
    initial_paths = [Path(__file__).resolve(), ROOT / 'MOTOR_ABI_PREREG.json', MOTOR_DIR / 'model.py', MOTOR_DIR / 'model_io.py', MOTOR, TRACE, TRIAL / 'RESULT.json']
    frozen = {str(p): sha(p) for p in initial_paths}
    # All heavyweight work is reachable only here, after explicit flag and header gate.
    import random
    import numpy as np
    import torch
    torch.set_num_threads(1)
    rng_python = random.getstate()
    rng_numpy = np.random.get_state()
    rng_torch = torch.get_rng_state().clone()
    sentinel = object()
    aliases = {name: sys.modules.get(name, sentinel) for name in ('model', 'common')}
    paths = list(sys.path)
    checks = {}
    summaries = []
    try:
        sys.path.insert(0, str(MOTOR_DIR))
        arch = module('_locomotion_abi_motor_arch', MOTOR_DIR / 'model.py')
        sys.modules['model'] = arch
        io = module('_locomotion_abi_motor_io', MOTOR_DIR / 'model_io.py')
        motor, payload = io.load_checked(MOTOR, MOTOR_SHA, require_final=True)
        checks['all_parameters_frozen'] = all(not p.requires_grad for p in motor.parameters())
        checks['state256_physical112'] = motor.identity()['state_dims'] == [128, 64, 64] and int(motor.base.physical_mask.bool().sum()) == 112
        for key in ('source_hashes', 'component_source_hashes', 'stream_hashes'):
            for path, digest in payload.get(key, {}).items():
                if sha(path) != digest:
                    raise ValueError('Strict payload source drift: ' + path)
                frozen[path] = digest
        weights = {k: v.detach().clone() for k, v in motor.state_dict().items()}
        with np.load(TRACE, allow_pickle=False) as archive:
            obs = torch.tensor(archive['X238'][:32].copy(), dtype=torch.float64)
        checks['obs32_finite'] = tuple(obs.shape) == (32, 238) and bool(torch.isfinite(obs).all())
        with torch.no_grad():
            for name, turn in (('neutral', 0.), ('left_half', .5), ('right_half', -.5)):
                state = motor.initial_state(1)
                base_state = motor.base.initial_state(1)
                command = torch.tensor([[max(turn, 0.) / 3., turn]], dtype=torch.float64)
                exact_mean = exact_action = exact_base_state = True
                mean_max = action_max = 0.
                for x in obs:
                    base_mean, base_state = motor.base.step(x[None], base_state)
                    mean, state = motor.step(x[None], command, state)
                    exact_base_state = exact_base_state and torch.equal(state[0], base_state)
                    exact_mean = exact_mean and torch.equal(mean, base_mean)
                    exact_action = exact_action and torch.equal(mean.tanh(), base_mean.tanh())
                    mean_max = max(mean_max, float((mean - base_mean).abs().max()))
                    action_max = max(action_max, float((mean.tanh() - base_mean.tanh()).abs().max()))
                checks[name + '_continuous_base_exact32'] = exact_base_state
                if turn == 0.:
                    checks['neutral_zero_residual_mean_exact32'] = exact_mean
                    checks['neutral_zero_residual_action_exact32'] = exact_action
                summaries.append(dict(name=name, steps=32, external_command=command[0].tolist(), effective_RIGHT=[0., min(turn, 0.)], effective_LEFT=[max(turn, 0.) / 3., max(turn, 0.)], base_mean_exact=exact_mean, base_action_exact=exact_action, max_mean_delta=mean_max, max_action_delta=action_max))
            for name, command in (('independent_positive_turn_zero_first', [0., .5]), ('independent_speed_zero_turn', [.1, 0.])):
                state = motor.initial_state(1)
                snapshot = tuple(s.clone() for s in state)
                try:
                    motor.step(obs[0:1], torch.tensor([command], dtype=torch.float64), state)
                except ValueError as exc:
                    checks[name + '_expected_rejected'] = 'coupled f=relu(turn)/3' in str(exc)
                    checks[name + '_no_state_mutation'] = all(torch.equal(a, b) for a, b in zip(snapshot, state))
                    summaries.append(dict(name=name, external_command=command, rejected=True, error=str(exc)))
                else:
                    checks[name + '_expected_rejected'] = False
        checks['all_state_dict_tensors_unchanged'] = all(torch.equal(v, motor.state_dict()[k]) for k, v in weights.items())
        checks['no_physical_backend_imported'] = not any(name in sys.modules for name in FORBIDDEN)
        # Include every actual imported project Python file in post-load freeze recheck.
        for loaded in list(sys.modules.values()):
            filename = getattr(loaded, '__file__', None)
            if filename:
                p = Path(filename).resolve()
                if p.is_relative_to(PROJECT) and p.suffix == '.py':
                    frozen.setdefault(str(p), sha(p))
        checks['source_and_checkpoint_hashes_unchanged'] = all(sha(p) == digest for p, digest in frozen.items())
    finally:
        sys.path[:] = paths
        for name, previous in aliases.items():
            if previous is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
        torch.set_rng_state(rng_torch)
    checks['model_common_aliases_restored'] = all(sys.modules.get(k, sentinel) is v for k, v in aliases.items())
    after_numpy = np.random.get_state()
    checks['rng_restored'] = random.getstate() == rng_python and torch.equal(torch.get_rng_state(), rng_torch) and after_numpy[0] == rng_numpy[0] and np.array_equal(after_numpy[1], rng_numpy[1]) and after_numpy[2:] == rng_numpy[2:]
    result = dict(kind='FROZEN_MOTOR789F_FIXED_OBSERVATION_ABI_DIAGNOSTIC_V1', all_checks_passed=all(checks.values()), checks=checks, header=h, observation_rows='latest actual V_empty X238[:32]; continuous state per valid command; cold state between commands', source_hashes=frozen, cases=summaries, physical_steps=0, optimizer_updates=0, sensory_or_brain_models_loaded=False, network_decisions=96, standalone_base_reference_decisions=96, scope='Motor ABI only. Conditional fixed observations are not new body trajectories, pivot ability, speed control, or neck deployment.')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    if not result['all_checks_passed']:
        raise ValueError('ABI checks failed; evidence retained at ' + str(out))
    print(json.dumps(dict(result=str(out), all_checks_passed=True, physical_steps=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-parent-nn-only', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not args.execute_parent_nn_only:
        parser.error('Explicit parent NN-only execution flag required')
    execute(args.out)


if __name__ == '__main__':
    main()
