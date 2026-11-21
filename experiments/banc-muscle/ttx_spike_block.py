"""Paired, virtual TTX-inspired spike-generation block on the BANC muscle bench.

No drug, concentration, diffusion, channel kinetics, or animal procedure is modeled.
The conductance point neurons have no sodium channel state. At the 500 ms handoff,
an effectively unreachable spike threshold is assigned to the selected root IDs;
subthreshold integration and already emitted delayed events continue normally.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STANCE = HERE / '20261120T081056Z-stance-warmup'
sys.path.insert(0, str(STANCE))
from warm_release import (HANDOFF, POST, SETTLE, support, BodyObservations,
                          ConductanceNetwork, DT_MS, Interfaces, MuscleTeacher,
                          body, load_pose, mj, sha, MODEL)

CHECKPOINT = HERE / '20261120T074805Z-linear-interfaces/verified/PARAMETERS.npz'
POSE = STANCE / 'stance.json'
ARMS = ('control', 'all_neurons', 'sensory_roots', 'motor_roots')
THRESHOLD_OFFSET_MV = 1_000_000.0  # symbolic spike block, not pharmacology


def digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def run_arm(arm, model, obs, io, stance):
    q = np.asarray(stance['qpos'], dtype=float)
    tonic = np.asarray(stance['tonic_activation'], dtype=float)
    net = ConductanceNetwork(len(io.roots), dt_ms=DT_MS)
    net.set_connectivity(*io.connectivity(io.fractions))
    teacher = MuscleTeacher(model, q, tonic)
    data = load_pose(model, q)
    data.act[:] = tonic
    data.ctrl[:] = tonic
    mj.mj_forward(model, data)
    initial_angle = data.qpos[obs.tibia_qpos].copy()
    height = float(data.subtree_com[0, 2])
    drive = np.zeros(net.n)
    trace = np.zeros(len(io.motor))
    neural_command = np.zeros(model.nu)
    block = np.zeros(net.n, dtype=bool)
    if arm == 'all_neurons':
        block[:] = True
    elif arm == 'sensory_roots':
        block[io.sensors] = True
        block[io.feco] = True
    elif arm == 'motor_roots':
        block[io.motor] = True
    elif arm != 'control':
        raise ValueError(arm)
    # Unique roots matter because candidate pools can share a root.
    blocked_roots = io.roots[block].astype(str).tolist()
    record = {k: [] for k in ('time_s', 'qpos', 'qvel', 'activation', 'loads',
                              'torso', 'depth_mm', 'stand', 'six_foot',
                              'applied_command', 'teacher_on')}
    events = {k: [] for k in ('neural_ms', 'sensory_spikes', 'motor_spikes',
                              'all_spikes', 'neural_command', 'input_nA')}
    for tick in range(HANDOFF + POST + 1):
        z = obs.measure(data)
        depth = min((c.dist for c in data.contact if obs.floor in c.geom), default=0.)
        record['time_s'].append(float(data.time))
        for key, value in [('qpos', data.qpos), ('qvel', data.qvel),
                           ('activation', data.act), ('loads', z['ground_load_w']),
                           ('torso', z['torso'])]:
            record[key].append(np.copy(value))
        record['depth_mm'].append(depth)
        record['stand'].append(support(z, depth, height))
        record['six_foot'].append(support(z, depth, height, True))
        if tick in (SETTLE, HANDOFF):
            if not all(record['six_foot'][-1001:]):
                raise RuntimeError(f'preparation lost six-foot support at {tick}')
        if tick == HANDOFF + POST:
            break
        use_teacher = tick < HANDOFF
        if tick == HANDOFF and block.any():
            offset = np.zeros(net.n)
            offset[block] = THRESHOLD_OFFSET_MV
            offset.flags.writeable = False
            net.threshold_offset_mV = offset
        if tick % 5 == 0:
            data.ctrl[:] = teacher.command(data) if use_teacher else neural_command
            drive.fill(0.)
            if tick >= SETTLE:
                drive[io.sensors] = np.clip(z['ground_load_w'][io.feet] * io.load_gain, 0, .12)
                signal = np.where(io.angle,
                                  (z['tibia_angle_rad'] - initial_angle)[io.feco_feet],
                                  z['tibia_velocity_rad_s'][io.feco_feet])
                drive[io.feco] = np.clip(signal * io.feco_gain, 0, .12)
            spiked = net.step(drive)
            if tick >= HANDOFF and np.any(spiked & block):
                raise AssertionError('blocked root emitted a spike')
            trace *= np.exp(-DT_MS / 10.)
            trace += spiked[io.motor]
            neural_command = np.clip(io.motor_gain @ trace, 0, 1)
            for key, value in [('neural_ms', net.t_ms),
                               ('sensory_spikes', int(spiked[io.sensors].sum())),
                               ('motor_spikes', int(spiked[io.motor].sum())),
                               ('all_spikes', int(spiked.sum())),
                               ('neural_command', neural_command),
                               ('input_nA', drive[io.sensors])]:
                events[key].append(np.copy(value))
        record['teacher_on'].append(use_teacher)
        record['applied_command'].append(data.ctrl.copy())
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise AssertionError('external forces were applied')
        mj.mj_step(model, data)
        mj.mj_forward(model, data)
    if any(data.warning.number):
        raise RuntimeError('MuJoCo warning')
    arr = {key: np.asarray(value) for key, value in (record | events).items()}
    if not all(np.isfinite(value).all() for value in arr.values()):
        raise RuntimeError('nonfinite trajectory')
    post = slice(HANDOFF, None)
    neural_post = slice(HANDOFF // 5, None)
    def first(mask):
        found = np.flatnonzero(mask)
        return float(found[0]) * model.opt.timestep * 1000 if found.size else None
    result = {
        'arm': arm, 'blocked_root_count': len(blocked_roots),
        'blocked_roots_sha256': hashlib.sha256('\n'.join(blocked_roots).encode()).hexdigest(),
        'native_steps': HANDOFF + POST, 'neural_ticks': net.steps,
        'preparation_six_foot': bool(arr['six_foot'][SETTLE:HANDOFF + 1].all()),
        'post_teacher_ticks': int(arr['teacher_on'][post].sum()),
        'post_neural_spikes': int(arr['all_spikes'][neural_post].sum()),
        'post_sensory_spikes': int(arr['sensory_spikes'][neural_post].sum()),
        'post_motor_spikes': int(arr['motor_spikes'][neural_post].sum()),
        'first_stance_violation_ms': first(~arr['stand'][post]),
        'first_nonfoot_ms': first(arr['loads'][post, -1] > 0),
        'first_com_drop_10pct_ms': first(arr['torso'][post, 0] < .9 * height),
        'post_entire_stand': bool(arr['stand'][post].all()),
        'post_tail_stand': bool(arr['stand'][-1001:].all()),
        'post_tail_foot_weight': float(arr['loads'][-1001:, :6].sum(axis=1).mean()),
        'post_peak_foot_weight': float(arr['loads'][post, :6].sum(axis=1).max()),
        'post_peak_command': float(arr['applied_command'][post].max()),
        'post_command_integral': float(arr['applied_command'][post].sum() * model.opt.timestep),
        'post_com_z_end_mm': float(arr['torso'][-1, 0]),
    }
    return arr, result


def main(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    stance = json.loads(POSE.read_text())
    model = body(stance['floor_contact_time_ms'])
    if stance['body_xml_sha256'] != sha(MODEL):
        raise AssertionError('body source changed')
    if stance['joint_order'] != [model.joint(j).name for j in range(model.njnt)]:
        raise AssertionError('joint identity changed')
    if stance['actuator_order'] != [model.actuator(j).name for j in range(model.nu)]:
        raise AssertionError('muscle identity changed')
    obs = BodyObservations(model)
    io = Interfaces(model)
    io.restore(CHECKPOINT)
    results, arrays = {}, {}
    for arm in ARMS:
        a, result = run_arm(arm, model, obs, io, stance)
        arrays[arm] = a
        results[arm] = result
        np.savez_compressed(output / f'{arm}.npz', **a)
        print(arm, result['post_neural_spikes'], result['first_nonfoot_ms'],
              result['post_tail_foot_weight'], flush=True)
    prefix_fields = ('qpos', 'qvel', 'activation', 'loads', 'torso', 'depth_mm',
                     'stand', 'six_foot', 'neural_command', 'all_spikes')
    for arm in ARMS[1:]:
        for key in prefix_fields:
            n = HANDOFF + 1 if key in arrays[arm] and len(arrays[arm][key]) == HANDOFF + POST + 1 else HANDOFF // 5
            np.testing.assert_array_equal(arrays['control'][key][:n], arrays[arm][key][:n], err_msg=f'{arm}/{key}')
    with np.load(STANCE / 'results/warm_release.npz', allow_pickle=False) as old:
        for old_key, key in [('qpos', 'qpos'), ('qvel', 'qvel'), ('activation', 'activation'),
                             ('loads', 'loads'), ('torso', 'torso'), ('neural_command', 'neural_command')]:
            np.testing.assert_array_equal(old[old_key], arrays['control'][key], err_msg=f'legacy/{key}')
    result = {
        'identity': {'pose_sha256': sha(POSE), 'checkpoint_sha256': sha(CHECKPOINT),
                     'body_xml_sha256': sha(MODEL), 'network_sha256': sha(HERE / '20261120T061904Z-independent/NETWORK.npz'),
                     'script_sha256': sha(Path(__file__)), 'dt_native_s': model.opt.timestep,
                     'dt_neural_ms': DT_MS, 'handoff_ms': 500, 'post_ms': 250,
                     'threshold_offset_mV': THRESHOLD_OFFSET_MV},
        'model_intervention': 'unreachable spike threshold at handoff for specified point neurons; not dose or channel block calibration',
        'pre_handoff_exact_match': True,
        'control_exact_legacy_replay': True,
        'training_updates': 0,
        'arms': results,
    }
    (output / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.output)
