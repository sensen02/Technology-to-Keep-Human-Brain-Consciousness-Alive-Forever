"""Neural-only experimental entry. No CPG, posture servo or target controller.
Compiled zero-floor derivative, never edits canonical XML or frozen pose.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
import run_neural_muscle_closure as loop
from engine.embodied.declared_pose_projection import project_scalar_ranges
from engine.embodied.source_snapshot import build_source_snapshot
from engine.embodied.closure_clock import grid_steps

OMIT = ('hairplate_angle', 'campaniform_strain', 'bristle_touch')


def validate_duration(value):
    grid_steps(value, loop.DT_COMMAND)
    if value < .02 or value > 1.0:
        raise ValueError('duration must be 0.02..1 seconds on the command clock')
    return value


def engineering_checks(arrays, model, manifest):
    """Engineering interventions, not biological/locomotion success criteria."""
    motors = sorted({i for row in manifest['rows'] for i in row['local_indices']})
    controls = arrays['normal_controls']
    return dict(
        zero_floor=bool(np.all(model.actuator_ctrlrange[:, 0] == 0)),
        finite_traces=all(bool(np.isfinite(a).all()) for a in arrays.values()),
        silence_zero_commands=not bool(arrays['output_silence_controls'].any()),
        sensory_cut_zero_commands=not bool(arrays['sensory_cut_controls'].any()),
        motor_cut_zero_after_15ms=not bool(arrays['motor_cut_controls'][3:].any()),
        neural_commands_present=bool(controls.any()),
        motor_spikes_present=bool(arrays['normal_spike_counts'][:, motors].any()),
        repeat_exact=all(np.array_equal(arrays['normal_'+kind], arrays['normal_repeat_'+kind])
                         for kind in ('trajectory', 'controls', 'spike_counts')),
        replay_exact=np.array_equal(arrays['normal_trajectory'], arrays['replay_baseline_trajectory']),
        canonical_xml_unchanged=loop.cp.sha256_file(loop.cp.CANONICAL_XML) == loop.cp.CANONICAL_XML_SHA256)


def feedback_measurements(arrays):
    """A changed command under matched mechanical push isolates feedback from replay.

    Nonzero differences are observations, not evidence of improved balance.
    """
    def difference(a, b):
        return float(np.max(np.abs(np.asarray(a, float)-np.asarray(b, float))))
    return dict(
        stimulus_changes_motor_spikes=not np.array_equal(arrays['normal_spike_counts'], arrays['feedback_push_spike_counts']),
        stimulus_command_max_difference=difference(arrays['normal_controls'], arrays['feedback_push_controls']),
        feedback_vs_replay_command_max_difference=difference(arrays['feedback_push_controls'], arrays['replay_push_controls']),
        feedback_vs_replay_body_max_difference=difference(arrays['feedback_push_trajectory'], arrays['replay_push_trajectory']))


def source_hashes():
    # Freeze before experiments, not after: mutation during execution must be detected.
    import sys
    root = Path(__file__).resolve().parent
    sources = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.suffix == '.py' and path.is_relative_to(root) and not any(
                    part in ('venv', 'venv_body', '.venv', 'vendor') for part in path.parts):
                sources[str(path)] = loop.cp.sha256_file(path)
    sources[str(loop.r39.ROUND37_POSE)] = loop.cp.sha256_file(loop.r39.ROUND37_POSE)
    return sources


def input_hashes():
    """Explicit model/network inputs; large assets stay external, hash-pinned.

    The source ZIP is not a complete environment/data archive.
    """
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    xml = loop.cp.CANONICAL_XML.resolve()
    paths = {xml, loop.POOLS.resolve(), loop.LOOKUP.resolve(),
             (Path(__file__).resolve().parent/'data/flywire/banc_connectome.npz').resolve()}
    # Use the exact loader's resolved local/cache paths, not XML-relative guesses.
    spec = _load_mjcf(str(xml))
    for asset in list(spec.meshes) + list(spec.textures):
        if asset.file:
            paths.add(Path(asset.file).resolve())
    return {str(path): loop.cp.sha256_file(path) for path in sorted(paths)}


def compact_evidence(evidence, groups):
    """Retain all declared sensory cells and explicitly prove omitted input is zero."""
    current = evidence['sensory_current_nA']
    indices = np.array(sorted({int(i) for row in groups.values()
                               for ids in row.values() for i in ids}), dtype=np.int64)
    outside = np.ones(current.shape[1], dtype=bool)
    outside[indices] = False
    outside_max = (np.max(np.abs(current[:, outside]), axis=1) if outside.any()
                   else np.zeros(current.shape[0]))
    if np.any(outside_max != 0):
        raise ValueError('Undeclared sensory input outside retained indices; refusing compact evidence')
    arrays = {key: evidence[key] for key in (
        'interval_start_s', 'interval_end_s', 'applied_source_interval',
        'decoded_next_controls', 'activation')}
    arrays['sensory_current_nA'] = current[:, indices].copy()
    arrays['outside_sensory_input_max_abs_nA'] = outside_max
    return indices, arrays


def environment_snapshot():
    """Version inventory, not a claim that dependencies were archived."""
    import importlib.metadata
    import platform
    import sys
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={dist.metadata['Name']: dist.version
                          for dist in importlib.metadata.distributions() if dist.metadata['Name']},
                thread_environment={key: os.environ.get(key) for key in
                                    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MUJOCO_GL')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=.05)
    args = parser.parse_args()
    validate_duration(args.duration)
    import mujoco
    with loop.mg.guarded_run('neural_muscle_closure', **loop.mg.BUDGETS['neural_muscle_closure']) as guard:
        inputs = input_hashes()
        model, historical, _, joints, geoms = loop.prepare_body()
        pose, projection = project_scalar_ranges(model, historical)
        data = mujoco.MjData(model)
        pose[2] = loop.cp.seat_root(model, data, pose)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        gate = loop.cp.stance_facts(model, data).as_dict()
        # Explicit derivative removes residual non-neural floor, not a posture controller.
        native_ranges = model.actuator_ctrlrange.copy()
        model.actuator_ctrlrange[:, 0] = 0
        if np.any(model.actuator_ctrlrange[:, 1] < 0):
            raise ValueError('body cannot accept nonnegative muscle activation')
        built = loop.build_neural_tier(loop.TierConfig(background_mean_nA=0, background_sd_nA=0))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        manifest = loop.build_motor_manifest(json.loads(loop.POOLS.read_text()), built['tier']['root_ids'], names)
        groups, _ = loop.channel_index_map(int(built['n']), built['tier']['root_ids'])
        weight = float(model.body_mass.sum()*np.linalg.norm(model.opt.gravity))
        push = dict(start_s=.001, end_s=.002, force_model=[.1*weight, 0., 0.])
        summaries, arrays = {}, {}
        sources = source_hashes()
        environment = environment_snapshot()
        normal_commands = None
        for name, arm, perturbation in (
                ('normal', 'normal', None), ('normal_repeat', 'normal', None),
                ('replay_baseline', 'replay', None), ('feedback_push', 'normal', push),
                ('replay_push', 'replay', push), ('sensory_cut', 'sensory_cut', push),
                ('motor_cut', 'motor_cut', push), ('output_silence', 'spike_output_silence', push)):
            evidence = {}
            summary, trajectory, commands, spikes = loop.episode(
                model, pose, joints, geoms, built, manifest, groups, arm, args.duration,
                replay=normal_commands, push=perturbation, external_touch=False,
                disabled_channels=OMIT, evidence=evidence)
            indices, captured = compact_evidence(evidence, groups)
            arrays['sensory_local_indices'] = indices
            arrays.update({name+'_'+key: value for key, value in captured.items()})
            summaries[name] = summary
            arrays[name+'_trajectory'] = trajectory
            arrays[name+'_controls'] = commands
            arrays[name+'_spike_counts'] = spikes
            if name == 'normal':
                normal_commands = commands
            print(name, 'escape_s=', summary['escape_time_s'], flush=True)
        checks = engineering_checks(arrays, model, manifest)
        feedback = feedback_measurements(arrays)
        dest = Path(__file__).resolve().parent/'outputs/experiment_platform'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        dest.mkdir(parents=True)
        sources.update({str(path.resolve()): inputs[str(path.resolve())] for path in
                        (loop.cp.CANONICAL_XML, loop.POOLS, loop.LOOKUP)})
        build_source_snapshot(sources, dest/'source_snapshot.zip')
        # A result is publishable only if its recorded implementation stayed frozen.
        if any(loop.cp.sha256_file(path) != digest for path, digest in {**sources, **inputs}.items()):
            raise RuntimeError('Source changed during execution; refusing result publication')
        np.savez_compressed(dest/'traces.npz', **arrays, projected_pose=pose, native_control_ranges=native_ranges)
        result = dict(execution_status='completed' if all(checks.values()) else 'engineering_checks_failed',
            engineering_checks=checks, feedback_measurements=feedback,
            autonomous_goal_status='NOT_ACHIEVED', biological_mapping_verified=False,
            controller_policy='No CPG/PD/target error/scheduled motor command. Neural pool rate translation only; replay is a labelled control arm.',
            branch='Compiled zero control floor + scalar-range projected initial pose',
            sensory_policy='Body claw/hook only, unknown channels disabled, zero background, no artificial tactile current',
            duration_s=args.duration, projection=projection, initial_contact_gate=gate,
            perturbation=push, episodes=summaries, motor_manifest=manifest,
            tier_root_ids=list(map(str, built['tier']['root_ids'])), sensory_groups=groups,
            input_hashes=inputs,
            traces_sha256=loop.cp.sha256_file(dest/'traces.npz'),
            environment_snapshot=environment,
            clocks={key: evidence[key] for key in ('dt_body_s', 'dt_neural_s', 'dt_command_s')},
            evidence_schema=dict(version=1, sensory_local_indices_key='sensory_local_indices',
                sensory_current_columns='sorted union of all sensory groups; indices address tier_root_ids',
                input_held_convention=evidence['input_held_convention'],
                activation_convention=evidence['activation_convention'],
                command_convention=evidence['command_convention'],
                body_nq=int(model.nq), body_nv=int(model.nv), body_na=int(model.na), body_nu=int(model.nu),
                motor_translation=loop.asdict(loop.MotorOutputConfig()),
                replay_source_arms={'replay_baseline': 'normal', 'replay_push': 'normal'}),
            source_snapshot_scope='Project Python and small XML/JSON inputs only; network NPZ and mesh assets are external hash-pinned dependencies, not embedded. Environment not archived.',
            source_snapshot_sha256=loop.cp.sha256_file(dest/'source_snapshot.zip'),
            source_hashes=sources, memory_guard=guard('completed'),
            limitations=['Unvalidated cell tuning and muscle recruitment.',
                        'Time-step convergence not established; falls and feedback differences are observations, not successful locomotion.',
                        'Replay receives identical external force but baseline commands; not neural control.',
                        'Initial projection is initialization only; no runtime pose correction.'])
        (dest/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('Completed experimental bundle:', dest)
        if not all(checks.values()):
            raise RuntimeError(f'Engineering checks failed; evidence saved in {dest}: {checks}')


if __name__ == '__main__':
    main()
