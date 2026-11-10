"""Short deterministic physics regression; no training, no history writes.

Run with NEW/env/bin/python -B NEW/check_physics.py. JSON prints to stdout.
Optional --output is explicit; recorder uses an automatically removed temp dir.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import numpy as np
import mujoco as mj
import physics as P

OLD = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261109T202355Z')

def load_old():
    # Old entry imports historical frame_audit/repair_core. Keep only the temporary
    # import-path mutation; default dependencies injected from snapshot for pairing.
    saved = list(sys.path)
    try:
        sys.path.insert(0, str(OLD))
        spec = importlib.util.spec_from_file_location('_physics_regression_old_core_v2', OLD / 'core_v2.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved


def run(ticks=12):
    if not 10 <= ticks <= 20:
        raise ValueError('Short regression ticks must be 10..20')
    started = time.perf_counter()
    old_module = load_old()
    old = old_module.Core(act_mod=P.act_mod, sensory=P.sensory)
    new = P.Core()
    assert np.array_equal(old.d.qpos, new.d.qpos)
    assert np.array_equal(old.d.qvel, new.d.qvel)
    qpos0, qvel0 = new.d.qpos.copy(), new.d.qvel.copy()
    errors = dict(qpos=0., qvel=0., loads=0., padz=0., obs495=0., target=0.)
    def compare_obs():
        expected = old.obs()
        actual = new.obs()
        errors['obs495'] = max(errors['obs495'], float(np.max(np.abs(expected - actual))))
        saved = (new.d.qpos.copy(), new.d.qvel.copy(), new.buf.copy(),
                 new._padz_prev.copy(), new._pad_rate.copy(), new.actuator.target_rad.copy(), new.d.time)
        assert np.array_equal(actual, new.obs())
        assert np.array_equal(actual, new.obs495())
        for a, b in zip(saved[:-1], (new.d.qpos, new.d.qvel, new.buf, new._padz_prev,
                                    new._pad_rate, new.actuator.target_rad)):
            assert np.array_equal(a, b)
        assert saved[-1] == new.d.time
    compare_obs()
    actions = np.random.default_rng(20261110).uniform(-.4, .4, (ticks, P.NJ))
    for action in actions:
        a, b = old.tick(action), new.tick(action)
        for key, x, y in [('qpos', old.d.qpos, new.d.qpos), ('qvel', old.d.qvel, new.d.qvel),
                           ('loads', a['loads'], b['loads']), ('padz', a['padz'], b['padz']),
                           ('target', old.actuator.target_rad, new.actuator.target_rad)]:
            errors[key] = max(errors[key], float(np.max(np.abs(x - y))))
        compare_obs()
    assert max(errors.values()) <= 1e-11, errors
    assert new.obs238(np.zeros(P.NJ), .1).shape == (238,)
    assert new.critic_extras().shape == (16,)
    # Reset clears derivative; compatibility with exactly one old read per tick.
    new.reset_from(qpos0, qvel0)
    assert np.array_equal(new._pad_rate, np.zeros(P.NL))
    old.reset_from(qpos0, qvel0)
    assert np.array_equal(new.obs(), old.obs())
    # Same action held 5ms must integrate precisely five 1ms command updates.
    rate_action = np.linspace(-.2, .2, P.NJ)
    target0 = new.actuator.target_rad.copy()
    for _ in range(5):
        old.tick(rate_action)
    new.tick_policy(rate_action, 5)
    assert np.array_equal(old.actuator.target_rad, new.actuator.target_rad)
    assert np.array_equal(old.d.qpos, new.d.qpos)
    target_error = float(np.max(np.abs(new.actuator.target_rad -
        np.clip(target0 + 5 * P.PDT * P.MAX_RATE * rate_action,
                new.actuator.target_bounds_rad[:, 0], new.actuator.target_bounds_rad[:, 1]))))
    assert target_error < 1e-14
    assert new.actuator.command_ticks == 5
    # Prove COM-reference wrench matches mj_applyFT generalized-force mapping.
    new.apply_adhesion(new.measure()['loads'])
    direct, via_wrench = np.zeros(new.plant.nv), np.zeros(new.plant.nv)
    for body in range(1, new.plant.nbody):
        force, torque = new.d.xfrc_applied[body, :3], new.d.xfrc_applied[body, 3:]
        mj.mj_applyFT(new.plant, new.d, force, torque, new.d.xipos[body], body, direct)
        jp, jr = np.zeros((3, new.plant.nv)), np.zeros((3, new.plant.nv))
        mj.mj_jacBodyCom(new.plant, new.d, jp, jr, body)
        via_wrench += jp.T @ force + jr.T @ torque
    wrench_error = float(np.max(np.abs(direct - via_wrench)))
    assert wrench_error <= 1e-12
    # Recorder chunks smaller than 80 force multi-file readback and bounded rows.
    new.reset()
    rec = []
    with tempfile.TemporaryDirectory(prefix='physics-check-') as folder:
        recorder = P.NativeRecorder(folder, chunk_size=31)
        new.recorder = recorder
        for action in actions[:2]:
            m = new.tick(action)
            rec.append((float(new.d.time), new.d.subtree_com[new.th].copy(),
                        m['loads'].copy(), float(m['nonfoot']), m['speed'].copy()))
            assert len(recorder.rows) < 31
        native_summary = recorder.close()
        new.recorder = None
        before_times, after_times = [], []
        residuals, hashes = [], []
        count = 0
        readback_keys = None
        shapes = {}
        for chunk in native_summary['chunks']:
            path = Path(chunk['path'])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == chunk['sha256']
            hashes.append(digest)
            with np.load(path, allow_pickle=False) as z:
                rows = chunk['rows']
                assert z['qpos_after'].shape == (rows, 49)
                assert z['qvel_after'].shape == (rows, 48)
                assert z['loads_native'].shape == (rows, 6)
                assert z['action'].shape == (rows, 42)
                assert z['com_after'].shape == (rows, 3)
                assert z['contact_offsets'].shape == (rows + 1,)
                assert z['contact_offsets'][-1] == len(z['contact_position_world'])
                assert z['adhesion_application_offsets'].shape == (rows + 1,)
                assert z['adhesion_application_offsets'][-1] == len(
                    z['adhesion_application_body_leg_point_force_native'])
                # Recover every body COM wrench from explicit adhesion application points.
                for row in range(rows):
                    a0, a1 = z['adhesion_application_offsets'][row:row + 2]
                    reconstructed = np.zeros_like(z['adhesion_body_wrench_native'][row])
                    for application in z['adhesion_application_body_leg_point_force_native'][a0:a1]:
                        body = int(application[0])
                        point, force = application[2:5], application[5:8]
                        reconstructed[body, :3] += force
                        reconstructed[body, 3:] += np.cross(
                            point - z['adhesion_reference_body_com'][row, body], force)
                    # Wrench is held for40 steps. Point/COM references are from its
                    # refresh time, and are explicitly distinct from current geometry.
                    assert np.allclose(reconstructed, z['adhesion_body_wrench_native'][row],
                                       rtol=0, atol=1e-12)
                for key in z.files:
                    assert np.isfinite(z[key]).all(), key
                before_times.extend(z['time_before_s'])
                after_times.extend(z['time_after_s'])
                residuals.extend(z['force_balance_residual_native'][:, 2] / P.W)
                count += rows
                readback_keys = sorted(z.files)
                shapes = {key: list(z[key].shape[1:]) for key in z.files
                          if key != 'contact_offsets'}
        before_times, after_times = np.asarray(before_times), np.asarray(after_times)
        assert count == 160 == native_summary['native_samples']
        assert len(hashes) == 6
        assert np.allclose(after_times - before_times, P.NATIVE_DT, rtol=0, atol=1e-15)
        assert np.allclose(np.diff(after_times), P.NATIVE_DT, rtol=0, atol=1e-15)
        assert np.array_equal(after_times[:-1], before_times[1:])
        report_summary = P.summarise(rec, 160, initial_heading=new.initial_heading,
                                     native_summary=native_summary)
        assert all(v['status'] == 'UNKNOWN' for v in report_summary['roadmap_D1_D8'].values())
        # Backward / transverse examples preserve old signed-X gate and XY denominator.
        synthetic = [(0., np.array([0., 0., 1.3]), np.full(6, P.W/6), 0., np.full(6, .02)),
                     (1., np.array([-.1, .2, 1.3]), np.full(6, P.W/6), 0., np.full(6, .02))]
        s = P.summarise(synthetic, 80, initial_heading=[0., 1.])
        assert not s['gates']['forward']
        assert np.isclose(s['slip_ratio_max'], .02 / np.hypot(.1, .2))
        assert np.isclose(s['initial_heading_diagnostics']['forward_signed_native_per_s'], .2)
        # Preserve public historical summary bit-for-bit, excluding added diagnostics.
        # Exact independent reference equations, no further historical import needed.
        assert report_summary['forward_speed_native_per_s'] == float(
            np.linalg.norm((rec[-1][1] - rec[0][1])[:2]) / (rec[-1][0] - rec[0][0]))
        native_report = dict(samples=count, chunks=len(hashes), sha256=hashes,
                             time_start_s=float(before_times[0]), time_end_s=float(after_times[-1]),
                             per_step_shapes=shapes, keys=readback_keys,
                             force_balance_abs_z_residual_p95_over_body_weight=float(
                                 np.quantile(np.abs(residuals), .95)),
                             force_balance_is_diagnostic_only=True,
                             nonfoot_force_peak_native=native_summary['nonfoot_force_peak_native'])
    source_paths = [Path(P.__file__).resolve(), Path(__file__).resolve(), *sorted(P.SOURCES.glob('*.py'))]
    return dict(status='PASS', all_passed=True,
                source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                tests=['old_core_v2_pair_deterministic', 'obs495_single_call_compatibility',
                'obs495_repeated_reads_idempotent', 'reset_zero_pad_derivative',
                'obs238_and_privileged16_shapes', '1ms_vs_5ms_same_rate_integration',
                'adhesion_COM_generalized_wrench', 'native_chunks_readback_time_shapes_hash',
                'legacy_XY_slip_denominator_and_world_X_gate', 'initial_heading_diagnostics',
                'missing_original_acceptance_UNKNOWN'], ticks=ticks,
                deterministic_max_abs_errors=errors, target_integration_max_abs_error=target_error,
                adhesion_wrench_max_abs_error=wrench_error, recorder=native_report,
                elapsed_s=time.perf_counter()-started,
                note='Short engineering regression only; no training or walking acceptance.',
                identities=dict(physics=str(Path(P.__file__).resolve()),
                    physics_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
                    snapshot_sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(P.SOURCES.glob('*.py'))},
                    PAD_XML=str(P.PAD_XML), POSE_FILE=str(P.POSE_FILE),
                    mujoco=mj.__version__, numpy=np.__version__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticks', type=int, default=12)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.ticks)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text + '\n')
    print(text)
