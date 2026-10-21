"""P2 fixed P1 commands across body clocks; replay diagnostic, not neural control."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
import time
import uuid
from pathlib import Path
import numpy as np
import run_neural_muscle_closure as loop
import run_experiment_platform as platform
from tools_verify_experiment_platform import verify_bundle
from engine.embodied.source_snapshot import build_source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    verified = verify_bundle(args.bundle, check_live_sources=True, check_live_inputs=True)
    source = json.loads((args.bundle / 'result.json').read_text())
    import mujoco
    with loop.mg.guarded_run('p2_body_clock', **loop.mg.BUDGETS['neural_muscle_closure']):
        frozen = platform.source_hashes()
        frozen[str(Path(__file__).resolve())] = loop.cp.sha256_file(__file__)
        with np.load(args.bundle / 'traces.npz', allow_pickle=False) as saved:
            commands = saved['normal_controls'].copy()
            pose = saved['projected_pose'].copy()
            reference = saved['normal_trajectory'].copy()
        duration = source['duration_s']
        command_dt = source['clocks']['dt_command_s']
        arrays = {'commands': commands, 'projected_pose': pose}
        experiments = {}
        for label, dt in (('base', .0001), ('half', .00005), ('quarter', .000025)):
            model, _, _, _, _ = loop.prepare_body()
            model.opt.timestep = dt
            model.actuator_ctrlrange[:, 0] = 0
            data = mujoco.MjData(model)
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            thorax = model.body('Thorax').id
            initial = float(data.xpos[thorax, 2])
            states, heights, times = [], [], []
            escape = None
            steps = loop.grid_steps(command_dt, dt)
            for k, control in enumerate(commands):
                data.ctrl[:] = control
                for _ in range(steps):
                    data.xfrc_applied[:] = 0
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if escape is None and data.xpos[thorax, 2] < .9 * initial:
                        escape = float(data.time)
                state = np.r_[data.qpos, data.qvel, data.act].copy()
                if not np.isfinite(state).all() or any(w.number for w in data.warning):
                    raise RuntimeError('nonfinite body or MuJoCo warning')
                if abs(data.time - (k+1)*command_dt) > 1e-10:
                    raise RuntimeError('body clock drift')
                states.append(state)
                heights.append(float(data.xpos[thorax, 2]))
                times.append(float(data.time))
            arrays[label+'_trajectory'] = np.asarray(states)
            arrays[label+'_height_model'] = np.asarray(heights)
            arrays[label+'_time_s'] = np.asarray(times)
            experiments[label] = dict(body_dt_s=dt, duration_s=duration,
                                     escape_time_s=escape, initial_height_model=initial)
        baseline_exact = bool(np.array_equal(arrays['base_trajectory'], reference))
        if not baseline_exact:
            raise AssertionError('fixed command base body does not match P1 exactly')
        comparisons = {}
        for a, b in (('base', 'half'), ('half', 'quarter'), ('base', 'quarter')):
            height_error = float(np.max(np.abs(arrays[a+'_height_model'] - arrays[b+'_height_model'])))
            ea, eb = experiments[a]['escape_time_s'], experiments[b]['escape_time_s']
            escape_error = None if ea is None or eb is None else abs(ea-eb)
            comparisons[a+'_vs_'+b] = dict(max_height_difference_model=height_error,
                escape_time_difference_s=escape_error,
                declared_height_005_pass=height_error <= .05,
                declared_escape_1ms_pass=escape_error is not None and escape_error <= .001)
        if any(loop.cp.sha256_file(p) != h for p, h in {**frozen, **source['input_hashes']}.items()):
            raise RuntimeError('source/input changed during replay')
        dest = Path(__file__).resolve().parent / 'outputs/p2_body_clock' / (time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        dest.mkdir(parents=True)
        build_source_snapshot(frozen, dest/'source_snapshot.zip')
        np.savez_compressed(dest/'traces.npz', **arrays)
        result = dict(status='COMPLETED_FIXED_COMMAND_DIAGNOSTIC', source_bundle=str(args.bundle.resolve()),
            source_result_sha256=verified['result_sha256'], source_hashes=frozen,
            source_snapshot_sha256=loop.cp.sha256_file(dest/'source_snapshot.zip'),
            traces_sha256=loop.cp.sha256_file(dest/'traces.npz'), experiments=experiments,
            comparisons=comparisons, base_matches_p1_exact=baseline_exact,
            environment=platform.environment_snapshot(),
            limitations=['Fixed normal P1 commands; no neural integration or feedback here.',
                         'Zero-floor derivative and projected initial pose unchanged.',
                         'Height/escape tolerances from existing protocol; not full-state convergence or behavior success.'])
        (dest/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('OUTPUT', str(dest), flush=True)
        print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
