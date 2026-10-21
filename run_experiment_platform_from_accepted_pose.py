"""Export a NEW 100x90 neural command archive from the same experiment-platform pipeline with the ONLY change being the
initial pose: the accepted declared-cone static equilibrium of the corrected/reissued batch.

Same branch and same controller policy as /run/media/sensen/Data2/cell_wound_prototype/run_experiment_platform.py
(compiled zero control floor + neural-muscle closure, no CPG/PD/target error/scheduled motor command); the model, the
gains, the friction, the force scaling and the sensory policy are untouched. Only the initial pose and the initial
contact gate differ, and only the 'normal' arm is exported (declared scope reduction: the derived archive needs the
normal-arm commands).

Manual research implementation, not a product. No standing/walking claim.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

os.environ['MUJOCO_GL'] = 'egl'
import run_experiment_platform as platform  # noqa: E402  (same pipeline, imported, not modified)
import run_neural_muscle_closure as loop  # noqa: E402
from engine.embodied.source_snapshot import build_source_snapshot  # noqa: E402

OMIT = platform.OMIT
ACCEPTED = Path('/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-20261015T161259/'
                'equilibrium_best_state.npz')
sha = loop.cp.sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pose-npz', default=str(ACCEPTED))
    ap.add_argument('--duration', type=float, default=0.5)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    platform.validate_duration(args.duration)

    import mujoco
    pose_npz = Path(args.pose_npz)
    st = np.load(pose_npz)
    pose = st['qpos'].copy()
    act0 = st['act'].copy()
    out = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parent / 'outputs/experiment_platform' /
        (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]))
    out.mkdir(parents=True, exist_ok=False)

    with loop.mg.guarded_run('neural_muscle_closure', **loop.mg.BUDGETS['neural_muscle_closure']) as guard:
        inputs = platform.input_hashes()
        sources = platform.source_hashes()
        model, historical, _, joints, geoms = loop.prepare_body()
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        gate = loop.cp.stance_facts(model, data).as_dict()
        native_ranges = model.actuator_ctrlrange.copy()
        model.actuator_ctrlrange[:, 0] = 0
        if np.any(model.actuator_ctrlrange[:, 1] < 0):
            raise ValueError('body cannot accept nonnegative muscle activation')
        built = loop.build_neural_tier(loop.TierConfig(background_mean_nA=0, background_sd_nA=0))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        manifest = loop.build_motor_manifest(json.loads(loop.POOLS.read_text()), built['tier']['root_ids'], names)
        groups, _ = loop.channel_index_map(int(built['n']), built['tier']['root_ids'])
        evidence = {}
        t0 = time.time()
        summary, trajectory, commands, spikes = loop.episode(
            model, pose, joints, geoms, built, manifest, groups, 'normal', args.duration,
            replay=None, push=None, external_touch=False, disabled_channels=OMIT, evidence=evidence)
        wall = time.time() - t0
        indices, captured = platform.compact_evidence(evidence, groups)
        arrays = dict(sensory_local_indices=indices)
        arrays.update({'normal_' + k: v for k, v in captured.items()})
        arrays['normal_trajectory'] = trajectory
        arrays['normal_controls'] = commands
        arrays['normal_spike_counts'] = spikes

    checks = dict(finite_traces=all(bool(np.isfinite(a).all()) for a in arrays.values()),
                  neural_commands_present=bool(np.asarray(commands).any()),
                  zero_floor=bool(np.all(model.actuator_ctrlrange[:, 0] == 0)),
                  canonical_xml_unchanged=loop.cp.sha256_file(loop.cp.CANONICAL_XML) == loop.cp.CANONICAL_XML_SHA256)
    sources.update({str(p.resolve()): inputs[str(p.resolve())] for p in
                    (loop.cp.CANONICAL_XML, loop.POOLS, loop.LOOKUP)})
    build_source_snapshot(sources, out / 'source_snapshot.zip')
    if any(loop.cp.sha256_file(p) != d for p, d in {**sources, **inputs}.items()):
        raise RuntimeError('Source changed during execution; refusing result publication')
    np.savez_compressed(out / 'traces.npz', **arrays, projected_pose=pose, native_control_ranges=native_ranges)

    result = dict(execution_status='completed' if all(checks.values()) else 'engineering_checks_failed',
                  engineering_checks=checks, autonomous_goal_status='NOT_ACHIEVED', biological_mapping_verified=False,
                  controller_policy=('No CPG/PD/target error/scheduled motor command. Neural pool rate translation only; '
                                     'replay is a labelled control arm.'),
                  branch=('Compiled zero control floor + scalar-range projected initial pose, DERIVED EXPORT whose only change is '
                          'the initial pose (accepted declared-cone static equilibrium)'),
                  same_pipeline_as=str(Path(platform.__file__).resolve()),
                  same_pipeline_sha256=sha(Path(platform.__file__).resolve()),
                  derivation=dict(only_change='initial pose and the resulting initial contact gate',
                                  unchanged=['model and canonical XML', 'controller policy and branch',
                                             'gains (none applied)', 'friction', 'force scaling', 'sensory policy',
                                             'command cadence and duration'],
                                  arm_exported='normal only',
                                  pose_npz=str(pose_npz), pose_npz_sha256=sha(pose_npz),
                                  pose_in_npz_key='qpos'),
                  sensory_policy=('Body claw/hook only, unknown channels disabled, zero background, no artificial tactile current'),
                  duration_s=args.duration, initial_contact_gate=gate, episode=summary, motor_manifest=manifest,
                  tier_root_ids=list(map(str, built['tier']['root_ids'])), sensory_groups=groups,
                  normal_controls_shape=list(np.asarray(commands).shape),
                  wall_seconds=round(wall, 1), guard_indices=indices.shape[0],
                  manual_research_not_product=True, standing=False, walking=False,
                  not_established='absolute force values, physical units, biological validity')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print('export dir:', out)
    print('normal_controls shape:', np.asarray(commands).shape, '| wall %.1f s' % wall)
    print('checks:', checks)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
