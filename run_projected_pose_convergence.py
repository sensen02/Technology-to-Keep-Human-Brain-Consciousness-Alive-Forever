"""Predeclared derivative-pose and clock convergence study, not stability tuning.
手工研究原型，不是成品. Frozen pose/XML untouched; compiled zero floor only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_neural_muscle_closure as loop
from engine.embodied.declared_pose_projection import project_scalar_ranges
from engine.embodied.source_snapshot import build_source_snapshot

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/projected_pose_convergence'
OMIT = ('hairplate_angle', 'campaniform_strain', 'bristle_touch')


def main():
    import mujoco
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start = time.monotonic()
    dest = OUT/time.strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True, exist_ok=False)
    model, historical, gate, joints, geoms = loop.prepare_body()
    pose, projection = project_scalar_ranges(model, historical)
    data = mujoco.MjData(model)
    pose[2] = loop.cp.seat_root(model, data, pose)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    projected_gate = loop.cp.stance_facts(model, data).as_dict()
    model.actuator_ctrlrange[:, 0] = 0
    built = loop.build_neural_tier(loop.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    manifest = loop.build_motor_manifest(json.loads(loop.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, _ = loop.channel_index_map(int(built['n']), built['tier']['root_ids'])
    arrays = dict(historical_pose=historical, projected_pose=pose, tier_root_ids=np.asarray(built['tier']['root_ids']))
    episodes = {}
    # Separate body-only and neural-only refinements; command sampling stays fixed.
    clocks = [('base', .0001, .0005), ('body_half', .00005, .0005),
              ('neural_half', .0001, .00025), ('both_half', .00005, .00025),
              ('both_quarter', .000025, .000125)]
    for clock, dt_body, dt_neural in clocks:
        model.opt.timestep = dt_body
        for arm in ('normal', 'sensory_cut', 'motor_cut'):
            name = clock+'_'+arm
            with patch.object(loop, 'DT_BODY', dt_body), patch.object(loop, 'DT_NEURAL', dt_neural):
                summary, trajectory, controls, spikes = loop.episode(model, pose, joints, geoms,
                    built, manifest, groups, arm, .1, external_touch=False, disabled_channels=OMIT)
            summary.update(body_dt_s=dt_body, neural_dt_s=dt_neural,
                motor_spike_total=sum(r['motor_spikes'] for r in summary['records']),
                network_spike_total=int(spikes.sum()))
            episodes[name] = summary
            for key, value in [('trajectory', trajectory), ('controls', controls), ('spike_counts', spikes)]:
                arrays[name+'_'+key] = value
            print(name, 'motor', summary['motor_spike_total'], 'escape', summary['escape_time_s'], flush=True)
    comparisons = {}
    for clock, _, _ in clocks[1:]:
        for arm in ('normal', 'sensory_cut', 'motor_cut'):
            a, b = episodes['base_'+arm], episodes[clock+'_'+arm]
            ah = np.array([r['height_mm_declared'] for r in a['records']])
            bh = np.array([r['height_mm_declared'] for r in b['records']])
            delta = None if a['escape_time_s'] is None or b['escape_time_s'] is None else b['escape_time_s']-a['escape_time_s']
            comparisons[clock+'_'+arm] = dict(height_max_abs_delta_mm=float(abs(ah-bh).max()),
                escape_delta_s=delta, motor_spike_delta=b['motor_spike_total']-a['motor_spike_total'],
                controls_max_abs_delta=float(abs(arrays['base_'+arm+'_controls']-arrays[clock+'_'+arm+'_controls']).max()),
                predeclared_descriptive_tolerance=dict(height_mm=.05, escape_s=.001),
                within_descriptive_tolerance=bool(abs(ah-bh).max() <= .05 and delta is not None and abs(delta) <= .001))
    checks = dict(canonical_xml_unchanged=loop.cp.sha256_file(loop.cp.CANONICAL_XML) == loop.cp.CANONICAL_XML_SHA256,
        projected_scalar_ranges_valid=projection['scalar_limit_report']['within_declared_scalar_ranges'],
        no_body_floor_initial=projected_gate['n_body_floor_contact'] == 0,
        sensory_cuts_silent=all(episodes[c+'_sensory_cut']['network_spike_total'] == 0 for c,_,_ in clocks),
        sensory_cuts_zero_control=all(not arrays[c+'_sensory_cut_controls'].any() for c,_,_ in clocks))
    assert all(checks.values()), checks
    np.savez_compressed(dest/'convergence_traces.npz', **arrays)
    # Actual canonical body keyframes from the base normal trajectory, not a servo surrogate.
    model.opt.timestep = .0001
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0, 1.2]; cam.distance = 9.; cam.azimuth = 130.; cam.elevation = -25.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k in zip(axes, (0, 9, 19)):
        row = arrays['base_normal_trajectory'][k]
        data.qpos[:] = row[:model.nq]
        data.qvel[:] = row[model.nq:model.nq+model.nv]
        data.act[:] = row[model.nq+model.nv:]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        ax.imshow(renderer.render()); ax.set_title(f'{(k+1)*5} ms'); ax.axis('off')
    renderer.close()
    fig.suptitle('Manual prototype: projected initial pose, actual BANC-driven body (not standing)')
    fig.tight_layout(); fig.savefig(dest/'body_keyframes.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, s in episodes.items():
        times = [r['time_end_s']*1000 for r in s['records']]
        axes[0].plot(times, [r['height_mm_declared'] for r in s['records']], label=name)
        axes[1].plot(times, [r['motor_spikes'] for r in s['records']])
    axes[0].set_ylabel('Thorax height (model mm)'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Motor spikes / 5ms'); axes[1].set_xlabel('Time (ms)')
    fig.suptitle('Projected pose and separate clock refinements — manual prototype')
    fig.tight_layout(); fig.savefig(dest/'convergence.png', dpi=150); plt.close(fig)
    sources = {}
    import sys
    for module in list(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and Path(p).resolve().is_relative_to(HERE) and not any(x in ('venv', 'venv_body', 'vendor') for x in Path(p).parts):
            sources[str(Path(p).resolve())] = loop.cp.sha256_file(p)
    sources[str(Path(__file__).resolve())] = loop.cp.sha256_file(__file__)
    build_source_snapshot(sources, dest/'source_snapshot.zip')
    report = dict(label='手工研究原型，不是成品', status='MEASURED_NOT_STANDING_ACCEPTANCE',
        projection=projection, historical_contact_gate=gate, projected_contact_gate=projected_gate,
        clocks=clocks, checks=checks, episodes=episodes, comparisons=comparisons,
        source_hashes=sources, canonical_xml_sha256=loop.cp.sha256_file(loop.cp.CANONICAL_XML),
        historical_pose_source=str(loop.r39.ROUND37_POSE), historical_pose_sha256=loop.cp.sha256_file(loop.r39.ROUND37_POSE),
        limits=['Derivative pose is coordinate projection, not a sourced anatomical stance or equilibrium.',
                'Boundary joint placement still permits native soft-limit forces; does not remove passive physics.',
                'Fixed 5ms command clock unchanged; neural refinement can alter discrete spike thresholds.',
                'Predeclared descriptive tolerances are engineering comparisons, not biological validation.',
                'No gain, polarity or connectivity changes; zero-floor compiled branch explicitly separate.'], wall_s=time.monotonic()-start)
    (dest/'convergence_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('OUTPUT', dest, flush=True)


if __name__ == '__main__':
    with loop.mg.guarded_run('projected_pose_convergence', **loop.mg.BUDGETS['neural_muscle_closure']):
        main()
