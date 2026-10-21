"""Actual canonical-body/BANC paired baseline. 手工研究原型，不是成品.
No gain search, background, PD, CPG or motor injection. Frozen XML untouched.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from pathlib import Path
import numpy as np
import run_neural_muscle_closure as loop
from engine.embodied.source_snapshot import build_source_snapshot

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/canonical_paired_baseline'
# Omit unsupported all-hairplate mean joint, foot-force->all CS, and
# foot-contact->all bristles assignments. Remaining FeCO sign/gain are hypotheses.
OMIT = ('hairplate_angle', 'campaniform_strain', 'bristle_touch')


def main():
    import mujoco
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    started = time.monotonic()
    dest = OUT/time.strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True, exist_ok=False)
    model, pose, gate, joints, geom_leg = loop.prepare_body()
    original_floor = model.actuator_ctrlrange.copy()
    built = loop.build_neural_tier(loop.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    manifest = loop.build_motor_manifest(json.loads(loop.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, stats = loop.channel_index_map(int(built['n']), built['tier']['root_ids'])
    pointer = HERE/'outputs/neural_muscle_closure/current_run.json'
    frozen = json.loads(pointer.read_text())
    identity_path = pointer.parent/frozen['path']/'neural_identity.json'
    identity = json.loads(identity_path.read_text())
    if list(map(str, built['tier']['root_ids'])) != list(map(str, identity['tier_root_ids'])):
        raise ValueError('tier roots differ from frozen identity')
    if groups != identity['sensory_groups']:
        raise ValueError('sensory groups differ from frozen identity')
    arrays = {'seated_qpos': pose, 'tier_root_ids': np.asarray(built['tier']['root_ids'])}
    episodes = {}
    specs = [
        ('native_body', 'normal', False, OMIT, False),
        ('native_sensory_cut', 'sensory_cut', False, OMIT, False),
        ('zero_floor_body', 'normal', False, OMIT, True),
        ('zero_floor_sensory_cut', 'sensory_cut', False, OMIT, True),
        ('zero_floor_motor_cut', 'motor_cut', False, OMIT, True),
        ('zero_floor_replay', 'replay', False, OMIT, True),
        ('zero_floor_declared_touch', 'normal', True, ('hairplate_angle', 'campaniform_strain'), True),
    ]
    for name, arm, touch, omitted, zero_floor in specs:
        model.actuator_ctrlrange[:] = original_floor
        if zero_floor:
            model.actuator_ctrlrange[:, 0] = 0.
        replay = arrays.get('zero_floor_body_controls') if arm == 'replay' else None
        result, trajectory, commands, spikes = loop.episode(
            model, pose, joints, geom_leg, built, manifest, groups, arm, .1,
            replay=replay, external_touch=touch, disabled_channels=omitted)
        result.update(model_branch='compiled_zero_ctrl_floor_only' if zero_floor else 'frozen_native',
                      experiment=name, external_touch_enabled=touch,
                      motor_spike_total=sum(r['motor_spikes'] for r in result['records']),
                      network_spike_total=int(spikes.sum()),
                      maximum_control=float(commands.max()),
                      max_activation=max(r['activation_max'] for r in result['records']))
        episodes[name] = result
        for key, value in [('trajectory', trajectory), ('controls', commands), ('spike_counts', spikes)]:
            arrays[name+'_'+key] = value
        print(name, 'motor_spikes', result['motor_spike_total'], 'network_spikes', result['network_spike_total'], 'escape', result['escape_time_s'], flush=True)
    model.actuator_ctrlrange[:] = original_floor
    checks = {
        'canonical_xml_unchanged': loop.cp.sha256_file(loop.cp.CANONICAL_XML) == loop.cp.CANONICAL_XML_SHA256,
        'compiled_floor_restored': bool(np.array_equal(model.actuator_ctrlrange, original_floor)),
        'sensory_cut_zero_input': all(r['sensory_current_min_nA'] == r['sensory_current_max_nA'] == 0 for name in ('native_sensory_cut', 'zero_floor_sensory_cut') for r in episodes[name]['records']),
        'zero_floor_sensory_cut_zero_controls': bool(np.all(arrays['zero_floor_sensory_cut_controls'] == 0)),
        'zero_floor_sensory_cut_zero_activation': episodes['zero_floor_sensory_cut']['max_activation'] == 0,
        'replay_exact_body': bool(np.array_equal(arrays['zero_floor_body_trajectory'], arrays['zero_floor_replay_trajectory'])),
        'replay_exact_controls': bool(np.array_equal(arrays['zero_floor_body_controls'], arrays['zero_floor_replay_controls'])),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    np.savez_compressed(dest/'paired_traces.npz', **arrays)
    (dest/'neural_identity.json').write_text(json.dumps(dict(tier_root_ids=list(map(str, built['tier']['root_ids'])), sensory_groups=groups, motor_manifest=manifest), indent=2)+'\n')
    import sys
    sources = {str(Path(__file__).resolve()): loop.cp.sha256_file(__file__)}
    for module in list(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and Path(p).resolve().is_relative_to(HERE) and not any(part in ('venv', 'venv_body', 'vendor') for part in Path(p).parts):
            sources[str(Path(p).resolve())] = loop.cp.sha256_file(p)
    build_source_snapshot(sources, dest/'source_snapshot.zip')
    report = dict(label='手工研究原型，不是成品', status='MEASURED_BASELINE_NOT_STANDING_ACCEPTANCE',
        duration_s=.1, n_neurons=int(built['n']), clocks_s=dict(body=loop.DT_BODY, neural=loop.DT_NEURAL, command=loop.DT_COMMAND),
        initial_pose_gate=gate, checks=checks, episodes=episodes,
        provenance=dict(canonical_xml=str(loop.cp.CANONICAL_XML), canonical_xml_sha256=loop.cp.sha256_file(loop.cp.CANONICAL_XML),
            identity_path=str(identity_path), identity_sha256=loop.cp.sha256_file(identity_path),
            connectome_sha256=loop.cp.sha256_file(HERE/'data/flywire/banc_connectome.npz'),
            lookup_sha256=loop.cp.sha256_file(loop.LOOKUP), source_hashes=sources,
            background_nA=0, edge_scale=.2, sensory_config=loop.asdict(loop.ChannelConfig()), motor_config=loop.asdict(loop.MotorOutputConfig())),
        limits=['Initial historical pose joint-range violations retained and reported; not a valid standing benchmark.',
            'Claw/hook input sign and encoding gains are declared uncalibrated hypotheses, not root-specific tuning.',
            'Unknown hairplate, CS and body-driven bristle assignments disabled in primary arms.',
            'Touch arm is diagnostic: binary foot contact and local 0.3nA injection into LF bristles are declared proxies, not receptor mechanics.',
            'Native XML positive control floor is non-neural actuation. Compiled zero-floor branch changes only lower control bounds in memory.',
            'Motor cut begins at 15ms and cannot erase preceding activation history.',
            'Replay only checks determinism, never scientific closed-loop success.',
            'Glutamate inhibitory sign remains a model assumption; no target-specific receptor validation.'], wall_s=time.monotonic()-started)
    (dest/'paired_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, result in episodes.items():
        t = [r['time_end_s']*1000 for r in result['records']]
        ax[0].plot(t, [r['height_mm_declared'] for r in result['records']], label=name)
        ax[1].plot(t, [r['motor_spikes'] for r in result['records']])
        ax[2].plot(t, [r['activation_max'] for r in result['records']])
    ax[0].set_ylabel('Thorax height (model mm)')
    ax[1].set_ylabel('Motor spikes / 5ms')
    ax[2].set_ylabel('Maximum activation'); ax[2].set_xlabel('Time (ms)')
    ax[0].legend(fontsize=8)
    fig.suptitle('Manual research prototype — actual body/BANC paired baseline; NOT standing acceptance')
    fig.tight_layout(); fig.savefig(dest/'paired_baseline.png', dpi=150); plt.close(fig)
    hashes = {str(p): loop.cp.sha256_file(p) for p in dest.iterdir() if p.is_file()}
    (dest/'artifact_manifest.json').write_text(json.dumps(dict(files=hashes), indent=2)+'\n')
    print('OUTPUT', dest, 'wall_s', report['wall_s'], flush=True)
    return dest


if __name__ == '__main__':
    with loop.mg.guarded_run('canonical_paired_baseline', **loop.mg.BUDGETS['neural_muscle_closure']):
        main()
