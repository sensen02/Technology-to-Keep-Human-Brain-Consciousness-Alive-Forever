"""Commensurate-normaliser branch: isolate the effect of the sensory normalisers.

EXPLICIT ASSUMPTION (declared, not fitted to behaviour):
  Each sensory channel should be divided by a *measured or model-declared* physical scale so
  that one physical unit of deviation produces a comparable signal across channels:
    A1 angle      : median half-range of the modelled leg joints taken from the XML joint ranges
    A2 velocity   : that angle scale divided by a declared 0.1 s time constant
    A3 force      : the model's per-leg weight share (total_mass*|g|/6), a measured quantity
  The proprioceptive REFERENCE is deliberately left at the episode's starting pose, so this
  branch isolates the normalisers from the separate reference-zero change tested earlier.

Unchanged: current scale, network parameters, connectivity, transmitter labels, motor mapping,
reference pose. Controls: output-silence and exact command replay.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, re, signal, statistics, sys, time, hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_duration_probe as duration
import run_fine_clock_branch as fine
from engine.embodied import branch_episode as branch
from engine.embodied.sensory_channels import ChannelConfig, ChannelSensoryEncoder
closure = duration.closure
OUT = HERE / 'outputs/model_branches/commensurate_normaliser'
XML = HERE / 'outputs/model_branches/fine_clock/fruitfly_zero_floor_25us.xml'
US = 25
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    base, pose, _, joints, geom_leg = closure.prepare_body()
    weight = float(model.body_mass.sum()) * abs(float(model.opt.gravity[2]))
    # A1: median half-range over the modelled leg joints that declare a finite range.
    half = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        if not name.startswith('joint_'):
            continue
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            half.append(0.5 * float(hi - lo))
    angle_scale = float(statistics.median(half))
    declared_tau = 0.1
    cfg = ChannelConfig(reference_angle_rad=angle_scale, reference_velocity_rad_s=angle_scale / declared_tau,
                        reference_force=weight / 6.0,
                        provenance='commensurate-normaliser branch: angle scale = median half-range of modelled leg '
                                   'joints; velocity scale = that / declared 0.1 s; force scale = per-leg weight share')
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(base.nu)]
    manifest = closure.build_motor_manifest(json.loads(closure.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])

    def encoder(channel_groups, n_neurons, config=None):
        return ChannelSensoryEncoder(channel_groups, n_neurons, cfg)

    frozen = {str(p): sha(p) for p in (XML, HERE / 'engine/embodied/branch_episode.py',
                                       HERE / 'engine/embodied/sensory_channels.py',
                                       HERE / 'run_neural_muscle_closure.py')}
    episodes, results, arrays = {}, {}, {'seated_qpos': pose}
    with patch.object(branch, 'ChannelSensoryEncoder', encoder):
        for arm in ('normal', 'spike_output_silence'):
            summary, result = fine.run_arm(model, pose, joints, geom_leg, built, manifest, groups, US, arm)
            episodes[f'commensurate_{arm}'] = summary
            results[f'commensurate_{arm}'] = result
        summary, result = fine.run_arm(model, pose, joints, geom_leg, built, manifest, groups, US, 'replay',
                                       replay=results['commensurate_normal']['controls'])
        episodes['commensurate_replay'] = summary
        results['commensurate_replay'] = result
    for key, res in results.items():
        arrays.update({key + '_' + k: v for k, v in res.items()})
    checks = {
        'replay_exact': all(np.array_equal(results['commensurate_normal'][k], results['commensurate_replay'][k])
                            for k in ('controls', 'body_qpos', 'body_qvel', 'body_act', 'heights_mm', 'spike_counts')),
        'silence_zero_drive': all(np.all(results['commensurate_spike_output_silence'][k] == 0)
                                  for k in ('body_act', 'body_ctrl')),
        'all_finite': all(np.isfinite(results[a][k]).all() for a in results
                          for k in ('body_qpos', 'body_qvel', 'body_act', 'body_ctrl', 'heights_mm')),
        'sources_unchanged': {str(p): sha(p) for p in frozen} == frozen,
        'reference_left_at_start_pose': True,
    }
    assert all(checks.values()), checks
    baseline_doc = json.loads((HERE / 'outputs/model_branches/fine_clock/fine_clock_results.json').read_text())
    base_ep = baseline_doc['episodes']['25us_normal']
    dn = results['commensurate_normal']
    report_branch = dict(endpoint_heights_mm=dn['heights_mm'][-1].tolist(),
                         com_drop_mm=float(dn['heights_mm'][0, 2] - dn['heights_mm'][-1, 2]),
                         act_max=float(dn['body_act'].max()),
                         escape_time_s=episodes['commensurate_normal']['escape_time_s'])
    baseline = dict(endpoint_heights_mm=base_ep['endpoint_heights_mm'],
                    com_drop_mm=float(base_ep['com_drop_mm']) if 'com_drop_mm' in base_ep else None,
                    escape_time_s=base_ep['escape_time_s'])
    npz = OUT / 'commensurate_normaliser_traces.npz'
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    doc = dict(schema_version=1, status='PASS_BRANCH_INVARIANTS',
               label='manual experimental branch with DECLARED normaliser assumptions; not a cure claim',
               declared_assumptions=dict(
                   A1_angle_scale_rad=angle_scale, A1_source='median half-range of modelled leg joints (XML joint ranges)',
                   A2_velocity_scale_rad_s=angle_scale / declared_tau, A2_declared_tau_s=declared_tau,
                   A3_force_scale=weight / 6.0, A3_source='model per-leg weight share',
                   reference_pose='UNCHANGED: still the episode starting pose',
                   not_changed=['current_nA_per_unit', 'reference pose', 'network parameters', 'connectivity',
                                'transmitter labels', 'motor mapping']),
               model=dict(xml=str(XML), xml_sha256=sha(XML), body_dt_s=US * 1e-6, weight_model_units=weight),
               episodes=episodes, branch=report_branch, published_25us_baseline=baseline,
               checks=checks, provenance=dict(source_hashes=frozen, npz_sha256=sha(npz),
                                              mujoco_version=mujoco.__version__),
               limitations=['Four arms, one timestep, one pose; no parameter search.',
                            'The normaliser values are declared model-derived scales, not literature measurements.',
                            'This isolates the normalisers from the reference-zero change tested earlier.',
                            'Standing criteria are unchanged; a worse or equal result is reported as such.'],
               memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(doc['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        doc['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    (OUT / 'commensurate_normaliser_results.json').write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(branch=report_branch, baseline=baseline, checks=checks,
                          scales=doc['declared_assumptions']), indent=2))
    return doc


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s commensurate deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('commensurate_normaliser', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
