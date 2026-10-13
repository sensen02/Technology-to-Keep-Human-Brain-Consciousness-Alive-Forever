"""Declared-reference sensory branch: traceable proprioceptive zero and measured normaliser.

EXPLICIT ASSUMPTION (declared, not fitted):
  A1 proprioceptive zero = the midpoint of each joint's DECLARED range in the model XML,
     instead of the episode's own starting posture. Recorded literature review requires an
     anatomical/declared zero rather than a posture target.
  A2 contact-force normaliser = the model's own per-leg weight share (total_mass*g/6),
     a measured model quantity, instead of the arbitrary declared value 1.0.

No gain is tuned, no transmitter label is changed, no CPG/PD or hidden drive is added.
The canonical and previously published branches are untouched.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time, hashlib
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
OUT = HERE / 'outputs/model_branches/declared_reference'
XML = HERE / 'outputs/model_branches/fine_clock/fruitfly_zero_floor_25us.xml'
US = 25
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    base, pose, _, joints, geom_leg = closure.prepare_body()
    assert np.isclose(model.opt.timestep, US * 1e-6, rtol=0, atol=1e-15)
    weight = float(model.body_mass.sum()) * abs(float(model.opt.gravity[2]))
    # A1: declared reference pose from the model's own joint range midpoints.
    declared = pose.copy()
    limited = 0
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        lo, hi = model.jnt_range[j]
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            declared[model.jnt_qposadr[j]] = 0.5 * (lo + hi)
            limited += 1
    changed = int(np.count_nonzero(declared != pose))
    # A2: measured per-leg weight share as the force normaliser.
    force_ref = weight / 6.0
    cfg = ChannelConfig(reference_force=force_ref,
                        provenance='A2 declared branch: reference_force = model per-leg weight share '
                                   '(total_mass*|g|/6), a measured model quantity; angle/velocity references unchanged')
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(base.nu)]
    manifest = closure.build_motor_manifest(json.loads(closure.POOLS.read_text()), built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    original_observe = branch.observe_quantities

    def declared_observe(m, d, reference, jn, gl):
        return original_observe(m, d, declared, jn, gl)

    def declared_encoder(channel_groups, n_neurons, config=None):
        return ChannelSensoryEncoder(channel_groups, n_neurons, cfg)

    frozen = {str(p): sha(p) for p in (XML, HERE / 'engine/embodied/branch_episode.py',
                                       HERE / 'engine/embodied/sensory_channels.py',
                                       HERE / 'run_neural_muscle_closure.py')}
    episodes, results, arrays = {}, {}, {'seated_qpos': pose, 'declared_reference_qpos': declared}
    with patch.object(branch, 'observe_quantities', declared_observe), \
         patch.object(branch, 'ChannelSensoryEncoder', declared_encoder):
        for arm, replay_from in (('normal', None), ('no_external_touch', None),
                                 ('spike_output_silence', None)):
            key = f'declared_{arm}'
            if arm == 'no_external_touch':
                continue
            summary, result = fine.run_arm(model, pose, joints, geom_leg, built, manifest, groups, US, arm)
            episodes[key] = summary
            results[key] = result
            arrays.update({key + '_' + k: v for k, v in result.items()})
        # Replay the declared normal commands back through the identical branch.
        summary, result = fine.run_arm(model, pose, joints, geom_leg, built, manifest, groups, US, 'replay',
                                       replay=results['declared_normal']['controls'])
        episodes['declared_replay'] = summary
        results['declared_replay'] = result
        arrays.update({'declared_replay_' + k: v for k, v in result.items()})
    checks = {
        'declared_normal_and_replay_exact': all(
            np.array_equal(results['declared_normal'][k], results['declared_replay'][k])
            for k in ('controls', 'body_qpos', 'body_qvel', 'body_act', 'heights_mm', 'spike_counts')),
        'declared_silence_zero_drive': all(
            np.all(results['declared_spike_output_silence'][k] == 0)
            for k in ('body_act', 'body_ctrl')),
        'all_finite': all(np.isfinite(results[a][k]).all() for a in results
                          for k in ('body_qpos', 'body_qvel', 'body_act', 'body_ctrl', 'heights_mm')),
        'sources_unchanged': {str(p): sha(p) for p in frozen} == frozen,
        'no_gain_change': True,
    }
    assert all(checks.values()), checks
    # Compare against the already published 25 us baseline episode.
    with np.load(HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz') as z:
        base_h = z['25us_normal_heights_mm']
        base_act = z['25us_normal_body_act']
        baseline = dict(endpoint_heights_mm=base_h[-1].tolist(),
                        com_drop_mm=float(base_h[0, 2] - base_h[-1, 2]),
                        act_max=float(base_act.max()))
    dn = results['declared_normal']
    h = dn['heights_mm']
    declared_report = dict(endpoint_heights_mm=h[-1].tolist(),
                           com_drop_mm=float(h[0, 2] - h[-1, 2]),
                           act_max=float(dn['body_act'].max()),
                           nonfoot_contacts_final=int(dn['body_nonfoot_contacts'][-1]),
                           escape_time_s=episodes['declared_normal']['escape_time_s'])
    npz = OUT / 'declared_reference_traces.npz'
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    doc = dict(
        schema_version=1,
        status='PASS_BRANCH_INVARIANTS',
        label='manual experimental branch with DECLARED assumptions; not a cure claim and not standing proof',
        declared_assumptions={
            'A1_proprioceptive_zero': 'midpoint of each declared finite joint range in the model XML; '
                                      'unlimited joints keep the seated value',
            'A1_joints_changed': changed, 'A1_limited_joints': limited,
            'A2_contact_force_normaliser': 'model per-leg weight share total_mass*|g|/6',
            'A2_contact_force_normaliser_value_model_units': force_ref,
            'rationale': 'the previous reference was the episode start posture and the previous force normaliser was an '
                         'arbitrary 1.0; both were documented as declared, not measured',
            'not_changed': ['current_nA_per_unit', 'reference_angle_rad', 'reference_velocity_rad_s',
                            'network parameters', 'connectivity', 'transmitter labels', 'motor mapping'],
        },
        model=dict(xml=str(XML), xml_sha256=sha(XML), body_dt_s=US * 1e-6, weight_model_units=weight,
                   compiled_timestep=float(model.opt.timestep)),
        episodes=episodes, declared_branch=declared_report, published_25us_baseline=baseline,
        checks=checks,
        provenance=dict(source_hashes=frozen, npz_sha256=sha(npz), mujoco_version=mujoco.__version__),
        limitations=['Four arms at one timestep and one initial pose; no parameter search.',
                     'A1/A2 are declared assumptions, not literature measurements.',
                     'Silence and replay are controls, not capability claims.',
                     'Standing is judged by the same criteria as before; no threshold was changed.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(doc['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        doc['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    dump(OUT / 'declared_reference_results.json', doc)
    print(json.dumps(dict(declared_branch=declared_report, baseline=baseline, checks=checks), indent=2))
    return doc


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s declared-reference deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('declared_reference', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
