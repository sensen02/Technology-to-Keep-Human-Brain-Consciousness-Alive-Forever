"""Closed-loop run on the tarsus-DOF extension.

Runs the SAME real feel->nerve->muscle->body loop as the published branches, on the extended
body model, and compares against the published 25 us baseline. The sensory front end is left
exactly as before (the new tarsus joint is excluded from the joint-deviation group, and the
joint list keeps its original seven entries), so the only difference between the two models is
the added tarsus degree of freedom and its two new muscles per leg.

Arms: normal, spike-output silence, exact command replay. No CPG/PD, no hidden drive, no gain
or parameter change, no transmitter or connectivity invention.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
import run_fine_clock_branch as fine
import run_zero_floor_branch as zero
from engine.embodied.motor_output import build_motor_manifest
OUT = HERE / 'outputs/model_branches/tarsus_dof'
XML = Path(os.environ.get('TARSUS_XML', str(OUT / 'fruitfly_tarsus_dof.xml')))
TAG = os.environ.get('TARSUS_TAG', '')
POOLS_EXPORT = OUT / 'tarsus_motor_pools.json'
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
US = 25
NEW_TARGETS = (('tarsus_depressor_muscle', 'depressor'), ('tarsus_levator_muscle', 'levator'))
EFFECTOR = {'LF': 'front_leg', 'RF': 'front_leg', 'LM': 'middle_leg', 'RM': 'middle_leg',
            'LH': 'hind_leg', 'RH': 'hind_leg'}
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def branch_prepare(model):
    """Branch-local body preparation: same protocol as the core, tolerating the extra tarsus joint.

    The core asserts exactly seven leg joints; this branch keeps the SAME seven joints for every
    sensory and clock purpose and simply ignores the added tarsus joint, which is recorded here.
    """
    import mujoco
    data = mujoco.MjData(model)
    closure.r39.apply_round37_pose(model, data, json.loads(closure.r39.ROUND37_POSE.read_text()))
    base = data.qpos.copy()
    base[2] = closure.cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    gate = closure.cp.stance_facts(model, data).as_dict()
    joints, added = {}, []
    for leg in closure.LEGS:
        ids = [j for j in range(model.njnt)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_' + leg)]
        original = [j for j in ids if 'tarsus' not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower()]
        extra = [j for j in ids if j not in original]
        tib = [j for j in original if 'tibia' in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).lower()]
        if len(original) != 7 or len(tib) != 1:
            raise ValueError(f'leg {leg}: expected 7 original joints and 1 tibia, got {len(original)}/{len(tib)}')
        if len(extra) != 1:
            raise ValueError(f'leg {leg}: expected exactly one added tarsus joint, got {len(extra)}')
        added.append(dict(leg=leg, joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, extra[0]),
                          qpos_address=int(model.jnt_qposadr[extra[0]]), dof_address=int(model.jnt_dofadr[extra[0]])))
        joints[leg] = dict(q=[int(model.jnt_qposadr[j]) for j in original],
                           tib_q=int(model.jnt_qposadr[tib[0]]), tib_v=int(model.jnt_dofadr[tib[0]]))
    geom_leg = {}
    for g in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').lower()
        body = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '').lower()
        match = [leg for leg in closure.LEGS if ('tars' in name + body and (name.startswith(leg) or body.startswith(leg)))]
        if len(match) == 1:
            geom_leg[g] = match[0]
    if set(geom_leg.values()) != set(closure.LEGS):
        raise ValueError('foot geometry mapping incomplete')
    return base, gate, joints, geom_leg, added


def augmented_pools(pool_result, csv_path):
    """Add the two new muscles per leg from the exported annotation; no cell is inferred."""
    exported = json.loads(Path(csv_path).read_text())
    assert exported['feather_sha256'] == sha(FEATHER), 'exported pools are stale'
    records = [dict(r) for r in exported['records']]
    detail = {r['actuator']: dict(target=r['muscle_target'], cells=r['n_pool'],
                                  roots=r['pool_root_ids'][:12], cell_types=r['pool_cell_types'])
              for r in records}
    out = json.loads(json.dumps(pool_result))
    out['measured']['resolved'] = out['measured']['resolved'] + records
    return out, detail


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    assert np.isclose(model.opt.timestep, US * 1e-6, rtol=0, atol=1e-15)
    base_model = closure.cp.load_model()
    pose, gate, joints, geom_leg, added = branch_prepare(model)
    assert len(pose) == model.nq
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    pools, pool_detail = augmented_pools(json.loads(closure.POOLS.read_text()), POOLS_EXPORT)
    manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    frozen = {str(p): sha(p) for p in (XML, closure.cp.CANONICAL_XML, zero.XML,
                                       HERE / 'engine/embodied/branch_episode.py',
                                       HERE / 'engine/embodied/motor_output.py',
                                       HERE / 'engine/embodied/sensory_channels.py')}
    episodes, results, arrays = {}, {}, {'seated_qpos': pose}
    for arm in ('normal', 'spike_output_silence'):
        summary, result = fine.run_arm(model, pose, joints, geom_leg, built, manifest, groups, US, arm)
        episodes[f'tarsus_{arm}'] = summary
        results[f'tarsus_{arm}'] = result
    summary, result = fine.run_arm(model, pose, joints, geom_leg, built, manifest, groups, US, 'replay',
                                   replay=results['tarsus_normal']['controls'])
    episodes['tarsus_replay'] = summary
    results['tarsus_replay'] = result
    for key, res in results.items():
        arrays.update({key + '_' + k: v for k, v in res.items()})
    checks = {
        'replay_exact': all(np.array_equal(results['tarsus_normal'][k], results['tarsus_replay'][k])
                            for k in ('controls', 'body_qpos', 'body_qvel', 'body_act', 'heights_mm', 'spike_counts')),
        'silence_zero_drive': all(np.all(results['tarsus_spike_output_silence'][k] == 0)
                                  for k in ('body_act', 'body_ctrl')),
        'all_finite': all(np.isfinite(results[a][k]).all() for a in results
                          for k in ('body_qpos', 'body_qvel', 'body_act', 'body_ctrl', 'heights_mm')),
        'sources_unchanged': {str(p): sha(p) for p in frozen} == frozen,
        'new_actuators_have_pools': all(len(r['local_indices']) > 0 for r in manifest['rows']
                                        if 'Tarsus_' in r['actuator']),
    }
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    base_doc = json.loads((HERE / 'outputs/model_branches/fine_clock/fine_clock_results.json').read_text())
    base_ep = base_doc['episodes']['25us_normal']
    dn = results['tarsus_normal']
    branch_report = dict(endpoint_heights_mm=dn['heights_mm'][-1].tolist(),
                         final_root_mm=float(dn['heights_mm'][-1, 0]),
                         escape_time_s=episodes['tarsus_normal']['escape_time_s'],
                         act_max=float(dn['body_act'].max()),
                         nonfoot_contacts_final=int(dn['body_nonfoot_contacts'][-1]))
    baseline = dict(endpoint_heights_mm=base_ep['endpoint_heights_mm'],
                    escape_time_s=base_ep['escape_time_s'])
    new_act = [r['actuator'] for r in manifest['rows'] if 'Tarsus_' in r['actuator']]
    idx = [r['actuator_index'] for r in manifest['rows'] if 'Tarsus_' in r['actuator']]
    tarsus_command = results['tarsus_normal']['controls'][:, idx]
    npz = OUT / f'tarsus_dof_traces{TAG}.npz'
    np.savez_compressed(npz, **arrays)
    doc = dict(schema_version=1, status='PASS_BRANCH_INVARIANTS',
               label='manual labelled model extension; closed-loop measurement, not a cure and not walking',
               extension=dict(added_dof=6, added_actuators=12, new_actuators=new_act,
                              tarsus_joint_angles=added, pool_detail=pool_detail,
                              tarsus_command_max=float(np.abs(tarsus_command).max()),
                              tarsus_command_mean=float(np.abs(tarsus_command).mean())),
               episodes=episodes, branch=branch_report, published_25us_baseline=baseline,
               comparison=dict(escape_time_delta_s=branch_report['escape_time_s'] - baseline['escape_time_s'],
                               final_root_delta_mm=branch_report['final_root_mm'] - baseline['endpoint_heights_mm'][0]),
               checks=checks, gate=gate,
               provenance=dict(source_hashes=frozen, npz_sha256=sha(npz), mujoco_version=mujoco.__version__,
                               branch_xml=str(XML), branch_xml_sha256=sha(XML)),
               limitations=['One pose, one timestep, one tier; the added joint axis and attachment geometry are '
                            'DECLARED assumptions, not measurements.',
                            'The sensory front end is unchanged: the new tarsus joint is not sensed.',
                            'A worse outcome is a legitimate result: an unvalidated free DOF can add a collapse mode.',
                            'No gain, parameter, connectivity or transmitter label was changed.'],
               memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(doc['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        doc['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    (OUT / f'tarsus_dof_results{TAG}.json').write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(branch=branch_report, baseline=baseline,
                          comparison=doc['comparison'],
                          tarsus_command_max=doc['extension']['tarsus_command_max'],
                          pools={k: v['cells'] for k, v in pool_detail.items()}), indent=2))
    return doc


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s tarsus branch deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('tarsus_dof_branch', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
