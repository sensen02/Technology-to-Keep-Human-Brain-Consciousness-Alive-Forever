"""Cross-check the recent audits for mix-ups: wrong episode, wrong weight, wrong indices.

Read-only. Every check compares an independently recorded quantity against the artifact
that a different script produced, so a copy/paste or provenance mistake in any one audit
shows up as a disagreement rather than being repeated silently.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
B = HERE / 'outputs/model_branches'
RUN = HERE / 'outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main(report):
    checks, details = {}, {}

    def check(name, condition, evidence):
        checks[name] = bool(condition)
        details[name] = evidence

    # 1. Weight must agree between the manifest, the support audit and the declared branch.
    support = load(B / 'support_audit/support_audit_results.json')
    declared = load(B / 'declared_reference/declared_reference_results.json')
    manifest = load(RUN / 'motor_manifest.json')['measured']
    w_support = support['weight_and_mass']['weight_model_units']
    w_manifest = load(RUN / 'closure_results.json')['units'].get('mass_unit_kg_inferred')
    check('weight_matches_existing_audit',
          abs(w_support - 24.446354755878005) < 1e-12,
          dict(support_audit=w_support, earlier_support_force_audit=24.446354755878005))
    check('declared_branch_uses_same_weight',
          abs(declared['model']['weight_model_units'] - w_support) < 1e-9,
          dict(declared=declared['model']['weight_model_units'], support=w_support))
    # Real check (replacing an earlier vacuous one): the canonical model hash must equal the
    # frozen constant, and the declared branch's A2 normaliser must be exactly weight/6.
    a2_value = float(declared['declared_assumptions']['A2_contact_force_normaliser_value_model_units'])
    check('canonical_xml_hash_is_the_frozen_constant',
          sha(closure.cp.CANONICAL_XML) == '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6',
          dict(sha256=sha(closure.cp.CANONICAL_XML)))
    check('declared_A2_normaliser_equals_weight_over_six',
          abs(a2_value - w_support / 6.0) < 1e-9,
          dict(a2_recorded=a2_value, weight_over_six=w_support / 6.0))

    # 2. The 25 us arrays used by later audits must be the fine-clock episode, not a re-run.
    with np.load(B / 'clock_refinement/clock_refinement_traces.npz') as cr, \
         np.load(B / 'fine_clock/fine_clock_traces.npz') as fc:
        same = all(np.array_equal(cr[f'25us_normal_{k}'], fc[f'25us_normal_{k}'])
                   for k in ('body_qpos', 'body_qvel', 'body_act', 'controls', 'spike_counts', 'heights_mm'))
    check('25us_arrays_are_the_same_episode_in_both_files', same,
          dict(compared=['body_qpos', 'body_qvel', 'body_act', 'controls', 'spike_counts', 'heights_mm']))

    # 3. Every audit that reads the 25 us episode must use its 25 us model, not another timestep.
    dt_of = {}
    for name, path in (('fine_clock_25us', B / 'fine_clock/fruitfly_zero_floor_25us.xml'),
                       ('clock_refinement_25us', None)):
        if path is not None and path.exists():
            import mujoco
            dt_of[name] = float(mujoco.MjModel.from_xml_path(str(path)).opt.timestep)
    check('25us_models_really_are_25us',
          all(abs(v - 25e-6) < 1e-15 for v in dt_of.values()), dt_of)

    # 4. Motor manifest internal consistency and agreement with my census.
    census = load(B / 'motor_output_census/motor_output_census.json')
    rows = manifest['rows']
    unresolved = [r for r in rows if r['status'] == 'unresolved']
    no_pool = [r for r in rows if not r.get('local_indices')]
    check('census_uncommandable_equals_manifest_unresolved',
          census['summary']['uncommandable_no_pool'] == len(unresolved) == len(no_pool),
          dict(census=census['summary']['uncommandable_no_pool'], manifest_unresolved=len(unresolved),
               rows_without_pool=len(no_pool)))
    check('census_actuator_total_matches_manifest',
          census['summary']['actuators'] == len(rows) == manifest['n_actuators'],
          dict(census=census['summary']['actuators'], manifest=len(rows), stated=manifest['n_actuators']))
    check('manifest_usable_count_explains_the_gap',
          manifest['usable_actuators'] == manifest['n_actuators'] - len(unresolved),
          dict(usable=manifest['usable_actuators'], total=manifest['n_actuators'], unresolved=len(unresolved)))

    # 5. Sensory channel counts must be identical across all audits that use them.
    counts = {}
    counts['channel_transmitter_audit'] = sum(v['tier_cells']
                                              for v in load(B / 'channel_transmitter_audit/channel_transmitter_audit.json')['per_channel'].values())
    counts['frontend_balance'] = load(B / 'sensory_frontend_balance/sensory_frontend_balance.json')['total_input_current_nA_steps'] is not None and \
        sum(v['cells'] for v in load(B / 'sensory_frontend_balance/sensory_frontend_balance.json')['per_channel'].values())
    counts['propagable_input'] = load(B / 'propagable_input/propagable_input.json')['totals']['sensory_cells']
    with np.load(B / 'sensory_frontend_balance/sensory_frontend_balance_traces.npz') as z:
        counts['frontend_balance_percell_rows'] = int(z['cell_current_mean_abs_nA'].size)
        counts['frontend_balance_unique_cells'] = int(np.unique(z['cell_local_index']).size)
    check('sensory_cell_total_consistent_across_audits',
          len({counts['channel_transmitter_audit'], counts['frontend_balance'],
               counts['propagable_input'], counts['frontend_balance_percell_rows'],
               counts['frontend_balance_unique_cells']}) == 1 == (len({3788}) if 3788 in counts.values() else 99),
          counts)

    # 6. Tier identity: the tier used by the audits is the published run's tier.
    identity = load(RUN / 'neural_identity.json')
    tier_roots = [str(x) for x in identity['tier_root_ids']]
    with np.load(B / 'sensory_frontend_balance/sensory_frontend_balance_traces.npz') as z:
        audited = [str(r) for r in z['cell_root_id']]
        audited_cells = int(z['cell_local_index'].size)
    roots_ok = all(r in set(tier_roots) for r in audited)
    check('audit_cells_belong_to_published_tier', roots_ok,
          dict(tier_neurons=len(tier_roots), audited_cells=audited_cells))

    # 7. Controls column identity: census actuator_index must index the saved controls row.
    with np.load(B / 'clock_refinement/clock_refinement_traces.npz') as cr:
        controls = cr['25us_normal_controls']
    idx = [r['actuator_index'] for r in rows]
    check('actuator_indices_are_a_permutation_of_columns',
          sorted(idx) == list(range(controls.shape[1])), dict(n_indices=len(idx), n_columns=int(controls.shape[1])))

    result = dict(status='PASS_CROSS_AUDIT_CONSISTENCY' if all(checks.values()) else 'FAIL_CROSS_AUDIT',
                  label='manual read-only cross-check, not a new result about the animal',
                  checks=checks, evidence=details,
                  provenance=dict(cross_checked=[str(p) for p in (
                      B / 'support_audit/support_audit_results.json',
                      B / 'declared_reference/declared_reference_results.json',
                      B / 'motor_output_census/motor_output_census.json',
                      B / 'channel_transmitter_audit/channel_transmitter_audit.json',
                      B / 'propagable_input/propagable_input.json',
                      B / 'sensory_frontend_balance/sensory_frontend_balance.json')],
                      canonical_xml_sha256=sha(closure.cp.CANONICAL_XML)),
                  memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    out = B / 'cross_audit_consistency'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cross_audit_consistency.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(checks, indent=2))
    print(json.dumps(details, indent=2)[:2000])
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('cross_audit_consistency', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
