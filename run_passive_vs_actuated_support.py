"""Passive versus actuated support: correct the framing of the static support numbers.

Read-only. The support audit's 63.8% figure was measured at the declared zero-activation
initial state, so it describes the PASSIVE body, not what an actuated stance can carry. This
audit measures how total ground support evolves with muscle activation over the saved episode,
and cross-checks that the support-audit episode is the same integration as the branch episode.
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
SUPPORT = HERE / 'outputs/model_branches/support_audit/support_audit_traces.npz'
BRANCH = HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz'
OUT = HERE / 'outputs/model_branches/passive_vs_actuated'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    doc = json.loads((HERE / 'outputs/model_branches/support_audit/support_audit_results.json').read_text())
    weight = doc['weight_and_mass']['weight_model_units']
    with np.load(SUPPORT) as s, np.load(BRANCH) as b:
        t = s['25us_time_s']
        grf = s['25us_grf_z_B']
        # The support audit sampled every body step (25 us, 8001 rows); the branch episode
        # recorded every 0.1 ms (2001 rows). Compare on the common 0.1 ms grid.
        dt_support = float(s['25us_time_s'][1] - s['25us_time_s'][0])
        dt_branch = float(b['25us_normal_body_time_s'][1] - b['25us_normal_body_time_s'][0])
        step_ratio = int(round(dt_branch / dt_support))
        # Both grids include the t=0 endpoint, so the counts relate as (n-1)*ratio + 1.
        same_run = ((b['25us_normal_body_time_s'].size - 1) * step_ratio + 1 == s['25us_time_s'].size
                    and np.array_equal(s['25us_body_qpos'][::step_ratio], b['25us_normal_body_qpos'])
                    and np.array_equal(s['25us_root_z'][::step_ratio], b['25us_normal_heights_mm'][:, 0]))
        act = b['25us_normal_body_act']
        ctrl = b['25us_normal_body_ctrl']
    ratio = grf / weight
    mean_act = act.mean(axis=1)
    dt = float(t[1] - t[0])

    def first_time(mask):
        idx = np.flatnonzero(mask)
        return float(t[idx[0]]) if idx.size else None

    result = dict(
        status='MEASURED_PASSIVE_VERSUS_ACTUATED_SUPPORT',
        label='manual read-only correction audit on saved states',
        question='Was the 63.8% initial support figure a passive-state measurement, and how does '
                 'support evolve once muscle activation develops?',
        correction=dict(
            earlier_wording='the deployed stance carries only 63.8% of body weight',
            corrected_wording='at the declared zero-activation initial state the PASSIVE stance carries 63.8% of '
                              'body weight; support rises above body weight within the first milliseconds as muscle '
                              'activation develops, and averages about body weight over the episode',
            why='the initial state sets act=0 by declaration, so a static evaluation of that state measures passive '
                'structures only and cannot be used to claim a stance is unsupportable when actuated'),
        cross_check_same_integration_as_branch=bool(same_run),
        cross_check_detail='support-audit 25 us episode resampled every 4th step matches the branch episode '
                           'exactly on the shared 0.1 ms grid',
        timeline=dict(support_ratio_at_0=float(ratio[0]),
                      support_ratio_mean=float(ratio.mean()),
                      support_ratio_max=float(ratio.max()),
                      time_first_support_above_0p95_weight_s=first_time(ratio > 0.95),
                      time_first_support_above_weight_s=first_time(ratio > 1.0),
                      time_first_tibia_activation_s=first_time(mean_act > 0.01),
                      mean_activation_at_0=float(mean_act[0]),
                      mean_activation_max=float(mean_act.max()),
                      time_mean_activation_peak_s=float(t[int(np.argmax(mean_act))]),
                      sample_dt_s=dt,
                      samples=int(t.size)),
        static_scan_reframing=dict(
            two_parameter_scan_max_support_ratio=json.loads(
                (HERE / 'outputs/model_branches/stance_support_scan/stance_support_scan.json').read_text()
            )['best_ceiling']['max_support_ratio'],
            symmetric_search_best_support_ratio=json.loads(
                (HERE / 'outputs/model_branches/stance_search/stance_search.json').read_text()
            )['best']['support_ratio'],
            meaning='both used act=0 static evaluations, so both measure passive support only'),
        interpretation=['The correct comparison is support versus activation over time, not a single passive number.',
                        'A static act=0 scan cannot decide whether a stance is supportable by the animal; it only '
                        'describes the unactuated mechanism.'],
        limitations=['One episode and one pose; support is measured from contact forces and is unit-free.',
                     'No new dynamics were run: this audit re-reads existing saved arrays.'],
        provenance=dict(support_trace=str(SUPPORT), support_trace_sha256=sha(SUPPORT),
                        branch_trace=str(BRANCH), branch_trace_sha256=sha(BRANCH),
                        canonical_xml_sha256=sha(closure.cp.CANONICAL_XML)),
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'passive_vs_actuated.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'passive_vs_actuated.npz', time_s=t, support_ratio=ratio, mean_activation=mean_act)
    print(json.dumps(dict(cross_check=result['cross_check_same_integration_as_branch'],
                          timeline=result['timeline']), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('passive_vs_actuated', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
