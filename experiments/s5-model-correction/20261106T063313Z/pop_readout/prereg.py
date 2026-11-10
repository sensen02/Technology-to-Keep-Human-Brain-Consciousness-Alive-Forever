"""Freeze the experiment design, the input hashes and the reading rule. Cheap, no simulation.

Run once.  Refuses to overwrite an existing PREREGISTRATION.json.
"""
import sys
from pathlib import Path

import numpy as np

import common as C

sys.path[:0] = [str(C.BASE), str(C.R), str(C.R / 'tools')]
import runtime as rt


def frozen_inputs():
    paths = [Path(p) for p in rt.files()]
    for dt, tag in C.DTS:
        paths += [C.CONFIGS / tag / 'WARM.npz', C.CONFIGS / tag / 'WARM_CACHE.pkl',
                  C.CONFIGS / tag / 'PARAMETERS.json', C.CONFIGS / tag / 'WARM_IDENTITY.json',
                  C.CONFIGS / tag / 'CONTROLLED_RELEASE.npz']
        for ph in C.PHASES:
            folder = C.SEALED_RUNS / f'{tag}_paired_{ph}'
            paths += [folder / 'RESULT.json', folder / 'NATIVE.npz',
                      folder / 'BODY.npz', folder / 'NEURAL.npz']
    paths += [C.NUMERICS_SUMMARY, C.IDENTITIES,
              C.TEACHER / 'BODY.npz', C.TEACHER / 'NATIVE.npz', C.TEACHER / 'TEACHER.npz',
              C.META, C.B888_TO_ROOT, C.IDENTITIES_NPZ, C.MAPPING,
              C.PRIOR / 'SUMMARY.json', C.PRIOR / 'R0_BITWISE.json',
              C.PRIOR / 'READOUT_R1.npz', C.PRIOR / 'READOUT_STATIC.json']
    return list(dict.fromkeys(str(p) for p in paths))


def main():
    if (C.O / 'PREREGISTRATION.json').exists():
        raise SystemExit('PREREGISTRATION.json already exists; refusing to overwrite')
    cfg = {
        'stage': 'S5 WHOLE-POPULATION READOUT TEST: replace the 204-cell / 66-pool readout with a '
                 'ridge readout over ALL 12000 filtered cell rates (P1/P2) and, as a control, over '
                 'the 204 motor cells only (P3); apply every readout to the SAME sealed neural warm '
                 'states at both neural timesteps. Zero new warm replays, zero new teacher '
                 'recordings, zero new connectome content.',
        'question': 'Can a readout that uses the WHOLE neural population let the same sealed neural '
                    'warm states physically support the body?',
        'parent_analysis': [
            str(C.RT1 / 'PARENT_ANALYSIS_READOUT_BOTTLENECK.zh-CN.md'),
            str(C.RT1 / 'PARENT_ANALYSIS_HOW_DT_MATTERS.zh-CN.md'),
            str(C.RT1 / 'PARENT_NOTE_OUTCOME_FRAGILITY.zh-CN.md'),
        ],
        'interpretation_label': 'OFFLINE ENGINEERING FITTING of the readout interface '
                               '(configurable engineering gains), NOT biological calibration. '
                               'The neurons still generate the command; no non-neural generator '
                               'produces any motion. No claim of walking, autonomous support, '
                               'biological validity or absolute force units, and no claim that the '
                               'brain model is invalid.',
        'readouts': {
            'R0': 'CONTROL: the current readout, clip(limits/100 * (old_matrix @ pool_rates), '
                  '+/- limits). Must reproduce the sealed NUMERICS releases bitwise at both '
                  'neural timesteps.',
            'P1': 'ridge over ALL 12000 filtered cell rates with NO constant term: '
                  'command = clip(W1 @ r_12000, +/- limits). Every part of the torque, including '
                  'its DC level, must come from the neural rate levels themselves.',
            'P2': 'the same 12000 features after explicit offset removal: each cell\'s FIT-WINDOW '
                  'mean is subtracted before the ridge, so the readout carries one fixed bias '
                  'vector b2 = -W2 @ m_fit: command = clip(W2 @ (r_12000 - m_fit), +/- limits) = '
                  'clip(W2 @ r_12000 + b2, +/- limits). This is the declared test of whether slow '
                  'drift / absolute rate level dominates.',
            'P3': 'CONTROL: ridge over the 204 cells the current readout uses, with the SAME '
                  'protocol as P2 (fit-window means removed): command = clip(W3 @ (r_204 - m3) '
                  '+ 0, +/- limits). Used to confirm the parent\'s 0.27-0.41 held-out R^2 range '
                  'inside this pipeline.',
        },
        'why_P1_has_no_constant_term': 'With a free per-joint intercept, "ridge on r" and "ridge '
                                       'on r minus the per-cell fit-window mean" are algebraically '
                                       'the same readout apart from how the target mean is '
                                       'handled, so the P1/P2 contrast would be empty. P1 therefore '
                                       'has no constant term - the same structure as the current '
                                       '204-cell readout, which also has none - and P2 is the only '
                                       'variant that adds a fitted offset. An intercept variant '
                                       'P1i is reported as an INTERFACE-ONLY diagnostic (no '
                                       'physical release) so the parent\'s published 0.881 can be '
                                       'located on the same axis.',
        'fit_protocol': {
            'features': 'the sealed 5 ms-filtered per-cell rate of each of the 12000 cells '
                        '(WARM.npz cell_filtered_rates) at the START of each 1 ms command '
                        'interval: source = vstack([zeros(12000), cell_filtered_rates[:-1]]), the '
                        'exact mirror of the sealed pool convention command_source_rates = '
                        'vstack([zeros(66), pool_rates[:-1]]).  A cell counts as active if its '
                        'filtered rate is non-zero; no cell is selected, dropped or reweighted '
                        'before the ridge.',
            'target': 'teacher applied joint torque, BODY.npz ctrl sampled at the 1 ms command '
                      'boundaries (0.5 ms index 400+2k, k=0..1199 -> 0.200..1.398 s).  This '
                      'convention reproduces the parent published teacher RMS 3.022 (full '
                      'window) and 3.003 (held-out window).',
            'fit_window': [C.FIT_LO, C.FIT_HI],
            'fit_window_s': [0.2, 0.8],
            'held_window': [C.HELD_LO, C.HELD_HI],
            'held_window_s': [0.8, 1.4],
            'method': 'ridge regression, dual form (n=600 intervals << d=12000 features), no '
                      'feature scaling beyond the declared per-cell de-meaning of P2/P3.',
            'alpha_selection': 'alpha is chosen INSIDE the fit window only, by %d contiguous '
                               'blocked folds of the 600 fit intervals: alpha = argmax of the '
                               'mean held-fold R^2 over the fixed grid %r. The held-out 0.8-1.4 s '
                               'window never enters the choice. The whole CV curve is reported.'
                               % (C.CV_FOLDS, C.ALPHAS),
            'heldout_r2': 'R^2 = 1 - SSE/SST over all 42 joints and all held-out intervals, SST '
                          'about the held-out per-joint mean.',
            'single_matrix': 'ONE fitted matrix per readout, taken from the 0.5 ms cache, is used '
                             'for BOTH neural timesteps and all five phases. Nothing is refitted '
                             'per timestep or per phase.',
            'apply': 'at release the readout consumes the runtime\'s own 5 ms-filtered per-cell '
                     'rate vector, updated once per neural step with the same decay '
                     'exp(-neural_dt/0.005) the sealed warm replay used, initialised from the '
                     'sealed warm state at the release phase.',
        },
        'cell_annotation_requirement': 'For P1/P2 the top-50 cells by readout row norm are joined '
                                       'to %s by banc_888_id (NOT root_id: on this tier 12000 ids '
                                       'all resolve as banc_888_id while only 11561 resolve as '
                                       'root_id). Class/type/region are taken only from that '
                                       'table. The explicit question: are the dominant cells '
                                       'sensory/proprioceptive-like or motor-like? The teacher '
                                       'torque is a PD function of body state, so a readout '
                                       'dominated by state-encoding sensory cells can reconstruct '
                                       'torque WITHOUT the connectome generating a motor command.'
                                       % str(C.META),
        'release_grid': {'phases_s': C.PHASES, 'duration_s': C.DURATION,
                         'neural_dt_s': [d for d, _ in C.DTS],
                         'count': len(C.TAGS) * len(C.PHASES) * len(C.DTS),
                         'breakdown': '10 control reproductions of sealed R0 + 10 P1 + 10 P2 '
                                      '+ 10 P3 = 40 releases'},
        'unchanged': 'body model, old_matrix and limits (used only by R0), connectivity '
                     '(ACTUAL_We/ACTUAL_Wi), ConductanceParams, 5 ms pool/cell rate filter, 1 ms '
                     'causal command clock, RH excitatory gain 0.375 applied exactly once at '
                     'release, 42 ideal joint torque actuators, 12.5 us physics step, zero '
                     'qfrc_applied/xfrc_applied, no external force, no pose write, no translator '
                     'actor.',
        'warm_state_reuse': 'The body is PD-driven during warm-up, so the sealed warm neural state '
                            'does not depend on the readout. The sealed WARM_CACHE.pkl states are '
                            'reused unchanged for every readout; that is what makes the comparison '
                            'paired. No warm replay is run and no teacher is re-recorded.',
        'control_requirement': 'R0 must reproduce the sealed NUMERICS releases bitwise for all five '
                               'phases at both neural timesteps (NATIVE.npz rows, BODY.npz arrays, '
                               'NEURAL.npz shared arrays and the RESULT.json J).  If it fails, the '
                               'heavy job stops before any P1/P2/P3 number exists.',
        'evidence_rule_inherited': 'Physical outcomes are not robust to sub-millisecond input '
                                   'phase (0.5 ms: a 0.5 ms input shift moved 1.3 s J from 1.590 '
                                   'to 3.251; at 0.0625 ms the J spread across input shifts is '
                                   '0.926). Therefore no single-phase number is a verdict, no '
                                   'timestep and no model is selected, and every conclusion is '
                                   'reported per phase and only claimed when it is consistent '
                                   'across phases and across both timesteps.',
        'reading_rule': {
            'SUPPORT_ACHIEVED': 'some readout passes all D1-D8 (all 9 D checks) in at least 4 of '
                                'the 5 release phases at the 0.5 ms timestep AND in at least 4 of '
                                'the 5 release phases at the 0.0625 ms timestep. Nothing weaker '
                                'counts as support.',
            'PARTIAL': 'no readout achieves SUPPORT_ACHIEVED, but some readout strictly increases '
                       'the D-gate pass count (number of the 9 D checks that pass, per phase) '
                       'versus the sealed R0 baseline in at least 4 of 5 phases at the 0.5 ms '
                       'timestep AND in at least 4 of 5 phases at the 0.0625 ms timestep.',
            'NO_IMPROVEMENT': 'otherwise. Reported as such, without softening.',
            'scope_note': 'The rule is per readout and then aggregated: the headline verdict is '
                          'SUPPORT_ACHIEVED if ANY readout satisfies it. All three readouts are '
                          'reported individually so a reader can see whether the result needed one '
                          'readout or all three.',
            'descriptive': 'Per readout, per timestep and per phase: J, foot support mean, minimum '
                           'COM height, first positive non-foot contact, all 9 D booleans and the '
                           'D pass count, plus the per-phase spread (min/median/max). A mean is '
                           'never the verdict.',
            'no_selection': 'No timestep, readout or phase is selected as the answer; the 0.5 ms '
                            'and 0.0625 ms results are both reported and any disagreement between '
                            'them is itself a reported result.',
        },
        'reproducibility_requirement': 'For every release the applied command must be recomputed '
                                       'from the saved arrays alone (unpack the saved 12000-bit '
                                       'spike words, re-run the 5 ms filter from the sealed warm '
                                       'state, apply the saved W/b, clip) and must match the '
                                       'recorded applied command. Checked in verify.py, not by the '
                                       'runner.',
        'static_report_requirement': 'Per readout: correlation with the teacher torque (overall and '
                                     'the per-joint median), sign agreement, command RMS versus '
                                     'teacher RMS, saturation fraction, and held-out R^2; plus the '
                                     'chosen alpha and the CV curve; plus the cell-annotation '
                                     'decomposition of the dominant weights.',
        'inherited_measured_facts': {
            'source': 'parent, from saved arrays; reproduced here as checks, not re-litigated',
            'current_readout_r_vs_teacher': -0.0831,
            'current_readout_saturation_fraction': 0.784,
            'current_readout_held_r2_66_pool_rates': 0.328,
            'readout_fit_interface_r': 0.889, 'readout_fit_saturation': 0.00065,
            'readout_fit_physical_releases_passing_D_gates': '0 of 30',
            'readout_fit_min_com_range': [0.49, 0.79], 'D5_band': 1.1375, 'D8_floor': 0.9658,
            'ridge_all_12000_held_r2': 0.881, 'ridge_alpha': 100.0,
            'ridge_all_12000_held_r2_after_100ms_detrend': 0.841,
            'shuffle_control_r2': -0.405,
            'ridge_204_motor_cells_held_r2_range': [0.27, 0.41],
            'top200_single_cells_held_r2': 0.354, 'top200_overlap_with_204': 0,
            'teacher_torque_rms': 3.003,
            'dt_0p5ms_J_input_shift': [1.590, 3.251],
            'dt_0p0625ms_J_spread_across_shifts': 0.926,
        },
        'budget': dict(heavy_jobs=1, releases=40, warm_replays=0, teacher_recordings=0,
                       new_neural_replay_seconds=0.0, physics_release_seconds=8.0,
                       expected_wall_s_per_release=4.0),
        'abort_conditions': ['an input hash changes', 'R0 is not bitwise identical to the sealed '
                             'release', 'a needed sealed warm cache is missing', 'held-out R^2 of '
                             'every whole-population readout is below 0.5 (the go/no-go signal)',
                             'non-finite state or a MuJoCo warning',
                             'another heavy python job is already running'],
        'go_no_go': 'The offline fit is run FIRST and its held-out R^2 is reported to the parent '
                    'before any physics. Below 0.5 held-out R^2 for P1 and P2 the physical grid is '
                    'still run only if the parent says so; the default is to report and stop.',
        'units': 'absolute forces and SI units NOT_ESTABLISHED; values are dimensionless ratios of '
                 'the sealed model only',
        'boundaries': ['do not claim walking', 'do not claim autonomous support',
                       'do not claim biological validity or absolute force units',
                       'do not claim the brain model is invalid',
                       'do not select a timestep or a model',
                       'offline engineering fitting, not biological calibration'],
        'sealed_dirs_written': 'none; every write goes to this task output directory',
        'forbidden_writes': [str(C.R / 'GOAL_TRACKER.zh-CN.md'), str(C.R / 'engine')],
        'importance': '核心必需',
    }
    C.dump(C.O / 'PREREGISTRATION.json', cfg)

    paths = frozen_inputs()
    before = {p: C.sha(p) for p in paths}
    C.dump(C.O / 'INPUT_HASHES_BEFORE.json', before)
    srcs = sorted(C.SRC.glob('*.py'))
    C.dump(C.O / 'CHANGE_LEDGER.json', dict(
        old=[dict(path=p, sha256=h) for p, h in before.items()],
        new=[dict(path=str(p), sha256=C.sha(p),
                  reason='New frozen source for the whole-population readout test; no sealed '
                         'file, no engine/ module and no GOAL_TRACKER is modified.')
             for p in srcs],
        note='Sealed read-only inputs are hashed before and after; the heavy job asserts equality.'))
    C.dump(C.O / 'SOURCE_FREEZE.json', dict(
        source_dir=str(C.SRC),
        files={str(p): C.sha(p) for p in srcs},
        frozen=True,
        note='Code was written and hashed before the heavy job started; verify.py re-hashes it.'))
    print('PREREGISTERED', len(before), 'inputs', flush=True)


if __name__ == '__main__':
    main()
