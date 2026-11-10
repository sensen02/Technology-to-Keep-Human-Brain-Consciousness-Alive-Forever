"""Freeze the experiment design, the input hashes and the reading rule. Cheap, no simulation.

Run once. Refuses to overwrite an existing PREREGISTRATION.json.
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
                  C.CONFIGS / tag / 'PARAMETERS.json', C.CONFIGS / tag / 'WARM_IDENTITY.json']
        for ph in C.PHASES:
            folder = C.SEALED_RUNS / C.sealed_label(tag, ph)
            paths += [folder / 'RESULT.json', folder / 'NATIVE.npz',
                      folder / 'BODY.npz', folder / 'NEURAL.npz']
    paths += [C.NUMERICS_SUMMARY, C.IDENTITIES,
              C.TEACHER / 'BODY.npz', C.TEACHER / 'NATIVE.npz', C.TEACHER / 'TEACHER.npz']
    return list(dict.fromkeys(str(p) for p in paths))


def main():
    if (C.O / 'PREREGISTRATION.json').exists():
        raise SystemExit('PREREGISTRATION.json already exists; refusing to overwrite')
    cfg = {
        'stage': 'S5 readout-interface correction: offline linear readout replacing the '
                 'current saturating readout, applied to the SAME sealed neural warm states. '
                 'Zero new warm replays, zero new teacher recordings.',
        'parent_analysis': [
            str(C.RT1 / 'PARENT_ANALYSIS_READOUT_BOTTLENECK.zh-CN.md'),
            str(C.RT1 / 'PARENT_ANALYSIS_HOW_DT_MATTERS.zh-CN.md'),
        ],
        'interpretation_label': 'OFFLINE ENGINEERING FITTING of the readout interface '
                               '(configurable engineering gains), NOT biological calibration. '
                               'The neurons still generate the command; no non-neural generator '
                               'produces any motion. No claim of walking, autonomous support, '
                               'biological validity or absolute force units.',
        'readouts': {
            'R0': 'current readout: clip(limits/100 * (old_matrix @ rates), +/- limits). '
                  'Control; must reproduce the sealed releases bitwise.',
            'R1': 'least-squares linear readout fitted on 0.2-0.8 s only: '
                  'command = clip(W_fit @ rates + b_fit, +/- limits). No other change.',
            'R2': 'R1 with one declared engineering gain s: '
                  'command = clip(s * (W_fit @ rates + b_fit), +/- limits), where '
                  's = RMS(teacher torque over 0.2-0.8 s) / RMS(unclipped R1 command over 0.2-0.8 s), '
                  'both elementwise over all 42 joints and all 600 intervals.',
        },
        'fit_protocol': {
            'features': 'sealed WARM.npz command_source_rates (1200x66) of the 0.5 ms neural '
                        'timestep cache; the same causal 1 ms command clock the runtime uses '
                        '(interval-start source rates; command at interval k uses rates completed '
                        'at the end of interval k-1).',
            'target': 'teacher applied joint torque, BODY.npz ctrl sampled at the 1 ms command '
                      'boundaries: 0.5 ms index 400+2k, k=0..1199 -> 0.200..1.398 s. '
                      'This convention reproduces the parent published teacher RMS 3.022 '
                      '(full window) and 3.003 (held-out window).',
            'fit_window': [C.FIT_LO, C.FIT_HI],
            'fit_window_s': [0.2, 0.8],
            'held_window': [C.HELD_LO, C.HELD_HI],
            'held_window_s': [0.8, 1.4],
            'method': 'ordinary least squares with a per-joint intercept (numpy.linalg.lstsq, '
                      'rcond=None). No ridge, no feature selection, no per-joint window tuning.',
            'single_matrix': 'ONE fitted 42x66 matrix + bias is used for BOTH neural timesteps. '
                             'The readout is not refitted per timestep and not refitted per phase.',
            'heldout_r2': 'R^2 = 1 - SSE/SST computed over all 42 joints and all held-out '
                          'intervals, SST about the held-out per-joint mean.',
        },
        'release_grid': {'phases_s': C.PHASES, 'duration_s': C.DURATION,
                         'neural_dt_s': [d for d, _ in C.DTS],
                         'count': 3 * len(C.PHASES) * len(C.DTS),
                         'breakdown': '10 reproductions of sealed R0 + 10 R1 + 10 R2 = 30 releases'},
        'unchanged': 'body model, old_matrix and limits (used only by R0), connectivity, '
                     'ConductanceParams, 5 ms pool rate filter, 1 ms causal command clock, '
                     'RH excitatory gain 0.375 applied exactly once at release, 42 ideal joint '
                     'torque actuators, 12.5 us physics step, zero qfrc_applied/xfrc_applied, '
                     'no external force, no pose write, no translator actor.',
        'warm_state_reuse': 'The body is PD-driven during warm-up, so the warm neural state does '
                            'not depend on the readout. The sealed WARM_CACHE.pkl states are '
                            'therefore reused unchanged for every readout; this is what makes the '
                            'comparison paired. No warm replay is run.',
        'control_requirement': 'R0 must reproduce the sealed NUMERICS releases bitwise for all '
                               'five phases at both neural timesteps: NATIVE.npz rows, BODY.npz '
                               'arrays, NEURAL.npz arrays and the RESULT.json J must all match '
                               'exactly. If this fails, the heavy job stops and reports; no R1/R2 '
                               'number is reported as a result.',
        'evidence_rule_inherited': 'Physical outcomes are not robust to sub-millisecond input '
                                   'phase (0.5 ms: a 0.5 ms input shift moved 1.3 s J from 1.590 '
                                   'to 3.251; at 0.0625 ms the J spread across input shifts is '
                                   '0.926). Therefore no single-phase number is a verdict, no '
                                   'timestep and no model is selected, and every conclusion is '
                                   'reported per phase and only claimed if it holds across phases.',
        'reading_rule': {
            'primary_A_physical': 'A readout counts as an improvement ONLY IF BOTH hold: '
                                  '(A1) J_new < J_R0 in at least 4 of 5 phases at the 0.5 ms '
                                  'timestep AND in at least 4 of 5 phases at the 0.0625 ms '
                                  'timestep (strict inequality, same phase axis); and '
                                  '(A2) the D-gate pass count (number of phases whose full '
                                  'simulated release passes all D1-D8/D4b checks) strictly '
                                  'increases over R0 at BOTH timesteps.',
            'secondary_B_interface_only': 'If A1 holds but A2 does not, the result is reported as '
                                          '"readout-level improvement only; the physical support '
                                          'condition is still not met", and explicitly NOT as '
                                          'the body being supported.',
            'descriptive': 'Per readout, per timestep, per phase spread (min/median/max) of J, '
                           'foot support and minimum COM are always reported; a mean alone is '
                           'never the verdict. Ancillary metrics (foot support mean, min COM, '
                           'first positive non-foot contact, per-gate counts) are reported per '
                           'phase with the same no-single-number rule.',
            'no_selection': 'No timestep, readout or phase is selected as the answer. The '
                            '0.5 ms and 0.0625 ms timesteps are both reported; disagreement '
                            'between them is itself a reported result.',
        },
        'static_report_requirement': 'Per-joint command range versus the joint limit, saturation '
                                     'fraction, correlation with the teacher torque (overall, '
                                     'per-joint distribution, sign agreement), and readout '
                                     'stability: fit on 0.2-0.8 s versus fit on 0.8-1.4 s and '
                                     'compare the two matrices.',
        'inherited_measured_facts': {
            'source': 'parent, from saved arrays; not re-litigated',
            'R0_r_vs_teacher': -0.0831, 'R0_perjoint_median_r': 0.0052,
            'R0_sign_agreement': 0.4653, 'R0_command_rms_ratio': 2.393,
            'R0_saturation_fraction': 0.784,
            'linear_readout_train_r2': 0.542, 'linear_readout_held_r2': 0.326,
            'teacher_torque_rms': 3.003,
            'dt_0p5ms_J_input_shift': [1.590, 3.251], 'dt_0p0625ms_J_spread_across_shifts': 0.926,
        },
        'budget': dict(heavy_jobs=1, releases=30, warm_replays=0, teacher_recordings=0,
                       new_neural_replay_seconds=0.0, physics_release_seconds=6.0,
                       expected_wall_s_per_release=4.0),
        'abort_conditions': ['an input hash changes', 'R0 is not bitwise identical to the sealed '
                             'release', 'a needed sealed warm cache is missing',
                             'non-finite state or a MuJoCo warning',
                             'another heavy python job is already running'],
        'units': 'absolute forces and SI units NOT_ESTABLISHED; values are dimensionless ratios '
                 'of the sealed model only',
        'boundaries': ['do not claim walking', 'do not claim autonomous support',
                       'do not claim biological validity or absolute force units',
                       'do not claim the brain model is invalid',
                       'do not select a timestep or a model',
                       'offline engineering fitting, not biological calibration'],
        'sealed_dirs_written': 'none; every write goes to this task output directory',
        'importance': '核心必需',
    }
    C.dump(C.O / 'PREREGISTRATION.json', cfg)

    paths = frozen_inputs()
    before = {p: C.sha(p) for p in paths}
    C.dump(C.O / 'INPUT_HASHES_BEFORE.json', before)
    C.dump(C.O / 'CHANGE_LEDGER.json', dict(
        old=[dict(path=p, sha256=h) for p, h in before.items()],
        new=[dict(path=str(p), sha256=C.sha(p),
                  reason='New frozen source for the readout-correction experiment; no sealed '
                         'file and no engine/ module is modified.')
             for p in sorted(C.SRC.glob('*.py'))],
        note='Sealed read-only inputs are hashed before and after; the heavy job asserts equality.'))
    C.dump(C.O / 'SOURCE_FREEZE.json', dict(
        source_dir=str(C.SRC),
        files={str(p): C.sha(p) for p in sorted(C.SRC.glob('*.py'))},
        frozen=True,
        note='Code was written and hashed before the heavy job started; verify.py re-hashes it.'))
    print('PREREGISTERED', len(before), 'inputs', flush=True)


if __name__ == '__main__':
    main()
