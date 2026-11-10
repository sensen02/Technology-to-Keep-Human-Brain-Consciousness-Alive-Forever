"""Step 1 of the gain sweep: freeze the pre-registration and the input hashes.

Nothing is simulated here. Run order:
    prereg.py  ->  static.py (cheap checks)  ->  offline.py (4 warms + rate/information
    diagnostics, the go/no-go the parent gets BEFORE any physics)  ->  run.py (single serial
    heavy job: the releases)  ->  verify.py  ->  report.py
"""
import sys
from pathlib import Path

import numpy as np

import common as C

sys.path[:0] = [str(C.BASE), str(C.R), str(C.R / 'tools')]
import runtime as rt

js, sha, dump = C.js, C.sha, C.dump


def register():
    """Everything that a reader must be able to check was not moved after the fact."""
    from engine.embodied.sensory_channels import ChannelConfig
    from engine.neural_cond import ConductanceParams
    cfg = ChannelConfig()
    assert cfg.current_nA_per_unit == C.REF_GAIN, cfg.current_nA_per_unit
    pre = dict(
        stage='S5/P2 gain sweep: sense-drive gain vs sparseness, motor information and support',
        question='Does lowering the sensory encoder gain bring this spiking network into a '
                 'sparse operating regime and restore task-relevant motor information and '
                 'physical support?',
        what_changes='exactly one number: ChannelConfig.current_nA_per_unit, at '
                     f'{C.GAINS} (the current value is {C.REF_GAIN})',
        what_does_not_change=['reference_force', 'reference_angle_rad',
                              'reference_velocity_rad_s', 'per-channel weights',
                              'connectivity matrices A', 'ConductanceParams defaults',
                              'physical body / model XML', 'readout limits',
                              '5 ms rate filter', '1 ms causal command clock',
                              'RH excitation gain .375 applied once at release',
                              'physics timestep 12.5 us', 'teacher trajectory'],
        boundary='changing the encoder gain is an ENGINEERING parameter change, NOT biological '
                 'calibration; absolute currents, forces and SI units are NOT_ESTABLISHED; no '
                 'walking, autonomous support or biological validity is claimed; the brain model '
                 'is NOT declared invalid; no timestep and no model is selected',
        gains_nA_per_unit=C.GAINS, reference_gain=C.REF_GAIN,
        neural_dt_s=C.DT_MAIN,
        second_timestep_s=C.DT_FINE,
        command_dt_s=C.CDT, duration_s=C.DURATION, physics_dt_s=C.PHYSICS_DT,
        rate_filter_tau_s=C.FILTER_TAU, RH_exc_gain_at_release=C.RH_GAIN,
        release_phases_s=C.PHASES,
        neuron_parameters=vars(ConductanceParams()),
        # ---- (a) currents
        currents=dict(
            source=str(C.CURRENTS), recorded_at_gain=C.REF_GAIN,
            rule='currents(g) = TRUE_CURRENTS * (g / %g); at g = %g the factor is exactly 1.0 '
                 'so the stream is the recorded array bitwise' % (C.REF_GAIN, C.REF_GAIN),
            declaration='for g != %g this is a LINEAR RESCALING of the recorded encoder output, '
                        'not a re-recording and not a new teacher run; the encoder multiplies '
                        'each channel quantity by the gain exactly once, which makes the '
                        'rescaling exact up to float rounding (checked statically)' % C.REF_GAIN,
            teacher_recordings_in_this_task=0,
            gain_at_0p3_reproduction_gate='the 0.5 ms warm replay at gain 0.3 must reproduce the '
                                          'sealed numerics/dt_0p5ms WARM.npz and WARM_CACHE.pkl '
                                          'bitwise (all 17 phases, all network variables, rates, '
                                          'spike-history, packed spikes); failure aborts the task'),
        # ---- (b) warm replay
        warm=dict(per_gain=1, timesteps='0.5 ms for all four gains; additionally 0.0625 ms for '
                                        'the top-two gains (see second_timestep_rule)',
                  start='rest: v=rest_mV, ge=gi=0, adaptation=0, no spikes, t_ms=0; '
                        'ConductanceNetwork(12000) with the sealed A matrices',
                  commands=1200, window_s=[0.2, 1.4],
                  physics_states='the 17 physical handoff states and the assisted reference '
                                 'statistics are taken from the sealed teacher cache '
                                 '(DATA/PHYSICAL_CACHE.pkl). The teacher body is driven by the '
                                 'teacher PD law only and the network does not touch it, so those '
                                 'states are gain-independent; this task makes no teacher '
                                 'recording',
                  saved=['17 physical handoff states (deep-copied network, rates, spike-history, '
                         'MuJoCo integration state, assisted support/COM statistics)',
                         'all-12000 spikes (packed) for the 1200 commands',
                         '12000 filtered cell rates at every 1 ms command boundary',
                         '66 pool rates and per-leg rates']),
        # ---- (c) information metric
        information=dict(
            target='teacher applied joint torque at the 1200 command boundaries '
                   '(BODY.npz ctrl rows 400+2k, 0.2-1.4 s)',
            features='that gain own 5 ms-filtered per-cell rates at the command-interval start '
                     '(causal: zeros at 0.2 s, then the previous interval-end state)',
            fit_window='intervals 0-599 (0.2-0.8 s)', held_window='intervals 600-1199 '
                                                                  '(0.8-1.4 s)',
            estimator='ridge, no separate constant column: per-cell fit-window mean removed '
                      'from the features and the per-joint fit-window mean torque added back as '
                      'one fixed offset (the previous stage P2/P3 shape)',
            alpha_rule='alpha = gamma * s with s = mean squared feature norm of the fit window; '
                       'gamma grid %r; gamma chosen by %d contiguous blocked CV folds INSIDE the '
                       'fit window only; the held-out window never enters the choice' %
                       (C.GAMMAS, C.CV_FOLDS),
            feature_sets=dict(motor_204='the 204 motor readout cells (all annotated '
                                        'super_class=motor)',
                              all_12000='every tier cell'),
            reporter=['R^2 in the fit window', 'R^2 in the held-out window', 'chosen gamma and '
                      'alpha', 'offset RMS', 'feature mean/RMS Hz',
                      'same estimator on the sealed gain-0.3 trace must reproduce the parent '
                      'numbers 0.43 (204 cells) and 0.86 (12000 cells)'],
            inherit_check='both R^2 are always reported; the all-cell value is the control for '
                          'whether the motor-cell value merely follows the whole population'),
        # ---- (d) rate diagnostics
        diagnostics=dict(
            mean_rate_per_cell='mean of the 5 ms filtered rate over the 1200 interval ends '
                               '(this definition reproduces the parent numbers 163.8 Hz sensory, '
                               '94.5 Hz motor, 120.5 Hz on the 204 readout cells)',
            temporal_cv='per-cell std/mean over the window; the reported motor CV is the MEDIAN '
                        'over the 204 readout cells (reproduces the parent 0.72)',
            silent='zero spikes in the 1.2 s window',
            per_class='super_class from banc_888_meta.feather joined by banc_888_id; cross-checked '
                      'against data/flywire/banc_connectome.npz ann_super_class by root_id',
            dense_relief_rule=dict(
                frozen_before_running=True,
                conditions={
                    'A': 'readout-204 mean rate <= %.1f Hz' % C.DENSE_READOUT_MEAN_HZ,
                    'B': 'share of the 12000 cells above 100 Hz <= %.2f' % C.DENSE_HYPER_SHARE,
                    'C': 'share of the 12000 cells with zero spikes <= %.2f' %
                         C.DENSE_SILENT_SHARE,
                    'D': 'population mean rate >= %.1f Hz (the network must still fire)' %
                         C.DENSE_MIN_MEAN_HZ},
                statement='DENSE_REGIME_RELIEVED only if A and B and C and D hold; every component '
                          'is reported regardless, and the raw per-class numbers are the primary '
                          'evidence')),
        # ---- (e) physical releases
        physical=dict(
            readout='ridge on the 204 motor cells only (the anatomically correct path): command = '
                    'clip(W @ (r_204 - mean_fit) + b, +/- limits); W, b, mean_fit are that gain '
                    'own fit, i.e. the same matrix reported in the information table',
            whole_population_control='for the single best-information gain only: the same five '
                                     'releases with the 12000-cell ridge readout',
            phases=C.PHASES, duration_s=C.DURATION,
            gates='unchanged dynamic_support_metrics.evaluate_native D1-D8 plus the sealed D1 '
                  'restatement (max positive non-foot ground force == 0); D8 floor = 0.8*H0 = '
                  '0.9658',
            objective='unchanged train_dynamic_support.objective with RELEASE set to the '
                      '-.2 s run length',
            no_force='qfrc_applied and xfrc_applied identically zero during every release; no '
                     'pose write, no external force, no assistance; RH excitatory gain .375 '
                     'applied exactly once at release',
            fragility='no single phase and no single timestep is a verdict; every conclusion is '
                      'reported per phase and the 0.0625 ms arm is labelled a single-timestep '
                      'observation'),
        # ---- second timestep rule (frozen before running)
        second_timestep_rule=dict(
            rule='rank the four gains by motor-cell HELD-OUT R^2 at 0.5 ms (higher is better, '
                 'ties broken by the smaller gain); run the 0.0625 ms arm for at most the top two',
            timing='the ranking is read from the offline part, i.e. before any physics',
            cost='2 extra warms at 0.0625 ms + 5 releases per selected gain',
            why_extra_warms='the neural state depends on the timestep, so a 0.0625 ms release '
                            'starts from its own 0.0625 ms warm state; the physical handoff '
                            'states are the same gain-independent sealed teacher states'),
        best_information_gain_rule='the rank-1 gain by motor-cell held-out R^2 at 0.5 ms; its '
                                   'five releases are repeated with the whole-population readout',
        # ---- budget and order
        budget=dict(releases_max=C.MAX_RELEASES,
                    releases_planned='4 gains x 5 phases (motor readout, 0.5 ms) = 20, + 5 '
                                     'whole-population releases for the best gain, + up to 10 '
                                     'at 0.0625 ms for the top two gains = at most 35',
                    warms_planned='%d (4 gains at 0.5 ms + 2 gains at 0.0625 ms)' % C.MAX_WARMS,
                    teacher_recordings=0,
                    order='offline part (warms, rate diagnostics, information R^2) is reported to '
                          'the parent as a go/no-go signal BEFORE the release job starts'),
        # ---- reading rule, frozen
        reading_rule=dict(
            INFORMATION_RESTORED='any gain has motor-cell held-out R^2 >= %.1f (versus %.2f at '
                                 'the current gain) AND its all-cell R^2 is reported so the '
                                 'reader can see whether it simply inherits the population value'
                                 % (C.RESTORED_MOTOR_R2, C.BASELINE_MOTOR_R2),
            DRIVE_DEPENDENT_SUPPORT='any gain passes ALL D1-D8 (and the D1 restatement) in >= %d '
                                    'of 5 phases at the timestep tested' % C.SUPPORT_PHASES_NEEDED,
            otherwise='NO_SUPPORT_IMPROVEMENT, stated as such',
            no_threshold_may_move_after_the_fact=True,
            single_phase_never_a_verdict=True),
        selection=dict(timestep_selected=None, model_selected=None, gain_selected=None,
                       note='the sweep is descriptive; the task asks whether a regime and the '
                            'support condition exist, not which setting is adopted'),
        importance='核心必需')
    dump(C.O / 'PREREGISTRATION.json', pre)
    print('PREREGISTERED', flush=True)


def hashes():
    paths = list(dict.fromkeys(rt.files() + [
        C.CURRENTS, C.PHYSICAL, C.TEACHER / 'BODY.npz', C.TEACHER / 'TEACHER.npz',
        C.TEACHER / 'NATIVE.npz', C.META, C.FLYWIRE, C.IDENTITIES_NPZ,
        C.TRANSLATOR_IDENTITIES, C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM.npz',
        C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM_CACHE.pkl',
        C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM_IDENTITY.json',
        C.SEALED_CONFIGS / 'dt_0p0625ms' / 'WARM.npz',
        C.SEALED_CONFIGS / 'dt_0p0625ms' / 'WARM_CACHE.pkl',
        C.SEALED_NUMERICS / 'PREREGISTRATION.json', C.SEALED_NUMERICS / 'SUMMARY.json',
        C.POPREADOUT / 'READOUT_STATIC.json', C.POPREADOUT / 'READOUT_FIT.npz',
        C.POPREADOUT / 'SUMMARY.json', C.POPREADOUT / 'CELL_ANNOTATION.json',
        C.O / 'PREREGISTRATION.json']))
    # Source files are frozen separately (SOURCE_FREEZE) with a declared-fix mechanism, so an
    # unrelated edit of this task's own code cannot masquerade as a changed sealed input.
    dump(C.O / 'INPUT_HASHES_BEFORE.json', {str(p): sha(p) for p in paths})
    dump(C.O / 'SOURCE_FREEZE.json',
         dict(files={str(p): sha(p) for p in sorted(C.SRC.glob('*.py'))},
              note='every source file of this task as frozen before the first simulation; a '
                   'later change must be declared in CHANGE_LEDGER.tool_fixes with its reason '
                   'and must be detectable by verify.py'))
    dump(C.O / 'CHANGE_LEDGER.json', dict(
        old=[dict(path=str(p), sha256=sha(p), role='read-only input, byte-identical before and '
                                                   'after this task')
             for p in paths if C.SRC not in p.parents and p.parent != C.O],
        new=[dict(path=str(p), sha256=sha(p),
                  reason='new file of this task (engineering parameter sweep); no sealed module '
                         'is edited: the encoder gain is passed as a construction argument')
             for p in sorted(C.SRC.glob('*.py'))],
        tool_fixes=[],
        note='GOAL_TRACKER.zh-CN.md, engine/ and every sealed directory are untouched by this '
             'task; the encoder gain is supplied as a separate ChannelConfig instance'))
    print('HASHED', len(paths), flush=True)


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'prereg'
    C.O.mkdir(parents=True, exist_ok=True)
    if action == 'prereg':
        assert not (C.O / 'PREREGISTRATION.json').exists(), 'already pre-registered'
        register()
    elif action == 'hash':
        assert (C.O / 'PREREGISTRATION.json').exists(), 'run prereg first'
        hashes()
    else:
        raise SystemExit('unknown action ' + action)


if __name__ == '__main__':
    main()
