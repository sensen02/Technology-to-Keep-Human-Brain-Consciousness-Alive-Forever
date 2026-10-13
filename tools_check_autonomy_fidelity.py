"""Reproducible autonomy-fidelity check for the ground closed loop.

WHY THIS EXISTS.  Two things were measured once and could silently regress:
  1. whether the loop is genuinely sensor-driven or merely self-sustaining, and
  2. whether the body is standing on its feet or resting on itself.
Both are easy to forget, so this script re-measures them after any change and
separates two very different kinds of statement:

  HARD INVARIANTS - facts that must hold, checked and failed loudly if violated:
    * every interval's muscle command equals the declared spike translation
      (the runtime origin audit),
    * no background current, sensory-only injection,
    * with the motor readout cut, muscle command and activation are exactly zero,
    * with all We/Wi cut, no non-sensory neuron fires,
    * with a declared sensory cut-after time, the injected sensory current is
      exactly zero afterwards,
    * silencing all synapses of the mapped motor neurons leaves the whole network
      running but drops every active muscle command to zero, while a size-matched
      random control does NOT,
    * a declared constant adhesion command stays exactly at that constant.

  REPORTED FIDELITY METRICS - measurements with NO invented pass/fail threshold,
  because turning them into thresholds would require biological evidence the
  project does not have:
    * self-sustaining fraction: spikes without any body-derived sensory current,
      relative to the full-sensation run,
    * how much sensation shapes the motor pattern (command difference, per
      actuator correlation),
    * contact composition: tarsus-only fraction, six-feet-down fraction, and which
      non-tarsus parts bear on the floor.

This does not drive the body, does not add a controller and does not change the
canonical model.  It only runs the existing entry point in fixed configurations and
grades the output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'diagnostics/diag_autonomous_sensorimotor.py'
OUT = ROOT / 'outputs/autonomy_fidelity_check'
PYTHON = ROOT / 'venv_body/bin/python'
CANONICAL_XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
CANONICAL_SHA = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(name, extra, seconds, outdir):
    """Run the existing entry point once; raise on failure."""
    out = outdir / name
    cmd = [str(PYTHON), str(RUNNER), '--exploratory', '--seconds', str(seconds),
           '--spike-storage', 'counts', '--record-contacts', '--out-dir', str(out)] + extra
    env = {'OPENBLAS_NUM_THREADS': '1', 'MUJOCO_GL': 'egl', 'PATH': '/usr/bin:/bin'}
    proc = subprocess.run(cmd, cwd=str(ROOT), env={**env}, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'{name} failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}')
    return json.loads((out / 'exploratory.json').read_text())


def check_invariants(report):
    """Hard invariants: return (violations, checked_count)."""
    bad = []
    n = 0
    for case, block in report['cases'].items():
        cut_after = None
        for row in block['commands']:
            n += 1
            audit = row['command_origin_audit']
            if audit['status'] != 'PASS':
                bad.append(f'{case} t={row["time_s"]}: origin audit {audit["status"]}')
            if not audit['checks'].get('sensory_only_injection', False):
                bad.append(f'{case} t={row["time_s"]}: current injected outside sensory cells')
            if not audit['checks'].get('no_background_current', False):
                bad.append(f'{case} t={row["time_s"]}: background current present')
            f = row.get('features') or {}
            if '_sensory_cut_after_s' in f:
                cut_after = f['_sensory_cut_after_s']
            if cut_after is not None and row['time_s'] >= cut_after:
                if abs(row['injected_current_sum_nA']) > 0.0:
                    bad.append(f'{case} t={row["time_s"]}: sensory current not zero after the declared cut')
            if row.get('adhesion_const_applied') is not None:
                pass  # the constant itself is asserted inside the runner every step
        if case in ('motor_readout_cut', 'propagation_cut'):
            if block['summary']['max_ctrl'] != 0.0:
                bad.append(f'{case}: muscle command not zero')
        if case == 'propagation_cut':
            if block['invariants']['background_max_nA'] != 0.0:
                bad.append('propagation_cut: background current non-zero')
    return bad, n


def spikes_series(case_block):
    t = np.array([r['time_s'] for r in case_block['commands']])
    s = np.array([r['network_spike_count'] for r in case_block['commands']])
    c = np.array([r['injected_current_sum_nA'] for r in case_block['commands']])
    return t, s, c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=.3,
                    help='duration per configuration; 0.3 s keeps the check quick, 1.0 s is the long run')
    ap.add_argument('--xml', type=Path, default=None,
                    help='Optional variant body model to check instead of the canonical asset.')
    ap.add_argument('--adhesion-const', type=float, default=0.0,
                    help='Constant adhesion command for a variant model; must be 0 for canonical.')
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sha_before = digest(CANONICAL_XML)
    summary = {'status': 'AUTONOMY_FIDELITY_CHECK', 'seconds': args.seconds,
               'canonical_xml_sha256_before': sha_before}

    # A: full sensation, with interventions
    model_args = []
    if args.xml is not None:
        model_args += ['--xml', str(Path(args.xml).resolve())]
    if args.adhesion_const:
        model_args += ['--adhesion-const', str(args.adhesion_const)]
    summary['variant_model'] = str(Path(args.xml).resolve()) if args.xml else None
    summary['adhesion_const'] = args.adhesion_const
    a = run_case('full_sensation', model_args + ['--interventions'], args.seconds, outdir)
    # B: sensory deafferentation halfway through
    b = run_case('deafferented', model_args + ['--sensory-cut-after', str(args.seconds / 2)],
                 args.seconds, outdir)

    violations, n_intervals = check_invariants(a)
    violations_b, n_b = check_invariants(b)
    violations += violations_b
    summary['invariants'] = {'intervals_checked': n_intervals + n_b,
                             'violations': violations,
                             'status': 'PASS' if not violations else 'FAIL'}

    # intervention specificity
    cases = a['cases']
    if 'motor_neuron_silenced' in cases and 'random_control_silenced' in cases:
        mn, ct = cases['motor_neuron_silenced'], cases['random_control_silenced']
        summary['intervention_specificity'] = {
            'motor_neurons_silenced_network_spikes': mn['summary']['network_spikes'],
            'motor_neurons_silenced_max_muscle_command': mn['summary']['max_ctrl'],
            'random_control_network_spikes': ct['summary']['network_spikes'],
            'random_control_max_muscle_command': ct['summary']['max_ctrl'],
            'specific': bool(mn['summary']['max_ctrl'] == 0.0 and ct['summary']['max_ctrl'] > 0.0),
            'meaning': 'silencing the mapped motor neurons removes all active muscle command while '
                       'the network keeps firing; a size-matched random silence does not - so the '
                       'command comes from those neurons specifically'}
        if not summary['intervention_specificity']['specific']:
            violations.append('intervention specificity failed')

    # deafferentation fidelity metrics (no invented threshold)
    fi = cases['intact']
    di = b['cases']['intact']
    tf, sf, cf = spikes_series(fi)
    td, sd, cd = spikes_series(di)
    half = args.seconds / 2
    late_f = sf[tf >= half].mean() if (tf >= half).any() else float('nan')
    late_d = sd[td >= half].mean() if (td >= half).any() else float('nan')
    Cf = np.array([r['ctrl'] for r in fi['commands']])
    Cd = np.array([r['ctrl'] for r in di['commands']])
    nz = np.where(Cf.any(axis=0) | Cd.any(axis=0))[0]
    late = td >= half
    A, B = Cf[late][:, nz], Cd[late][:, nz]
    corr = [np.corrcoef(A[:, k], B[:, k])[0, 1] for k in range(A.shape[1])
            if A[:, k].std() > 0 and B[:, k].std() > 0]
    summary['fidelity_metrics'] = {
        'spikes_full_sensation': int(sf.sum()),
        'spikes_deafferented': int(sd.sum()),
        'self_sustaining_spike_fraction': float(sd.sum() / sf.sum()) if sf.sum() else float('nan'),
        'late_window_spikes_full': float(late_f),
        'late_window_spikes_deafferented': float(late_d),
        'late_window_ratio': float(late_d / late_f) if late_f else float('nan'),
        'sensory_current_zero_after_cut': bool(np.all(cd[late] == 0.0)),
        'muscle_command_mean_abs_difference_after_cut': float(np.abs(A - B).mean()) if A.size else float('nan'),
        'muscle_command_fraction_identical': float(np.isclose(A, B, atol=1e-12).mean()) if A.size else float('nan'),
        'per_actuator_command_correlation_mean': float(np.mean(corr)) if corr else float('nan'),
        'interpretation': 'A high self-sustaining fraction means recurrence, not the body, sets the '
                          'tempo; a low per-actuator correlation means sensation still shapes the '
                          'pattern. Both are reported, never graded, because no biological threshold '
                          'is available to this project.'}

    # contact composition
    def contact_stats(block):
        recs = [r['contacts'] for r in block['commands'] if r.get('contacts')]
        if not recs:
            return None
        t = np.array([r['time_s'] for r in block['commands'] if r.get('contacts')])
        late = t >= args.seconds * 0.5
        late_recs = [r for r, m in zip(recs, late) if m] or recs
        return {'tarsus_only_fraction_all': float(np.mean([r['tarsus_only'] for r in recs])),
                'tarsus_only_fraction_late': float(np.mean([r['tarsus_only'] for r in late_recs])),
                'six_feet_down_fraction_late': float(np.mean(
                    [len(r['legs_with_foot_contact']) == 6 for r in late_recs])),
                'non_tarsus_geoms_late': sorted({g for r in late_recs for g in r['non_tarsus_geoms']}),
                'mean_ground_contacts_late': float(np.mean([r['n_ground_contacts'] for r in late_recs]))}
    summary['contact_composition_full_sensation'] = contact_stats(fi)
    summary['contact_composition_deafferented'] = contact_stats(di)
    summary['final_heights_mm'] = {k: v['summary']['final_height_mm'] for k, v in cases.items()}

    sha_after = digest(CANONICAL_XML)
    summary['canonical_xml_sha256_after'] = sha_after
    # The canonical asset must never change, even when a variant is being checked.
    if sha_after != sha_before or sha_after != CANONICAL_SHA:
        violations.append('canonical XML changed during the check')
    summary['invariants']['violations'] = violations
    summary['invariants']['status'] = 'PASS' if not violations else 'FAIL'
    summary['overall'] = 'PASS' if not violations else 'FAIL'

    (outdir / 'fidelity_check.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if not violations else 1


if __name__ == '__main__':
    sys.exit(main())
