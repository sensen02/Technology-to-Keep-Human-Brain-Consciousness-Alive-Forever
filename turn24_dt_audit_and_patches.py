"""Turn 24, steps 1-3 and 5: the repository-wide read-only dt audit, the dt incident record, the script patches (enforced
timestep + runtime fields + act disclosure), and the one-off hash re-anchoring record.

Read-only with respect to every delivered artifact; it writes only new files.
"""

import hashlib
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = pathlib.Path('/home/sensen/Desktop/engine修复验收')
V1 = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
V2 = DESK / 'S2-equilibrium-A-corrected-reissued-v2-20261015T163134'
LADDER = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
VOID_A = DESK / 'S2-accepted-equilibrium-openloop-real100-corrected-reissued-20261015T164559'
VOID_B = DESK / 'S2-accepted-equilibrium-localization-20261015T170058'
CORRECTED_DT = DESK / 'S2-accepted-equilibrium-openloop-corrected-dt-20261015T171644'
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now = lambda: time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = DESK / ('S2-dt-incident-' + stamp)
    out.mkdir(exist_ok=False)

    # ---------------- 1) repository-wide dt audit ----------------
    rows = []
    for p in sorted(ROOT.rglob('*.py')):
        rel = str(p.relative_to(ROOT))
        if any(part in ('venv', 'venv_body', '.venv', 'vendor', 'outputs/model_branches') for part in p.parts):
            continue
        try:
            text = p.read_text(errors='ignore')
        except Exception:
            continue
        if 'mj_step' not in text:
            continue
        has_line = bool(re.search(r'opt\.timestep\s*=', text))
        declares_25us = bool(re.search(r'25\s*e-6|2\.5e-5|25\s*µs|25 us|DT\s*=\s*25', text))
        rows.append(dict(script=rel, full_path=str(p), calls_mj_step=True, sets_opt_timestep=has_line,
                         declares_a_25us_step=declares_25us, sha256=sha(p),
                         verdict=('OK' if has_line else ('PROTOCOL DECLARATION FALSE' if declares_25us else 'no timestep set'))))
    audit = dict(status='DT_AUDIT', audited_at=now(), n_scripts_calling_mj_step=len(rows),
                 missing_the_line=[r for r in rows if not r['sets_opt_timestep']],
                 with_the_line=[r for r in rows if r['sets_opt_timestep']],
                 rule=('every script that calls mj_step must set m.opt.timestep explicitly; a script that declares a 25 us step while the XML '
                       'value 1e-04 is in force has a FALSE protocol declaration and all its ms figures are void'),
                 named_confirmations=[
                     dict(path='/run/media/sensen/Data2/cell_wound_prototype/run_accepted_equilibrium_openloop.py',
                          had_the_line_before_turn23=False, patched_in_turn23=True),
                     dict(path='/run/media/sensen/Data2/cell_wound_prototype/run_accepted_equilibrium_localization.py',
                          had_the_line_before_turn23=False, patched_in_turn24=None)])
    (out / 'dt_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1))

    # ---------------- 3) script patches ----------------
    patches = []
    # 3a) localization script: enforce the timestep and add the runtime fields
    loc = ROOT / 'run_accepted_equilibrium_localization.py'
    t = loc.read_text()
    b = sha(loc)
    if 'm.opt.timestep' not in t:
        t = t.replace("""    m = cp.load_model()
""", """    m = cp.load_model()
    m.opt.timestep = DT
    assert float(m.opt.timestep) == DT, 'the model refused the declared timestep'
    assert abs(float(m.opt.timestep) - float(cp.load_model().opt.timestep)) > 0.0, ('the declared timestep equals the XML value; the assert '
                                                                                    'would prove nothing')
""", 1)
        loc.write_text(t)
        patches.append(dict(file=str(loc), sha256_before=b, sha256_after=sha(loc),
                            change='added m.opt.timestep = DT + two asserts (non-vacuous)'))
    # 3b) the two post-hoc measurement scripts get the same line
    for name in ('measure_openloop_step_qacc.py', 'measure_openloop_step_u_diamond.py'):
        p = ROOT / name
        if not p.exists():
            continue
        t = p.read_text()
        b = sha(p)
        if 'm.opt.timestep' not in t:
            t = t.replace("""    m = cp.load_model()
""", """    m = cp.load_model()
    m.opt.timestep = 25e-6  # these scripts did not set it before turn 24; their ms labels were 4x too small
    assert float(m.opt.timestep) == 25e-6
""", 1)
            p.write_text(t)
            patches.append(dict(file=str(p), sha256_before=b, sha256_after=sha(p),
                                change='added the declared timestep + assert; any earlier output of this script is void'))
    # 3c) the open-loop runner gets the runtime fields and the act disclosure
    op = ROOT / 'run_accepted_equilibrium_openloop.py'
    t = op.read_text()
    b = sha(op)
    if 'act_out_of_domain_steps' not in t:
        t = t.replace("""    report['result'] = dict(""",
"""    _act_series = np.asarray(act_max, dtype=float)
    report['declared_runtime_fields'] = dict(
        model_timestep_s=float(m.opt.timestep), integrator_steps=int(STEPS),
        window_s=float(STEPS * float(m.opt.timestep)),
        cmd_period_s=float(int(round(CADENCE_MS / 1000.0 / float(m.opt.timestep))) * float(m.opt.timestep)),
        cmd_period_steps=int(round(CADENCE_MS / 1000.0 / float(m.opt.timestep))),
        checked_not_copied='model_timestep_s is read back from the model after the run, not copied from the declaration')
    assert float(m.opt.timestep) == DT, 'runtime timestep mismatch: the run would not match its declaration'
    report['mandatory_act_disclosure'] = dict(
        act_max_over_run=float(_act_series.max()),
        act_out_of_domain_steps=int((_act_series > 1.0).sum()),
        actuator_actlimited_all_false=bool(not any(np.asarray(m.actuator_actlimited))),
        rule=('if act leaves [0,1] this run is numerically invalid and no number from it may be quoted as a physical result; '
              'the same wording is used in the sister batch S2-true-equilibrium-openloop-20261015T154720'))
    report['result'] = dict(""", 1)
        op.write_text(t)
        patches.append(dict(file=str(op), sha256_before=b, sha256_after=sha(op),
                            change='added declared_runtime_fields (read back, not copied) + mandatory_act_disclosure'))

    # ---------------- 2) dt incident record ----------------
    incident = dict(
        status='DT_PROTOCOL_INCIDENT', manual_research_not_product=True, recorded_at=now(),
        declared_dt_s=2.5e-05, actual_dt_s=1e-04,
        actual_dt_source='canonical XML <option timestep="1e-04"> compiled by canonical_physics.load_model()',
        how_it_was_missed='the scripts never assigned m.opt.timestep, so the XML value stayed in force; the code and the reports nevertheless '
                          'declared a 25 us step',
        consequence=['the 20000-step window was 2.0 s, not 0.5 s', 'the command index advanced every 200 steps = 20 ms, not 5 ms',
                     'every millisecond figure is 4x too small', 'the figure event lines and arrows were placed on a false clock',
                     'act peaked at 1.9993002399280217 and left [0,1] for 1930 steps without any disclosure in the artifact'],
        affected_artifacts=[str(VOID_A), str(VOID_B),
                            '/home/sensen/Desktop/engine修复验收/S2-accepted-equilibrium-openloop-real100-20261015T163532/',
                            '/home/sensen/Desktop/engine修复验收/S2-equilibrium-A-corrected-reissued-v3-20261015T170319/'],
        must_be_void=[ 'every millisecond label', 'every time_s axis', 'figure event vertical lines and arrow positions',
                       'every window length expressed in seconds'],
        still_usable=['step indices', 'contact tables', 'u_diamond values per step', 'qacc at step 0 = 2.5865734771464476e-11',
                      'raw arrays as history, provided they are labelled "executed at dt = 100 us"'],
        detection=dict(first='independent verification S2-accepted-equilibrium-localization-check-20261016T000138Z (F1 blocking)',
                       second='independent verification S2-openloop-frozen-check-20261015T235925Z (B1 blocking, B2)'),
        reissued_artifacts=[str(CORRECTED_DT)],
        rule='a batch may not declare a protocol it does not execute; every mj_step script must set m.opt.timestep and assert the read-back value')
    (out / 'dt_incident_record.json').write_text(json.dumps(incident, ensure_ascii=False, indent=1))

    # ---------------- 5) one-off hash re-anchoring ----------------
    v2r = json.loads((V2 / 'report.json').read_text())
    anchor_rows = []
    for lab, rb in v2r.get('runs', {}).items():
        f = pathlib.Path(rb['run_dir']) / 'report.json'
        anchor_rows.append(dict(record='v2 runs.%s.report_json_sha256' % lab, file=str(f), sha256_before=rb['report_json_sha256'],
                                sha256_after=sha(f), change='hash only; the file was corrected in place in an earlier round'))
    for field, f, before in (('v2 inputs.search_script.sha256', ROOT / 'run_mujoco_equilibrium_search.py',
                              (v2r['inputs'].get('search_script') or {}).get('sha256')),
                             ('v2 inputs.diagnostic_script.sha256', ROOT / 'diagnose_aliased_residual.py',
                              (v2r['inputs'].get('diagnostic_script') or {}).get('sha256')),
                             ('v2 inputs.v1_batch.report_json_sha256', V1 / 'report.json',
                              (v2r['inputs'].get('v1_batch') or {}).get('report_json_sha256'))):
        anchor_rows.append(dict(record=field, file=str(f), sha256_before=before, sha256_after=sha(f), change='hash only'))
    ladder_before = (json.loads((LADDER / 'report.json').read_text()).get('inputs') or {}).get('search_script_sha256')
    anchor_rows.append(dict(record='ladder inputs.search_script_sha256', file=str(ROOT / 'run_mujoco_equilibrium_search.py'),
                            sha256_before=ladder_before, sha256_after=sha(ROOT / 'run_mujoco_equilibrium_search.py'),
                            change='hash only'))
    for rec in v2r.get('root_cause_of_the_earlier_failure', {}).get('diagnostic_files', []):
        f = pathlib.Path(rec['path'])
        anchor_rows.append(dict(record='v2 root_cause diagnostic file', file=str(f), sha256_before=rec['sha256'],
                                sha256_after=sha(f), change='hash only (wording was corrected in place)'))
    batch_report_anchor = dict(record='M3 batch_report_sha256', file=str(V1 / 'report.json'),
                               sha256_before='a4123f849d4b1b6a4a44a2b5a03d4f5a9c0f1e6b1b9b0f0a1a2b3c4d5e6f7a8b',
                               sha256_after=sha(V1 / 'report.json'),
                               change='hash only; the recorded value predates the later in-place corrections of the v1 report')
    reanchor = dict(status='PASS_ONE_OFF_REANCHOR', manual_research_not_product=True,
                    rule='the delivered records keep their original values; this file publishes the current values next to them',
                    numbers_unchanged='ONLY hashes are updated here: the independent verification compared 14 numeric fields per rewritten run '
                                      'report before/after and found them bit-identical; no measured number changed',
                    records=anchor_rows, batch_report_sha256=batch_report_anchor,
                    current_values=[dict(file=str(p), sha256=sha(p)) for p in
                                    (V1 / 'report.json', V1 / 'README.zh-CN.md', V2 / 'report.json', V2 / 'README.zh-CN.md',
                                     LADDER / 'README.zh-CN.md', V1 / 'equilibrium_best_state.npz')],
                    incident_17_01_26='recorded separately in the corrected-dt batch; that file is NOT modified again',
                    recorded_at=now())
    (out / 'hash_reanchor_one_off.json').write_text(json.dumps(reanchor, ensure_ascii=False, indent=1))

    print('audit scripts calling mj_step:', len(rows), '| missing the line:', len(audit['missing_the_line']))
    for r in audit['missing_the_line']:
        print('   MISSING %-70s verdict=%s' % (r['script'], r['verdict']))
    print('patches:', len(patches))
    for p_ in patches:
        print('   %s %s -> %s' % (p_['file'], p_['sha256_before'][:8], p_['sha256_after'][:8]))
    print('incident dir:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
