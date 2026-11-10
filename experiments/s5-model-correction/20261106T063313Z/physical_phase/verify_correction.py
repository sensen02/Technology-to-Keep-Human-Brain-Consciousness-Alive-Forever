"""Correction of one mis-specified verification check (V9) plus file-inventory reclassification.

Why this file exists
  verify.py was frozen with the pre-registration and is not edited afterwards. Its check V9 asserted
  that the whole first native row (20 columns) is bitwise identical across S0/S1/S2 at each release
  phase. That assertion was mis-specified: the saved native row is
  [time, foot_force(6), nonfoot_pos, nonfoot_signed, ncontact, com_z, qpos(7), mean(ctrl^2),
   max|qfrc_applied[:6]|], and two of those entries CANNOT be identical between conditions:
    * column 18 is the mean squared motor command of the first 1 ms interval, which is exactly where
      the perturbation enters (the condition's own warm rates produce its own first command);
    * the foot-force columns 1..6 come from mujoco mj_contactForce after mj_forward, i.e. from the
      solved constraint forces, which depend on the applied control of that instant.
  The physical state at the release start is identical; only the control-dependent evaluation of the
  same instant differs. This script verifies the corrected statement on the saved arrays, so the
  failure of V9 as written is documented rather than hidden or silently rewritten.

Subcommands
  check      write VERIFICATION_CORRECTION.json (corrected V9b + re-affirmation of the other checks)
  inventory  reclassify this file, VERIFICATION_CORRECTION.json and VERIFY.log inside FILE_INVENTORY.json
"""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np

O = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/physical_phase')
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
CONDITIONS = ['S0', 'S1', 'S2']
IDENT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics/TRANSLATOR_IDENTITIES.json')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def js(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=default) + '\n')


COL = dict(time=0, foot=slice(1, 7), nonfoot_pos=7, nonfoot_signed=8, ncontact=9, com_z=10,
           qpos=slice(11, 18), mean_ctrl2=18, max_qfrc_applied=19)


def check():
    ident = js(IDENT)
    limits = np.asarray(ident['limits'])
    matrix = np.asarray(ident['old_matrix'])
    per_phase = {}
    ok = True
    for ph in PHASES:
        rows = {}
        body = {}
        agg = {}
        for cond in CONDITIONS:
            run = O / 'RUNS' / ('%s_paired_%s' % (cond, ph))
            rows[cond] = np.load(run / 'NATIVE.npz')['rows'][0]
            body[cond] = np.load(run / 'BODY.npz')
            agg[cond] = np.load(O / 'CONFIGS' / cond / 'WARM_AGG.npz')
        state_cols = ['time', 'nonfoot_pos', 'nonfoot_signed', 'ncontact', 'com_z', 'qpos',
                      'max_qfrc_applied']
        state_same = {c: all(np.array_equal(rows['S0'][COL[c]], rows[k][COL[c]])
                             for k in ('S1', 'S2')) for c in state_cols}
        qvel_same = all(np.array_equal(body['S0']['qvel'][0], body[k]['qvel'][0]) for k in ('S1', 'S2'))
        foot_diff = {k: [float(rows[k][1 + i] - rows['S0'][1 + i]) for i in range(6)] for k in ('S1', 'S2')}
        foot_any = {k: bool(np.any(np.asarray(foot_diff[k]) != 0.0)) for k in ('S1', 'S2')}
        # the first motor command is this condition's own readout of its own warm rates
        ctrl_ok = {}
        for cond in CONDITIONS:
            first = np.clip(limits / 100 * (matrix @ agg[cond]['cap_rates_' + str(ph)]), -limits, limits)
            ctrl_ok[cond] = bool(abs(float(np.mean(first ** 2)) - float(rows[cond][COL['mean_ctrl2']])) < 1e-12)
        ctrl_differs = bool(rows['S0'][COL['mean_ctrl2']] != rows['S1'][COL['mean_ctrl2']]
                            and rows['S0'][COL['mean_ctrl2']] != rows['S2'][COL['mean_ctrl2']])
        passed = all(state_same.values()) and qvel_same and ctrl_differs and all(ctrl_ok.values())
        ok = ok and passed
        per_phase[str(ph)] = dict(
            passed=bool(passed),
            physical_state_columns_identical_across_conditions=state_same,
            release_start_qvel_identical=qvel_same,
            observed_ranges=dict(time=float(rows['S0'][0]), com_z=float(rows['S0'][COL['com_z']]),
                                 qpos=[float(v) for v in rows['S0'][COL['qpos']]],
                                 nonfoot_force_columns=[float(rows['S0'][7]), float(rows['S0'][8])],
                                 ncontact=int(rows['S0'][9])),
            foot_force_columns_differ_immediately=foot_any,
            foot_force_difference_vs_S0=foot_diff,
            mean_ctrl2={c: float(rows[c][COL['mean_ctrl2']]) for c in CONDITIONS},
            mean_ctrl2_equals_readout_of_own_warm_rates=ctrl_ok,
            first_motor_command_differs_as_designed=ctrl_differs)
    ver = js(O / 'VERIFICATION.json')
    others = {k: v['passed'] for k, v in ver['checks'].items() if not k.startswith('V9')}
    rec = dict(
        stage='correction of verification check V9 (mis-specified as pre-registered)',
        simulation=False,
        mis_specification=('V9 as frozen in verify.py required the whole 20-column first native row to be '
                           'bitwise identical across S0/S1/S2. Column 18 (mean ctrl^2) is where the '
                           'perturbation enters by design, and the foot-force columns come from mujoco '
                           'mj_contactForce after mj_forward, i.e. from solved constraint forces that '
                           'depend on the applied control of that instant. Both must differ; the '
                           'physical state must not.'),
        corrected_claim=('The physical starting state is identical across conditions: time, COM height, the '
                         'seven free-root pose coordinates, the release-start joint velocities, the '
                         'non-foot contact count/force (all zero) and max|qfrc_applied| match bitwise. The '
                         'only between-condition differences at the first recorded instant are the solved '
                         'foot contact forces and the applied motor command, which are exactly the '
                         'designed perturbation.'),
        corrected_check_passed=bool(ok),
        per_phase=per_phase,
        other_checks_reaffirmed=others,
        note=('verify.py itself is left byte-identical to the pre-registered version: its V9 verdict of '
              'false is correct for the check as written and stays in VERIFICATION.json.'))
    dump(O / 'VERIFICATION_CORRECTION.json', rec)
    print('V9b CORRECTED CHECK', 'PASSED' if ok else 'FAILED', flush=True)
    if not ok:
        raise SystemExit(1)


def inventory():
    p = O / 'FILE_INVENTORY.json'
    inv = js(p)
    patch = {
        'verify_correction.py': ('核心必需', '修正被误规格化的 V9 校验并复核释放起始物理状态逐位一致；'
                                             '冻结的 verify.py 不改动'),
        'VERIFICATION_CORRECTION.json': ('核心必需', 'V9 修正校验的结果（物理状态列逐位一致、'
                                                     '首条命令与当瞬足接触力按设计不同）'),
        'VERIFY.log': ('重要辅助', 'verify.py 的原始 stdout（含唯一一条误规格化检查未通过）'),
    }
    changed = []
    for e in inv['files']:
        if e['relative'] in patch and e['importance'] != patch[e['relative']][0]:
            e['importance'], e['purpose'] = patch[e['relative']]
            changed.append(e['relative'])
    counts = {}
    for e in inv['files']:
        counts[e['importance']] = counts.get(e['importance'], 0) + 1
    inv['counts_by_importance'] = counts
    inv['entries'] = len(inv['files'])
    inv['total_bytes'] = sum(e['bytes'] for e in inv['files'])
    inv['reclassified'] = changed
    inv['reclassification_note'] = ('report.py is frozen with the pre-registration, so the three files '
                                    'added to the delivery after the frozen build are reclassified here '
                                    '(see VERIFICATION_CORRECTION.json) rather than by editing report.py.')
    dump(p, inv)
    print('INVENTORY reclassified', changed, counts, flush=True)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'check'
    {'check': check, 'inventory': inventory}[action]()
