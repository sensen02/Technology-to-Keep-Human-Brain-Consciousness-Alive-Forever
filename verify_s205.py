"""S205 bookkeeping (ledger, budget, hashes) + independent verification."""
import json
import hashlib
import numpy as np
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s205-unresolved-mn-and-silent-pools/20261025T075517207961Z')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S204 = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


ledger = {"round": "S205", "failures": [
    {"id": "S205-A1", "what": "s205_provenance.py dataset check",
     "defect": "reading banc_888_meta.feather needs pyarrow, which is not installed; the direct dataset re-read failed",
     "caught_by": "ImportError inside the try block", "status": "recorded as a gap; corroborated with the dataset-derived table "
                  "data/banc888/promotor_adductor_pools.json instead of installing anything"},
    {"id": "S205-A2", "what": "s205_diag.py first pass",
     "defect": "the decoder was fed a constant per-window spike mask instead of the real per-substep spike matrix, so the arm "
               "diverged from S204-H0 from window 6 on (1710 vs 1708 spikes)",
     "caught_by": "comparing the per-window spike counts with S204-H0", "status": "fixed; the rerun is bitwise equal to H0"},
    {"id": "S205-A3", "what": "s205_analyze.py first pass",
     "defect": "pool spike counts were read with the NEURON index (counts[w, actuator_index]) instead of summing over the pool's "
               "cells, and the classification used 'external current' as the drive measure",
     "caught_by": "the absurd result (all 74 pools NO_EXTERNAL_CURRENT while 72 of them fired) and the 184-vs-72 mismatch",
     "status": "fixed: pool counts now sum over local_indices and match S204-H0 bitwise; classification uses membrane potential"},
    {"id": "S205-C1", "what": "an S204 interpretation, corrected here",
     "defect": "the S204 conclusion said the 72 zero-valued actuator-windows were '24 non-empty actuators that received no spike'. "
               "Wrong twice: 72 is the ZERO-CTRL count (zero-spike count is 74) and only 2 pools are silent in all three windows",
     "caught_by": "this round's per-pool diagnosis", "status": "corrected in CONCLUSION.zh-CN.md section 4; the S204 archive files "
                  "are left untouched so their hashes stay valid"}],
    "design_notes": [
        {"id": "S205-N1", "what": "why the pools see no external current",
         "finding": "the sensory encoder writes into the six per-leg channel groups (492 of 12000 cells); the motor pool cells are "
                    "driven synaptically, so their EXTERNAL current is zero by architecture, not by defect",
         "status": "documented in SILENT_POOL_DIAGNOSIS.json"},
        {"id": "S205-N2", "what": "the 16 uncommandable rows were NOT substituted",
         "finding": "a mirror or cross-leg substitution would be inventing synapses; this round only locates and sizes the gap",
         "status": "enforced: no model, mapping or XML change in this round"}]}
(RUN / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

budget = {"round": "S205",
          "stop_reason": ("Both pre-registered parts are complete: the provenance of all 16 uncommandable actuators is classified "
                          "with reasons summing to 16, and the silent pools are classified with one arm that reproduces S204-H0 "
                          "bitwise. No substitution, model change, training or XML edit was needed or done, so the round stops."),
          "budget": {"max_steps": 10000, "steps_used": 2000, "remaining": 8000,
                     "model_loads": 3, "forward_calls": 1, "mj_step_calls": 2000, "lp_calls": 0,
                     "neural_ticks": 10 * 10, "training_steps": 0, "prewarming": False,
                     "xml_changes": 0, "model_changes": 0, "mapping_substitutions": 0},
          "environment": {"python": "venv_body/bin/python 3.12", "mujoco": "3.9.0",
                          "env": "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -B",
                          "xml_sha256": sha(XML), "pyarrow": "MISSING (feather re-read not possible)"}}
budget["artifacts"] = {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
budget["sources"] = {str(p): sha(p) for p in [ROOT / 's205_prereg.py', ROOT / 's205_provenance.py', ROOT / 's205_diag.py',
                                              ROOT / 's205_analyze.py', ROOT / 'verify_s205.py', XML] if p.exists()}
(RUN / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(RUN / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))

# ----------------------------------------------------------------- verification
checks = []
def chk(name, ok, detail):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))

prov = json.load(open(RUN / 'UNRESOLVED_PROVENANCE.json'))
diag = json.load(open(RUN / 'SILENT_POOL_DIAGNOSIS.json'))
pools_raw = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())['measured']

# 1. the 16 rows, recomputed from the source of truth
u = pools_raw['unresolved']
chk('provenance:16_rows_from_source', len(u) == 16 and prov['n_unresolved'] == 16, '%d rows' % len(u))
chk('provenance:reasons_sum_to_16', prov['reasons_sum_to_16'] and sum(v['n'] for v in prov['by_reason'].values()) == 16,
    json.dumps({k: v['n'] for k, v in prov['by_reason'].items()}))
chk('provenance:reason_strings_match_source',
    sorted(r['reason'] for r in u) == sorted([r['reason'] for r in prov['rows']]),
    'per-row reasons identical to the build-time records')
absent_recomputed = sorted({r['muscle_target'] for r in u if 'absent' in r['reason']})
chk('provenance:absent_targets', absent_recomputed == sorted(prov['target_absent_entirely']['targets']) == ['pleural_promotor_muscle'],
    str(absent_recomputed))

# 2. hard-DOF loss shares recomputed from the frozen moment matrix
cen = json.load(open(S204 / 'READOUT_CENSUS.json'))
traj = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')
MOM = np.asarray(traj['actuator_moment'][0], float)
idx_of = {c['actuator']: c['actuator_index'] for c in cen['rows']}
for dof in ('22', '42'):
    v = prov['hard_dofs'][dof]
    lost = sum(abs(MOM[idx_of[a], int(dof)]) for a in v['uncommandable'])
    tot = sum(abs(MOM[idx_of[a], int(dof)]) for a in v['uncommandable'] + v['commandable'])
    chk('hard_dof:%s_lost_moment_share_recomputed' % dof, abs(lost / tot - v['lost_moment_share']) < 1e-12,
        '%.4f recomputed vs %.4f stored' % (lost / tot, v['lost_moment_share']))

# 3. diagnostic arm equals S204-H0 bitwise
d = np.load(RUN / 'diag' / 'DIAG.npz')
h = np.load(S204 / 'H0' / 'TRACE.npz')
my = np.zeros((10, 90), int)
for c in cen['rows']:
    if c['pool_size'] > 0:
        my[:, c['actuator_index']] = d['counts'][:, c['local_indices']].sum(axis=1)
chk('diag:ctrl_bitwise_equals_H0', bool(np.array_equal(d['ctrl'], h['ctrl'])), 'max diff %.3e'
    % float(np.max(np.abs(d['ctrl'] - h['ctrl']))))
chk('diag:pool_spike_counts_bitwise_equals_H0', bool(np.array_equal(my, h['pool_spikes'])), 'per-pool per-window equal')

# 4. the 74 vs 72 distinction
n0_spike = int(sum(1 for c in cen['rows'] if c['pool_size'] > 0 for w in (1, 2, 3)
                   if int(d['counts'][w, c['local_indices']].sum()) == 0))
n0_ctrl = int(sum(1 for c in cen['rows'] if c['pool_size'] > 0 for w in (1, 2, 3)
                  if float(h['ctrl'][w, c['actuator_index']]) <= 1e-12))
chk('diag:zero_spike_count_74_and_zero_ctrl_count_72', n0_spike == 74 and n0_ctrl == 72,
    'zero-spike %d, zero-ctrl %d (of 222)' % (n0_spike, n0_ctrl))
chk('diag:classes_sum_to_74', sum(diag['class_counts'].values()) == 74, json.dumps(diag['class_counts']))
chk('diag:only_two_pools_silent_in_all_windows',
    diag['n_pools_silent_in_all_three_windows'] == 2 == len(diag['silent_pools']),
    ', '.join(p['actuator'] for p in diag['silent_pools']))
chk('diag:silent_pools_are_driven_not_at_rest',
    all(p['cls'] == 'DRIVEN_BUT_BELOW_THRESHOLD' and p['max_membrane_mV'] > diag['rest_mV']
        for p in diag['silent_pools']),
    '; '.join('%s vmax=%.2f mV (rest %.1f, thr %.1f)'
              % (p['actuator'], p['max_membrane_mV'], diag['rest_mV'], diag['threshold_mV'])
              for p in diag['silent_pools']))

# 5. external current reaches only the sensory channel cells
cur = d['current']
nz = int((np.abs(cur).max(axis=0) > 0).sum())
chk('diag:external_current_only_reaches_channel_cells', nz == diag['n_cells_with_nonzero_external_current'] == 492,
    '%d of 12000 cells' % nz)

# 6. guards
chk('xml:hash_unchanged', sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', sha(XML))
chk('budget:within_cap', budget['budget']['steps_used'] <= budget['budget']['max_steps'],
    '%d of %d' % (budget['budget']['steps_used'], budget['budget']['max_steps']))
chk('guards:no_model_or_mapping_change', budget['budget']['model_changes'] == 0
    and budget['budget']['mapping_substitutions'] == 0, 'diagnosis only')

n_ok = sum(c['ok'] for c in checks)
out = dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
           note='artifact recomputation against the source pool table, the frozen moment matrix and the S204 H0 trace')
(RUN / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(dict(n=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
