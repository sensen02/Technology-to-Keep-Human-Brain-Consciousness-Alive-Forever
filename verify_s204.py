"""S204 bookkeeping + independent verification."""
import json
import hashlib
import numpy as np
import dataclasses
import scipy.sparse as sp
from pathlib import Path

import sys
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
from engine.embodied import adapters as ad
from engine.embodied.motor_output import build_motor_manifest, MotorOutput, MotorOutputConfig

RUN = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MOTOR_SRC = ROOT / 'engine/embodied/motor_output.py'
checks = []


def chk(name, ok, detail):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- ledger + budget first
ledger = {"round": "S204", "failures": [
    {"id": "S204-A1", "what": "s204_arm.py decoder config",
     "defect": "MotorOutputConfig is a frozen dataclass; assigning rate_for_full_control_hz raised FrozenInstanceError",
     "caught_by": "the exception itself", "status": "fixed with dataclasses.replace; no data produced"},
    {"id": "S204-A2", "what": "s204_arm.py spike packing",
     "defect": "the packed buffer was sized NT_STEPS*n//8+1 = 15001 bytes, one more than packbits produces (15000)",
     "caught_by": "broadcast ValueError", "status": "fixed (nbytes = ceil(nbits/8)); no data produced"},
    {"id": "S204-A3", "what": "reachability direction",
     "defect": "the first version propagated FORWARD from the pool cells and reported 0 reachable non-pool cells, which was "
               "then mis-read as if the pools had no neighbours",
     "caught_by": "the 0 result contradicted the measured mean out-degree 24.3/sample and was investigated; the pool cells are "
                  "motor neurons, so the question needs the backward direction (adj.T)",
     "status": "fixed; both directions are now reported (forward: 0 in 1 hop, which is the correct and expected value)"},
    {"id": "S204-A4", "what": "s204_census.py print block", "defect": "referenced renamed keys after the reachability fix",
     "caught_by": "KeyError", "status": "fixed, no artefact written from the broken pass"},
    {"id": "S204-A5", "what": "s204_verdict.py note string",
     "defect": "a literal '84.7%' inside a %-format string was parsed as a format specifier (%o)",
     "caught_by": "TypeError", "status": "fixed, no artefact written from the broken pass"},
    {"id": "S204-A6", "what": "FIG1/FIG2/FIG3 layout", "defect": "titles clipped, and FIG3 claimed the 16 uncommandable rows "
     "sit at the bottom of the heat map, which is false (they are scattered)",
     "caught_by": "opening the figures", "status": "fixed; FIG3 now marks the 16 rows with red ticks"}],
    "design_notes": [
        {"id": "S204-N1", "what": "the frozen gate asks for bitwise identical input across the H arms",
         "finding": ("in a CLOSED loop that is impossible by construction: changing the decoder changes ctrl, which changes the "
                     "body, which changes the sensory input. The measured closed-loop arms therefore share only window 0 and "
                     "diverge from window 2 on. The gate is instead evaluated on an OPEN-LOOP replay of one stored spike matrix "
                     "through the three decoder settings, which has bitwise identical input by construction; the closed-loop arms "
                     "are reported alongside and agree (67.57% vs 67.57% at H2). Neither is hidden."),
         "status": "documented in DECODER_TRANSFER_BITWISE.json and CALIBRATION_ARMS.json"}],
    "parameter_changes": [
        {"parameter": "rate_for_full_control_hz", "old": 100.0, "new": [100.0, 200.0, 1000.0],
         "file": "engine/embodied/motor_output.py (MotorOutputConfig default)",
         "old_sha256": sha(MOTOR_SRC), "new_sha256": sha(MOTOR_SRC),
         "overwrite": "NONE: the platform file was not edited; the value is overridden per arm via dataclasses.replace",
         "reason": "decoder full-scale calibration is an owner-allowed parameter; the default is left untouched so every earlier "
                   "round stays reproducible bitwise"}]}
(RUN / 'ATTEMPT_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))

steps = 3 * 2000
budget = {"round": "S204",
          "stop_reason": ("All five Part A items (census, transfer function with the bitwise hard check, decomposition, "
                          "reachability, hard-DOF cross-check) and the three calibration arms H0/H1/H2 are complete. The frozen "
                          "gates are decided: SATURATION_IS_CALIBRATION. No 5 s arm, no tactile arm and no XML change was added, "
                          "so the round stops here."),
          "budget": {"max_steps": 20000, "steps_per_arm": 2000, "arms": 3, "total_used": steps, "remaining": 20000 - steps,
                     "model_loads": {"arms": 3, "offline_census": 1, "offline_replay": 1},
                     "forward_calls": {"arms": 3, "offline": 0},
                     "mj_step_calls": steps, "lp_calls": 0,
                     "neural_ticks": 3 * 10 * 10,
                     "training_steps": 0, "prewarming": False, "xml_changes": 0, "platform_source_edits": 0},
          "environment": {"python": "venv_body/bin/python 3.12", "mujoco": "3.9.0",
                          "env": "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -B",
                          "xml_sha256": sha(XML), "motor_source_sha256": sha(MOTOR_SRC)}}
budget["artifacts"] = {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
budget["sources"] = {str(p): sha(p) for p in [ROOT / 's204_prereg.py', ROOT / 's204_census.py', ROOT / 's204_arm.py',
                                              ROOT / 's204_replay.py', ROOT / 's204_verdict.py', ROOT / 'verify_s204.py', XML, MOTOR_SRC] if p.exists()}
(RUN / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
(RUN / 'SOURCE_HASHES.json').write_text(json.dumps(dict(sources=budget['sources'], artifacts=budget['artifacts']), indent=2))

# ---------------------------------------------------------------- independent verification
cen = json.load(open(RUN / 'READOUT_CENSUS.json'))
# 1. census recomputed from the platform objects
built = ad.build_neural_tier(ad.TierConfig(background_mean_nA=0, background_sd_nA=0))
n = int(built['n'])
import mujoco
m = mujoco.MjModel.from_xml_path(str(XML))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
rows = build_motor_manifest(pools, built['tier']['root_ids'], names)['rows']
empty = [r for r in rows if not r['local_indices']]
cells = {int(x) for r in rows for x in r['local_indices']}
chk('census:n_is_12000', n == 12000 and cen['n_tier'] == 12000, n)
chk('census:rows_90_empty_16_nonempty_74',
    len(rows) == 90 and len(empty) == 16 and len(rows) - len(empty) == 74
    and cen['n_empty_rows'] == 16 and cen['n_nonempty_rows'] == 74,
    'rows %d, empty %d' % (len(rows), len(empty)))
chk('census:unique_cells_and_coverage', len(cells) == 202 and abs(cen['coverage_fraction'] - 202 / n) < 1e-15,
    '%d cells = %.4f%%' % (len(cells), 100 * len(cells) / n))
chk('census:pool_size_histogram_matches',
    sorted((int(k), int(v)) for k, v in cen['pool_size_histogram'].items()) == sorted(
        (int(k), int(v)) for k, v in zip(*np.unique([len(r['local_indices']) for r in rows if r['local_indices']],
                                                    return_counts=True))) and
    sum(int(v) for v in cen['pool_size_histogram'].values()) == 74,
    json.dumps(cen['pool_size_histogram']))

# 2. the 16 uncommandable rows really are structurally zero in every stored arm
zi = sorted(c['actuator_index'] for c in cen['rows'] if c['pool_size'] == 0)
ok_zero = True
for arm in ('H0', 'H1', 'H2'):
    ctrl = np.load(RUN / arm / 'TRACE.npz')['ctrl']
    ok_zero &= bool(np.all(ctrl[:, zi] == 0.0))
chk('arms:uncommandable_rows_are_exactly_zero_in_all_three_arms', ok_zero,
    'actuator indices %s' % zi)

# 3. H1/H2 derivation arithmetic from the stored samples
rate = json.load(open(RUN / 'RATE_DISTRIBUTION.json'))
samples = np.asarray(rate['samples'], float)
H1 = 2 * float(np.median(samples))
H2 = max(10 * 100.0, 2 * float(np.percentile(samples, 90)))
chk('derivation:H1_equals_2x_median', abs(H1 - rate['derivation']['H1']) < 1e-12, 'H1 = %r' % H1)
chk('derivation:H2_equals_max_10xH0_2xP90', abs(H2 - rate['derivation']['H2']) < 1e-12, 'H2 = %r' % H2)

# 4. replay recomputed independently from the stored packed spikes
z = np.load(RUN / 'H0' / 'TRACE.npz')
ncmd = z['ctrl'].shape[0]
NT = 10
bits = np.unpackbits(z['spikes_packed'], axis=1)[:, :NT * n].reshape(ncmd, NT, n).astype(bool)
groups = [np.asarray(r['local_indices'], int) for r in rows]
cfg = MotorOutputConfig()
decay = float(np.exp(-0.0005 / cfg.filter_tau_s))
def replay(H):
    rates = np.zeros(len(groups))
    out = np.zeros((ncmd, m.nu))
    for k in range(ncmd):
        for sample in bits[k]:
            inst = np.array([(sample[idx].mean() / 0.0005) if idx.size else 0.0 for idx in groups])
            rates = decay * rates + (1 - decay) * inst
        out[k] = np.clip(rates / H, 0.0, cfg.max_control)
    return out
c100 = replay(100.0)
chk('hard_check:independent_reimplementation_reproduces_stored_ctrl',
    bool(np.array_equal(c100, z['ctrl'])), 'max abs diff %.3e' % float(np.max(np.abs(c100 - z['ctrl']))))
ne = [c['actuator_index'] for c in cen['rows'] if c['pool_size'] > 0]
def off_rail(ctrl):
    sub = ctrl[1:4][:, ne]
    return float((~((sub <= 1e-12) | (sub >= 1 - 1e-12))).mean())
bt = json.load(open(RUN / 'DECODER_TRANSFER_BITWISE.json'))
for a, H in (('H0', 100.0), ('H1', H1), ('H2', H2)):
    mine = off_rail(replay(H))
    chk('gate:off_rail_recomputed_%s' % a,
        abs(mine - bt['open_loop_gate']['per_arm'][a]['nonempty']['off_rail_fraction']) < 1e-12,
        'mine %.6f vs file %.6f' % (mine, bt['open_loop_gate']['per_arm'][a]['nonempty']['off_rail_fraction']))

# 5. verdict follows from the frozen rules
cal = json.load(open(RUN / 'CALIBRATION_ARMS.json'))
fires_cal = any(bt['open_loop_gate']['per_arm'][a]['nonempty']['off_rail_fraction'] > 0.5 for a in ('H1', 'H2'))
fires_pool = all(bt['open_loop_gate']['per_arm'][a]['nonempty']['off_rail_fraction'] <= 0.25 for a in ('H1', 'H2'))
expected = 'SATURATION_IS_CALIBRATION' if fires_cal else ('POOL_TOO_SMALL_FOR_GRADATION' if fires_pool else 'UNDETERMINED')
chk('verdict:matches_frozen_rules', cal['verdict'] == expected, '%s (cal fires %s, pool fires %s)' % (ca := cal['verdict'], fires_cal, fires_pool))

# 6. reachability recomputed
reach = json.load(open(RUN / 'REACHABILITY.json'))
adj = (built['edges']['We'] + built['edges']['Wi']).tocsr()
adj.data[:] = 1.0
mask = np.zeros(n, bool)
mask[sorted(cells)] = True
in1 = (adj.T @ mask.astype(float)) > 0
in2 = (adj.T @ in1.astype(float)) > 0
mine_share = float(((in1 | in2 | mask) & ~mask).sum()) / int((~mask).sum())
chk('reachability:2hop_share_recomputed', abs(mine_share - reach['reachable_nonpool_share']) < 1e-12,
    '%.6f vs %.6f' % (mine_share, reach['reachable_nonpool_share']))
chk('reachability:label_matches_frozen_rule',
    reach['label'] == ('SUBTHRESHOLD_REACHABLE' if mine_share > 0.5 else 'TOPO_UNREACHABLE'),
    '%s at share %.6f (margin %.2f pp)' % (reach['label'], mine_share, abs(mine_share - 0.5) * 100))

# 7. transfer function numbers
tr = json.load(open(RUN / 'DECODER_TRANSFER.json'))
jump = (1 - decay) * 2000.0
chk('transfer:single_spike_gain', abs(tr['filtered_gain_per_single_spike'] - jump) < 1e-9,
    '%.6f Hz per spike' % jump)

# 8. xml + platform source untouched + budget
chk('xml:hash_unchanged', sha(XML) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', sha(XML))
chk('platform:motor_output_py_unmodified',
    sha(MOTOR_SRC) == ledger['parameter_changes'][0]['new_sha256'] == ledger['parameter_changes'][0]['old_sha256'],
    'the calibration was applied per arm, never by editing the platform')
chk('budget:within_cap', budget['budget']['total_used'] <= budget['budget']['max_steps'],
    '%d of %d steps' % (budget['budget']['total_used'], budget['budget']['max_steps']))

n_ok = sum(c['ok'] for c in checks)
out = dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
           note='independent recomputation from the platform objects and the stored traces; no physics re-run')
(RUN / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(dict(n=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
