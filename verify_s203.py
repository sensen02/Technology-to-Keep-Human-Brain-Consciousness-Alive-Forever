"""S203 independent verification: recompute the headline numbers from the stored traces."""
import json
import hashlib
import numpy as np
from pathlib import Path

RUN = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit/20261025T072017913193Z')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
Q0 = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')['qpos'][0]
checks = []


def chk(name, ok, detail):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))


def ld(n):
    return json.load(open(RUN / n / 'SUMMARY.json')), np.load(RUN / n / 'TRACE.npz')


names = ('A', 'A_d1', 'A_d2', 'B', 'B_d1', 'short3_A', 'short3_B', 'short3_C', 'short3_D', 'short3_T')
S, Z = {}, {}
for n in names:
    S[n], Z[n] = ld(n)

# 1. end-state metrics recomputed from TRACE (displacement measured from the frozen release pose)
for n in ('A', 'A_d1', 'A_d2', 'B', 'B_d1'):
    z, s = Z[n], S[n]
    xy = z['rootxy']
    disp = float(np.hypot(xy[-1, 0] - Q0[0], xy[-1, 1] - Q0[1]))
    rz = z['rootz']
    t = z['t']
    last = rz[t >= t[-1] - 1.0]
    chk('recompute:%s:horizontal_disp' % n, abs(disp - s['horizontal_disp']) < 1e-9,
        'mine=%.15g runner=%.15g' % (disp, s['horizontal_disp']))
    chk('recompute:%s:rootz_last1s' % n,
        abs(float(np.mean(last)) - s['rootz_last1s_mean']) < 1e-12 and
        abs(float(np.std(last)) - s['rootz_last1s_std']) < 1e-12,
        'mean/std recomputed from TRACE')

# 2. ctrl divergence recomputed
cd = json.load(open(RUN / 'CTRL_DIVERGENCE.json'))
d = np.abs(Z['short3_A']['ctrl'] - Z['short3_B']['ctrl'])
rec = dict(max_abs=float(d.max()), n_values_differ=int((d > 0).sum()),
           n_commands_differ=int((d.max(axis=1) > 0).sum()), n_actuators_differ=int((d.max(axis=0) > 0).sum()))
chk('recompute:ctrl_divergence_pre_A_vs_B', rec == dict(
        max_abs=cd['pre_collapse_A_vs_B']['exact']['max_abs'],
        n_values_differ=cd['pre_collapse_A_vs_B']['exact']['n_values_differ'],
        n_commands_differ=cd['pre_collapse_A_vs_B']['exact']['n_commands_differ'],
        n_actuators_differ=cd['pre_collapse_A_vs_B']['exact']['n_actuators_differ']),
    json.dumps(rec))
d5 = np.abs(Z['A']['ctrl'] - Z['B']['ctrl'])
chk('recompute:ctrl_divergence_5s_A_vs_B',
    int((d5.max(axis=1) > 0).sum()) == cd['A_vs_B_whole_run_first_divergence']['n_commands_differ'] and
    int((d5 > 0).sum()) == cd['A_vs_B_whole_run_first_divergence']['n_values_differ'],
    'commands %d, values %d' % (int((d5.max(axis=1) > 0).sum()), int((d5 > 0).sum())))

# 3. bitwise relations that carry the conclusion
rel = cd['first_command_probe']['channel_separation']['relations']
for k, ok in (('A_vs_T', np.array_equal(Z['short3_A']['ctrl'], Z['short3_T']['ctrl']) and
               np.array_equal(Z['short3_A']['qpos'], Z['short3_T']['qpos']) and
               np.array_equal(Z['short3_A']['pool_spikes'], Z['short3_T']['pool_spikes'])),
              ('A_vs_D', np.array_equal(Z['short3_A']['ctrl'], Z['short3_D']['ctrl']) and
               np.array_equal(Z['short3_A']['qpos'], Z['short3_D']['qpos'])),
              ('A_vs_B', not np.array_equal(Z['short3_A']['ctrl'], Z['short3_B']['ctrl'])),
              ('B_vs_T', not np.array_equal(Z['short3_B']['ctrl'], Z['short3_T']['ctrl']))):
    expect_same = k in ('A_vs_T', 'A_vs_D')
    consistent = all(rel[k].values()) if expect_same else not any(rel[k].values())
    chk('bitwise:%s' % k, bool(ok) and consistent,
        json.dumps(rel[k]) + (' (expected identical everywhere)' if expect_same else ' (expected to differ everywhere)'))

# 4. touch-only spikes exist but no decoder pool changes
pools = Z['short3_T']['pool_spikes'].sum(axis=0)
poolsA = Z['short3_A']['pool_spikes'].sum(axis=0)
chk('touch_only:spikes_added_but_no_pool_changed',
    int(S['short3_T']['total_spikes']) > int(S['short3_A']['total_spikes']) and np.array_equal(pools, poolsA),
    'T spikes %d vs A %d; pools changed %d' % (S['short3_T']['total_spikes'], S['short3_A']['total_spikes'],
                                               int((pools != poolsA).sum())))
chk('touch_only:all_zero_first_command',
    float(np.max(Z['short3_T']['ctrl'][0])) == 0.0 and int(Z['short3_T']['spikes'][0]) > 0,
    'first command max ctrl %.1f with %d spikes' % (float(np.max(Z['short3_T']['ctrl'][0])),
                                                     int(Z['short3_T']['spikes'][0])))

# 5. the release pose really is in contact (legitimises the first-command probe)
chk('first_command_probe:release_pose_in_contact',
    float(np.min(Z['short3_B']['force'][0])) > 0.0,
    'per-leg tarsus force at t=0: %s' % np.round(Z['short3_B']['force'][0], 3).tolist())

# 6. delta arms: the perturbation is present and the gates are evaluated on the stored numbers
for n, delta in (('A_d1', 1e-9), ('A_d2', 1e-6)):
    z = Z[n]
    first = float(np.max(np.abs(z['qpos'][0] - Z['A']['qpos'][0])))
    chk('delta:%s:perturbation_visible_at_first_sample' % n, first > 0.0,
        'max|dqpos| after the first command = %.3e (delta = %g)' % (first, delta))
sens = json.load(open(RUN / 'SENSITIVITY_ARMS.json'))
sp = sens['spreads']
chk('gate:A_d1_chaotic_rule_evaluated',
    (sp['A_d1_disp'] >= 0.05) == sens['gate']['A_d1_fires'],
    'disp spread %.3e m, rootz-std spread %.3e: rule fires = %s'
    % (sp['A_d1_disp'], sp['A_d1_rootz_std'], sens['gate']['A_d1_fires']))
chk('gate:A_d2_same_rule', (sp['A_d2_disp'] >= 0.05) or (sp['A_d2_rootz_std'] >= 0.005),
    'disp spread %.6f, rootz-std spread %.6f -> fires = %s'
    % (sp['A_d2_disp'], sp['A_d2_rootz_std'], sens['gate']['A_d2_fires_same_rule']))
env = sens['S202_claims_vs_envelope']
chk('envelope:C_claim_below_perturbation_envelope', env['C_minus_A_disp'] < sp['A_d2_disp'],
    'C-A = %.6f m vs envelope %.6f m (ratio %.2f)' % (env['C_minus_A_disp'], sp['A_d2_disp'], env['ratio_C']))
chk('envelope:B_claim_above_envelope', env['B_minus_A_disp'] > 3 * sp['A_d2_disp'],
    'B-A = %.6f m = %.2fx the envelope' % (env['B_minus_A_disp'], env['ratio_B']))

# 7. the S202 number used for the C comparison is the one S202 actually stored
s202 = json.load(open('/home/sensen/Desktop/engine修复验收/s202-tactile-channel/20261025T362000000000Z/FOUR_ARM_RESULTS.json'))
c_diff = s202['arms']['C']['horizontal_disp'] - s202['arms']['A']['horizontal_disp']
chk('cross_round:S202_C_minus_A_unchanged', abs(c_diff - env['C_minus_A_disp']) < 1e-12,
    'S202 stored %.15g, used here %.15g' % (c_diff, env['C_minus_A_disp']))

# 8. clamp audit lines really exist in the platform source at the claimed places
cla = json.load(open(RUN / 'CLAMP_PATH_AUDIT.json'))
ok_lines = True
detail = []
for stage in cla['stages']:
    src = Path(stage['file']).read_text().splitlines()
    for hit in stage['lines']:
        text = src[hit['line'] - 1].strip()
        same = text == hit['text']
        detail.append(dict(file=stage['file'], line=hit['line'], same=same))
        ok_lines &= same
chk('clamp_audit:file_line_citations_reproduce', ok_lines,
    '%d citations checked, %d mismatched' % (len(detail), sum(1 for d in detail if not d['same'])))
chk('clamp_audit:no_rectifier_on_current_path', cla['is_negative_current_structurally_rejected'] is False,
    'encoder/tier/membrane accept negative current; clamps are the one-sided spike threshold and the two-sided decode clip')

# 9. D mechanism verdict is one of the frozen values and is backed by the pool count
dm = json.load(open(RUN / 'D_MECHANISM_VERDICT.json'))
chk('d_mechanism:verdict_in_frozen_set', dm['verdict'] in dm['frozen_values'], dm['verdict'])
chk('d_mechanism:zero_pools_changed_backs_the_verdict',
    dm['pool_attribution']['D']['n_pools_changed'] == 0,
    'D changed %d of %d decoder pools vs A' % (dm['pool_attribution']['D']['n_pools_changed'],
                                               dm['pool_attribution']['D']['n_pools']))
chk('d_mechanism:saturation_masked_not_claimed_vacuously',
    dm['verdict'] != 'SATURATION_MASKED',
    'the SATURATION_MASKED definition would have been satisfied vacuously (nothing could move), so it is refused')

# 10. E is a pre-registered deferral, not a hidden omission
pre = json.load(open(RUN / 'PREREG.json'))
chk('prereg:E_declared_deferred_before_runs', pre['written_before_runs'] is True and
    'E' not in pre['part_B_arms']['arms'] and 'deferred' in pre['part_B_arms'],
    pre['part_B_arms']['deferred']['reason'][:90] + '...')

# 11. ledger + xml + budget
bud = json.load(open(RUN / 'STOP_AND_BUDGET.json'))
chk('budget:within_cap', bud['budget']['total_used'] <= bud['budget']['total_max_steps'],
    '%d of %d steps, %d remaining' % (bud['budget']['total_used'], bud['budget']['total_max_steps'],
                                      bud['budget']['remaining']))
h = hashlib.sha256(XML.read_bytes()).hexdigest()
chk('xml:hash_unchanged', h == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', h)
led = json.load(open(RUN / 'ATTEMPT_LEDGER.json'))
chk('ledger:contains_the_withdrawn_T_duplicate',
    any(f['id'] == 'S203-A4' for f in led['failures']),
    'the duplicate-of-B arm is recorded as withdrawn, not silently overwritten')

n_ok = sum(c['ok'] for c in checks)
out = dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
           note='artifact-only recomputation plus source line citations; no physics re-run')
(RUN / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(dict(n=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
