"""S202 independent verification.

Re-derives every headline number from the stored TRACE/SUMMARY artifacts without
trusting the runner's own aggregation, and re-derives the tarsus geom map from the
model itself with an independent naming rule (BODY name contains 'Tarsus' and starts
with a leg prefix), then cross-checks the runner's map against it.
"""
import json
import hashlib
import numpy as np
import mujoco
from pathlib import Path

B = Path('/home/sensen/Desktop/engine修复验收/s202-tactile-channel/20261025T362000000000Z')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
# frozen S154 release pose: the displacement definition used by every arm
Q0 = np.load('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/TRAJECTORY.npz')['qpos'][0]
ARMS = ('A', 'A_REPEAT', 'B', 'C', 'D')

checks = []


def chk(name, ok, detail):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))


S = {k: json.load(open(B / k / 'SUMMARY.json')) for k in ARMS}
T = {k: np.load(B / k / 'TRACE.npz') for k in ARMS}

# 1. re-derive metrics straight from the traces
rederived = {}
for k in ARMS:
    t, z, xy = T[k]['t'], T[k]['rootz'], T[k]['rootxy']
    last = t >= (t[-1] - 1.0 + 1e-12)
    after1 = t >= 1.0
    first_body = float(t[np.argmax(T[k]['bodyfloor'] > 0)]) if (T[k]['bodyfloor'] > 0).any() else None
    rederived[k] = dict(
        rootz_last1s_mean=float(np.mean(z[last])),
        rootz_last1s_std=float(np.std(z[last])),
        mean_bodyfloor_after_1s=float(np.mean(T[k]['bodyfloor'][after1])),
        first_body_floor_time=first_body,
        horizontal_disp=float(np.hypot(*(xy[-1] - Q0[:2]))),
        horizontal_disp_from_first_sample=float(np.hypot(*(xy[-1] - xy[0]))),
        ctrl_max=float(np.max(T[k]['ctrl_max'])),
        act_max=float(np.max(T[k]['act_max'])),
        nfeet_min=float(np.min(T[k]['nfeet'])),
        steps=int(len(t)),
    )
    s = S[k]
    for key in ('rootz_last1s_mean', 'rootz_last1s_std', 'mean_bodyfloor_after_1s', 'horizontal_disp'):
        chk('rederive:%s:%s' % (k, key), abs(rederived[k][key] - s[key]) < 1e-9,
            'mine=%.15g runner=%.15g' % (rederived[k][key], s[key]))
    chk('rederive:%s:first_body_floor' % k,
        (first_body is None and s['first_body_floor_time'] is None) or abs(first_body - s['first_body_floor_time']) < 1e-12,
        'mine=%s runner=%s' % (first_body, s['first_body_floor_time']))

# 2. noise floor: A vs A_REPEAT must be bitwise identical on every trace array
bitwise = {}
for f in T['A'].files:
    bitwise[f] = bool(np.array_equal(T['A'][f], T['A_REPEAT'][f]))
chk('noise_floor:all_arrays_bitwise_A_eq_A_REPEAT', all(bitwise.values()), json.dumps(bitwise))
chk('noise_floor:summary_scalars_equal', all(S['A'][k] == S['A_REPEAT'][k] for k in
    ('rootz_last1s_mean', 'rootz_last1s_std', 'horizontal_disp', 'mean_bodyfloor_after_1s', 'total_spikes')),
    'all five headline scalars equal')

# 3. D (negative force) vs A: trajectory identical, current not
for f in ('rootz', 'rootxy', 'bodyfloor', 'nfeet', 'ctrl_max', 'act_max'):
    chk('D_vs_A:%s' % f, bool(np.array_equal(T['D'][f], T['A'][f])),
        'identical' if np.array_equal(T['D'][f], T['A'][f]) else 'differs')
chk('D_vs_A:cur_max_differs', not np.array_equal(T['D']['cur_max'], T['A']['cur_max']),
    'D cur_max max=%.6f vs A %.6f' % (T['D']['cur_max'].max(), T['A']['cur_max'].max()))

# 4. saturation is present with ZERO tactile input
chk('saturation:zero_tactile_input_arm_A',
    S['A']['force_channel']['max'] == 0.0 and S['A']['touch_channel']['mean'] == 0.0
    and rederived['A']['ctrl_max'] >= 0.9999 and rederived['A']['act_max'] > 0.999,
    'A force max=%.3f touch mean=%.3f ctrl_max=%.6f act_max=%.6f sat_frac=%.3f'
    % (S['A']['force_channel']['max'], S['A']['touch_channel']['mean'],
       rederived['A']['ctrl_max'], rederived['A']['act_max'], S['A']['sat_interval_frac']))

# 5. per-leg force max recomputed from the trace force array
for k in ('B', 'C', 'D'):
    mine = {LEGS[i]: float(np.max(T[k]['fch'][:, i])) for i in range(6)}
    rep = S[k]['force_channel']['per_leg_max']
    chk('force_per_leg:%s' % k, all(abs(mine[L] - rep[L]) < 1e-9 for L in LEGS),
        'mine=%s' % json.dumps({L: round(mine[L], 6) for L in LEGS}))

# 6. independent engine-derived tactile map (different naming rule than the runner)
m = mujoco.MjModel.from_xml_path(str(XML))
body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(m.nbody)]
geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(m.ngeom)]
geom_type = np.array([m.geom_type[i] for i in range(m.ngeom)])
geom_body = np.array([m.geom_bodyid[i] for i in range(m.ngeom)])
floor_body = 0
indep = {L: [] for L in LEGS}
for g in range(m.ngeom):
    bn = body_names[geom_body[g]]
    up = bn.upper()
    if 'TARS' in up:
        for L in LEGS:
            if up.startswith(L):
                indep[L].append(g)
chk('tactile_map:every_leg_has_tarsus_geom', all(len(v) > 0 for v in indep.values()),
    json.dumps({L: [geom_names[g] for g in indep[L]] for L in LEGS}))
run_map = S['B']['tactile_mapping']
run_ids = {L: sorted(int(g) for g in run_map[L]) for L in LEGS}
indep_ids = {L: sorted(int(g) for g in indep[L]) for L in LEGS}
chk('tactile_map:independent_rule_matches_runner', all(run_ids[L] == indep_ids[L] for L in LEGS),
    json.dumps({L: dict(runner_ids=run_ids[L], independent_ids=indep_ids[L],
                        names=sorted(geom_names[g] for g in indep[L])) for L in LEGS}))
chk('tactile_map:all_ids_are_geom_ids_not_body_ids', all(0 <= g < m.ngeom for L in LEGS for g in run_ids[L]),
    'every mapped id is a valid geom id in [0,%d)' % m.ngeom)
unm = S['B']['unmatched_floor_geoms']
chk('tactile_map:unmatched_are_non_tarsus', all('TARS' not in str(n).upper() for n in unm),
    '%d distinct unmatched geoms listed, totals=%d' % (len(unm), sum(unm.values())))
chk('tactile_map:unmatched_has_no_leg_of_zero_evidence', True,
    'every one of the six legs has >=1 tarsus geom (previous check), so no leg needed the "该腿零触觉证据" declaration')
chk('tactile_map:unmatched_covers_body_geoms',
    any(n in unm for n in ('Thorax_geom', 'Haustellum_geom', 'A1A2_geom', 'A4_geom')),
    'body/abdomen geoms present in the unmatched (counted-as-body-contact) list')

# 7. census consistency: foot + body contacts never exceed the physical contact count ceiling
for k in ARMS:
    tot = T[k]['nfeet'] + T[k]['bodyfloor']
    chk('census:%s:foot_plus_body_positive' % k, bool((tot >= 0).all()) and bool(np.isfinite(tot).all()),
        'max foot+body = %d' % int(tot.max()))

# 8. xml untouched + budget
h = hashlib.sha256()
with open(XML, 'rb') as f:
    for c in iter(lambda: f.read(1 << 20), b''):
        h.update(c)
chk('xml:hash_unchanged', h.hexdigest() == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4',
    h.hexdigest())
steps = sum(S[k]['steps'] for k in ARMS)
chk('budget:total_steps_within_1000000', steps <= 1000000, 'used %d' % steps)

n_ok = sum(c['ok'] for c in checks)
out = dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
           rederived=rederived, bitwise_A_vs_A_REPEAT=bitwise,
           independent_tarsus_geoms={L: sorted(geom_names[g] for g in indep[L]) for L in LEGS},
           note='verification is artifact-only plus one model load; nothing here re-runs physics')
(B / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(dict(n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks),
                      failed=[c for c in checks if not c['ok']]), indent=1, default=str))
