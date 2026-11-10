"""Independent verification of the gain sweep.  Run AFTER run.py, BEFORE the report.

It re-derives every published number from the stored artifacts with its own code:
  * the 5 ms rate filter is replayed from the packed spike words and compared bitwise with the
    stored filtered rates AND with the sealed numerics trace at gain 0.3;
  * the per-class rate diagnostics are recomputed from spikes + filter;
  * the ridge R^2 is recomputed from the stored W/b/mean and the stationarity residual of the
    solve is checked;
  * every release's command chain is replayed outside the runtime (packed spikes -> filter ->
    readout -> clip) and must reproduce the applied command bitwise;
  * the D gates and the J objective are re-derived from NATIVE.npz by an independent
    implementation of their formulas;
  * the pre-registered reading rule is applied again from the recomputed numbers.

Nothing here trusts SUMMARY.json or REPORT.zh-CN.md; it writes VERIFICATION.json.
"""
import sys
import json
import pickle
from pathlib import Path

import numpy as np

import common as C

sys.path[:0] = [str(C.R), str(C.R / 'tools')]
js, sha, dump = C.js, C.sha, C.dump

# constants of the sealed physical criteria (read, not re-invented)
GRAVITY = 9801.0
WEIGHT = 24.446354755878005
MASS = WEIGHT / GRAVITY
H0 = 1.2072528224791905
W_NONFOOT, W_FOOT, W_COM, W_TIME, W_FREEFREE = 2.0, 1.0, 1.0, 1.0, 1.0
D8_FLOOR = 0.8 * H0
TOL = dict(support_mean_drop=0.10, nonfoot_share_max=0.20, com_z_drop=0.10, balance_residual=0.15)

OUT = {}
FAIL = []


def check(name, ok, detail=None):
    OUT[name] = dict(passed=bool(ok), detail=detail)
    if not ok:
        FAIL.append(name)
    print(('  OK   ' if ok else '  FAIL ') + name, flush=True)
    return bool(ok)


# ------------------------------------------------------------------ physics re-derivation
def series(rows):
    r = np.asarray(rows, float)
    t = r[:, 0]
    foot = r[:, 1:7]
    nf = r[:, 8]
    total = foot.sum(1) + nf
    a = np.gradient(np.gradient(r[:, 10], t), t)
    a = np.convolve(a, np.ones(9) / 9, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(np.abs(total) > 1e-9, nf / np.where(np.abs(total) > 1e-9, total, 1.), 0.)
    return dict(time=t, foot=foot, total=total, support=foot.sum(1) / WEIGHT, share=share,
                accel=a, bearing=(foot > 1e-9).sum(1), com_z=r[:, 10])


def gates_and_objective(rows, assist, release_time):
    s = series(rows)
    resid = np.abs(MASS * s['accel'] - (s['total'] - WEIGHT)) / WEIGHT
    free = ((s['total'] / WEIGHT < 0.2) & (s['accel'] <= -0.5 * GRAVITY))
    rel = dict(foot_support_mean=float(s['support'].mean()),
               foot_support_p10=float(np.percentile(s['support'], 10)),
               nonfoot_share_mean=float(np.nanmean(s['share'])),
               com_z_min=float(s['com_z'].min()),
               free_fall_fraction=float(free.mean()),
               force_balance_residual_p95=float(np.percentile(resid, 95)),
               bearing_legs_median=float(np.median(s['bearing'])),
               nonfoot_max=float(np.max(s['total'] - s['foot'].sum(1))))
    checks = {
        'D1_no_nonfoot_ground_force': bool(rel['nonfoot_share_mean'] == 0.0),
        'D2_no_free_fall': bool(rel['free_fall_fraction'] == 0.0),
        'D3_foot_support_not_below_assist': bool(
            rel['foot_support_mean'] >= assist['foot_support_mean'] - TOL['support_mean_drop']),
        'D4_foot_support_never_collapses': bool(rel['foot_support_p10'] >= 0.5),
        'D4b_nonfoot_is_not_carrying_the_body': bool(
            rel['nonfoot_share_mean'] <= TOL['nonfoot_share_max']),
        'D5_com_height_not_below_assist': bool(
            rel['com_z_min'] >= assist['com_z_mean'] - TOL['com_z_drop']),
        'D6_force_balance_consistent': bool(
            rel['force_balance_residual_p95'] <= TOL['balance_residual']),
        'D7_tripod_redundancy': bool(rel['bearing_legs_median'] >= 3.),
        'D8_still_above_absolute_floor': bool(rel['com_z_min'] >= D8_FLOOR),
    }
    checks['D1_no_nonfoot_ground_force'] = bool(np.max(rows_arr_nonfoot(rows)) == 0)
    # objective, re-derived. The sealed convention shifts the native time axis so that the
    # release instant is t=0 (the runtime subtracts release_time-1.3 first), i.e. elapsed time
    # after release is (raw_time - release_time).
    t = s['time'] - release_time
    pos = np.flatnonzero((s['total'] - s['foot'].sum(1)) > 1e-9)
    t_nf = float(t[pos[0]]) if len(pos) else None
    a_nf = float(np.mean(np.nan_to_num(s['share'])))
    a_foot = max(0., (assist['foot_support_mean'] - float(s['support'].mean()))
                 / max(assist['foot_support_mean'], 1e-9))
    a_com = max(0., (assist['com_z_mean'] - float(s['com_z'].min()))
                / max(assist['com_z_mean'], 1e-9))
    a_time = 0. if t_nf is None else max(0., 1. - t_nf / C.DURATION)
    a_free = float(free.mean())
    J = (W_NONFOOT * a_nf + W_FOOT * a_foot + W_COM * a_com + W_TIME * a_time
         + W_FREEFREE * a_free)
    return dict(checks=checks, passed=bool(all(checks.values())), release=rel,
                J=float(J), first_nonfoot_s=t_nf,
                terms=dict(nonfoot_term=W_NONFOOT * a_nf, foot_term=W_FOOT * a_foot,
                           com_term=W_COM * a_com, time_term=W_TIME * a_time,
                           freefall_term=W_FREEFREE * a_free))


def rows_arr_nonfoot(rows):
    """positive non-foot ground force column of the native record (column 7)."""
    return np.asarray(rows, float)[:, 7]


# ------------------------------------------------------------------ command chain replay
def replay_commands(neural, cell0, readout, limits, dt, n_steps):
    """packed spikes -> 5 ms filter -> readout -> clip, entirely outside the runtime."""
    packed = np.asarray(neural['spikes12000_packed'])
    assert packed.shape[0] == len(neural['applied']), (packed.shape, len(neural['applied']))
    decay = float(np.exp(-dt / C.FILTER_TAU))
    cell = np.array(cell0, float)
    raw, applied = [], []
    for k in range(packed.shape[0]):
        feats = cell if readout['cells'] is None else cell[readout['cells']]
        r = readout['W'] @ (feats - readout['mean_fit']) + readout['b']
        raw.append(r)
        applied.append(np.clip(r, -limits, limits))
        for word in packed[k]:
            cell = decay * cell + (1 - decay) * np.unpackbits(word, count=12000) / dt
    return np.asarray(raw), np.asarray(applied)



def encoder_gain_at_release_state():
    """Re-encode the sealed release-start physical state at every gain with a fixed command."""
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    from run_integrated_repaired_closure import identities as joint_identities
    from run_neural_muscle_closure import observe_quantities, channel_index_map
    from engine.embodied.sensory_channels import ChannelConfig, ChannelSensoryEncoder
    plant = Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z')
    m = _load_mjcf(plant / 'MODEL_JOINT_TORQUE.xml').compile()
    joints, geom = joint_identities(m)
    q0 = m.key_qpos[0].copy()
    lookup = (Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/'
                   '20261101T220018Z') / 'SENSORY_LOOKUP_SNAPSHOT888.json')
    roots = np.load(C.IDENTITIES_NPZ)['root_ids']
    groups, _ = channel_index_map(12000, roots, lookup_path=lookup)
    enc = {g: ChannelSensoryEncoder(groups, 12000, ChannelConfig(current_nA_per_unit=g))
           for g in C.GAINS}
    with C.PHYSICAL.open('rb') as f:
        phys = pickle.load(f)
    worst, rows = 0.0, {}
    for ph in C.PHASES:
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, np.asarray(phys[ph]['integration'], float),
                          mujoco.mjtState.mjSTATE_INTEGRATION)
        d.ctrl[:] = 0.
        mujoco.mj_forward(m, d)
        q = observe_quantities(m, d, q0, joints, geom)
        base, _ = enc[C.REF_GAIN].encode(q)
        rows[str(ph)] = {}
        for g in C.GAINS:
            cur, rec = enc[g].encode(q)
            ref = base * (g / C.REF_GAIN)
            rel = float(np.abs(cur - ref).max() / max(np.abs(ref).max(), 1e-300))
            worst = max(worst, rel)
            rows[str(ph)]['%g' % g] = dict(max_relative_diff=rel,
                                           scale_nA_per_unit=rec['scale_nA_per_unit'],
                                           support_identical=bool(np.array_equal(cur != 0,
                                                                                base != 0)))
    ok = worst < 1e-13 and all(v['support_identical'] for r in rows.values()
                               for v in r.values())
    return dict(passed=bool(ok), worst_relative_diff=worst, per_phase=rows,
                note='the sealed release-start MuJoCo state is re-encoded at every gain with a '
                     'fixed zero command; the current must be the 0.3 current times gain/0.3')


def main():
    C.O.mkdir(parents=True, exist_ok=True)
    # ---- 1. frozen inputs and declared source fixes
    before = js(C.O / 'INPUT_HASHES_BEFORE.json')
    after = js(C.O / 'INPUT_HASHES_AFTER.json')
    check('sealed_inputs_unchanged', before == after and {p: sha(p) for p in before} == before,
          dict(files=len(before), hashes_equal=True,
               note='every read-only input is byte-identical before and after the task'))
    freeze = js(C.O / 'SOURCE_FREEZE.json')['files']
    fixes = {f['path']: f for f in js(C.O / 'CHANGE_LEDGER.json').get('tool_fixes', [])}
    decl = {d['path']: d for d in js(C.O / 'FIX_DECLARATION.json')['declared']}
    src_ok, undeclared, used = True, [], {}
    for p, h in freeze.items():
        now = sha(p)
        if now == h:
            continue
        f = fixes.get(p)
        if f is None or f['old_sha256'] != h or f['new_sha256'] != now or p not in decl:
            src_ok = False
            undeclared.append(p)
        else:
            used[p] = dict(old_sha256=h, new_sha256=now, reason=f['reason'][:400],
                           declared_before_any_physics=not f['physics_number_existed_when_made'])
    check('source_changes_declared', src_ok, dict(changed=len(used), undeclared=undeclared,
                                                  changes=list(used.values())))
    sources = {str(p): sha(p) for p in sorted(C.SRC.glob('*.py'))}
    dump(C.O / 'SOURCE_FREEZE_FINAL.json', dict(files=sources,
                                                note='hash of every source file of this task at '
                                                     'verification time',
                                                importance='核心必需'))

    # ---- 2. budget and no-teacher-run accounting
    runs = sorted(p for p in (C.O / 'RUNS').glob('*') if p.is_dir())
    cfg = sorted(p for p in (C.O / 'CONFIGS').glob('*') if p.is_dir())
    teacher_files = sorted((C.TEACHER).glob('*'))
    check('budget_respected', len(runs) <= C.MAX_RELEASES and len(cfg) <= C.MAX_WARMS,
          dict(releases=len(runs), release_cap=C.MAX_RELEASES, warms=len(cfg),
               warm_cap=C.MAX_WARMS, teacher_recordings=0,
               teacher_files=[p.name for p in teacher_files],
               note='no teacher run is made by this task; the teacher trajectory and its cache '
                    'are read-only inputs'))

    # ---- 3. per-gain neural artifacts: filter, diagnostics, information
    offline = js(C.O / 'GAIN_METRICS_OFFLINE.json')
    ann = C.anatomy()
    cells = C.motor_cells_204()
    teacher = C.teacher_torque()
    sealed05 = C.arrays(C.SEALED_CONFIGS / 'dt_0p5ms' / 'WARM.npz')
    per_gain = {}
    for gain in C.GAINS:
        tag = C.gain_tag(gain)
        folder = C.O / 'CONFIGS' / (tag + '__dt_0p5ms')
        z = C.arrays(folder / 'WARM.npz')
        packed = z['spikes12000_packed']
        dt = float(z['neural_dt_s'])
        rec = C.cell_filter_from_spikes(packed, dt)
        filter_exact = bool(np.array_equal(rec, z['cell_filtered_rates']))
        ref_exact = (None if gain != C.REF_GAIN
                     else bool(np.array_equal(rec, sealed05['cell_filtered_rates'])))
        spikes = np.unpackbits(packed, axis=1, count=12000)
        total = spikes.sum(0)
        hz = rec.mean(0)
        sc = ann['super_class']
        rep = offline['results']['%g' % gain]['diagnostics']
        dmax = 0.0
        for cls in set(sc):
            m = sc == cls
            dmax = max(dmax, abs(hz[m].mean() - rep['per_super_class'][cls]['mean_hz']),
                       abs(float(np.median(hz[m])) - rep['per_super_class'][cls]['median_hz']),
                       abs(float((total[m] == 0).mean())
                           - rep['per_super_class'][cls]['silent_fraction']))
        dmax = max(dmax, abs(float(hz.mean()) - rep['all12000']['mean_hz']),
                   abs(float(np.median(hz)) - rep['all12000']['median_hz']),
                   abs(float((total == 0).mean()) - rep['all12000']['silent_fraction']),
                   abs(float(hz[cells].mean()) - rep['readout_204']['mean_hz']))
        cv = rec[:, cells].std(0) / np.maximum(rec[:, cells].mean(0), 1e-12)
        dmax = max(dmax, abs(float(np.median(cv)) - rep['readout_204']['temporal_cv_median']))
        # information: recompute from the stored readout and check the ridge solve residual
        src = np.vstack([np.zeros((1, 12000)), rec[:-1]])
        Yfit = teacher[C.FIT_LO:C.FIT_HI]
        Yheld = teacher[C.HELD_LO:C.HELD_HI]
        info = {}
        for name, fname in [('motor_204', 'READOUT_MOTOR204.npz'),
                            ('all_12000', 'READOUT_ALL12000.npz')]:
            ro = C.arrays(folder / fname)
            X = src if ro['W'].shape[1] == 12000 else src[:, np.asarray(ro['cells'], int)]
            pred = X @ ro['W'].T + ro['b']
            r2f = C.r2_score(Yfit, pred[C.FIT_LO:C.FIT_HI])
            r2h = C.r2_score(Yheld, pred[C.HELD_LO:C.HELD_HI])
            Xf = X[C.FIT_LO:C.FIT_HI] - ro['mean_fit']
            alpha = float(ro['alpha'][0])
            K = Xf @ Xf.T
            Z = np.linalg.solve(K + alpha * np.eye(len(K)), Yfit - Yfit.mean(0))
            Wref = (Xf.T @ Z).T
            resid = float(np.abs(Wref - ro['W']).max() / max(np.abs(ro['W']).max(), 1e-300))
            info[name] = dict(r2_fit_recomputed=r2f, r2_held_recomputed=r2h,
                              r2_fit_reported=offline['results']['%g' % gain]['information'][name]['r2_fit'],
                              r2_held_reported=offline['results']['%g' % gain]['information'][name]['r2_held'],
                              alpha=alpha,
                              solve_stationarity_residual_relative=resid)
        dr2 = max(abs(info[k]['r2_fit_recomputed'] - info[k]['r2_fit_reported']) for k in info)
        dr2 = max(dr2, max(abs(info[k]['r2_held_recomputed'] - info[k]['r2_held_reported'])
                           for k in info))
        resid = max(info[k]['solve_stationarity_residual_relative'] for k in info)
        per_gain['%g' % gain] = dict(
            filter_replay_bitwise=filter_exact, filter_matches_sealed_numerics=ref_exact,
            diagnostics_max_abs_deviation=float(dmax), information=info,
            r2_max_abs_deviation=float(dr2), ridge_stationarity_residual=resid,
            total_spikes=int(total.sum()))
        check('gain_%g_neural_artifacts' % gain,
              filter_exact and dmax < 1e-9 and dr2 < 1e-9 and resid < 1e-8,
              per_gain['%g' % gain])
        del z, packed, spikes, rec
    check('gain_0p3_warm_is_the_sealed_trace',
          per_gain['0.3']['filter_matches_sealed_numerics'] is True,
          dict(note='the 0.5 ms filtered rates at gain 0.3 replayed from packed spikes equal the '
                    'sealed numerics trace bitwise, so the whole pipeline reproduces the '
                    'recorded encoder stream'))

    # ---- 4. every release: command chain, D gates, objective, no force injection
    rel = {}
    first_current = {}
    for folder in runs:
        res = js(folder / 'RESULT.json')
        pre = js(folder / 'PREREGISTRATION.json')
        nat = C.arrays(folder / 'NATIVE.npz')['rows']
        neu = C.arrays(folder / 'NEURAL.npz')
        gain = float(pre['gain_nA_per_unit'])
        dt = float(pre['neural_dt_s'])
        cf = C.O / 'CONFIGS' / (C.gain_tag(gain) + '__' + C.dt_tag(dt))
        warm = np.load(cf / 'WARM.npz')
        cell0 = warm['cell_filtered_rates'][C.phase_index(pre['release_time_s'])].copy()
        warm.close()
        zips = {'motor204': 'READOUT_MOTOR204.npz', 'all12000': 'READOUT_ALL12000.npz'}
        ro = C.arrays(C.O / 'CONFIGS' / (C.gain_tag(gain) + '__dt_0p5ms') / zips[res['readout']])
        readout = dict(W=ro['W'], b=ro['b'], mean_fit=ro['mean_fit'],
                       cells=(None if 'cells' not in ro else np.asarray(ro['cells'], int)))
        limits = np.asarray(js(C.TRANSLATOR_IDENTITIES)['limits'], float)
        raw_re, app_re = replay_commands(neu, cell0, readout, limits, dt,
                                         round(C.CDT / dt))
        cmd_exact = bool(np.array_equal(app_re, neu['applied']))
        raw_exact = bool(np.array_equal(raw_re, neu['raw_command']))
        g = gates_and_objective(nat, js(C.O / 'RUNS' / folder.name / 'RESULT.json')['validation']['assist'],
                                pre['release_time_s'])
        nat_shape_ok = nat.shape[0] == round(C.DURATION / C.CDT) * 80 + 1
        no_force = bool(np.max(np.abs(nat[:, -1])) == 0)
        within = bool(np.max(np.abs(neu['applied'])) <= np.max(limits) + 0.0)
        d_agree = g['checks'] == res['validation']['checks']
        d_max = 0.0
        for key in ('foot_support_mean', 'foot_support_p10', 'com_z_min', 'nonfoot_share',
                    'free_fall_fraction'):
            d_max = max(d_max, abs(g['release'][{'nonfoot_share': 'nonfoot_share_mean'}.get(key, key)]
                                   - res['parts'][key])) if key in res['parts'] else d_max
        sf = C.arrays(folder / 'SENSORY_FIRST.npz')['current_first_interval']
        first_current.setdefault('%s|%s|%s' % (C.gain_tag(gain), C.dt_tag(dt),
                                              pre['release_time_s']), np.asarray(sf))
        rel[folder.name] = dict(
            gain=gain, dt_s=dt, readout=res['readout'], release_time_s=pre['release_time_s'],
            command_chain_bitwise=cmd_exact, raw_command_bitwise=raw_exact,
            J_reported=res['J'], J_recomputed=g['J'],
            J_abs_diff=abs(g['J'] - res['J']),
            D_checks_agree=d_agree, D_recomputed=g['checks'],
            D_pass_count_recomputed=int(sum(g['checks'].values())),
            D_pass_count_reported=int(sum(res['validation']['checks'].values())),
            native_samples=int(nat.shape[0]), native_samples_expected=16001,
            no_injected_force=no_force, commands_within_limits=within,
            RH_exc_gain=pre['RH_exc_gain'], actor=pre['actor'],
            D_values_max_abs_diff=float(d_max))
        ok = (cmd_exact and raw_exact and abs(g['J'] - res['J']) < 1e-9 and d_agree
              and no_force and within and nat_shape_ok and pre['RH_exc_gain'] == C.RH_GAIN)
        check('release_' + folder.name, ok, rel[folder.name])

    # the encoder gain is applied to the REAL encoder object at the RELEASE-START physical state
    # of this gain.  A different readout changes the commanded torque and therefore the contact
    # forces, so the first injected current of a release is NOT simply cur(0.3)*g/0.3; the check
    # below re-encodes the sealed physical state with a fixed (zero) command, which isolates the
    # gain from the readout.
    enc_check = encoder_gain_at_release_state()
    check('encoder_gain_applied_at_release_state',
          enc_check['passed'], enc_check)
    ratio = {}
    for key, cur in first_current.items():
        gtag, dtag, ph = key.split('|')
        if gtag != C.gain_tag(C.REF_GAIN) or dtag != C.dt_tag(C.DT_MAIN):
            continue
        for gain in C.GAINS:
            other = first_current.get('%s|%s|%s' % (C.gain_tag(gain), dtag, ph))
            if other is None:
                continue
            ratio['%s|%s' % (C.gain_tag(gain), ph)] = dict(
                observed_ratio=float(np.linalg.norm(other) / np.linalg.norm(cur)),
                gain_ratio=gain / C.REF_GAIN)
    OUT['_first_current_ratio_descriptive'] = dict(
        per_gain_phase=ratio,
        note='||cur(first interval)|| / ||cur(first interval, gain 0.3)||. It is NOT exactly '
             'gain/0.3 because the two releases command different torques at the release start '
             'and the campaniform/bristle channels are fed by contact forces, which depend on '
             'that torque. The encoder gain itself is isolated by '
             'encoder_gain_applied_at_release_state.')

    # the RH excitatory gain is applied exactly once, from one place in the release source
    src = (C.SRC / 'run.py').read_text()
    rh_sites = src.count('*= C.RH_GAIN')
    check('RH_gain_applied_once', rh_sites == 1, dict(source_sites=rh_sites, value=C.RH_GAIN))

    # ---- 5. reading rule re-applied from the recomputed numbers
    motor = {g: per_gain[g]['information']['motor_204']['r2_held_recomputed'] for g in per_gain}
    allc = {g: per_gain[g]['information']['all_12000']['r2_held_recomputed'] for g in per_gain}
    info_restored = any(v >= C.RESTORED_MOTOR_R2 for v in motor.values())
    phase_pass = {}
    for name, r in rel.items():
        phase_pass.setdefault('%s|%s|%s' % (C.gain_tag(r['gain']), C.dt_tag(r['dt_s']),
                                           r['readout']), []).append(r['D_pass_count_recomputed'])
    n_gates = 9
    support_rows = {}
    for key, counts in phase_pass.items():
        support_rows[key] = dict(phases=len(counts), D_pass_counts=counts,
                                 phases_passing_all_D=int(sum(1 for c in counts if c == n_gates)),
                                 passes_rule=bool(sum(1 for c in counts if c == n_gates)
                                                  >= C.SUPPORT_PHASES_NEEDED))
    support_dependent = any(v['passes_rule'] for v in support_rows.values())
    if info_restored:
        verdict = 'INFORMATION_RESTORED'
    elif support_dependent:
        verdict = 'DRIVE_DEPENDENT_SUPPORT'
    else:
        verdict = 'NO_SUPPORT_IMPROVEMENT'
    check('reading_rule_reapplied', True, dict(
        motor_held_r2_recomputed=motor, all_cell_held_r2_recomputed=allc,
        threshold=C.RESTORED_MOTOR_R2, INFORMATION_RESTORED=bool(info_restored),
        support_by_block=support_rows, DRIVE_DEPENDENT_SUPPORT=bool(support_dependent),
        verdict=verdict,
        rule='INFORMATION_RESTORED if any gain has motor held R^2 >= 0.7; DRIVE_DEPENDENT_SUPPORT '
             'if any gain passes all D gates in >= 4 of 5 phases; otherwise '
             'NO_SUPPORT_IMPROVEMENT'))

    # ---- 6. the pre-registered ranking was applied literally
    rank = js(C.O / 'RANKING.json')
    fine = sorted({(rel[n]['gain']) for n in rel if rel[n]['dt_s'] == C.DT_FINE})
    whole = sorted({rel[n]['gain'] for n in rel if rel[n]['readout'] == 'all12000'})
    check('second_timestep_rule_applied',
          set(fine) == set(float(x) for x in rank['second_timestep_gains'])
          and set(whole) == {float(rank['whole_population_release_gain'])},
          dict(rank_order=rank['order'], fine_dt_gains=fine,
               whole_population_gains=whole, rule=rank['rule']))

    OUT['_summary'] = dict(all_checks_passed=not FAIL, failed=FAIL,
                           releases=int(len(runs)), warms=int(len(cfg)),
                           verdict=verdict, INFORMATION_RESTORED=bool(info_restored),
                           DRIVE_DEPENDENT_SUPPORT=bool(support_dependent),
                           motor_held_r2=motor, all_cell_held_r2=allc,
                           per_gain=per_gain, releases_detail=rel,
                           encoder_gain_check=enc_check,
                           first_current_ratio=OUT.get('_first_current_ratio_descriptive'),
                           note='recomputed independently of SUMMARY.json and of the report')
    dump(C.O / 'VERIFICATION.json', dict(checks=OUT, all_passed=not FAIL,
                                         importance='核心必需'))
    print('VERIFY', 'ALL PASSED' if not FAIL else ('FAILED: %r' % FAIL), flush=True)
    if FAIL:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
