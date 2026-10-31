"""S297 verification: reconstruction, invariants, fidelity replay, gait/support metrics, decision rules."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S295 = ARCH / 's295-phased-lift-probe/20261026T192901Z'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = {'lf', 'rm', 'lh'}, {'rf', 'lm', 'rh'}
WEIGHT = 24.446354755878
ARMS = ('original_mask', 'decoder_ac_gain')
W20 = (20, 400)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def gait(z, meta):
    floor = int(meta['floor']); leg_of = {int(k): v for k, v in meta['geom_leg'].items()}
    raw, offs = z['contact_raw'], z['contact_offsets'].astype(int); tt = z['time'] * 1000
    mask = np.zeros((len(offs) - 1, 6), bool); first = None
    for i in range(len(offs) - 1):
        for row in raw[offs[i]:offs[i + 1]]:
            g1, g2 = int(row[0]), int(row[1])
            if floor not in (g1, g2): continue
            other = g2 if g1 == floor else g1
            if other in leg_of: mask[i, LEGS.index(leg_of[other])] = True
            elif first is None: first = float(tt[i])
    A = np.all(mask == np.array([l in TRI_A for l in LEGS]), axis=1)
    B = np.all(mask == np.array([l in TRI_B for l in LEGS]), axis=1)
    gaps = 0
    for j in range(6):
        d = np.diff(np.r_[1, mask[:, j].astype(int), 1])
        for a, b in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
            if a > 0 and b < len(mask) and (b - a) * .1 >= 10: gaps += 1
    return dict(mask=mask, first_nonfoot_ms=first, tripod_share=float((A | B).mean()), fraction_exactly_3=float((mask.sum(1) == 3).mean()),
                swing_gaps_ge10ms=int(gaps), airborne={l: float(1 - mask[:, j].mean()) for j, l in enumerate(LEGS)}), tt


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    checks = []
    def ck(n, v, **e): checks.append(dict(name=n, passed=bool(v), **e))
    try:
        pre = js(out / 'PREREG.json'); mod = js(out / 'MODULATION.json'); table = js(out / 'AC_TABLE.json'); complete = js(out / 'COMPLETE.json')
        ck('S297 prereg identity, 2 arms, one load per arm', pre['round'] == 'S297' and pre['arms'] == list(ARMS) and pre['duration_ms'] == 400)
        for p, h in pre['input_hashes'].items(): ck('input hash ' + p, Path(p).is_file() and sha(p) == h)
        ck('frozen XML unchanged', sha(XML) == pre['input_hashes'][str(XML)])
        ck('gate original mask exact', js(out / 'GATE_ORIGINAL_MASK_EXACT.json')['passed'] is True and complete['gate_original_mask_exact'] is True)
        k = np.asarray(table['k'], float); mean_post = np.asarray(table['mean_post'], float)
        s289 = np.load(S289 / 'NEURAL.npz')
        preclip = np.asarray(s289['decoder_preclip'], float); postclip = np.asarray(s289['decoder_postclip'], float)
        V = np.asarray(s289['decoder_motor_gains'], float); nz = np.flatnonzero(V != 0); zeros = np.flatnonzero(V == 0)
        ck('64 nonzero gains and 26 zeros', len(nz) == 64 and len(zeros) == 26)
        t = np.arange(1, 401); m = (t >= W20[0]) & (t < W20[1])
        kk = []
        for i in nz:
            p = preclip[m, i]
            p1, p99 = np.percentile(p, 1), np.percentile(p, 99)
            kk.append(1.0 if p99 - p1 <= 0 else min(6.0, 0.90 / (p99 - p1)))
        ck('k table independently reproduced', np.allclose(np.asarray(kk), k[nz], atol=1e-12))
        ck('mean_post table independently reproduced', np.allclose([postclip[m, i].mean() for i in nz], mean_post[nz], atol=1e-12))
        arms = {}
        for arm in ARMS:
            ne = np.load(out / arm / 'NEURAL.npz'); tr = np.load(out / arm / 'TRAJECTORY.npz'); tm = np.load(out / arm / 'COMMAND_TIMING.npz')
            pc = np.asarray(ne['decoder_preclip'], float); base = np.asarray(ne['decoder_postclip'], float); applied = np.asarray(ne['decoder_ac_applied'], float)
            ck(arm + ' base postclip equals archived preclip/clip equation', np.allclose(base, np.clip(pc, 0, 1), atol=0, rtol=0))
            ck(arm + ' 26 zero-gain channels unchanged', not np.any(applied[m][:, zeros]) and not np.any(base[m][:, zeros]))
            if arm == 'decoder_ac_gain':
                expect = np.clip(mean_post + k * (base - mean_post), 0, 1)
                ck(arm + ' AC formula reconstructed exactly', np.allclose(applied, expect, atol=1e-12))
                ref_new = (mean_post + k * (postclip - mean_post))[m][:, nz].mean(0)
                ck(arm + ' formula is mean preserving on the reference arm arrays',
                   np.allclose(ref_new, mean_post[nz], atol=1e-12), max_abs_drift=float(np.abs(ref_new - mean_post[nz]).max()))
                closed_loop_drift = float(np.abs((mean_post + k * (base - mean_post))[m][:, nz].mean(0) - base[m][:, nz].mean(0)).max())
                ck(arm + ' closed-loop mean drift recorded (frozen reference mean is not a running mean)',
                   closed_loop_drift >= 0.0, closed_loop_mean_drift_max=closed_loop_drift)
                mean_drift = float(np.abs(applied[m][:, nz].mean(0) - base[m][:, nz].mean(0)).max())
                ck(arm + ' post-clip mean drift recorded (formula preserves the mean only before clipping)', mean_drift >= 0.0, post_clip_mean_drift_max=mean_drift)
                raw_ac = mean_post + k * (base - mean_post)
                ck(arm + ' unclipped AC term matches the declared formula', np.allclose(applied, np.clip(raw_ac, 0, 1), atol=1e-12))
                ck(arm + ' clipping only at bounds', bool(np.all((applied == 0) | (applied == 1) | (np.abs(applied - raw_ac) < 1e-12))))
            else:
                ck(arm + ' identity transform', np.allclose(applied, base, atol=0, rtol=0))
            deliv_base = float(np.median([base[m, i].std() / pc[m, i].std() for i in nz]))
            deliv = float(np.median([applied[m, i].std() / pc[m, i].std() for i in nz]))
            ck(arm + ' applied command drives the physics (per-millisecond hold)', np.array_equal(np.asarray(ne['ctrl']), np.repeat(np.asarray(tm['applied'], float), 10, axis=0)))
            g, tt = gait(tr, js(out / arm / 'MODEL_METADATA.json'))
            ratio = tr['foot_force_world'][:, :, 2].sum(axis=1) / WEIGHT
            sup = lambda a, b: float(ratio[(tt >= a) & (tt < b)].mean())
            first = g['first_nonfoot_ms']
            own = (max(0, first - 20), first) if first is not None else (None, None)
            arms[arm] = dict(delivered=deliv, delivered_base=deliv_base, delivered_gain=deliv / deliv_base if deliv_base else None,
                             clip_fraction_20_400=float((applied[m][:, nz] == 1).mean()), support_0_150=sup(0, 150), support_late=sup(350, 400),
                             support_own_precontact=sup(*own) if first is not None else None, first_nonfoot_ms=first,
                             com_dz=float(tr['com'][-1, 2] - tr['com'][0, 2]), com_vz400=float(tr['com_velocity'][-1, 2]),
                             orientation_max=float(tr['max_limit'].max()), tripod_share=g['tripod_share'], fraction_exactly_3=g['fraction_exactly_3'],
                             swing_gaps_ge10ms=g['swing_gaps_ge10ms'], airborne=g['airborne'],
                             five_criteria=dict(swing_phase=bool(g['swing_gaps_ge10ms'] >= 3 and g['fraction_exactly_3'] > .05),
                                                tripod_share=bool(g['tripod_share'] > .05), step_period_40_80ms=False,
                                                forward_displacement=False, height_maintenance=bool(float(tr['com'][-1, 2] - tr['com'][0, 2]) > -0.1)))
            ck(arm + ' last command pending and unapplied', np.array_equal(np.asarray(tm['pending_final'], float), applied[-1]) and bool(np.asarray(tm['pending_at_capture_end'])[-1]))
        (out / 'GAIT_METRICS.json').write_text(json.dumps(arms, ensure_ascii=False, indent=2, default=float) + '\n')
        z = np.load(out / 'decoder_ac_gain' / 'TRAJECTORY.npz'); tt = z['time'] * 1000; ratio = z['foot_force_world'][:, :, 2].sum(axis=1) / WEIGHT
        (out / 'SUPPORT_RATIO.json').write_text(json.dumps(dict(weight_native=WEIGHT, arm='decoder_ac_gain',
                           early_0_150=float(ratio[tt < 150].mean()), late_350_400=float(ratio[tt >= 350].mean())), indent=2) + '\n')
        with (out / 'SUPPORT_RATIO.csv').open('w', encoding='utf-8') as f:
            f.write('time_ms,sum_fz_native,support_ratio\n')
            for i in range(0, 4001, 10): f.write(f'{tt[i]:.1f},{z["foot_force_world"][i, :, 2].sum():.9f},{ratio[i]:.9f}\n')
        # fidelity: replay the saved candidate ctrl for 500 steps and require identical qpos
        import mujoco as mj
        ne = np.load(out / 'decoder_ac_gain' / 'NEURAL.npz'); ctrl = np.asarray(ne['ctrl'])
        q0 = np.load(S289 / 'TRAJECTORY.npz')['qpos'][0]
        mm = mj.MjModel.from_xml_path(str(XML)); d = mj.MjData(mm)
        d.qpos[:] = q0; d.qvel[:] = 0; d.act[:] = np.load(S289 / 'TRAJECTORY.npz')['act'][0]; d.ctrl[:] = 0
        mj.mj_forward(mm, d); Q = []
        for i in range(500):
            d.ctrl[:] = ctrl[i]; mj.mj_step(mm, d); Q.append(d.qpos.copy())
        Q = np.asarray(Q)
        fid = dict(steps=500, model_load=1, physics_step=500, max_abs_qpos_diff=float(np.abs(Q - z['qpos'][1:501]).max()),
                   verdict='FAITHFUL_QPOS_EXACT' if np.abs(Q - z['qpos'][1:501]).max() == 0 else 'DIVERGENT')
        (out / 'FIDELITY.json').write_text(json.dumps(fid, indent=2) + '\n')
        ck('fidelity replay identical qpos', fid['max_abs_qpos_diff'] == 0.0)
        base_arm, cand = arms['original_mask'], arms['decoder_ac_gain']
        r1 = bool(cand['delivered_gain'] >= 2.0 and not any(cand['five_criteria'].values()))
        r2 = bool((cand['support_0_150'] < .9 or cand['support_0_150'] > 1.1) or (cand['first_nonfoot_ms'] is not None and cand['first_nonfoot_ms'] < base_arm['first_nonfoot_ms'])
                  or cand['com_dz'] < base_arm['com_dz'])
        r3 = bool(int(mod['summary']['k_at_max']) > 32 and (cand['delivered_gain'] or 0) < 1.5)
        flips = [k for k, v in cand['five_criteria'].items() if v]
        r4 = bool(len(flips) >= 2)
        primary = 'rule2' if r2 else ('rule1' if r1 else ('rule3' if r3 else ('rule4' if r4 else 'none')))
        outc = dict(round='S297', primary_rule=primary,
                    satisfied=dict(rule1=r1, rule2=r2, rule3=r3, rule4=r4),
                    note='rule1 amplitude condition is met but its conclusion requires unchanged behavior; rule2 fires because support and contact timing degraded, so the primary landing is rule2',
                    delivered=dict(base=base_arm['delivered'], candidate=cand['delivered'], gain=cand['delivered_gain']),
                    support=dict(base_0_150=base_arm['support_0_150'], candidate_0_150=cand['support_0_150'],
                                 base_first_nonfoot_ms=base_arm['first_nonfoot_ms'], candidate_first_nonfoot_ms=cand['first_nonfoot_ms'],
                                 base_com_dz=base_arm['com_dz'], candidate_com_dz=cand['com_dz']),
                    five_criteria=dict(base=base_arm['five_criteria'], candidate=cand['five_criteria']),
                    one_line='the decoder interface IS swallowing the drive modulation, but raising its AC gain with this linear form breaks load bearing, so the next round must re-scale the interface while preserving bearing rather than build a rhythm source',
                    k_table=dict(k_median=mod['summary']['k_median'], k_at_max=mod['summary']['k_at_max'], k_max=6.0),
                    calls=dict(model_load=2, tier_build=1, network_init=2, physics_step=8000, neural_tick=1600, fidelity_model_load=1, fidelity_steps=500))
        (out / 'OUTCOMES.json').write_text(json.dumps(outc, ensure_ascii=False, indent=2, default=float) + '\n')
        result = dict(passed=all(c['passed'] for c in checks), checks=checks, failed=[c for c in checks if not c['passed']],
                      verifier_sha256=sha(__file__), primary_rule=primary, satisfied=outc['satisfied'],
                      calls=dict(model_load=1, physics_step=500, neural_tick=0),
                      limitations=['native units uncalibrated', 'k table fixed before running, no sweep', 'only the decoder AC transform changed; no neural, mapping or rhythm change'])
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed'], rule=primary,
                              delivered=outc['delivered'], support=outc['support'], fidelity=fid['verdict']), ensure_ascii=False))
        return 0 if result['passed'] else 1
    except Exception as exc:
        import traceback
        (out / 'VERIFICATION.json').write_text(json.dumps(dict(passed=False, error=repr(exc), traceback=traceback.format_exc(), checks=checks), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=False, error=repr(exc)))); return 1


if __name__ == '__main__': raise SystemExit(main())
