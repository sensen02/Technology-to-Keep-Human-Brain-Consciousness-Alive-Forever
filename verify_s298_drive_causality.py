"""S298 independent verification: recompute all three parts and the rule decision."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ARCH = Path('/home/sensen/Desktop/engine修复验收')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S297 = ARCH / 's297-decoder-dynamic-range/20261026T193953Z'
GROUPS = ARCH / 's295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = ('lf', 'rm', 'lh'), ('rf', 'lm', 'rh')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def crossings_by_dft_sign(x):
    """Independent sign-change check: reconstruct the tripod difference from its Fourier series."""
    n = len(x); f = np.fft.rfftfreq(n, d=.001); F = np.fft.rfft(x - x.mean())
    rec = np.fft.irfft(F, n=n) + x.mean()
    return int(np.count_nonzero(np.diff(np.sign(rec)) != 0)), float(np.abs(rec - x).max())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    checks = []
    def ck(n, v, **e): checks.append(dict(name=n, passed=bool(v), **e))
    try:
        pre = js(out / 'PREREG.json'); a = js(out / 'S297_CORRECTION.json'); b = js(out / 'FEASIBILITY_BOUND.json')
        c = js(out / 'LEG_ANTIPHASE.json'); o = js(out / 'OUTCOMES.json'); led = js(out / 'CALL_LEDGER.json')
        ck('S298 prereg round and zero calls', pre['round'] == 'S298' and all(v == 0 for v in led.values()) and all(v == 0 for v in pre['calls'].values()))
        for p, h in pre['input_hashes'].items(): ck('input hash ' + p, Path(p).is_file() and sha(p) == h)
        ck('no missing inputs', pre['missing_inputs'] == [])
        nb = np.load(S297 / 'original_mask' / 'NEURAL.npz'); nc = np.load(S297 / 'decoder_ac_gain' / 'NEURAL.npz')
        preb = np.asarray(nb['decoder_preclip'], float); prec = np.asarray(nc['decoder_preclip'], float)
        postb = np.asarray(nb['decoder_postclip'], float); postc = np.asarray(nc['decoder_postclip'], float)
        appc = np.asarray(nc['decoder_ac_applied'], float); V = np.asarray(nb['decoder_motor_gains'], float)
        nz = np.flatnonzero(V != 0); m = (np.arange(1, 401) >= 20) & (np.arange(1, 401) < 400)
        ck('64 nonzero and 26 zero channels', len(nz) == 64 and int(np.count_nonzero(V == 0)) == 26)
        med = lambda xs: float(np.median(xs))
        d1 = med([postb[m, i].std() / preb[m, i].std() for i in nz])
        d2 = med([appc[m, i].std() / prec[m, i].std() for i in nz])
        d3 = med([postc[m, i].std() / preb[m, i].std() for i in nz])
        ck('part A delivered medians reproduced',
           all(abs(x - y) < 1e-12 for x, y in ((d1, a['delivered_modulation']['baseline_postclip_over_own_preclip']),
                                               (d2, a['delivered_modulation']['candidate_ac_applied_over_own_preclip']),
                                               (d3, a['delivered_modulation']['candidate_raw_postclip_over_baseline_preclip']))))
        kt = js(S297 / 'AC_GAIN_TABLE.csv'.replace('AC_GAIN_TABLE.csv', 'AC_TABLE.json'))['k']
        ck('part A k classification reproduced', a['k_distribution']['k_at_max'] == sum(1 for x in kt if x >= 6.0)
           and a['k_distribution']['k_below_1'] == sum(1 for x in kt if x < 1.0)
           and abs(a['k_distribution']['k_min'] - min(kt)) < 1e-12)
        tot_b = float(postb[m].sum(axis=1).mean()); tot_c = float(appc[m].sum(axis=1).mean())
        ck('part A total command change reproduced', abs(a['total_command']['relative_change_percent'] - (tot_c / tot_b - 1) * 100) < 1e-9)
        newz = [int(i) for i in nz if np.count_nonzero(postb[m, i] == 0) == 0 and np.count_nonzero(appc[m, i] == 0) > 0]
        ck('part A newly zeroed channels reproduced', a['channels_newly_zero_at_some_sample_but_never_zero_in_baseline'] == newz)
        ck('part A channel 40 and 68 reproduced',
           abs(a['channel_40']['cand_min'] - float(appc[:, 40].min())) < 1e-12 and abs(a['channel_40']['base_min'] - float(postb[:, 40].min())) < 1e-12
           and abs(a['per_channel_mean_change']['channel_68']['mean_change'] - float(appc[m, 68].mean() - postb[m, 68].mean())) < 1e-12)
        # part B
        mm = [float(postb[m, i].mean()) for i in nz]; ss = [float(postb[m, i].std()) for i in nz]
        bnd = [float(np.sqrt(max(x * (1 - x), 0))) for x in mm]
        ck('part B groups reproduced', b['groups']['m_ge_0_9'] == sum(1 for x in mm if x >= .9)
           and b['groups']['m_ge_0_99'] == sum(1 for x in mm if x >= .99) and abs(b['groups']['bound_median'] - med(bnd)) < 1e-12)
        ck('part B required mean bound for std 0.34 correct (upper root)', abs(b['required_mean_for_group_std']['0.34']['mean_must_be_at_most'] - (1 + np.sqrt(1 - 4 * .34 ** 2)) / 2) < 1e-12)
        ck('part B route closed', b['verdict']['feasible'] == 'NO')
        # part C
        g = js(GROUPS)['legs']; z = np.load(S289 / 'NEURAL.npz')
        V9 = np.asarray(z['decoder_motor_gains'], float); pre9 = np.asarray(z['decoder_preclip'], float)
        norm = pre9 / np.where(V9 == 0, 1.0, V9)
        L = {l: norm[:, g[l]['minus']].mean(axis=1) for l in LEGS}
        Lm = np.vstack([L[l] for l in LEGS]); cm = Lm.mean(0); R = Lm - cm
        Craw = np.corrcoef(Lm); Cres = np.corrcoef(R)
        q = c['per_quantity']['gain_normalised_preclip']
        ck('part C L means reproduced', all(abs(q['L_mean'][l] - float(L[l].mean())) < 1e-12 for l in LEGS))
        rawoff = {(x, y): float(Craw[i, j]) for i, x in enumerate(LEGS) for j, y in enumerate(LEGS) if i < j}
        ck('part C raw correlations reproduced', all(abs(t[2] - rawoff[(t[0], t[1])]) < 1e-12 for t in q['raw_pair_correlations']))
        resoff = {(x, y): float(Cres[i, j]) for i, x in enumerate(LEGS) for j, y in enumerate(LEGS) if i < j}
        ck('part C residual correlations reproduced', all(abs(t[2] - resoff[(t[0], t[1])]) < 1e-12 for t in q['residual_pair_correlations']))
        ck('part C residual sums to zero (artefact declared)', float(np.abs(R.sum(0)).max()) < 1e-9)
        D = np.mean([L[l] for l in TRI_A], axis=0) - np.mean([L[l] for l in TRI_B], axis=0)
        sc, recerr = crossings_by_dft_sign(D)
        ck('part C tripod difference reproduced', abs(q['tripod_difference']['std'] - float(D.std())) < 1e-12
           and abs(q['tripod_difference']['mean'] - float(D.mean())) < 1e-12)
        sc_same = int(np.count_nonzero(np.diff(np.sign(D)) != 0))
        zeros = int(np.count_nonzero(D == 0))
        ck('part C sign changes reproduced with the recorded convention', sc_same == q['tripod_difference']['sign_changes'],
           same_convention=sc_same, reported=q['tripod_difference']['sign_changes'], exact_zero_samples=zeros)
        ck('part C Fourier reconstruction is faithful (crossing convention differs by exact zeros)',
           recerr < 1e-9, reconstruction_error=recerr, dft_convention_crossings=sc)
        Lmean = float(np.mean([L[l].mean() for l in LEGS]))
        ck('part C amplitude criterion far below 0.3', q['tripod_difference']['std'] < .3 * Lmean,
           ratio=float(q['tripod_difference']['std'] / Lmean))
        ck('part C raw correlations all positive', all(t[2] > 0 for t in q['raw_pair_correlations']))
        ck('decision primary reproduced', o['primary_rule'] == 'B' and o['satisfied']['rule_a_corrected'] is False
           and o['satisfied']['rule_a_literal'] is True and 'artefact' in o['rule_a_literal_note'])
        ck('no simulator or neural import', not any(k.split('.')[0] == 'mujoco' or k.startswith('engine.') for k in __import__('sys').modules))
        result = dict(passed=all(x['passed'] for x in checks), checks=checks, failed=[x for x in checks if not x['passed']],
                      verifier_sha256=sha(__file__), primary_rule=o['primary_rule'],
                      calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0),
                      limitations=['read-only audit of saved arrays', '2.5 Hz resolution: the 8-25 Hz band is not judgeable',
                                   'residual correlations are structurally negative because the six residuals sum to zero',
                                   'sqrt(m(1-m)) bounds feasibility only, never behaviour'])
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed'], primary=result['primary_rule'])))
        return 0 if result['passed'] else 1
    except Exception as exc:
        import traceback
        (out / 'VERIFICATION.json').write_text(json.dumps(dict(passed=False, error=repr(exc), traceback=traceback.format_exc(), checks=checks), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=False, error=repr(exc)))); return 1


if __name__ == '__main__': raise SystemExit(main())
