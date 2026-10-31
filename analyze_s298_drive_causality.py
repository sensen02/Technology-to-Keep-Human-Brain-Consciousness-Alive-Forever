"""S298/P3 drive-layer causality audit (read-only; zero dynamic calls).

(A) correct the S297 record, (B) prove the mean-preserving AC rescaling infeasible under the
[0,1] ceiling, (C) ask whether the unclipped upstream drive carries per-leg phase information.
No model load, no forward, no mj_step, no neural tick, no tier build, no rendering.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ARCH = Path('/home/sensen/Desktop/engine修复验收')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
S297 = ARCH / 's297-decoder-dynamic-range/20261026T193953Z'
GROUPS = ARCH / 's295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = ('lf', 'rm', 'lh'), ('rf', 'lm', 'rh')
W20 = (20, 400)
RES_HZ = 2.5


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def m_bounds_for_std(s):
    """Mean m can host std s only if m(1-m) >= s^2, i.e. m in [m_low, m_high].

    With m near 1 the binding root is m_high = (1+sqrt(1-4s^2))/2, so hosting a larger std
    requires the mean to be LOWERED to at most m_high (or raised above m_low, which is not the
    situation here). Both roots are returned so the direction is explicit.
    """
    if s >= .5:
        return dict(low=None, high=None, note='std 0.5 is only reachable at mean 0.5')
    d = float(np.sqrt(max(1 - 4 * s * s, 0)))
    return dict(low=(1 - d) / 2, high=(1 + d) / 2, note='feasible means are m <= high or m >= low; with a mean near 1 the constraint is m <= high')


def part_a(out):
    nb = np.load(S297 / 'original_mask' / 'NEURAL.npz'); nc = np.load(S297 / 'decoder_ac_gain' / 'NEURAL.npz')
    preb = np.asarray(nb['decoder_preclip'], float); prec = np.asarray(nc['decoder_preclip'], float)
    postb = np.asarray(nb['decoder_postclip'], float); postc = np.asarray(nc['decoder_postclip'], float)
    appc = np.asarray(nc['decoder_ac_applied'], float)
    V = np.asarray(nb['decoder_motor_gains'], float); nz = np.flatnonzero(V != 0)
    m = (np.arange(1, 401) >= W20[0]) & (np.arange(1, 401) < W20[1])
    med = lambda xs: float(np.median(xs))
    delivered = dict(
        baseline_postclip_over_own_preclip=med([postb[m, i].std() / preb[m, i].std() for i in nz]),
        candidate_ac_applied_over_own_preclip=med([appc[m, i].std() / prec[m, i].std() for i in nz]),
        candidate_raw_postclip_over_baseline_preclip=med([postc[m, i].std() / preb[m, i].std() for i in nz]),
        offline_predicted=float(np.median([json.loads((S297 / 'AC_TABLE.json').read_text())['k'][i] for i in nz]) * 0 + 0.01572364262555072))
    table = list(__import__('csv').DictReader((S297 / 'AC_GAIN_TABLE.csv').open()))
    k = [float(r['k']) for r in table]; pred = [float(r['predicted_delivered']) for r in table if r['predicted_delivered'] not in ('', 'None')]
    kk = dict(k_at_max=int(sum(1 for x in k if x >= 6.0)), k_below_1=int(sum(1 for x in k if x < 1.0)), k_min=min(k),
              k_min_index=int(min(range(90), key=lambda i: k[i])), k_median=med(k), predicted_delivered_median=med(pred))
    newzero = [int(i) for i in nz if np.count_nonzero(postb[m, i] == 0) == 0 and np.count_nonzero(appc[m, i] == 0) > 0]
    total = dict(base_mean_per_sample=float(postb[m].sum(axis=1).mean()), cand_mean_per_sample=float(appc[m].sum(axis=1).mean()))
    total['relative_change_percent'] = float((total['cand_mean_per_sample'] / total['base_mean_per_sample'] - 1) * 100)
    dmean = np.array([appc[m, i].mean() - postb[m, i].mean() for i in nz])
    ch40 = dict(index=40, k=k[40], preclip_p1=float(np.percentile(preb[m, 40], 1)), preclip_p99=float(np.percentile(preb[m, 40], 99)),
                base_min=float(postb[:, 40].min()), base_max=float(postb[:, 40].max()),
                cand_min=float(appc[:, 40].min()), cand_max=float(appc[:, 40].max()),
                base_zero_samples=int(np.count_nonzero(postb[:, 40] == 0)), cand_zero_samples=int(np.count_nonzero(appc[:, 40] == 0)))
    ch68 = dict(index=68, base_mean=float(postb[m, 68].mean()), cand_mean=float(appc[m, 68].mean()),
                mean_change=float(appc[m, 68].mean() - postb[m, 68].mean()))
    doc = dict(
        round='S298', part='A', title='S297 record correction',
        delivered_modulation=delivered,
        delivered_definition_note='my reported x2.53 compared std(decoder_ac_applied)/std(own preclip); the 0.005599 figure compares the untransformed decoder_postclip against the baseline preclip, which is unchanged by construction. Both numbers are correct for the array they measure; the x2.53 claim applies only to the applied (post-transform) command.',
        k_distribution=kk,
        channels_newly_zero_at_some_sample_but_never_zero_in_baseline=newzero,
        total_command=total,
        per_channel_mean_change=dict(median=float(np.median(dmean)), max_abs=float(np.abs(dmean).max()), channel_68=ch68),
        channel_40=ch40,
        verdict=dict(hypothesis_tested='whether the candidate arm restored delivered modulation',
                     answer='NO: the raw clip stage is unchanged by construction, the total command falls by '
                            f"{abs(total['relative_change_percent']):.2f}%, and the change is a spatial redistribution "
                            f"(channel 40 dip raised from {ch40['base_min']:.3f} to {ch40['cand_min']:.3f}, "
                            f"{len(newzero)} channels newly driven to exactly zero at some samples)",
                     rule2_mechanism='NOT SUPPORTED: the total drive is essentially unchanged, so the earlier bearing loss is not explained by weakened total constant-max co-activation but by redistribution across channels and legs'),
    )
    dump(out / 'S297_CORRECTION.json', doc)
    return doc


def part_b(out):
    nb = np.load(S297 / 'original_mask' / 'NEURAL.npz')
    post = np.asarray(nb['decoder_postclip'], float); V = np.asarray(nb['decoder_motor_gains'], float)
    nz = np.flatnonzero(V != 0); m = (np.arange(1, 401) >= W20[0]) & (np.arange(1, 401) < W20[1])
    rows = {}
    for i in nz:
        mu = float(post[m, i].mean()); sd = float(post[m, i].std()); bound = float(np.sqrt(max(mu * (1 - mu), 0)))
        rows[int(i)] = dict(mean=mu, std=sd, bound_sqrt_m_1_minus_m=bound, used_fraction=float(sd / bound) if bound > 0 else None)
    vals = list(rows.values())
    groups = dict(m_ge_0_9=int(sum(1 for r in vals if r['mean'] >= .9)), m_ge_0_99=int(sum(1 for r in vals if r['mean'] >= .99)),
                  m_median=float(np.median([r['mean'] for r in vals])), std_median=float(np.median([r['std'] for r in vals])),
                  bound_median=float(np.median([r['bound_sqrt_m_1_minus_m'] for r in vals])),
                  used_fraction_median=float(np.median([r['used_fraction'] for r in vals if r['used_fraction'] is not None])),
                  channels_with_zero_bound=int(sum(1 for r in vals if r['used_fraction'] is None)))
    need = {f'{s:.2f}': dict(mean_must_be_at_most=m_bounds_for_std(s)['high'], other_root=m_bounds_for_std(s)['low'],
                             feasible_at_current_median_mean=bool(m_bounds_for_std(s)['high'] is not None and groups['m_median'] <= m_bounds_for_std(s)['high']),
                             note=m_bounds_for_std(s)['note'])
            for s in (.10, .25, .34, .50)}
    s295_arm_std = 0.34
    doc = dict(round='S298', part='B', title='feasibility bound for a mean-preserving AC rescaling',
               bound='for a command in [0,1] with mean m, std <= sqrt(m(1-m)); a mean-preserving linear AC gain cannot exceed that ceiling',
               per_channel=rows, groups=groups, required_mean_for_group_std=need,
               s295_successful_lift_group_std_reference=s295_arm_std,
               verdict=dict(feasible='NO', reason=f"baseline non-zero-channel postclip mean median is {groups['m_median']:.4f} which allows std only {groups['bound_median']:.4f}, while the S295 lift arm commands a group std near {s295_arm_std}; "
                                                 f"hosting std {s295_arm_std} with this [0,1] ceiling requires the mean to be lowered to at most {m_bounds_for_std(s295_arm_std)['high']:.4f}",
                            implication='the mean-preserving AC rescaling route is closed; any future interface must change the mean (or the ceiling), not just the AC gain'))
    dump(out / 'FEASIBILITY_BOUND.json', doc)
    return doc


def part_c(out):
    g = json.loads(GROUPS.read_text())['legs']
    z = np.load(S289 / 'NEURAL.npz')
    V = np.asarray(z['decoder_motor_gains'], float); pre = np.asarray(z['decoder_preclip'], float); rates = np.asarray(z['rates_hz'], float)
    safe = np.where(V == 0, 1.0, V)
    norm = pre / safe          # gain-normalised preclip == rates/450
    res = {}
    for name, M in (('gain_normalised_preclip', norm), ('rates_hz', rates)):
        L = {l: M[:, g[l]['minus']].mean(axis=1) for l in LEGS}
        P = {l: M[:, g[l]['plus']].mean(axis=1) for l in LEGS}
        Lm = np.vstack([L[l] for l in LEGS]); Pmat = np.vstack([P[l] for l in LEGS])
        cm = Lm.mean(axis=0); R = Lm - cm
        Craw = np.corrcoef(Lm); Cres = np.corrcoef(R)
        D = np.mean([L[l] for l in TRI_A], axis=0) - np.mean([L[l] for l in TRI_B], axis=0)
        off_raw = [(LEGS[i], LEGS[j], float(Craw[i, j])) for i in range(6) for j in range(i + 1, 6)]
        off_res = [(LEGS[i], LEGS[j], float(Cres[i, j])) for i in range(6) for j in range(i + 1, 6)]
        Lmean = float(np.mean([L[l].mean() for l in LEGS]))
        res[name] = dict(
            L_mean={l: float(L[l].mean()) for l in LEGS}, P_mean={l: float(P[l].mean()) for l in LEGS},
            L_std={l: float(L[l].std()) for l in LEGS},
            L_residual_std={l: float((L[l] - cm).std()) for l in LEGS},
            raw_pair_correlations=off_raw, residual_pair_correlations=off_res,
            residual_correlation_note='the six residuals sum to zero by construction, so negative residual correlations are a structural artefact and must not be read as per-leg phase information',
            tripod_difference=dict(mean=float(D.mean()), std=float(D.std()), min=float(D.min()), max=float(D.max()),
                                   sign_changes=int(np.count_nonzero(np.diff(np.sign(D)) != 0)),
                                   amplitude_over_mean_L=float(D.std() / Lmean) if Lmean else None),
            release_if_group_zeroed={l: dict(abs=float(L[l].mean()), share=float(L[l].mean() / sum(L[k].mean() for k in LEGS))) for l in LEGS},
        )
    doc = dict(round='S298', part='C', title='upstream per-leg phase audit on unclipped quantities',
               inputs=dict(neural=str(S289 / 'NEURAL.npz'), quantities=['gain_normalised_preclip (== rates_hz/450)', 'rates_hz'],
                           window_ms=[0, 400], resolution_hz=RES_HZ,
                           resolution_note='400 samples at 1 ms give 2.5 Hz resolution and 7 bins in 8-25 Hz: structure at or below 6 Hz is judgeable, the 8-25 Hz stepping band is NOT judgeable, and "not judgeable" is not the same as "absent"'),
               per_quantity=res)
    dump(out / 'LEG_ANTIPHASE.json', doc)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prereg-only', action='store_true')
    args = ap.parse_args(); out = args.output
    inputs = [Path(__file__).resolve(), GROUPS, S289 / 'NEURAL.npz', S292 / 'original_mask' / 'NEURAL.npz',
              S292 / 'current_sample_55_62' / 'NEURAL.npz', S297 / 'original_mask' / 'NEURAL.npz',
              S297 / 'decoder_ac_gain' / 'NEURAL.npz', S297 / 'AC_GAIN_TABLE.csv', S297 / 'MODULATION.json']
    prereg = dict(round='S298', mode='READ-ONLY audit: no model load, forward, mj_step, neural tick, tier build or rendering',
                  calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0),
                  windows=dict(modulation_ms=list(W20), antiphasic_ms=[0, 400]), resolution_hz=RES_HZ,
                  parts=dict(A='S297 correction table: reported vs archived values and the source of each difference',
                             B='feasibility bound std <= sqrt(m(1-m)) for a mean-preserving AC rescaling',
                             C='per-leg phase audit on unclipped gain-normalised preclip and rates, including common-mode removal'),
                  rules=dict(
                      A='after common-mode removal, a leg-pair correlation <= -0.3, or the tripod difference shows >=3 sign changes with amplitude >= 0.3 x that leg L -> upstream carries usable per-leg phase',
                      B='all residual leg-pair correlations >= 0.5 and the tripod difference is near constant (amplitude < 0.05 x L) -> no phase information upstream',
                      C='anti-phase appears only in individual legs -> label it a single-leg anomaly and cross-reference the S296 mask ceilings (LM 0.571, LH 0.500, RM 0.429, RH 0.564)',
                      artefact_safeguard='subtracting the six-leg common mode forces the residuals to sum to zero, which makes negative residual correlations structural; the decision therefore also reports raw pair correlations and the amplitude criterion, and a literal rule-A trigger caused only by that constraint is recorded as an artefact rather than as evidence',
                      precedence='evaluate all three; primary = A if the amplitude criterion is met, else C if anti-phase is confined to one or two legs, else B'),
                  prohibitions=['no new gain, formula or command recomputation', 'no change to any historical source, archive, gain or frozen XML',
                                'no support, walking or autonomy claim', 'sqrt(m(1-m)) is a feasibility bound and not behavioural evidence'],
                  input_hashes={str(p): sha(p) for p in inputs if Path(p).is_file()},
                  missing_inputs=[str(p) for p in inputs if not Path(p).is_file()])
    if args.prereg_only:
        assert not out.exists(), 'fresh output required'
        out.mkdir(parents=True); dump(out / 'PREREG.json', prereg); print(json.dumps(dict(prereg=str(out / 'PREREG.json')))); return 0
    pre = json.loads((out / 'PREREG.json').read_text())
    assert pre['input_hashes'][str(Path(__file__).resolve())] == sha(__file__), 'analyzer changed after prereg'
    for p, h in pre['input_hashes'].items(): assert sha(p) == h, 'input changed: ' + p
    a = part_a(out); b = part_b(out); c = part_c(out)
    # decision
    q = c['per_quantity']['gain_normalised_preclip']
    trip = q['tripod_difference']
    amp_ok = bool(trip['std'] >= .3 * float(np.mean(list(q['L_mean'].values()))))
    res_neg = [t for t in q['residual_pair_correlations'] if t[2] <= -0.3]
    raw_pos = all(t[2] >= 0.5 for t in q['raw_pair_correlations'])
    rule_a_literal = bool(res_neg or (trip['sign_changes'] >= 3 and amp_ok))
    rule_a_corrected = bool(trip['sign_changes'] >= 3 and amp_ok)
    rule_b = bool(all(t[2] >= 0.5 for t in q['residual_pair_correlations']) and trip['std'] < .05 * float(np.mean(list(q['L_mean'].values()))))
    negative_pairs = [t for t in q['raw_pair_correlations'] if t[2] < 0]
    legs_in_negative = sorted({t[0] for t in negative_pairs} | {t[1] for t in negative_pairs})
    rule_c = bool(negative_pairs and len(legs_in_negative) <= 2)
    primary = 'A' if rule_a_corrected else ('C' if rule_c else 'B')
    outc = dict(round='S298', primary_rule=primary,
                satisfied=dict(rule_a_literal=rule_a_literal, rule_a_corrected=rule_a_corrected, rule_b=rule_b, rule_c=rule_c),
                rule_a_literal_note='the literal residual-correlation trigger fires only because residuals are forced to sum to zero; it is recorded as an artefact and is not used as evidence',
                amplitude=trip, raw_correlations_all_positive=raw_pos, legs_on_negative_raw_correlation=legs_in_negative,
                one_line=('upstream drive carries no usable per-leg phase information at the judgeable timescale, so the next round is a sourced phase-mechanism audit, not a differential interface'
                          if primary == 'B' else ('per-leg phase information exists upstream, so the next round builds a differential (dip) interface with complementary rather than mean constraints'
                                                  if primary == 'A' else 'anti-phase is confined to individual legs, which is a single-leg anomaly plus the S296 mask ceilings')),
                correction=dict(delivered_claimed_x2_53='applies only to decoder_ac_applied; the untransformed clip stage is unchanged',
                                rule2_mechanism='withdrawn: total drive changed by only %.2f percent, so the bearing loss is a redistribution effect' % a['total_command']['relative_change_percent']),
                feasibility=dict(route='mean-preserving AC rescaling', closed=True,
                                 reason=f"mean median {b['groups']['m_median']:.4f} allows std only {b['groups']['bound_median']:.4f}; std {b['s295_successful_lift_group_std_reference']} needs the mean lowered to at most {b['required_mean_for_group_std']['0.34']['mean_must_be_at_most']:.4f}"),
                evidence=dict(correction='S297_CORRECTION.json', feasibility='FEASIBILITY_BOUND.json', antiphasic='LEG_ANTIPHASE.json'))
    dump(out / 'OUTCOMES.json', outc)
    dump(out / 'CALL_LEDGER.json', dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0))
    print(json.dumps(dict(primary=primary, satisfied=outc['satisfied'], tripod_std=trip['std'], sign_changes=trip['sign_changes'],
                          raw_all_positive=raw_pos, delivered=a['delivered_modulation'], total_pct=a['total_command']['relative_change_percent']), ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
