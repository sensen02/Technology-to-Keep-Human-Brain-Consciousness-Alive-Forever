"""S296/P3 read-only audit of lift/press drive time structure in saved neural trials.

Zero dynamic calls: no model, no forward, no step, no neural tick, no tier, no rendering.
Commands are taken verbatim from the stored decoder_postclip arrays (400 x 90, 1 ms grid)
and cross-checked against COMMAND_TIMING.decoded; nothing is recomputed with a new formula.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ARCH = Path('/home/sensen/Desktop/engine修复验收')
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
GROUPS = ARCH / 's295-phased-lift-probe/20261026T192901Z/DIRECTION_GROUPS.json'
NOTES = Path('/run/media/sensen/Data2/cell_wound_prototype/s290_provenance_notes.json')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = ('lf', 'rm', 'lh'), ('rf', 'lm', 'rh')
THRESHOLDS = (0.3, 0.5, 0.8, 1.0)
DWELL_MS = 10
BAND = (8.0, 25.0)
CONTACT_MS = 253.4
WINDOWS = {'common_0_150': (0., 150.), 'pre_nonfoot_20': (CONTACT_MS - 20, CONTACT_MS), 'late_253_4_400': (CONTACT_MS, 400.1)}
ARMS = {'s289_decoder_original': S289, 's292_original_mask': S292 / 'original_mask', 's292_current_sample_55_62': S292 / 'current_sample_55_62'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def runs(mask1d):
    idx = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask1d, bool), False].astype(int)) != 0)
    return [(int(a), int(b)) for a, b in zip(idx[0::2], idx[1::2])]


def threshold_table(series, dt_ms=1.0):
    out = {}
    for thr in THRESHOLDS:
        segs = [(a, b) for a, b in runs(series >= thr - 1e-12)]
        segs_ms = [dict(start_ms=a * dt_ms, end_ms=b * dt_ms, duration_ms=(b - a) * dt_ms) for a, b in segs]
        long_segs = [s for s in segs_ms if s['duration_ms'] >= DWELL_MS - 1e-9]
        out[str(thr)] = dict(reached=bool(segs), first_cross_ms=(segs_ms[0]['start_ms'] if segs_ms else None),
                             segment_count=len(segs_ms), dwell_ge10ms_count=len(long_segs),
                             longest_dwell_ms=(max([s['duration_ms'] for s in segs_ms]) if segs_ms else 0.0),
                             total_ms=float(np.count_nonzero(series >= thr - 1e-12) * dt_ms),
                             segments=segs_ms)
    return out


def spectrum(series, dt_s=.001):
    x = np.asarray(series, float) - np.mean(series)
    n = len(x); f = np.fft.rfftfreq(n, d=dt_s); p = np.abs(np.fft.rfft(x)) ** 2
    tot = p[1:].sum()
    band = (f >= BAND[0]) & (f <= BAND[1])
    pk = int(np.argmax(p[band])) if band.any() else None
    return dict(n=n, resolution_hz=float(f[1] - f[0]), band=BAND,
                peak_hz=float(f[band][pk]) if band.any() else None,
                band_power_fraction=float(p[band].sum() / tot) if tot > 0 else None,
                total_power_excluding_dc=float(tot),
                band_bins=int(band.sum()))


def window_stats(series, windows):
    out = {}
    for k, (a, b) in windows.items():
        m = (np.arange(len(series)) >= a) & (np.arange(len(series)) < b)
        out[k] = dict(start_ms=a, end_ms=b, samples=int(m.sum()), mean=float(series[m].mean()) if m.any() else None,
                      std=float(series[m].std()) if m.any() else None, max=float(series[m].max()) if m.any() else None)
    return out


def airborne_fraction(arm_dir, leg_of, floor):
    z = np.load(arm_dir / 'TRAJECTORY.npz')
    raw, offs = z['contact_raw'], z['contact_offsets'].astype(int)
    n = len(offs) - 1
    mask = np.zeros((n, len(LEGS)), bool)
    for i in range(n):
        for row in raw[offs[i]:offs[i + 1]]:
            g1, g2 = int(row[0]), int(row[1])
            if floor not in (g1, g2): continue
            other = g2 if g1 == floor else g1
            leg = leg_of.get(other)
            if leg is not None: mask[i, LEGS.index(leg)] = True
    return {leg: float(1 - mask[:, j].mean()) for j, leg in enumerate(LEGS)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--prereg-only', action='store_true')
    args = ap.parse_args(); out = args.output
    groups = json.loads(GROUPS.read_text())['legs']
    notes = json.loads(NOTES.read_text())
    zero = {r['index']: r for r in notes['per_zero_actuator']}
    members = {}
    for leg in LEGS:
        prov = {r['index']: r for r in groups[leg]['per_actuator']}
        members[leg] = {}
        for kind in ('minus', 'plus'):
            keep = [i for i in groups[leg][kind] if prov[i]['max_abs_moment'] > 0]
            members[leg][kind] = dict(indices=keep, names=[prov[i]['actuator'] for i in keep],
                                      zero_gain=[i for i in keep if i in zero],
                                      ceiling=float(len([i for i in keep if i not in zero]) / len(keep)) if keep else None)
    inputs = [Path(__file__).resolve(), GROUPS, NOTES]
    for arm, d in ARMS.items(): inputs += [d / 'NEURAL.npz', d / 'COMMAND_TIMING.npz']
    inputs += [S292 / 'original_mask' / 'TRAJECTORY.npz', S292 / 'original_mask' / 'MODEL_METADATA.json']
    prereg = dict(
        round='S296', mode='READ-ONLY saved-array audit: no model, forward, physics step, neural tick, tier build or rendering',
        calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0),
        command_source='stored decoder_postclip (400x90) on the 1 ms grid, verified equal to COMMAND_TIMING.decoded; applied array kept for cross-checks only, never recomputed',
        projection=dict(
            L_definition='L_leg(t) = arithmetic mean over that leg\'s minus-group member commands',
            P_definition='P_leg(t) = arithmetic mean over that leg\'s plus-group member commands',
            calibration='L=1.0 equals the command level of the successful S295 lift arm, so L is a lift-arm equivalent fraction directly comparable to the S295 differential lift 1.598-2.122',
            member_filter='only members used by S295 and carrying a nonzero S84 signed-moment entry',
            caution='linear projection proxy, not a lift prediction; the direction groups are only valid at the S292 t=0 pose; S295 established a systematic sign convention mismatch, so minus=lift and plus=press are the empirically observed labels and their relation to the S84 moment sign convention is UNDETERMINED',
            ceiling_definition='L_ceiling = active members / members; masked members can never contribute, so L is structurally capped per leg'),
        thresholds=list(THRESHOLDS), dwell_ms=DWELL_MS, band_hz=BAND, windows=WINDOWS, contact_reference_ms=CONTACT_MS,
        tripods=dict(A=list(TRI_A), B=list(TRI_B)),
        decision_rules=dict(
            A='no leg reaches L=0.3, or every >=0.3 dwell is shorter than 10 ms -> the drive lacks swing-scale magnitude and dwell; the coverage route is closed because the only sourced channels were 55 and 62 and S292 already ran them',
            B='a >=10 ms dwell exists at 0.3-0.5 but no leg reaches 0.8 -> lift intent is present but amplitude is insufficient; a sourced amplitude definition is required before any gain change',
            C='a leg reaches L>=0.8 with a >=10 ms dwell yet that leg never leaves the ground in the saved trajectory (airborne fraction <5%) -> drive present, effect absent; the landing point is drive-versus-load timing mismatch',
            D='all six legs show anti-phase structure in 8-25 Hz with >=3 periods -> rhythm already exists at the drive level, contradicting the S293 no-gait finding; the next round must explain the transmission failure',
            precedence='evaluate all four; report every satisfied rule; primary = A if A else C if C else D if D else B'),
        members=members, input_hashes={str(p): sha(p) for p in inputs if Path(p).is_file()},
        missing_inputs=[str(p) for p in inputs if not Path(p).is_file()],
    )
    if args.prereg_only:
        assert not out.exists(), 'fresh output required'
        out.mkdir(parents=True); dump(out / 'PREREG.json', prereg); print(json.dumps(dict(prereg=str(out / 'PREREG.json')))); return 0
    existing = json.loads((out / 'PREREG.json').read_text())
    assert existing['input_hashes'][str(Path(__file__).resolve())] == sha(__file__), 'analyzer changed after prereg'
    for p, h in existing['input_hashes'].items(): assert sha(p) == h, 'input changed: ' + p

    result = {'members': members, 'arms': {}}
    for arm, d in ARMS.items():
        ne = np.load(d / 'NEURAL.npz'); tm = np.load(d / 'COMMAND_TIMING.npz')
        cmd = np.asarray(ne['decoder_postclip'], float)
        assert cmd.shape == (400, 90) and np.array_equal(cmd, np.asarray(tm['decoded']))
        series = {}
        for leg in LEGS:
            L = cmd[:, members[leg]['minus']['indices']].mean(axis=1)
            P = cmd[:, members[leg]['plus']['indices']].mean(axis=1)
            series[leg] = dict(L=L, P=P)
        arm_out = {'per_leg': {}}
        for leg in LEGS:
            L, P = series[leg]['L'], series[leg]['P']
            ratio = np.divide(L, P, out=np.full_like(L, np.nan), where=P > 1e-12)
            arm_out['per_leg'][leg] = dict(
                L=dict(mean=float(L.mean()), std=float(L.std()), min=float(L.min()), max=float(L.max()),
                       ceiling=members[leg]['minus']['ceiling'], normalized_mean=float(L.mean() / members[leg]['minus']['ceiling']) if members[leg]['minus']['ceiling'] else None,
                       thresholds=threshold_table(L), spectrum=spectrum(L), windows=window_stats(L, WINDOWS), series=L.tolist()),
                P=dict(mean=float(P.mean()), std=float(P.std()), min=float(P.min()), max=float(P.max()),
                       ceiling=members[leg]['plus']['ceiling'],
                       thresholds=threshold_table(P), spectrum=spectrum(P), windows=window_stats(P, WINDOWS), series=P.tolist()),
                L_over_P=dict(mean=float(np.nanmean(ratio)) if np.isfinite(ratio).any() else None,
                              fraction_with_P_positive=float(np.mean(P > 1e-12))))
        A = np.mean([series[l]['L'] for l in TRI_A], axis=0); B = np.mean([series[l]['L'] for l in TRI_B], axis=0)
        D = A - B
        corr = np.corrcoef(np.vstack([series[l]['L'] for l in LEGS]))
        dead = np.abs(D) > .05
        sgn = np.sign(D[dead]); crossings = int(np.count_nonzero(np.diff(sgn) != 0)) if sgn.size else 0
        arm_out['coupling'] = dict(pairwise_L_correlation={LEGS[i]: {LEGS[j]: float(corr[i, j]) for j in range(6)} for i in range(6)},
                                   tripod_A_mean=float(A.mean()), tripod_B_mean=float(B.mean()),
                                   tripod_difference=dict(mean=float(D.mean()), std=float(D.std()), min=float(D.min()), max=float(D.max()),
                                                          sign_changes=crossings, approx_periods=float(crossings / 2),
                                                          anti_phase_ge3_periods=bool(crossings / 2 >= 3),
                                                          fraction_positive=float(np.mean(D > .05)), fraction_negative=float(np.mean(D < -.05))),
                                   leg_correlation_with_difference={l: float(np.corrcoef(series[l]['L'], D)[0, 1]) for l in LEGS})
        result['arms'][arm] = arm_out
    # per-leg contact status from the saved trajectory (S293 baseline)
    meta = json.loads((S292 / 'original_mask' / 'MODEL_METADATA.json').read_text())
    air = airborne_fraction(S292 / 'original_mask', {int(k): v for k, v in meta['geom_leg'].items()}, int(meta['floor']))
    result['airborne_fraction_baseline'] = air
    # sensitivity between the two S292 arms
    base, cand = result['arms']['s292_original_mask']['per_leg'], result['arms']['s292_current_sample_55_62']['per_leg']
    sens = {}
    for leg in LEGS:
        Lb, Lc = np.array(base[leg]['L']['series']), np.array(cand[leg]['L']['series'])
        sens[leg] = dict(L_correlation=float(np.corrcoef(Lb, Lc)[0, 1]) if Lb.std() > 0 and Lc.std() > 0 else None,
                         L_max_abs_difference=float(np.abs(Lb - Lc).max()),
                         L_max_base=base[leg]['L']['max'], L_max_candidate=cand[leg]['L']['max'],
                         threshold_table_identical={str(t): base[leg]['L']['thresholds'][str(t)]['segments'] == cand[leg]['L']['thresholds'][str(t)]['segments'] for t in THRESHOLDS},
                         peak_hz_base=base[leg]['L']['spectrum']['peak_hz'], peak_hz_candidate=cand[leg]['L']['spectrum']['peak_hz'])
    result['sensitivity_s292'] = dict(per_leg=sens,
                                      note='compares the two S292 arms only; no new gain or command is computed here')
    # decision rules from recorded numbers
    legs_reach_03 = [l for l in LEGS if result['arms']['s292_original_mask']['per_leg'][l]['L']['thresholds']['0.3']['dwell_ge10ms_count'] > 0]
    any_reach_03 = [l for l in LEGS if result['arms']['s292_original_mask']['per_leg'][l]['L']['thresholds']['0.3']['reached']]
    any_reach_08_dwell = [l for l in LEGS if result['arms']['s292_original_mask']['per_leg'][l]['L']['thresholds']['0.8']['dwell_ge10ms_count'] > 0]
    mid_dwell = [l for l in LEGS if result['arms']['s292_original_mask']['per_leg'][l]['L']['thresholds']['0.5']['dwell_ge10ms_count'] > 0]
    airborne_legs = [l for l in LEGS if air[l] < .05]
    c_legs = [l for l in any_reach_08_dwell if l in airborne_legs]
    d_flag = bool(result['arms']['s292_original_mask']['coupling']['tripod_difference']['anti_phase_ge3_periods'])
    rules = dict(A=bool(not legs_reach_03), B=bool(mid_dwell and not any_reach_08_dwell), C=bool(c_legs), D=d_flag)
    primary = 'A' if rules['A'] else ('C' if rules['C'] else ('D' if rules['D'] else ('B' if rules['B'] else 'NONE')))
    result['rules'] = dict(satisfied=rules, primary=primary,
                           evidence=dict(legs_with_dwell_at_03=legs_reach_03, legs_reaching_03=any_reach_03,
                                         legs_with_dwell_at_08=any_reach_08_dwell, legs_with_dwell_at_05=mid_dwell,
                                         legs_never_airborne=airborne_legs, legs_08_dwell_but_grounded=c_legs,
                                         tripod_sign_changes=result['arms']['s292_original_mask']['coupling']['tripod_difference']['sign_changes']))
    dump(out / 'DRIVE_PROJECTION.json', result)
    for arm in ARMS:
        pl = result['arms'][arm]['per_leg']
        lines = ['time_ms,' + ','.join(f'L_{l},P_{l}' for l in LEGS)]
        for i in range(400):
            lines.append(f'{i + 1},' + ','.join(f'{pl[l]["L"]["series"][i]:.9f},{pl[l]["P"]["series"][i]:.9f}' for l in LEGS))
        (out / ('DRIVE_%s.csv' % arm)).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(out / 'THRESHOLD_CROSSINGS.json', {arm: {leg: dict(L=result['arms'][arm]['per_leg'][leg]['L']['thresholds'],
                                                             P=result['arms'][arm]['per_leg'][leg]['P']['thresholds']) for leg in LEGS} for arm in ARMS})
    dump(out / 'SPECTRUM.json', {arm: {leg: dict(L=result['arms'][arm]['per_leg'][leg]['L']['spectrum'],
                                                 P=result['arms'][arm]['per_leg'][leg]['P']['spectrum']) for leg in LEGS} for arm in ARMS})
    dump(out / 'LEG_COUPLING.json', {arm: result['arms'][arm]['coupling'] for arm in ARMS})
    dump(out / 'SENSITIVITY_S292.json', result['sensitivity_s292'])
    dump(out / 'OUTCOMES.json', dict(round='S296', primary_rule=primary, satisfied=rules, evidence=result['rules']['evidence'],
                                     airborne_fraction_baseline=air, ceilings={leg: dict(minus=members[leg]['minus']['ceiling'], plus=members[leg]['plus']['ceiling']) for leg in LEGS},
                                     no_dynamic_calls=True))
    dump(out / 'CALL_LEDGER.json', dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0))
    print(json.dumps(dict(rules=rules, primary=primary,
                          Lmax={l: round(result['arms']['s292_original_mask']['per_leg'][l]['L']['max'], 3) for l in LEGS},
                          ceil={l: members[l]['minus']['ceiling'] for l in LEGS})))
    return 0


if __name__ == '__main__': raise SystemExit(main())
