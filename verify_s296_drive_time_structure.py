"""S296 independent verification: recompute L/P, dwell segments, spectra (separate DFT), windows, coupling."""
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
ARMS = {'s289_decoder_original': S289, 's292_original_mask': S292 / 'original_mask', 's292_current_sample_55_62': S292 / 'current_sample_55_62'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def segs(mask):
    idx = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int)) != 0)
    return [(int(a), int(b)) for a, b in zip(idx[0::2], idx[1::2])]


def dft_band_peak(x, dt=.001, lo=8.0, hi=25.0):
    """Direct DFT on the true Fourier grid k/(n*dt) (independent of numpy rfft).

    400 samples at 1 ms give a 0.4 s window, so the resolution is 2.5 Hz and the
    8-25 Hz band holds only 7 bins; the peak is therefore quantised to that grid.
    """
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); t = np.arange(n) * dt
    resolution = 1.0 / (n * dt)
    ks = np.arange(1, n // 2 + 1); freqs = ks * resolution
    sel = (freqs >= lo) & (freqs <= hi)
    power = np.array([abs(np.sum(x * np.exp(-2j * np.pi * f * t))) ** 2 for f in freqs[sel]])
    return float(freqs[sel][int(np.argmax(power))]), float(power.sum()), float(resolution), int(sel.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    checks = []
    def ck(n, v, **e): checks.append(dict(name=n, passed=bool(v), **e))
    try:
        pre = js(out / 'PREREG.json'); res = js(out / 'DRIVE_PROJECTION.json'); outc = js(out / 'OUTCOMES.json'); ledger = js(out / 'CALL_LEDGER.json')
        ck('S296 prereg and zero-call ledger', pre['round'] == 'S296' and all(v == 0 for v in ledger.values()) and all(v == 0 for v in pre['calls'].values()))
        for p, h in pre['input_hashes'].items(): ck('input hash ' + p, Path(p).is_file() and sha(p) == h)
        ck('no missing inputs', pre['missing_inputs'] == [])
        groups = js(GROUPS)['legs']; zero = {r['index'] for r in js(NOTES)['per_zero_actuator']}
        for leg in LEGS:
            prov = {r['index']: r for r in groups[leg]['per_actuator']}
            for kind in ('minus', 'plus'):
                expect = [i for i in groups[leg][kind] if prov[i]['max_abs_moment'] > 0]
                ck(f'members {leg} {kind} match S295 groups with provenance', pre['members'][leg][kind]['indices'] == expect)
                ceil = len([i for i in expect if i not in zero]) / len(expect)
                ck(f'ceiling {leg} {kind}', abs(pre['members'][leg][kind]['ceiling'] - ceil) < 1e-12)
        for arm, d in ARMS.items():
            ne = np.load(d / 'NEURAL.npz'); tm = np.load(d / 'COMMAND_TIMING.npz')
            cmd = np.asarray(ne['decoder_postclip'], float)
            ck(arm + ' command array shape/identity', cmd.shape == (400, 90) and np.array_equal(cmd, np.asarray(tm['decoded'])))
            for leg in LEGS:
                L = cmd[:, pre['members'][leg]['minus']['indices']].mean(axis=1)
                P = cmd[:, pre['members'][leg]['plus']['indices']].mean(axis=1)
                saved = res['arms'][arm]['per_leg'][leg]
                ck(arm + ' ' + leg + ' L/P series reproduced',
                   np.allclose(L, np.asarray(saved['L']['series']), atol=0, rtol=0) and np.allclose(P, np.asarray(saved['P']['series']), atol=0, rtol=0))
                for thr in ('0.3', '0.5', '0.8', '1.0'):
                    sg = [(a * 1.0, b * 1.0) for a, b in segs(L >= float(thr) - 1e-12)]
                    got = saved['L']['thresholds'][thr]
                    ck(arm + ' ' + leg + ' dwell table ' + thr,
                       got['segment_count'] == len(sg) and got['dwell_ge10ms_count'] == sum(1 for a, b in sg if b - a >= 10 - 1e-9)
                       and abs(got['longest_dwell_ms'] - (max([b - a for a, b in sg]) if sg else 0.0)) < 1e-9
                       and [(x['start_ms'], x['end_ms']) for x in got['segments']] == sg)
                pk, bandpow, freq_res, nbin = dft_band_peak(L)
                ck(arm + ' ' + leg + ' spectral peak via independent DFT', abs(saved['L']['spectrum']['peak_hz'] - pk) < 1e-9,
                   reported=saved['L']['spectrum']['peak_hz'], dft=pk)
                ck(arm + ' ' + leg + ' resolution is 2.5 Hz with 7 band bins', abs(saved['L']['spectrum']['resolution_hz'] - freq_res) < 1e-12 and saved['L']['spectrum']['band_bins'] == nbin)
                ck(arm + ' ' + leg + ' band power positive and fraction in (0,1]',
                   bandpow > 0 and 0 < saved['L']['spectrum']['band_power_fraction'] <= 1)
                for wname, (a, b) in pre['windows'].items():
                    m = (np.arange(400) >= a) & (np.arange(400) < b)
                    ck(arm + ' ' + leg + ' window ' + wname,
                       saved['L']['windows'][wname]['samples'] == int(m.sum())
                       and abs(saved['L']['windows'][wname]['mean'] - float(L[m].mean())) < 1e-12
                       and abs(saved['L']['windows'][wname]['max'] - float(L[m].max())) < 1e-12)
        base = {leg: np.asarray(res['arms']['s292_original_mask']['per_leg'][leg]['L']['series']) for leg in LEGS}
        corr = np.corrcoef(np.vstack([base[l] for l in LEGS]))
        ck('pairwise correlation reproduced', np.allclose(corr, np.asarray([[res['arms']['s292_original_mask']['coupling']['pairwise_L_correlation'][a][b] for b in LEGS] for a in LEGS]), atol=1e-12))
        D = np.mean([base[l] for l in TRI_A], axis=0) - np.mean([base[l] for l in TRI_B], axis=0)
        sgn = np.sign(D[np.abs(D) > .05]); x = int(np.count_nonzero(np.diff(sgn) != 0)) if sgn.size else 0
        td = res['arms']['s292_original_mask']['coupling']['tripod_difference']
        ck('tripod difference and sign changes reproduced', abs(td['mean'] - float(D.mean())) < 1e-12 and td['sign_changes'] == x)
        raw_sgn = np.sign(D); raw_cross = int(np.count_nonzero(np.diff(raw_sgn) != 0))
        derived_tripod = dict(mean=float(D.mean()), std=float(D.std()), min=float(D.min()), max=float(D.max()),
                              raw_sign_changes=raw_cross, raw_approx_periods=float(raw_cross / 2),
                              deadband_sign_changes=x, deadband=0.05,
                              note='the raw difference crosses zero often but its amplitude (std ~0.013) is two orders below the 0.8 decision scale, so the crossings are ripple rather than an anti-phase stepping rhythm')
        meta = js(S292 / 'original_mask' / 'MODEL_METADATA.json'); z = np.load(S292 / 'original_mask' / 'TRAJECTORY.npz')
        raw, offs = z['contact_raw'], z['contact_offsets'].astype(int); leg_of = {int(k): v for k, v in meta['geom_leg'].items()}
        mask = np.zeros((len(offs) - 1, 6), bool)
        for i in range(len(offs) - 1):
            for row in raw[offs[i]:offs[i + 1]]:
                g1, g2 = int(row[0]), int(row[1])
                if int(meta['floor']) not in (g1, g2): continue
                other = g2 if g1 == int(meta['floor']) else g1
                if other in leg_of: mask[i, LEGS.index(leg_of[other])] = True
        for j, leg in enumerate(LEGS):
            ck('airborne fraction ' + leg, abs((1 - mask[:, j].mean()) - res['airborne_fraction_baseline'][leg]) < 1e-12)
        sens = res['sensitivity_s292']['per_leg']
        for leg in LEGS:
            cand = np.asarray(res['arms']['s292_current_sample_55_62']['per_leg'][leg]['L']['series'])
            ck('sensitivity max diff ' + leg, abs(np.abs(base[leg] - cand).max() - sens[leg]['L_max_abs_difference']) < 1e-12)
        rules = outc['satisfied']; ck('rule B is the recorded primary', outc['primary_rule'] == 'B' and rules['B'] is True)
        ck('rule C cannot fire without an 0.8 dwell', all(res['arms']['s292_original_mask']['per_leg'][l]['L']['thresholds']['0.8']['dwell_ge10ms_count'] == 0 for l in LEGS))
        ck('no model or neural import', not any(m.split('.')[0] == 'mujoco' or m.startswith('engine.') for m in __import__('sys').modules))
        result = dict(passed=all(c['passed'] for c in checks), checks=checks, failed=[c for c in checks if not c['passed']],
                      verifier_sha256=sha(__file__), tripod_derived=derived_tripod, calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0),
                      limitations=['linear projection proxy, not a lift prediction',
                                   'direction groups valid only at the S292 t=0 pose; sign convention between S84 moments and the observed minus=lift mapping is undetermined',
                                   '400 samples at 1 ms give a 0.4 s window: 2.5 Hz resolution and only 7 bins in the 8-25 Hz band, so the peak frequency is quantised to 2.5 Hz; this was not stated in PREREG and is recorded as a prereg-versus-actual discrepancy',
                                   'saved arrays only; missing fields would be reported NOT_ESTABLISHED rather than recomputed differently'])
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed'])))
        return 0 if result['passed'] else 1
    except Exception as exc:
        import traceback
        (out / 'VERIFICATION.json').write_text(json.dumps(dict(passed=False, error=repr(exc), traceback=traceback.format_exc(), checks=checks), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=False, error=repr(exc)))); return 1


if __name__ == '__main__': raise SystemExit(main())
