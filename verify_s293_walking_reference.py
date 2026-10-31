"""Independent S293 verification: recompute saved-trial gait metrics and audit the citation table.

Reads only saved arrays, the frozen XML and the archived JSON; never imports a simulator,
tier or neural module.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S292 = Path('/home/sensen/Desktop/engine修复验收/s292-current-sample-two-channel-calibration/20261026T181800Z')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
A, B = {'lf', 'rm', 'rh'}, {'rf', 'lm', 'lh'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def masks(arm, strict):
    meta = js(S292 / arm / 'MODEL_METADATA.json')
    leg_of = {int(k): v for k, v in meta['geom_leg'].items()}; floor = int(meta['floor'])
    z = np.load(S292 / arm / 'TRAJECTORY.npz')
    raw, offs = z['contact_raw'], z['contact_offsets'].astype(int)
    out = np.zeros((len(offs) - 1, len(LEGS)), bool)
    for i in range(len(offs) - 1):
        for row in raw[offs[i]:offs[i + 1]]:
            g1, g2, dist = int(row[0]), int(row[1]), float(row[2])
            if floor not in (g1, g2): continue
            other = g2 if g2 != floor else g1
            if strict and dist > 0: continue
            leg = leg_of.get(other)
            if leg is not None: out[i, LEGS.index(leg)] = True
    return z, out


def segments(m1d):
    """Maximal True runs as (start inclusive, end exclusive) sample indices."""
    idx = np.flatnonzero(np.diff(np.r_[False, np.asarray(m1d, bool), False].astype(int)) != 0)
    return [(int(a), int(b)) for a, b in zip(idx[0::2], idx[1::2])]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    checks = []
    def ck(name, val, **extra): checks.append(dict(name=name, passed=bool(val), **extra))
    try:
        pre = js(out / 'PREREG.json')
        ck('S293 round and zero-call prereg', pre['round'] == 'S293' and pre['calls'] == dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0))
        for p, h in pre['input_hashes'].items(): ck('preregistered input ' + p, Path(p).is_file() and sha(p) == h)
        ref = js(out / 'REFERENCE_TABLE.json')
        for e in ref['entries']:
            src = e['source']
            ck('citation completeness ' + e['id'], src['doi'].startswith('10.7554/') and src['pmcid'].startswith('PMC') and src['url'].startswith('https://pmc.ncbi.nlm.nih.gov/') and len(e['quote']) > 15)
        ck('both primary sources used', {e['source']['pmcid'] for e in ref['entries']} == {'PMC3545443', 'PMC7932689'})
        ck('19 reference entries recorded', len(ref['entries']) == 19)
        d = {e['id']: e for e in ref['derived']}
        ck('derived arithmetic displacement', math.isclose(d['displacement_400ms_representative']['value'], 0.028 * .4 * 1000, rel_tol=0, abs_tol=1e-12))
        ck('derived arithmetic step count', math.isclose(d['steps_400ms_fast']['value'], 400 / 60, rel_tol=0, abs_tol=1e-12))
        ck('derived duty inference flagged', d['duty_fast_inference']['value'] == .5 and 'not a printed' in d['duty_fast_inference']['evidence'])
        ck('joint amplitudes left unestablished', d['leg_joint_amplitude_reference']['value'] is None and d['body_length_mm']['value'] is None)
        # independent XML joint-range recomputation
        jr = js(out / 'MODEL_JOINT_RANGE.json'); tree = ET.fromstring(XML.read_bytes())
        parsed = {el.get('name'): (float(el.get('range').split()[0]), float(el.get('range').split()[1])) for el in tree.findall('.//joint') if el.get('range')}
        ck('joint range count matches XML', len(jr['joints']) == len(parsed) and jr['xml_sha256'] == sha(XML))
        for row in jr['joints']:
            lo, hi = parsed[row['joint']]
            ck('joint range ' + row['joint'], row['range_low_rad'] == lo and row['range_high_rad'] == hi and math.isclose(row['span_deg'], math.degrees(hi - lo), rel_tol=0, abs_tol=1e-12))
        gait = js(out / 'GAIT_COMPARISON.json'); ledger = js(out / 'CALL_LEDGER.json')
        ck('ledger all zero', all(v == 0 for v in ledger.values()))
        for arm in ('original_mask', 'current_sample_55_62'):
            saved = gait['arms'][arm]
            z, margin = masks(arm, False); _, strict = masks(arm, True)
            t = z['time'] * 1000
            ck(arm + ' sample count/duration', saved['samples'] == len(t) == 4001 and math.isclose(saved['duration_ms'], float(t[-1]), abs_tol=1e-9))
            ck(arm + ' strict-vs-margin difference', saved['both_definition_difference'] == int(np.count_nonzero(margin != strict)))
            for j, leg in enumerate(LEGS):
                s = saved['per_leg'][leg]
                st = segments(margin[:, j]); air = [((b - a) * .1, a == 0 or b == len(margin)) for a, b in segments(~margin[:, j])]
                ck(arm + ' ' + leg + ' airborne fraction', math.isclose(s['airborne_fraction'], 1 - float(margin[:, j].mean()), abs_tol=1e-12))
                ck(arm + ' ' + leg + ' stance episodes', s['stance_episodes_margin'] == len(st) and math.isclose(s['total_stance_ms'], float(margin[:, j].sum() * .1), abs_tol=1e-9))
                ck(arm + ' ' + leg + ' longest airborne and edges', math.isclose(s['longest_airborne_ms'], max([a[0] for a in air], default=0.0), abs_tol=1e-9) and s['airborne_gaps_ge10ms_all'] == sum(1 for a in air if a[0] >= 10) and s['airborne_gaps_ge10ms_interior'] == sum(1 for a in air if a[0] >= 10 and not a[1]) and s['airborne_gaps_lt10ms'] == sum(1 for a in air if 0 < a[0] < 10) and s['airborne_gaps_total'] == len(air))
                ck(arm + ' ' + leg + ' flicker definition', s['longest_stance_ms'] > 0 and s['airborne_fraction'] >= 0)
            a = np.all(margin == np.array([l in A for l in LEGS]), axis=1); b = np.all(margin == np.array([l in B for l in LEGS]), axis=1)
            ck(arm + ' tripod fractions/transitions', math.isclose(saved['fraction_tripod_either'], float(np.mean(a | b)), abs_tol=1e-12) and saved['tripod_transitions'] == int(np.count_nonzero(np.diff((a | b).astype(int)) != 0)))
            hist = {str(k): int(np.count_nonzero(margin.sum(axis=1) == k)) for k in range(7)}
            ck(arm + ' stance count histogram', hist == saved['stance_count_hist'])
            com = z['com']
            ck(arm + ' COM endpoints and drops', math.isclose(saved['com']['dz_native'], float(com[-1, 2] - com[0, 2]), abs_tol=1e-12) and saved['com']['z0'] == float(com[0, 2]) and saved['com']['z400'] == float(com[-1, 2]))
            ck(arm + ' horizontal displacement/path', math.isclose(saved['com']['horizontal_displacement_native'], float(np.hypot(com[-1, 0] - com[0, 0], com[-1, 1] - com[0, 1])), abs_tol=1e-12) and math.isclose(saved['com']['horizontal_path_length_native'], float(np.hypot(*np.diff(com[:, :2], axis=0).T).sum()), abs_tol=1e-12))
            ck(arm + ' no swing claim consistent with interior gaps', (saved['any_swing'] is False) == (saved['airborne_gaps_ge10ms_interior'] < 3))
            ck(arm + ' all five reference checks recorded', [c['id'] for c in saved['reference_checks']] == ['stepping_present', 'tripod_alternation', 'step_period', 'forward_travel', 'height_maintenance'])
            ck(arm + ' no check claimed satisfied beyond contact facts', all(c['satisfied'] is False for c in saved['reference_checks'][1:]))
        fig = js(out / 'FIGURE.json')
        ck('figure written and hashed', 'path' in fig and Path(fig['path']).is_file() and sha(fig['path']) == fig['sha256'])
        ck('no simulator or neural module imported', not any(m.split('.')[0] in ('mujoco',) or m.startswith('engine.') for m in __import__('sys').modules))
        result = dict(passed=all(c['passed'] for c in checks), checks=checks, failed=[c for c in checks if not c['passed']],
                      verifier_sha256=sha(__file__),
                      calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0),
                      limitations=['reference values are verbatim quotes from fetched PMC text; no re-fetch is performed here',
                                    'native units of the saved trial are not calibrated, so mm-vs-native comparisons are qualitative',
                                    'no per-joint walking angle amplitude or printed duty factor exists in the fetched text'])
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed']), ensure_ascii=False))
        return 0 if result['passed'] else 1
    except Exception as exc:
        import traceback
        result = dict(passed=False, checks=checks, error=repr(exc), traceback=traceback.format_exc(), verifier_sha256=sha(__file__))
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=False, error=repr(exc), checks=len(checks)), ensure_ascii=False)); return 1


if __name__ == '__main__': raise SystemExit(main())
