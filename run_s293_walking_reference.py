"""S293/P3 real-walking reference acquisition + saved-trial gait comparison (read-only).

No simulator, tier or neural import: reads only the frozen XML, saved S292 arms and
public primary text. Reference numbers are only recorded as verbatim quotes with
DOI/PMC provenance; anything not stated in the fetched text is NOT_ESTABLISHED.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
ARMS = ('original_mask', 'current_sample_55_62')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
TRI_A, TRI_B = {'lf', 'rm', 'rh'}, {'rf', 'lm', 'lh'}

MENDES = dict(title='Quantification of gait parameters in freely walking wild type and sensory deprived Drosophila melanogaster',
              authors='Mendes CS, Bartos I, Akay T, Márka S, Mann RS', year=2013, journal='eLife 2:e00231',
              doi='10.7554/eLife.00231', pmcid='PMC3545443', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC3545443/')
CHUN = dict(title='Drosophila uses a tripod gait across all walking speeds, and the geometry of the tripod is important for speed control',
            authors='Chun C, Biswas T, Bhandawat V', year=2021, journal='eLife 10:e65878',
            doi='10.7554/eLife.65878', pmcid='PMC7932689', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC7932689/')

# Every value below is a verbatim excerpt from the fetched primary text of that article.
REFERENCE = [
 dict(id='speed_range', quantity='free-walking average speed range', value=[7.2, 44.7], unit='mm/s', source=MENDES, section='Results, step parameters',
      quote='Average speeds of each fly varied between 7.2 and 44.7 mm/s with 28 mm/s as the most representative speed'),
 dict(id='speed_representative', quantity='most representative average speed', value=28.0, unit='mm/s', source=MENDES, section='Results, step parameters',
      quote='Average speeds of each fly varied between 7.2 and 44.7 mm/s with 28 mm/s as the most representative speed'),
 dict(id='speed_classes', quantity='speed class boundaries', value=[19.9, 20.0, 33.9, 34.0], unit='mm/s', source=MENDES, section='Table 1 caption',
      quote='Slow \\[≤19.9 mm/s\\] | Medium \\[20–33.9 mm/s\\] | Fast \\[≥34 mm/s\\]'),
 dict(id='step_frequency_steady', quantity='step period steady state and frequency at fast speeds', value=60.0, unit='ms (period); 16 cycles/s', source=MENDES, section='Results, step parameters',
      quote='step period varies inversely with average speed up to approximately 30 mm/s, when step period reaches a steady state of about ∼60 ms, corresponding to a frequency of 16 cycles per s'),
 dict(id='stance_swing_scaling', quantity='dependence of stance and swing duration on speed', value='stance shortens, swing largely constant, equal at fastest', unit='qualitative', source=MENDES, section='Results, step parameters',
      quote='stance phase duration becomes shorter while swing phase duration remains largely constant; at the fastest speeds the durations of both swing and stance phases equalize'),
 dict(id='tripod_fraction', quantity='tripod-gait share of stance combinations by speed class', value={'slow': 31.37, 'medium': 51.63, 'fast': 64.98}, unit='percent of steps', source=MENDES, section='Table 1',
      quote='Tripod | 31.37 | Tripod | 51.63 | Tripod | 64.98'),
 dict(id='metachronal_fast', quantity='hindleg period equals metachronal lag at fast speeds (tripod signature)', value='HLagF ≈ Period', unit='qualitative', source=MENDES, section='Results, step parameters',
      quote='Data extracted from the fast set reveal that hindleg period equals the metachronal lag (HLagF ≈ PH), typical of the tripod gait'),
 dict(id='metachronal_slow', quantity='period-dependent metachronal lag at slow speeds (tetrapod signature)', value='HLagF = 0.505*PH + 27.6', unit='ms', source=MENDES, section='Results, step parameters',
      quote='flies walking slower than 20 mm/s showed a period-dependent metachronal lag with a regression curve of (HLagF = 0.505* PH + 27.6), typical of the tetrapod gait'),
 dict(id='step_length_scaling', quantity='step length dependence on speed', value='increases almost linearly with speed', unit='qualitative', source=MENDES, section='Results, step parameters',
      quote='step length increased almost linearly with speed'),
 dict(id='walk_track_length', quantity='recorded straight-walk distance per trial', value=1.27, unit='cm (average)', source=MENDES, section='Results, high resolution tracking',
      quote='The distance recorded was on average 1.27 cm'),
 dict(id='m_tripod_all_speeds', quantity='gait is M-tripod across the entire speed range', value='M-tripod', unit='qualitative', source=CHUN, section='Introduction',
      quote='Flies employ a modified tripod (M-tripod) gait throughout their entire speed range with only a small dependence on speed'),
 dict(id='com_speed_midstance_max', quantity='horizontal COM speed profile within a step', value='maximum at mid-stance', unit='qualitative', source=CHUN, section='Results',
      quote='the CoM typically reached a maximum horizontal speed at mid-stance'),
 dict(id='com_height_midstance_max', quantity='vertical COM profile and speed dependence', value='mid-stance height maximum; more erect when faster', unit='qualitative', source=CHUN, section='Results',
      quote='It is important to note that there is a mid-stance maximum in the height of the fly and that the flies are more erect when they are walking faster'),
 dict(id='step_distance_scaling', quantity='distance per tripod stance increases, duration decreases with speed', value='qualitative', unit='qualitative', source=CHUN, section='Results',
      quote='The distance traveled over the tripod gait cycle also increases with speed'),
 dict(id='interleg_delays', quantity='normalised interleg stance delays within and across tripods', value={'within_tripod': 'small', 'opposing_tripod': 0.5}, unit='fraction of cycle', source=CHUN, section='Results',
      quote='the normalized delays between the legs of the same tripod were small throughout the entire speed range'),
 dict(id='opposing_phase', quantity='normalised delay across opposing tripods', value=0.5, unit='fraction of cycle', source=CHUN, section='Results',
      quote='the normalized delays across legs in the opposing tripods were 0.5'),
 dict(id='nonconforming_steps', quantity='share of steps not conforming to any gait at very low speed', value=4.0, unit='percent', source=CHUN, section='Results',
      quote='A small percentage (about 4%) of steps at very low speeds did not conform to any gait'),
 dict(id='com_tracking_resolution', quantity='vertical tracking resolution of COM', value=20.0, unit='um', source=CHUN, section='Results',
      quote='the vertical resolution being 20 μm'),
 dict(id='steps_analyzed', quantity='number of steps analysed', value=500, unit='steps (>500)', source=CHUN, section='Introduction',
      quote="we analyzed a fly's gait over >500 steps during which the fly is always walking straight"),
]
DERIVED = [
 dict(id='duty_fast_inference', quantity='duty factor at the fastest speeds if stance and swing equalise', value=0.5, unit='fraction',
      evidence='derived from stance_swing_scaling; not a printed duty-factor value', source=MENDES),
 dict(id='displacement_400ms_representative', quantity='forward distance covered in 400 ms at the representative speed', value=11.2, unit='mm',
      evidence='0.028 m/s x 0.4 s; exact arithmetic on speed_representative', source=MENDES),
 dict(id='steps_400ms_fast', quantity='number of 60 ms step periods within a 400 ms trial at fast speeds', value=400 / 60, unit='periods',
      evidence='400/60; exact arithmetic on step_frequency_steady; higher speeds up to ~20 Hz are not in the fetched claims', source=MENDES),
 dict(id='leg_joint_amplitude_reference', quantity='measured per-joint femur-tibia / coxa walking angle amplitudes', value=None, unit='deg',
      evidence='NOT_ESTABLISHED_IN_FETCHED_TEXT: neither fetched article reports per-joint walking angle amplitudes in degrees', source=None),
 dict(id='body_length_mm', quantity='body length in mm used for BL normalisation', value=None, unit='mm',
      evidence='NOT_ESTABLISHED_IN_FETCHED_TEXT: Mendes normalises to body size but the fetched text gives no BL value in mm', source=MENDES),
]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def joint_ranges():
    tree = ET.fromstring(XML.read_bytes())
    out = []
    for j in tree.findall('.//joint'):
        rng = j.get('range')
        if not rng: continue
        lo, hi = (float(x) for x in rng.split())
        out.append(dict(joint=j.get('name'), range_low_rad=lo, range_high_rad=hi, span_rad=hi - lo,
                        range_low_deg=float(np.degrees(lo)), range_high_deg=float(np.degrees(hi)), span_deg=float(np.degrees(hi - lo))))
    return out


def stance_masks(arm_dir, leg_of, floor, strict):
    z = np.load(arm_dir / 'TRAJECTORY.npz')
    raw, offs = z['contact_raw'], z['contact_offsets'].astype(int)
    n = 4001
    mask = np.zeros((n, len(LEGS)), bool)
    for i in range(n):
        for row in raw[offs[i]:offs[i + 1]]:
            g1, g2, dist = int(row[0]), int(row[1]), float(row[2])
            if floor not in (g1, g2): continue
            other = g2 if g1 == floor else g1
            leg = leg_of.get(other)
            if leg is None: continue
            if strict and dist > 0: continue
            mask[i, LEGS.index(leg)] = True
    return z, mask


def runs(mask1d):
    """Canonical maximal True runs as (start inclusive, end exclusive) sample indices."""
    idx = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask1d, bool), False].astype(int)) != 0)
    return list(zip(idx[0::2], idx[1::2]))


def gaps(mask1d):
    """Complement (airborne) intervals of a leg's stance mask, in ms.

    Intervals touching a trial edge are returned too, flagged edge=True: a leg that
    lifts early and never returns has one long airborne interval, which must not be
    silently dropped by edge filtering.
    """
    out = []
    for a, b in runs(~np.asarray(mask1d, bool)):
        out.append(dict(start_index=int(a), end_index=int(b), duration_ms=float((b - a) * .1),
                        edge=bool(a == 0 or b == len(mask1d))))
    return out


def gait(arm_dir, leg_of, floor):
    z, strict = stance_masks(arm_dir, leg_of, floor, strict=True)
    _, margin = stance_masks(arm_dir, leg_of, floor, strict=False)
    t = z['time'] * 1000.0
    res = dict(arm=str(arm_dir.name), samples=int(len(t)), duration_ms=float(t[-1]), per_leg={}, both_definition_difference=int(np.count_nonzero(strict != margin)))
    flicker = 0; swings_ge10 = 0; cycles = 0
    for j, leg in enumerate(LEGS):
        stances = runs(margin[:, j]); strict_stances = runs(strict[:, j])
        s = np.array([a for a, b in stances]); e = np.array([b for a, b in stances])
        strict_s = np.array([a for a, b in strict_stances])
        gs = gaps(margin[:, j]); g = np.array([x['duration_ms'] for x in gs], float)
        td = max(0, len(s) - int(margin[0, j]))
        flicker += int(np.count_nonzero((g > 0) & (g < 10)))
        swings_ge10 += int(np.count_nonzero((g >= 10) & np.array([not x['edge'] for x in gs], bool)))
        cycles += int(len(g))
        res['per_leg'][leg] = dict(
            stance_fraction_margin=float(margin[:, j].mean()), stance_fraction_strict=float(strict[:, j].mean()),
            airborne_fraction=float(1 - margin[:, j].mean()),
            stance_episodes_margin=int(len(s)), stance_episodes_strict=int(len(strict_s)),
            first_touch_ms=float(t[s[0]]) if len(s) else None,
            touchdowns_after_t0_margin=td,
            lift_offs_margin=int(len(e) - int(margin[-1, j])),
            longest_stance_ms=float(np.max(e - s) * .1) if len(s) else 0.0,
            longest_airborne_ms=float(g.max()) if g.size else 0.0,
            longest_airborne_is_edge=bool(any(x['edge'] for x in gs) and g.max() == max([x['duration_ms'] for x in gs if x['edge']] or [0])),
            airborne_gaps_total=int(len(g)),
            airborne_gaps_ge10ms_all=int(np.count_nonzero(g >= 10)),
            airborne_gaps_ge10ms_interior=int(np.count_nonzero((g >= 10) & np.array([not x['edge'] for x in gs], bool))),
            airborne_gaps_lt10ms=int(np.count_nonzero((g > 0) & (g < 10))),
            total_stance_ms=float(margin[:, j].sum() * .1),
            total_airborne_ms=float(g.sum()))
    stance_set = margin
    count = stance_set.sum(axis=1)
    a = np.all(stance_set == np.array([l in TRI_A for l in LEGS]), axis=1)
    b = np.all(stance_set == np.array([l in TRI_B for l in LEGS]), axis=1)
    res.update(total_touchdowns=int(sum(v['touchdowns_after_t0_margin'] for v in res['per_leg'].values())),
               short_contact_flicker_lt10ms=int(flicker), airborne_gaps_ge10ms_interior=int(swings_ge10), airborne_gap_count=int(cycles),
               any_swing=bool(swings_ge10 >= 3),
               stance_count_hist={str(k): int(np.count_nonzero(count == k)) for k in range(7)},
               fraction_exactly_3_legs=float(np.mean(count == 3)), fraction_tripod_A=float(a.mean()), fraction_tripod_B=float(b.mean()),
               fraction_tripod_either=float(np.mean(a | b)), fraction_all_6_legs=float(np.mean(count == 6)),
               tripod_transitions=int(np.count_nonzero(np.diff((a | b).astype(int)) != 0)),
               spectral_resolution_hz=2.5,
               spectral_note='400 ms window gives 2.5 Hz bins; a 16 Hz rhythm is not resolvable from Fz here, so stride period is judged from the contact raster, not from the spectrum')
    com = z['com']; fz = z['foot_force_world'][:, :, 2]
    res['com'] = dict(x0=float(com[0, 0]), y0=float(com[0, 1]), z0=float(com[0, 2]),
                      x400=float(com[-1, 0]), y400=float(com[-1, 1]), z400=float(com[-1, 2]),
                      horizontal_displacement_native=float(np.hypot(com[-1, 0] - com[0, 0], com[-1, 1] - com[0, 1])),
                      horizontal_path_length_native=float(np.hypot(*np.diff(com[:, :2], axis=0).T).sum()),
                      dz_native=float(com[-1, 2] - com[0, 2]), z_min=float(com[:, 2].min()), z_max=float(com[:, 2].max()),
                      z_argmin_ms=float(t[int(np.argmin(com[:, 2]))]))
    spec = {}
    for j, leg in enumerate(LEGS):
        sig = fz[:, j] - fz[:, j].mean()
        freq = np.fft.rfftfreq(len(sig), d=.0001)
        power = np.abs(np.fft.rfft(sig)) ** 2
        band = (freq >= .5) & (freq <= 30)
        pk = int(np.argmax(power[band]))
        spec[leg] = dict(peak_hz=float(freq[band][pk]), power_fraction_in_band=float(power[band].sum() / max(power[1:].sum(), 1e-30)),
                         fz_mean=float(fz[:, j].mean()), fz_std=float(fz[:, j].std()))
    res['spectrum'] = spec
    res['fz_total_sum_mean'] = float(fz.sum(axis=1).mean())
    res['body_nonfoot_contact_fraction'] = float(np.mean(z['body_penetrating_contacts'] > 0))
    return res, z


def compare(g):
    """Descriptive comparison against the quoted reference; never a walking-success claim."""
    swings = {leg: v['airborne_gaps_ge10ms_interior'] for leg, v in g['per_leg'].items()}
    checks = []
    checks.append(dict(id='stepping_present',
                       reference='a step needs a swing phase: feet leave the ground and re-contact repeatedly (fast-speed duty inferred 0.5)',
                       observed=dict(airborne_gaps_ge10ms_total=g['airborne_gaps_ge10ms_interior'],
                                     short_contact_flicker_lt10ms=g['short_contact_flicker_lt10ms'],
                                     longest_airborne_ms_per_leg={leg: v['longest_airborne_ms'] for leg, v in g['per_leg'].items()},
                                     legs_with_3_or_more_swings=[l for l, n in swings.items() if n >= 3]),
                       satisfied=bool(g['any_swing'])))
    checks.append(dict(id='tripod_alternation',
                       reference='tripod gait in 31-65% of steps; two triads alternating with opposing-triad delay 0.5 of cycle',
                       observed=dict(fraction_tripod_either=g['fraction_tripod_either'], tripod_transitions=g['tripod_transitions'],
                                     stance_count_hist=g['stance_count_hist']),
                       satisfied=bool(g['fraction_tripod_either'] > .05 and g['tripod_transitions'] >= 2)))
    period_ok = g['any_swing'] and g['fraction_tripod_either'] > .05
    checks.append(dict(id='step_period', reference='~60 ms period at fast speeds (16 cycles/s); comparison band 40-80 ms',
                       observed=dict(measurable=period_ok, note='needs repeated lift-off/touch-down per leg and alternation'),
                       satisfied=False))
    checks.append(dict(id='forward_travel', reference='~11.2 mm of straight walking per 400 ms at 28 mm/s',
                       observed=dict(horizontal_displacement_native=g['com']['horizontal_displacement_native'],
                                     horizontal_path_length_native=g['com']['horizontal_path_length_native'],
                                     unit='native; unit scale NOT_ESTABLISHED'),
                       satisfied=False))
    checks.append(dict(id='height_maintenance', reference='mid-stance height maximum, more erect at higher speed, no sustained sinking',
                       observed=dict(dz_native=g['com']['dz_native'], z_min=g['com']['z_min'], z_argmin_ms=g['com']['z_argmin_ms'],
                                     body_nonfoot_contact_fraction=g['body_nonfoot_contact_fraction']),
                       satisfied=False))
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args(argv).output
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute --output required')
    inputs = [Path(__file__).resolve(), XML] + [S292 / a / f for a in ARMS for f in ('TRAJECTORY.npz', 'MODEL_METADATA.json', 'CONTACTS.json')] + [S292 / 'PREREG.json', S292 / 'METRICS.json', S292 / 'MANIFEST_SHA256.json']
    out.mkdir(parents=True)
    dump(out / 'PREREG.json', dict(
        round='S293', goal='P3 obtain a citable real-walking kinematic reference and compare the saved S292 neural-driven trial against it; no simulator calls',
        mode='offline: frozen XML parse + saved S292 arrays + public primary text quotes', input_hashes={str(p): sha(p) for p in inputs},
        calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0),
        reference_sources=[MENDES, CHUN],
        predictions_before_analysis=dict(
            any_swing_expected=False, duty_expected_near_1=False, tripod_alternation_expected_false=True,
            height_expected_to_sink=True, forward_displacement_expected_small=True),
        acceptance_criteria=['a swing phase exists (duty<1) and repeats with a measurable period',
                             'stance sets alternate between the two tripods for a sizeable fraction of time',
                             'forward displacement comparable to reference speed x duration while height is maintained and no non-foot contact occurs'],
        not_claimed=['walking success', 'biological equivalence of our decoded drive', 'absolute force or unit calibration', 'reference joint-angle amplitudes (not reported in fetched text)'],
    ))
    (out / Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    dump(out / 'REFERENCE_TABLE.json', dict(acquisition_date_utc='2026-10-26', fetched_by='web_fetch of PMC full text; verbatim excerpts', entries=REFERENCE, derived=DERIVED,
                                            unestablished=[e['id'] for e in DERIVED if e['value'] is None] + ['duty factor printed value', 'per-joint walking angle amplitudes', 'step length in mm', 'BL in mm']))
    jr = joint_ranges(); dump(out / 'MODEL_JOINT_RANGE.json', dict(xml=str(XML), xml_sha256=sha(XML), joints=jr,
        note='MuJoCo range in radians (compiler angle=radian); span is the mechanically expressible excursion, not an observed walking amplitude'))
    arms = {}
    for arm in ARMS:
        meta = json.loads((S292 / arm / 'MODEL_METADATA.json').read_text())
        leg_of = {int(k): v for k, v in meta['geom_leg'].items()}; floor = int(meta['floor'])
        arms[arm], _ = gait(S292 / arm, leg_of, floor)
        arms[arm]['forward_travel_reference_mm'] = 11.2
        arms[arm]['reference_checks'] = compare(arms[arm])
    dump(out / 'GAIT_COMPARISON.json', dict(arms=arms, primary_arm='original_mask',
        interpretation='Reference is published free-walking Drosophila kinematics; the saved trial is a 400 ms neural-driven trial with no walking goal implemented. Failing these checks is a null result, not evidence of a broken decoder.'))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        metas = {a: json.loads((S292 / a / 'MODEL_METADATA.json').read_text()) for a in ARMS}
        m1 = stance_masks(S292 / 'original_mask', {int(k): v for k, v in metas['original_mask']['geom_leg'].items()}, int(metas['original_mask']['floor']), strict=False)[1]
        m2 = stance_masks(S292 / 'current_sample_55_62', {int(k): v for k, v in metas['current_sample_55_62']['geom_leg'].items()}, int(metas['current_sample_55_62']['floor']), strict=False)[1]
        fig = plt.figure(figsize=(12, 12))
        ax0 = fig.add_subplot(3, 1, 1)
        ax0.imshow(m1.T.astype(float), aspect='auto', cmap='Greys', interpolation='nearest', extent=[0, 400, len(LEGS) - .5, -.5])
        ax0.set_yticks(range(len(LEGS))); ax0.set_yticklabels([l.upper() for l in LEGS]); ax0.set_xlabel('ms')
        ax0.set_title('original_mask stance raster (black=in contact): reference at ~16 Hz over 400 ms implies 6-7 stance/swing cycles, i.e. 6-7 black bands and gaps per leg')
        ax1 = fig.add_subplot(3, 1, 2)
        ax1.plot(np.arange(4001) * .1, m1.sum(axis=1), lw=.8, label='original_mask')
        ax1.plot(np.arange(4001) * .1, m2.sum(axis=1), lw=.8, alpha=.7, label='current_sample_55_62')
        ax1.axhline(3, ls='--', c='k', lw=.8, label='exact tripod = 3 stance legs')
        ax1.set_ylabel('legs in contact'); ax1.set_xlabel('ms'); ax1.set_ylim(0, 6.5); ax1.legend(fontsize=7)
        ax1.set_title('contact count: 4-5 legs planted throughout, tripod count (3) essentially absent')
        ax2 = fig.add_subplot(3, 1, 3)
        z = np.load(S292 / 'original_mask' / 'TRAJECTORY.npz'); fz = z['foot_force_world'][:, :, 2]; t = z['time'] * 1000
        for j, leg in enumerate(LEGS): ax2.plot(t, fz[:, j], lw=.7, label=leg.upper())
        for k in range(1, 7): ax2.axvline(60 * k, ls=':', c='r', lw=.6)
        ax2.set_title('per-leg foot Fz (native); red dotted lines = 60 ms reference step period')
        ax2.set_xlabel('ms'); ax2.legend(fontsize=7, ncol=6)
        fig.tight_layout(); fig.savefig(out / 'GAIT_REFERENCE_VS_TRIAL.png', dpi=160)
        dump(out / 'FIGURE.json', dict(path=str(out / 'GAIT_REFERENCE_VS_TRIAL.png'), sha256=sha(out / 'GAIT_REFERENCE_VS_TRIAL.png'),
                                       panels=['stance raster original_mask', 'stance count both arms', 'per-leg Fz with 60 ms reference grid']))
    except Exception as exc:
        dump(out / 'FIGURE.json', dict(error=repr(exc)))
    dump(out / 'CALL_LEDGER.json', dict(model_load=0, forward=0, physics_step=0, neural_tick=0, tier_build=0, render=0))
    print(json.dumps(dict(entries=len(REFERENCE), arms={a: dict(touchdowns=v['total_touchdowns'], tripod=v['fraction_tripod_either'], dz=v['com']['dz_native'], horiz=v['com']['horizontal_displacement_native']) for a, v in arms.items()}), ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
