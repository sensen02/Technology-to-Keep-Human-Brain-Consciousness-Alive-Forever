"""S294 part 1 (offline, zero dynamic calls): vertical support ratio of the saved baseline.

support_ratio(t) = sum of foot-floor normal forces / (model mass x gravity). Also checks the
vertical momentum identity m*dvz = integral(sum Fz - W) dt, which validates the force and
weight accounting against the saved COM velocity without loading a simulator.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z/original_mask'
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
FIRST_NONFOOT = 253.4


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_weight():
    tree = ET.fromstring(XML.read_bytes())
    mass = sum(float(e.get('mass')) for e in tree.iter('inertial'))
    g = float(tree.find('.//option').get('gravity').split()[2])
    return mass, -g, mass * -g


def analyse(arm_dir, weight, dt=.0001):
    z = np.load(arm_dir / 'TRAJECTORY.npz')
    t = z['time'] * 1000.0
    fz = z['foot_force_world'][:, :, 2]
    total = fz.sum(axis=1)
    ratio = total / weight
    com = z['com']; vz = z['com_velocity'][:, 2] if 'com_velocity' in z else np.gradient(com[:, 2], dt)
    def win(a, b):
        m = (t >= a - 1e-9) & (t < b - 1e-9)
        return dict(start_ms=a, end_ms=b, samples=int(m.sum()), ratio_mean=float(ratio[m].mean()),
                    ratio_min=float(ratio[m].min()), ratio_max=float(ratio[m].max()),
                    frac_ge_1=float(np.mean(ratio[m] >= 1)), frac_ge_0_9=float(np.mean(ratio[m] >= .9)),
                    frac_le_0_1=float(np.mean(ratio[m] <= .1)),
                    total_fz_mean=float(total[m].mean()), com_z_start=float(com[m][0, 2]), com_z_end=float(com[m][-1, 2]),
                    com_dz=float(com[m][-1, 2] - com[m][0, 2]), vz_mean=float(vz[m].mean()), vz_min=float(vz[m].min()))
    windows = {k: win(*v) for k, v in dict(early_0_50=(0, 50), early_0_150=(0, 150), before_first_nonfoot=(0, FIRST_NONFOOT),
                                            after_first_nonfoot=(FIRST_NONFOOT, 400), late_350_400=(350, 400)).items()}
    # momentum identity over the pre-contact window: m*dvz should equal integral(Fz - W) dt
    m0 = t < FIRST_NONFOOT
    impulse = np.trapezoid(total[m0] - weight, t[m0] * .001)
    mass = weight / 9801.0
    pred_dvz = impulse / mass
    obs_dvz = float(vz[m0][-1] - vz[m0][0])
    per_leg = {leg: dict(mean_share_of_weight=float((fz[m0, j].mean()) / weight),
                         max_share_of_weight=float(fz[m0, j].max() / weight),
                         contact_fraction=float(np.mean(fz[:, j] > 1e-9))) for j, leg in enumerate(LEGS)}
    first_below = {}
    for thr in (0.9, 0.5, 0.1):
        idx = np.flatnonzero(ratio < thr)
        first_below[str(thr)] = float(t[idx[0]]) if idx.size else None
    return dict(arm=arm_dir.name, samples=int(len(t)), duration_ms=float(t[-1]), weight_native=weight,
                ratio_t0=float(ratio[0]), ratio_series_min=float(ratio.min()), ratio_series_max=float(ratio.max()),
                first_time_ratio_below_ms=first_below, windows=windows, per_leg=per_leg,
                momentum_check=dict(window='0-253.4ms', impulse_native_s=float(impulse), predicted_dvz=float(pred_dvz),
                                    observed_dvz=float(obs_dvz), abs_error=float(abs(pred_dvz - obs_dvz)),
                                    verdict='CONSISTENT' if abs(pred_dvz - obs_dvz) < .05 * max(abs(obs_dvz), 1e-9) else 'INCONSISTENT'),
                free_fall_verdict=('NOT_FREE_FALL_FROM_T0: ratio near 1 early with non-negative COM rise' if windows['early_0_150']['ratio_mean'] > .9 and windows['early_0_150']['com_dz'] >= 0
                                   else 'FREE_FALL_OR_SUPPORT_LOSS_EARLY'))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    mass, g, weight = model_weight()
    assert abs(weight - 24.446354755878) < 1e-9, weight
    res = dict(model=dict(xml=str(XML), xml_sha256=sha(XML), mass=mass, gravity=g, weight_native=weight),
               definition='support_ratio(t) = sum over foot-floor contacts of the upward world normal force / (m g); native units, absolute calibration NOT_ESTABLISHED',
               first_nonfoot_contact_reference_ms=FIRST_NONFOOT,
               arms={})
    for label, d in (('s292_original_mask', S292), ('s289_decoder_original', S289)):
        res['arms'][label] = analyse(d, weight)
    # csv for the primary arm
    z = np.load(S292 / 'TRAJECTORY.npz'); t = z['time'] * 1000; fz = z['foot_force_world'][:, :, 2]
    lines = ['time_ms,sum_fz_native,support_ratio,' + ','.join('fz_' + l for l in LEGS)]
    for i in range(0, len(t), 10):
        lines.append(f'{t[i]:.1f},{fz[i].sum():.9f},{fz[i].sum() / weight:.9f},' + ','.join(f'{v:.9f}' for v in fz[i]))
    (out / 'SUPPORT_RATIO.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    res['input_hashes'] = {str(p): sha(p) for p in (Path(__file__).resolve(), XML, S292 / 'TRAJECTORY.npz', S289 / 'TRAJECTORY.npz')}
    res['calls'] = dict(model_load=0, forward=0, physics_step=0, neural_tick=0)
    (out / 'SUPPORT_RATIO.json').write_text(json.dumps(res, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    a = res['arms']['s292_original_mask']
    print(json.dumps(dict(weight=weight, ratio_0_150=a['windows']['early_0_150']['ratio_mean'],
                          ratio_0_253=a['windows']['before_first_nonfoot']['ratio_mean'],
                          ratio_after=a['windows']['after_first_nonfoot']['ratio_mean'],
                          com_dz_0_150=a['windows']['early_0_150']['com_dz'],
                          com_dz_after=a['windows']['after_first_nonfoot']['com_dz'],
                          momentum=a['momentum_check']['verdict'], verdict=a['free_fall_verdict']), ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
