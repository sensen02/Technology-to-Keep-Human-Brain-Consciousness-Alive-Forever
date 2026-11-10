"""Offline validation of sensory encoder v2 against v1. No neural dynamics, no physics.

Fixtures are exact: they change one physical quantity at a time, so anything the encoder
reports as identical really is identical, and the rank figures are algebraic facts of the
declared transduction, not statistics.
"""
from pathlib import Path
import json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import encoder_v2 as ev2

W = ev2.BODY_WEIGHT_NATIVE


def main():
    mod = ev2._load_v1()
    v1 = mod.SensoryEncoder()
    enc = ev2.SensoryEncoderV2(v1)
    names = enc.joint_names
    nidx = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(5)
    lo, hi = v1.joint_ranges_rad.T
    q = (lo + hi) / 2
    v = np.zeros(42)
    load = np.full(6, W / 6)

    def diff(a, b):
        return float(np.abs(a - b).max())

    def changed_ports(a, b):
        return int(np.sum(a != b))

    cases = {}
    # 1) tibia angular velocity sign
    vp, vn = v.copy(), v.copy()
    j = nidx['joint_LMTibia_pitch']
    vp[j], vn[j] = +6.0, -6.0
    cases['tibia_rate_sign'] = dict(
        v1_max_abs_diff=diff(v1.encode(q, vp, load), v1.encode(q, vn, load)),
        v2_max_abs_diff=diff(enc.encode(q, vp, load), enc.encode(q, vn, load)),
        v2_ports_changed=changed_ports(enc.encode(q, vp, load), enc.encode(q, vn, load)))
    # 2) coxa axis aliasing: equal *normalised* displacement on two different axes makes the
    #    v1 mean identical by construction, so v1 cannot tell them apart at all
    def norm_to_rad(ji, target):
        l, h = v1.joint_ranges_rad[ji]
        return l + target * (h - l)

    qy, qr = q.copy(), q.copy()
    jy, jr = nidx['joint_LFCoxa_yaw'], nidx['joint_LFCoxa_roll']
    qy[jy] = norm_to_rad(jy, 0.5 + 0.10)
    qr[jr] = norm_to_rad(jr, 0.5 + 0.10)
    cases['coxa_axis_separation'] = dict(
        v1_max_abs_diff=diff(v1.encode(qy, v, load), v1.encode(qr, v, load)),
        v2_max_abs_diff=diff(enc.encode(qy, v, load), enc.encode(qr, v, load)))
    # 3) trochanter angle, previously invisible
    qt = q.copy()
    qt[nidx['joint_LHTrochanter_pitch']] += .20
    cases['trochanter_angle_visible'] = dict(
        v1_max_abs_diff=diff(v1.encode(q, v, load), v1.encode(qt, v, load)),
        v2_max_abs_diff=diff(enc.encode(q, v, load), enc.encode(qt, v, load)))
    # 4) trochanter velocity, still declared unobserved
    vt = v.copy()
    vt[nidx['joint_LHTrochanter_pitch']] = 5.0
    cases['trochanter_rate_still_unobserved'] = dict(
        v1_max_abs_diff=diff(v1.encode(q, v, load), v1.encode(q, vt, load)),
        v2_max_abs_diff=diff(enc.encode(q, v, load), enc.encode(q, vt, load)))
    # 5) coxa pitch velocity, still unobserved
    vc = v.copy()
    vc[nidx['joint_RMCoxa_pitch']] = 5.0
    cases['coxa_pitch_rate_still_unobserved'] = dict(
        v1_max_abs_diff=diff(v1.encode(q, v, load), v1.encode(q, vc, load)),
        v2_max_abs_diff=diff(enc.encode(q, v, load), enc.encode(q, vc, load)))
    # 6) load still visible
    l2 = load.copy(); l2[1] = 0.
    cases['foot_load_visible'] = dict(
        v1_max_abs_diff=diff(v1.encode(q, v, load), v1.encode(q, v, l2)),
        v2_max_abs_diff=diff(enc.encode(q, v, load), enc.encode(q, v, l2)))

    A1 = np.zeros((90, 24))
    for k, leg in enumerate(v1.foot_order):
        A1[v1.knee_indices[k], 4 * k + 0] = 1.
        for ax in v1.coxa_indices[k]:
            A1[ax, 4 * k + 1] = 1. / 3.
        A1[42 + v1.knee_indices[k], 4 * k + 2] = 1.
        A1[84 + k, 4 * k + 3] = 1.
    A2 = enc.independence_matrix()
    rank = dict(v1=int(np.linalg.matrix_rank(A1)), v2=int(np.linalg.matrix_rank(A2)),
                input_dimension=90)
    # current bounds and port usage
    probe = enc.encode(q, v * 0 + rng.normal(0, 20, 42), rng.uniform(-W, 3 * W, 6))
    bounds = dict(min=float(probe.min()), max=float(probe.max()),
                  within_declared=bool(probe.min() >= 0. and probe.max() <= .06),
                  nonzero_entries=int(np.count_nonzero(probe)),
                  v1_nonzero=int(np.count_nonzero(v1.encode(q, v, load))))
    out = dict(label='offline transduction comparison, no neural dynamics',
               cases=cases, rank=rank, bounds=bounds,
               ports=dict(v2_ports=int(len(enc.used_ports)), v2_unique=int(len(np.unique(enc.used_ports))),
                          v1_ports=int(sum(len(a) for leg in v1.foot_order
                                           for a in v1.channel_local_indices[leg].values())),
                          same_port_set=bool(np.array_equal(np.sort(enc.used_ports),
                                                            np.sort(np.concatenate([a for leg in v1.foot_order
                                                                                    for a in v1.channel_local_indices[leg].values()]))))),
               distinct_signals=dict(v1=24, v2=enc.n_signals),
               expectation=dict(tibia_rate_sign='v1 identical, v2 different',
                                coxa_axis_separation='v1 identical, v2 different',
                                trochanter_angle_visible='v1 identical, v2 different',
                                trochanter_rate_still_unobserved='both identical (declared)',
                                coxa_pitch_rate_still_unobserved='both identical (declared)',
                                foot_load_visible='both different'))
    ok = (cases['tibia_rate_sign']['v1_max_abs_diff'] == 0
          and cases['tibia_rate_sign']['v2_max_abs_diff'] > 0
          and cases['coxa_axis_separation']['v1_max_abs_diff'] == 0
          and cases['coxa_axis_separation']['v2_max_abs_diff'] > 1e-9
          and cases['trochanter_angle_visible']['v1_max_abs_diff'] == 0
          and cases['trochanter_angle_visible']['v2_max_abs_diff'] > 0
          and cases['trochanter_rate_still_unobserved']['v2_max_abs_diff'] == 0
          and cases['coxa_pitch_rate_still_unobserved']['v2_max_abs_diff'] == 0
          and cases['foot_load_visible']['v2_max_abs_diff'] > 0
          and rank['v2'] > rank['v1']
          and out['ports']['same_port_set']
          and bounds['within_declared'])
    out['all_expected'] = bool(ok)
    (HERE / 'ENCODER_V2_VALIDATION.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
    assert ok, 'encoder v2 did not behave as declared'


if __name__ == '__main__':
    main()
