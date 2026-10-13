"""Are the model's a/b actuators fibre clusters of ONE muscle, or separate muscles?

WHY THIS MATTERS.  Twelve actuators are unresolved because the model splits `trochanter_flexor`
and `sterno-tergo-trochanter_extensor` into an `_a` and a `_b` actuator while the documented
BANC naming has no a/b subdivision.  Round26/27 showed the remaining static-balance shortfall
sits on exactly those actuators (one of them would need -5.1% activation), so whether a and b
are two fibre clusters of one muscle decides whether one documented motor pool may legitimately
drive both (with a capacity-normalisation rule) or whether they must stay separate.

WHAT IS MEASURED, from the COMPILED model, with a built-in control:
  * each tendon's attachment SITE positions, expressed relative to that leg's own proximal
    anchor, for the a actuator, the b actuator and - as a control - the other muscle of the
    same family that is already resolved separately (e.g. accessory_trochanter_flexor);
  * the minimum site-to-site distance between the two sets, compared with the same quantity
    for the control pair, so "close" is judged against a same-leg reference, not an absolute;
  * the moment arm of each actuator about every joint of that leg, and the SIGN AGREEMENT
    between a and b versus between a and the control.

VERDICT RULES (fixed before looking at the numbers):
  CONSISTENT_WITH_FIBRE_CLUSTERS  (a,b separation <= control separation AND sign agreement
                                   >= 0.9)
  MORE_CONSISTENT_WITH_SEPARATE_MUSCLES (a,b separation > control separation AND sign
                                   agreement < 0.9)
  MIXED / CANNOT_DETERMINE otherwise, reported as such rather than forced into a verdict.

This is geometric evidence about the MODEL, not anatomical registration: the original
segmentation/cluster definition lives upstream and is not available locally, so a
CONSISTENT verdict supports a labelled, capacity-normalised shared pool - it does not prove
innervation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402

OUT = ROOT / 'outputs/ab_cluster_geometry_round27'
XML = r15.XML
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
# (a, b, control) actuator names per leg; the control is a neighbouring muscle of the same
# family that already has its own documented pool, so it is known to be a separate muscle.
FAMILIES = {
    'trochanter_flexor': ('{leg}F_trochanter_flexor_a', '{leg}F_trochanter_flexor_b',
                          '{leg}F_accesory_trochanter_flexor'),
    'sterno_tergo_extensor': ('{leg}F_sterno-tergo-trochanter_extensor_a',
                             '{leg}F_sterno-tergo-trochanter_extensor_b',
                             '{leg}F_trochanter_extensor'),
}
SIGN_AGREEMENT_AGREE = 0.9


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    names = [model.actuator(i).name or '' for i in range(model.nu)]
    index_of = {n: i for i, n in enumerate(names)}

    def tendon_sites(act_index):
        """Site names attached to this actuator's tendon, plus their world positions."""
        tid = int(model.actuator_trnid[act_index, 0])
        if int(model.actuator_trntype[act_index]) != int(mj.mjtTrn.mjTRN_TENDON):
            return []
        adr, num = int(model.tendon_adr[tid]), int(model.tendon_num[tid])
        out_sites = []
        for k in range(adr, adr + num):
            if int(model.wrap_type[k]) == int(mj.mjtWrap.mjWRAP_SITE):
                sid = int(model.wrap_objid[k])
                out_sites.append((mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, sid) or f'site{sid}',
                                  np.asarray(data.site_xpos[sid], float).copy()))
        return out_sites

    def moment_arms(act_index, leg):
        """Finite-difference moment arm of this actuator's tendon about each of the leg's joints."""
        arms = {}
        tid = int(model.actuator_trnid[act_index, 0])
        for j in range(model.njnt):
            nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or ''
            if not nm.startswith('joint_' + leg):
                continue
            qa = int(model.jnt_qposadr[j])
            base = float(data.qpos[qa])
            h = 1e-4
            vals = []
            for dq in (-h, +h):
                data.qpos[qa] = base + dq
                mj.mj_forward(model, data)
                vals.append(float(data.ten_length[tid]))
            data.qpos[qa] = base
            mj.mj_forward(model, data)
            arms[nm] = (vals[1] - vals[0]) / (2 * h)
        return arms

    def min_site_distance(sites_a, sites_b):
        if not sites_a or not sites_b:
            return None
        pa = np.array([s[1] for s in sites_a])
        pb = np.array([s[1] for s in sites_b])
        return float(np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)))

    results = []
    for family, (a_t, b_t, c_t) in FAMILIES.items():
        for leg in LEGS:
            a_n, b_n, c_n = a_t.format(leg=leg), b_t.format(leg=leg), c_t.format(leg=leg)
            if a_n not in index_of:
                continue
            ia, ib = index_of[a_n], index_of[b_n]
            ic = index_of.get(c_n)
            sa, sb = tendon_sites(ia), tendon_sites(ib)
            sc = tendon_sites(ic) if ic is not None else []
            d_ab = min_site_distance(sa, sb)
            d_ac = min_site_distance(sa, sc) if sc else None
            arms_a, arms_b = moment_arms(ia, leg), moment_arms(ib, leg)
            joints = sorted(set(arms_a) & set(arms_b))
            agree = [int(np.sign(arms_a[j]) == np.sign(arms_b[j])) for j in joints
                     if abs(arms_a[j]) > 1e-9 or abs(arms_b[j]) > 1e-9]
            sign_agreement = float(np.mean(agree)) if agree else float('nan')
            arms_c = moment_arms(ic, leg) if ic is not None else {}
            joints_c = sorted(set(arms_a) & set(arms_c))
            agree_c = [int(np.sign(arms_a[j]) == np.sign(arms_c[j])) for j in joints_c
                       if abs(arms_a[j]) > 1e-9 or abs(arms_c[j]) > 1e-9]
            sign_agreement_c = float(np.mean(agree_c)) if agree_c else float('nan')
            if d_ab is None or d_ac is None or not np.isfinite(sign_agreement):
                verdict = 'CANNOT_DETERMINE'
            elif d_ab <= d_ac and sign_agreement >= SIGN_AGREEMENT_AGREE:
                verdict = 'CONSISTENT_WITH_FIBRE_CLUSTERS'
            elif d_ab > d_ac and sign_agreement < SIGN_AGREEMENT_AGREE:
                verdict = 'MORE_CONSISTENT_WITH_SEPARATE_MUSCLES'
            else:
                verdict = 'MIXED'
            results.append({
                'family': family, 'leg': leg, 'a': a_n, 'b': b_n, 'control': c_n,
                'n_sites_a': len(sa), 'n_sites_b': len(sb), 'n_sites_control': len(sc),
                'min_site_distance_a_b_mm': d_ab,
                'min_site_distance_a_control_mm': d_ac,
                'sign_agreement_a_b': sign_agreement,
                'sign_agreement_a_control': sign_agreement_c,
                'n_joints_compared': len(joints),
                'moment_arms_a': arms_a, 'moment_arms_b': arms_b,
                'verdict': verdict})

    verdicts = {}
    for r in results:
        verdicts[r['verdict']] = verdicts.get(r['verdict'], 0) + 1
    summary = {
        'status': 'AB_CLUSTER_GEOMETRY',
        'question': 'are the model a/b actuators fibre clusters of one muscle (so one documented pool '
                    'may drive both with a capacity rule) or separate muscles (so they must stay apart)?',
        'source_sha256': {str(XML): r15.digest(XML)},
        'rules_fixed_before_measuring': {
            'CONSISTENT_WITH_FIBRE_CLUSTERS': 'a-b site separation <= control separation AND moment-arm '
                                              'sign agreement >= %.2f' % SIGN_AGREEMENT_AGREE,
            'MORE_CONSISTENT_WITH_SEPARATE_MUSCLES': 'a-b separation > control separation AND sign '
                                                     'agreement < %.2f' % SIGN_AGREEMENT_AGREE,
            'MIXED': 'anything else, reported as such'},
        'verdict_counts': verdicts,
        'results': results,
        'limitations': [
            'Finite-difference moment arms are pose dependent; geometric derivative is NOT physiological '
            'torque direction.',
            'Homology by name and joint axis is not anatomical registration.',
            'The original segmentation/cluster definition lives upstream and is not available locally, so '
            'a CONSISTENT verdict supports a LABELLED capacity-normalised shared pool and does not prove '
            'innervation.',
            'No model file is modified.'],
        'scope': 'geometry bookkeeping only; no simulation, no controller',
    }
    assert r15.digest(XML) == xml_before
    (out / 'ab_cluster_geometry.json').write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    labs = [f"{r['leg']}\n{r['family'][:12]}" for r in results]
    x = np.arange(len(results))
    axs[0].bar(x - .2, [r['min_site_distance_a_b_mm'] for r in results], .4, label='a vs b')
    axs[0].bar(x + .2, [r['min_site_distance_a_control_mm'] for r in results], .4,
               label='a vs control (separate muscle)')
    axs[0].set_xticks(x); axs[0].set_xticklabels(labs, fontsize=6, rotation=90)
    axs[0].set_ylabel('minimum site distance (mm)')
    axs[0].set_title('Are a and b closer to each other than to a known separate muscle?')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3, axis='y')
    axs[1].bar(x - .2, [r['sign_agreement_a_b'] for r in results], .4, label='a vs b')
    axs[1].bar(x + .2, [r['sign_agreement_a_control'] for r in results], .4, label='a vs control')
    axs[1].axhline(SIGN_AGREEMENT_AGREE, ls='--', c='green', label='fibre-cluster rule')
    axs[1].set_xticks(x); axs[1].set_xticklabels(labs, fontsize=6, rotation=90)
    axs[1].set_ylabel('moment-arm sign agreement')
    axs[1].set_title('Do a and b act in the same directions?')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3, axis='y')
    fig.suptitle('a/b actuator geometry: model evidence, NOT anatomical registration')
    fig.tight_layout(); fig.savefig(out / 'ab_cluster_geometry.png', dpi=110)

    print(json.dumps({'verdict_counts': verdicts,
                      'per_pair': [{'leg': r['leg'], 'family': r['family'],
                                    'ab_mm': None if r['min_site_distance_a_b_mm'] is None else round(r['min_site_distance_a_b_mm'], 4),
                                    'a_control_mm': None if r['min_site_distance_a_control_mm'] is None else round(r['min_site_distance_a_control_mm'], 4),
                                    'sign_ab': None if not np.isfinite(r['sign_agreement_a_b']) else round(r['sign_agreement_a_b'], 3),
                                    'verdict': r['verdict']} for r in results]}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
