"""Are the model's `_a`/`_b` actuator pairs really two bundles of one muscle?

DECISION THIS SETTLES.  Several reading of the muscle asset turn on one question: if the model
splits a muscle into `_a` and `_b` actuators, are those two FIBRE CLUSTERS of one muscle (so one
motor pool must drive both, and their forces are constrained to move together) or two SEPARATE
MUSCLES (so they keep separate pools and separate commands)?  Guessing wrong silently removes a
degree of freedom from every stance and gait result.

WHAT IS MEASURED, and why it is decisive.  For each pair the file reports

  * `|r_a - r_b|` relative to `|r_a|`: how different the two actuators' generalized-force
    directions are.  Two bundles of one muscle pull through the same attachment group, so their
    moment arms coincide and the difference is small; two different muscles pull on different
    attachments and the difference is large.
  * `|r_a + r_b|` relative to `|r_a| + |r_b|`: how close the pair is to being collinear.  For a
    bundle pair this ratio is ~1 even when the two columns have different scales, because they
    are the SAME direction.
  * the same two numbers at several leg poses, because a coincidental match at one pose proves
    nothing.

Plus the ATTACHMENT EVIDENCE from the asset's own site names: the tergopleural promotor sites are
named `..._Miller28a_p*` and `..._Miller28b_p*` (two clusters of one attachment group), while the
pleural promotor is `..._Miller30_p*` (a different site), and the sterno-tergo pair carries no
attachment-group token at all.

Diagnostic only: this file changes no model and grants no runtime authorisation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

OUT = HERE / 'outputs/bundle_vs_muscle_round34'
XML = r15.XML
PENETRATION_MM = -0.050


def site_provenance(xml: Path):
    """Which attachment-group token does each actuator's tendon use?"""
    s = xml.read_text()
    groups = {}
    for m in re.finditer(r'<spatial name="([LR][FMH][CF]?_[A-Za-z0-9_\-]+?)_tendon">(.*?)</spatial>',
                         s, re.S):
        tendon, body = m.group(1), m.group(2)
        sites = re.findall(r'site="([^"]+)"', body)
        toks = set()
        for st in sites:
            for t in re.findall(r'Miller\d+[a-z]?|lff_[a-z]+_[a-z]+_[a-z]+_[a-z]|bifemlh_r|pp\d+',
                                st):
                toks.add(t)
        groups[tendon] = {'n_sites': len(sites), 'tokens': sorted(toks),
                          'site_names': sites[:4]}
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    prov = site_provenance(XML)

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key = data.qpos.copy()

    # a handful of different leg postures, so a single-pose coincidence cannot decide anything
    poses = {'keyframe': key.copy()}
    gen = np.random.default_rng(3)
    for k in range(3):
        q = key.copy()
        for leg in r16.LEGS:
            for j in ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw',
                      'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch'):
                jid = model.joint(f'joint_{leg}{j}').id
                lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
                q[model.jnt_qposadr[jid]] = float(np.clip(
                    q[model.jnt_qposadr[jid]] + gen.normal(0, 0.25), lo, hi))
        poses[f'sampled_{k}'] = q

    pairs = []
    for n in names:
        m = re.match(r'^(.*)_([ab])$', n)
        if not m:
            continue
        partner = f'{m.group(1)}_{"b" if m.group(2) == "a" else "a"}'
        if partner in names and n < partner:
            pairs.append((m.group(1), n, partner))

    rows = []
    for pose_name, q in poses.items():
        data.qpos[:] = q; data.qvel[:] = 0
        mj.mj_forward(model, data)
        R, _ = r15.actuator_moment_matrix(model, data, mj)
        for base, a, b in pairs:
            ra, rb = R[:, names.index(a)], R[:, names.index(b)]
            na, nb = float(np.linalg.norm(ra)), float(np.linalg.norm(rb))
            diff = float(np.linalg.norm(ra - rb))
            ssum = float(np.linalg.norm(ra + rb))
            rows.append({'pose': pose_name, 'base': base, 'a': a, 'b': b,
                         'norm_a': na, 'norm_b': nb,
                         'rel_direction_difference': diff / max(na, 1e-12),
                         'collinearity': ssum / max(na + nb, 1e-12)})

    # aggregate per pair across poses: the worst-case (largest) direction difference is the
    # conservative reading, because at some pose every pair can look aligned by accident
    agg = {}
    for r in rows:
        d = agg.setdefault(r['base'], {'rel_direction_difference': [], 'collinearity': [],
                                       'norm_ratio': []})
        d['rel_direction_difference'].append(r['rel_direction_difference'])
        d['collinearity'].append(r['collinearity'])
        d['norm_ratio'].append(min(r['norm_a'], r['norm_b']) / max(r['norm_a'], r['norm_b'], 1e-12))
    summary = []
    for base, d in sorted(agg.items()):
        tokens = prov.get(f'{base}_tendon') or prov.get(base, {})
        entry = {'base': base,
                 'rel_direction_difference_min': float(np.min(d['rel_direction_difference'])),
                 'rel_direction_difference_max': float(np.max(d['rel_direction_difference'])),
                 'collinearity_min': float(np.min(d['collinearity'])),
                 'collinearity_max': float(np.max(d['collinearity'])),
                 'norm_ratio_max': float(np.max(d['norm_ratio'])),
                 'attachment_tokens': tokens.get('tokens', []),
                 'n_sites': tokens.get('n_sites')}
        entry['reading'] = ('SAME DIRECTION ACROSS ALL POSES -> consistent with two fibre '
                            'clusters of one muscle'
                            if entry['rel_direction_difference_max'] < 0.10
                            else 'DIFFERENT DIRECTIONS -> behaves like two separate actuators, '
                                 'so coupling them removes a real degree of freedom')
        summary.append(entry)
    summary.sort(key=lambda e: e['rel_direction_difference_max'])

    report = {
        'status': 'BUNDLE_VS_MUSCLE_ROUND34',
        'question': 'are the model\'s `_a`/`_b` actuator pairs two fibre clusters of one muscle '
                    '(one pool, forces locked together) or two separate actuators?',
        'method': 'generalized-force direction comparison over 4 leg postures + the asset\'s own '
                  'attachment-group tokens',
        'poses': list(poses),
        'per_pair_per_pose': rows,
        'summary': summary,
        'conclusion_rule': {
            'same_muscle': 'rel_direction_difference_max < 0.10 at every pose',
            'separate_muscles': 'rel_direction_difference_max >= 0.10 at some pose',
            'note': 'the threshold is a reading aid, not a measurement of biology; the per-pose '
                    'values are the evidence'},
        'source_sha256': {str(XML): r15.digest(XML)},
        'limitations': [
            'A shared pull direction is necessary but not sufficient evidence for one muscle: two '
            'muscles can pull the same way.',
            'Moment arms are evaluated at four poses only.',
            'No model file is modified and no runtime use is authorised.'],
        'wall_seconds': float(time.time() - t0),
    }
    assert r15.digest(XML) == xml_before
    (out / 'bundle_vs_muscle.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    y = np.arange(len(summary))
    ax[0].barh(y, [e['rel_direction_difference_max'] for e in summary], color='tab:purple')
    ax[0].set_yticks(y); ax[0].set_yticklabels([e['base'] for e in summary], fontsize=7)
    ax[0].axvline(0.10, ls='--', c='red', label='reading aid: 0.10')
    ax[0].set_xlabel('max over poses of  |r_a - r_b| / max(|r_a|,|r_b|)')
    ax[0].set_title('Do the two actuators of a pair pull the same way?\n'
                    'small = consistent with one muscle; large = separate actuators')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='x')
    ax[1].barh(y, [e['collinearity_min'] for e in summary], color='tab:cyan')
    ax[1].set_yticks(y); ax[1].set_yticklabels([e['base'] for e in summary], fontsize=7)
    ax[1].axvline(0.99, ls='--', c='red', label='reading aid: 0.99')
    ax[1].set_xlabel('min over poses of  |r_a + r_b| / (|r_a|+|r_b|)')
    ax[1].set_title('How collinear is the pair?\n1.0 = the same direction with different scale')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis='x')
    fig.tight_layout(); fig.savefig(out / 'bundle_vs_muscle.png', dpi=110)

    print(json.dumps({'n_pairs': len(summary),
                      'same_muscle_like': [e['base'] for e in summary
                                           if e['rel_direction_difference_max'] < 0.10],
                      'separate_like': [e['base'] for e in summary
                                        if e['rel_direction_difference_max'] >= 0.10]},
                     indent=2))
    print(out)


if __name__ == '__main__':
    main()
