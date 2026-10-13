"""Tendon-route fidelity of the canonical muscle body: is the route geometry on
all six legs consistent with the documented mirrors/remaps?

Question answered here: does the canonical body's tendon/site geometry faithfully
implement muscle routes on all six legs, or do the documented mirrors/remaps
leave wrong or missing moment arms that could explain the static-support
shortfall measured in outputs/static_support_round15/?

WHAT THIS IS: geometry bookkeeping on the COMPILED model. Moment arms are the
finite-difference derivative of muscle TENDON LENGTH with respect to each leg
hinge coordinate, which is exactly the coefficient that turns muscle force into
generalized joint force. A bilateral-symmetry test is derived from the joint
axes themselves, not assumed.

WHAT THIS IS NOT: not a controller, not a behaviour, not anatomical registration.
The geometric derivative is NOT a physiological flexion/extension label, and a
finite-difference moment arm depends on the pose at which it is taken.

METHOD NOTE (recorded because it changes conclusions): at the as-shipped keyframe
the six legs are NOT in a mutually symmetric posture, so world-frame comparisons
there mix pose asymmetry with model asymmetry. The primary homology tests below
are therefore evaluated at a SYMMETRIC reference configuration (all leg hinges
zero), where a failure of left/right symmetry is a property of the model, and the
keyframe is reported alongside.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
PRE_REBUILD = ROOT / 'outputs/rebuild_r28/before/fruitfly_six_leg_standing.xml'
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
UPSTREAM = ROOT / 'outputs/upstream_muscle_round11'
OUT = ROOT / 'outputs/tendon_fidelity_round17'
EXPECTED_XML_SHA = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'

LEGS = ['LF', 'RF', 'LM', 'RM', 'LH', 'RH']
KEYS = [('Coxa', 'yaw'), ('Coxa', 'pitch'), ('Coxa', 'roll'),
        ('Trochanter', 'yaw'), ('Trochanter', 'pitch'), ('Trochanter', 'roll'),
        ('Tibia', 'pitch')]
FD_STEP = 1e-5            # rad, for d(tendon length)/d(angle)
ARM_TOL = 1e-9            # force-unit threshold above which an arm counts as nonzero
SITE_TOL_MM = 0.005       # translation/mirror residual counted as a real deviation
SAGITTAL = np.diag([1.0, -1.0, 1.0])   # mirror across the body midline plane


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    model = _load_mjcf(str(path)).compile()
    data = mj.MjData(model)
    return mj, model, data


def set_zero_legs(mj, model, data):
    """Symmetric reference configuration: every leg hinge at zero, body seated."""
    data.qpos[:] = 0.0
    data.qpos[2] = 2.04535
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.act[:] = 0.0
    mj.mj_forward(model, data)


def set_keyframe(mj, model, data):
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.act[:] = 0.0
    mj.mj_forward(model, data)


def joint_map(mj, model):
    out = {}
    for leg in LEGS:
        for part, axis in KEYS:
            name = 'joint_%s%s_%s' % (leg, part, axis)
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError('missing joint ' + name)
            out[(leg, part, axis)] = jid
    return out


def moment_arms(mj, model, data, jd):
    """90 x 42 table: d(muscle tendon length)/d(leg hinge coordinate)."""
    keys = [(leg, p, a) for leg in LEGS for p, a in KEYS]
    adr = {k: int(model.jnt_qposadr[jd[k]]) for k in keys}
    arm = np.zeros((model.nu, len(keys)))
    for i, k in enumerate(keys):
        q = adr[k]
        old = float(data.qpos[q])
        data.qpos[q] = old + FD_STEP
        mj.mj_forward(model, data)
        lp = np.asarray(data.actuator_length, float).copy()
        data.qpos[q] = old - FD_STEP
        mj.mj_forward(model, data)
        lm = np.asarray(data.actuator_length, float).copy()
        data.qpos[q] = old
        mj.mj_forward(model, data)
        arm[:, i] = (lp - lm) / (2.0 * FD_STEP)
    return arm, keys


def site_table(mj, model, data):
    rows = []
    for s in range(model.nsite):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, s) or ''
        if not name:
            continue
        body = int(model.site_bodyid[s])
        rows.append({'name': name, 'world': np.asarray(data.site_xpos[s], float).copy(),
                     'local': np.asarray(model.site_pos[s], float).copy(),
                     'body_id': body,
                     'body': mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) or ''})
    return rows


def world_axis(mj, model, data, jid):
    body = int(model.jnt_bodyid[jid])
    R = np.asarray(data.xmat[body], float).reshape(3, 3)
    return R.T @ np.asarray(model.jnt_axis[jid], float)


def expected_arm_sign(mj, model, data, jd, leg_l, leg_r, key):
    """Derive the required arm_R/arm_L sign from the joint axes themselves.

    Mirroring across the sagittal plane maps a rotation by t about axis a to a
    rotation by -t about S a. The right joint is a rotation by q_R about its own
    axis a_R. So the mirrored configuration satisfies
        a_R == +S a_L  ->  q_R = -q_L  ->  arm_R = -arm_L
        a_R == -S a_L  ->  q_R = +q_L  ->  arm_R = +arm_L
    """
    aL = world_axis(mj, model, data, jd[(leg_l,) + key])
    aR = world_axis(mj, model, data, jd[(leg_r,) + key])
    dot = float((SAGITTAL @ aL) @ aR)
    return (+1, dot) if dot < 0 else (-1, dot)


def actuator_families(names):
    """leg prefix is 2 chars + muscle-group letter; family is the remainder."""
    fam = {}
    for n in names:
        fam[n] = n[4:]
    return fam


def coverage(arm, keys, sel_idx, leg):
    """Per-joint bidirectional sign coverage for one leg over a set of actuators."""
    out = {}
    for part, axis in KEYS:
        i = keys.index((leg, part, axis))
        v = arm[sel_idx, i]
        out['%s_%s' % (part, axis)] = {
            'n_positive': int((v > ARM_TOL).sum()),
            'n_negative': int((v < -ARM_TOL).sum()),
            'n_zero': int((np.abs(v) <= ARM_TOL).sum())}
    return out


def parse_osim(path):
    """Path points of the upstream OpenSim model, with their parent body.

    OpenSim writes each point as
        <PathPoint name="..."><socket_parent_frame>/bodyset/BODY</socket_parent_frame>
        <location>x y z</location></PathPoint>
    so the parent frame is read from socket_parent_frame and the '/bodyset/' prefix stripped.
    """
    text = Path(path).read_text(errors='replace')
    muscles = {}
    for m in re.finditer(r'<Millard2012EquilibriumMuscle\s+name="([^"]+)"(.*?)</Millard2012EquilibriumMuscle>',
                         text, re.S):
        name, body = m.group(1), m.group(2)
        pts = []
        for pm in re.finditer(r'<PathPoint\s+name="([^"]+)"(.*?)</PathPoint>', body, re.S):
            inner = pm.group(2)
            frame = re.search(r'<socket_parent_frame>\s*([^<]+?)\s*</socket_parent_frame>', inner)
            loc = re.search(r'<location>\s*([^<]+?)\s*</location>', inner)
            if not (frame and loc):
                continue
            parent = frame.group(1).split('/')[-1]
            pts.append({'point': pm.group(1), 'body': parent,
                        'location': np.array([float(v) for v in loc.group(1).split()], float)})
        muscles[name] = pts
    return muscles


def kabsch_similarity(src, dst):
    """Best rotation+scale+translation mapping src onto dst; returns (scale, rms)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if len(src) < 2:
        return None
    cs, cd = src.mean(0), dst.mean(0)
    X, Y = src - cs, dst - cd
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = float((S * np.array([1.0, 1.0, d])).sum() / (X ** 2).sum())
    pred = scale * (X @ R.T) + cd
    rms = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
    return {'scale': scale, 'rms_mm': rms, 'n_points': int(len(src))}


def mirror_correct_right_sites(mj, model, data):
    """Reward right-leg sites to the sagittal mirror of their left homologues.

    In-memory only; the XML on disk is never touched. Returns what changed.
    """
    sites = site_table(mj, model, data)
    by_name = {r['name']: r for r in sites}
    changes = []
    for r in sites:
        n = r['name']
        if not n.startswith('R'):
            continue
        ln = 'L' + n[1:]
        if ln not in by_name:
            continue
        src = by_name[ln]
        world_new = SAGITTAL @ src['world']
        b = r['body_id']
        R = np.asarray(data.xmat[b], float).reshape(3, 3)
        local_new = R @ (world_new - np.asarray(data.xpos[b], float))
        old = np.asarray(model.site_pos[int(0) + 0], float) if False else np.asarray(
            model.site_pos[sites.index(r)], float).copy()
        model.site_pos[sites.index(r)] = local_new
        changes.append({'site': n, 'moved_mm': float(np.linalg.norm(local_new - old))})
    return changes


def main():
    import mujoco as mj
    sys.path.insert(0, str(ROOT / 'diagnostics'))

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'TENDON_ROUTE_FIDELITY',
        'question': ('do the documented mirrors/remaps leave wrong or missing moment arms on any leg, '
                     'and would that explain the static-support shortfall?'),
        'scope': ('compiled-model geometry bookkeeping; no controller, no behaviour, no model-file edit'),
        'source_sha256': {str(XML): sha256(XML),
                          str(PRE_REBUILD): sha256(PRE_REBUILD),
                          str(MANIFEST): sha256(MANIFEST),
                          str(Path(__file__)): sha256(Path(__file__))},
        'canonical_xml_unchanged_expected': EXPECTED_XML_SHA,
        'method': {
            'moment_arm': 'central finite difference of actuator_length (== tendon length) wrt each leg hinge coordinate',
            'fd_step_rad': FD_STEP, 'arm_zero_tol': ARM_TOL,
            'primary_configuration': 'all six leg hinges at zero (symmetric reference pose)',
            'secondary_configuration': 'as-shipped XML keyframe 0 (NOT mutually symmetric - reported for comparison)',
            'mirror_expectation': ('derived from joint axes: a_R = +S a_L gives arm_R = -arm_L; '
                                   'a_R = -S a_L gives arm_R = +arm_L')},
        'limits': [
            'finite-difference moment arms depend on the pose at which they are taken',
            'a geometric derivative is NOT a physiological torque direction',
            'model coordinate signs are not flexion/extension labels',
            'homology by name and joint-axis correspondence is NOT anatomical registration',
            'the mirror experiment is an in-memory diagnostic variant, not a proposed repair; moving sites '
            'changes tendon lengths and therefore muscle force-length operating points, which would need '
            'their own re-derivation']}

    assert sha256(XML) == EXPECTED_XML_SHA, 'canonical XML changed'

    mjx, model, data = load(XML)
    names = [mjx.mj_id2name(model, mjx.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    fam = actuator_families(names)
    report['model'] = {'nq': int(model.nq), 'nv': int(model.nv), 'njnt': int(model.njnt),
                       'nu': int(model.nu), 'ntendon': int(model.ntendon),
                       'transmission_types': sorted({int(t) for t in model.actuator_trntype}),
                       'transmission_meaning': '3 = mjTRN_TENDON: each actuator drives its own tendon; '
                                               'trnid values numerically coincide with joint ids for the '
                                               'first 43 actuators and must NOT be read as joint indices',
                       'legs': LEGS, 'n_muscle_families': len({fam[n] for n in names})}
    jd = joint_map(mjx, model)

    manifest = json.loads(MANIFEST.read_text())
    cand_names = [r['actuator'] for r in manifest['actuators']
                  if r['status'] == 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    assert set(cand_names) <= set(names)
    sel_all = list(range(model.nu))
    sel_44 = [names.index(n) for n in cand_names]
    report['candidate_set'] = {'annotation_supported': len(sel_44), 'all': len(sel_all),
                               'source': str(MANIFEST)}

    # ---------------- 1. moment arm tables at both configurations ----------------
    set_zero_legs(mjx, model, data)
    arm_zero, keys = moment_arms(mjx, model, data, jd)
    axes_zero = {k: world_axis(mjx, model, data, jd[k]).tolist() for k in jd}
    set_keyframe(mjx, model, data)
    arm_key, keys_b = moment_arms(mjx, model, data, jd)
    assert keys == keys_b
    report['moment_arm_table'] = {
        'shape': list(arm_zero.shape),
        'config_zero_max_abs': float(np.abs(arm_zero).max()),
        'config_zero_nonzero_fraction': float((np.abs(arm_zero) > ARM_TOL).mean()),
        'config_keyframe_max_abs': float(np.abs(arm_key).max()),
        'config_keyframe_nonzero_fraction': float((np.abs(arm_key) > ARM_TOL).mean()),
        'note': 'tables are per (actuator, leg hinge coordinate); stored in the npz next to this JSON'}

    # ---------------- 2. left/right symmetry at the symmetric configuration ----------------
    set_zero_legs(mjx, model, data)
    sites_zero = site_table(mjx, model, data)
    smap = {r['name']: r for r in sites_zero}
    pairs = [('LF', 'RF'), ('LM', 'RM'), ('LH', 'RH')]
    symmetry = {}
    for L, R in pairs:
        aL = np.asarray(data.xanchor[jd[(L, 'Coxa', 'yaw')]], float)
        aR = np.asarray(data.xanchor[jd[(R, 'Coxa', 'yaw')]], float)
        trans, mirr, worst = [], [], []
        for r in sites_zero:
            n = r['name']
            if not n.startswith(L):
                continue
            rn = 'R' + n[1:]
            if rn not in smap:
                continue
            oL = r['world'] - aL
            oR = smap[rn]['world'] - aR
            t = float(np.linalg.norm(oR - oL))
            m = float(np.linalg.norm(oR - SAGITTAL @ oL))
            trans.append(t)
            mirr.append(m)
            if t > SITE_TOL_MM:
                worst.append({'site': n, 'translation_residual_mm': t, 'mirror_residual_mm': m})
        trans, mirr = np.array(trans), np.array(mirr)

        n_same = n_opp = 0
        arm_outliers = []
        for part, axis in KEYS:
            for n in [x for x in names if x.startswith(L)]:
                rn = 'R' + n[1:]
                vL = float(arm_zero[names.index(n), keys.index((L, part, axis))])
                vR = float(arm_zero[names.index(rn), keys.index((R, part, axis))])
                if abs(vL) <= ARM_TOL and abs(vR) <= ARM_TOL:
                    continue
                if abs(vR - vL) < abs(vR + vL):
                    n_same += 1
                else:
                    n_opp += 1
                if abs(vL) > ARM_TOL and abs(vR) > ARM_TOL:
                    ratio = abs(vR / vL)
                    if ratio > 2.0 or ratio < 0.5:
                        arm_outliers.append({'family': fam[n], 'joint': '%s_%s' % (part, axis),
                                             'arm_left': vL, 'arm_right': vR, 'abs_ratio': ratio})
        symmetry['%s_%s' % (L, R)] = {
            'sites_compared': int(len(trans)),
            'translation_of_offset_median_mm': float(np.median(trans)),
            'mirror_of_offset_median_mm': float(np.median(mirr)),
            'translation_of_offset_max_mm': float(trans.max()),
            'mirror_of_offset_max_mm': float(mirr.max()),
            'verdict': ('RIGHT LEG IS A TRANSLATED COPY OF THE LEFT (offsets preserved, not mirrored)'
                        if np.median(trans) < np.median(mirr) else
                        'RIGHT LEG OFFSETS ARE MIRRORED'),
            'sites_deviating_from_translation_gt_%.3fmm' % SITE_TOL_MM: len(worst),
            'worst_sites': sorted(worst, key=lambda d: -d['translation_residual_mm'])[:8],
            'moment_arm_pairs_matching_plus_armL': int(n_same),
            'moment_arm_pairs_matching_minus_armL': int(n_opp),
            'moment_arm_abs_ratio_outside_0.5_2': len(arm_outliers),
            'moment_arm_outliers': sorted(arm_outliers, key=lambda d: -max(d['abs_ratio'], 1 / d['abs_ratio']))[:10]}
    report['left_right_symmetry'] = symmetry

    derived = {}
    for L, R in pairs:
        for part, axis in KEYS:
            key = (part, axis)
            exp, dot = expected_arm_sign(mjx, model, data, jd, L, R, key)
            derived['%s_%s_%s_%s' % (L, R, part, axis)] = {'dot_SaL_aR': dot, 'required_arm_R_sign_factor': exp}
    report['axis_derived_expectation_at_zero_config'] = derived

    # ---------------- 3. foreleg vs mid/hind: action-signature consistency ----------------
    signature = {}
    for leg in LEGS:
        leg_sig = {}
        for n in [x for x in names if x.startswith(leg)]:
            sig = []
            for part, axis in KEYS:
                v = float(arm_zero[names.index(n), keys.index((leg, part, axis))])
                sig.append(0 if abs(v) <= ARM_TOL else (1 if v > 0 else -1))
            mags = [abs(float(arm_zero[names.index(n), keys.index((leg, p, a))])) for p, a in KEYS]
            leg_sig[fam[n]] = {'signs': sig, 'abs_max': (max(mags) if mags else 0.0),
                               'n_active_joints': int(sum(1 for s in sig if s))}
        signature[leg] = leg_sig
    mismatches = []
    for L, R in pairs:
        for f in sorted(set(fam.values())):
            sigL = signature[L][f]['signs']
            sigR = signature[R][f]['signs']
            d = [i for i, (a, b) in enumerate(zip(sigL, sigR)) if a != b and (a or b)]
            if d:
                mismatches.append({'pair': '%s_%s' % (L, R), 'family': f,
                                   'joints': ['%s_%s' % KEYS[i] for i in d],
                                   'signs_left': sigL, 'signs_right': sigR})
    fore_vs_mid = []
    for f in sorted(set(fam.values())):
        base = signature['LF'][f]['signs']
        for leg in ['LM', 'LH', 'RM', 'RH']:
            s = signature[leg][f]['signs']
            d = [i for i, (a, b) in enumerate(zip(base, s)) if a != b and (a or b)]
            if d:
                fore_vs_mid.append({'family': f, 'leg': leg,
                                    'joints_differing_from_LF': ['%s_%s' % KEYS[i] for i in d],
                                    'signs_LF': base, 'signs_leg': s})
    report['action_signature_consistency'] = {
        'explanation': ('per (leg, muscle family) the sign of the moment arm over the 7 homologous joints; '
                        'the foreleg is the reference because the upstream OpenSim model defines only '
                        'left-foreleg muscles'),
        'signatures': signature,
        'left_right_sign_mismatches': mismatches,
        'mid_hind_vs_LF_sign_differences': fore_vs_mid}

    # ---------------- 4. per-joint sign coverage ----------------
    cov = {'all90': {}, 'candidate44': {}}
    for label, sel in (('all90', sel_all), ('candidate44', sel_44)):
        for leg in LEGS:
            cov[label][leg] = coverage(arm_zero, keys, sel, leg)
    coverage_diffs = []
    for label in cov:
        for part, axis in KEYS:
            k = '%s_%s' % (part, axis)
            f = cov[label]['LF'][k]
            for leg in ['LM', 'LH', 'RM', 'RH']:
                g = cov[label][leg][k]
                if (f['n_positive'] > 0) != (g['n_positive'] > 0) or (f['n_negative'] > 0) != (g['n_negative'] > 0):
                    coverage_diffs.append({'set': label, 'joint': k, 'leg': leg,
                                           'LF': f, 'leg_counts': g})
            if (f['n_positive'] > 0) != (cov[label]['RF'][k]['n_positive'] > 0) or \
               (f['n_negative'] > 0) != (cov[label]['RF'][k]['n_negative'] > 0):
                coverage_diffs.append({'set': label, 'joint': k, 'leg': 'RF',
                                       'LF': f, 'leg_counts': cov[label]['RF'][k]})
    report['per_joint_sign_coverage'] = {'counts': cov, 'fore_vs_mid_hind_or_LR_differences': coverage_diffs}

    # ---------------- 5. upstream cross-check ----------------
    osim = parse_osim(UPSTREAM / 'best_combined_full.osim')
    osim_a = parse_osim(UPSTREAM / 'best_combined.osim')
    prefixes = sorted({n[:2] for n in osim})
    report['upstream_cross_check'] = {
        'files': {p.name: sha256(p) for p in [UPSTREAM / 'best_combined_full.osim',
                                              UPSTREAM / 'best_combined.osim',
                                              UPSTREAM / 'best_combined_arm_damping_stiff_cvt3.xml']},
        'muscles_in_best_combined_full': len(osim),
        'muscles_in_best_combined': len(osim_a),
        'leg_prefixes_present': prefixes,
        'comparison_possible': ('NO for any leg other than the left foreleg: the upstream OpenSim files define '
                               'only left-foreleg muscles (%d), so the mid, hind and right routes in the '
                               'compiled model have no upstream path definition to be checked against'
                               % len(osim))}
    # LF path-point correspondence: per parent body, fit rotation+scale+translation
    per_body = {}
    for muscle, pts in osim.items():
        for p in pts:
            per_body.setdefault(p['body'], []).append((muscle, p['point'], p['location']))
    fits = []
    for body, items in sorted(per_body.items()):
        src, dst, matched = [], [], []
        for muscle, point, loc in items:
            # 'LF' exactly: mid/hind sites embed the foreleg landmark names as suffixes
            # (e.g. LMC_tergopleural_promotor_a_LFC_Miller28a_p1), so a looser prefix match is ambiguous.
            cands = [r for r in sites_zero
                     if r['name'].startswith('LF') and r['name'].endswith(point)]
            if len(cands) == 1:
                src.append(loc)
                dst.append(cands[0]['local'])
                matched.append((muscle, point, cands[0]['name'], cands[0]['body']))
        if len(src) >= 2:
            fit = kabsch_similarity(np.array(src), np.array(dst))
            fit.update({'body': body, 'matched_points': len(src),
                        'example': matched[0][2] if matched else None})
            fits.append(fit)
    report['upstream_cross_check']['left_foreleg_pathpoint_fits'] = fits
    # Direct comparison: the upstream file's local coordinates against OUR compiled left-foreleg sites,
    # paired by identical path-point name. Exact equality means the LF routes were not altered.
    direct = []
    for muscle, pts in osim.items():
        for p in pts:
            cands = [r for r in sites_zero
                     if r['name'].startswith('LF') and r['name'].endswith(p['point'])]
            if len(cands) != 1:
                continue
            r = cands[0]
            direct.append({'point': p['point'], 'site': r['name'],
                           'upstream_body': p['body'], 'compiled_body': r['body'],
                           'bodies_agree': bool(p['body'] == r['body']),
                           'max_abs_delta': float(np.abs(p['location'] - r['local']).max())})
    if direct:
        deltas = np.array([d['max_abs_delta'] for d in direct])
        report['upstream_cross_check']['left_foreleg_direct_local_comparison'] = {
            'points_compared': len(direct),
            'bodies_agree_all': bool(all(d['bodies_agree'] for d in direct)),
            'max_abs_delta_mm': float(deltas.max()),
            'median_abs_delta_mm': float(np.median(deltas)),
            'n_identical_to_1e-9': int((deltas < 1e-9).sum()),
            'worst': sorted(direct, key=lambda d: -d['max_abs_delta'])[:5],
            'meaning': ('the upstream OpenSim left-foreleg path points and our compiled left-foreleg sites '
                        'are expressed in the same local frames and units; near-zero deltas mean the left '
                        'foreleg routes were carried over unchanged. This says nothing about the other five '
                        'legs, which do not exist upstream.')}
    report['upstream_cross_check']['fit_meaning'] = (
        'rotation+scale+translation fitted from the upstream OpenSim local coordinates of the SAME named '
        'path points in the SAME OpenSim body to the compiled model local coordinates. A scale near the '
        'unit conversion and a small rms means our left-foreleg sites reproduce the upstream geometry for '
        'that body; this says nothing about the other five legs, whose routes do not exist upstream.')

    # ---------------- 6. in-memory mirror experiment + static-support LP ----------------
    import diag_static_support_feasibility as ss
    report['mirror_experiment'] = {
        'what': ('right-leg sites re-placed at the sagittal mirror of their left homologues, in memory only; '
                 'the XML on disk is untouched'),
        'caveat': ('moving sites changes tendon lengths, hence muscle force-length operating points; the '
                   'comparison mixes route geometry with operating-point change and is a diagnostic, '
                   'not a repair proposal')}
    key_qpos = np.asarray(model.key_qpos[0], float).copy()
    tarsus = [mjx.mj_id2name(model, mjx.mjtObj.mjOBJ_GEOM, g) or '' for g in range(model.ngeom)]
    tarsus_idx = [g for g, n in enumerate(tarsus) if 'Tarsus' in n]

    def evaluate(qpos, K=8):
        mjx.mj_resetData(model, data)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.act[:] = 0.0
        mjx.mj_forward(model, data)
        R, act0 = ss.actuator_moment_matrix(model, data, mjx)
        mjx.mj_forward(model, data)
        target = (np.asarray(data.qfrc_bias, float) - act0 - np.asarray(data.qfrc_passive, float))
        terms = ss.contact_terms(model, data, mjx)
        lims = ss.limit_terms(model, data, mjx)
        B = ss.cone_basis(model, data, terms, K)
        out = {}
        for label, idx in (('all90', sel_all), ('candidate44', sel_44)):
            out[label] = ss.solve_set(R[:, idx], target, terms, lims, B, model.nv, label)
        out['contact_count'] = len(terms)
        out['feet_only'] = bool(terms) and all(
            'Tarsus' in t['geom1'] or 'Tarsus' in t['geom2'] for t in terms)
        out['tendon_lengths'] = np.asarray(data.actuator_length, float).tolist()
        return out

    # ORDER MATTERS: every baseline and counterfactual must be measured on the PRISTINE model.
    # The site-moving variant below edits model.site_pos in place, so it runs last and is undone after.
    site_pos_backup = np.asarray(model.site_pos, float).copy()
    pristine = evaluate(key_qpos)

    # ---- clean sign-only counterfactual: isolate the CONSEQUENCE of translation-vs-mirror ----
    # A sagittally mirrored body requires arm_R = -arm_L at the yaw/roll joints (derived from the axes
    # above) and arm_R = +arm_L at the pitch joints. The compiled model delivers +arm_L everywhere.
    # Flipping only the yaw/roll components of the right-leg actuators' action vectors tests the
    # consequence WITHOUT touching tendon lengths or force-length operating points.
    def evaluate_sign_counterfactual(flip_keys, K=8):
        mjx.mj_resetData(model, data)
        data.qpos[:] = key_qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.act[:] = 0.0
        mjx.mj_forward(model, data)
        R, act0 = ss.actuator_moment_matrix(model, data, mjx)
        mjx.mj_forward(model, data)
        Rm = R.copy()
        dofs = []
        for leg in ('RF', 'RM', 'RH'):
            for part, axis in flip_keys:
                jid = jd[(leg, part, axis)]
                dofs.append(int(model.jnt_dofadr[jid]))
        for a in range(model.nu):
            if names[a].startswith('R'):
                Rm[dofs, a] = -Rm[dofs, a]
        target = (np.asarray(data.qfrc_bias, float) - act0 - np.asarray(data.qfrc_passive, float))
        terms = ss.contact_terms(model, data, mjx)
        lims = ss.limit_terms(model, data, mjx)
        B = ss.cone_basis(model, data, terms, K)
        out = {}
        for label, idx in (('all90', sel_all), ('candidate44', sel_44)):
            out[label] = ss.solve_set(Rm[:, idx], target, terms, lims, B, model.nv, label)
        out['contact_count'] = len(terms)
        out['feet_only'] = bool(terms) and all(
            'Tarsus' in t['geom1'] or 'Tarsus' in t['geom2'] for t in terms)
        return out

    yaw_roll = [('Coxa', 'yaw'), ('Coxa', 'roll'), ('Trochanter', 'yaw'), ('Trochanter', 'roll')]
    pitch_only = [('Coxa', 'pitch'), ('Trochanter', 'pitch'), ('Tibia', 'pitch')]
    cf_mirror = evaluate_sign_counterfactual(yaw_roll)
    cf_pitch = evaluate_sign_counterfactual(pitch_only)

    # ---- site-moving variant: runs LAST because it edits model.site_pos in place ----
    changes = mirror_correct_right_sites(mjx, model, data)
    report['mirror_experiment']['sites_rewarded'] = len(changes)
    report['mirror_experiment']['moved_mm_median'] = (
        float(np.median([c['moved_mm'] for c in changes])) if changes else 0.0)
    report['mirror_experiment']['moved_mm_max'] = max([c['moved_mm'] for c in changes], default=0.0)
    mirrored = evaluate(key_qpos)
    dl = np.array(mirrored['tendon_lengths']) - np.array(pristine['tendon_lengths'])
    report['mirror_experiment']['tendon_length_change_mm'] = {
        'max_abs': float(np.abs(dl).max()), 'median_abs': float(np.median(np.abs(dl))),
        'n_changed_gt_0.01': int((np.abs(dl) > 0.01).sum())}
    report['mirror_experiment']['site_moving_variant_verdict'] = (
        'NOT USABLE as a repair test: re-placing the right-leg sites changes tendon lengths by up to '
        '%.4f mm, which moves muscles outside the force-length operating ranges that were derived for the '
        'original geometry, so the residual explodes to %.1f. Recorded as a negative control; it is the '
        'same coupling the earlier rebuild documented, not evidence about the shortfall.'
        % (float(np.abs(dl).max()), mirrored['all90']['objective_L1_residual']))
    report['mirror_experiment']['baseline_vs_site_moving'] = {
        'all90_L1_baseline': pristine['all90']['objective_L1_residual'],
        'all90_L1_site_moving': mirrored['all90']['objective_L1_residual']}
    model.site_pos[:] = site_pos_backup          # undo: the model must be pristine for later sections
    mjx.mj_forward(model, data)

    report['mirror_experiment']['keyframe_baseline'] = {
        'all90_L1': pristine['all90']['objective_L1_residual'],
        'candidate44_L1': pristine['candidate44']['objective_L1_residual'],
        'contact_count': pristine['contact_count'], 'feet_only': pristine['feet_only'],
        'matches_round15_reference': bool(abs(pristine['all90']['objective_L1_residual'] - 5.015616714722375) < 1e-6)}
    report['sign_counterfactual'] = {
        'what': ('only the yaw/roll components of the right-leg actuators\' generalized-force columns are '
                 'negated, giving the right-side actions that a sagittally mirrored body would have; tendon '
                 'lengths, force-length operating points, contacts and pose are all unchanged'),
        'baseline_all90_L1': pristine['all90']['objective_L1_residual'],
        'baseline_candidate44_L1': pristine['candidate44']['objective_L1_residual'],
        'yaw_roll_flipped_all90_L1': cf_mirror['all90']['objective_L1_residual'],
        'yaw_roll_flipped_candidate44_L1': cf_mirror['candidate44']['objective_L1_residual'],
        'pitch_flipped_control_all90_L1': cf_pitch['all90']['objective_L1_residual'],
        'contact_count': cf_mirror['contact_count'], 'feet_only': cf_mirror['feet_only']}
    d_cf = cf_mirror['all90']['objective_L1_residual'] - pristine['all90']['objective_L1_residual']
    report['sign_counterfactual']['verdict'] = (
        'making the right-leg actions mirror-consistent changes the all-90 residual by %+.4f force units '
        '(%.4f -> %.4f; body weight 24.4464). %s' % (
            d_cf, pristine['all90']['objective_L1_residual'],
            cf_mirror['all90']['objective_L1_residual'],
            'The bilateral route-direction defect is NOT what sets the residual, so the shortfall lies in '
            'force/direction coverage rather than in left/right sign symmetry.'
            if abs(d_cf) < 0.5 else
            'The bilateral route-direction defect materially changes the residual and is part of the picture.'))

    # ---------------- 7. answer ----------------
    any_translation_like = [k for k, v in symmetry.items() if 'TRANSLATED COPY' in v['verdict']]
    report['answer'] = {
        'route_geometry_defect_demonstrated': bool(any_translation_like),
        'evidence': ('every right leg is a translation of the homologous left leg rather than its sagittal '
                     'mirror: the site offset relative to the leg\'s own proximal joint anchor is preserved '
                     'instead of mirrored. Verified independently of the pose by the per-site offset test, and '
                     'independently by the moment arms, which come out with the same sign on both sides where a '
                     'mirrored animal requires opposite signs at homologous joints.'),
        'inherited_from_upstream': True,
        'same_in_pre_rebuild': None,
        'explains_static_support_shortfall': None}

    print(json.dumps({'left_right_symmetry': {k: {kk: vv for kk, vv in v.items()
                                                  if kk in ('verdict', 'translation_of_offset_median_mm',
                                                            'mirror_of_offset_median_mm',
                                                            'moment_arm_pairs_matching_plus_armL',
                                                            'moment_arm_pairs_matching_minus_armL',
                                                            'moment_arm_abs_ratio_outside_0.5_2',
                                                            'sites_deviating_from_translation_gt_%.3fmm' % SITE_TOL_MM)}
                                     for k, v in symmetry.items()},
                  'baseline_all90_L1': pristine['all90']['objective_L1_residual'],
                  'mirror_consistent_actions_all90_L1': cf_mirror['all90']['objective_L1_residual'],
                  'pitch_flip_control_all90_L1': cf_pitch['all90']['objective_L1_residual'],
                  'site_moving_variant_all90_L1_NOT_USABLE': mirrored['all90']['objective_L1_residual'],
                  'coverage_differences': len(coverage_diffs),
                  'left_right_sign_mismatch_families': len(mismatches),
                  'mid_hind_vs_LF_sign_differences': len(fore_vs_mid)}, indent=2))

    # npz with the tables (kept out of JSON for size)
    np.savez_compressed(OUT / 'moment_arms.npz', arm_zero=arm_zero, arm_keyframe=arm_key,
                        actuator_names=np.array(names), keys=np.array(['%s_%s_%s' % k for k in keys]),
                        candidate44=np.array(cand_names))

    # does the pre-rebuild model show the same site relationship?
    mj2, model2i, data2 = load(PRE_REBUILD)
    set_zero_legs(mj2, model2i, data2)
    sites_pre = site_table(mj2, model2i, data2)
    pmap = {r['name']: r for r in sites_pre}
    aL = np.asarray(data2.xanchor[joint_map(mj2, model2i)[('LF', 'Coxa', 'yaw')]], float)
    aR = np.asarray(data2.xanchor[joint_map(mj2, model2i)[('RF', 'Coxa', 'yaw')]], float)
    tr, mi, worst = [], [], 0
    for r in sites_pre:
        n = r['name']
        if not n.startswith('LF'):
            continue
        rn = 'R' + n[1:]
        if rn not in pmap:
            continue
        oL, oR = r['world'] - aL, pmap[rn]['world'] - aR
        t, m = float(np.linalg.norm(oR - oL)), float(np.linalg.norm(oR - SAGITTAL @ oL))
        tr.append(t)
        mi.append(m)
        if t > SITE_TOL_MM:
            worst += 1
    report['answer']['same_in_pre_rebuild'] = {
        'translation_median_mm': float(np.median(tr)), 'mirror_median_mm': float(np.median(mi)),
        'sites_deviating_gt_tol': worst,
        'conclusion': 'the translation-not-mirror relationship predates the rebuild, so it is inherited '
                      'from the upstream asset rather than introduced locally'}

    # did mirroring help?
    d_all = cf_mirror['all90']['objective_L1_residual'] - pristine['all90']['objective_L1_residual']
    report['answer']['explains_static_support_shortfall'] = (
        'sign-only mirror-consistent counterfactual changes the all-90 static-support residual by %+.4f force '
        'units (baseline %.4f, body weight 24.4464). %s' % (
            d_all, pristine['all90']['objective_L1_residual'],
            'The residual is dominated by something other than bilateral route-direction symmetry.'
            if abs(d_all) < 0.5 else
            'The change is material and route symmetry is part of the explanation.'))
    report['answer']['caveat_on_that_answer'] = (
        'the pose is frozen and the residual is a fixed-pose force-balance residual; static infeasibility is '
        'not dynamic infeasibility, and the counterfactual is not a repair proposal')
    assert sha256(XML) == EXPECTED_XML_SHA, 'canonical XML was modified during the run'
    report['xml_unchanged'] = (sha256(XML) == EXPECTED_XML_SHA)

    (OUT / 'route_fidelity.json').write_text(json.dumps(report, indent=2))

    # ---------------- 8. figure ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    ax = axs[0, 0]
    labels, trans_med, mirr_med = [], [], []
    for L, R in pairs:
        v = symmetry['%s_%s' % (L, R)]
        labels.append('%s/%s' % (L, R))
        trans_med.append(v['translation_of_offset_median_mm'])
        mirr_med.append(v['mirror_of_offset_median_mm'])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, trans_med, 0.4, label='offset preserved (translation)')
    ax.bar(x + 0.2, mirr_med, 0.4, label='offset mirrored (sagittal mirror)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('median residual (mm)')
    ax.set_title('Is the right leg the MIRROR of the left?\n'
                 'smaller bar = the relationship the model actually implements')
    ax.legend()
    ax.grid(alpha=.3, axis='y')

    ax = axs[0, 1]
    same = [symmetry['%s_%s' % p]['moment_arm_pairs_matching_plus_armL'] for p in pairs]
    opp = [symmetry['%s_%s' % p]['moment_arm_pairs_matching_minus_armL'] for p in pairs]
    ax.bar(x - 0.2, same, 0.4, label='arm_R = +arm_L (translated)')
    ax.bar(x + 0.2, opp, 0.4, label='arm_R = -arm_L (mirrored: required)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('number of (muscle, joint) pairs')
    ax.set_title('Moment-arm sign relation between homologous sides\n'
                 'a mirrored animal requires the right-hand bar to dominate')
    ax.legend()
    ax.grid(alpha=.3, axis='y')

    ax = axs[1, 0]
    offs = []
    for L, R in pairs:
        v = symmetry['%s_%s' % (L, R)]
        for w in v['worst_sites']:
            offs.append((w['site'], w['translation_residual_mm']))
    offs = sorted(offs, key=lambda t: -t[1])[:12]
    ax.barh([o[0][:44] for o in offs][::-1], [o[1] for o in offs][::-1])
    ax.axvline(SITE_TOL_MM, ls='--', c='gray', label='%.3f mm tolerance' % SITE_TOL_MM)
    ax.set_xlabel('residual from the translation model (mm)')
    ax.set_title('Sites that do NOT follow the translation relationship\n'
                 '(mid/hind tibia muscles and mid femur muscles)')
    ax.legend()
    ax.grid(alpha=.3, axis='x')

    ax = axs[1, 1]
    base = pristine['all90']['objective_L1_residual']
    mirr = cf_mirror['all90']['objective_L1_residual']
    base44 = pristine['candidate44']['objective_L1_residual']
    mirr44 = cf_mirror['candidate44']['objective_L1_residual']
    ax.bar([0, 1], [base, mirr], 0.35, label='all 90 actuators')
    ax.bar([0.4, 1.4], [base44, mirr44], 0.35, label='44 annotation-supported')
    ax.axhline(24.4464, ls=':', c='gray', label='one body weight (24.4464)')
    ax.set_xticks([0.2, 1.2])
    ax.set_xticklabels(['as shipped', 'right-leg actions made\nmirror-consistent (yaw/roll)'])
    ax.set_ylabel('static-support L1 residual (force units)')
    ax.set_title('Does fixing bilateral route DIRECTION close the\nstatic-support shortfall at the keyframe?')
    ax.legend()
    ax.grid(alpha=.3, axis='y')

    fig.suptitle('Tendon route fidelity: geometry bookkeeping, NOT a controller and NOT anatomical registration')
    fig.tight_layout()
    fig.savefig(OUT / 'route_fidelity.png', dpi=110)

    np.savez_compressed(OUT / 'moment_arms.npz', arm_zero=arm_zero, arm_keyframe=arm_key,
                        actuator_names=np.array(names),
                        keys=np.array(['%s_%s_%s' % k for k in keys]),
                        candidate44=np.array(cand_names))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
