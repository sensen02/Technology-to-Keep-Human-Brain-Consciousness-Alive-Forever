"""Round18: derive a bilateral (sagittal-mirror) repair of the right-leg muscle paths
and measure whether it closes more of the static-support shortfall than the sign-only
counterfactual from round17.

WHAT IS BEING TESTED, AND WHY IT IS NOT ASSUMED.
    Round17 measured, convention-free, that each right leg in the canonical model is a
    TRANSLATION of its left homologue rather than its sagittal mirror (site offsets taken
    relative to the leg's own proximal anchor: LF/RF median 0.00072 mm under translation vs
    0.0888 mm under mirroring; LM/RM 1e-7 vs 0.0521; LH/RH 1e-9 vs 0.0599).  This file
    builds the mirror as an actual CANDIDATE MODEL VARIANT and measures the consequence.
    The mirror plane is DERIVED from the model's own homologous body pairs, not assumed to
    be y = 0, and the residual of that derivation is reported.

WHAT THIS DOES NOT ESTABLISH.
    That the fly stands, that the repair is anatomically correct, or that the residual is
    meaningful as biology.  The variant is a candidate physical correction derived from a
    documented axis-based argument.  It does NOT modify or replace the canonical asset, and
    a reduced residual is NOT stability, NOT controllability, NOT a controller and NOT
    autonomy.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))          # tools_repair_muscle_lengthrange lives at the repo root

import diag_static_support_feasibility as r15    # noqa: E402  validated LP helpers
import diag_posture_support_search as r16        # noqa: E402  pose facts / solve wrappers
import diag_tendon_route_fidelity as r17         # noqa: E402  moment arms, site tables

CANONICAL = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
CANONICAL_SHA = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'
VARIANT_DIR = ROOT / 'outputs/muscles_six_leg_standing_bilateral'
OUT = ROOT / 'outputs/bilateral_repair_round18'
SITES_ONLY = VARIANT_DIR / 'fruitfly_six_leg_standing_bilateral_sites_only.xml'
SITES_BANDS = VARIANT_DIR / 'fruitfly_six_leg_standing_bilateral_sites_bands.xml'
LEGS = ['LF', 'RF', 'LM', 'RM', 'LH', 'RH']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leg_token(name):
    m = re.match(r'^([LR])([FMH])', name)
    return m.group(0) if m else None


def homolog_name(name):
    t = leg_token(name)
    if t is None or not t.startswith('R'):
        return None
    return 'L' + name[1:]


# ----------------------------------------------------------------- 1. derive mirror
def derive_mirror(mj, model, data, thorax_id):
    """Derive the sagittal plane from homologous body pairs, in the Thorax frame.

    The Thorax frame is used because it is the body's own frame: expressing the plane there
    makes it pose-independent for every fixed (non-free) joint.  The reference configuration
    is the model's build configuration with all leg joints at zero.
    """
    mj.mj_forward(model, data)
    Xt = np.asarray(data.xmat[thorax_id], float).reshape(3, 3)
    xt = np.asarray(data.xpos[thorax_id], float)

    def to_thorax(pos_world):
        return Xt.T @ (np.asarray(pos_world, float) - xt)

    pairs = []
    for b in range(model.nbody):
        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or ''
        t = leg_token(bn)
        if t is None or not t.startswith('L'):
            continue
        # bn starts with 'L' here, so the counterpart is the same name with the leading
        # token flipped L -> R.  (homolog_name() maps R -> L and would return None for a
        # left body, and passing None into mj_name2id segfaults the C library.)
        hn = 'R' + bn[1:]
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, hn) < 0:
            continue
        hb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, hn)
        if hb < 0:
            continue
        pL = to_thorax(data.xpos[b])
        pR = to_thorax(data.xpos[hb])
        pairs.append({'left_body': bn, 'right_body': hn, 'left': pL, 'right': pR,
                      'mid': 0.5 * (pL + pR), 'diff': pR - pL})
    if len(pairs) < 6:
        raise RuntimeError('too few homologous body pairs to derive a mirror plane')

    D = np.array([p['diff'] for p in pairs])
    mean_dir = D.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)
    # independent direction estimate: principal axis of the difference cloud
    _, s, Vt = np.linalg.svd(D - D.mean(axis=0), full_matrices=False)
    svd_dir = Vt[0] / np.linalg.norm(Vt[0])
    if float(svd_dir @ mean_dir) < 0:
        svd_dir = -svd_dir
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(mean_dir @ svd_dir), -1, 1))))

    n = mean_dir
    mids = np.array([p['mid'] for p in pairs])
    c_scalar = float(np.mean(mids @ n))
    c_vec = c_scalar * n

    def reflect(p):
        return p - 2.0 * n * (float(n @ (p - c_vec)))

    res = []
    for p in pairs:
        res.append(float(np.linalg.norm(reflect(p['left']) - p['right'])))
    res = np.asarray(res)
    mirrored_assumption = np.asarray([float(np.linalg.norm(p['right'] - p['left'])) for p in pairs])
    return {
        'frame': 'Thorax body frame (pose-independent for the fixed joints)',
        'reference_configuration': 'fresh MjData: free joint at origin, all leg joints zero',
        'normal_in_thorax_frame': n.tolist(),
        'plane_offset_along_normal_mm': c_scalar,
        'svd_principal_direction': svd_dir.tolist(),
        'angle_between_mean_and_svd_direction_deg': angle_deg,
        'n_homologous_body_pairs': len(pairs),
        'translation_residual_mm': {'median': float(np.median(mirrored_assumption)),
                                    'max': float(mirrored_assumption.max())},
        'mirror_residual_mm': {'median': float(np.median(res)), 'max': float(res.max())},
        'derivation_note': ('The residual is how far the model is from left-right mirror '
                            'symmetry: a real mirror pair would give ~0 here, whereas a '
                            'translation copy gives the leg separation.'),
        'pairs': [{'left_body': p['left_body'], 'right_body': p['right_body'],
                   'mirror_residual_mm': float(r)} for p, r in zip(pairs, res)],
        '_normal': n, '_offset': c_scalar, '_c_vec': c_vec, '_reflect': reflect,
    }


# ------------------------------------------------------- 2. build the variant tree
def build_sites_only(mirror, mj, model, data, thorax_id):
    """Return (tree, change list): right-leg sites moved to the mirror of their left homologues."""
    mj.mj_forward(model, data)
    Xt = np.asarray(data.xmat[thorax_id], float).reshape(3, 3)
    xt = np.asarray(data.xpos[thorax_id], float)
    reflect = mirror['_reflect']

    def to_thorax(pos_world):
        return Xt.T @ (np.asarray(pos_world, float) - xt)

    sites = r17.site_table(mj, model, data)
    by_name = {s['name']: s for s in sites}
    body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or '' for b in range(model.nbody)]

    tree = ET.parse(str(CANONICAL))
    root = tree.getroot()
    xml_sites = {}
    for b in root.iter('body'):
        for s in b.findall('site'):
            xml_sites[(b.get('name'), s.get('name'))] = s

    changes = []
    for s in sites:
        nm = s['name']
        if not nm.startswith('R'):
            continue
        ln = homolog_name(nm)
        if ln is None or ln not in by_name:
            raise RuntimeError('no left homolog for ' + nm)
        left = by_name[ln]
        world_left = np.asarray(left['world'], float)
        world_new = Xt @ reflect(to_thorax(world_left)) + xt     # mirror in Thorax frame
        # express in THIS site's parent body frame at the same reference configuration
        b = s['body_id']
        R = np.asarray(data.xmat[b], float).reshape(3, 3)
        local_new = R.T @ (world_new - np.asarray(data.xpos[b], float))

        key = (s['body'], nm)
        if key not in xml_sites:
            raise RuntimeError('site not found in XML: %s / %s' % key)
        old = np.asarray([float(v) for v in (s['local'])], float)
        el = xml_sites[key]
        el.set('pos', ' '.join('%.9g' % v for v in local_new))
        changes.append({'site': nm, 'body': s['body'],
                        'left_homolog': ln, 'left_body': left['body'],
                        'parent_is_homolog_of_left_site_body':
                            ('L' + s['body'][1:]) == left['body'] if s['body'].startswith('R')
                            else s['body'] == left['body'],
                        'old_local': old.tolist(), 'new_local': local_new.tolist(),
                        'moved_mm': float(np.linalg.norm(local_new - old))})
    return tree, changes


def site_level_leg_test(mj, model, data, thorax_id, mirror, label):
    """Convention-free test of how a right leg's site offsets relate to its left homolog's.

    Offsets are taken RELATIVE TO THE LEG'S OWN coxa anchor, so a shared origin, a shared
    orientation convention and the animal's midline offset all cancel out of the comparison.
    Translation residual: ||o_R - o_L||.  Mirror residual: ||o_R - S o_L|| with S the linear
    part of the reflection about the derived plane.  A right leg that is a translation copy
    scores low on the first and high on the second; a mirrored leg the other way round.
    """
    mj.mj_forward(model, data)
    n = np.asarray(mirror['normal_in_thorax_frame'], float)
    S = np.eye(3) - 2.0 * np.outer(n, n)
    sites = r17.site_table(mj, model, data)
    by_name = {s['name']: s for s in sites}
    anchors = {}
    for leg in LEGS:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, leg + 'Coxa')
        anchors[leg] = np.asarray(data.xpos[bid], float) if bid >= 0 else None
    out = {}
    for leg in LEGS:
        if not leg.startswith('R'):
            continue
        lg = 'L' + leg[1:]
        if anchors[leg] is None or anchors[lg] is None:
            continue
        tr, mi = [], []
        for s in sites:
            nm = s['name']
            if leg_token(nm) != leg:
                continue
            ln = 'L' + nm[1:]
            if ln not in by_name:
                continue
            oL = np.asarray(by_name[ln]['world'], float) - anchors[lg]
            oR = np.asarray(s['world'], float) - anchors[leg]
            tr.append(float(np.linalg.norm(oR - oL)))
            mi.append(float(np.linalg.norm(oR - S @ oL)))
        out[leg] = {'n_site_pairs': len(tr),
                    'translation_residual_mm': {'median': float(np.median(tr)), 'max': float(np.max(tr))},
                    'mirror_residual_mm': {'median': float(np.median(mi)), 'max': float(np.max(mi))},
                    'verdict': ('mirror-like' if np.median(mi) < np.median(tr) else 'translation-like')}
    return {'configuration': label, 'per_leg': out}


def construction_residual(mj, model, data, thorax_id, mirror):
    """After the repair, how exactly does each moved right site sit at the mirrored left position?

    The reflection is defined in the Thorax frame, so the world positions are converted into
    that frame first; applying it to raw world coordinates would silently be wrong whenever the
    Thorax body is not at the origin with an identity orientation.
    """
    mj.mj_forward(model, data)
    Xt = np.asarray(data.xmat[thorax_id], float).reshape(3, 3)
    xt = np.asarray(data.xpos[thorax_id], float)
    sites = r17.site_table(mj, model, data)
    by_name = {s['name']: s for s in sites}
    worst = 0.0
    worst_site = None
    n_checked = 0
    for s in sites:
        nm = s['name']
        if not nm.startswith('R'):
            continue
        ln = 'L' + nm[1:]
        if ln not in by_name:
            continue
        wL = np.asarray(by_name[ln]['world'], float)
        target = Xt @ mirror['_reflect'](Xt.T @ (wL - xt)) + xt
        got = np.asarray(s['world'], float)
        r = float(np.linalg.norm(got - target))
        if r > worst:
            worst, worst_site = r, nm
        n_checked += 1
    return {'n_sites_checked': n_checked, 'max_residual_mm': worst, 'worst_site': worst_site,
            'note': 'expected to be at XML round-off level: the construction writes 9 significant '
                    'digits, so ~1e-8 mm is normal and anything larger means the check or the '
                    'construction used the wrong frame'}


def write_xml(tree, dest, provenance):
    dest.parent.mkdir(parents=True, exist_ok=True)
    txt = ET.tostring(tree.getroot(), encoding='unicode')
    txt = '<?xml version="1.0" encoding="utf-8"?>\n' + txt
    dest.write_text(txt)
    return dest


# ----------------------------------------------------------- 3. measurement helpers
def compile_model(path):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    import mujoco as mj
    model = _load_mjcf(str(path)).compile()
    model.actuator_ctrlrange[:, 0] = 0.0     # memory only: allow an exactly zero command
    data = mj.MjData(model)
    return mj, model, data


def candidate44_index(model):
    manifest = json.loads((ROOT / 'outputs/muscle_anatomy_resolution/resolution.json').read_text())
    import mujoco as mj
    names = [model.actuator(i).name or '' for i in range(model.nu)]
    index_of = {}
    for i, n in enumerate(names):
        index_of.setdefault(n, i)
    want = [r['actuator'] for r in manifest['actuators']
            if r['status'] == 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    return [index_of[w] for w in want if w in index_of]


def measure(model_path, label):
    """Static-support residual at the model's own keyframe, plus validation facts."""
    mj, model, data = compile_model(model_path)
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    tarsus = [g for g in range(model.ngeom)
              if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus}
    thorax = model.body('Thorax').id
    facts = r16.pose_facts(model, data, mj, tarsus_names, thorax)

    trn = np.asarray(model.actuator_trnid[:, 0], int)
    lg = np.asarray(data.actuator_length, float)
    lr = np.asarray(model.actuator_lengthrange, float)
    frac = (lg - lr[:, 0]) / np.maximum(lr[:, 1] - lr[:, 0], 1e-12)
    ten_ident = float(np.abs(lg - np.asarray(data.ten_length, float)[trn]).max())

    sets = {'all90': list(range(model.nu)), 'candidate44': candidate44_index(model)}
    sols = r16.solve_pose(model, data, mj, sets, label)
    valid = r16.validate_equilibrium(model, data, mj, label)
    return {
        'label': label,
        'dims': {'nq': int(model.nq), 'nv': int(model.nv), 'njnt': int(model.njnt),
                 'nu': int(model.nu), 'ntendon': int(model.ntendon)},
        'keyframe_thorax_height_mm': facts['thorax_height_mm'],
        'keyframe_contacts': facts['contact_count'],
        'keyframe_contacts_feet_only': facts['contacts_are_feet_only_tarsus'],
        'active_joint_limit_rows': facts['active_joint_limit_rows'],
        'qfrc_bias_absmax': facts['qfrc_bias_absmax'],
        'qfrc_passive_absmax': facts['qfrc_passive_absmax'],
        'actuator_length_equals_tendon_length_max_abs': ten_ident,
        'muscle_band_fraction_min': float(frac.min()),
        'muscle_band_fraction_max': float(frac.max()),
        'n_muscles_outside_band': int(((frac < 0) | (frac > 1)).sum()),
        'all90_L1': float(sols['all90']['objective_L1_residual']),
        'all90_max_abs': float(sols['all90']['max_abs_residual']),
        'all90_saturated_actuators': int((np.asarray(sols['all90']['activation']) > 1 - 1e-9).sum()),
        'candidate44_L1': float(sols['candidate44']['objective_L1_residual']),
        'equilibrium_validation': valid,
        '_model': model, '_data': data, '_mj': mj, '_factors': facts,
        '_lengths': lg.copy(), '_bands': lr.copy(), '_fracs': frac.copy(),
    }


def moment_arm_mirror_check(model, data, mj, label):
    """How many homologous muscle/joint pairs satisfy the axis-derived mirror relation."""
    jd = r17.joint_map(mj, model)
    arm, keys = r17.moment_arms(mj, model, data, jd)
    names = [model.actuator(i).name or '' for i in range(model.nu)]
    idx = {}
    for i, n in enumerate(names):
        idx.setdefault(n, i)
    # derived from the declared axes: mirroring negates the pseudo-vector, so a mirrored
    # animal needs arm_R = -arm_L on yaw/roll and +arm_L on pitch.
    sign_for = {'yaw': -1.0, 'roll': -1.0, 'pitch': +1.0}
    agree, disagree, skipped = 0, 0, 0
    detail = []
    for i, nm in enumerate(names):
        if not nm.startswith('R'):
            continue
        ln = 'L' + nm[1:]
        if ln not in idx:
            skipped += 1
            continue
        j = idx[ln]
        for k, (leg, part, axis) in enumerate(keys):
            if leg != 'RF' and leg != 'RM' and leg != 'RH':
                continue
            expected = sign_for[axis] * arm[j, keys.index((ln[:2], part, axis))]
            got = arm[i, k]
            if abs(expected) < 1e-9 or abs(got) < 1e-9:
                continue
            if np.sign(expected) == np.sign(got):
                agree += 1
            else:
                disagree += 1
                detail.append({'muscle': nm, 'joint': '%s%s_%s' % (leg, part, axis),
                               'arm_right': float(got), 'expected_from_left': float(expected)})
    return {'configuration': label, 'pairs_matching_axis_derived_mirror': agree,
            'pairs_violating': disagree, 'skipped_no_homolog': skipped,
            'violations_sample': detail[:12],
            '_arm': arm, '_keys': keys, '_names': names}


def coverage_summary(arm, keys, sel_idx, label):
    """Per-joint how many actuators can push each way: a joint with only one sign is one-way."""
    rows = []
    one_way_all = one_way_44 = 0
    for k, (leg, part, axis) in enumerate(keys):
        a_all = arm[:, k]
        a_sel = arm[sel_idx, k]
        pos_all = int((a_all > 1e-9).sum()); neg_all = int((a_all < -1e-9).sum())
        pos_s = int((a_sel > 1e-9).sum()); neg_s = int((a_sel < -1e-9).sum())
        if pos_all == 0 or neg_all == 0:
            one_way_all += 1
        if pos_s == 0 or neg_s == 0:
            one_way_44 += 1
        rows.append({'joint': '%s%s_%s' % (leg, part, axis),
                     'all90_pos': pos_all, 'all90_neg': neg_all,
                     'candidate44_pos': pos_s, 'candidate44_neg': neg_s})
    return {'label': label, 'one_way_joints_all90': one_way_all,
            'one_way_joints_candidate44': one_way_44,
            'n_joints': len(keys), 'per_joint': rows}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    sha_before = sha256(CANONICAL)
    if sha_before != CANONICAL_SHA:
        raise SystemExit('canonical XML hash changed before starting')

    mj, model, data = compile_model(CANONICAL)
    thorax = model.body('Thorax').id
    # reference configuration for all geometry work: fresh MjData (free joint at origin,
    # leg joints zero) -- the model's own build configuration.
    data_ref = mj.MjData(model)
    mirror = derive_mirror(mj, model, data_ref, thorax)

    report = {
        'status': 'BILATERAL_REPAIR_EXPERIMENT',
        'question': ('does a derived sagittal-mirror repair of the right-leg muscle paths close '
                     'more of the static-support shortfall than the round17 sign-only counterfactual?'),
        'scope': ('diagnostic; the variant is a CANDIDATE physical correction, is NOT anatomically '
                  'registered, and never replaces the canonical asset. A reduced residual is NOT '
                  'standing, NOT stability, NOT controllability, NOT autonomy.'),
        'canonical': {'path': str(CANONICAL), 'sha256_before': sha_before,
                      'expected_sha256': CANONICAL_SHA, 'unchanged_so_far': sha_before == CANONICAL_SHA},
        'mirror_derivation': {k: v for k, v in mirror.items() if not k.startswith('_')},
        'round17_reference': {
            'sign_only_counterfactual_all90_L1': 4.007590543171839,
            'canonical_all90_L1': 5.015616714722375,
            'source': 'outputs/tendon_fidelity_round17/route_fidelity.json (read, not recomputed)',
            'negative_control_reported_by_round17': ('round17 reported that moving right sites to '
                                                     'mirrored positions without re-deriving bands '
                                                     'blew the residual up to ~3585. THIS script '
                                                     'measures 22.81 for its own sites-only variant: '
                                                     'the same idea but with the derived plane offset '
                                                     'and a parent-frame conversion. The two numbers '
                                                     'differ by two orders of magnitude and the cause '
                                                     'was NOT identified here; the magnitude of the '
                                                     'sites-only negative control is therefore not '
                                                     'settled, only its direction (bands must be '
                                                     're-derived).')},
        'sources_sha256': {str(p): sha256(p) for p in
                           (CANONICAL, HERE / 'diag_static_support_feasibility.py',
                            HERE / 'diag_posture_support_search.py',
                            HERE / 'diag_tendon_route_fidelity.py',
                            ROOT / 'tools_repair_muscle_lengthrange.py')},
        'limitations': [
            'The mirror plane is derived from homologous body positions; the derivation residual '
            'is reported and is not zero, because the model is not mirror-symmetric to begin with.',
            'The right-leg bodies are NOT moved, only the tendon sites, so the right leg keeps the '
            'skeleton the canonical model already has.',
            'Friction cone linearised; actuator force affine in activation at the frozen pose; '
            'force-length/velocity and moment arms frozen at the measured pose.',
            'Static feasibility is NOT stability, NOT controllability, NOT a controller, NOT standing.',
            'Coordinate signs are model coordinates, not anatomical flexion labels.'],
    }

    # ---- canonical baseline ------------------------------------------------
    canon = measure(CANONICAL, 'canonical')
    report['canonical_measurement'] = {k: v for k, v in canon.items() if not k.startswith('_')}
    canon_mirror = moment_arm_mirror_check(canon['_model'], canon['_data'], canon['_mj'],
                                           'canonical zero-leg configuration')
    _c44 = candidate44_index(canon['_model'])
    report['moment_arm_mirror_check'] = {'canonical': {k: v for k, v in canon_mirror.items()
                                                      if not k.startswith('_')}}
    report['joint_sign_coverage'] = {'canonical': coverage_summary(
        canon_mirror['_arm'], canon_mirror['_keys'], _c44,
        'canonical zero-leg configuration')}

    # ---- variant A: mirrored sites only (the negative control) --------------
    tree, changes = build_sites_only(mirror, mj, model, data_ref, thorax)
    moved = np.array([c['moved_mm'] for c in changes])
    report['site_level_leg_test_canonical'] = site_level_leg_test(
        mj, model, data_ref, thorax, mirror, 'canonical, reference configuration')
    report['site_changes'] = {
        'n_sites_moved': len(changes),
        'n_on_shared_thorax_body': int(sum(1 for c in changes if c['body'] == 'Thorax')),
        'n_on_right_leg_bodies': int(sum(1 for c in changes if c['body'] != 'Thorax')),
        'all_parent_bodies_are_homologs_of_left_site_bodies':
            bool(all(c['parent_is_homolog_of_left_site_body'] for c in changes)),
        'moved_mm': {'min': float(moved.min()), 'median': float(np.median(moved)),
                     'max': float(moved.max())},
        'changes': changes}
    write_xml(tree, SITES_ONLY, 'round18 bilateral repair: right-leg sites mirrored, bands untouched')
    sites_only = measure(SITES_ONLY, 'sites_only')
    report['sites_only_measurement'] = {k: v for k, v in sites_only.items() if not k.startswith('_')}

    # ---- variant B: mirrored sites + re-derived operating bands -------------
    from tools_repair_muscle_lengthrange import plan as plan_bands, apply_to_tree, verify_plan
    mj_b, model_b, data_b = compile_model(SITES_ONLY)
    bands, brep = plan_bands(model_b, data_b, 'keyframe')
    ok, worst = verify_plan(brep)
    tree_b = ET.parse(str(SITES_ONLY))
    n_bands = apply_to_tree(tree_b.getroot(), bands)
    write_xml(tree_b, SITES_BANDS,
              'round18 bilateral repair: mirrored right-leg sites + re-derived force-length bands')
    sites_bands = measure(SITES_BANDS, 'sites_plus_bands')
    _mj_v, _model_v, _data_v = compile_model(SITES_BANDS)
    report['construction_residual'] = construction_residual(_mj_v, _model_v, _data_v, thorax, mirror)
    report['site_level_leg_test_repaired'] = site_level_leg_test(
        _mj_v, _model_v, _data_v, thorax, mirror, 'repaired variant, reference configuration')
    report['band_rederivation'] = {
        'tool': 'tools_repair_muscle_lengthrange.plan/apply_to_tree (reused, unmodified)',
        'posture': 'keyframe',
        'n_bands_written': int(n_bands),
        'n_scaled': int(brep['n_scaled']),
        'n_skipped': len(brep.get('skipped', [])),
        'invariance_ok': bool(ok),
        'invariance_worst_abs_frac_minus_frac_fore': float(worst),
        'scaled_sample': brep['scaled'][:6]}
    report['sites_plus_bands_measurement'] = {k: v for k, v in sites_bands.items()
                                              if not k.startswith('_')}

    # ---- tendon length and band change accounting --------------------------
    dL = sites_bands['_lengths'] - canon['_lengths']
    names = [canon['_model'].actuator(i).name or '' for i in range(canon['_model'].nu)]
    changed = int((np.abs(dL) > 1e-9).sum())
    report['tendon_and_band_changes'] = {
        'n_muscles_tendon_length_changed': changed,
        'tendon_length_change_mm': {'min': float(dL.min()), 'median': float(np.median(dL)),
                                    'max': float(dL.max()),
                                    'max_abs': float(np.abs(dL).max())},
        'n_muscles_band_changed': int((np.abs(sites_bands['_bands'] - canon['_bands']) > 1e-12).any(axis=1).sum()),
        'per_muscle': [{'muscle': names[i], 'tendon_len_before': float(canon['_lengths'][i]),
                        'tendon_len_after': float(sites_bands['_lengths'][i]),
                        'delta_mm': float(dL[i]),
                        'band_before': canon['_bands'][i].tolist(),
                        'band_after': sites_bands['_bands'][i].tolist(),
                        'frac_before': float(canon['_fracs'][i]),
                        'frac_after': float(sites_bands['_fracs'][i])}
                       for i in range(len(names)) if abs(dL[i]) > 1e-9][:60]}

    # ---- moment arms after the repair (zero-leg configuration) -------------
    _, model_c, data_c = compile_model(SITES_BANDS)
    mj.mj_forward(model_c, data_c)          # fresh data == zero-leg reference configuration
    arm_after = moment_arm_mirror_check(model_c, data_c, mj,
                                        'repaired variant zero-leg configuration')
    report['moment_arm_mirror_check']['repaired_variant'] = {k: v for k, v in arm_after.items()
                                                             if not k.startswith('_')}
    _c44b = candidate44_index(model_c)
    report['joint_sign_coverage']['repaired_variant'] = coverage_summary(
        arm_after['_arm'], arm_after['_keys'], _c44b,
        'repaired variant zero-leg configuration')

    # ---- verdict -----------------------------------------------------------
    report['verdict'] = {
        'canonical_all90_L1': canon['all90_L1'],
        'sites_only_all90_L1': sites_only['all90_L1'],
        'sites_plus_bands_all90_L1': sites_bands['all90_L1'],
        'round17_sign_only_all90_L1': 4.007590543171839,
        'canonical_candidate44_L1': canon['candidate44_L1'],
        'sites_plus_bands_candidate44_L1': sites_bands['candidate44_L1'],
        'reduced_vs_canonical': bool(sites_bands['all90_L1'] < canon['all90_L1']),
        'beats_sign_only_counterfactual': bool(sites_bands['all90_L1']
                                               < 4.007590543171839),
        'feasible_within_tolerance': bool(sites_bands['all90_L1'] < r15.FEASIBLE_SUM_TOL),
    }
    report['execution'] = {'wall_seconds': float(time.time() - t0), 'lp_caps': {
        'time_limit_s': r15.LP_TIME_LIMIT_S, 'maxiter': r15.LP_MAXITER,
        'cone_directions': r15.CONE_DIRECTIONS}, 'mujoco': mj.__version__}

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    labels = ['canonical', 'sites only\n(negative control)', 'sites + bands\n(repair)']
    all90 = [canon['all90_L1'], sites_only['all90_L1'], sites_bands['all90_L1']]
    c44 = [canon['candidate44_L1'], sites_only['candidate44_L1'], sites_bands['candidate44_L1']]
    x = np.arange(3)
    axs[0].bar(x - 0.2, all90, 0.4, label='all 90 actuators')
    axs[0].bar(x + 0.2, c44, 0.4, label='44 annotation-supported')
    axs[0].axhline(4.007590543171839, ls=':', c='red',
                   label='round17 sign-only counterfactual (4.008)')
    axs[0].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
    axs[0].set_yscale('symlog', linthresh=1e-3)
    axs[0].set_xticks(x); axs[0].set_xticklabels(labels, fontsize=8)
    axs[0].set_ylabel('static-support LP L1 residual (force units)')
    axs[0].set_title('Keyframe static-support residual\nbody weight scale = 24.446')
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.3, axis='y')

    axs[1].hist(np.abs(dL), bins=40, color='tab:blue')
    axs[1].set_xlabel('|tendon length change| (mm)')
    axs[1].set_ylabel('muscles')
    axs[1].set_title('Tendon length change from mirroring\n%d of %d muscles changed'
                     % (changed, len(names)))
    axs[1].grid(alpha=.3)

    ca = report['moment_arm_mirror_check']['canonical']
    cb = report['moment_arm_mirror_check']['repaired_variant']
    axs[2].bar([0, 1], [ca['pairs_matching_axis_derived_mirror'], cb['pairs_matching_axis_derived_mirror']],
               0.35, label='match axis-derived mirror')
    axs[2].bar([0.4, 1.4], [ca['pairs_violating'], cb['pairs_violating']], 0.35,
               color='tab:red', label='violate')
    axs[2].set_xticks([0.2, 1.2]); axs[2].set_xticklabels(['canonical', 'repaired'])
    axs[2].set_ylabel('homologous muscle/joint pairs')
    axs[2].set_title('Do right-leg moment arms obey the\naxis-derived mirror relation?')
    axs[2].legend(fontsize=7); axs[2].grid(alpha=.3, axis='y')
    fig.suptitle('Round18 bilateral repair: static force balance only\n'
                 'candidate correction, NOT standing, NOT stability, NOT autonomy')
    fig.tight_layout()
    fig.savefig(OUT / 'bilateral_repair.png', dpi=110)

    report['canonical']['sha256_after'] = sha256(CANONICAL)
    report['canonical']['unchanged'] = report['canonical']['sha256_after'] == CANONICAL_SHA
    report['variant_files'] = {'sites_only': str(SITES_ONLY), 'sites_plus_bands': str(SITES_BANDS),
                               'sites_only_sha256': sha256(SITES_ONLY),
                               'sites_plus_bands_sha256': sha256(SITES_BANDS)}
    (OUT / 'bilateral_repair.json').write_text(json.dumps(report, indent=2))

    print(json.dumps({'mirror_normal': mirror['normal_in_thorax_frame'],
                      'mirror_offset_mm': mirror['plane_offset_along_normal_mm'],
                      'mirror_residual_median_mm': mirror['mirror_residual_mm']['median'],
                      'translation_residual_median_mm': mirror['translation_residual_mm']['median'],
                      'sites_moved': len(changes), 'max_move_mm': float(moved.max()),
                      'canonical_all90_L1': canon['all90_L1'],
                      'canonical_candidate44_L1': canon['candidate44_L1'],
                      'sites_only_all90_L1': sites_only['all90_L1'],
                      'sites_plus_bands_all90_L1': sites_bands['all90_L1'],
                      'sites_plus_bands_candidate44_L1': sites_bands['candidate44_L1'],
                      'tendon_len_changed': changed,
                      'n_muscles_outside_band_after': sites_bands['n_muscles_outside_band'],
                      'mirror_pairs_canonical': [ca['pairs_matching_axis_derived_mirror'],
                                                 ca['pairs_violating']],
                      'mirror_pairs_repaired': [cb['pairs_matching_axis_derived_mirror'],
                                                cb['pairs_violating']],
                      'one_way_joints_all90_canonical_vs_repaired': [
                          report['joint_sign_coverage']['canonical']['one_way_joints_all90'],
                          report['joint_sign_coverage']['repaired_variant']['one_way_joints_all90']],
                      'one_way_joints_candidate44_canonical_vs_repaired': [
                          report['joint_sign_coverage']['canonical']['one_way_joints_candidate44'],
                          report['joint_sign_coverage']['repaired_variant']['one_way_joints_candidate44']],
                      'construction_residual_max_mm': report['construction_residual']['max_residual_mm'],
                      'canonical_unchanged': report['canonical']['unchanged'],
                      'wall_seconds': report['execution']['wall_seconds']}, indent=2))
    print(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
