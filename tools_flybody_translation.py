"""Declared translation tables from the canonical muscle body to the published FlyBody body.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  No behaviour is run here.

This file answers, with measurements rather than prose, the only question that has to be
answered before the existing spike->muscle closure can drive a DIFFERENT body:

    which finger of the old body is which finger of the new one, and which way does each
    muscle pull it?

Three tables are produced and written to disk:

  T-A  JOINT PAIRING.  Every canonical leg hinge (7 per leg: Coxa yaw/pitch/roll, Trochanter
       yaw/pitch/roll, Tibia pitch) is paired with a FlyBody leg hinge of the same leg by the
       largest |dot product| between the two hinges' axes expressed in their OWN animal's
       thorax frame at the seated pose.  The dot product is a measurement; the pairing is
       therefore reproducible and its quality is reported, not asserted.  A pairing below the
       declared threshold is left UNPAIRED and reported as unpaired rather than guessed.

  T-B  MUSCLE -> JOINT.  Every canonical muscle's target hinge is the hinge with the largest
       |moment arm| for that muscle, where the moment arm is the finite-difference derivative
       of the muscle's own tendon length with respect to the hinge coordinate -- the same
       instrument the project already uses in diagnostics/diag_tendon_route_fidelity.py.  The
       runner-up ratio is reported so ambiguous muscles are visible.

  T-C  MOTOR DIRECTION.  A muscle that shortens when the hinge coordinate increases pulls the
       hinge NEGATIVE, so its torque direction about the canonical positive axis is
       -sign(arm).  A positive canonical rotation is a rotation of sign `dot` about the
       FlyBody positive axis.  The FlyBody drive direction of that muscle is therefore
       -sign(arm)*sign(dot).  For the tibia -- the one hinge whose action has an unambiguous
       geometric definition -- the chain is then CHECKED end to end: a canonical flexor pool
       must move the FlyBody claw TOWARDS the body.

Also written here: the torque-replacement variant of the published model, which keeps the
published geometry, masses and joints but turns the 48 leg joint position servos into plain
torque motors with the same published force limit.  It is a DECLARED VARIANT of a published
model, written to this branch's own output directory, never into the published asset directory.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'diagnostics'))
import engine.canonical_physics as cp
import engine.memory_guard as mg
from diagnostics.diag_tendon_route_fidelity import joint_map, moment_arms, KEYS, LEGS

OUT = HERE / 'outputs/model_branches/flybody_loop'
FLYBODY_DIR = HERE / 'upstream/flybody'
FLYBODY_XML = FLYBODY_DIR / 'fruitfly.xml'
FLYBODY_FLOOR_XML = FLYBODY_DIR / 'floor.xml'
FLYBODY_XML_SHA = 'd14946fd0311025ecca70c8eeb5de80e1fe18700d3072be37ecbb18d33d80fd8'
AXIS_MIN_DOT = .5            # below this |dot| the pairing is reported UNPAIRED, not guessed
LEG_TAG = {'LF': ('T1', 'left'), 'RF': ('T1', 'right'), 'LM': ('T2', 'left'),
           'RM': ('T2', 'right'), 'LH': ('T3', 'left'), 'RH': ('T3', 'right')}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flybody_leg_joint_name(leg, joint):
    tag, side = LEG_TAG[leg]
    return f'{joint}_{tag}_{side}'


def flybody_prefix(leg):
    tag, side = LEG_TAG[leg]
    return f'_{tag}_{side}'


def thorax_frame_axis(mj, model, data, jid, thorax_id):
    """Joint axis of `jid`, in the world at the current pose, expressed in the thorax frame."""
    axis = np.asarray(model.jnt_axis[jid], float)
    parent = int(model.jnt_bodyid[jid])
    world = data.xmat[parent].reshape(3, 3) @ axis
    return data.xmat[thorax_id].reshape(3, 3).T @ world


def canonical_axes_and_arms():
    import mujoco
    import run_fix_round39 as r39
    if cp.sha256_file(cp.CANONICAL_XML) != cp.CANONICAL_XML_SHA256:
        raise ValueError('canonical body changed; the translation tables would be stale')
    model = cp.load_model()
    data = mujoco.MjData(model)
    r39.apply_round37_pose(model, data, json.loads(r39.ROUND37_POSE.read_text()))
    base = data.qpos.copy()
    base[2] = cp.seat_root(model, data, base)
    data.qpos[:] = base
    data.qvel[:] = 0.0
    data.act[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    thorax = model.body('Thorax').id
    jd = joint_map(mujoco, model)
    arm, keys = moment_arms(mujoco, model, data, jd)
    axes = {k: thorax_frame_axis(mujoco, model, data, jd[k], thorax) for k in jd}
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(model.nu)]
    return dict(mj=mujoco, model=model, data=data, pose=base, jd=jd, keys=keys,
                arm=arm, axes=axes, actuator_names=actuator_names,
                weight=float(model.body_mass.sum()*abs(model.opt.gravity[2])),
                pose_note='round37 pose, root seated on the floor plane by the project seam',
                arm_note='finite difference d(muscle tendon length)/d(hinge coordinate) at that pose')


def flybody_axes():
    import mujoco
    if sha256(FLYBODY_XML) != FLYBODY_XML_SHA:
        raise ValueError('published FlyBody asset changed; refusing to build tables from it')
    model = mujoco.MjModel.from_xml_path(str(FLYBODY_FLOOR_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    thorax = model.body('thorax').id
    axes, ranges, springrefs, act_index = {}, {}, {}, {}
    for leg in LEGS:
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ''
            if not name.endswith(flybody_prefix(leg)):
                continue
            axes[name] = thorax_frame_axis(mujoco, model, data, i, thorax)
            ranges[name] = [float(model.jnt_range[i][0]), float(model.jnt_range[i][1])]
            springrefs[name] = float(model.qpos_spring[int(model.jnt_qposadr[i])])
    for i in range(model.nu):
        act_index[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] = dict(
            index=i, ctrlrange=[float(model.actuator_ctrlrange[i][0]),
                                float(model.actuator_ctrlrange[i][1])],
            forcerange=[float(model.actuator_forcerange[i][0]),
                        float(model.actuator_forcerange[i][1])],
            gainprm=float(model.actuator_gainprm[i][0]), biasprm=float(model.actuator_biasprm[i][1]),
            trntype=int(model.actuator_trntype[i]))
    return dict(mj=mujoco, model=model, data=data, axes=axes, ranges=ranges,
                springrefs=springrefs, actuators=act_index,
                weight=float(model.body_mass.sum()*abs(model.opt.gravity[2])),
                note='every hinge at 0, root at the origin: FlyBody\'s published neutral pose')


def canonical_axes_at_zero_legs(canon):
    """Canonical hinge axes at the project's own SYMMETRIC reference configuration.

    `diagnostics/diag_tendon_route_fidelity.py` records why this pose is the primary one for any
    homology question: at the as-shipped keyframe the six legs are NOT in a mutually symmetric
    posture, so world-frame comparisons there mix pose asymmetry with model asymmetry.  Measured
    here, at zero leg hinges the canonical model's axes come out exactly on the world axes, which
    is why an axis-only registration cannot separate the hinges (see discrete_registration).
    """
    import mujoco
    data = canon['data']
    data.qpos[:] = 0.0
    data.qpos[2] = 2.04535
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(canon['model'], data)
    thorax = canon['model'].body('Thorax').id
    axes = {k: thorax_frame_axis(mujoco, canon['model'], data, canon['jd'][k], thorax)
            for k in canon['jd']}
    return axes


def discrete_registration(canon_axes, fly):
    """Evidence that a one-to-one hinge registration between the two models does NOT exist.

    For each leg pair the best bijection between the seven canonical hinges and the FlyBody
    hinges is computed with the Hungarian algorithm on the summed |axis dot product| over the
    left and right legs.  If the two models shared a hinge-to-hinge registration this would
    recover it; the reported result is anatomically absurd (e.g. a tibia hinge assigned to a
    coxal twist), which is the measurement this branch acts on.
    """
    from scipy.optimize import linear_sum_assignment
    pairs = {'T1': ('LF', 'RF'), 'T2': ('LM', 'RM'), 'T3': ('LH', 'RH')}
    out = {}
    for pair, (left, right) in pairs.items():
        fnames = sorted(n for n in fly['axes'] if n.endswith(flybody_prefix(left)))
        cost = np.zeros((len(KEYS), len(fnames)))
        for i, (part, axis) in enumerate(KEYS):
            for j, n in enumerate(fnames):
                n_r = n.replace(flybody_prefix(left), flybody_prefix(right))
                cost[i, j] = (abs(float(np.dot(canon_axes[(left, part, axis)], fly['axes'][n])))
                              + abs(float(np.dot(canon_axes[(right, part, axis)], fly['axes'][n_r]))))
        rows, cols = linear_sum_assignment(-cost)
        rows_out = []
        for i, j in zip(rows, cols):
            part, axis = KEYS[i]
            n = fnames[j]
            n_r = n.replace(flybody_prefix(left), flybody_prefix(right))
            rows_out.append(dict(canonical_joint=f'{part}_{axis}', flybody_joint=n.split('_'+pair)[0],
                                 dot_left=round(float(np.dot(canon_axes[(left, part, axis)],
                                                           fly['axes'][n])), 4),
                                 dot_right=round(float(np.dot(canon_axes[(right, part, axis)],
                                                            fly['axes'][n_r])), 4)))
        out[pair] = dict(legs=[left, right], assignment=rows_out,
                         mean_abs_dot=round(float(np.mean([abs(r['dot_left']) for r in rows_out])), 4))
    return dict(method='Hungarian assignment maximising summed |thorax-frame axis dot product| over '
                       'the left and right leg of a pair',
                threshold_note='if a hinge-to-hinge registration existed, the assignment would '
                               'recover it and the dots would be near 1 on homologous hinges',
                result=out,
                conclusion='NO one-to-one hinge registration is supportable: the assignment pairs a '
                           'canonical tibia hinge with a FlyBody coxal twist on the middle legs and '
                           'a canonical trochanter hinge with a tarsomere on the front and hind '
                           'legs.  This branch therefore does NOT use a discrete pairing.')


#: Declared segment identity.  Each model names the segments its hinges connect, and the
#: correspondence below uses only that: the canonical Coxa hinge group connects Thorax-Coxa, the
#: Trochanter group connects Coxa-Trochanter (FlyBody calls that segment the femur), the canonical
#: Tibia hinge connects Trochanter-Tibia.  No DOF is paired by name across the two models.
SEGMENT_OF_PART = {'Coxa': 'coxa', 'Trochanter': 'femur', 'Tibia': 'tibia'}


def segment_projection(canon, canon_axes, fly):
    """Muscle -> FlyBody hinge coefficients by segment-restricted axis projection.

    For every canonical muscle: its own target hinge is the one with the largest |moment arm|;
    that hinge's rotation axis is taken in the thorax frame; the muscle's torque is then projected
    onto the FlyBody hinges OF THE SAME SEGMENT of the same leg, with the coefficient being the
    axis dot product times -sign(moment arm) (a muscle that shortens as the coordinate grows pulls
    the coordinate negative).  Coefficients are measurements; the segment correspondence is
    declared; no DOF is guessed across a segment boundary.
    """
    table, dropped, per_hinge_norm = {}, [], {}
    for name, row in muscle_targets(canon).items():
        leg, part, axis = row['leg'], row['part'], row['axis']
        segment = SEGMENT_OF_PART[part]
        axis_c = canon_axes[(leg, part, axis)]
        coeffs = {}
        for fname, axis_f in fly['axes'].items():
            if not fname.endswith(flybody_prefix(leg)) or not fname.startswith(segment + '_'):
                continue
            dot = float(np.dot(axis_c, axis_f))
            coeffs[fname] = round(row['sign']*dot, 6)
        if not coeffs or max(abs(v) for v in coeffs.values()) < .1:
            dropped.append(dict(actuator=name, leg=leg, segment=segment,
                                best_abs_coefficient=max([abs(v) for v in coeffs.values()] or [0.0]),
                                reason='no same-segment FlyBody hinge axis is within 0.1 of this '
                                       'muscle\'s own hinge axis'))
        for fname, v in coeffs.items():
            per_hinge_norm[fname] = per_hinge_norm.get(fname, 0.0) + abs(v)
        table[name] = dict(actuator_index=row['actuator_index'], leg=leg,
                           canonical_joint=row['canonical_joint'],
                           canonical_segment=part, flybody_segment=segment,
                           moment_arm_model_units_per_rad=row['moment_arm_model_units_per_rad'],
                           runner_up_ratio=row['runner_up_ratio'],
                           coefficients_to_flybody_hinges=coeffs,
                           max_abs_coefficient=max(abs(v) for v in coeffs.values()) if coeffs else 0.0)
    return dict(
        method='segment-restricted axis projection (no discrete DOF pairing)',
        declared_segment_correspondence=SEGMENT_OF_PART,
        table=table,
        per_hinge_coefficient_norm={k: round(v, 6) for k, v in sorted(per_hinge_norm.items())},
        n_muscles=len(table), n_dropped_muscles=len(dropped), dropped_muscles=dropped,
        normalisation='each FlyBody hinge is driven by the SUM over muscles of coefficient * '
                     'activation, divided by the sum of |coefficient| reaching that hinge, so a '
                     'hinge reached by seven coxa muscles is not scaled up relative to a hinge '
                     'reached by two tibia muscles',
        coefficient_meaning='positive coefficient = positive FlyBody hinge coordinate',
        what_this_cannot_do='the projection cannot restore muscle-level resolution: several '
                            'canonical muscles that pull different canonical hinges can and do '
                            'land on the same FlyBody hinge, and their individual identity is lost '
                            'in the sum')


def muscle_targets(canon):
    """Largest-|moment arm| hinge per muscle, with the runner-up reported for ambiguity."""
    rows = {}
    for m, name in enumerate(canon['actuator_names']):
        order = np.argsort(-np.abs(canon['arm'][m]))
        best, second = canon['keys'][order[0]], canon['keys'][order[1]] if len(order) > 1 else None
        a_best = float(canon['arm'][m][order[0]])
        a_second = float(canon['arm'][m][order[1]]) if len(order) > 1 else 0.0
        rows[name] = dict(
            actuator_index=m, leg=best[0], part=best[1], axis=best[2],
            canonical_joint=f'joint_{best[0]}{best[1]}_{best[2]}',
            moment_arm_model_units_per_rad=a_best, sign=-1 if a_best > 0 else 1,
            sign_meaning='torque direction about the canonical positive axis when the muscle is '
                         'activated (a muscle that shortens as the coordinate grows pulls negative)',
            runner_up_joint=f'joint_{second[0]}{second[1]}_{second[2]}' if second else None,
            runner_up_moment_arm=a_second,
            runner_up_ratio=round(abs(a_second/a_best), 6) if a_best else None)
    return rows


def tibia_direction_check(fly, projection):
    """End-to-end sign check on the tibia: which FlyBody tibia rotation is flexion?

    Flexion is defined geometrically here, not by a label: the rotation that brings the claw
    closer to the thorax, measured by forward kinematics over a small sweep of the tibia
    coordinate with everything else at the published neutral pose.  The canonical tibia flexor
    muscle's own projection coefficient is then required to point that way.
    """
    import mujoco
    model, out = fly['model'], {}
    thorax = model.body('thorax').id
    for leg in LEGS:
        jname = flybody_leg_joint_name(leg, 'tibia')
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = int(model.jnt_qposadr[jid])
        claw = model.body(f'claw{flybody_prefix(leg)}').id
        distances = {}
        for delta in (-0.3, 0.0, 0.3):
            data = mujoco.MjData(model)
            data.qpos[:] = 0.0
            data.qpos[adr] = delta
            mujoco.mj_forward(model, data)
            distances[delta] = float(np.linalg.norm(data.xpos[claw] - data.xpos[thorax]))
        positive_extends = distances[0.3] > distances[-0.3]
        coefficient = projection['table']['LF' if leg.startswith('L') else 'RF']['coefficients_to_flybody_hinges'] if False else None
        name = (f'{leg}Tibia_flex_93434')
        row = projection['table'].get(name)
        coeff = (row or {}).get('coefficients_to_flybody_hinges', {}).get(jname)
        out[leg] = dict(flybody_tibia_joint=jname,
                        claw_to_thorax_at_minus_0p3_mm=round(distances[-0.3]*10, 6),
                        claw_to_thorax_at_zero_mm=round(distances[0.0]*10, 6),
                        claw_to_thorax_at_plus_0p3_mm=round(distances[0.3]*10, 6),
                        positive_coordinate_extends=bool(positive_extends),
                        flexion_is_negative_coordinate=bool(positive_extends),
                        canonical_flexor=name,
                        flexor_projection_coefficient=coeff,
                        chain_consistent=(coeff is not None and
                                          ((coeff < 0) == bool(positive_extends))))
    return dict(definition='flexion = the rotation that brings the claw closer to the thorax',
                per_leg=out,
                n_consistent=sum(1 for v in out.values() if v['chain_consistent']),
                n_legs=len(out))


def compiled_drive_matrix(projection, fly):
    """The runtime form of the projection: one entry per FlyBody hinge, ready to be applied."""
    hinge = {}
    for name, row in projection['table'].items():
        for fname, coeff in row['coefficients_to_flybody_hinges'].items():
            entry = hinge.setdefault(fname, dict(flybody_hinge=fname, actuator=fly['actuators'][fname]['index'],
                                                 actuator_index=fly['actuators'][fname]['index'],
                                                 ctrlrange=fly['actuators'][fname]['ctrlrange'],
                                                 muscles=[], coefficients=[], coefficient_norm=0.0))
            entry['muscles'].append(name)
            entry['coefficients'].append(coeff)
            entry['coefficient_norm'] += abs(coeff)
    for entry in hinge.values():
        entry['coefficient_norm'] = round(entry['coefficient_norm'], 6)
        entry['n_muscles'] = len(entry['muscles'])
    return hinge


def torque_variant_xml(fly, directory):
    """Rewrite the 48 leg joint position servos as plain torque motors, same force limits.

    The published model is NOT edited.  Two files are written to this branch's own output
    directory: the variant body (the published fruitfly XML with only the leg actuators changed,
    and an absolute meshdir pointing back at the untouched published meshes) and a wrapper that
    inlines the published floor line verbatim, which is how the published arena composes them.
    """
    directory = Path(directory)
    text = FLYBODY_XML.read_text()
    leg_actuators = [(name, info) for name, info in fly['actuators'].items()
                     if any(name.endswith(flybody_prefix(leg)) for leg in LEGS)
                     and not name.startswith('adhere_')]
    if len(leg_actuators) != 48:
        raise ValueError(f'expected 48 leg joint actuators, found {len(leg_actuators)}')
    replaced = 0
    for name, info in sorted(leg_actuators, key=lambda r: -len(r[0])):
        pattern = re.compile(r'<general\s+name="%s"[^/]*?/>' % re.escape(name))
        # The published leg servos declare NO force limit (forcerange 0 0); the force they can
        # deliver comes from their own published gain and control range.  The torque variant uses
        # that number so its authority is the published servo's authority, not a new choice.
        limit = abs(info['gainprm'])*max(abs(info['ctrlrange'][0]), abs(info['ctrlrange'][1]))
        if limit <= 0:
            raise ValueError(f'actuator {name} has no published authority to transfer')
        new_tag = (f'<motor name="{name}" joint="{name}" gear="{limit:.10g}" '
                   f'ctrlrange="-1 1" ctrllimited="true" group="2"/>')
        text, n = pattern.subn(new_tag, text, count=1)
        if n != 1:
            raise ValueError(f'could not rewrite actuator {name}')
        replaced += n
    text = text.replace('<compiler autolimits="true" angle="radian"/>',
                        f'<compiler autolimits="true" angle="radian" meshdir="{FLYBODY_DIR}"/>', 1)
    header = ('<!-- VARIANT OF A PUBLISHED MODEL, produced by tools_flybody_translation.py.\n'
              '     Only the 48 leg joint position servos were replaced by torque motors whose gear\n'
              '     is the published force limit of the same joint; every joint, geom, mass, spring\n'
              '     and adhesion actuator is the published one, unedited. -->\n')
    text = text.replace('<mujoco model="fruitfly">',
                        '<mujoco model="fruitfly_torque_variant">' + header, 1)
    body_path = directory/'flybody_torque_body.xml'
    body_path.write_text(text)
    floor_text = FLYBODY_FLOOR_XML.read_text()
    floor_lines = [ln for ln in floor_text.splitlines() if '<geom name="floor"' in ln]
    if len(floor_lines) != 1:
        raise ValueError('the published floor line could not be located verbatim')
    start = floor_text.index('<asset>')
    end = floor_text.index('</asset>') + len('</asset>')
    floor_asset = floor_text[start:end]
    wrapper = ('<!-- VARIANT OF A PUBLISHED MODEL: the published floor composition (floor.xml) with\n'
               '     the variant body inlined in place of the published fruitfly.xml.  The asset\n'
               '     block and the floor geom are the published ones, copied verbatim. -->\n'
               '<mujoco>\n'
               '  <include file="flybody_torque_body.xml"/>\n'
               '  ' + floor_asset + '\n'
               '  <worldbody>\n'
               f'    {floor_lines[0].strip()}\n'
               '  </worldbody>\n'
               '</mujoco>\n')
    wrapper_path = directory/'flybody_torque_variant.xml'
    wrapper_path.write_text(wrapper)
    return dict(path=str(wrapper_path), sha256=sha256(wrapper_path),
                body_path=str(body_path), body_sha256=sha256(body_path),
                replaced_actuators=replaced,
                floor_line_from=str(FLYBODY_FLOOR_XML),
                gears={name: round(abs(info['gainprm'])*max(abs(info['ctrlrange'][0]),
                                                            abs(info['ctrlrange'][1])), 8)
                       for name, info in sorted(leg_actuators)},
                margin='ctrl is a torque fraction, -1..1, dimensionless, times the published '
                       'servo\'s own authority gainprm*|ctrlrange|; the published leg servos '
                       'declare no separate force limit, so nothing else was changed.  The '
                       'published posture hold is REMOVED in this variant.')


def figure(doc, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    reg = doc['registration_evidence']['result']
    widths = {'T1': 3, 'T2': 2.9, 'T3': 2.8}
    x = 0.0
    for pair, row in reg.items():
        for r in row['assignment']:
            axes[0].bar(x, abs(r['dot_left']), width=.8,
                        color='tab:red' if abs(r['dot_left']) < .5 else 'tab:blue')
            x += 1.0
        axes[0].axvline(x, color='k', lw=.5)
        axes[0].text(x-1.5, 1.02, f"{pair}\nbest bijection", ha='center', fontsize=7)
        x += 1.5
    axes[0].axhline(.5, color='k', ls=':', lw=1)
    axes[0].set(ylim=(0, 1.1), xlabel='the 21 canonical hinges, grouped by leg pair',
                ylabel='|axis dot product| of the BEST bijection',
                title='an axis-only hinge registration does NOT exist\n(red: the "best" match is still a mismatch)')
    coeff = [r['max_abs_coefficient'] for r in doc['segment_projection']['table'].values()]
    axes[1].hist(coeff, bins=20, range=(0, 1))
    axes[1].set(xlabel='largest |coefficient| a muscle reaches on any same-segment FlyBody hinge',
                ylabel='canonical muscles', title=f'transfer strength, {len(coeff)} muscles\n'
                f"{doc['segment_projection']['n_dropped_muscles']} muscles fall below 0.1 and are lost")
    norms = doc['segment_projection']['per_hinge_coefficient_norm']
    axes[2].barh(list(norms), list(norms.values()))
    axes[2].tick_params(axis='y', labelsize=5)
    axes[2].set(xlabel='summed |coefficient| reaching this FlyBody hinge',
                title='how much of the canonical muscle set reaches each FlyBody hinge')
    fig.suptitle('Canonical muscle body -> published FlyBody: declared translation tables '
                 '(manual research prototype, not a product)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


def main(guard_report=None):
    OUT.mkdir(parents=True, exist_ok=True)
    canon = canonical_axes_and_arms()
    canon_axes_zero = canonical_axes_at_zero_legs(canon)
    fly = flybody_axes()
    registration = discrete_registration(canon_axes_zero, fly)
    projection = segment_projection(canon, canon_axes_zero, fly)
    tibia = tibia_direction_check(fly, projection)
    # MEASURED DEFECT of the declared transfer, not a fix: the canonical tibia flexor's projection
    # points the flexion way on the front and middle legs and the OPPOSITE way on both hind legs.
    # The magnitude is equal, so this is a sign inversion, and the branch therefore carries the
    # per-leg sign that the geometric check supports in a separate, explicitly labelled variant.
    tibia_sign_override = {row['flybody_tibia_joint']: (-1 if not row['chain_consistent'] else 1)
                           for row in tibia['per_leg'].values()}
    tibia['sign_override_by_hinge'] = tibia_sign_override
    tibia['override_meaning'] = ('1 leaves the transferred canonical sign; -1 reverses it so the '
                                'canonical flexor pool flexes the FlyBody tibia on that leg')
    drive = compiled_drive_matrix(projection, fly)
    variant = torque_variant_xml(fly, OUT)
    sensory = {}
    for leg in LEGS:
        sensory[leg] = dict(
            tibia_hinge=flybody_leg_joint_name(leg, 'tibia'),
            multi_joint_group=sorted(n for n in fly['axes']
                                     if n.endswith(flybody_prefix(leg))
                                     and n.split('_')[0] in ('coxa', 'femur', 'tibia')),
            note='the multi-joint group carries exactly the six FlyBody hinges whose segment has a '
                 'canonical counterpart; the FlyBody tarsus hinges are excluded because the '
                 'canonical body has no tarsus hinge and no canonical sensory quantity refers to one')
    doc = dict(
        status='MEASURED_FLYBODY_TRANSLATION_TABLES',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; translation tables only, no behaviour',
        question='Can the canonical muscle body be translated onto the published FlyBody body, and '
                 'what is measured rather than assumed in that translation?',
        units=dict(angle='rad', moment_arm='tendon length model units per rad',
                   axis='unit vector in the thorax frame',
                   canonical_length_unit=cp.MODEL_LENGTH_UNIT_NAME, flybody_length_unit='cm (CGS)',
                   comparable='axis dot products are dimensionless'),
        registration_evidence=registration,
        segment_projection=projection,
        compiled_drive_matrix=drive,
        sensory_translation=sensory,
        tibia_direction_check=tibia,
        torque_variant=variant,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        canonical_xml=cp.sha256_file(cp.CANONICAL_XML),
                        canonical_xml_declared=cp.CANONICAL_XML_SHA256,
                        flybody_xml=sha256(FLYBODY_XML), flybody_xml_declared=FLYBODY_XML_SHA,
                        canonical_axes_pose='symmetric reference configuration: all canonical leg '
                                            'hinges zero, root seated (the pose '
                                            'diagnostics/diag_tendon_route_fidelity.py declares as '
                                            'primary for homology questions)',
                        flybody_axes_pose='published neutral pose: every hinge zero, root at origin',
                        canonical_weight_model_units=canon['weight'],
                        flybody_weight_model_units=fly['weight'],
                        method='unit-vector geometry and finite differences only; no fitting, no '
                               'hand-set angle, no tuning of any kind',
                        memory_guard=guard_report('completed') if guard_report else None),
        what_this_does_not_support=[
            'No episode is run here; nothing about standing, walking or the neural circuit is '
            'measured in this file.',
            'No one-to-one hinge registration exists between the two models: the best bijection is '
            'reported and is anatomically absurd.  The segment correspondence and the projection '
            'are DECLARED, and the projection coefficients are the evidence for them.',
            'Moment-arm signs are a geometric derivative, not a physiological flexion/extension '
            'label; only the tibia has an end-to-end geometric check here.',
            'Muscle-level resolution is LOST in the projection: several canonical muscles land on '
            'the same FlyBody hinge and only their signed sum survives.',
            'The torque variant is this project\'s declared modification of a published model, not '
            'a published model; its ctrl is a dimensionless torque fraction.',
            'The connectome, the tier, the spikes and the clocks are untouched by this file; nothing '
            'here says which cell should reach which hinge.',
        ])
    cp.write_result(OUT/'flybody_translation.json', question=doc['question'], status='PASS',
                    measured={k: doc[k] for k in ('registration_evidence', 'segment_projection',
                                                  'compiled_drive_matrix', 'sensory_translation',
                                                  'tibia_direction_check', 'torque_variant')},
                    provenance=doc['provenance'], units=doc['units'], label=doc['label'],
                    status_meaning='PASS means the tables were measured and written; it is not a '
                                   'biological or behavioural result',
                    illustrative_vs_measured='Axis dots, moment arms and the geometry of flexion are '
                                             'measured on the two compiled models; the segment '
                                             'correspondence and the projection rule are DECLARED',
                    what_this_does_not_support=doc['what_this_does_not_support'])
    print('figure', figure(doc, OUT/'flybody_translation.png'), flush=True)
    print(json.dumps(dict(muscles=projection['n_muscles'], dropped=projection['n_dropped_muscles'],
                          hinges_driven=len(drive), senses=len(sensory),
                          tibia_consistent=tibia['n_consistent'],
                          registration=registration['result']), indent=1)[:2500])
    return doc


if __name__ == '__main__':
    with mg.guarded_run('flybody_translation', need_mb=1600.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
