"""B-path step 2 (declared): restore the native antagonist sign structure per leg.

Measured defect (109000Z): 5 joints have one-sided authority in the 66 trusted pool
(LHCoxa_yaw, LHCoxa_roll, LMCoxa_yaw, RMCoxa_yaw, RMCoxa_roll) and 2 remain one-sided
even with all 90 actuators (LHCoxa_roll, RMCoxa_roll). A joint that cannot be driven
in one direction cannot be held by ANY bounded muscle activation.

Rule (uniform, applied to every affected muscle, no per-joint tuning):
1. Work out, at the operating pose, each muscle's moment-arm sign on its dominant joint.
2. Take the native left-front leg as the reference for that joint class, and decide
   whether the target leg has a global orientation flip (majority vote of signs).
3. For every muscle whose sign disagrees with the reference after that flip, reflect its
   insertion site 180 degrees about its own joint axis through the joint anchor, inside
   that site's own body frame. Magnitudes, muscle names, force parameters, tendon routes
   and the actuator/neuron mapping are untouched; only which side of the joint a muscle
   pulls from changes.

Everything is then re-measured from scratch (authority, contacts, LP). A negative result
is reported as such; the rule is fixed before the measurement, not tuned after it.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
import xml.etree.ElementTree as ET

S = Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z')
R = Path('/home/sensen/Desktop/engine修复验收/s5-joint-authority/20261028T109000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-rebuild/20261028T113000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def force_matrix(m, d, q, counts):
    d.qpos[:] = q; d.qvel[:] = 0; d.ctrl[:] = 0
    d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0
    d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
    f0 = d.actuator_force.copy()
    Mo = np.zeros((m.nv, m.nu))
    if d.actuator_moment.size == m.nv * m.nu:
        Mo = d.actuator_moment.reshape(m.nv, m.nu).copy()
    else:
        for j in range(m.nu):
            adr = int(d.moment_rowadr[j]); nnz = int(d.moment_rownnz[j])
            Mo[d.moment_colind[adr:adr + nnz], j] = d.actuator_moment[adr:adr + nnz]
    d.act[:] = 1.; mujoco.mj_forward(m, d); counts['forward'] += 1
    f1 = d.actuator_force.copy()
    return Mo * (f1 - f0)[None, :], d


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time()
    counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    h = np.load(S / 'HANDOVER_BEFORE.npz')
    b = np.load(S / 'BODY.npz')
    mapping = json.loads((S / 'MAPPING.json').read_text())
    weight = float(json.loads((S / 'GAINS.json').read_text())['weight'])
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    authority = json.loads((R / 'JOINT_AUTHORITY.json').read_text())
    targets = sorted({r['joint'] for r in authority if not r['two_sided90']}
                     | {r['joint'] for r in authority if not r['two_sided66']})
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Declared muscle-attachment geometry rebuild for joints that lack two-sided authority. Only the side of a muscle insertion changes; no muscle, tendon route, force parameter, contact or neural mapping is added or removed.',
        rule='reflect insertion sites 180 deg about their own joint axis so each muscle matches the native left-front sign structure for its joint class, after removing any global per-leg orientation flip by majority vote',
        targets=targets,
        fixed_criteria=dict(authority='42/42 leg joints two-sided with all 90 actuators, and 40+/42 with the trusted 66 pool',
                            static_lp='trusted66 outer residual <= 1e-6 at the handover pose',
                            geometry='six feet contact, no non-foot contact, front legs not crossing the midline, joint limits respected'),
        accepted_before_measurement=True,
        source_hashes={str(p): sha(p) for p in [Path(__file__), S / 'MODEL.xml', S / 'HANDOVER_BEFORE.npz',
                                                R / 'JOINT_AUTHORITY.json', S / 'MAPPING.json']}))

    # ---------- reference signs from the native left-front leg ----------
    m = _load_mjcf(str(S / 'MODEL.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    q = h['qpos'].copy()
    A, d = force_matrix(m, d, q, counts)
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    jname = {j: m.joint(j).name for j in jid}
    dof_of_joint = {j: int(m.jnt_dofadr[j]) for j in jid}
    dom = np.abs(A).argmax(axis=0)
    import re
    def muscle_class(name):
        return re.sub(r'^[LR][FMH][^_]*_', '', name)
    def joint_class(name):
        return re.sub(r'^[LR][FMH]', '', name.split('_', 1)[1])
    ref = {}
    for j in jid:
        nm = jname[j]
        if not nm.startswith('joint_LF'):
            continue
        cls = joint_class(nm)
        row = dof_of_joint[j]
        for a in range(m.nu):
            if abs(A[row, a]) > 1e-9:   # muscles actually acting on this joint, not just the dominant one
                ref[(cls, muscle_class(m.actuator(a).name))] = int(np.sign(A[row, a]))
    dump(O / 'LF_REFERENCE_SIGNS.json', {f'{k[0]}|{k[1]}': v for k, v in ref.items()})
    debug = []

    # ---------- decide which insertion sites to reflect ----------
    tree = ET.parse(S / 'MODEL.xml'); root = tree.getroot()
    site_owner = {}
    for body in root.iter('body'):
        for site in body.findall('site'):
            site_owner[site.get('name')] = body.get('name')
    site_el = {s.get('name'): s for s in root.iter('site')}
    tendon_sites = {t.get('name'): [c.get('site') for c in t] for t in root.findall('./tendon/spatial')}
    act_el = {a.get('name'): a for a in root.findall('./actuator/*')}
    body_of = {m.body(i).name: i for i in range(m.nbody)}
    changes = []
    for target in targets:
        j = next(j for j in jid if jname[j] == target)
        cls = joint_class(target)
        dofrow = A[dof_of_joint[j], :]
        actors = [a for a in range(m.nu) if abs(dofrow[a]) > 1e-9]
        if not actors:
            continue
        signs = {a: int(np.sign(dofrow[a])) for a in actors}
        # per-leg global flip: majority vote against the reference sign of the same class/name tail
        votes = []
        for a in actors:
            key = (cls, muscle_class(m.actuator(a).name))
            if key in ref:
                votes.append(signs[a] * ref[key])
        flip = -1 if sum(votes) < 0 else 1
        debug.append(dict(joint=target, actors=[m.actuator(a).name for a in actors],
                          signs={m.actuator(a).name: signs[a] for a in actors},
                          votes=int(len(votes)), flip=int(flip),
                          ref_for_class={k[1]: v for k, v in ref.items() if k[0] == cls}))
        for a in actors:
            key = (cls, muscle_class(m.actuator(a).name))
            if key not in ref:
                continue
            if signs[a] * flip * ref[key] > 0:
                continue
            # reflect this muscle's insertion site about the joint axis, in its own body frame
            aname = m.actuator(a).name
            tname = act_el[aname].get('tendon')
            sites = tendon_sites.get(tname, [])
            child_body = m.body(m.jnt_bodyid[j]).name
            cand = [s for s in sites if site_owner.get(s) == child_body]
            if not cand:
                continue
            sname = cand[-1]   # distal-most site on the child body
            bodyid = body_of[child_body]
            axis = np.array(m.jnt_axis[j], dtype=float)
            axis /= np.linalg.norm(axis)
            anchor = np.array(m.jnt_pos[j], dtype=float)
            el = site_el[sname]
            old = np.array([float(x) for x in el.get('pos').split()])
            rel = old - anchor
            # Rodrigues rotation by pi about the axis
            new = anchor + (2 * np.outer(axis, axis) - np.eye(3)) @ rel
            el.set('pos', ' '.join(format(v, '.17g') for v in new))
            changes.append(dict(joint=target, actuator=aname, site=sname, body=child_body,
                                old_pos=old, new_pos=new))
    dump(O / 'MATCH_DEBUG.json', debug)
    out = O / 'MODEL_REBUILT.xml'
    tree.write(out, encoding='utf8', xml_declaration=True)
    dump(O / 'SITE_CHANGES.json', dict(changed_sites=len(changes), changes=changes,
                                       note='Reflection by pi about each joint axis; magnitudes, tendon routes and names unchanged.'))
    # ---------- re-measure ----------
    m2 = _load_mjcf(str(out)).compile(); counts['model_load'] += 1
    d2 = mujoco.MjData(m2)
    # Declared 089 calibration: because reflections change tendon lengths, re-centre every
    # muscle length range (translation only, width preserved) so normalized length is 1.0
    # at the operating pose.
    lr_old = m2.actuator_lengthrange.copy()
    d2.qpos[:] = q; d2.qvel[:] = 0; d2.act[:] = .0001; d2.ctrl[:] = 0
    mujoco.mj_forward(m2, d2); counts['forward'] += 1
    Lnow = d2.actuator_length.copy()
    gp2 = m2.actuator_gainprm; lmin2, lmax2 = gp2[:, 0], gp2[:, 1]
    width2 = lr_old[:, 1] - lr_old[:, 0]; L02 = width2 / (lmax2 - lmin2)
    new0 = Lnow - (1.0 - lmin2) * L02
    m2.actuator_lengthrange[:, 0] = new0; m2.actuator_lengthrange[:, 1] = new0 + width2
    dump(O / 'LENGTHRANGE_RECENTER.json', dict(n=90, note='translation only, width preserved, at the operating pose; engineering calibration'))

    A2, d2 = force_matrix(m2, d2, q, counts)
    A266 = A2 @ G
    rows = []
    for j in jid:
        row = int(m2.jnt_dofadr[j]); r66 = A266[row, :]; r90 = A2[row, :]
        rows.append(dict(joint=jname[j],
                         neg66=float(r66[r66 < 0].sum()), pos66=float(r66[r66 > 0].sum()),
                         neg90=float(r90[r90 < 0].sum()), pos90=float(r90[r90 > 0].sum()),
                         two_sided66=bool((r66 < -1e-3).any() and (r66 > 1e-3).any()),
                         two_sided90=bool((r90 < -1e-3).any() and (r90 > 1e-3).any())))
    dump(O / 'JOINT_AUTHORITY_AFTER.json', rows)
    # contact rows and LP at the same pose
    blocks = []
    for c in range(d2.ncon):
        co = d2.contact[c]; a1, a2 = int(co.geom1), int(co.geom2)
        p1 = m2.geom_type[a1] == mujoco.mjtGeom.mjGEOM_PLANE; p2 = m2.geom_type[a2] == mujoco.mjtGeom.mjGEOM_PLANE
        if not (p1 or p2):
            continue
        g = a2 if p1 else a1; nm = m2.geom(g).name
        if not any(nm.startswith(l) and 'Tars' in nm for l in LEGS):
            continue
        jp = np.zeros((3, m2.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jac(m2, d2, jp, jr, co.pos, int(m2.geom_bodyid[g])); counts['jac'] += 1
        blocks.append(jp.T)
    J = np.column_stack(blocks)
    E = d2.efc_J.reshape(d2.nefc, m2.nv).copy() if d2.efc_J.size == d2.nefc * m2.nv else np.zeros((d2.nefc, m2.nv))
    lim = np.flatnonzero(np.asarray(d2.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
    Lrow = E[lim].T if lim.size else np.zeros((48, 0))
    d2.qpos[:] = q; d2.qvel[:] = 0; d2.act[:] = .0001; d2.ctrl[:] = 0
    mujoco.mj_forward(m2, d2); counts['forward'] += 1
    base = d2.qfrc_actuator.copy(); bias = d2.qfrc_bias.copy(); passive = d2.qfrc_passive.copy()
    lp = {}
    for tag, H in (('trusted66', G), ('all90', np.eye(90))):
        for cone, inner in (('outer', False), ('inner', True)):
            counts['lps'] += 1
            n = H.shape[1]
            B = np.column_stack([A2 @ H, J, Lrow]) / weight
            y = (bias - passive - base) / weight
            nc = J.shape[1] // 3; nl = Lrow.shape[1]
            rws = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
            for jj in range(nc):
                mr = np.zeros((4, B.shape[1] + 1))
                dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for k, (sx, sy) in enumerate(dirs):
                    mr[k, n + 3 * jj:n + 3 * jj + 3] = [sx, sy, -1.]
                rws.append(mr); rhs.append(np.zeros(4))
            obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
            bounds = [(0., 1.)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
            sol = linprog(obj, A_ub=np.vstack(rws), b_ub=np.concatenate(rhs), bounds=bounds, method='highs')
            lp[f'{tag}_{cone}'] = dict(residual=float(sol.fun) if sol.success else None,
                                       feasible=bool(sol.success and sol.fun <= 1e-6))
    summary = dict(changed_sites=len(changes), targets=targets,
                   two_sided_66=int(sum(r['two_sided66'] for r in rows)),
                   two_sided_90=int(sum(r['two_sided90'] for r in rows)),
                   lp=lp, model_sha256=sha(out),
                   criteria_met=bool(sum(r['two_sided90'] for r in rows) == 42
                                     and sum(r['two_sided66'] for r in rows) >= 40
                                     and lp['trusted66_outer']['feasible']))
    dump(O / 'RESULTS.json', summary)
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
    (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
