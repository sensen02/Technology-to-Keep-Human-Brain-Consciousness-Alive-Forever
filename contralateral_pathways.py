"""Contralateral pathway attempt, driven by the signed K=2 solution.

K-ladder (117000Z) showed the settled stance becomes exactly balanceable with signed
activation capped at |a|<=2 (K*=2; residual 0.0), while the mid-walk handover pose needs
K*=16. So the settled stance needs about two units of signed magnitude, i.e. muscle
force in the direction opposite to the existing attachment for the pools that the signed
solution drives negative.

This step mirrors ONLY the distal attachment of muscles that the K*=2 signed solution
drives clearly negative (activation < -0.5), each such muscle once, using the same
already-authorised reflection rule (180 deg about its own joint axis in its own body
frame). Everything else - names, tendon routes, Fmax, passive bias, neural mapping,
LF native sites - is untouched; length ranges are re-centred at each pose by the
declared 089 calibration.

Reported for both poses and both pools; criterion fixed in advance: bounded
0<=a<=1 outer residual <= 1e-6 for the 66 pool AND the 90 pool.

Budget: 2 model loads, <=200 forwards, 0 body integration, 0 neural steps.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog
import xml.etree.ElementTree as ET

REB = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-rebuild/20261028T113000Z')
KL = Path('/home/sensen/Desktop/engine修复验收/s5-k-ladder/20261028T117000Z')
S74 = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
S107 = Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-contralateral-pathways/20261028T119000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def measure(xml, q, weight, mapping, counts):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    m = _load_mjcf(str(xml)).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    lr0 = m.actuator_lengthrange.copy()
    d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
    mujoco.mj_forward(m, d); counts['forward'] += 1
    Lnow = d.actuator_length.copy(); gp = m.actuator_gainprm
    lmin, lmax = gp[:, 0], gp[:, 1]; width = lr0[:, 1] - lr0[:, 0]; L0 = width / (lmax - lmin)
    new0 = Lnow - (1.0 - lmin) * L0
    m.actuator_lengthrange[:, 0] = new0; m.actuator_lengthrange[:, 1] = new0 + width
    d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
    d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0; d.qacc_warmstart[:] = 0
    mujoco.mj_forward(m, d); counts['forward'] += 1
    base = d.qfrc_actuator.copy()
    Mo = np.zeros((m.nv, m.nu))
    if d.actuator_moment.size == m.nv * m.nu:
        Mo = d.actuator_moment.reshape(m.nv, m.nu).copy()
    else:
        for j in range(m.nu):
            adr = int(d.moment_rowadr[j]); nnz = int(d.moment_rownnz[j])
            Mo[d.moment_colind[adr:adr + nnz], j] = d.actuator_moment[adr:adr + nnz]
    d.act[:] = 1.; mujoco.mj_forward(m, d); counts['forward'] += 1
    f1 = d.actuator_force.copy()
    d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
    f0 = d.actuator_force.copy()
    A = Mo * (f1 - f0)[None, :]
    blocks = []
    for c in range(d.ncon):
        co = d.contact[c]; a1, a2 = int(co.geom1), int(co.geom2)
        p1 = m.geom_type[a1] == mujoco.mjtGeom.mjGEOM_PLANE; p2 = m.geom_type[a2] == mujoco.mjtGeom.mjGEOM_PLANE
        if not (p1 or p2):
            continue
        g = a2 if p1 else a1; nm = m.geom(g).name
        if not any(nm.startswith(l) and 'Tars' in nm for l in LEGS):
            continue
        jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jac(m, d, jp, jr, co.pos, int(m.geom_bodyid[g])); counts['jac'] += 1
        blocks.append(jp.T)
    J = np.column_stack(blocks) if blocks else np.zeros((48, 0))
    E = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
    lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
    Lrow = E[lim].T if lim.size else np.zeros((48, 0))
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    return dict(A=A, base=base, J=J, L=Lrow, G=G, bias=d.qfrc_bias.copy(), passive=d.qfrc_passive.copy(),
                weight=weight, nc=J.shape[1] // 3, nl=Lrow.shape[1], contacts=len(blocks))


def solve(P, H, lo, hi, inner=False):
    n = H.shape[1]; nc = P['nc']; nl = P['nl']
    B = np.column_stack([P['A'] @ H, P['J'], P['L']]) / P['weight']
    y = (P['bias'] - P['passive'] - P['base']) / P['weight']
    rws = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
    for j in range(nc):
        mr = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mr[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rws.append(mr); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bnd = [(lo, hi)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rws), b_ub=np.concatenate(rhs), bounds=bnd, method='highs')
    return (float(sol.fun), sol.x[:n].copy()) if sol.success else (None, None)


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    weight = float(json.loads((S74 / 'GAINS.json').read_text())['weight'])
    mapping = json.loads((S74 / 'MAPPING.json').read_text())
    q_settled = np.load(S74 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    q_hand = np.load(S107 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    names = [a.get('name') for a in ET.parse(REB / 'MODEL_REBUILT.xml').getroot().findall('./actuator/*')]
    dump(O / 'PREREGISTRATION.json', dict(
        scope='One contralateral pathway for muscles that the K*=2 signed solution drives negative; measure bounded LP at the settled and handover poses for both pools.',
        K_ladder=dict(settled_K_star=2.0, handover_K_star=16.0, bounded_0_1_settled=0.044910249883673,
                      note='117000Z labelled its 0<=a<=1 reference with a signed K=1 value by mistake (0.010333); the true bounded reference is 0.044910 as measured in 115000Z and re-measured here.'),
        threshold='activation < -0.5 in the K=2 signed solution at the settled pose',
        criterion='bounded 0<=a<=1 outer residual <= 1e-6 for the 66 pool AND the 90 pool',
        budget=dict(model_load=3, forward=200, body_step=0, neural_step=0),
        source_hashes={str(p): sha(p) for p in [Path(__file__), REB / 'MODEL_REBUILT.xml', S74 / 'HANDOVER_BEFORE.npz',
                                                S107 / 'HANDOVER_BEFORE.npz', S74 / 'GAINS.json', S74 / 'MAPPING.json',
                                                KL / 'K_LADDER.json']}))
    # reference measurements on the unmodified rebuilt model
    P0 = measure(REB / 'MODEL_REBUILT.xml', q_settled, weight, mapping, counts)
    counts['lps'] += 2
    b0_66 = solve(P0, P0['G'], 0., 1.)[0]; b0_90 = solve(P0, np.eye(90), 0., 1.)[0]
    s0_66, x_signed = solve(P0, P0['G'], -2., 2.)
    pick = [int(i) for i in range(len(x_signed)) if x_signed[i] < -0.5] if x_signed is not None else []
    # expand pool indices back to actuators
    pool_to_act = []
    for row in mapping['rows']:
        if row['local_indices']:
            pool_to_act.append(row['actuator_index'])
    target_actuators = sorted({pool_to_act[i] for i in pick})
    dump(O / 'SELECTION.json', dict(negative_threshold=-0.5, pools=[int(i) for i in pick],
                                    actuator_indices=target_actuators,
                                    actuator_names=[names[i] for i in target_actuators],
                                    bounded_before=dict(residual66=b0_66, residual90=b0_90),
                                    signed_K2_settled=s0_66))
    # build the mirrored model: reflect the distal attachment of each selected muscle
    tree = ET.parse(REB / 'MODEL_REBUILT.xml'); root = tree.getroot()
    owner = {}
    for body in root.iter('body'):
        for s in body.findall('site'):
            owner[s.get('name')] = body.get('name')
    tendon_sites = {t.get('name'): [c.get('site') for c in t] for t in root.findall('./tendon/spatial')}
    act_el = {a.get('name'): a for a in root.findall('./actuator/*')}
    site_el = {s.get('name'): s for s in root.iter('site')}
    m0 = P0  # matrices only; need joint info from a compiled model
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    mm = _load_mjcf(str(REB / 'MODEL_REBUILT.xml')).compile()
    changes = []
    for ai in target_actuators:
        aname = names[ai]
        row = np.abs(P0['A'][:, ai])
        dof = int(row.argmax())
        jid = int(mm.dof_jntid[dof])
        child = mm.body(int(mm.jnt_bodyid[jid])).name
        tname = act_el[aname].get('tendon')
        cand = [s for s in tendon_sites.get(tname, []) if owner.get(s) == child]
        if not cand:
            continue
        sname = cand[-1]
        axis = np.array(mm.jnt_axis[jid], dtype=float); axis /= np.linalg.norm(axis)
        anchor = np.array(mm.jnt_pos[jid], dtype=float)
        el = site_el[sname]
        old = np.array([float(x) for x in el.get('pos').split()])
        new = anchor + (2 * np.outer(axis, axis) - np.eye(3)) @ (old - anchor)
        el.set('pos', ' '.join(format(v, '.17g') for v in new))
        changes.append(dict(actuator=aname, site=sname, body=child, joint=mm.joint(jid).name,
                            old_pos=old, new_pos=new))
    out = O / 'MODEL_CONTRALATERAL.xml'
    tree.write(out, encoding='utf8', xml_declaration=True)
    dump(O / 'SITE_CHANGES.json', dict(mirrored=len(changes), changes=changes,
                                       note='Only muscles that the signed K=2 solution drove below -0.5 were mirrored, once each; same reflection rule as before.'))
    # measure after
    Ps = measure(out, q_settled, weight, mapping, counts)
    Ph = measure(out, q_hand, weight, mapping, counts)
    res = {}
    for tag, P in (('settled', Ps), ('handover', Ph)):
        for pool, H in (('trusted66', P['G']), ('all90', np.eye(90))):
            for cone, inner in (('outer', False), ('inner', True)):
                counts['lps'] += 1
                r, _ = solve(P, H, 0., 1., inner=inner)
                res[f'{tag}_{pool}_{cone}'] = r
    ok = bool(res['settled_trusted66_outer'] is not None and res['settled_trusted66_outer'] <= 1e-6
              and res['settled_all90_outer'] is not None and res['settled_all90_outer'] <= 1e-6)
    dump(O / 'RESULTS.json', dict(mirrored_sites=len(changes), bounded_residuals=res, criterion_met=ok,
                                  before=dict(settled66=b0_66, settled90=b0_90), model_sha256=sha(out)))
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
    (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(mirrored=len(changes), before=dict(s66=b0_66, s90=b0_90), after=res, criterion_met=ok), indent=2))


if __name__ == '__main__':
    main()
