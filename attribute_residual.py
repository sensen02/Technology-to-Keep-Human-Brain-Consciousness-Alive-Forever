"""Residual attribution for the rebuilt-muscle static LP (declared, bounded).

Question: the settled-stance LP still leaves 0.0449x bodyweight, and a 0.5x-32x muscle
force sweep barely moved it. Before any amplitude calibration, attribute that residual
to the rows and column families that actually carry it.

Budget, fixed before any call: 1 model load, <=200 forwards, 0 body integration,
0 neural steps. Branches are also fixed in the pre-registration and are not changed
after seeing numbers.

Nothing here changes a model parameter, a neural mapping, Fmax or the passive bias.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf
import xml.etree.ElementTree as ET

REB = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-rebuild/20261028T113000Z')
PREV = Path('/home/sensen/Desktop/engine修复验收/s5-joint-authority/20261028T109000Z')
S74 = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
S = Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-residual-attribution/20261028T115000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def solve(B, y, bounds, inner, nc, nl, n):
    rws = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
    for j in range(nc):
        mr = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mr[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rws.append(mr); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bnd = list(bounds) + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rws), b_ub=np.concatenate(rhs), bounds=bnd, method='highs')
    if not sol.success:
        return None, None, sol.message
    x = sol.x
    res = B @ x[:-1] - y
    return float(x[-1]), res, sol.message


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    q = np.load(S74 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    weight = float(json.loads((S74 / 'GAINS.json').read_text())['weight'])
    mapping = json.loads((S74 / 'MAPPING.json').read_text())
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Residual attribution + two ablations on the rebuilt model at the settled stance. No parameter, mapping or geometry change.',
        budget=dict(model_load=1, forward=200, body_step=0, neural_step=0),
        rows='root 6 / 42 leg joints / 3 active joint-limit rows',
        column_families='66 muscle columns / contact cone columns (3 per foot contact) / active limit columns',
        ablations=dict(A='muscle activation bounds relaxed to (-inf, inf)', B='muscle activation fixed at 0; only contact and limit forces'),
        branches_fixed_before_measurement=[
            'if ablation B residual > 1e-6: stop the muscle-geometry line; the carrying gap is in contact/foot placement/posture; return to pose+contact search within the registered domain',
            'if ablation B <= 1e-6 and ablation A <= 1e-6: continue with amplitude calibration to the native LF positive/negative ratio, criterion 66-pool outer <= 1e-6 fixed in advance',
            'if ablation A > 1e-6 while B <= 1e-6: list the irreducible rows and the physical term (bias / passive / limit) they belong to; fix that term next round instead of geometry'],
        source_hashes={str(p): sha(p) for p in [Path(__file__), REB / 'MODEL_REBUILT.xml', S74 / 'HANDOVER_BEFORE.npz',
                                                S74 / 'GAINS.json', S74 / 'MAPPING.json',
                                                PREV / 'JOINT_AUTHORITY.json', REB / 'JOINT_AUTHORITY_AFTER.json',
                                                S / 'MODEL.xml']}))
    m = _load_mjcf(str(REB / 'MODEL_REBUILT.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    # declared 089 recentring at this pose (same as 114000Z)
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
    blocks = []; geoms = []
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
        blocks.append(jp.T); geoms.append(nm)
    J = np.column_stack(blocks)
    E = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
    lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
    Lrow = E[lim].T if lim.size else np.zeros((48, 0))
    bias = d.qfrc_bias.copy(); passive = d.qfrc_passive.copy()
    nc = J.shape[1] // 3; nl = Lrow.shape[1]
    B = np.column_stack([A @ G, J, Lrow]) / weight
    y = (bias - passive - base) / weight
    np.savez_compressed(O / 'LP_INPUTS.npz', A=A, G=G, J=J, L=Lrow, base=base, bias=bias, passive=passive,
                        weight=weight, qpos=q, contact_geoms=np.array(geoms))
    # ---- baseline LP ----
    t0, res0, msg = solve(B, y, [(0., 1.)] * 66, False, nc, nl, 66); counts['lps'] += 1
    # ---- ablation A: unbounded muscles ----
    tA, resA, msgA = solve(B, y, [(None, None)] * 66, False, nc, nl, 66); counts['lps'] += 1
    # ---- ablation B: muscles zero ----
    tB, resB, msgB = solve(B, y, [(0., 0.)] * 66, False, nc, nl, 66); counts['lps'] += 1
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    jnames = [m.joint(j).name for j in jid]
    row_names = [f'root_{k}' for k in range(6)] + jnames + [f'limit_{int(r)}' for r in lim]
    fam = dict(muscle=slice(0, 66), contact=slice(66, 66 + 3 * nc), limit=slice(66 + 3 * nc, 66 + 3 * nc + nl))
    rows = []
    for r in range(48):
        rows.append(dict(row=r, name=row_names[r], residual=float(res0[r]),
                         abs_residual=abs(float(res0[r])),
                         b_component=float(y[r]),
                         support_muscle=float(np.abs(B[r, fam['muscle']]).sum()),
                         support_contact=float(np.abs(B[r, fam['contact']]).sum()),
                         support_limit=float(np.abs(B[r, fam['limit']]).sum()) if nl else 0.0,
                         bias=float(bias[r] / weight), passive=float(passive[r] / weight),
                         base_floor=float(base[r] / weight)))
    total = sum(x['abs_residual'] for x in rows)
    for x in rows:
        x['share'] = x['abs_residual'] / total if total else 0.
    worst = sorted(rows, key=lambda x: -x['abs_residual'])[:12]
    out = dict(
        bodyweight_native=weight,
        baseline=dict(residual=t0, l2=float(np.linalg.norm(res0)), contacts=nc, limits=nl, message=msg),
        ablation_A_unbounded_muscles=dict(residual=tA, message=msgA),
        ablation_B_zero_muscles=dict(residual=tB, message=msgB),
        rows_with_share_above_1pct=[{k: x[k] for k in ('row', 'name', 'abs_residual', 'share',
                                                       'support_muscle', 'support_contact', 'support_limit',
                                                       'bias', 'passive', 'base_floor')} for x in rows if x['share'] > .01],
        worst_rows=[{k: x[k] for k in ('row', 'name', 'abs_residual', 'support_muscle', 'support_contact', 'support_limit')} for x in worst],
        intuitive=dict(
            residual_in_bodyweight=t0,
            equivalent_force_units=float(t0 * weight),
            body_mass_g=float(weight / abs(m.opt.gravity[2]) * 1000),
            equivalent_grams_of_force=float(t0 * weight / abs(m.opt.gravity[2]) * 1000),
            note='Residual is distributed over 48 generalized directions, so the gram figure is a force-equivalent, not a weight on one foot.'),
        per_row=rows)
    dump(O / 'ATTRIBUTION.json', out)
    # ---- full +/- authority table before vs after ----
    before = json.loads((PREV / 'JOINT_AUTHORITY.json').read_text())
    after = json.loads((REB / 'JOINT_AUTHORITY_AFTER.json').read_text())
    table = []
    for b_, a_ in zip(before, after):
        table.append(dict(joint=b_['joint'],
                          before_neg66=b_['neg66'], before_pos66=b_['pos66'],
                          after_neg66=a_['neg66'], after_pos66=a_['pos66'],
                          before_neg90=b_['neg90'], before_pos90=a_['pos90'],
                          after_neg90=a_['neg90'], after_pos90=a_['pos90'],
                          two_sided_before66=b_['two_sided66'], two_sided_after66=a_['two_sided66'],
                          two_sided_before90=b_['two_sided90'], two_sided_after90=a_['two_sided90']))
    dump(O / 'AUTHORITY_FULL_TABLE.json', table)
    # ---- LF native muscle integrity ----
    def lf_signature(path):
        root = ET.parse(path).getroot()
        owner = {}
        for body in root.iter('body'):
            for s in body.findall('site'):
                owner[s.get('name')] = body.get('name')
        sites = {}
        for s in root.iter('site'):
            if s.get('name', '').startswith('LF'):
                sites[s.get('name')] = dict(body=owner.get(s.get('name')), pos=s.get('pos'))
        tendons = {}
        for t in root.findall('./tendon/spatial'):
            if t.get('name', '').startswith('LF'):
                tendons[t.get('name')] = [c.get('site') for c in t]
        acts = {}
        for a in root.findall('./actuator/*'):
            if a.get('name', '').startswith('LF'):
                acts[a.get('name')] = {k: a.get(k) for k in ('tendon', 'gainprm', 'biasprm', 'dynprm', 'ctrlrange', 'class')}
        blob = json.dumps(dict(sites=sites, tendons=tendons, actuators=acts), sort_keys=True)
        return len(sites), len(tendons), len(acts), hashlib.sha256(blob.encode()).hexdigest(), dict(sites=sites, tendons=tendons, actuators=acts)
    n0, t0n, a0n, h0, sig0 = lf_signature(S / 'MODEL.xml')
    n1, t1n, a1n, h1, sig1 = lf_signature(REB / 'MODEL_REBUILT.xml')
    lr_changed = json.loads((REB / 'SITE_CHANGES.json').read_text())['changed_sites']
    integrity = dict(lf_sites_before=n0, lf_sites_after=n1, lf_tendons_before=t0n, lf_tendons_after=t1n,
                     lf_actuators_before=a0n, lf_actuators_after=a1n,
                     signature_before=h0, signature_after=h1, identical=bool(h0 == h1),
                     changed_non_LF_sites=lr_changed,
                     note='Signature covers LF site names, owning bodies and positions, LF tendon site order, and LF actuator tendon/gainprm/biasprm/dynprm/ctrlrange. Length ranges are excluded because they were re-centred for all 90 muscles by the declared 089 calibration.')
    dump(O / 'LF_INTEGRITY.json', integrity)
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0,
                                    source_unchanged=all(sha(Path(p)) == h for p, h in
                                                         json.loads((O / 'PREREGISTRATION.json').read_text())['source_hashes'].items())))
    (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: out[k] for k in ('baseline', 'ablation_A_unbounded_muscles', 'ablation_B_zero_muscles',
                                          'intuitive')}, indent=2))
    print('LF integrity identical:', integrity['identical'], integrity['signature_before'][:16], integrity['signature_after'][:16])


if __name__ == '__main__':
    main()
