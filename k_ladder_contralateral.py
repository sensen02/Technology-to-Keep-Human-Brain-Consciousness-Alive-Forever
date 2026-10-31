"""K-ladder: how large a signed muscle magnitude is needed before the stance balances.

Why: at the settled stance the bounded LP (0<=a<=1) leaves 0.04491 bodyweight, the
nonnegative-unbounded LP only improves to 0.03670, and a SIGNED relaxation reaches 0.
Adding "one contralateral pathway per muscle" is mathematically equivalent to allowing
a signed magnitude up to about 1, so the smallest signed cap K* at which the residual
reaches 1e-6 is a lower bound on how many contralateral pathways are needed.

Fixed before running (see PREREGISTRATION): pose is the settled stance used by
114000Z/115000Z so 0.04491 is directly comparable; K in {0.5,1,2,4,8,16,32};
signed LP |a|<=K; both the settled and the handover pose are measured; every
activation vector is saved as an artifact so no number in the report is untraceable.

Budget: 1 model load, <=300 forwards, 0 body integration, 0 neural steps.
No geometry, Fmax, passive-bias, length-range or neural change is made here.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog

REB = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-rebuild/20261028T113000Z')
S74 = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
S107 = Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-k-ladder/20261028T117000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
KS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def build_matrices(xml, q, weight, mapping, counts):
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
    return dict(m=m, d=d, A=A, base=base, J=J, L=Lrow, G=G, bias=d.qfrc_bias.copy(),
                passive=d.qfrc_passive.copy(), weight=weight, geoms=geoms, nc=J.shape[1] // 3, nl=Lrow.shape[1],
                names=[m.actuator(i).name for i in range(m.nu)])


def solve_K(P, H, K, inner=False):
    A, base, J, L, bias, passive, weight = P['A'], P['base'], P['J'], P['L'], P['bias'], P['passive'], P['weight']
    n = H.shape[1]; nc = P['nc']; nl = P['nl']
    B = np.column_stack([A @ H, J, L]) / weight
    y = (bias - passive - base) / weight
    rws = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
    for j in range(nc):
        mr = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mr[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rws.append(mr); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bnd = [(-K, K)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rws), b_ub=np.concatenate(rhs), bounds=bnd, method='highs')
    if not sol.success:
        return dict(K=K, residual=None, success=False, message=sol.message)
    x = sol.x
    return dict(K=K, residual=float(x[-1]), success=True, activation=x[:n].copy())


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    weight = float(json.loads((S74 / 'GAINS.json').read_text())['weight'])
    mapping = json.loads((S74 / 'MAPPING.json').read_text())
    q_settled = np.load(S74 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    q_hand = np.load(S107 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Signed activation cap ladder |a|<=K at a fixed pose. The smallest K with residual<=1e-6 bounds how many contralateral muscle pathways would be needed.',
        budget=dict(model_load=1, forward=300, body_step=0, neural_step=0),
        Ks=list(KS),
        poses=dict(settled=str(S74 / 'HANDOVER_BEFORE.npz'), handover=str(S107 / 'HANDOVER_BEFORE.npz')),
        comparable_baseline=0.044910249883673,
        note='Only the SIGNED ladder defines K*; the nonnegative ladder plateaus at 0.0367 and is not used as a criterion.',
        branches=[
            'K* <= 1: build one contralateral pathway per muscle with the already-authorised geometry rule; LF 15 native muscles must stay byte-identical; require 66-pool AND 90-pool outer residual <= 1e-6 and report both',
            '1 < K* <= 2: allowed, but the report must state the required number of contralateral pathways and give both single- and double-pathway LP sets',
            'K* > 2: stop the geometry line; use a uniform engineering calibration with factor K* scaling only gainprm (Fmax), never the passive bias, length ranges or neural mapping; that changes LF native parameters and breaks the previous byte-identical claim, which must be stated; report same-factor 90-pool and unscaled negative results; 66-pool <= 1e-6 required to pass'],
        source_hashes={str(p): sha(p) for p in [Path(__file__), REB / 'MODEL_REBUILT.xml', S74 / 'HANDOVER_BEFORE.npz',
                                                S107 / 'HANDOVER_BEFORE.npz', S74 / 'GAINS.json', S74 / 'MAPPING.json']}))
    results = {}
    artifacts = {}
    for tag, q in (('settled', q_settled), ('handover', q_hand)):
        P = build_matrices(REB / 'MODEL_REBUILT.xml', q, weight, mapping, counts)
        rows = []
        saved = {}
        for K in KS:
            counts['lps'] += 1
            r = solve_K(P, P['G'], K)
            d = dict(K=K, residual=r['residual'],
                     activation_min=float(r['activation'].min()) if r['success'] else None,
                     activation_max=float(r['activation'].max()) if r['success'] else None,
                     feasible=bool(r['residual'] is not None and r['residual'] <= 1e-6))
            rows.append(d)
            if r['success']:
                saved[f'K_{K}'] = r['activation']
            print(json.dumps(dict(pose=tag, **d)), flush=True)
        # bounded reference for comparability
        counts['lps'] += 1
        rb = solve_K(P, P['G'], 1.0)
        np.savez_compressed(O / f'ACTIVATION_VECTORS_{tag}.npz', **saved)
        artifacts[tag] = sorted(saved)
        results[tag] = dict(rows=rows, bounded_0_1_residual=rb['residual'],
                            contacts=P['nc'], limit_rows=P['nl'],
                            K_star=next((d['K'] for d in rows if d['residual'] is not None and d['residual'] <= 1e-6), None))
    dump(O / 'K_LADDER.json', dict(results=results, artifacts=artifacts,
                                   K_star=results['settled']['K_star'],
                                   K_star_handover=results['handover']['K_star']))
    # which muscles need the largest negative activation at the largest K (traceable source of -17.76)
    P = build_matrices(REB / 'MODEL_REBUILT.xml', q_settled, weight, mapping, counts)
    counts['lps'] += 1
    big = solve_K(P, P['G'], 32.0)
    vec = big['activation']
    order = np.argsort(vec)[:8]
    dump(O / 'MOST_NEGATIVE_ACTIVATIONS.json', dict(
        K=32.0, activation_min=float(vec.min()), activation_max=float(vec.max()),
        values=[dict(index=int(i), name=P['names'][i], activation=float(vec[i])) for i in order],
        note='Signed LP allows negative activation, which a muscle cannot produce; these entries are what a contralateral pathway would have to supply.'))
    np.savez_compressed(O / 'K32_ACTIVATION_SETTLED.npz', **{'activation': vec, 'names': np.array(P['names'])})
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
    (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(K_star=results['settled']['K_star'], handover=results['handover']['K_star'],
                          settled_rows=results['settled']['rows']), indent=2))


if __name__ == '__main__':
    main()
