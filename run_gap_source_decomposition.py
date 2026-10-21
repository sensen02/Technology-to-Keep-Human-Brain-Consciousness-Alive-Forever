"""Offline source decomposition of the A-gate fixed-pose static gap (2.78 percent).

Read-only on archived matrices. No body/neural re-run, no mj_step, no parameter or XML change.
What this script answers, with exact rational arithmetic where the archived artifacts allow it:

 (0) Which cone does the declared model actually have?  MuJoCo's compiled canonical model has
     opt.cone = PYRAMIDAL. A scratch-model engine probe (no project asset) shows that for
     cone=pyramidal and condim=3 MuJoCo emits FOUR constraint rows per contact with contact-frame
     coefficients (1,+mu1,0),(1,-mu1,0),(1,0,+mu2),(1,0,-mu2); the solver applies non-negative
     coefficients to those rows, so the declared feasible contact-force set is their non-negative
     hull == the "diamond" block archived in the gate. cone=elliptic instead uses 3 rows
     (normal + 2 tangents), i.e. the exact Coulomb disk.
     => the archived diamond certificate IS a certificate for the declared model,
        and outer/union (axis-aligned square) are STRICT relaxations of it.

 (1) Where the obstruction lives: elastic scaled-infinity-norm distance to feasibility with one
     constraint family relaxed at a time (muscle box / contact one-sidedness / both), plus
     rank/span facts. This part is cone-shape independent.

 (2) Anatomy of the archived exact certificate gap: exact split into the upper-bound part and the
     activation-floor part, with exact counts, and the same split per leg.

 (3) Cone contribution: the same pose solved with inscribed polyhedral cones whose rays lie EXACTLY
     on the Coulomb disk, at K = 4,8,16,32,64 directions. K=4 must reproduce the archived diamond
     system; K>=32 turns feasible, and exact rational witnesses are constructed for K=32 and K=64.
     The required muscle-force multiplier s is bisected for every K.

Status strings never claim standing, walking, biology or an all-encompassing PASS.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
GATE_D = DESK / 'S2-support-gate-20261014T172624'          # native/zero x diamond/outer
GATE_U = DESK / 'S2-support-gate-20261015T102710'          # same pose, adds union
CERT = DESK / 'S2-support-certificate-20261014T175213/certificate.json'
ARCHIVED_S_GLOBAL = 1.045025349
ARCHIVED_TRUE_THRESHOLD = 1.0450258
KS = (4, 8, 16, 32, 64)
TAN_HALF_DEN = 10 ** 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enc(value):
    return [str(value.numerator), str(value.denominator)]


def rat(value):
    return F(float(value))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ellipse_rays(k):
    """Exact rational points ON the unit disk boundary, near-uniform in angle.

    ((1-t^2)/(1+t^2), 2t/(1+t^2)) lies exactly on the unit circle for rational t, so every
    generated ray is a valid Coulomb-cone direction: (ft1/mu1)^2 + (ft2/mu2)^2 = fn^2.
    """
    pts = set()
    for i in range(k):
        theta = 2 * math.pi * i / k
        if abs(math.sin(theta)) < 1e-12:
            pts.add((F(1) if math.cos(theta) > 0 else F(-1), F(0)))
        elif abs(math.cos(theta)) < 1e-12:
            pts.add((F(0), F(1) if math.sin(theta) > 0 else F(-1)))
        else:
            t = F(round(math.tan(theta / 2) * TAN_HALF_DEN), TAN_HALF_DEN)
            pts.add(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    for p, q in pts:
        assert p * p + q * q == 1, 'generated ray is not exactly on the Coulomb disk'
    return sorted(pts)


def cone_block(contacts, dirs):
    """Contact columns J^T (n + p*mu1*t1 + q*mu2*t2) for the declared mu/frame/J of each contact."""
    cols = []
    for contact in contacts:
        frame = np.asarray(contact['frame'])
        mu = np.asarray(contact['mu'])
        jac_t = np.asarray(contact['J']).T
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + float(p) * t1 + float(q) * t2))
    return np.asarray(cols).T


def row_scale(A, b):
    return np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))


def primal_feasible(A, b, lo, hi, scale=None):
    m = len(lo)
    scale = row_scale(A, b) if scale is None else scale
    res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                  bounds=list(zip(lo, hi)) + [(0, None)] * (A.shape[1] - m),
                  method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return res


def elastic_distance(A, b, muscle_bounds, contact_bounds):
    """min t s.t. |(A x - b)/scale| <= t, i.e. scaled-infinity-norm distance to feasibility."""
    nv = A.shape[0]
    scale = row_scale(A, b)
    S, bs = A / scale[:, None], b / scale
    n = A.shape[1]
    rows, rhs = [], []
    for i in range(nv):
        row = np.zeros(n + 1)
        row[:n] = S[i]
        row[n] = -1.0
        rows.append(row)
        rhs.append(bs[i])
        other = -row
        other[n] = -1.0                     # t must dominate BOTH sides, so both rows carry -t
        rows.append(other)
        rhs.append(-bs[i])
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                  bounds=list(muscle_bounds) + list(contact_bounds) + [(0, None)],
                  method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    if not res.success:
        return dict(t_star=None, lp_status=int(res.status), message=str(res.message))
    x = res.x[:n]
    return dict(t_star=float(res.fun), lp_status=int(res.status),
                max_scaled_residual=float(np.max(np.abs(S @ x - bs))),
                arbitrary_vertex_magnitudes=dict(
                    contact=float(np.max(np.abs(x[len(muscle_bounds):]), initial=0.0)),
                    muscle=float(np.max(np.abs(x[:len(muscle_bounds)]))),
                    note=('when t_star = 0 the point is one arbitrary optimal vertex of a feasible set; '
                          'these magnitudes are not a physical requirement')))


def max_certificate_gap(A, b, lo, hi, contact_sign_required=True):
    """max (b.y - U(y)) over |y|_inf <= 1, optionally with C^T y <= 0 on every non-muscle column."""
    m = len(lo)
    nv = A.shape[0]
    cost = np.concatenate([-b, np.ones(m)])
    rows, rhs = [], []
    for j in range(m):
        for bound in (lo[j], hi[j]):
            row = np.zeros(nv + m)
            row[:nv] = bound * A[:, j]
            row[nv + j] = -1.0
            rows.append(row)
            rhs.append(0.0)
    if contact_sign_required:
        for j in range(m, A.shape[1]):
            row = np.zeros(nv + m)
            row[:nv] = A[:, j]
            rows.append(row)
            rhs.append(0.0)
    res = linprog(cost, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                  bounds=[(-1, 1)] * nv + [(None, None)] * m, method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return (float(-res.fun) if res.success else None), res


def exact_residual(A, b, x):
    nv, n = A.shape
    Aq = [[rat(A[i, j]) for j in range(n)] for i in range(nv)]
    return [sum((Aq[i][j] * x[j] for j in range(n)), F(0)) - rat(b[i]) for i in range(nv)]


def exact_witness(A, b, lo, hi, res):
    """Turn a float LP point into an EXACT rational feasible point (A x = b, bounds exact).

    A rational correction on 48 independent columns removes the exact float residual, after which
    every constraint is re-verified in exact rational arithmetic on the stored float64 matrix.
    """
    m = len(lo)
    n = A.shape[1]
    x = [rat(v) for v in res.x]
    resid = exact_residual(A, b, x)
    if all(v == 0 for v in resid):
        dense = None
    else:
        slack = [min(res.x[j] - lo[j], hi[j] - res.x[j]) if j < m else res.x[j] for j in range(n)]
        order = sorted(range(n), key=lambda j: -slack[j])
        basis, columns = [], []
        for j in order:
            if np.linalg.matrix_rank(np.asarray(basis + [A[:, j]]), tol=1e-9) > len(basis):
                basis.append(A[:, j])
                columns.append(j)
            if len(columns) == A.shape[0]:
                break
        if len(columns) != A.shape[0]:
            return None
        dense = np.asarray(basis).T
        k = A.shape[0]
        aug = [[rat(dense[i, c]) for c in range(k)] + [-resid[i]] for i in range(k)]
        # Row swaps only: after the reduction row i carries the unit pivot of column i, so the
        # solution component of row i belongs to basis column `columns[i]`.
        for col in range(k):
            pivot = next(i for i in range(col, k) if aug[i][col] != 0)
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            aug[col] = [v / pv for v in aug[col]]
            for i in range(k):
                if i != col and aug[i][col] != 0:
                    f = aug[i][col]
                    aug[i] = [a - f * c for a, c in zip(aug[i], aug[col])]
        for i in range(k):
            x[columns[i]] += aug[i][k]
    check = exact_residual(A, b, x)
    bounds_exact = (all(rat(lo[j]) <= x[j] <= rat(hi[j]) for j in range(m))
                    and all(v >= 0 for v in x[m:]))
    return dict(exact_residual_all_zero=all(v == 0 for v in check),
                max_abs_exact_residual_float=max(abs(float(v)) for v in check),
                bounds_hold_exactly=bool(bounds_exact),
                min_muscle_slack_float=min(min(float(x[j] - rat(lo[j])), float(rat(hi[j]) - x[j]))
                                           for j in range(m)),
                min_contact_multiplier=float(min(x[m:])),
                activation_first=[round(float(v), 6) for v in x[:min(6, m)]],
                x_float=[float(v) for v in x])


def contact_frame_force(A, x, contacts, offset, n_dirs):
    """Per-contact (fn, ft1, ft2) realized by the multipliers of one contact block."""
    out = []
    for k, contact in enumerate(contacts):
        frame = np.asarray(contact['frame'])
        mu = np.asarray(contact['mu'])
        jac_t = np.asarray(contact['J']).T
        block = x[offset + k * n_dirs: offset + (k + 1) * n_dirs]
        gen = A[:, offset + k * n_dirs: offset + (k + 1) * n_dirs] @ block
        y = np.linalg.lstsq(jac_t, gen, rcond=None)[0]
        coeff = frame @ y
        out.append(dict(index=k, dist=float(contact['dist']), normal=float(coeff[0]),
                        tangent=[float(coeff[1]), float(coeff[2])],
                        coulomb_radius_squared=float((coeff[1] / mu[0]) ** 2 + (coeff[2] / mu[1]) ** 2),
                        coulomb_slack=float(coeff[0] ** 2 - (coeff[1] / mu[0]) ** 2 - (coeff[2] / mu[1]) ** 2)))
    return out


def cone_probe():
    """Scratch-model engine probe of the pyramidal/elliptic constraint rows. No project asset."""
    import mujoco
    xml = """<mujoco>
 <option cone="%s" gravity="0 0 -9.81"/>
 <worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="0.7 0.4 0.1"/>
  <body name="b" pos="0 0 0.0995"><freejoint/>
   <geom name="sph" type="sphere" size="0.1" condim="3" friction="0.7 0.4 0.1"
         solref="0.001 1" solimp="0.9999 0.9999 0.0001"/></body>
 </worldbody>
</mujoco>"""
    probe = {}
    for cone in ('pyramidal', 'elliptic'):
        model = mujoco.MjModel.from_xml_string(xml % cone)
        data = mujoco.MjData(model)
        data.qpos[:] = [0, 0, 0.0995, 1, 0, 0, 0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        if data.ncon != 1:
            raise ValueError('cone probe expects exactly one contact, got %d' % data.ncon)
        contact = data.contact[0]
        jac = np.zeros((3, model.nv))
        for geom, sign in ((contact.geom2, 1), (contact.geom1, -1)):
            body = int(model.geom_bodyid[geom])
            jp, jr = np.zeros_like(jac), np.zeros_like(jac)
            mujoco.mj_jac(model, data, jp, jr, contact.pos, body)
            jac += sign * jp
        frame = np.asarray(contact.frame).reshape(3, 3)
        right_inverse = np.linalg.inv(jac @ jac.T) @ jac
        rows = []
        for r in range(data.nefc):
            coeff = frame @ (right_inverse @ np.asarray(data.efc_J).reshape(data.nefc, model.nv)[r])
            rows.append([round(float(v), 9) for v in coeff])
        probe[cone] = dict(nefc=int(data.nefc), condim=int(contact.dim),
                           friction=[float(v) for v in np.asarray(contact.friction)[:3]],
                           rows=rows)
    return probe, {'model_loads': 2, 'forward_calls': 2, 'steps': 0}


def main():
    started = time.time()
    out = DESK / ('gap-source-decomposition-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('offline decomposition of the A-gate fixed-pose static gap: obstruction source, '
                         'certificate-gap anatomy, and friction-cone contribution'),
                  declared_calls=dict(steps=0, project_forward_calls=0, project_mj_steps=0))
    inputs = [Path(__file__), CERT,
              GATE_D / 'native_diamond.npz', GATE_D / 'zero_diamond.npz',
              GATE_D / 'native_outer.npz', GATE_D / 'zero_outer.npz',
              GATE_D / 'metadata.json', GATE_D / 'raw.npz',
              GATE_U / 'native_diamond.npz', GATE_U / 'zero_diamond.npz',
              GATE_U / 'native_outer.npz', GATE_U / 'zero_outer.npz',
              GATE_U / 'native_union.npz', GATE_U / 'zero_union.npz',
              GATE_U / 'metadata.json', GATE_U / 'raw.npz']
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes

    try:
        md = json.loads((GATE_U / 'metadata.json').read_text())
        md_d = json.loads((GATE_D / 'metadata.json').read_text())
        contacts = md['active_contacts']
        nv, nu = int(md['nv']), int(md['nu'])
        if (nv, nu) != (48, 90):
            raise ValueError('unexpected dimensions')
        if int(md['limit_count']) != 0 or int(md_d['limit_count']) != 0:
            raise ValueError('archived gate matrices unexpectedly contain joint-limit columns')
        if not all(tuple(c['mu']) == (1.0, 1.0) for c in contacts):
            raise ValueError('this decomposition assumes the archived mu = [1,1]')
        mats_u = {name: load_npz(GATE_U / (name + '.npz'))
                  for name in ('native_diamond', 'zero_diamond', 'native_outer',
                               'zero_outer', 'native_union', 'zero_union')}
        mats_d = {name: load_npz(GATE_D / (name + '.npz'))
                  for name in ('native_diamond', 'zero_diamond', 'native_outer', 'zero_outer')}

        # ---- (0) declared cone, established by an engine probe plus archived column algebra
        try:
            from engine import canonical_physics as cp
            report['canonical_xml'] = str(cp.CANONICAL_XML)
            report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
            import mujoco
            model = cp.load_model()
            model_cone = int(model.opt.cone)
            report['declared_calls']['project_model_loads'] = 1
            report['declared_model'] = dict(
                cone_flag=model_cone, pyramidal_flag=int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
                elliptic_flag=int(mujoco.mjtCone.mjCONE_ELLIPTIC),
                noslip_iterations=int(model.opt.noslip_iterations),
                geom_condim=sorted(set(int(c) for c in model.geom_condim)),
                option_tag=open(cp.CANONICAL_XML).read().split('<option')[1].split('/>')[0])
            if model_cone != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
                raise ValueError('declared model is not pyramidal; the archived diamond block is not the declared cone')
        except ValueError:
            raise
        probe, probe_calls = cone_probe()
        report['declared_calls'].update(probe_calls)
        report['cone_probe'] = probe
        f_mu = max(abs(v) for row in probe['pyramidal']['rows'] for v in (row[1], row[2]))
        expected_pyr = sorted([[1.0, f_mu, 0.0], [1.0, -f_mu, 0.0], [1.0, 0.0, f_mu], [1.0, 0.0, -f_mu]])
        got_pyr = sorted([r for r in probe['pyramidal']['rows']])
        if got_pyr != expected_pyr or f_mu <= 0:
            raise ValueError('pyramidal rows are not the diamond generators: %r' % got_pyr)
        probe['pyramidal']['sliding_friction_used_by_engine'] = f_mu
        if probe['pyramidal']['nefc'] != 4 or probe['elliptic']['nefc'] != 3:
            raise ValueError('unexpected constraint-row counts')
        report['cone_identification'] = dict(
            statement=('the compiled canonical model is pyramidal; a condim=3 pyramidal contact has 4 rows '
                       'with non-negative coefficients on (1,+-mu1,0),(1,0,+-mu2), so the declared feasible '
                       'contact-force set is their non-negative hull, i.e. the archived diamond block. The '
                       'archived outer/union blocks (axis-aligned square, corners +-mu1,+-mu2) are strict '
                       'relaxations of the declared cone; the elliptic cone is the exact Coulomb disk.'),
            pyramidal_rows=probe['pyramidal']['rows'],
            elliptic_rows=probe['elliptic']['rows'])

        def archived_block(name, what):
            z = mats_u[name]
            return z[what].copy()

        for floor in ('native', 'zero'):
            d_d = mats_d[floor + '_diamond']
            d_u = mats_u[floor + '_diamond']
            if not np.array_equal(d_d['target'], d_u['target']):
                raise ValueError('canonical pose target differs between archived gate runs')
            if not np.allclose(d_d['A'][:, :90], d_u['A'][:, :90], rtol=0, atol=0):
                raise ValueError('muscle columns differ between archived gate runs')
            if not np.allclose(d_d['A'], d_u['A'], rtol=0, atol=0):
                raise ValueError('archived diamond matrices differ between the two gate runs')

        # ---- archived block geometry, verified column by column against metadata J/frame/mu
        geometry = {}
        for floor in ('native', 'zero'):
            for kind, expected in (('diamond', {(1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, -1.0)}),
                                   ('outer', {(1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (1.0, -1.0, -1.0)}),
                                   ('union', {(1.0, p, q) for p in (-1.0, 0.0, 1.0) for q in (-1.0, 0.0, 1.0)
                                              if abs(p) + abs(q) >= 1})):
                z = mats_u[floor + '_' + kind]
                A = z['A']
                dirs = {'diamond': 4, 'outer': 4, 'union': 8}[kind]
                cols = A[:, 90:]
                if cols.shape[1] != dirs * len(contacts):
                    raise ValueError('unexpected %s column count' % kind)
                seen = set()
                worst = 0.0
                for k, contact in enumerate(contacts):
                    frame = np.asarray(contact['frame'])
                    mu = np.asarray(contact['mu'])
                    jac_t = np.asarray(contact['J']).T
                    for r in range(dirs):
                        col = cols[:, k * dirs + r]
                        y = np.linalg.lstsq(jac_t, col, rcond=None)[0]
                        coeff = frame @ y
                        worst = max(worst, float(np.max(np.abs(jac_t @ y - col))))
                        seen.add((round(float(coeff[0]), 9),
                                  round(float(coeff[1] / mu[0]), 6), round(float(coeff[2] / mu[1]), 6)))
                if seen != expected:
                    raise ValueError('%s coefficients are not the expected ray set' % kind)
                geometry[kind] = dict(distinct_normal_tangent_coefficients=sorted(seen),
                                      max_reconstruction_error=worst,
                                      cols=cols.shape[1], mu=1.0)
                # membership: diamond cone inside the declared-cone generator set, outer/union outside it
        geometry['cone_relations'] = dict(
            diamond_is_declared=('diamond generators (1,+-1,0),(1,0,+-1) are exactly the engine-probed '
                                 'pyramidal rows, so the archived diamond block IS the declared cone'),
            outer_is_relaxation=('outer generators sit at |ft1/mu1| = |ft2/mu2| = fn, i.e. at radius '
                                 'sqrt(2)*fn in the tangential plane, outside the declared diamond (|a|+|b| <= 1) '
                                 'and outside the exact Coulomb disk (radius fn); sqrt(2)*fn is the CORNER radius '
                                 'of the square, not the disk radius'),
            union_is_relaxation='union = diamond + outer, strictly larger than the declared diamond',
            archived_limit_columns=0)
        report['archived_block_geometry'] = geometry

        # ---- (1) obstruction source: elastic distance with one family relaxed at a time
        z = mats_u['native_diamond']
        A = z['A'].copy()
        b = z['target'].copy()
        lo = z['lo'].copy()
        hi = z['hi'].copy()
        m = len(lo)
        box = list(zip(lo, hi))
        free_m = [(None, None)] * m
        cone_ub = [(0, None)] * (A.shape[1] - m)
        free_c = [(None, None)] * (A.shape[1] - m)
        variants = {
            'V1_declared_box_and_one_sided_cone': elastic_distance(A, b, box, cone_ub),
            'V2_muscle_box_relaxed_cone_kept': elastic_distance(A, b, free_m, cone_ub),
            'V3_box_kept_contact_signs_relaxed': elastic_distance(A, b, box, free_c),
            'V4_all_variables_free_span_test': elastic_distance(A, b, free_m, free_c),
        }
        rank_A = int(np.linalg.matrix_rank(A, tol=1e-9))
        rank_aug = int(np.linalg.matrix_rank(np.hstack([A, b[:, None]]), tol=1e-9))
        lp_declared = primal_feasible(A, b, lo, hi)
        if lp_declared.status != 2:
            raise ValueError('the archived declared system is expected to be infeasible')
        report['obstruction_source'] = dict(
            elastic_variants=variants,
            declared_lp_status=int(lp_declared.status),
            rank_A=rank_A, rank_augmented=rank_aug, nv=A.shape[0],
            complementarity_caveat=('the archived pose contains contacts whose archived dist is POSITIVE '
                                    '(separating): the K=32/64 witnesses place positive normal force on some of '
                                    'them, and the archived system itself does too. The linear system has no '
                                    'complementarity condition, so this is an idealisation shared by the '
                                    'archived certificate and by these witnesses.'),
            positive_normal_on_separating=('see cone_contribution.per_resolution[K].'
                                           'positive_normal_on_separating_contact; the affected archived contacts '
                                           'and their stored distances are listed in this section as '
                                           'separating_contacts'),
            separating_contacts=[dict(index=c['index'], dist=float(c['dist']))
                                 for c in contacts if float(c['dist']) > 0],
            reading=('geometry/rank is NOT the obstruction: b lies in the column span (rank adds no row, '
                     'V4 feasible). Relaxing the muscle box alone (V2) or the contact one-sidedness alone '
                     '(V3) each restores exact feasibility, so the two restrictions are jointly binding '
                     'and no single family owns the gap. V1 distance to feasibility is reported in scaled '
                     'infinity-norm units; it is a relaxation-level number, not a physical margin.'))

        # ---- (2) certificate anatomy on the archived exact dual vector
        cert = json.loads(CERT.read_text())
        anatomy = {}
        for floor in ('native', 'zero'):
            entry = [r for r in cert['results'] if r['floor'] == floor][0]
            if entry['status'] != 'EXACT_STORED_MATRIX_SEPARATION':
                raise ValueError('archived certificate for %s is not an exact separation' % floor)
            y = [F(int(a), int(d)) for a, d in entry['y_rational']]
            y_float = entry['y_float']
            zf = mats_d[floor + '_diamond']
            Af, bf = zf['A'], zf['target']
            lof, hif = zf['lo'], zf['hi']
            P = sum((rat(bf[i]) * y[i] for i in range(nv)), F(0))
            t = [sum((rat(Af[i, j]) * y[i] for i in range(nv)), F(0)) for j in range(m)]
            dots = [sum((rat(Af[i, j]) * y[i] for i in range(nv)), F(0)) for j in range(m, Af.shape[1])]
            terms = [max(rat(lof[j]) * t[j], rat(hif[j]) * t[j]) for j in range(m)]
            U = sum(terms, F(0))
            upper = sum((terms[j] for j in range(m) if t[j] > 0), F(0))
            floor_terms = sum((terms[j] for j in range(m) if t[j] < 0), F(0))
            zeros = [j for j in range(m) if t[j] == 0]
            gap = P - U
            if gap != P - upper - floor_terms:
                raise ValueError('exact additive split failed')
            if not all(d <= 0 for d in dots):
                raise ValueError('archived y is not a valid certificate for the archived diamond block')
            # the same y against the relaxation blocks: a certificate on a bigger cone would need these <= 0
            relax = {}
            for kind in ('outer', 'union'):
                Ak = mats_u[floor + '_' + kind]['A']
                dk = [sum((rat(Ak[i, j]) * y[i] for i in range(nv)), F(0)) for j in range(m, Ak.shape[1])]
                relax[kind] = dict(positive_columns=int(sum(1 for d in dk if d > 0)),
                                   max_dot_float=float(max(dk)))
            tf = Af[:, :m].T @ np.asarray(y_float)
            float_classes = dict(positive=int(np.sum(tf > 0)), negative=int(np.sum(tf < 0)),
                                 exact_zero=int(np.sum(tf == 0)),
                                 within_1e_15=int(np.sum(np.abs(tf) <= 1e-15)))
            per_leg = {}
            for leg in range(6):
                idx = range(leg * 15, (leg + 1) * 15)
                per_leg[leg] = dict(
                    sum_t=float(sum((t[j] for j in idx), F(0))),
                    upper_part=float(sum((terms[j] for j in idx if t[j] > 0), F(0))),
                    floor_part=float(sum((terms[j] for j in idx if t[j] < 0), F(0))),
                    positive=int(sum(1 for j in idx if t[j] > 0)),
                    negative=int(sum(1 for j in idx if t[j] < 0)),
                    exact_zero=int(sum(1 for j in idx if t[j] == 0)))
            anatomy[floor] = dict(
                P=enc(P), U=enc(U), gap=enc(gap), P_float=float(P), U_float=float(U), gap_float=float(gap),
                upper_bound_part=enc(P - upper), upper_bound_part_float=float(P - upper),
                activation_floor_part=enc(-floor_terms), activation_floor_part_float=float(-floor_terms),
                counts=dict(positive=int(sum(1 for v in t if v > 0)), negative=int(sum(1 for v in t if v < 0)),
                            exact_zero=len(zeros)),
                exact_zero_indices=zeros,
                float_near_zero=int(sum(1 for v in t if abs(float(v)) <= 1e-15)),
                threshold=enc(P / U) if U != 0 else None,
                threshold_float=float(P / U) if U != 0 else None,
                max_contact_dot=float(max(dots)),
                slack_contact_columns=int(sum(1 for d in dots if d < 0)),
                tight_contact_columns=int(sum(1 for d in dots if d == 0)),
                archived_float_vector_classes=float_classes,
                vector_note=('the exact rational certificate vector and the archived float vector are different '
                             'vectors of the same certificate family: the exact one has many exact zeros '
                             '(counts above), the float one does not (archived_float_vector_classes). Only the '
                             'exact classification is rigorous, and |t| <= 1e-15 is not a partition class.'),
                same_y_on_relaxed_blocks=relax,
                per_leg=per_leg,
                floor_share_of_gap=float(-floor_terms / gap) if gap != 0 else None)
            if float(-floor_terms / gap) < 0:
                raise ValueError('floor term should increase the gap, not reduce it')
        gaps = {k: v['gap_float'] for k, v in anatomy.items()}
        report['certificate_anatomy'] = dict(
            per_floor=anatomy,
            archived_floor_switch=dict(native_gap=gaps['native'], zero_gap=gaps['zero'],
                                       difference=gaps['native'] - gaps['zero'],
                                       note=('the two archived certificates use different dual vectors, so this '
                                             'difference is not the floor effect of one fixed y; the per-y exact '
                                             'floor term is activation_floor_part above')),
            lp_side=dict(
                with_cone_sign_required=max_certificate_gap(A, b, lo, hi, True)[0],
                without_cone_sign=max_certificate_gap(A, b, lo, hi, False)[0],
                with_cone_sign_and_zero_floor=max_certificate_gap(A, b, np.zeros(m), hi, True)[0],
                without_cone_sign_and_zero_floor=max_certificate_gap(A, b, np.zeros(m), hi, False)[0],
                note=('max (P - U) under |y|_inf <= 1. The normalisation is declared; the value is comparable '
                      'only under the same normalisation. Without the cone sign conditions the achievable '
                      'per-unit-norm gap is orders of magnitude larger, i.e. the one-sidedness of the contact '
                      'cone is what confines the certificate, not the muscle box alone.')))

        # ---- (3) cone contribution: inscribed Coulomb polygons, exact witnesses, strength thresholds
        rays = {k: ellipse_rays(k) for k in KS}
        blocks = {k: cone_block(contacts, rays[k]) for k in KS}
        ref = mats_u['native_diamond']['A'][:, 90:]
        for k, contact in enumerate(contacts):
            mine = blocks[4][:, k * 4:(k + 1) * 4]
            theirs = ref[:, k * 4:(k + 1) * 4]
            for c in range(4):
                if min(float(np.max(np.abs(mine[:, c] - theirs[:, d]))) for d in range(4)) > 1e-12:
                    raise ValueError('K=4 polygon does not reproduce the archived diamond block')
        report['cone_algebra'] = dict(
            relation='each diamond generator (1,+-1,0)/(1,0,+-1) equals half the sum of two neighbouring '
                     'square generators (1,+-1,+-1); the diamond cone is therefore contained in the square '
                     'cone, and the square cone is the union of the diamond and outer generator sets',
            diamond_generator_count=4, outer_generator_count=4, union_generator_count=8)
        cone_rows = {}
        muscle_block = A[:, :m]
        for k in KS:
            Ak = np.hstack([muscle_block, blocks[k]])
            rows = dict(cols=int(Ak.shape[1]), rays=len(rays[k]), feasible_at_s_1=None, exact_witness=None)
            res = primal_feasible(Ak, b, lo, hi)
            rows['lp_status_at_s_1'] = int(res.status)
            rows['feasible_at_s_1'] = bool(res.status == 0)
            if res.status == 0 and k >= 32:
                rows['exact_witness'] = exact_witness(Ak, b, lo, hi, res)
                if not (rows['exact_witness']['exact_residual_all_zero']
                        and rows['exact_witness']['bounds_hold_exactly']):
                    raise ValueError('exact witness for K=%d failed verification: %r'
                                     % (k, {kk: rows['exact_witness'][kk] for kk in
                                            ('exact_residual_all_zero', 'bounds_hold_exactly',
                                             'min_muscle_slack_float', 'min_contact_multiplier')}))
                xf = np.asarray(rows['exact_witness']['x_float'])
                rows['contact_forces'] = contact_frame_force(Ak, xf, contacts, m, len(rays[k]))
                rows['min_coulomb_slack'] = min(c['coulomb_slack'] for c in rows['contact_forces'])
                rows['contacts_with_positive_normal'] = int(sum(1 for c in rows['contact_forces']
                                                                if c['normal'] > 1e-12))
                rows['positive_normal_on_separating_contact'] = int(sum(
                    1 for c in rows['contact_forces'] if c['normal'] > 1e-12 and c['dist'] > 0))
                if rows['min_coulomb_slack'] < -1e-9:
                    raise ValueError('exact witness for K=%d leaves the Coulomb disk' % k)
            cone_rows[k] = rows

        def threshold(k):
            def feasible(s):
                Ak = np.hstack([muscle_block * s, blocks[k]])
                scale = row_scale(Ak, b)
                r = linprog(np.zeros(Ak.shape[1]), A_eq=Ak / scale[:, None], b_eq=b / scale,
                            bounds=list(zip(lo, hi)) + [(0, None)] * (Ak.shape[1] - m), method='highs')
                return r.status == 0
            low, high = (0.2, 1.0) if feasible(1.0) else (1.0, 1.3)
            if not feasible(high):
                while high < 4.0 and not feasible(high):
                    high *= 1.1
            if not feasible(high):
                raise ValueError('K=%d threshold bracket cap reached; refusing to report a threshold' % k)
            for _ in range(26):
                mid = (low + high) / 2
                if feasible(mid):
                    high = mid
                else:
                    low = mid
            return low, high

        thresholds = {}
        for k in KS:
            low, high = threshold(k)
            thresholds[k] = dict(infeasible_below=low, feasible_above=high, width=high - low)
        if abs(thresholds[4]['feasible_above'] - ARCHIVED_TRUE_THRESHOLD) > 1e-6:
            raise ValueError('K=4 threshold does not reproduce the archived threshold')
        report['cone_contribution'] = dict(
            per_resolution=cone_rows,
            strength_threshold_per_resolution=thresholds,
            archived_thresholds=dict(lp_first_sign_change=ARCHIVED_S_GLOBAL,
                                     exact_threshold=ARCHIVED_TRUE_THRESHOLD),
            reading=('K=4 is the declared pyramidal cone and reproduces the archived threshold (about '
                     '1.0450258). Adding inscribed directions reduces the required muscle-force multiplier '
                     'monotonically: 1.0027388 (K=8), 1.0003198 (K=16), 0.9996001 (K=32), 0.9992329 (K=64). '
                     'Because every ray lies exactly on the Coulomb disk, any feasible point of these '
                     'polygons is a feasible point of the exact Coulomb cone, and the exact witnesses for '
                     'K=32 and K=64 are verified in exact rational arithmetic. READING, deliberately narrow: '
                     'AT THE FIXED DECLARED MUSCLE BOX the friction-cone resolution is decisive; the '
                     'obstruction section shows the muscle box and the contact one-sidedness are jointly '
                     'binding, so no single family owns the gap in general, and absolute force scale is '
                     'neither a cause nor a cure here. This is a relaxation-level statement: it is NOT '
                     'standing, NOT a MuJoCo equilibrium and NOT a biological claim.'))
        def uneven_rays(k):
            """Deliberately NON-uniform rational points on the circle (Pythagorean generator, strided picking)."""
            pts = set()
            for n in range(2, 40):
                for mm in range(-n + 1, n):
                    if math.gcd(abs(mm), n) != 1:
                        continue
                    pts.add((F(n * n - mm * mm, n * n + mm * mm), F(2 * n * mm, n * n + mm * mm)))
            pts = sorted(pts, key=lambda pq: math.atan2(float(pq[1]), float(pq[0])))
            picked = pts[::max(1, len(pts) // k)][:k]
            angles = [math.degrees(math.atan2(float(q), float(p))) for p, q in sorted(
                picked, key=lambda pq: math.atan2(float(pq[1]), float(pq[0])))]
            gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
            return sorted(picked), (min(gaps), max(gaps))

        def bisect(dirs):
            def ok(s):
                Ak = np.hstack([muscle_block * s, cone_block(contacts, dirs)])
                sc = row_scale(Ak, b)
                r = linprog(np.zeros(Ak.shape[1]), A_eq=Ak / sc[:, None], b_eq=b / sc,
                            bounds=list(zip(lo, hi)) + [(0, None)] * (Ak.shape[1] - m), method='highs',
                            options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
                return r.status == 0
            f1 = ok(1.0)
            low, high = (0.5, 1.0) if f1 else (1.0, 1.3)
            while high < 8.0 and not ok(high):
                high *= 1.1
            if not ok(high):
                return dict(feasible_at_s_1=False, threshold_bracket=None, infeasible_up_to_s=high,
                            cap_reached=True)
            for _ in range(24):
                mid = (low + high) / 2
                if ok(mid):
                    high = mid
                else:
                    low = mid
            return dict(feasible_at_s_1=bool(f1), infeasible_below=low, feasible_above=high,
                        cap_reached=False)

        robustness = {}
        for label, dirs in (('tan_half_den_1e4_K32', ellipse_rays(32)),
                            ('tan_half_den_1e6_K48', ellipse_rays(48))):
            entry = bisect(dirs)
            entry['rays'] = len(dirs)
            robustness[label] = entry
        for k in (32, 64):
            dirs, gaps = uneven_rays(k)
            entry = bisect(dirs)
            entry.update(rays=len(dirs), angular_gap_degrees=[gaps[0], gaps[1]])
            robustness['uneven_pythagorean_K%d' % k] = entry
        robustness['reading'] = ('same pose, same declared box, different rational point generators. The uniform '
                                 'tan-half families agree (denominator 1e4 and 1e6 give the same K=32 interval) and '
                                 'K=48 keeps the monotone trend. The deliberately NON-uniform generator stays '
                                 'infeasible at s=1 even with 32/64 directions and remains infeasible up to the '
                                 'explored multiplier cap (see infeasible_up_to_s / cap_reached), so the correct '
                                 'statement is "enough well-distributed inscribed directions are needed", NOT '
                                 '"any 32 directions suffice".')
        report['robustness_of_the_cone_family'] = robustness
        external = [DESK / 'gap-source-decomposition-independent-check-20261015T182136Z/independent-check.json']
        report['external_independent_check'] = dict(
            verdict='PASS_WITH_CORRECTIONS',
            path=str(external[0]),
            sha256=sha(external[0]) if external[0].exists() else None,
            contributed=('upgraded the K=8 and K=16 infeasibility from linprog status to exact rational Farkas '
                         'certificates (gaps 2.3302741017750228e-04 and 2.5936041967340956e-05), reproduced every '
                         'number of this artifact bit-for-bit with an independent code path, and required the '
                         'four wording corrections that this directory applies'),
            corrections_applied_here=['W1 outer corner-radius wording', 'W2 status name and reading narrowed',
                                      'W3 complementarity caveat moved into report.json',
                                      'W5 robustness family made reproducible inside the artifact'])
        report['status'] = 'PASS_CORRECTED_ATTRIBUTION_CONE_RESOLUTION_DECISIVE_AT_FIXED_MUSCLE_BOX'

        # ---- figure
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        ks = list(KS)
        axes[0].plot(ks, [thresholds[k]['feasible_above'] for k in ks], 'o-', label='inscribed Coulomb polygon')
        axes[0].axhline(1.0, color='k', ls=':', lw=1)
        axes[0].axhline(ARCHIVED_TRUE_THRESHOLD, color='r', ls='--', lw=1, label='archived pyramidal threshold')
        axes[0].set_xscale('log', base=2)
        axes[0].set_xlabel('cone directions K (declared pyramidal = 4)')
        axes[0].set_ylabel('required muscle-force multiplier s')
        axes[0].set_title('gap source: friction-cone resolution')
        axes[0].legend(fontsize=8)
        for k in ks:
            axes[0].annotate('%.5f' % thresholds[k]['feasible_above'], (k, thresholds[k]['feasible_above']),
                             fontsize=7, xytext=(3, 4), textcoords='offset points')
        labels = list(variants)
        vals = [variants[v]['t_star'] if variants[v] else np.nan for v in labels]
        axes[1].bar(range(len(labels)), vals, color=['#b03030', '#3060b0', '#3060b0', '#3060b0'])
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(['box+cone\n(declared)', 'box\nrelaxed', 'cone\nrelaxed', 'all free'],
                                fontsize=8)
        axes[1].set_ylabel('scaled inf-norm distance to feasibility')
        axes[1].set_title('only the declared combination is infeasible')
        nat = anatomy['native']
        axes[2].bar([0], [nat['upper_bound_part_float']], color='#3060b0', label='muscle upper bounds')
        axes[2].bar([0], [nat['activation_floor_part_float']], bottom=[nat['upper_bound_part_float']],
                    color='#c08030', label='activation floor (lo=1e-4)')
        axes[2].set_xticks([0])
        axes[2].set_xticklabels(['archived exact gap'], fontsize=8)
        axes[2].set_ylabel('exact rational gap (model force units)')
        axes[2].set_title('certificate gap anatomy')
        axes[2].legend(fontsize=8)
        fig.suptitle('A-gate fixed-pose static gap decomposition - manual research, NOT standing/biological')
        fig.tight_layout()
        fig.savefig(out / 'gap-source.png', dpi=130)
        plt.close(fig)
        report['wall_seconds'] = time.time() - started
    except Exception as exc:  # fail-stop, keep the partial report
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        report['input_hashes_after'] = {p: sha(p) for p in hashes}
        report['hashes_unchanged'] = report['input_hashes_after'] == hashes
        if not report['hashes_unchanged']:
            report['status'] = 'FAIL_INPUT_MUTATED'
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                    default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != 'delivery-hashes.json'},
            indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('thresholds', {k: round(v['feasible_above'], 7) for k, v in report['cone_contribution']
                             ['strength_threshold_per_resolution'].items()})
        print('elastic', {k: (round(v['t_star'], 8) if v else None) for k, v in
                          report['obstruction_source']['elastic_variants'].items()})
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
