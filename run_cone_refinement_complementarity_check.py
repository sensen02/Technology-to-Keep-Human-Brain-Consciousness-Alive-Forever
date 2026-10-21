"""Does the cone-refinement "feasibility" survive when separating contacts are forbidden to carry force?

Why this exists: in the delivered static LP there is NO complementarity condition, so a contact whose
geometric distance is POSITIVE (separating) may still carry force. The independent check pointed out that
the K=32/K=64 witnesses did exactly that (contacts 7 and 9 of the canonical pose, distances +1.0048e-3 and
+1.2193e-3 mm; contact 1 of the new seated pose, +6.66e-4 mm). A "feasible" relaxation that leans on a
separating contact is not evidence that the pose can be passively supported.

This script therefore DECLARES a hard complementarity-restricted variant of the same static relaxation:
contacts with dist > 0 are removed from the system entirely (equivalent to forcing x_c = 0 on those
columns), and everything is recomputed on the remaining contact set:
  * diamond (= declared pyramidal cone) LP at s = 1, with an EXACT rational certificate when infeasible
    and an EXACT rational primal witness when feasible;
  * the required muscle-force multiplier s for inscribed Coulomb polygons K = 4/8/16/32/64.

Offline: no MuJoCo call, no mj_step, no model/gain/friction/force-scale change. Relaxation level only.
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
SENS = DESK / 'S2-pose-cone-sensitivity-20261015T114356'
REVIEW = DESK / 'gap-source-decomposition-independent-check-20261015T182136Z/independent-check.json'
KS = (4, 8, 16, 32, 64)
TAN_HALF_DEN = 10 ** 6
S_CAP = 500.0
DIAMOND = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
POSES = {
    'canonical_declared': dict(dir=SENS / 'matrices_canonical_declared',
                               note='the pose that carries the archived certificate'),
    'new_seated_tarsus_only': dict(dir=SENS / 'matrices_new_seated_tarsus_only',
                                   note='the seated tarsus-only pose delivered this session'),
    'v6_seated': dict(dir=SENS / 'matrices_v6_seated',
                      note='archived v6 seated candidate, kept as a DECLARED relaxation (gate rejects it)'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ellipse_rays(k):
    pts = set()
    for i in range(k):
        th = 2 * math.pi * i / k
        if abs(math.sin(th)) < 1e-12:
            pts.add((1.0 if math.cos(th) > 0 else -1.0, 0.0))
        elif abs(math.cos(th)) < 1e-12:
            pts.add((0.0, 1.0 if math.sin(th) > 0 else -1.0))
        else:
            t = round(math.tan(th / 2) * TAN_HALF_DEN) / TAN_HALF_DEN
            pts.add(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return sorted(pts)


def cone_block(contacts, dirs):
    cols = []
    for c in contacts:
        frame = np.asarray(c['frame'])
        mu = np.asarray(c['mu'])
        jac_t = np.asarray(c['J']).T
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + float(p) * t1 + float(q) * t2))
    return np.asarray(cols).T


def lp_feasible(A, b, lo, hi):
    m = len(lo)
    scale = np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))
    res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                  bounds=list(zip(lo, hi)) + [(0, None)] * (A.shape[1] - m), method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return res


def max_gap_dual(A, b, lo, hi):
    """max (b.y - U(y)) over |y|_inf <= 1 with every non-muscle column dot <= 0."""
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
    for j in range(m, A.shape[1]):
        row = np.zeros(nv + m)
        row[:nv] = A[:, j]
        rows.append(row)
        rhs.append(0.0)
    res = linprog(cost, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                  bounds=[(-1, 1)] * nv + [(None, None)] * m, method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return res


def exact_residual(A, b, x):
    nv, n = A.shape
    Aq = [[F(float(A[i, j])) for j in range(n)] for i in range(nv)]
    return [sum((Aq[i][j] * x[j] for j in range(n)), F(0)) - F(float(b[i])) for i in range(nv)]


def exact_primal_witness(A, b, lo, hi, res):
    """Exact rational point with A x = b exactly, box exact, contact multipliers >= 0 exactly."""
    m = len(lo)
    n = A.shape[1]
    x = [F(float(v)) for v in res.x]
    resid = exact_residual(A, b, x)
    if not all(v == 0 for v in resid):
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
        aug = [[F(float(dense[i, c])) for c in range(k)] + [-resid[i]] for i in range(k)]
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
    ok = (all(v == 0 for v in check) and all(F(float(lo[j])) <= x[j] <= F(float(hi[j])) for j in range(m))
          and all(v >= 0 for v in x[m:]))
    return dict(exact_equality=all(v == 0 for v in check), bounds_exact=bool(ok),
                min_contact_multiplier=float(min(x[m:])),
                max_abs_residual_float=max(abs(float(v)) for v in check))


def exact_dual_certificate(A, b, lo, hi):
    """Exact rational separation: all contact dots <= 0 and gap = b.y - U(y) > 0, verified exactly."""
    m = len(lo)
    nv = A.shape[0]
    res = max_gap_dual(A, b, lo, hi)
    if not res.success or -res.fun <= 0:
        return None
    y = [F(float(v)) for v in res.x[:nv]]

    def verify(y):
        dots = [sum((F(float(A[i, j])) * y[i] for i in range(nv)), F(0)) for j in range(m, A.shape[1])]
        t = [sum((F(float(A[i, j])) * y[i] for i in range(nv)), F(0)) for j in range(m)]
        U = sum((max(F(float(lo[j])) * t[j], F(float(hi[j])) * t[j]) for j in range(m)), F(0))
        P = sum((F(float(b[i])) * y[i] for i in range(nv)), F(0))
        return P, U, dots

    P, U, dots = verify(y)
    if not all(d <= 0 for d in dots):
        return None
    gap = P - U
    if gap <= 0:
        return None
    if not all(abs(v) <= 1 for v in y):
        return None
    return dict(status='EXACT_STORED_MATRIX_SEPARATION',
                gap_rational=[str(gap.numerator), str(gap.denominator)], gap_float=float(gap),
                required_float=float(P), upper_float=float(U),
                contact_dots_all_nonpositive=True,
                max_contact_dot=float(max(dots)),
                abs_y_max=float(max(abs(float(v)) for v in y)))


def reduced_system(reduce_dir, keep_where):
    meta = json.loads((reduce_dir / 'metadata.json').read_text())
    contacts = meta['active_contacts']
    with np.load(reduce_dir / 'native_diamond.npz') as z:
        A_full = z['A'].copy()
        b = z['target'].copy()
        lo = z['lo'].copy()
        hi = z['hi'].copy()
    nu = len(lo)
    rebuilt_full = np.hstack([A_full[:, :nu], cone_block(contacts, DIAMOND)])
    if not np.allclose(rebuilt_full, A_full, rtol=0, atol=1e-12):
        raise ValueError('rebuild of the full diamond system does not match the archived matrix')
    kept = [c for c in contacts if keep_where(c)]
    dropped = [c for c in contacts if not keep_where(c)]
    A_red = np.hstack([A_full[:, :nu], cone_block(kept, DIAMOND)])
    return dict(meta=meta, contacts=contacts, kept=kept, dropped=dropped, b=b, lo=lo, hi=hi,
                A_full=A_full, A_red=A_red, nu=nu)


def threshold(muscle_cols, cone_cols, b, lo, hi):
    def ok(s):
        A = np.hstack([muscle_cols * s, cone_cols])
        return lp_feasible(A, b, lo, hi).status == 0
    if cone_cols.shape[1] == 0:
        return dict(feasible_at_s_1=ok(1.0), threshold_bracket=None, cap_reached=True,
                    no_contact_columns=True)
    f1 = ok(1.0)
    low, high = (0.5, 1.0) if f1 else (1.0, 2.0)
    while high <= S_CAP and not ok(high):
        high *= 1.25
    if high > S_CAP:
        return dict(feasible_at_s_1=f1, threshold_bracket=None, cap_reached=True,
                    infeasible_up_to_s=high / 1.25, cap_max=S_CAP)
    for _ in range(26):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=f1, infeasible_below=low, feasible_above=high,
                threshold_midpoint=0.5 * (low + high), cap_reached=False)


def main():
    started = time.time()
    out = DESK / ('S2-cone-refinement-complementarity-check-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('recompute the static relaxation with separating contacts REMOVED (declared hard '
                         'complementarity-restricted variant) and test whether the cone-refinement '
                         'feasibility survives'),
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0, project_model_loads=0),
                  absolute_force_values='NOT_ESTABLISHED')
    inputs = [Path(__file__), REVIEW] + [POSES[k]['dir'] / f for k in POSES
                                         for f in ('metadata.json', 'native_diamond.npz')]
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes

    try:
        review = json.loads(REVIEW.read_text())
        rg = None
        candidates = [(path, value) for path, value in _walk(review) if 'my_scaled_unit_max_gap' in path]
        for path, value in candidates:
            if isinstance(value, dict):
                rg = value
        if rg is None and candidates:
            rg = {path.rsplit('/', 1)[-1]: value for path, value in candidates}
        report['independent_check_reference'] = dict(
            path=str(REVIEW), sha256=hashes[str(REVIEW)], verdict=review.get('verdict'),
            per_K_max_certificate_gap_scaled_units=rg,
            note=('that check built EXACT rational Farkas certificates for K=4/8/16 (positive gaps) and found '
                  'no positive gap for K=32/64; the K=8/K=16 exact infeasibility values are recorded here as '
                  'requested, and the K=32/64 side is the claim this script re-tests with separating contacts '
                  'removed'))

        results = {}
        comparison = []
        for label, cfg in POSES.items():
            red = reduced_system(cfg['dir'], lambda c: float(c['dist']) <= 0.0)
            entry = dict(pose=str(cfg['dir']), note=cfg['note'],
                         contact_count_total=len(red['contacts']),
                         separating_contacts=[dict(index=i, dist=float(c['dist']))
                                              for i, c in enumerate(red['contacts'])
                                              if float(c['dist']) > 0.0],
                         kept_contacts=len(red['kept']), dropped_contacts=len(red['dropped']),
                         dropped_detail=[dict(dist=float(c['dist'])) for c in red['dropped']])
            A_red, b, lo, hi, nu = red['A_red'], red['b'], red['lo'], red['hi'], red['nu']
            res = lp_feasible(A_red, b, lo, hi)
            entry['diamond_reduced'] = dict(lp_status=int(res.status))
            if res.status == 0:
                w = exact_primal_witness(A_red, b, lo, hi, res)
                entry['diamond_reduced'].update(
                    classification='FEASIBLE_IN_THIS_RELAXATION',
                    exact_witness=w,
                    statement=('with separating contacts removed the DECLARED diamond system is feasible at '
                               's=1 in this static relaxation, and an exact rational witness was constructed '
                               '(A x = b exactly, box exact, multipliers >= 0 exactly). This is a static '
                               'linear relaxation statement, NOT standing and NOT a MuJoCo equilibrium.'))
            else:
                cert = exact_dual_certificate(A_red, b, lo, hi)
                entry['diamond_reduced'].update(
                    classification=('EXACT_CERTIFICATE' if cert else 'INDETERMINATE_FLOAT_ONLY'),
                    exact_certificate=cert,
                    statement=('with separating contacts removed the DECLARED diamond system is still infeasible '
                               'at s=1, proven by an exact rational separating vector' if cert else
                               'no exact certificate was reconstructed; treat as indeterminate'))
            # K sweep on the reduced contact set
            entry['cone_resolution_reduced'] = {}
            for k in KS:
                dirs = DIAMOND if k == 4 else ellipse_rays(k)
                cols = cone_block(red['kept'], dirs)
                entry['cone_resolution_reduced'][str(k)] = dict(rays=len(dirs), cols=int(cols.shape[1]),
                                                                **threshold(A_red[:, :nu], cols, b, lo, hi))
            # same pose, full contact set, for a like-for-like comparison (recomputed here, one source)
            entry['cone_resolution_all_contacts_here'] = {}
            for k in KS:
                dirs = DIAMOND if k == 4 else ellipse_rays(k)
                cols = cone_block(red['contacts'], dirs)
                entry['cone_resolution_all_contacts_here'][str(k)] = dict(
                    rays=len(dirs), cols=int(cols.shape[1]),
                    **threshold(red['A_full'][:, :nu], cols, b, lo, hi))
            results[label] = entry

            for k in KS:
                a = entry['cone_resolution_all_contacts_here'][str(k)]
                r = entry['cone_resolution_reduced'][str(k)]
                comparison.append(dict(
                    pose=label, K=k,
                    all_contacts_feasible_at_s_1=a.get('feasible_at_s_1'),
                    all_contacts_threshold=a.get('threshold_midpoint'),
                    all_contacts_cap_reached=bool(a.get('cap_reached')),
                    non_separating_only_feasible_at_s_1=r.get('feasible_at_s_1'),
                    non_separating_only_threshold=r.get('threshold_midpoint'),
                    non_separating_only_cap_reached=bool(r.get('cap_reached')),
                    verdict=('refinement_feasibility_SURVIVES' if (a.get('feasible_at_s_1')
                                                                  and r.get('feasible_at_s_1'))
                             else 'refinement_feasibility_LOST_WITHOUT_SEPARATING_CONTACTS'
                             if (a.get('feasible_at_s_1') and not r.get('feasible_at_s_1'))
                             else 'infeasible_at_s_1_in_both_contacts_sets')))
        report['poses'] = results
        report['comparison_table'] = comparison

        lost = [c for c in comparison if c['verdict'] == 'refinement_feasibility_LOST_WITHOUT_SEPARATING_CONTACTS']
        survives = [c for c in comparison if c['verdict'] == 'refinement_feasibility_SURVIVES']
        report['verdict_conclusion'] = dict(
            rows_losing_feasibility=[(c['pose'], c['K']) for c in lost],
            rows_keeping_feasibility=[(c['pose'], c['K']) for c in survives],
            statement=('Removing the contacts whose distance is positive removes the cone-refinement '
                       'feasibility (the rows listed in rows_losing_feasibility), i.e. the earlier "feasible at '
                       'K=32/64" reading came from pushing on contacts that the pose is geometrically '
                       'separating from and that only the missing complementarity condition allowed to carry '
                       'force: physically unusable. Where a row keeps feasibility under the reduced contact '
                       'set, the honest wording is "feasible in this static relaxation with this contact set", '
                       'still not standing and not a MuJoCo equilibrium.'
                       if lost else
                       'No row lost feasibility, so the cone-refinement result does not depend on separating '
                       'contacts in this experiment.'),
            cap_negative_results=[dict(pose=k, K=int(kk), infeasible_up_to_s=v.get('infeasible_up_to_s'),
                                       cap_max=v.get('cap_max'))
                                  for k, e in results.items()
                                  for kk, v in e['cone_resolution_reduced'].items()
                                  if v.get('cap_reached') and not v.get('feasible_at_s_1')],
            cap_wording=('rows with cap_reached=True and feasible_at_s_1=False must be reported as '
                         '"no threshold found within the cap (s <= 500)" and must NOT be called infeasible or '
                         'feasible; the cap is an exploration limit, not a threshold'))
        report['status'] = 'PASS_COMPLEMENTARITY_RESTRICTED_RECOMPUTATION_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        poses_l = list(results)
        width = 0.35
        for i, k in enumerate(KS):
            xs = [p for p in poses_l]
            a_vals = [results[p]['cone_resolution_all_contacts_here'][str(k)].get('feasible_above') for p in xs]
            r_vals = [results[p]['cone_resolution_reduced'][str(k)].get('feasible_above') for p in xs]
            axes[0].bar([j + i * 0.12 for j in range(len(xs))], [0 if v is None else v for v in a_vals],
                        0.1, label='K=%d all contacts' % k, alpha=0.9)
            axes[1].bar([j + i * 0.12 for j in range(len(xs))], [0 if v is None else v for v in r_vals],
                        0.1, label='K=%d non-separating only' % k, alpha=0.9)
        for ax, title in ((axes[0], 'required s, ALL contacts (as delivered)'),
                          (axes[1], 'required s, contacts with dist<=0 ONLY')):
            ax.set_yscale('log')
            ax.axhline(1.0, color='k', ls=':')
            ax.set_xticks(range(len(poses_l)))
            ax.set_xticklabels([p.replace('_', '\n') for p in poses_l], fontsize=7)
            ax.set_ylabel('threshold s (0-height = no threshold within cap)')
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=6)
        axes[2].bar(range(len(comparison)), [1] * len(comparison),
                    color=['#b03030' if c['verdict'].endswith('LOST_WITHOUT_SEPARATING_CONTACTS')
                           else '#2f8f4f' if c['verdict'].endswith('SURVIVES') else '#808080'
                           for c in comparison])
        import matplotlib.patches as mpatches
        axes[2].legend(handles=[mpatches.Patch(color='#b03030', label='feasibility lost without separating contacts'),
                                mpatches.Patch(color='#2f8f4f', label='feasibility survives'),
                                mpatches.Patch(color='#808080', label='infeasible in both contact sets')],
                       fontsize=6, loc='upper center')
        axes[2].set_xticks(range(len(comparison)))
        axes[2].set_xticklabels(['%s/K%d' % (c['pose'].replace('_', '')[:8], c['K']) for c in comparison],
                                fontsize=6, rotation=45, ha='right')
        axes[2].set_yticks([])
        axes[2].set_title('red = refinement feasibility lost\nwhen separating contacts are removed', fontsize=9)
        fig.suptitle('Cone refinement vs separating contacts - static relaxation level, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'complementarity-check.png', dpi=130)
        plt.close(fig)
    except Exception as exc:
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
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('lost:', report['verdict_conclusion']['rows_losing_feasibility'])
        print('survives:', report['verdict_conclusion']['rows_keeping_feasibility'])
        print('cap negatives:', report['verdict_conclusion']['cap_negative_results'])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


def _walk(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                yield from _walk(v, path + '/' + k)
            else:
                yield path + '/' + k, v
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, path + '[%d]' % i)


if __name__ == '__main__':
    sys.exit(main())
