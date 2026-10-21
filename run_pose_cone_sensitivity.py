"""Static-LP and friction-cone sensitivity for one or more saved poses, side by side.

For every declared pose this script
  * rebuilds the ideal-contact static system with the engine itself (one base forward plus one
    forward per muscle, exactly the construction the canonical gate uses) and asserts the model
    invariants the gate relies on;
  * describes the contact set and the geometric margins with mj_kinematics + mj_geomDistance;
  * writes the same npz layout as the canonical gate so the archived exact-certificate tool can be
    pointed at these directories;
  * reports, per pose: declared-cone LP status (diamond / outer / union, native and zero floor),
    the scaled-infinity-norm distance to feasibility with one constraint family relaxed at a time,
    and the required muscle-force multiplier s as a function of inscribed Coulomb-cone resolution.

Scope: static relaxation level only. Nothing here is standing, a MuJoCo equilibrium or biology, and
no pose is modified.
"""
from pathlib import Path
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
FLOOR_GEOM = 0
KS = (4, 8, 16, 32, 64)
TAN_HALF_DEN = 10 ** 6
DIST_CAP = 4.0
NEW_POSE_GLOB = 'S2-seated-tarsus-only-*'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newest_pose():
    cands = sorted(p for p in DESK.glob(NEW_POSE_GLOB) if (p / 'candidate_seated_pose.npz').exists())
    if not cands:
        raise FileNotFoundError('no seated-pose candidate under %s' % (DESK / NEW_POSE_GLOB))
    return cands[-1] / 'candidate_seated_pose.npz'


def ellipse_rays(k):
    pts = set()
    for i in range(k):
        theta = 2 * math.pi * i / k
        if abs(math.sin(theta)) < 1e-12:
            pts.add((1.0 if math.cos(theta) > 0 else -1.0, 0.0))
        elif abs(math.cos(theta)) < 1e-12:
            pts.add((0.0, 1.0 if math.sin(theta) > 0 else -1.0))
        else:
            t = round(math.tan(theta / 2) * TAN_HALF_DEN) / TAN_HALF_DEN
            pts.add(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    dirs = sorted(pts)
    for p, q in dirs:
        if abs(p * p + q * q - 1.0) > 1e-15:
            raise ValueError('generated direction is not on the Coulomb circle')
    return dirs


def cone_block(contacts, dirs):
    cols = []
    for c in contacts:
        frame = c['frame']
        mu = c['mu']
        jac_t = np.asarray(c['J']).T          # contact dict stores (3, nv); generalized force needs nv x 3
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + p * t1 + q * t2))
    return np.asarray(cols).T


def row_scale(A, b):
    return np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))


def lp(A, b, lo, hi, scale=None):
    m = len(lo)
    scale = row_scale(A, b) if scale is None else scale
    return linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                   bounds=list(zip(lo, hi)) + [(0, None)] * (A.shape[1] - m), method='highs',
                   options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))


def elastic(A, b, muscle_bounds, contact_bounds):
    nv, n = A.shape
    scale = row_scale(A, b)
    S, bs = A / scale[:, None], b / scale
    rows, rhs = [], []
    for i in range(nv):
        row = np.zeros(n + 1)
        row[:n] = S[i]
        row[n] = -1.0
        rows.append(row)
        rhs.append(bs[i])
        other = -row
        other[n] = -1.0
        rows.append(other)
        rhs.append(-bs[i])
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                  bounds=list(muscle_bounds) + list(contact_bounds) + [(0, None)], method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return dict(t_star=(float(res.fun) if res.success else None), lp_status=int(res.status),
                max_scaled_residual=(float(np.max(np.abs(S @ res.x[:n] - bs))) if res.success else None))


S_CAP = 500.0


def threshold(muscle_block, cone_cols, b, lo, hi):
    def ok(s):
        Ak = np.hstack([muscle_block * s, cone_cols])
        return lp(Ak, b, lo, hi).status == 0
    feasible_at_one = ok(1.0)
    low, high = (0.5, 1.0) if feasible_at_one else (1.0, 2.0)
    while high <= S_CAP and not ok(high):
        high *= 1.25
    if high > S_CAP:
        return dict(feasible_at_s_1=feasible_at_one, threshold_bracket=None, infeasible_up_to_s=high / 1.25,
                    cap_reached=True, cap_max=S_CAP)
    for _ in range(24):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=feasible_at_one, infeasible_below=low, feasible_above=high,
                threshold_midpoint=0.5 * (low + high), cap_reached=False, cap_max=S_CAP)


def main():
    started = time.time()
    out = DESK / ('S2-pose-cone-sensitivity-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('static ideal-contact LP and friction-cone sensitivity of saved poses; '
                         'relaxation level only'),
                  declared_calls=dict(project_mj_steps=0))
    poses = {
        'new_seated_tarsus_only': dict(pose=newest_pose(), allow_nonfoot=False,
                                       note='new pose built under the declared tarsus-only contract'),
        'v6_seated': dict(pose=DESK / 'S2-keypoint-fit-v6-seated-20261015T105022/v6_candidate_seated_pose.npz',
                          allow_nonfoot=True,
                          note=('archived v6 seated candidate; the canonical gate REJECTS it for non-foot '
                                'active contact, so its contacts are kept here as a DECLARED RELAXATION and '
                                'this row must not be read as a gate result')),
        'canonical_declared': dict(pose=ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz',
                                   allow_nonfoot=False,
                                   note='canonical declared seated pose (reference; carries the archived certificate)'),
    }
    inputs = [Path(__file__)] + [Path(c['pose']) for c in poses.values()]
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        m = cp.load_model()
        d = mujoco.MjData(m)
        report['declared_calls']['project_model_loads'] = 1
        if (m.nv, m.nu) != (48, 90):
            raise ValueError('unexpected model dimensions')
        if not np.all(m.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE)):
            raise ValueError('not native muscles')
        if np.any(m.actuator_forcelimited) or np.any(m.actuator_actlimited) or m.nplugin:
            raise ValueError('unsupported force coupling')
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
        other = [g for g in range(m.ngeom) if g != FLOOR_GEOM and 'Tarsus' not in gname(g)]
        report['declared_model'] = dict(cone_flag=int(m.opt.cone), nv=int(m.nv), nu=int(m.nu),
                                        geom_condim=sorted(set(int(c) for c in m.geom_condim)))

        def geom_dist(geom):
            return float(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, geom, DIST_CAP, np.zeros(6)))

        per_pose = {}
        for label, cfg in poses.items():
            pose_path = Path(cfg['pose'])
            with np.load(pose_path) as z:
                pose = z['projected_seated_qpos'].copy()
            if pose.shape != (m.nq,):
                raise ValueError('bad pose shape for %s' % label)
            d.qpos[:] = pose
            d.qvel[:] = 0
            d.act[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            report['declared_calls']['project_forward_calls'] = report['declared_calls'].get(
                'project_forward_calls', 0) + 1
            mujoco.mj_kinematics(m, d)
            clearances = {gname(g): geom_dist(g) for g in other}
            leg_dist = {}
            for pref in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
                gs = [g for g in foot if gname(g).startswith(pref)]
                leg_dist[pref] = float(min(geom_dist(g) for g in gs))
            contacts = []
            for c in d.contact:
                if c.efc_address < 0:
                    continue
                if c.dim != 3:
                    raise ValueError('requires condim3 contacts')
                jac = np.zeros((3, m.nv))
                for geom, sign in ((c.geom2, 1), (c.geom1, -1)):
                    body = int(m.geom_bodyid[geom])
                    jp, jr = np.zeros_like(jac), np.zeros_like(jac)
                    mujoco.mj_jac(m, d, jp, jr, c.pos, body)
                    jac += sign * jp
                contacts.append(dict(index=len(contacts), geom1=int(c.geom1), geom2=int(c.geom2),
                                     names=[gname(int(c.geom1)), gname(int(c.geom2))],
                                     dist=float(c.dist), frame=np.asarray(c.frame).reshape(3, 3).copy(),
                                     mu=np.asarray(c.friction)[:2].copy(), J=jac))
            nonfoot = [c for c in contacts if not any('Tarsus' in n for n in c['names'])]
            if nonfoot and not cfg['allow_nonfoot']:
                raise ValueError('%s has %d non-foot contacts but the declared gate would reject it'
                                 % (label, len(nonfoot)))
            base = d.qfrc_actuator.copy()
            passive = d.qfrc_passive.copy()
            bias = d.qfrc_bias.copy()
            target = bias - passive - base
            R = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d.act[:] = 0
                d.act[i] = 1
                mujoco.mj_forward(m, d)
                report['declared_calls']['project_forward_calls'] += 1
                R[:, i] = d.qfrc_actuator - base
            d.act[:] = 0
            mujoco.mj_forward(m, d)
            report['declared_calls']['project_forward_calls'] += 1
            lo = m.actuator_ctrlrange[:, 0].copy()
            hi = m.actuator_ctrlrange[:, 1].copy()
            mu_set = sorted({tuple(np.round(c['mu'], 12)) for c in contacts})
            blocks = dict(diamond=[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
                          outer=[(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)],
                          union=[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                 (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)])
            entry = dict(pose=str(pose_path), pose_sha256=hashes[str(pose_path)], allow_nonfoot=bool(cfg['allow_nonfoot']),
                         note=cfg['note'], root_z_mm=float(pose[2]),
                         contact_count=len(contacts), nonfoot_contact_count=len(nonfoot),
                         tarsus_contact_count=len(contacts) - len(nonfoot),
                         contact_mu_values=mu_set,
                         per_leg_min_tarsus_distance_mm=leg_dist,
                         min_nonfoot_clearance_mm=float(min(clearances.values())),
                         min_nonfoot_clearance_geom=min(clearances, key=clearances.get),
                         closest_5_nonfoot_geoms=sorted(({'geom': k, 'clearance_mm': v}
                                                         for k, v in clearances.items()),
                                                        key=lambda r: r['clearance_mm'])[:5],
                         lp={}, elastic={}, cone_resolution={}, threshold={})
            pose_dir = out / ('matrices_' + label)
            pose_dir.mkdir(exist_ok=False)
            for floor_kind, lo_use in (('native', lo), ('zero', np.zeros(m.nu))):
                for kind, dirs in blocks.items():
                    Ak = np.hstack([R, cone_block(contacts, dirs)])
                    scale = row_scale(Ak, target)
                    np.savez_compressed(pose_dir / ('%s_%s.npz' % (floor_kind, kind)), A=Ak,
                                        target=target, scale=scale, lo=lo_use, hi=hi)
                    res = lp(Ak, target, lo_use, hi)
                    entry['lp']['%s_%s' % (floor_kind, kind)] = dict(
                        status=int(res.status),
                        classification=('RELAXED_WITNESS' if res.status == 0
                                        else 'NUMERICALLY_INFEASIBLE' if res.status == 2 else 'INDETERMINATE'))
            (pose_dir / 'metadata.json').write_text(json.dumps(dict(
                nv=int(m.nv), nu=int(m.nu), ncon=int(d.ncon), limit_count=0,
                active_contacts=[dict(index=c['index'], geom1=c['geom1'], geom2=c['geom2'],
                                      dist=c['dist'], frame=c['frame'].tolist(),
                                      mu=c['mu'].tolist(), J=c['J'].tolist()) for c in contacts],
                mu_set=[list(v) for v in mu_set], pose=str(pose_path)), indent=2) + '\n')
            entry['matrix_dir'] = str(pose_dir)
            A_decl = np.hstack([R, cone_block(contacts, blocks['diamond'])])
            box = list(zip(lo, hi))
            entry['elastic'] = dict(
                declared_box_and_one_sided_cone=elastic(A_decl, target, box, [(0, None)] * (A_decl.shape[1] - m.nu)),
                muscle_box_relaxed=elastic(A_decl, target, [(None, None)] * m.nu,
                                           [(0, None)] * (A_decl.shape[1] - m.nu)),
                contact_signs_relaxed=elastic(A_decl, target, box, [(None, None)] * (A_decl.shape[1] - m.nu)),
                all_free=elastic(A_decl, target, [(None, None)] * m.nu,
                                 [(None, None)] * (A_decl.shape[1] - m.nu)))
            entry['rank'] = dict(A=int(np.linalg.matrix_rank(A_decl, tol=1e-9)),
                                 augmented=int(np.linalg.matrix_rank(np.hstack([A_decl, target[:, None]]), tol=1e-9)),
                                 nv=int(m.nv))
            for k in KS:
                dirs = ellipse_rays(k)
                cols = cone_block(contacts, dirs)
                th = threshold(R, cols, target, lo, hi)
                entry['cone_resolution'][str(k)] = dict(cols=int(cols.shape[1]), rays=len(dirs), **th)
            per_pose[label] = entry
        report['poses'] = per_pose
        report['reading'] = (
            'All numbers are ideal static LP relaxation statements on each pose\'s own generated contact '
            'set. For v6_seated the non-foot contacts are retained as a DECLARED RELAXATION because the '
            'canonical gate rejects that pose; its row is a diagnostic, not a gate result. Congruent cone '
            'resolution thresholds across poses mean the cone-discretisation effect is a property of the '
            'declared pyramidal cone rather than of one particular pose.')
        report['status'] = 'PASS_POSE_CONE_SENSITIVITY_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for label, entry in per_pose.items():
            ks = sorted(int(k) for k in entry['cone_resolution'])
            ys = [entry['cone_resolution'][str(k)].get('feasible_above') for k in ks]
            axes[0].plot(ks, [np.nan if y is None else y for y in ys], 'o-', label=label)
        axes[0].axhline(1.0, color='k', ls=':')
        capped = [lbl for lbl, e in per_pose.items()
                  if all(v.get('cap_reached') for v in e['cone_resolution'].values())]
        if capped:
            axes[0].text(0.02, 0.97, 'cap reached (no threshold up to s=%.0f):\n%s'
                         % (S_CAP, ', '.join(capped)), transform=axes[0].transAxes, fontsize=7,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', fc='#ffe8e8', ec='#b03030'))
        axes[0].set_xscale('log', base=2)
        axes[0].set_xlabel('inscribed Coulomb-cone directions K (declared pyramidal = 4)')
        axes[0].set_ylabel('required muscle-force multiplier s')
        axes[0].set_title('cone resolution vs required strength')
        axes[0].legend(fontsize=8)
        labels = list(per_pose)
        keys = ['declared_box_and_one_sided_cone', 'muscle_box_relaxed', 'contact_signs_relaxed', 'all_free']
        width = 0.2
        for i, k in enumerate(keys):
            vals = []
            for lbl in labels:
                v = per_pose[lbl]['elastic'][k]['t_star']
                vals.append(0.0 if v is None else v)
            axes[1].bar(np.arange(len(labels)) + (i - 1.5) * width, vals, width, label=k.replace('_', ' ')[:28])
        axes[1].set_xticks(np.arange(len(labels)))
        axes[1].set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=8)
        axes[1].set_ylabel('scaled inf-norm distance to feasibility')
        axes[1].set_title('obstruction by constraint family')
        axes[1].legend(fontsize=7)
        x = np.arange(len(labels))
        tf = [per_pose[l]['tarsus_contact_count'] for l in labels]
        nf = [per_pose[l]['nonfoot_contact_count'] for l in labels]
        axes[2].bar(x, tf, 0.45, label='tarsus contacts', color='#3060b0')
        axes[2].bar(x, nf, 0.45, bottom=tf, label='non-foot contacts', color='#b03030')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=8)
        axes[2].set_ylabel('active contacts')
        axes[2].set_title('contact composition (v6 kept as declared relaxation)')
        ax2 = axes[2].twinx()
        ax2.plot(x, [per_pose[l]['min_nonfoot_clearance_mm'] for l in labels], 'k^--', label='min non-foot clearance')
        ax2.set_ylabel('min non-foot clearance (mm)')
        ax2.legend(fontsize=7, loc='center right')
        axes[2].legend(fontsize=7, loc='upper left')
        fig.suptitle('Static LP + friction-cone sensitivity per pose - relaxation level only, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'pose-cone-sensitivity.png', dpi=130)
        plt.close(fig)
        report['wall_seconds'] = time.time() - started
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
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        for label, entry in report['poses'].items():
            print('%-24s contacts %2d (nonfoot %2d) minclear %.4f mm  diamond native %s  thr K=4 %s K=32 %s' % (
                label, entry['contact_count'], entry['nonfoot_contact_count'],
                entry['min_nonfoot_clearance_mm'], entry['lp']['native_diamond']['status'],
                entry['cone_resolution']['4'].get('feasible_above'),
                entry['cone_resolution']['32'].get('feasible_above')))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
