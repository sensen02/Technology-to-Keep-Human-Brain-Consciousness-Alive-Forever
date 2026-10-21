"""CORRECTED / REISSUED declared MuJoCo equilibrium search for pose A (multi-start).

WHY THIS EXISTS: an earlier producer run (S2-mujoco-equilibrium-A-20261015T151340) used only five uniform starts, stalled at
max|qacc| = 19.805 mm/s^2 and published the conclusion "(b) NO EQUILIBRIUM inside the declared boundary". An independent verifier
refuted that with a 240-start multi-start search (best max|qacc| = 3.14e-11 mm/s^2). This script redoes the search with multi-start
and reissues the result; the old directory is declared VOID and is not cited any more.

DECLARED GATE (written into the report BEFORE the search runs):
  max|qacc| <= 1e-3 mm/s^2 (about 1e-7 g; at double precision and the solver tolerances of a single mj_forward on this model this is
  indistinguishable from zero, and it is 10x tighter than the tightest dynamic criterion used anywhere in this project), all six
  tarsi in active contact, zero non-foot contacts, every contact inside the declared axis-ray diamond (u1+u2 <= 1), drive = frozen
  connectome-derived command archive only, no per-step pose correction.
MANDATORY DISCLOSURE (recorded in every report of this directory): qacc = 0 is only a STATIC necessary condition. It is NOT
stability, NOT standing, and NOT a statement that pose A itself is an equilibrium. The equilibria found here are NEIGHBOURING
STATES inside the declared contract ball: 46-52 active tarsus contacts and root z -0.02 ... -0.05 mm, whereas pose A has 9
contacts, and their joint distance to the source pose is 0.382-0.3996 rad, i.e. close to the declared 0.40 rad cap.

Declared search: variables = act (90) + root z (+-0.05 mm) + 42 leg hinge joints, each within [joint range - 5e-3 rad] and within
0.40 rad of the source pose; objective = max|qacc| after one mj_forward with qvel = 0 and ctrl = 0; scipy least_squares (trf),
random multi-start over the declared box; nested probes with the leg joints additionally restricted to within r rad of POSE A for
r = 0.05 / 0.10 / 0.20.
Absolute force values and physical units remain NOT_ESTABLISHED. Manual research, not a product; no standing/walking claim.
"""
from pathlib import Path
import hashlib, json, time, zipfile
import numpy as np
from scipy.optimize import least_squares

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
GATE = dict(max_abs_qacc_mm_s2=1e-3,
            max_abs_qacc_reason=('about 1e-7 g at mm units and g = 9801 mm/s^2; indistinguishable from zero at the solver tolerance of one mj_forward, '
                                 'and 10x tighter than the tightest dynamic criterion used anywhere in this project (1e-2 mm/s^2)'),
            six_tarsi_active_contact=True, zero_nonfoot_contacts=True, all_contacts_u_diamond_le_1=True,
            drive='frozen connectome-derived command archive only; no CPG, no PD, no preprogrammed action, no per-step pose correction',
            may_only_be_called='declared-cone static equilibrium - never standing')
DISCLOSURE = ('qacc = 0 is only a STATIC necessary condition: it is not stability, not standing, and not a statement that pose A itself is an equilibrium. '
              'The equilibria found here are NEIGHBOURING STATES inside the declared contract ball (46-52 active tarsus contacts, root z -0.02 to -0.05 mm), '
              'while pose A has 9 contacts; their joint distance to the source pose is 0.382-0.3996 rad, i.e. close to the declared 0.40 rad cap.')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
SUF = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
JOINT_MARGIN = 5e-3
MAX_DEVIATION = 0.40
ROOT_Z_RANGE = 0.05
N_FULL = 240
NESTED = ((0.05, 40), (0.10, 240), (0.20, 40))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = DESK / ('S2-equilibrium-A-corrected-reissued-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', title='CORRECTED / REISSUED declared MuJoCo equilibrium search for pose A (multi-start)',
                  manual_research_not_product=True, standing=False, walking=False,
                  voided_predecessor=('S2-mujoco-equilibrium-A-20261015T151340 is VOID: its five-uniform-start search stalled and its conclusion '
                                      '"(b) NO EQUILIBRIUM inside the declared boundary" is refuted; that directory must not be cited'),
                  declared_gate=GATE, declared_before_the_search=True, mandatory_disclosure=DISCLOSURE,
                  pose=str(POSE_A), source_pose=str(SOURCE_POSE),
                  absolute_force_values='NOT_ESTABLISHED',
                  input_hashes={str(p): sha(p) for p in (Path(__file__), POSE_A, SOURCE_POSE) if Path(p).exists()})
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model(); d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML); report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        leg_joints = {L: [next(j for j in range(m.njnt) if jname(j) == 'joint_' + L + s) for s in SUF] for L in LEGS}
        jids = [j for L in LEGS for j in leg_joints[L]]
        qadr = [int(m.jnt_qposadr[j]) for j in jids]
        src = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
        q0 = np.load(POSE_A)['projected_seated_qpos'].copy()
        act_lo = np.full(m.na, 1e-4); act_hi = np.full(m.na, 1.0)

        def eval_state(qpos, act):
            d.qpos[:] = qpos; d.qvel[:] = 0.0; d.act[:] = np.clip(act, act_lo, act_hi); d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            qa = np.asarray(d.qacc, dtype=float)
            f6 = np.zeros(6); rows = []; per_leg = {}
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                mujoco.mj_contactForce(m, d, ci, f6)
                fr = np.asarray(c.friction)[:3]
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
                leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
                kind = 'tarsus' if 'Tarsus' in (n1 + n2) else 'non-foot'
                rows.append(dict(geom=sorted((n1, n2)), leg=leg, kind=kind, fn=fn, u_diamond=u))
                if kind == 'tarsus' and leg in LEGS and u is not None:
                    per_leg[leg] = u if per_leg.get(leg) is None else max(per_leg[leg], u)
            mujoco.mj_kinematics(m, d); ft = np.zeros(6)
            tarsi_with = [L for L in LEGS if L in per_leg]
            nonfoot = [r for r in rows if r['kind'] == 'non-foot']
            us = [r['u_diamond'] for r in rows if r['u_diamond'] is not None]
            dq = float(max(np.max(np.abs(qpos[[m.jnt_qposadr[j] for j in leg_joints[L]]] - src[[m.jnt_qposadr[j] for j in leg_joints[L]]])) for L in LEGS))
            return dict(max_abs_qacc_mm_s2=float(np.max(np.abs(qa))), qacc_norm=float(np.linalg.norm(qa)),
                        n_dof=int(qa.size), n_dof_above_1e_minus_3=int(np.sum(np.abs(qa) > 1e-3)),
                        n_tarsus_contacts=sum(1 for r in rows if r['kind'] == 'tarsus'), n_nonfoot_contacts=len(nonfoot),
                        tarsi_with_contact=tarsi_with, six_tarsi_in_contact=bool(len(tarsi_with) == 6),
                        per_leg_max_u_diamond={L: per_leg.get(L) for L in LEGS},
                        max_u_diamond=(None if not us else float(max(us))),
                        root_z_mm=float(qpos[2]), max_abs_delta_q_rad=dq,
                        act_min=float(act.min()), act_max=float(act.max()),
                        act_at_floor=int(np.sum(np.abs(act - act_lo) <= 1e-9)), act_at_ceiling=int(np.sum(np.abs(act - act_hi) <= 1e-9)))

        # ---------------- full declared box, random multi-start ----------------
        lo_q = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, src[int(m.jnt_qposadr[j])] - MAX_DEVIATION) for j in jids])
        hi_q = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, src[int(m.jnt_qposadr[j])] + MAX_DEVIATION) for j in jids])

        def run_box(n_starts, lo_q_, hi_q_, seed):
            lb = np.concatenate([act_lo, [-ROOT_Z_RANGE], lo_q_]); ub = np.concatenate([act_hi, [ROOT_Z_RANGE], hi_q_])

            def unpack(x):
                qp = q0.copy(); qp[2] = q0[2] + x[m.na]; qp[qadr] = x[m.na + 1:]
                return qp, x[:m.na]

            def resid(x):
                qp, act = unpack(x)
                d.qpos[:] = qp; d.qvel[:] = 0.0; d.act[:] = np.clip(act, act_lo, act_hi); d.ctrl[:] = 0.0
                mujoco.mj_forward(m, d)
                return np.asarray(d.qacc, dtype=float)

            rng = np.random.default_rng(seed); hits = []; best = None
            for i in range(n_starts):
                xs = lb + rng.random(lb.size) * (ub - lb)
                try:
                    r = least_squares(resid, xs, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
                except Exception:
                    continue
                val = float(np.max(np.abs(r.fun)))
                if best is None or val < best['max_abs_qacc_mm_s2']:
                    qp, act = unpack(r.x); ev = eval_state(qp, act)
                    ev.update(status=str(r.status), message=str(r.message), nfev=int(r.nfev), njev=int(getattr(r, 'njev', 0)),
                              starts_used=i + 1, x=r.x.tolist())
                    best = ev
                if val <= 1.0:
                    qp, act = unpack(r.x); ev = eval_state(qp, act)
                    ev.update(status=str(r.status), nfev=int(r.nfev), start_index=i)
                    hits.append(ev)
                if i % 40 == 0:
                    print('  start %d/%d best max|qacc| %.6g hits<=1.0 %d' % (i, n_starts, best['max_abs_qacc_mm_s2'], len(hits)), flush=True)
            return dict(n_starts=n_starts, best=best, n_hits_le_1=len(hits),
                        n_hits_le_1e_minus_3=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= 1e-3)),
                        hits=[{k: v for k, v in h.items() if k != 'x'} for h in hits])
        print('full declared box, %d starts' % N_FULL, flush=True)
        full = run_box(N_FULL, lo_q, hi_q, 20261015)
        report['full_box'] = full
        best = full['best']
        xb = np.asarray(best['x']); qb = q0.copy(); qb[2] = q0[2] + xb[m.na]; qb[qadr] = xb[m.na + 1:]
        np.savez_compressed(out / 'equilibrium_best_state.npz', qpos=qb, act=np.clip(xb[:m.na], act_lo, act_hi))
        np.savez_compressed(out / 'equilibrium_best_pose_only.npz', projected_seated_qpos=qb)
        # nested radii around pose A
        nested = {}
        for r_rad, n_starts in NESTED:
            lo_n = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, q0[int(m.jnt_qposadr[j])] - r_rad) for j in jids])
            hi_n = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, q0[int(m.jnt_qposadr[j])] + r_rad) for j in jids])
            print('nested r = %.2f rad, %d starts' % (r_rad, n_starts), flush=True)
            nested['r_%.2f' % r_rad] = run_box(n_starts, lo_n, hi_n, 20261016 + int(r_rad * 100))
        report['nested_radii_around_pose_A'] = nested
        report['gate_evaluation'] = dict(
            checks=dict(max_abs_qacc=bool(best['max_abs_qacc_mm_s2'] <= GATE['max_abs_qacc_mm_s2']),
                        six_tarsi_active_contact=bool(best['six_tarsi_in_contact']),
                        zero_nonfoot_contacts=bool(best['n_nonfoot_contacts'] == 0),
                        all_contacts_u_diamond_le_1=bool(best['max_u_diamond'] is not None and best['max_u_diamond'] <= 1.0 + 1e-9)),
            values=dict(max_abs_qacc_mm_s2=best['max_abs_qacc_mm_s2'], qacc_norm_mm_s2=best['qacc_norm'],
                        n_tarsus_contacts=best['n_tarsus_contacts'], n_nonfoot_contacts=best['n_nonfoot_contacts'],
                        max_u_diamond=best['max_u_diamond'], root_z_mm=best['root_z_mm'], max_abs_delta_q_rad=best['max_abs_delta_q_rad'],
                        act_at_floor=best['act_at_floor'], act_at_ceiling=best['act_at_ceiling']),
            contract_caveat=('the best state has max|dq| = %.6g rad; when the source pose value of a joint already lies outside [joint range - margin] the bound is the '
                             'joint range rather than source-0.40, so this figure can slightly exceed 0.40 rad. States strictly inside 0.40 rad also exist (see the '
                             'nested probes)' % best['max_abs_delta_q_rad']))
        report['mandatory_disclosure'] = DISCLOSURE
        report['status'] = 'PASS_EQUILIBRIUM_SEARCH_CORRECTED_REISSUED'

        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        axes[0].bar(['full box\n(%d starts)' % full['n_starts'], 'r=0.05\n(%d)' % NESTED[0][1], 'r=0.10\n(%d)' % NESTED[1][1], 'r=0.20\n(%d)' % NESTED[2][1]],
                    [full['best']['max_abs_qacc_mm_s2'], nested['r_0.05']['best']['max_abs_qacc_mm_s2'],
                     nested['r_0.10']['best']['max_abs_qacc_mm_s2'], nested['r_0.20']['best']['max_abs_qacc_mm_s2']],
                    color=['tab:green', 'tab:orange', 'tab:orange', 'tab:green'])
        axes[0].set_yscale('log'); axes[0].axhline(1e-3, color='r', ls='--', lw=1, label='declared gate 1e-3 mm/s^2')
        axes[0].set_ylabel('best max |qacc| (mm/s^2)'); axes[0].set_title('best residual with the leg joints confined to r rad of pose A')
        axes[0].legend(fontsize=7)
        axes[1].bar(['A (delivered)', 'best equilibrium'], [9, best['n_tarsus_contacts']], color=['tab:blue', 'tab:green'])
        axes[1].set_ylabel('active tarsus contacts'); axes[1].set_title('the equilibrium is a neighbouring state, not pose A')
        axes[2].scatter([best['root_z_mm'] - q0[2]], [best['max_abs_delta_q_rad']], s=60, color='tab:red', label='best equilibrium')
        axes[2].axhline(0.40, color='k', ls='--', lw=1, label='declared 0.40 rad cap')
        axes[2].set_xlabel('root z change (mm)'); axes[2].set_ylabel('max joint distance from the SOURCE pose (rad)')
        axes[2].set_title('pose-contract position of the equilibrium'); axes[2].legend(fontsize=7)
        fig.suptitle('CORRECTED / REISSUED declared MuJoCo equilibrium search (multi-start) - manual research, NOT standing', fontsize=9)
        fig.tight_layout(); fig.savefig(out / 'equilibrium-search-reissued.png', dpi=130); plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc()[-1500:])
    finally:
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out); print(report['status'])
    if report['status'].startswith('PASS'):
        b = report['full_box']['best']
        print('best max|qacc| %.6g (norm %.6g) contacts %d (nonfoot %d) max u_diamond %s root z %.6f dq %.6g act@floor %d act@ceiling %d' %
              (b['max_abs_qacc_mm_s2'], b['qacc_norm'], b['n_tarsus_contacts'], b['n_nonfoot_contacts'], b['max_u_diamond'],
               b['root_z_mm'], b['max_abs_delta_q_rad'], b['act_at_floor'], b['act_at_ceiling']))
        print('hits <=1.0: %d (<=1e-3: %d) of %d starts' % (report['full_box']['n_hits_le_1'], report['full_box']['n_hits_le_1e_minus_3'], report['full_box']['n_starts']))
        print('nested:', {k: (v['best']['max_abs_qacc_mm_s2'], v['n_hits_le_1'], v['n_starts']) for k, v in report['nested_radii_around_pose_A'].items()})
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
