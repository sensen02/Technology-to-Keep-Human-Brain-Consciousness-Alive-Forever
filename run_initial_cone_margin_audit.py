"""Audit of the s = 0.95941 pose at the very start of the declared drive: is it already on the friction-cone boundary?

DECLARED QUESTION (fixed before running): for the delivered pose whose static s on the declared pyramidal (axis-ray diamond) cone is
0.9594120346009731 (file S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz), measure at step 0 and for the first
50 steps of the declared open-loop drive
  (a) the friction-cone utilisation u_diamond = |ft1|/(mu1*fn) + |ft2|/(mu2*fn) of EVERY tarsus contact and per foot (max per leg),
  (b) the overturning margin = signed distance from the whole-body CoM projection to the convex hull of the current tarsus contact
      points (negative = CoM outside the support polygon),
and report how many of those 51 states already have a contact on the declared boundary (u_diamond >= 1) and how many already have a
negative margin. If the pose starts ON the boundary, then "statically feasible inside the declared cone" and "dynamically
maintainable under MuJoCo's real cone" must be reported as separate statements with these initial numbers attached.

DRIVE AND DISCIPLINE: declared connectome archive 100 x 90 at 5 ms cadence
(outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz), ctrl = archive[min(k//200, 99)], dt = 25 us; qpos/qvel/act written
once before step 0 and only ctrl written inside the loop; step 0 is the initial state after mj_forward (no stepping).
Absolute force values and physical units remain NOT_ESTABLISHED; manual research, not a product; no standing/walking claim.
"""
from pathlib import Path
import hashlib, json, time, zipfile
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
POSE = DESK / 'S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz'
ARCHIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DT = 2.5e-5
CMD_DT = 0.005
N_AUDIT = 50          # steps 0..50 inclusive (51 states)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hull(points):
    pts = sorted(map(tuple, points))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lo = half(pts); up = half(pts[::-1])
    return lo[:-1] + up[:-1]


def main():
    out = DESK / ('S2-initial-cone-margin-audit-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('audit of the s = 0.95941 pose at step 0 and the first 50 steps of the declared open-loop drive: friction-cone '
                         'utilisation on the DECLARED axis-ray diamond per contact and per foot, plus the CoM-versus-support-hull margin'),
                  declared=dict(pose=str(POSE), pose_sha256=sha(POSE), static_s_on_declared_cone=0.9594120346009731,
                                drive=str(ARCHIVE), drive_sha256=sha(ARCHIVE), dt_s=DT, indexing='ctrl = archive[min(k // 200, 99)]',
                                states=('step 0 (initial state after mj_forward, no stepping) and steps 1..50'),
                                metric='u_diamond = |ft1|/(mu1*fn) + |ft2|/(mu2*fn) >= 1 means the declared diamond boundary is touched',
                                margin='signed distance from the CoM xy to the convex hull of the current tarsus contact points; negative = outside',
                                discipline='qpos/qvel/act written once; only ctrl inside the loop; no CPG, no PD, no preprogrammed action, no per-step pose correction'),
                  absolute_force_values='NOT_ESTABLISHED')
    report['input_hashes'] = {str(p): sha(p) for p in (Path(__file__), POSE, ARCHIVE) if Path(p).exists()}
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model(); m.opt.timestep = DT
        d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML); report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        controls = np.load(ARCHIVE)['normal_controls'].copy()
        hold = int(round(CMD_DT / DT))
        d.qpos[:] = np.load(POSE)['projected_seated_qpos'].copy()
        d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = controls[0]
        mujoco.mj_forward(m, d)
        states = []
        for k in range(N_AUDIT + 1):
            if k > 0:
                d.ctrl[:] = controls[min(k // hold, controls.shape[0] - 1)]
                mujoco.mj_step(m, d)
            f6 = np.zeros(6); per_contact = []; per_leg = {L: None for L in LEGS}; pts = []
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                if not ('Tarsus' in (n1 + n2)):
                    continue
                mujoco.mj_contactForce(m, d, ci, f6)
                fr = np.asarray(c.friction)[:3]
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
                leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
                per_contact.append(dict(geom=sorted((n1, n2)), leg=leg, fn=fn, ft1=ft1, ft2=ft2, u_diamond=u))
                if u is not None and leg in per_leg:
                    per_leg[leg] = u if per_leg[leg] is None else max(per_leg[leg], u)
                pts.append([float(c.pos[0]), float(c.pos[1])])
            margin = None
            if len(pts) >= 3:
                mujoco.mj_comPos(m, d)
                com = np.asarray(d.subtree_com[0])[:2]
                H = hull(pts); dmin = np.inf
                for i in range(len(H)):
                    a = np.asarray(H[i]); b = np.asarray(H[(i + 1) % len(H)])
                    e = b - a; nn = np.asarray([-e[1], e[0]]); nn = nn / (np.linalg.norm(nn) + 1e-12)
                    dmin = min(dmin, float(np.dot(com - a, nn)))
                margin = float(dmin)
            us = [c['u_diamond'] for c in per_contact if c['u_diamond'] is not None]
            states.append(dict(step=k, time_ms=k * DT * 1000.0, n_tarsus_contacts=len(per_contact),
                               max_u_diamond=(None if not us else float(max(us))),
                               n_contacts_on_boundary=int(sum(1 for u in us if u >= 1.0 - 1e-9)),
                               n_contacts_ge_0p95=int(sum(1 for u in us if u >= 0.95)),
                               per_leg_max_u_diamond={L: per_leg[L] for L in LEGS},
                               com_hull_margin_mm=margin,
                               root_z_mm=float(d.qpos[2]), max_abs_qvel=float(np.max(np.abs(d.qvel)))))
        report['states'] = states
        first = states[0]
        any_boundary = [s['step'] for s in states if s['n_contacts_on_boundary'] > 0]
        neg_margin = [s['step'] for s in states if s['com_hull_margin_mm'] is not None and s['com_hull_margin_mm'] < 0]
        report['answer'] = dict(
            step0=dict(max_u_diamond=first['max_u_diamond'], n_contacts_on_boundary=first['n_contacts_on_boundary'],
                       per_leg_max_u_diamond=first['per_leg_max_u_diamond'], com_hull_margin_mm=first['com_hull_margin_mm'],
                       root_z_mm=first['root_z_mm']),
            first_state_with_a_contact_on_the_declared_boundary=(None if not any_boundary else min(any_boundary)),
            n_states_with_a_contact_on_the_boundary=len(any_boundary),
            first_state_with_negative_com_hull_margin=(None if not neg_margin else min(neg_margin)),
            n_states_with_negative_margin=len(neg_margin),
            statement=('this pose is audited only at the start of the declared drive; the numbers say whether it is already pressed against the DECLARED friction boundary '
                       'and/or already outside the support hull, which is what separates "statically feasible inside the declared cone" from "dynamically maintainable '
                       'under MuJoCo\'s real cone". No standing or walking claim is made and absolute force values/units remain NOT_ESTABLISHED.'),
            not_standing='no standing, no walking, no completion claim')
        report['status'] = 'PASS_INITIAL_CONE_MARGIN_AUDIT_REPORTED'
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
        print(json.dumps(report['answer'], indent=1, default=str))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
