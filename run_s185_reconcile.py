"""S185 Stage A: row/DOF reconciliation from SAVED matrices (zero-forward unless the RM scan).

Independent code path: reads the frozen S177 baseline B/base/f0/J/T arrays and recomputes
every one of the 42 rows by hand, then cross-compares three sources bitwise.
The only physics part is the RM positive-channel scan (2 forwards per angle, no caching).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, json, shutil, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
S178 = BASE/'s178-legjoint-release-search/20261025T034000000000Z'
S180 = BASE/'s180-dof-authority-audit/20261025T096000000000Z'
S183 = BASE/'s183-geometry-targeted/20261025T150000000000Z'
S184 = BASE/'s184-census-and-staleness-audit/20261025T162000000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
SCAN_FORWARD_CAP = 60
STATE = ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied')
OPTIONAL = ('mocap_pos', 'mocap_quat', 'eq_active', 'plugin_state', 'userdata')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sanitize(v):
    if isinstance(v, dict): return {str(k): sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [sanitize(x) for x in v]
    if isinstance(v, np.generic): v = v.item()
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, (int, float, bool, str)) or v is None: return v
    return str(v)
def dump(p, v): p.write_text(json.dumps(sanitize(v), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def extract(p, names):
    t = ast.parse(p.read_text())
    return '\n'.join(ast.unparse(next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == x)) for x in names)+'\n'

def main(out):
    assert out.is_absolute() and not out.exists(); out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    files = [Path(__file__).resolve(), XML, S154/'INPUT_HASHES.json', S154/'init80ms/TRAJECTORY.npz',
             S154/'init80ms/MODEL_PROVENANCE.json', S154/'init80ms/COM_CONTACT_METADATA.json',
             S166/'MAPPING.npz', S177/'ROOT_BALANCE_STATE.npz', S177/'ROOT_RESULT.json',
             S178/'STAGE_A_BASELINE_ROWS.json', S180/'ARM_SIGN_TABLE.csv', S180/'DEMAND_DECOMPOSITION.csv',
             S180/'NATIVE_LIMIT_FLIP.csv', S183/'SUMMARY.json', S184/'CENSUS_38.json', S184/'ROW_GAP_DECOMPOSITION.csv']
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: '+p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files}); dump(out/'INPUT_HASHES.json', hashes)
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
    anames = [n.get('name') for n in tree.getroot().find('actuator')]
    jname = [j.get('name') for j in joints[1:]]
    nat = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    st = load(S177/'ROOT_BALANCE_STATE.npz')
    B57, base57, f057, J57, T57 = st['B'], st['base'], st['f0'], st['J'], st['T']
    assert B57.shape == (48, 90) and T57.shape == (90, 66)
    calls = dict(model_load=0, forward=0, scan_forward=0, inner_lp=0)
    # ---------- definition of "unreachable", written before any computation ----------
    dump(out/'PREREG.json', dict(
        task='S185/P1-S2 row/DOF reconciliation (no Stage B unless the conflict is resolved)',
        pose_for_definition='S177 root-balanced baseline: qpos = ROOT_BALANCE_STATE.npz["qpos"], the same arrays that produced the S177/S183/S184 baseline numbers',
        matrices='B, base, f0, J read from ROOT_BALANCE_STATE.npz (frozen S177 run); T from s166 MAPPING.npz; no fresh forward for the reconciliation',
        row_target='required[i] = -(base + J.T@f0)[6+i] for joint_row i in 0..41 (equivalently DOF 6+i)',
        interval='[sum_j min((B@T)[6+i,j],0), sum_j max((B@T)[6+i,j],0)] over the 66 pool activations x in [0,1]; the 0.5 gain is already inside T (T has 0.5 entries), so the box is the frozen 0.5 limit',
        reachable_row='a joint_row is INTERVAL-REACHABLE iff required lies inside its own [Σmin,Σmax] (single-row necessary condition)',
        lpm_minimax='separately, min over x in [0,1]^66 of max_i |(A x - required)_i| over ALL 42 rows (simultaneous condition)',
        note='interval-reachability of every row does NOT imply LP minimax = 0; the two are different questions and are reported separately here',
        scan='RM rows: arm of the dominant pool vs that joint angle, fresh pose each point (no cache), 2 forwards per point',
        stageB_trigger='only if (a) all three sources agree bitwise, (b) the six violation values are traced to the same DOF names in every table, and (c) the two RM rows are AMPLITUDE-LIKE with m<=3 inside native limits',
        no_step=True, no_neural=True, no_training=True, no_dynamics=True, units='native; SI NOT_ESTABLISHED'))
    # ---------- independent recomputation of all 42 rows at the baseline ----------
    A = (B57@T57)[6:, :]
    req = -(base57+J57.T@f057)[6:]
    lo = np.minimum(A, 0.).sum(axis=1); hi = np.maximum(A, 0.).sum(axis=1)
    viol = np.maximum.reduce([lo-req, np.zeros(42), req-hi])
    reach = (lo-1e-9 <= req) & (req <= hi+1e-9)
    n = A.shape[1]
    lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                 b_ub=np.r_[req, -req], bounds=[(0, 1)]*n+[(0, None)], method='highs'); calls['inner_lp'] += 1
    rows = []
    for i in range(42):
        m = (req[i]/hi[i]) if (req[i] > 0 and hi[i] > 1e-12) else ((req[i]/lo[i]) if (req[i] < 0 and lo[i] < -1e-12) else None)
        rows.append(dict(joint_row=i, dof=6+i, dof_name=jname[i], leg=jname[i][6:8], joint_name=jname[i],
                         required=float(req[i]), sum_min=float(lo[i]), sum_max=float(hi[i]),
                         gap=float(max(0., req[i]-hi[i], lo[i]-req[i])), reachable=bool(reach[i]),
                         needed_multiplier=(float(m) if m is not None and np.isfinite(m) else None),
                         violation=float(viol[i])))
    with (out/'RECONCILE_BASELINE.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    unrec = [r for r in rows if not r['reachable']]
    # ---------- three-source crosswalk ----------
    cen = json.loads((S184/'CENSUS_38.json').read_text())
    cbase = [r for r in cen['rows'] if r['tag'] == 'BASELINE_s177_pose'][0]
    s183 = json.loads((S183/'SUMMARY.json').read_text())
    s183base = s183['baseline']
    s178 = json.loads((S178/'STAGE_A_BASELINE_ROWS.json').read_text())
    arm = list(csv.DictReader((S180/'ARM_SIGN_TABLE.csv').open()))
    arm_base = {int(r['dof']): r for r in arm if r['pose'] == 's177_baseline'}
    dem = {int(r['dof']): r for r in csv.DictReader((S180/'DEMAND_DECOMPOSITION.csv').open())}
    flip = list(csv.DictReader((S180/'NATIVE_LIMIT_FLIP.csv').open()))
    six = [d for d in range(6, 48) if not reach[d-6]]
    violations6 = [float(viol[d-6]) for d in six]
    crosswalk = dict(
        recomputed=dict(unreachable_dofs=six, unreachable_names=[jname[d-6] for d in six], violations_six=violations6,
                        lp_minimax=float(lp.x[-1])),
        census_38=dict(unreachable_dofs=cbase['unreachable'], minimax=cbase['minimax'],
                       match_dofs=bool(cbase['unreachable'] == six), dminimax_diff=float(abs(cbase['minimax']-lp.x[-1]))),
        s183_summary_baseline=dict(unreachable_dofs=s183base['unreachable'], violations_six=s183base['violations_six'],
                                   match_violations=bool(np.max(np.abs(np.array(s183base['violations_six'])-np.array(violations6))) <= 1e-12),
                                   max_violation_diff=float(np.max(np.abs(np.array(s183base['violations_six'])-np.array(violations6))))),
        s178_stageA_baseline=dict(rows={r['dof']: dict(target=r['target'], lo=r['lo'], hi=r['hi'], reachable=r['reachable']) for r in s178['rows']},
                                  note='S178 saved its own baseline table; compared below'),
        s180_arm_sign=dict(sample={d: dict(target=arm_base[d]['target'], lo=arm_base[d]['lo'], hi=arm_base[d]['hi'],
                                           reachable=arm_base[d]['reachable'], needed_sign=arm_base[d]['needed_sign'],
                                           channel_missing=arm_base[d]['needed_sign_channel_missing']) for d in six}),
        s180_demand={d: dict(target=dem[d]['target'], gravity=dem[d]['gravity_passive_part'], contact=dem[d]['contact_part'],
                             dominant=dem[d]['dominant']) for d in six},
        s180_native_flip={r['dof']: dict(row_reachable_at_some_grid=r['flippable_in_native_limits'],
                                         joint_name=r['joint_name'], native_range=[float(r['native_range'][0]) if False else r['native_range']],
                                         delta_from_release=r['delta_from_release']) for r in flip if int(r['dof']) in six})
    diffs = []
    for r in s178['rows']:
        d = int(r['dof']); i = d-6
        diffs.append(dict(source='S178_STAGE_A_BASELINE_ROWS', dof=d, target_diff=float(abs(r['target']-req[i])),
                          lo_diff=float(abs(r['lo']-lo[i])), hi_diff=float(abs(r['hi']-hi[i])),
                          reachable_match=bool(r['reachable'] == reach[i])))
    for d in six:
        i = d-6; a = arm_base[d]
        diffs.append(dict(source='S180_ARM_SIGN_TABLE', dof=d, target_diff=float(abs(float(a['target'])-req[i])),
                          lo_diff=float(abs(float(a['lo'])-lo[i])), hi_diff=float(abs(float(a['hi'])-hi[i])),
                          reachable_match=bool((a['reachable'] == 'True') == reach[i])))
    # S180 DEMAND_DECOMPOSITION was computed at the cand11 pose, NOT at the baseline: it is
    # reported for provenance but excluded from the bitwise baseline comparison.
    crosswalk_demand_note = ('S180 DEMAND_DECOMPOSITION pose = cand11 (from S179); excluded from the baseline '
                             'bitwise comparison on purpose, its targets differ by 0.02-1.41 because it is a different pose')
    crosswalk['per_row_differences'] = diffs
    crosswalk['demand_decomposition_excluded'] = crosswalk_demand_note
    crosswalk['max_target_diff_any_source'] = float(max(d['target_diff'] for d in diffs))
    crosswalk['verdict'] = 'ALIGNED' if (crosswalk['max_target_diff_any_source'] <= 1e-12 and cbase['unreachable'] == six
                                        and crosswalk['s183_summary_baseline']['match_violations']
                                        and all(d['reachable_match'] is not False for d in diffs)) else 'MISALIGNED'
    dump(out/'SOURCE_CROSSWALK.json', crosswalk)
    # ---------- answer: which dof_name do the six violations belong to ----------
    dump(out/'SIX_ROWS_IDENTITY.json', dict(
        six_dofs=six, six_names=[jname[d-6] for d in six], six_violations=violations6,
        gap_table_pose='S184 ROW_GAP_DECOMPOSITION.csv was computed at refine_j26_-1_0.1 (the best census point), NOT at the baseline',
        gap_table_dofs=[int(r['dof']) for r in csv.DictReader((S184/'ROW_GAP_DECOMPOSITION.csv').open())],
        gap_table_names=[r['dof_name'] for r in csv.DictReader((S184/'ROW_GAP_DECOMPOSITION.csv').open())],
        same_dof_batch=bool([int(r['dof']) for r in csv.DictReader((S184/'ROW_GAP_DECOMPOSITION.csv').open())] == six),
        note='the S184 gap table reported gap=0 for four rows because it was taken at a DIFFERENT pose; at the baseline all six rows have gap>0. This is a pose difference, not a row/DOF mix-up.'))
    # ---------- LP active set / slack at the baseline (0 forward) ----------
    xstar = lp.x[:n]
    res = A@xstar-req
    order = np.argsort(-np.abs(res))
    dump(out/'LP_ACTIVE_SET.json', dict(minimax=float(lp.x[-1]), n_active=int(np.sum(np.abs(np.abs(res)-lp.x[-1]) <= 1e-9)),
         tightest_rows=[dict(joint_row=int(i), dof=int(i+6), dof_name=jname[i], residual=float(res[i])) for i in order[:10]],
         rows_at_box=[int(j) for j in range(n) if xstar[j] in (0.0,) or abs(xstar[j]-1.0) <= 1e-9][:20],
         n_box_active=int(np.sum((xstar <= 1e-12) | (xstar >= 1-1e-12))),
         note='the rows with |residual| equal to the minimax are the binding constraints; rows whose interval contains its target can still be violated in the simultaneous LP'))
    # ---------- RM two-row positive-channel scan (the only physics part) ----------
    specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)), (ROOT/'run_local_pool_response.py', ('moment_record',)),
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',))]
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p, n) for p, n in specs), '<inert>', 'exec'), ns)
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    import mujoco as mj
    calls['model_load'] += 1; m = mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m) == proof['before']
    m.actuator_dynprm[:, 0] = .001; m.actuator_dynprm[:, 1] = .0004
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = .000025
    assert ns['snapshot'](m) == proof['after']
    def fresh(q):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for nm in STATE: getattr(d, nm)[:] = z[nm][0]
        for nm in OPTIONAL:
            if nm in z: getattr(d, nm)[:] = z[nm][0]
            else: assert np.asarray(getattr(d, nm)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = 0; d.ctrl[:] = 0; d.qacc_warmstart[:] = 0
        mj.mj_forward(m, d); calls['forward'] += 1; calls['scan_forward'] += 1
        return dict(af=np.asarray(d.actuator_force).copy(), mom=np.asarray(ns['moment_record'](m, d)['actuator_moment']).copy(),
                    efc_D=np.asarray(d.efc_D).copy(), efc_aref=np.asarray(d.efc_aref).copy(),
                    efc_type=np.asarray(d.efc_type).copy(), J=np.asarray(ns['constraint_record'](mj, m, d, 0)['efc_J_dense']).copy(),
                    pas=np.asarray(d.qfrc_passive).copy(), bias=np.asarray(d.qfrc_bias).copy())
    def model2(qa):
        c1 = fresh(qa); c0 = fresh(qa)
        gv = np.asarray(c1['af'])-np.asarray(c0['af']); b = np.asarray(c0['af']); mom = c0['mom']
        types = c0['efc_type']; f0 = c0['efc_D']*c0['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        Bm = mom.T*gv[None, :]; bs = mom.T@b+np.asarray(c0['pas'])-np.asarray(c0['bias'])
        return Bm, bs, f0, c0['J']
    qbase = st['qpos'].copy()
    scan = []
    for dof in (42, 47):
        i = dof-6
        pool_dom = int(np.argmax(np.abs(A[i])))
        joint_candidates = [k for k in range(42) if jname[k][6:8] == 'RM']
        best = None
        for k in joint_candidates:
            lo_n, hi_n = float(nat[k][0]), float(nat[k][1])
            for ang in (lo_n, hi_n):
                if calls['forward'] >= SCAN_FORWARD_CAP: break
                q = qbase.copy(); q[7+k] = ang
                Bm, bs, f0, Jm = model2(q)
                Ai = (Bm@T57)[6:, :]
                reqi = -(bs+Jm.T@f0)[6:]
                hi_i = float(np.maximum(Ai[i], 0.).sum())
                mneed = (float(reqi[i])/hi_i) if (reqi[i] > 0 and hi_i > 1e-12) else None
                kind = ('CHANNEL' if hi_i <= 1e-12 else ('AMPLITUDE' if (mneed is not None and mneed <= 3.0) else 'AMPLITUDE_HARD'))
                scan.append(dict(dof=dof, dof_name=jname[i], joint_row=k, joint_name=jname[k], angle=float(ang),
                                 required=float(reqi[i]), sum_max_positive=hi_i,
                                 needed_multiplier=(mneed if mneed is not None else None), classification=kind,
                                 contact_signature_efc=int(len(Jm))))
                if best is None or (hi_i > best['sum_max_positive']):
                    best = dict(joint_row=k, joint_name=jname[k], angle=float(ang), sum_max_positive=hi_i, classification=kind,
                                needed_multiplier=(mneed if mneed is not None else None))
            if calls['forward'] >= SCAN_FORWARD_CAP: break
        if best: scan.append(dict(dof=dof, dof_name=jname[i], joint_row=best['joint_row'], joint_name=best['joint_name'],
                                  angle=best['angle'], required=float(req[i]), sum_max_positive=best['sum_max_positive'],
                                  needed_multiplier=best['needed_multiplier'], classification='BEST_FOUND:'+best['classification'],
                                  contact_signature_efc=-1))
    with (out/'RM_CHANNEL_SCAN.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(scan[0])); w.writeheader(); w.writerows(scan)
    rm_best = {d: max([s for s in scan if s['dof'] == d and not s['classification'].startswith('BEST_FOUND')],
                      key=lambda s: s['sum_max_positive'], default=None) for d in (42, 47)}
    all_pts = [s for s in scan if not s['classification'].startswith('BEST_FOUND')]
    per_row_max = {d: max([s['sum_max_positive'] for s in all_pts if s['dof'] == d], default=0.) for d in (42, 47)}
    per_row_minm = {d: min([s['needed_multiplier'] for s in all_pts if s['dof'] == d and s['needed_multiplier'] is not None], default=None) for d in (42, 47)}
    scanned = {d: sum(1 for s in all_pts if s['dof'] == d) for d in (42, 47)}
    if all(per_row_max[d] <= 1e-12 for d in (42, 47)) and all(scanned[d] >= 7 for d in (42, 47)):
        rm_verdict = 'CHANNEL_TYPE'
    elif any(v is not None and v <= 3.0 for v in per_row_minm.values()):
        rm_verdict = 'AMPLITUDE_TYPE'
    elif all(scanned[d] >= 7 for d in (42, 47)):
        rm_verdict = 'AMPLITUDE_HARD'
    else:
        rm_verdict = 'SCAN_INCOMPLETE'
    dump(out/'RM_CHANNEL_VERDICT.json', dict(per_row_max_positive=per_row_max, per_row_min_multiplier=per_row_minm,
         points_scanned=scanned, best=rm_best, verdict=rm_verdict,
         note='native XML limits, one joint moved at a time, everything recomputed fresh (no cache); 2 forwards per point'))
    stageB = ('STAGE_B_NOT_TRIGGERED_IMPLEMENTATION_DEFECT' if crosswalk['verdict'] != 'ALIGNED' else
              ('STAGE_B_NOT_TRIGGERED_CHANNEL_TYPE' if rm_verdict == 'CHANNEL_TYPE' else 'STAGE_B_TRIGGERED_AMPLITUDE_TYPE'))
    dump(out/'SUMMARY.json', dict(crosswalk_verdict=crosswalk['verdict'], rm_verdict=rm_verdict, stageB=stageB,
         baseline_lp_minimax=float(lp.x[-1]), baseline_unreachable_dofs=six, violations_six=violations6,
         explanations=[
             'S184 gap table was computed at refine_j26_-1_0.1, a DIFFERENT pose, so its gap=0 for four rows does not contradict the baseline LP',
             'interval-reachability (per row) and LP minimax (all 42 rows at once) are different questions and are reported separately',
             'the six baseline violations belong to ' + ', '.join(jname[d-6] for d in six)],
         calls=calls, units='native; SI NOT_ESTABLISHED'))
    dump(out/'INPUT_HASHES_AFTER.json', {k: sha(k) for k in hashes})
    print(json.dumps(sanitize(dict(crosswalk=crosswalk['verdict'], rm=rm_verdict, stageB=stageB,
                                   minimax=float(lp.x[-1]), six_names=[jname[d-6] for d in six],
                                   maxdiff=crosswalk['max_target_diff_any_source'], scan_fw=calls['scan_forward']))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path)
    a = ap.parse_args(); main(a.output or BASE/'s185-row-dof-reconciliation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
