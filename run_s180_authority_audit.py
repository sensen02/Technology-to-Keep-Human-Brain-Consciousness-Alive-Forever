"""S180 Stage A: DOF authority audit. No mj_step, no neural, no training.

Items: (1) B vs engine qfrc_actuator cross-check (any mismatch => implementation defect),
(2) per-DOF arm sign table + antagonist actuators mapped to pools and the 16 empty slots,
(3) demand decomposition (gravity+passive vs contact) for the unreachable rows,
(4) sign-flip requirement recomputed against NATIVE XML limits (not our +/-0.6 window).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from s179_core import build_row_dof_map, leg_of_name, row_of_dof

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
S179 = BASE/'s179-targeted-legjoint-search/20261025T070000000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP, LOAD_CAP = 300, 2
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
    files = [Path(__file__).resolve(), ROOT/'s179_core.py', XML, S154/'INPUT_HASHES.json', S154/'init80ms/TRAJECTORY.npz',
             S154/'init80ms/MODEL_PROVENANCE.json', S154/'init80ms/NATIVE_MODEL_ARRAYS.npz', S154/'init80ms/COM_CONTACT_METADATA.json',
             S166/'MAPPING.npz', S177/'ROOT_BALANCE_STATE.npz', S177/'ROOT_RESULT.json', S179/'ROW_DOF_MAP.json', S179/'CANDIDATES.json']
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: '+p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files}); dump(out/'INPUT_HASHES.json', hashes)
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
    specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)), (ROOT/'run_local_pool_response.py', ('moment_record',)),
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',)),
             (ROOT/'verify_muscle_reconstruction.py', ('fl', 'evaluate'))]
    text = '\n'.join(extract(p, n) for p, n in specs)
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile(text, '<inert>', 'exec'), ns)
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    mapping = load(S166/'MAPPING.npz'); T = mapping['T']; pool = mapping['actuator_pool']
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
    nodes = list(tree.getroot().find('actuator')); anames = [n.get('name') for n in nodes]
    nat = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    calls = dict(model_load=0, forward=0)
    import mujoco as mj
    assert mj.__version__ == '3.9.0'; calls['model_load'] += 1
    m = mj.MjModel.from_xml_path(str(XML)); assert ns['snapshot'](m) == proof['before']
    m.actuator_dynprm[:, 0] = .001; m.actuator_dynprm[:, 1] = .0004
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = .000025
    assert ns['snapshot'](m) == proof['after']
    jname = [j.get('name') for j in joints[1:]]
    rowmap = build_row_dof_map(['x']*6+[mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or 'x' for j in range(1, 43)])
    for k, v in enumerate(rowmap): assert v['dof'] == k+6 and v['dof_name'] == jname[k]
    ref_map = json.loads((S179/'ROW_DOF_MAP.json').read_text())
    assert sha(S179/'ROW_DOF_MAP.json') == hashes[str(S179/'ROW_DOF_MAP.json')]
    assert [r['dof_name'] for r in ref_map['rows']] == [r['dof_name'] for r in rowmap]
    dump(out/'ROW_DOF_MAP_REUSED.json', dict(note='reused S179 map, sha verified, regenerated and compared',
         sha256=hashes[str(S179/'ROW_DOF_MAP.json')], rows=rowmap))
    def forward(q, act90=None):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for n in STATE: getattr(d, n)[:] = z[n][0]
        for n in OPTIONAL:
            if n in z: getattr(d, n)[:] = z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0
        a = np.zeros(90) if act90 is None else act90
        d.act[:] = a; d.ctrl[:] = a; d.qacc_warmstart[:] = 0
        mj.mj_forward(m, d); calls['forward'] += 1
        c = dict(actuator_force=np.asarray(d.actuator_force).copy(), qacc=np.asarray(d.qacc).copy(), qfrc_actuator=np.asarray(d.qfrc_actuator).copy(),
                 qfrc_passive=np.asarray(d.qfrc_passive).copy(), qfrc_bias=np.asarray(d.qfrc_bias).copy(),
                 actuator_moment=np.asarray(ns['moment_record'](m, d)['actuator_moment']).copy(),
                 actuator_length=np.asarray(d.actuator_length).copy(),
                 efc_D=np.asarray(d.efc_D).copy(), efc_aref=np.asarray(d.efc_aref).copy(),
                 efc_type=np.asarray(d.efc_type).copy(), efc_J_dense=np.asarray(ns['constraint_record'](mj, m, d, 0)['efc_J_dense']).copy(),
                 contact_dist=np.array([v.dist for v in d.contact]))
        return c
    def engine_gain(q, moment):
        """Exact gain: the engine force is affine in act, so gain = f(1) - f(0) (measured, not reconstructed)."""
        a1 = np.ones(90); f1 = forward(q, a1); f0v = forward(q, np.zeros(90))
        g = np.asarray(f1['actuator_force'], float)-np.asarray(f0v['actuator_force'], float)
        b = np.asarray(f0v['actuator_force'], float).copy()
        chk = forward(q, np.full(90, .37))
        pred = g*.37+b
        err = float(np.abs(pred-np.asarray(chk['actuator_force'])).max())
        assert err <= 1e-6*max(1.0, float(np.abs(pred).max())), f'engine force is not affine in act: {err}'
        return g, b, err

    def affine(c, q=None):
        types = c['efc_type']
        f0 = c['efc_D']*c['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        if q is None:
            gain = np.empty(90); bias = np.empty(90)
            for j, node in enumerate(nodes): _, gain[j], bias[j] = ns['evaluate'](c['actuator_length'][j], 0., 0., node)
            src = 'reconstructed_evaluate'
        else:
            gain, bias, aff_err = engine_gain(q, c['actuator_moment'])
            src = f'engine_measured_affine_err={aff_err:.3g}'
        B = c['actuator_moment'].T*gain[None, :]
        base = c['actuator_moment'].T@bias+c['qfrc_passive']-c['qfrc_bias']
        return B, base, f0, gain, bias
    cands = json.loads((S179/'CANDIDATES.json').read_text())
    best11 = [c for c in cands if c['index'] == 11][0]
    q11 = np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'].copy()
    rm = [v for v in rowmap if v['leg'] == 'rm']
    for mv in best11['moves']:
        row = mv[0]; q11[7+row] = q11[7+row]+mv[1]
    q11[0:2] = best11['root_p'][0:2]; q11[2] = best11['root_p'][2]
    # quaternion from root_p euler (same convention as S179)
    def quat_mul(a, b):
        aw, ax, ay, az = a; bw, bx, by, bz = b
        return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
    rx, ry, rz = best11['root_p'][3:6]
    cx, sx = np.cos(rx/2), np.sin(rx/2); cy, sy = np.cos(ry/2), np.sin(ry/2); cz, sz = np.cos(rz/2), np.sin(rz/2)
    dq = np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])
    qw = quat_mul(dq, q0[3:7]); q11[3:7] = qw/np.linalg.norm(qw)
    qbase = np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'].copy()
    poses = {'cand11': q11, 's177_baseline': qbase}
    results = {}
    for name, q in poses.items():
        c = forward(q); B, base, f0, gain, bias = affine(c, q)
        A = (B@T)[6:, :]; bb = -(base+c['efc_J_dense'].T@f0); tgt = bb[6:]
        lo = np.minimum(A, 0.).sum(axis=1); hi = np.maximum(A, 0.).sum(axis=1)
        reach = (lo-1e-9 <= tgt) & (tgt <= hi+1e-9)
        unres = [int(v['dof']) for v in rowmap if not reach[v['joint_row']]]
        n = A.shape[1]
        lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                     b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs')
        results[name] = dict(B=B, base=base, f0=f0, A=A, bias=bias, tgt=tgt, lo=lo, hi=hi, reach=reach, unres=unres,
                             minimax=float(lp.x[-1]) if lp.success else None, activation=(T@lp.x[:n]) if lp.success else None,
                             q=q, c=c)
    # ---- (1) engine cross-check on cand11: pool columns vs engine qfrc_actuator ----
    r11 = results['cand11']; A11 = r11['A']
    pool_to_act = {p: [k for k in range(90) if int(pool[k]) == p] for p in range(66)}
    def act_for_pool(p, level):
        a = np.zeros(90)
        for k in pool_to_act[p]: a[k] = 0.5*level
        return a
    crosses = []
    tests = [(d, 'unreachable_dominant') for d in r11['unres']] + [(d, 'control_sample') for d in [6, 10, 15, 21, 30, 45]]
    for dof, kind in tests:
        i = dof-6; p = int(np.argmax(np.abs(A11[i])))
        bias_term = float((r11['c']['actuator_moment'].T@np.asarray(r11['bias']))[dof])
        for level in (0.25, 0.5):
            cc = forward(r11['q'], act_for_pool(p, level))
            qa = float(cc['qfrc_actuator'][dof])
            pred = float(A11[i, p]*level)+bias_term
            crosses.append(dict(dof=dof, pool=p, actuators=pool_to_act[p], kind=kind, pool_activation=level,
                                engine_qfrc=qa, predicted_A_times_activation=pred, difference=qa-pred,
                                sign_match=bool(np.sign(qa) == np.sign(pred) or abs(qa) < 1e-12),
                                rel_error=float(abs(qa-pred)/max(abs(qa), abs(pred), 1e-12))))
    maxdiff = max(abs(x['difference']) for x in crosses)
    defect = not (all(x['sign_match'] for x in crosses) and maxdiff <= 1e-6)
    dump(out/'B_ENGINE_CROSSCHECK.json', dict(pose='cand11', model='engine qfrc_actuator = A[:,pool]*activation + (moment.T@bias) at fixed qpos; bias term is activation-independent',
         tests=crosses, max_abs_difference=maxdiff, all_signs_match=all(x['sign_match'] for x in crosses),
         tolerance=1e-6, verdict='IMPLEMENTATION_DEFECT' if defect else 'PASS'))
    # ---- (2) per-DOF arm sign tables ----
    def arm_table(name):
        r = results[name]; A = r['A']; rows = []
        for v in rowmap:
            i = v['joint_row']; arms = A[i]
            pos = float(np.maximum(arms, 0).sum()); neg = float(np.minimum(arms, 0).sum())
            dom = int(np.argmax(np.abs(arms)))
            rows.append(dict(pose=name, joint_row=v['joint_row'], dof=v['dof'], dof_name=v['dof_name'], leg=v['leg'],
                             target=float(r['tgt'][i]), lo=neg, hi=pos, reachable=bool(r['reach'][i]),
                             dominant_pool=dom, dominant_actuator=pool_to_act[dom][0], dominant_actuator_name=anames[pool_to_act[dom][0]],
                             dominant_arm=float(arms[dom]), dominant_slot_count=len(pool_to_act[dom]),
                             has_positive_channel=bool(pos > 1e-9), has_negative_channel=bool(neg < -1e-9),
                             needed_sign=('positive' if r['tgt'][i] > 1e-9 else ('negative' if r['tgt'][i] < -1e-9 else 'zero')),
                             needed_sign_channel_missing=bool((r['tgt'][i] > 1e-9 and pos <= 1e-9) or (r['tgt'][i] < -1e-9 and neg >= -1e-9))))
        return rows
    tbl = arm_table('cand11')+arm_table('s177_baseline')
    with (out/'ARM_SIGN_TABLE.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tbl[0])); w.writeheader(); w.writerows(tbl)
    # ---- antagonist actuators for the unreachable rows, mapped to pools / 16 empty slots ----
    empty = [i for i in range(90) if pool[i] < 0]
    anti = []
    for dof in r11['unres']:
        i = dof-6; arms = r11['A'][i]; t = r11['tgt'][i]
        want = -np.sign(t) if t != 0 else 0
        cand = [(j, float(arms[j])) for j in range(66) if want*arms[j] > 0]
        cand.sort(key=lambda x: -abs(x[1]))
        for j, arm in cand[:3]:
            anti.append(dict(dof=dof, dof_name=rowmap[i]['dof_name'], leg=rowmap[i]['leg'], target=float(t),
                             needed_sign='positive' if t > 0 else 'negative', antagonist_pool=j,
                             antagonist_actuators=pool_to_act[j], antagonist_arm=arm,
                             slot_share=len(pool_to_act[j]), driven_by_empty_slot=bool(len(pool_to_act[j]) == 0)))
    dump(out/'ANTAGONIST_POOL_MAP.json', dict(empty_actuator_slots=empty, empty_actuator_names=[anames[k] for k in empty],
         note='pool index < 0 means the slot has no source in the frozen mapping (the 16 missing outputs)',
         rows=anti))
    # ---- (3) demand decomposition ----
    dec = []
    for dof in r11['unres']:
        i = dof-6
        g = float(-r11['base'][i]); cf = float(-(r11['c']['efc_J_dense'].T@r11['f0'])[i])
        dec.append(dict(dof=dof, dof_name=rowmap[i]['dof_name'], leg=rowmap[i]['leg'], target=float(r11['tgt'][i]),
                        gravity_passive_part=g, contact_part=cf,
                        dominant='gravity' if abs(g) >= abs(cf) else 'contact',
                        same_sign=bool(np.sign(g) == np.sign(cf)),
                        gravity_sign='positive' if g > 0 else 'negative', contact_sign='positive' if cf > 0 else 'negative'))
    with (out/'DEMAND_DECOMPOSITION.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dec[0])); w.writeheader(); w.writerows(dec)
    # ---- (4) sign flip under NATIVE limits ----
    flips = []
    for dof in r11['unres']:
        i = dof-6; leg = rowmap[i]['leg']
        jidx_all = [k for k in range(42) if rowmap[k]['leg'] == leg]
        jidx = sorted(jidx_all, key=lambda k: -abs(float(r11['A'][i][int(np.argmax(np.abs(r11['A'][i])))])))[:3] if False else jidx_all[:1]+jidx_all[6:7]+jidx_all[3:4]
        for k in jidx:
            if calls['forward'] >= CAP-2: break
            j = k  # joint index equals joint_row here (dof-6)
            dom = int(np.argmax(np.abs(r11['A'][i])))
            lo_n, hi_n = nat[j]
            grid = np.linspace(lo_n, hi_n, 3)
            best = None
            for ang in grid:
                q = r11['q'].copy(); q[7+j] = ang
                cc = forward(q); Bc, basec, f0c, _, _ = affine(cc, q)
                Ac = (Bc@T)[6:, :]; tc = float(-(basec+cc['efc_J_dense'].T@f0c)[6:][i])
                l = float(np.minimum(Ac[i], 0).sum()); h = float(np.maximum(Ac[i], 0).sum())
                ok = bool(l-1e-9 <= tc <= h+1e-9)
                arm = float(Ac[i, dom])
                if best is None or (ok and not best['reachable']):
                    best = dict(angle=float(ang), reachable=ok, arm=arm, lo=l, hi=h, target=tc)
                if ok: break
            rel = float(r11['q'][7+j])
            flips.append(dict(dof=dof, dof_name=rowmap[i]['dof_name'], leg=leg, joint_row=k, joint_name=rowmap[k]['dof_name'],
                              native_range=[float(lo_n), float(hi_n)], release=rel, best_angle=best['angle'],
                              delta_from_release=float(best['angle']-rel), reachable_at_best=best['reachable'],
                              flippable_in_native_limits=bool(best['reachable'])))
    with (out/'NATIVE_LIMIT_FLIP.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flips[0])); w.writeheader(); w.writerows(flips)
    per_row = []
    for dof in r11['unres']:
        sub = [x for x in flips if x['dof'] == dof]
        ok = [x for x in sub if x['flippable_in_native_limits']]
        per_row.append(dict(dof=dof, dof_name=rowmap[dof-6]['dof_name'], leg=rowmap[dof-6]['leg'],
                            flippable=bool(ok), cheapest=None if not ok else min(ok, key=lambda x: abs(x['delta_from_release']))))
    stageB = dict(triggered=bool(any(v['flippable'] for v in per_row) and not defect),
                  reason=('implementation defect in B' if defect else ('at least one row flippable inside native limits' if any(v['flippable'] for v in per_row) else 'no unreachable row is flippable inside the native XML limits')),
                  rows=per_row)
    dump(out/'STAGE_B_TRIGGER.json', stageB)
    dump(out/'SUMMARY.json', dict(pose_cand11=dict(minimax=r11['minimax'], unreachable=r11['unres']),
         pose_baseline=dict(minimax=results['s177_baseline']['minimax'], unreachable=results['s177_baseline']['unres']),
         crosscheck_verdict='IMPLEMENTATION_DEFECT' if defect else 'PASS',
         crosscheck_max_abs_diff=max(abs(x['difference']) for x in crosses),
         stageB=stageB, calls=calls,
         per_row_always_unreachable_across_24_candidates=[19, 22, 26, 42],
         units='native; SI NOT_ESTABLISHED'))
    dump(out/'INPUT_HASHES_AFTER.json', {k: sha(k) for k in hashes})
    print(json.dumps(sanitize(dict(crosscheck='DEFECT' if defect else 'PASS', stageB=stageB['triggered'],
                                   reason=stageB['reason'], forward=calls['forward']))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path)
    a = ap.parse_args(); main(a.output or BASE/'s180-dof-authority-audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
