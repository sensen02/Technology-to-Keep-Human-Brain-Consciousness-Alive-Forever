"""S176/P1-S2 widened pose-screening search: 1 probe forward per pose + small LP.

Reuses the frozen S169 physics helpers by AST extraction (same model, same tau patch,
same native soft-contact affine law). Root rows have zero activation authority, so the
score is max(|root wrench residual|, LP minimax over the 42 joint rows).
No mj_step, no neural ticks, no controller, no training. Parent pre-registers all bounds.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil, traceback
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S165 = BASE/'s165-static-support-corrected/20261024T231400000000Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S173 = BASE/'s173-accounted-near-initial-search/20261024T235544196037334Z/RUN'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP = 1200
RESERVED = 10
LEGS = ('lf','lm','lh','rf','rm','rh')
STATE = ('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied')
OPTIONAL = ('mocap_pos','mocap_quat','eq_active','plugin_state','userdata')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
PRIM=(int,float,bool,str,type(None))
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,PRIM): return v
    return None
def dump(p, v): p.write_text(json.dumps(clean(v), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def extract(p, names):
    t = ast.parse(p.read_text())
    return '\n'.join(ast.unparse(next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == x)) for x in names)+'\n'

class BudgetStop(Exception): pass
class AcceptedStop(Exception): pass

def main(out):
    assert out.is_absolute() and not out.exists(), 'fresh timestamp only'
    out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    ledger = dict(model_load=0, forward=0, probe_forward=0, gate_forward=0, confirmation_forward=0,
                  pose_evaluations=0, lp_solves=0, step=0, neural_ticks=0)
    def reserve(kind):
        if kind != 'model_load' and ledger['forward'] >= CAP: raise BudgetStop()
        ledger[kind] += 1
        if kind != 'model_load': ledger['forward'] += 1
        assert ledger['model_load'] <= 1 and ledger['forward'] <= CAP
        dump(out/'CALL_LEDGER.json', ledger)
    specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)),
             (ROOT/'run_local_pool_response.py', ('moment_record',)),
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',)),
             (ROOT/'verify_muscle_reconstruction.py', ('fl', 'evaluate'))]
    files = [Path(__file__).resolve(), XML, S154/'INPUT_HASHES.json', S154/'init80ms/TRAJECTORY.npz',
             S154/'init80ms/MODEL_PROVENANCE.json', S154/'init80ms/NATIVE_MODEL_ARRAYS.npz',
             S154/'init80ms/COM_CONTACT_METADATA.json', S165/'STATIC_AFFINE_000.npz', S165/'ZERO_V_000.npz',
             S166/'MAPPING.npz', S173/'EVAL_001.npz', S173/'FROZEN_SEARCH.npz']+[p for p, _ in specs]
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: '+p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files})
    dump(out/'INPUT_HASHES.json', hashes)
    manifest = {}
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
        manifest[p] = dict(sha256=h, copy=str(t))
    dump(out/'SOURCE_FREEZE_MANIFEST.json', manifest)
    text = '\n'.join(extract(p, n) for p, n in specs)
    (out/'SOURCE_FREEZE'/'INERT_FUNCTIONS.py').write_text(text)
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile(text, str(out/'SOURCE_FREEZE'/'INERT_FUNCTIONS.py'), 'exec'), ns)
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    T = load(S166/'MAPPING.npz')['T']; assert T.shape == (90, 66) and np.count_nonzero(T) == 74
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
    nodes = list(tree.getroot().find('actuator'))
    assert joints[0].get('type') == 'free' and len(joints) == 43 and len(q0) == 49
    ranges = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    indices = np.r_[2, np.arange(7, 49)].astype(int)
    radius = np.full(42, .35)
    lower = np.r_[q0[2]*(1-.15), np.maximum(q0[7:]-radius, ranges[:, 0])]
    upper = np.r_[q0[2]*(1+.15), np.minimum(q0[7:]+radius, ranges[:, 1])]
    assert np.all(lower < upper) and np.all(q0[indices] >= lower) and np.all(q0[indices] <= upper)
    width = upper-lower; x0 = (q0[indices]-lower)/width
    dump(out/'PREREG.json', dict(
        task='S176/P1-S2 widened pose-interval screening for static support', start='one S154 release qpos; no additional starts',
        variables='root z + all 42 original leg joints = 43; fixed root xy/orientation and all nonleg qpos',
        bounds=dict(root_z=[float(lower[0]), float(upper[0])], leg_joint_radius_rad=0.35, leg_joint_bounds=[lower[1:].tolist(), upper[1:].tolist()],
                    note='widened once by owner authorization; frozen here, never expanded after seeing results'),
        activation='real 66 pools, gain .5, 16 missing slots zero (unchanged mapping)', per_pose_cost='1 zero-activation probe forward; no inner least squares, no finite differences',
        score='score=max(max|bb[0:6]|, LPminmax over rows 6..47 with x in [0,1]); bb=-(base+J.T f0)',
        why='S175 proved rows0-5 have zero activation authority, so root rows are pure geometry; screening targets the real obstruction',
        search='deterministic coordinate pattern search, step 0.25 of range, halving on no improvement, stop below 2^-6',
        accept='score<=1e-8 then confirmation forward with the optimized activation; independent reload only if that passes',
        budget=dict(total_forward_cap=CAP, reserved_for_confirmation=RESERVED, max_lp_per_pose=1),
        no_step=True, no_neural=True, no_training=True, no_deployment=True, units='native; SI NOT_ESTABLISHED'))
    import mujoco as mj
    from scipy.optimize import lsq_linear
    assert mj.__version__ == '3.9.0'
    for name in ('control', 'act_dyn', 'act_gain', 'act_bias', 'passive', 'sensor'):
        g = getattr(mj, 'get_mjcb_'+name, None); assert g is None or g() is None
    reserve('model_load'); m = mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m) == proof['before']
    m.actuator_dynprm[:, 0] = .001; m.actuator_dynprm[:, 1] = .0004
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = .000025
    assert ns['snapshot'](m) == proof['after']
    assert np.all(m.actuator_actearly == 0) and np.array_equal(m.actuator_actadr, np.arange(90))
    assert np.all(m.actuator_gaintype == int(mj.mjtGain.mjGAIN_MUSCLE)) and np.all(m.actuator_biastype == int(mj.mjtBias.mjBIAS_MUSCLE))
    assert np.all(m.actuator_dyntype == int(mj.mjtDyn.mjDYN_MUSCLE)) and np.all(m.actuator_actnum == 1)
    assert not np.any(m.actuator_forcelimited) and not np.any(m.actuator_actlimited)
    floors = set(meta['floor_geom_ids']); geomleg = {int(k): LEGS.index(v) for k, v in meta['geom_map'].items()}
    depthcap = 0.03442910871489635
    def forward(q, pool, kind):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for n in STATE: getattr(d, n)[:] = z[n][0]
        for n in OPTIONAL:
            if n in z: getattr(d, n)[:] = z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = T@pool; d.ctrl[:] = d.act; d.qacc_warmstart[:] = 0
        init = {n: np.asarray(getattr(d, n)).copy() for n in (*STATE, *OPTIONAL)}
        reserve(kind); mj.mj_forward(m, d)
        assert all(np.array_equal(getattr(d, n), init[n]) for n in (*STATE, *OPTIONAL))
        c = ns['constraint_record'](mj, m, d, ledger['forward']); c.update(ns['moment_record'](m, d))
        for n in ('qacc', 'act_dot', 'actuator_force', 'actuator_length',
                  'qfrc_actuator', 'qfrc_passive', 'qfrc_bias', 'efc_D', 'efc_aref', 'efc_type', 'efc_frictionloss', 'qM'):
            c[n] = np.asarray(getattr(d, n)).copy()
        assert 'actuator_moment' in c and c['actuator_moment'].shape == (90, 48)
        assert 'efc_J_dense' in c and c['efc_J_dense'].shape[0] == len(c['efc_type'])
        c['contact_dist'] = np.array([v.dist for v in d.contact]); c['contact_friction'] = np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5)
        M = np.empty((48, 48)); mj.mj_fullM(m, M, d.qM); c['M'] = M
        return c
    def affine(c):
        types = c['efc_type']; assert np.all(np.isin(types, [0, 3, 4, 5, 6])), 'unsupported native row type'
        assert not np.any(c['efc_frictionloss'])
        f0 = c['efc_D']*c['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        gain = np.empty(90); bias = np.empty(90)
        for j, node in enumerate(nodes): _, gain[j], bias[j] = ns['evaluate'](c['actuator_length'][j], 0., 0., node)
        moment = c['actuator_moment']; B = moment.T*gain[None, :]; base = moment.T@bias+c['qfrc_passive']-c['qfrc_bias']
        return B, base, f0, moment
    def geometry(c, f0):
        depth = 0.; nonfoot = 0.; bearing = set(); margins = []
        for ci, (g1, g2) in enumerate(c['contact_geom']):
            g1 = int(g1); g2 = int(g2); leg = geomleg.get(g2 if g1 in floors else g1) if ((g1 in floors) != (g2 in floors)) else None
            pen = max(0., -float(c['contact_dist'][ci])); addr = int(c['contact_efc_address'][ci]); dim = int(c['contact_dim'][ci])
            if leg is None: nonfoot = max(nonfoot, pen)
            else: depth = max(depth, pen)
            if addr < 0: continue
            count = int(c['contact_row_count'][ci])
            if dim == 1: local = np.r_[f0[addr], 0., 0.]
            else:
                assert dim == 3 and count == 4
                e = f0[addr:addr+4]; mu = c['contact_friction'][ci, :2]; local = np.r_[e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])]
            margins.append(float(local[0]-sum(abs(local[1:]/mu))))
            if leg is not None and local[0] > 0 and c['contact_dist'][ci] <= 0: bearing.add(leg)
        return depth, nonfoot, len(bearing), min(margins, default=0.)
    J = None
    def score_pose(x, kind='probe_forward', save=None):
        q = q0.copy(); q[indices] = lower+x*width
        c = forward(q, np.zeros(66), kind)
        B, base, f0, moment = affine(c); Jm = c['efc_J_dense']
        bb = -(base+Jm.T@f0); rootviol = float(abs(bb[:6]).max())
        A = (B@T)[6:, :]; tgt = bb[6:]; n = A.shape[1]
        r = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                    b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs',
                    options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        ledger['lp_solves'] += 1; assert r.success, r.message
        t = float(r.x[-1]); pools = r.x[:n]
        depth, nonfoot, nlegs, margin = geometry(c, f0)
        ledger['pose_evaluations'] += 1
        rec = dict(evaluation=ledger['pose_evaluations'], forward=ledger['forward'], qpos=q, score=max(rootviol, t), root_violation=rootviol,
                   joint_minimax=t, lp_activation=np.asarray(pools), B=B, base=base, f0=f0, J=Jm, contact_geom=c['contact_geom'],
                   contact_dist=c['contact_dist'], contact_dim=c['contact_dim'], contact_efc_address=c['contact_efc_address'],
                   contact_row_count=c['contact_row_count'], efc_type=c['efc_type'], foot_depth=depth, nonfoot_depth=nonfoot,
                   bearing_legs=nlegs, friction_margin=margin, qacc_probe=c['qacc'], act_dot_probe=c['act_dot'])
        if save is not None: np.savez_compressed(save, **rec)
        return rec
    rows = []; cache = {}; cache_hits = 0; best = None
    def probe_pose(x, note):
        nonlocal cache_hits, best
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in cache:
            cache_hits += 1
            return cache[key]
        rec = score_pose(x)
        cache[key] = rec
        rows.append(dict(evaluation=rec['evaluation'], note=note, score=rec['score'], root_violation=rec['root_violation'],
                         joint_minimax=rec['joint_minimax'], foot_depth=rec['foot_depth'], nonfoot_depth=rec['nonfoot_depth'],
                         bearing_legs=rec['bearing_legs'], friction_margin=rec['friction_margin'], qpos=rec['qpos'].tolist()))
        dump(out/'SCREEN_EVALUATIONS.json', rows); dump(out/'CALL_LEDGER.json', ledger)
        if best is None or rec['score'] < best['score']:
            best = dict(rec)
            np.savez_compressed(out/'BEST_SCREENED_POSE.npz', **rec)
            dump(out/'BEST_SCREENED_POSE.json', {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)})
        return rec
    status = 'SCREEN_FAILED'
    try:
        gate = load(S173/'EVAL_001.npz')
        g = score_pose(x0, 'gate_forward'); ledger['gate_forward'] = 1
        best = dict(g); np.savez_compressed(out/'BEST_SCREENED_POSE.npz', **g)
        assert np.max(abs(g['qpos']-gate['qpos'])) <= 1e-12
        gate_err = dict(B=float(abs(g['B']-gate['B']).max()), base=float(abs(g['base']-gate['base']).max()), f0=float(abs(g['f0']-gate['f0']).max()))
        dump(out/'EQUIVALENCE_GATE.json', dict(reference='S173 RUN/EVAL_001.npz probe at release pose', errors=gate_err, tolerance=1e-10, passed=bool(max(gate_err.values()) <= 1e-10)))
        assert max(gate_err.values()) <= 1e-10, 'reimplementation does not reproduce frozen production probe'
        cache[np.asarray(x0, dtype=np.float64).tobytes()] = g
        rows.append(dict(evaluation=g['evaluation'], note='release_gate', score=g['score'], root_violation=g['root_violation'], joint_minimax=g['joint_minimax'],
                         foot_depth=g['foot_depth'], nonfoot_depth=g['nonfoot_depth'], bearing_legs=g['bearing_legs'], friction_margin=g['friction_margin'], qpos=g['qpos'].tolist()))
        dump(out/'SCREEN_EVALUATIONS.json', rows)
        x = x0.copy(); cur = dict(g); cur_score = g['score']; step = .25; sweeps = 0; accepted_moves = 0
        while step >= 1/64 and ledger['forward'] < CAP:
            sweeps += 1; improved = False
            for j in range(43):
                for direction in (+1, -1):
                    y = x.copy(); y[j] = np.clip(y[j]+direction*step, 0., 1.)
                    if y[j] == x[j]: continue
                    rec = probe_pose(y, f'sweep{sweeps}_coord{j}_step{step:+.4f}')
                    if rec['score'] < cur_score - 1e-15:
                        cur = rec; cur_score = rec['score']; x = y; improved = True; accepted_moves += 1
                        break
                if ledger['forward'] >= CAP: break
            if not improved: step /= 2
        dump(out/'SEARCH_TERMINATION.json', dict(reason='step below 2^-6 or forward cap', step=step, sweeps=sweeps, accepted_moves=accepted_moves,
                                                 cache_hits=cache_hits, forwards=ledger['forward']))
        if best['score'] <= 1e-8:
            q = best['qpos']; act = T@best['lp_activation']
            conf = forward(q, best['lp_activation'], 'confirmation_forward'); ledger['confirmation_forward'] = 1
            Bc, basec, f0c, _ = affine(conf); resc = Bc@act+basec+conf['efc_J_dense'].T@f0c
            depth, nonfoot, nlegs, margin = geometry(conf, f0c)
            ok = bool(abs(resc).max() <= 1e-8 and abs(conf['qacc']).max() <= 1e-6 and depth <= depthcap and nonfoot <= 1e-12 and nlegs >= 3 and margin >= -1e-10 and not np.any(conf['act_dot']))
            np.savez_compressed(out/'CONFIRMATION.npz', qpos=q, activation=act, residual48=resc, qacc=conf['qacc'], act_dot=conf['act_dot'], B=Bc, base=basec, f0=f0c, J=conf['efc_J_dense'],
                                foot_depth=depth, nonfoot_depth=nonfoot, bearing_legs=nlegs, friction_margin=margin)
            dump(out/'CONFIRMATION.json', dict(passed=ok, max_abs_residual=float(abs(resc).max()), max_abs_qacc=float(abs(conf['qacc']).max()),
                                               foot_depth=depth, nonfoot_depth=nonfoot, bearing_legs=nlegs, friction_margin=margin))
            status = 'CONFIRMED_STATIC_CANDIDATE' if ok else 'SCREEN_PASSED_CONFIRMATION_FAILED'
    except BudgetStop:
        status = 'SCREEN_BUDGET_STOP'
    except Exception:
        dump(out/'FAILED.json', dict(traceback=traceback.format_exc(), ledger=ledger, no_retry=True)); status = 'FAILED_STOP_NO_RETRY'; raise
    finally:
        dump(out/'SUMMARY.json', dict(status=status, ledger=ledger, best={k: v for k, v in (best or {}).items() if k not in ('B', 'base', 'f0', 'J', 'qpos', 'lp_activation', 'qacc_probe', 'act_dot_probe')},
                                       units='native; SI NOT_ESTABLISHED', trajectory=False, integration=False, neural_controller=False, deployment=False,
                                       limitation='66 constant pools are an interface upper bound; root rows have zero activation authority; no standing/walking claim'))
        after = {p: sha(p) for p in hashes}
        dump(out/'INPUT_HASHES_AFTER.json', after)
        if rows:
            with (out/'SCREEN_EVALUATIONS.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'qpos']); w.writeheader()
                for r in rows: w.writerow({k: v for k, v in r.items() if k != 'qpos'})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    main(a.output or BASE/'s176-screened-pose-search'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
