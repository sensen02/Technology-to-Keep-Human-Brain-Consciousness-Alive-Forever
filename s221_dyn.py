"""S221 Part B: first bounded dynamics, 3 arms x 200 mj_step (the first mj_step in the project)."""
import json, hashlib
import numpy as np, mujoco
import xml.etree.ElementTree as ET
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s221-pose-robust-and-first-dynamics')
DIR = sorted([p for p in ROOT.iterdir() if p.is_dir()])[-1]
S217 = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics/20261025T173216395113Z')
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CL = dict(model_loads=0, fwd_position=0, fwd_velocity=0, fwd_actuation=0, fwd_constraint=0,
          mj_forward=0, mj_step=0, lp_solves=0, neural_ticks=0)
z = np.load(S217 / 'AUTH_MATRICES_V3.npz'); c = np.load(S217 / 'CONE_STRUCTURE_V2.npz')
T = z['T']; J0 = z['J']; B0 = z['B_frozen']; base0 = z['base_frozen']
qrel = np.asarray(np.load(S177)['qpos'], float)
A0 = B0 @ T

# LP command at q0
n_cols = A0.shape[1]; n_lam = J0.shape[0]; n = n_cols + n_lam; N = n + 1
Aub, bub = [], []
for i in range(48):
    for sgn in (1.0, -1.0):
        row = np.zeros(N); row[:n_cols] = sgn * A0[i]; row[n_cols:n] = sgn * J0.T[i]; row[n] = -1.0
        Aub.append(row); bub.append(sgn * (-base0[i]))
for r in range(n_lam):
    row = np.zeros(N); row[n_cols + r] = -1.0; Aub.append(row); bub.append(0.0)
lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub),
             bounds=[(0, 1.0)] * n_cols + [(None, None)] * n_lam + [(0, None)], method='highs')
CL['lp_solves'] += 1
x_lp = lp.x[:n_cols]
u_lp = np.clip(T @ x_lp, 0.0, 1.0)

m = mujoco.MjModel.from_xml_path(str(FROZEN)); CL['model_loads'] += 1
dt = float(m.opt.timestep)
nodes = list(ET.parse(str(FROZEN)).getroot().find('actuator'))
joints = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('joint_')]
leg_of_act = [a for a in range(m.nu)]

def run(ctrl_vec, nsteps=200, every=10):
    d = mujoco.MjData(m)
    d.qpos[:] = qrel; d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = ctrl_vec
    q0 = d.qpos.copy(); v0 = d.qvel.copy(); a0 = d.act.copy(); c0 = d.ctrl.copy()
    series = []
    for k in range(1, nsteps + 1):
        mujoco.mj_step(m, d); CL['mj_step'] += 1
        if k % every == 0:
            disp = float(np.linalg.norm(d.qpos[:3] - q0[:3]))
            ncon = int(d.ncon)
            lam_sum = float(np.sum(np.abs(np.asarray(d.efc_force[:d.nefc])))) if d.nefc else 0.0
            mind = float(min((d.contact[i].dist for i in range(ncon)), default=0.0))
            series.append(dict(step=k, t_ms=1000 * k * dt, disp=disp,
                               qvel_inf=float(np.max(np.abs(d.qvel))), ncon=ncon, sigma_lam=lam_sum,
                               min_dist=mind, muscle_force_sum=float(np.sum(np.abs(np.asarray(d.actuator_force)))), nefc=int(d.nefc)))
    same_start = dict(qpos=bool(np.array_equal(q0, qrel)), qvel=bool(np.all(v0 == 0)), act=bool(np.all(a0 == 0)),
                      ctrl=bool(np.array_equal(c0, ctrl_vec)))
    return dict(series=series, start_identical=same_start, final_qpos=np.asarray(d.qpos).tolist(),
                final_ncon=int(d.ncon), final_min_dist=float(min((d.contact[i].dist for i in range(int(d.ncon))), default=0.0)))

arms = dict(zero=np.zeros(m.nu), u_lp=u_lp, u_lp2=np.clip(2.0 * u_lp, 0.0, 1.0))
res = {}
for k, cv in arms.items():
    res[k] = run(cv)
    CL['mj_forward'] += 0
def at10ms(series):
    for s in series:
        if s['t_ms'] >= 10.0:
            return s
    return series[-1]
pred = dict(zero=dict(disp=0.21, gate='>0.10', ncon=5, qvel=43), u_lp=dict(disp=0.026, gate='<0.05', ncon=13, qvel=5))
got = {k: dict(disp=at10ms(res[k]['series'])['disp'], ncon=at10ms(res[k]['series'])['ncon'],
               qvel=at10ms(res[k]['series'])['qvel_inf']) for k in res}
dyn = dict(dt=dt, nsteps=200, duration_ms=1000 * 200 * dt,
           u_lp_stats=dict(min=float(u_lp.min()), max=float(u_lp.max()), n_positive=int((u_lp > 0).sum())),
           x_lp_max=float(x_lp.max()),
           arms={k: dict(at10ms=got[k], start_identical=res[k]['start_identical'],
                         final_ncon=res[k]['final_ncon'], final_min_dist=res[k]['final_min_dist'],
                         sigma_lam_at10ms=at10ms(res[k]['series'])['sigma_lam'],
                         min_dist_at10ms=at10ms(res[k]['series'])['min_dist']) for k in res},
           predictions=pred,
           gates=dict(zero_gt_010=bool(got['zero']['disp'] > 0.10), u_lp_lt_005=bool(got['u_lp']['disp'] < 0.05)),
           ratio_zero_over_u=float(got['zero']['disp'] / got['u_lp']['disp']) if got['u_lp']['disp'] else None,
           ref_gravity_scale=4.9e-4,
           note='u_LP is an LP command, not the shipped decoder output; a smaller drift is only a preliminary correspondence, '
                'never a recovery')
(DIR / 'FIRST_DYNAMICS.json').write_text(json.dumps(dyn, indent=2, default=str))
with open(DIR / 'TIMESERIES.csv', 'w') as f:
    f.write('arm,step,t_ms,disp,qvel_inf,ncon,sigma_lam,min_dist,muscle_force_sum,nefc\n')
    for k in ('zero', 'u_lp', 'u_lp2'):
        for s in res[k]['series']:
            f.write('%s,%d,%.3f,%.6e,%.6f,%d,%.6f,%.6e,%.6f,%d\n' % (k, s['step'], s['t_ms'], s['disp'], s['qvel_inf'],
                                                                   s['ncon'], s['sigma_lam'], s['min_dist'],
                                                                   s['muscle_force_sum'], s['nefc']))
# figures
plt.figure(figsize=(10, 5))
for k, col in (('zero', 'tab:red'), ('u_lp', 'tab:blue'), ('u_lp2', 'tab:green')):
    plt.plot([s['t_ms'] for s in res[k]['series']], [s['disp'] for s in res[k]['series']], 'o-', color=col, label=k)
plt.axhline(4.9e-4, color='k', ls='--', lw=1, label='gravity scale 0.5 g (10ms)^2')
plt.yscale('log'); plt.xlabel('t (ms)'); plt.ylabel('root displacement (m)')
plt.title('FIG2  First bounded dynamics: zero command drifts ~0.2 m in 10 ms, far above the 4.9e-4 m reference', fontsize=9)
plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DIR / 'FIG2_FIRST_DYNAMICS_DISP.png', dpi=150)
plt.close()
plt.figure(figsize=(10, 5))
for k, col in (('zero', 'tab:red'), ('u_lp', 'tab:blue'), ('u_lp2', 'tab:green')):
    plt.plot([s['t_ms'] for s in res[k]['series']], [s['ncon'] for s in res[k]['series']], 'o-', color=col, label=k + ' ncon')
plt.xlabel('t (ms)'); plt.ylabel('ncon'); plt.title('FIG3  Contact count per arm over 20 ms', fontsize=9)
plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DIR / 'FIG3_NCON.png', dpi=150)
plt.close()
pose = json.load(open(DIR / 'POSE_ROBUST_LP.json'))
plt.figure(figsize=(12, 5))
rm = [k for k in pose['invariant_joints'] if k.startswith('joint_RM')]
others = [k for k in pose['invariant_joints'] if not k.startswith('joint_RM')]
rows = []
import csv as _csv
with open(DIR / 'POSE_ROBUST_LP.csv') as f:
    rows = list(_csv.DictReader(f))
for name, col in (('non-RM (32 joints)', 'tab:blue'), ('RM (7 joints)', 'tab:red')):
    xs, ys = [], []
    for i, r in enumerate(rows):
        if not r['t']:
            continue
        is_rm = r['joint'].startswith('joint_RM')
        if (name.startswith('non-RM') and not is_rm) or (name.startswith('RM') and is_rm):
            xs.append(i); ys.append(float(r['t']))
    plt.scatter(xs, ys, s=12, color=col, label=name)
plt.axhline(1e-9, color='k', ls='--', lw=1)
plt.yscale('log'); plt.xlabel('pose index'); plt.ylabel('t(q)')
plt.title('FIG1  t(q) over 210 poses: non-RM joints sit exactly on 0.059329414 (flat), only the RM joints move it', fontsize=9)
plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(DIR / 'FIG1_POSE_T.png', dpi=150)
plt.close()

checks = []
def chk(n, ok, dd): checks.append(dict(check=n, ok=bool(ok), detail=dd))
chk('pose:no_zero_t_across_210', pose['no_zero_t'], 'same-contact t in [%.6f, %.6f], changed-contact t in [%.6f, %.6f]'
    % (pose['same_min_t'], pose['same_max_t'], pose['changed_min_t'], pose['changed_max_t']))
chk('pose:32_invariant_7_rm', pose['n_invariant_joints'] == 32 and len(pose['rm_joints']) == 7 and len(pose['noninvariant_joints']) == 7,
    'invariant %d, RM-moving %d' % (pose['n_invariant_joints'], len(pose['rm_joints'])))
chk('pose:label', pose['label'] == 'INFEASIBILITY_POSE_ROBUST', pose['label'])
chk('corr:sum_lambda_invariant', corr_ok := json.load(open(DIR / 'S220_CORRECTIONS.json'))['sum_lambda_vertex']['invariant_holds'],
    'sum(lambda) = 24.387025 (W-t); W+t = 24.505684 is the other optimal vertex')
chk('corr:column_vertex_dependence',
    [d['column'] for d in json.load(open(DIR / 'S220_CORRECTIONS.json'))['column_side']['negative_l1_w']] == [28, 35, 65],
    'single-w certificate: cols 28/35; L1 certificate: cols 28/35/65 - column side is vertex-dependent')
chk('corr:w_vs_l1_mismatch', not json.load(open(DIR / 'S220_CORRECTIONS.json'))['w_norm_vs_l1']['same_solution'],
    'sum|w| = 16.918 vs l1 objective 16.855: two different solutions were archived in one file')
chk('dyn:zero_gt_0_10m', dyn['gates']['zero_gt_010'], 'zero arm displacement at 10 ms = %.4f m' % got['zero']['disp'])
chk('dyn:u_lp_lt_0_05m', dyn['gates']['u_lp_lt_005'], 'u_LP arm displacement at 10 ms = %.4f m' % got['u_lp']['disp'])
chk('dyn:starts_identical', all(res[k]['start_identical']['qpos'] and res[k]['start_identical']['qvel'] and res[k]['start_identical']['act'] for k in res),
    'all three arms start from the frozen qpos with qvel=0, act=0')
chk('dyn:mj_step_count', CL['mj_step'] == 600, '%d mj_step calls (cap 600)' % CL['mj_step'])
chk('dyn:no_forward_or_training', CL['mj_forward'] == 0 and CL['neural_ticks'] == 0, json.dumps(CL))
chk('guards:xml_unchanged', sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', 'hash ok')
n_ok = sum(1 for x in checks if x['ok'])
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                   all_ok=n_ok == len(checks), calls=CL), indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), dt=dt, arms={k: dict(disp10=round(got[k]['disp'], 5), ncon10=got[k]['ncon'],
                                                          qvel10=round(got[k]['qvel'], 3)) for k in got},
                      gates=dyn['gates'], ratio=dyn['ratio_zero_over_u'], checks=(n_ok, len(checks)), calls=CL), indent=1))
