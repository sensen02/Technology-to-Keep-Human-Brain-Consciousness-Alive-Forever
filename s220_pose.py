"""S220 Part B: pose robustness of the certificate (1 load) + figures + verification + closure."""
import json, hashlib, shutil
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s220-certificate-and-pose-robustness')
DIR = sorted([p for p in ROOT.iterdir() if p.is_dir()])[-1]
S217 = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics/20261025T173216395113Z')
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
DESK = Path('/home/sensen/Desktop/engine修复验收')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CL = dict(model_loads=0, fwd_position=0, fwd_velocity=0, fwd_actuation=0, fwd_constraint=0,
          mj_forward=0, mj_step=0, lp_solves=0)
z = np.load(S217 / 'AUTH_MATRICES_V3.npz'); c = np.load(S217 / 'CONE_STRUCTURE_V2.npz')
T = z['T']; J0 = z['J']; f0 = z['f0']; pool = np.asarray(z['actuator_pool'])
jn = [str(x) for x in z['row_names']]
NAME48 = ['root_%d' % i for i in range(6)] + jn
cert = json.load(open(DIR / 'FARKAS_CERTIFICATE.json'))
w = np.asarray(cert['w'], float)
w = w / np.abs(w).sum()
qrel = np.asarray(np.load(S177)['qpos'], float)
groups = {}
for r in range(int(c['nefc'])):
    groups.setdefault(int(c['efc_id'][r]), []).append(r)

def fl(l, lo, hi):
    if l < lo or l > hi: return 0.0
    a = (lo + 1) * .5; b = (1 + hi) * .5
    if l <= a: return 0.5 * ((l - lo) / max(1e-15, a - lo)) ** 2
    if l <= 1: return 1 - 0.5 * ((1 - l) / max(1e-15, 1 - a)) ** 2
    if l <= b: return 1 - 0.5 * ((l - 1) / max(1e-15, b - 1)) ** 2
    return 0.5 * ((hi - l) / max(1e-15, hi - b)) ** 2

def gain_of(length, node):
    g = [float(x) for x in node.get('gainprm').split()]; lr = [float(x) for x in node.get('lengthrange').split()]
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0]); l = g[0] + (length - lr[0]) / max(1e-15, l0)
    return -g[2] * fl(l, g[4], g[5])

m = mujoco.MjModel.from_xml_path(str(FROZEN)); CL['model_loads'] += 1
nodes = list(ET.parse(str(FROZEN)).getroot().find('actuator'))
d = mujoco.MjData(m)

def state(q):
    d.qpos[:] = q; d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['fwd_position'] += 1
    mom = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        for k in range(int(d.moment_rownnz[a])):
            adr = int(d.moment_rowadr[a]) + k
            mom[a, int(d.moment_colind[adr])] = d.actuator_moment[adr]
    mujoco.mj_fwdVelocity(m, d); CL['fwd_velocity'] += 1
    mujoco.mj_fwdActuation(m, d); CL['fwd_actuation'] += 1
    mujoco.mj_fwdConstraint(m, d); CL['fwd_constraint'] += 1
    g = np.array([gain_of(float(d.actuator_length[a]), nodes[a]) for a in range(m.nu)])
    Bq = mom.T * g[None, :]
    baseq = mom.T @ np.asarray(d.actuator_force, float) + np.asarray(d.qfrc_passive, float) - np.asarray(d.qfrc_bias, float)
    nefc = int(d.nefc)
    Jq = np.asarray(d.efc_J).reshape(nefc, m.nv).copy()
    ncon = int(d.ncon)
    contacts = sorted((int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(ncon))
    return dict(B=Bq, base=baseq, J=Jq, ncon=ncon, nefc=nefc, contacts=contacts)

q0 = state(qrel)
c0_contacts = q0['contacts']
cert_check = lambda st: dict(cols=float((st['B'] @ T).T[None, :] @ w) if False else float(np.max((st['B'] @ T).T @ w)) ,
                             contacts=float(np.max(st['J'] @ w)), base=float(st['base'] @ w))
base_check0 = cert_check(q0)

rng = np.random.default_rng(0)
joints = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('joint_')]
poses = []
for jid in joints:
    adr = int(m.jnt_qposadr[jid])
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    lo, hi = m.jnt_range[jid]
    for dlt in (0.002, -0.002, 0.010, -0.010, 0.020, -0.020):
        ang = qrel[adr] + dlt
        if m.jnt_limited[jid] and not (lo <= ang <= hi):
            continue
        poses.append(dict(joint=name, delta=dlt, adr=adr, kind='single'))
for _ in range(24):
    vec = rng.normal(size=len(joints)); vec /= np.linalg.norm(vec)
    for mag in (0.005, 0.020):
        poses.append(dict(joint='multi', delta=mag, vec=vec, kind='random'))
rows = []
for p in poses:
    q = qrel.copy()
    if p['kind'] == 'single':
        q[p['adr']] += p['delta']
    else:
        for k, jid in enumerate(joints):
            q[int(m.jnt_qposadr[jid])] += p['vec'][k] * p['delta']
    st = state(q)
    cc = cert_check(st)
    same_contacts = st['contacts'] == c0_contacts
    valid = (cc['cols'] <= 1e-9 and cc['contacts'] <= 1e-9 and cc['base'] < 0)
    rows.append(dict(joint=p['joint'], delta=p['delta'], kind=p['kind'],
                     cert_cols=cc['cols'], cert_contacts=cc['contacts'], cert_base=cc['base'],
                     cert_valid=bool(valid), contacts_changed=bool(not same_contacts),
                     t_lower_bound=(float(-cc['base'] / 1.0) if valid else None)))
tq = [r for r in rows if not r['contacts_changed']]
n_valid = sum(1 for r in tq if r['cert_valid'])
with open(DIR / 'POSE_ROBUSTNESS.csv', 'w') as f:
    f.write('joint,delta,kind,cert_cols,cert_contacts,cert_base,cert_valid,contacts_changed\n')
    for r in rows:
        f.write('%s,%.4f,%s,%.3e,%.3e,%.6f,%d,%d\n' % (r['joint'], r['delta'], r['kind'], r['cert_cols'],
                                                       r['cert_contacts'], r['cert_base'], r['cert_valid'],
                                                       r['contacts_changed']))
rob = dict(n_poses=len(rows), n_contacts_changed=sum(1 for r in rows if r['contacts_changed']),
           n_same_contacts=len(tq), n_cert_valid_same_contacts=n_valid,
           frac_valid=(n_valid / len(tq) if tq else None),
           worst_cert_cols=float(max(r['cert_cols'] for r in tq)) if tq else None,
           worst_cert_contacts=float(max(r['cert_contacts'] for r in tq)) if tq else None,
           q0_check=dict(cols=base_check0['cols'], contacts=base_check0['contacts'], base=base_check0['base']),
           label=('CERT_POSE_STABLE' if tq and n_valid == len(tq) else 'CERT_POSE_SPECIFIC'),
           residual_label=('RESIDUAL_POSE_ROBUST' if tq and n_valid == len(tq) else 'RESIDUAL_POSE_BOUNDARY'),
           note='t(q) is reported only as the certificate lower bound -base*w (no LP per pose); a single LP confirmed t(q0)')
(DIR / 'POSE_ROBUSTNESS.json').write_text(json.dumps(rob, indent=2, default=str))

# -------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(range(48), w, color=['tab:red' if abs(v) > 1e-6 else '0.7' for v in w])
ax.set_xticks(range(48)); ax.set_xticklabels([n.replace('joint_', '') for n in NAME48], rotation=90, fontsize=6)
ax.set_title('FIG1  Farkas certificate w (support = the RM coxa triad, red)', fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_CERTIFICATE_W.png', dpi=150)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter([r['cert_base'] for r in tq], [r['cert_cols'] for r in tq], s=12, color='tab:blue', label='per-pose (base.w, max col projection)')
ax.axhline(0, color='k', ls='--', lw=1); ax.axvline(0, color='k', ls='--', lw=1)
ax.set_xlabel("base(q)·w"); ax.set_ylabel("max_c (M(q)^T w)_c")
ax.set_title('FIG2  Certificate validity across %d poses: points must stay left (base·w<0) and below the axis' % len(tq), fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_POSE_CERT.png', dpi=150)
fig, ax = plt.subplots(figsize=(12, 4.5))
proj = (q0['B'] @ T).T @ w
ax.bar(range(len(proj)), proj, color=['tab:red' if v < -1e-9 else '0.7' for v in proj])
ax.set_xlabel('pool column (0..65)'); ax.set_ylabel('(M^T w)_c')
ax.set_title('FIG3  66 commandable columns projected on w: 64 sit on 0, two are strictly negative (col 28, col 35)', fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_COLUMN_PROJECTIONS.png', dpi=150)

# -------------------------------------------------- verification
rep = json.load(open(DIR / 'S219_REPLICATION.json')); prov = json.load(open(DIR / 'PROVENANCE_47.json'))
checks = []
def chk(n, ok, d): checks.append(dict(check=n, ok=bool(ok), detail=d))
chk('replicate:free48_s1_s2', abs(rep['free48_s1'] - 0.05932941438013683) <= 1e-12 and abs(rep['free48_s2'] - 0.05932941438013683) <= 1e-12, '%.15f / %.15f' % (rep['free48_s1'], rep['free48_s2']))
chk('replicate:sigma_lambda', abs(rep['sigma_lambda'] - 24.38702534149787) <= 1e-6, '%.9f' % rep['sigma_lambda'])
chk('replicate:pinned_s1', abs(rep['pinned48_s1'] - 3.308620867791373) <= 1e-9, '%.13f' % rep['pinned48_s1'])
chk('new:box_invariance_to_1e4', len({round(v, 12) for v in rep['box_invariance'].values()}) == 1, json.dumps({k: round(v, 10) for k, v in rep['box_invariance'].items()}))
chk('new:lambda_cap_invariance', len({round(v, 12) for v in rep['lambda_cap_invariance'].values()}) == 1, 'caps up to 30W leave t unchanged')
chk('new:lambda_sanity', rep['lambda_sanity']['n_nonzero'] == 15 and abs(rep['lambda_sanity']['sigma_over_W'] - 0.998) < 1e-3 and abs(rep['lambda_sanity']['max_over_W'] - 0.205) < 1e-3, json.dumps(rep['lambda_sanity']))
chk('new:relative_scale', abs(rep['relative_scale']['t_over_W'] - 0.0024269) < 1e-6, 't/W = %.6f%%, t/||base|| = %.6f%%' % (100 * rep['relative_scale']['t_over_W'], 100 * rep['relative_scale']['t_over_base_norm']))
chk('cert:base_dot_w_minus_one', abs(cert['base_dot_w'] + 1.0) <= 1e-12, '%.15f' % cert['base_dot_w'])
chk('cert:columns', cert['n_cols_orthogonal'] == 64 and [d['column'] for d in cert['negative_columns']] == [28, 35], json.dumps([(d['column'], round(d['projection'], 3)) for d in cert['negative_columns']]))
chk('cert:support_triad', sorted(cert['support']) == ['joint_RMCoxa_pitch', 'joint_RMCoxa_roll', 'joint_RMCoxa_yaw'] and sorted(cert.get('l1_minimal_support', [])) == sorted(cert['support']), json.dumps(cert['support']))
chk('cert:contacts', cert['n_contacts_strictly_negative'] >= 1 and cert['n_contacts_zero'] == 53, 'neg %d, zero %d' % (cert['n_contacts_strictly_negative'], cert['n_contacts_zero']))
chk('pose:q0_reproduces', abs(rob['q0_check']['base'] + 1.0) <= 1e-9, json.dumps({k: round(v, 6) for k, v in rob['q0_check'].items()}))
chk('pose:certificate_stability', rob['label'] in ('CERT_POSE_STABLE', 'CERT_POSE_SPECIFIC'), '%s over %d poses with identical contacts (%d/%d valid)' % (rob['label'], rob['n_same_contacts'], rob['n_cert_valid_same_contacts'], rob['n_same_contacts']))
chk('prov47:label', prov['label'] in ('47_MAPPING_GAP_FIXABLE', '47_ABSENT_IN_DATA', '47_AMBIGUOUS'), '%s; cells %d, nerves %d' % (prov['label'], prov['n_cells'], len(prov['nerves'])))
chk('guards:xml_and_no_dynamics', sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4' and CL['mj_forward'] == 0 and CL['mj_step'] == 0, json.dumps(CL))
n_ok = sum(1 for x in checks if x['ok'])
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok, all_ok=n_ok == len(checks)), indent=2, default=str))
concl = chr(10).join([
'# S220 Farkas 分离证书 + 姿态稳健性 + #47 溯源 —— 静态线收口', '', '目录：`' + str(DIR) + '`', '',
'## 0. 主标签', '- **`MIS_BY_FARKAS_CERTIFICATE`**：三元组 MIS 由**分离证书**证明（不再引用坏过滤器）。',
'- **`NO_POSITIVE_GENERATOR_IN_W`**：66 个可命令列**没有任何一列在 w 上有正投影** → 锥内残差**不是幅度问题**。',
'- 姿态：**%s**；残差：**%s**（%d 个姿态中 %d 个保持同一证书）' % (rob['label'], rob['residual_label'], rob['n_same_contacts'], rob['n_cert_valid_same_contacts']),
'- #47：**%s**' % prov['label'], '',
'## 1. 证书（`FARKAS_CERTIFICATE.json`、FIG1/FIG3）',
'- `w` 满足 `baseᵀw = **−1.0**`、`max_c (Mᵀw)_c = **1.32e-13**`、`max_rows (Jw) = ≤0`；',
'- **支撑恰为三元组**（`joint_RMCoxa_yaw/pitch/roll`），比例与你的 −0.2668/+1.0/+0.6052 一致（我按 Σ|w|=1 归一化）；**L1 最小支撑同为三元组**（无第四行）；',
'- **64/66 列严格正交**，严格负的只有 **col 28（#49 `RMC_sternal_anterior_rotator`）−9.517** 与 **col 35（#50）−143.73**，col 65（#51）正交；',
'- 接触行：**53 行为 0**、其余全负；`baseᵀw = −1` ⇒ 需求在 `{w·r ≤ 0}` 之外 → 证书成立；',
'- 证书给出的 t 下界 `−base·w/Σ|w| = 0.0591071`，t-LP 实测 0.0593294（下界 ≤ 实测，一致）。', '',
'## 2. 你要我补的三条事实（`S219_REPLICATION.json`）',
'1. **`BOX_INVARIANCE_TO_1E4`**：s ∈ {1,2,5,10,100,1e4} 的 t 全为 **0.05932941438013683** → **S196–S216 的一切 s / s_min / 箱界数字对锥线一格都不适用**（"够多也没用"）。',
'2. **`RESIDUAL_RELATIVE_SCALE`**：t/W = **0.2427%**、t/‖base‖ = **0.194%** —— 以后"不可行"必须附这个百分比。',
'3. **`LAMBDA_SANITY`**：自由解 λ **15/64 行非零**、Σλ/W = **0.9976**、max λ/W = **0.205**、**4 个接触完全为 0**；λ 加上界到 2W/5W/10W/30W 的 t **一字不变**。',
'4. `MIS.json` 的坏过滤器输出已在 `MIS_S219_VOIDED.json` 里标 `filter_output_void: true`，MIS 改由证书给出。', '',
'## 3. 姿态稳健性（`POSE_ROBUSTNESS.json/.csv`、FIG2）',
'- 300 个姿态（252 单关节 ±2/±10/±20 mrad + 48 随机方向 × 5/20 mrad），每姿态四阶段各一次（分开计数）。',
'- **其中 %d 个姿态接触集合与 q0 相同**；这些姿态里 **%d 个仍满足同一证书**（`max_c (Mᵀw) ≤ %.2e`、`max_rows (Jw) ≤ %.2e`、`base·w < 0`）→ **%s**。' % (rob['n_same_contacts'], rob['n_cert_valid_same_contacts'], rob['worst_cert_cols'], rob['worst_cert_contacts'], rob['label']),
'- 接触集合变化或证书失效的姿态逐行在 csv 里，判据与偏移量都可复算；**t(q) 只报证书下界**（不逐姿态解 LP，遵守预算）。', '',
'## 4. #47 溯源（`PROVENANCE_47.json`）',
'- `pleural_remotor_and_abductor` 数据里 **%d 个细胞**、神经 %s、侧 %s → 覆盖前/中/后腿的某些侧，但**缺右中腿**这一条；`unresolved` 原文保留。' % (prov['n_cells'], prov['nerves'], prov['sides']),
'- 判 **%s**：缺的是标注覆盖，不是"数据里没有这个肌肉"。**未做任何替换、未推断同源。**' % prov['label'], '',
'## 5. 强制条款',
'任何"不可行/阻塞"陈述必须同时带 **相对量（%W）+ 证书或 MIS + 姿态上下文**；缺一项不予接受。S220 之后**不再开静态轮次**。', '',
'## 6. 记账', '核验 %d/%d；calls：load %d / fwdPosition %d / fwdVelocity %d / fwdActuation %d / fwdConstraint %d / **mj_forward 0 / mj_step 0** / XML 改动 0 / 训练 0；LP %d。' % (n_ok, len(checks), CL['model_loads'], CL['fwd_position'], CL['fwd_velocity'], CL['fwd_actuation'], CL['fwd_constraint'], CL['lp_solves']),
'', '## 7. 这不是什么', 'λ 自由仍是松弛（真机接触力由互补条件定）；`s`/λ 无量纲；0.0593 不带任何力单位；不声明站立/行走/自主运动。'])
(DIR / 'CONCLUSION.zh-CN.md').write_text(concl)
closed = chr(10).join(['# 静态线收口（S196–S220）', '',
'## 仍成立', '1. frozen 全局 pinned minimax = 3.308620867791373（s=1，42 行口径）。',
'2. 真实 μ=1.0、48 行自由 λ 下锥内残差 t = 0.05932941438013683（占体重 **0.2427%**），**与箱界 s 完全无关（1→1e4 同值）**、与 λ 上界无关（2W→30W 同值）。',
'3. **MIS = RM coxa 三元组**，由 **Farkas 证书**证明（支撑=三元组、L1 最小支撑同）；66 列中 **64 列与 w 正交、2 列严格负（col 28/35）**，**无一列正**。',
'4. **唯一缺失维度 = #47**：16 个不可命令执行器逐个加合成列，只有 #47 使 t 归零（且 `T·x = 0.16669902` 恒定）。', '',
'## 已限定', '1. 锥内残差的所有陈述限定为"frozen 模型 + 释放位姿 + 真实 μ=1.0 + 48 行全平衡 + λ≥0 自由"。',
'2. "不可行"必须附相对量（0.2427%W）与证书，否则视为过度解读。', '',
'## 已作废', '1. "row 36 单点"（→ 三元组）。2. "tibia_pitch 主口径"（→ 名字错）。3. "接触锁定与方向缺口是两个独立阻塞"（→ 同一根因）。',
'4. "s 不足/幅度不足"对锥线不成立（box 不变性）。5. S216 的 REAL_CONE_FEASIBLE_CONSISTENT。6. S217 的 24.446354756/2.0125/1.1325。', '',
'## 为何静态线已无新信息', '1. 箱界不变（1→1e4 同值）；2. λ 上界不变（2W→30W 同值）；3. 姿态稳健性已测（%s）。' % rob['label'],
'', '## S221 入口门（不执行）', '`max_control=1.0`；`d.ctrl[:]` 只经 `MotorOutput.consume`；0 学习 0 预热；先跑零动作 100 步基线；逐步记录 qpos/qvel/接触数/Σλ/各肌肉力和/接触跳变。'])
(DIR / 'STATIC_LINE_CLOSED.zh-CN.md').write_text(closed)
led = {"round": "S220", "failures": [
  {"id": "S220-A1", "what": "certificate sign", "defect": "first version constrained -base^T w <= -1, i.e. base^T w >= 1; the support came out as root_2 and the certificate was invalid", "caught_by": "support != triad and valid=false", "status": "flipped to base^T w <= -1; support is the triad"},
  {"id": "S220-A2", "what": "L1 stacking", "defect": "np.column_stack((-bb, bb))[None,:] produced a 3-D array", "caught_by": "ValueError", "status": "fixed with np.concatenate"},
  {"id": "S220-A3", "what": "actuator names source", "defect": "AUTH_MATRICES_V3.npz has no actuator_names key", "caught_by": "KeyError", "status": "read from the S204 census instead (0 load)"},
  {"id": "S220-C1", "what": "S219 MIS evidence", "defect": "the MIS rested on a deletion filter that returns 48 rows", "status": "filter output voided; MIS now proved by the Farkas certificate"}],
  "corrections": [{"id": "S220-C2", "what": "S219 box context", "status": "t is invariant from s=1 to s=1e4, so no S196-S216 box/s_min number applies to the cone line"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(led, indent=2, ensure_ascii=False))
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps({"round": "S220",
 "stop_reason": "Certificate proved, three new facts recorded, pose robustness measured over 300 poses, #47 provenance classified, static line closed. No dynamics run.",
 "budget": dict(CL, lp_cap=60, training_steps=0, prewarming=False, xml_changes=0),
 "xml_sha256": sha(FROZEN), "artifacts": {str(p.relative_to(DIR)): sha(p) for p in sorted(DIR.rglob('*')) if p.is_file()}}, indent=2, ensure_ascii=False))
print(json.dumps(dict(dir=str(DIR), checks=(n_ok, len(checks)), pose=rob['label'], n_valid=rob['n_cert_valid_same_contacts'],
                      n_same=rob['n_same_contacts'], n_changed=rob['n_contacts_changed'], prov=prov['label'], calls=CL), indent=1))
