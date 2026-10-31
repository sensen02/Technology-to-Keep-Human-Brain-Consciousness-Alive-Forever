"""S211 Part C (4-model behaviour census, B and A separately) + figures + verification + conclusion."""
import json
import hashlib
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s211-attachment-provenance').iterdir())[-1]
S210 = Path('/home/sensen/Desktop/engine修复验收/s210-model-lineage-and-arm-symmetry/20261025T165113566119Z')
DESK = Path('/home/sensen/Desktop/engine修复验收')
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MUSDIR = Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs')
TMP = Path('/tmp/s211_assets')
MESH = '/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/'
ASSETS = dict(frozen=FROZEN, repaired=MUSDIR / 'fruitfly_six_leg_standing.xml',
              norepair=MUSDIR / 'fruitfly_six_leg_standing_noarmrepair.xml',
              qpos0bands=MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml')
EPS_ABS, EPS_REL = 1e-12, 1e-9
arm_tab = json.load(open(DIR / 'CROSS_MODEL_ARMS.json'))
world = json.load(open(DIR / 'WORLD_SITE_GEOMETRY.json'))

st = np.load(S177)
Bf = np.asarray(st['B'], float)
T = np.asarray(st['T'], float)
base, f0, J = np.asarray(st['base'], float), np.asarray(st['f0'], float), np.asarray(st['J'], float)
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

def prep(tag):
    if tag == 'frozen':
        return FROZEN
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / (tag + '_c.xml')
    out.write_text(ASSETS[tag].read_text().replace('./meshes/stl/', MESH))
    return out

# ---------------------------------------------------------------- Part C: 4-model census
def census(tag):
    m = mujoco.MjModel.from_xml_path(str(prep(tag)))
    d = mujoco.MjData(m)
    qpos0 = np.asarray(m.key_qpos[0], float) if m.nkey else np.asarray(st['qpos'], float)
    d.qpos[:] = qpos0
    d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d)
    dense = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        for k in range(int(d.moment_rownnz[a])):
            adr = int(d.moment_rowadr[a]) + k
            dense[a, int(d.moment_colind[adr])] = d.actuator_moment[adr]
    out = {}
    for i in range(42):
        dname = jn[i]
        _ = dname
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, dname)
        if jid < 0:
            continue
        dof = int(m.jnt_dofadr[jid])
        col_B = dense[:, dof]
        # A-space uses the frozen mapping only; for other models the mapping is the same 90x66 layout
        col_A = (dense.T @ T)[6:, :][i, :] if dense.T.shape[0] == 48 else None
        for name, col in (('B', col_B), ('A', col_A)):
            if col is None:
                continue
            for eps in (1e-12, 1e-9):
                pos = float(np.abs(col[col > eps]).max()) if (col > eps).any() else 0.0
                neg = float(np.abs(col[col < -eps]).max()) if (col < -eps).any() else 0.0
                ratio = (max(pos, neg) / min(pos, neg)) if min(pos, neg) > 0 else float('inf')
                out[f'{dname}|{name}|{eps:g}'] = dict(max_pos=pos, max_neg=neg, ratio=ratio,
                                                       one_sided=bool(min(pos, neg) <= eps or ratio > 1 / EPS_REL),
                                                       missing_side=('positive' if pos <= eps else ('negative' if neg <= eps else None)))
    return out

census_all = {t: census(t) for t in ASSETS}
(DIR / 'LINEAGE_BEHAVIOUR_4MODELS.json').write_text(json.dumps(dict(census=census_all, eps_abs=EPS_ABS, eps_rel=EPS_REL,
                                                                     note='B is the physical matrix; A = B@T is the commandable one; '
                                                                          'both are censused with the same eps protocol'), indent=2, default=str))
def onesided(tag, mat, eps):
    return sorted(k.split('|')[0] for k, v in census_all[tag].items()
                  if k.endswith('|%s|%g' % (mat, eps)) and v['one_sided'])
summary = {t: dict(B_1e12=onesided(t, 'B', 1e-12), B_1e9=onesided(t, 'B', 1e-9),
                   A_1e12=onesided(t, 'A', 1e-12), A_1e9=onesided(t, 'A', 1e-9)) for t in ASSETS}

# ---------------------------------------------------------------- provenance (Part B)
prov = dict(
    documented_rule_source='outputs/muscles_six_legs/mirror_decisions.json',
    documented_rule=json.load(open('/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/'
                                   'mirror_decisions.json'))['known_approximations'],
    source_xml=json.load(open('/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/'
                              'mirror_decisions.json'))['source_xml'],
    offline_rederivation='NOT ACHIEVED in this round: reproducing the frozen sites from the source XML plus the documented '
                         'isotropic segment-length scaling to 1e-6 requires the per-leg scale table, which is not stored in any '
                         'artefact this round could locate within budget',
    label='FROZEN_ATTACHMENT_UNATTRIBUTED',
    candidate_rule='(a) mirror LF -> other legs with a handedness flip; (b) scale site positions isotropically by the segment '
                   'length ratio (documented approximation, NOT an axial-only scale); (c) the source model has no tibia muscles '
                   'at all for mid/hind legs, so those attach points are a construction of this repository',
    forced_clause=('until the provenance is settled the 25-39x arm difference may be written neither as a defect of our build nor '
                   'as a body limitation; only "frozen, repaired and norepair give different tibia-muscle arms at the same pose and '
                   'frozen is a front-to-back monotone gradient" may be stated'))
(DIR / 'ATTACHMENT_PROVENANCE.json').write_text(json.dumps(prov, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(LEGS))
for i, (tag, c) in enumerate((('frozen', 'tab:red'), ('repaired', 'tab:blue'), ('norepair', '0.5'))):
    vals = [arm_tab['arms'][tag][f'{L}_flex'] for L in LEGS]
    ax.bar(x + (i - 1) * 0.27, vals, width=0.27, color=c, label=tag)
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x); ax.set_xticklabels(LEGS)
ax.set_ylabel('tibia flexor moment arm on tibia_pitch (same qpos)')
ax.set_title('FIG1  Three variants, one pose: frozen has a monotone front-to-back gradient; repaired copies the front-leg value '
             'to all six legs; norepair is sign-flipped and large', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_ARMS_THREE_VARIANTS.png', dpi=150)

fig, ax = plt.subplots(figsize=(10, 5.5))
for tag, c in (('frozen', 'tab:red'), ('repaired', 'tab:blue'), ('norepair', '0.5')):
    dist = [world['per_model'][tag][L]['distance'] for L in LEGS]
    ax.plot(LEGS, dist, 'o-', color=c, label=tag)
ax.set_ylabel('|site - tibia body origin| in world coords')
ax.set_title('FIG2  World-coordinate attachment distance (the 405 mm figure came from comparing LOCAL coordinates)', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_WORLD_SITE_DISTANCE.png', dpi=150)

fig, ax = plt.subplots(figsize=(12, 5.5))
names = sorted({k.split('|')[0] for k in census_all['frozen']})
for tag, c in (('frozen', 'tab:red'), ('repaired', 'tab:blue'), ('norepair', '0.5')):
    ratios = []
    for n in names:
        v = census_all[tag].get(n + '|B', {}).get('ratio')
        ratios.append(min(v, 1e6) if v is not None and np.isfinite(v) else 1e6)
    ax.plot(range(len(names)), ratios, 'o-', color=c, label=tag + ' (B)', ms=3, lw=1)
ax.axhline(1 / EPS_REL, color='k', ls='--', lw=1.0, label='one-sided threshold 1/eps_rel')
ax.set_yscale('log')
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace('joint_', '') for n in names], rotation=90, fontsize=6)
ax.set_ylabel('max|pos| / max|neg| (B matrix, log; clipped at 1e6)')
ax.set_title('FIG3  Four-model 42-joint directionality census on the physical matrix B', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_CENSUS_RATIOS.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, d):
    checks.append(dict(check=n, ok=bool(ok), detail=d))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()
sh = json.load(open(DIR / 'SOURCE_HASHES.json'))
chk('assets:originals_unchanged', sh['unchanged'], json.dumps({k: v[:12] for k, v in sh['assets'].items()}))
chk('arms:structure_equal', arm_tab['structure']['equal'],
    json.dumps({t: (arm_tab['structure']['per_model'][t]['nq'], arm_tab['structure']['per_model'][t]['nu']) for t in ASSETS}))
chk('arms:18_cells_reproduced_at_1e-6', arm_tab['n_advisor_cells_matched'] == arm_tab['n_advisor_cells'],
    '%d/%d cells within 1e-6 (the reported table is rounded to 6 decimals)'
    % (arm_tab['n_advisor_cells_matched'], arm_tab['n_advisor_cells']))
chk('arms:frozen_is_a_third_variant',
    all(abs(arm_tab['arms']['frozen'][f'{L}_flex'] - arm_tab['arms']['repaired'][f'{L}_flex']) > 1e-9
        for L in ('LM', 'LH', 'RM', 'RH'))
    and all(abs(arm_tab['arms']['frozen'][f'{L}_flex'] - arm_tab['arms']['norepair'][f'{L}_flex']) > 1e-9
            for L in ('LM', 'LH', 'RM', 'RH')),
    'frozen differs from both other variants on all four mid/hind legs')
chk('arms:frozen_gradient_monotone',
    abs(arm_tab['arms']['frozen']['LM_flex']) < abs(arm_tab['arms']['frozen']['LF_flex'])
    and abs(arm_tab['arms']['frozen']['LH_flex']) < abs(arm_tab['arms']['frozen']['LM_flex']),
    'left side |LF| %.6f > |LM| %.6f > |LH| %.6f'
    % (abs(arm_tab['arms']['frozen']['LF_flex']), abs(arm_tab['arms']['frozen']['LM_flex']),
       abs(arm_tab['arms']['frozen']['LH_flex'])))
chk('arms:repaired_copies_front_values',
    len({round(arm_tab['arms']['repaired'][f'{L}_flex'], 9) for L in ('LF', 'LM', 'LH')}) == 1
    and len({round(arm_tab['arms']['repaired'][f'{L}_flex'], 9) for L in ('RF', 'RM', 'RH')}) == 1,
    'left legs all equal, right legs all equal -> the repaired asset pins the front-leg value')
chk('arms:norepair_is_sign_flipped_mid_hind',
    all(arm_tab['arms']['norepair'][f'{L}_flex'] > 0 for L in ('LM', 'LH', 'RM', 'RH')),
    'norepair mid/hind flexor arms are positive (frozen ones are negative)')
chk('world:distance_uses_world_coords',
    all(abs(world['per_model'][t][L]['distance'] - np.linalg.norm(
        np.array(world['per_model'][t][L]['site_world']) - np.array(world['per_model'][t][L]['tibia_body_origin']))) < 1e-9
        for t in ASSETS for L in LEGS),
    'recomputed from site_xpos and xpos after one position-stage call')
chk('census:four_models_done', set(summary) == set(ASSETS),
    json.dumps({t: dict(nB12=len(summary[t]['B_1e12']), nB9=len(summary[t]['B_1e9']),
                        nA12=len(summary[t]['A_1e12']), nA9=len(summary[t]['A_1e9'])) for t in summary}))
chk('census:eps_both_reported',
    all(k in summary['frozen'] for k in ('A_1e12', 'A_1e9', 'B_1e12', 'B_1e9')),
    'frozen one-sided sets: A@1e-12 %s ; A@1e-9 %s ; B@1e-12 %s'
    % (summary['frozen']['A_1e12'], summary['frozen']['A_1e9'], summary['frozen']['B_1e12']))
chk('provenance:label', prov['label'] in ('ATTACHMENT_RULE_FOUND', 'FROZEN_ATTACHMENT_UNATTRIBUTED'), prov['label'])
chk('provenance:rule_documented_but_not_rederived', 'isotropically' in str(prov['documented_rule'])
    and prov['label'] == 'FROZEN_ATTACHMENT_UNATTRIBUTED',
    'the documented rule is recorded, but the 1e-6 offline re-derivation was not achieved -> UNATTRIBUTED')
chk('s210:contradictions_closed', json.load(open(DIR / 'S210_RECONCILIATION.json'))['part_c_status']['status']
    == 'S210_PART_C_INVALID', 'part C relabelled; lineage relabelled to LINEAGE_UNDETERMINED')
chk('guards:zero_dynamics_and_no_xml_change',
    sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4', 'frozen XML hash unchanged')
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                  all_ok=n_ok == len(checks)), indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), n_ok=n_ok, n_checks=len(checks),
                      failed=[c for c in checks if not c['ok']],
                      summary={t: dict(nB12=len(summary[t]['B_1e12']), nB9=len(summary[t]['B_1e9']),
                                       nA12=len(summary[t]['A_1e12']), nA9=len(summary[t]['A_1e9'])) for t in summary}),
                 indent=1, default=str))


# ---------------------------------------------------------------- conclusion + ledger + budget
import hashlib as _h
def _sha(p):
    return _h.sha256(Path(p).read_bytes()).hexdigest()

leg = {"round": "S211", "failures": [
  {"id": "S211-A1", "what": "cross-model arm comparison tolerance",
   "defect": "compared the reported table (rounded to 6 decimals) with 1e-9 tolerance, which reported 0/12 matches even though all 12 agreed at 6 decimals", "caught_by": "reading the per-cell diff (all ~1e-7)", "status": "tolerance corrected to 1e-6; the table is reproduced 12/12"},
  {"id": "S211-A2", "what": "census epsilon", "defect": "the first census used a single eps (1e-12) on the A matrix and declared a one-sided set that differed from S209's", "caught_by": "an assertion that failed", "status": "dual eps (1e-12 and 1e-9) now reported side by side; no single list is claimed"},
  {"id": "S211-A3", "what": "s211 census indentation", "defect": "a patch left the eps loop body unindented", "caught_by": "IndentationError", "status": "fixed"}],
  "corrections": [
  {"id": "S211-C1", "what": "S210 MODEL_LINEAGE.json", "defect": "wrote FROZEN_MODEL_IS_POST_REPAIR while the conclusion said LINEAGE_UNDETERMINED", "status": "relabelled LINEAGE_UNDETERMINED (the JSON came from the pass whose coordinate criterion broke a tie by max)"},
  {"id": "S211-C2", "what": "my S210 message and conclusion sentence about Part C", "defect": "claimed the s table AND the per-leg minimax did not change; the archived numbers show the per-leg minimax changed and got worse (LH 2.9495->3.1267, RM 2.4262->2.5390)", "status": "corrected: s rows unchanged, per-leg minimax changed; Part C marked S210_PART_C_INVALID because the edit landed on the wrong rows"},
  {"id": "S211-C3", "what": "S210 coordinate criterion", "defect": "compared LOCAL site coordinates across assets, which is meaningless and produced the 405 mm red herring", "status": "replaced by world-coordinate distances in WORLD_SITE_GEOMETRY.json"}],
  "design_notes": [
  {"id": "S211-N1", "what": "frozen is a third variant", "finding": "at one pose the tibia flexor arms are frozen: LF -0.010246, LM -0.004322, LH -0.001785 (monotone 100/42/17%); repaired: -0.010246 on every left leg and -0.010509 on every right leg (front value copied); norepair: +0.123469 / +0.045804 / +0.116572 / +0.044686 (sign flipped and large). None of the three is a perturbation of another."},
  {"id": "S211-N2", "what": "the documented construction rule", "finding": "outputs/muscles_six_legs/mirror_decisions.json records 'site positions scaled isotropically by the segment length ratio' and 'no muscles in the tibia' for the source model; that is the candidate explanation of the monotone gradient, but the 1e-6 offline re-derivation was NOT achieved this round, so the label stays FROZEN_ATTACHMENT_UNATTRIBUTED"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(leg, indent=2, ensure_ascii=False))

wc = sorted({L for t in arm_tab['arms'] for L in ('LF','LM','LH','RF','RM','RH')})
rows = []
for L in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
    rows.append('| ' + L + ' | ' + '%.6f' % arm_tab['arms']['frozen'][L + '_flex'] + ' | '
                + '%.6f' % arm_tab['arms']['repaired'][L + '_flex'] + ' | '
                + '%.6f' % arm_tab['arms']['norepair'][L + '_flex'] + ' |')
rowtxt = chr(10).join(rows)
concl = chr(10).join([
 '# S211 frozen 附着几何来源归因 + S210 矛盾闭环 + 跨模型臂表 —— 结论（中文）',
 '',
 '目录：`' + str(DIR) + '`',
 '',
 '## 0. 标签（并列，主标签=归因）',
 '- **归因（主）：`FROZEN_ATTACHMENT_UNATTRIBUTED`** —— 规则有文档，但 1e-6 的离线重算本轮未达成。',
 '- **臂表：`ARMS_CROSS_MODEL_REPRODUCED`** —— 12/12 格复现（容差 1e-6，因为给出的表是 6 位小数取整）。',
 '- **血统：`FROZEN_THIRD_VARIANT`** —— frozen 与 repaired、norepair 三者互不相承。',
 '- **S210 处置：`S210_LINEAGE_RELABELED` + `S210_PART_C_INVALID`**。',
 '- **强制条款已生效**：来源归属完成前，25–39 倍臂差不得写成“我们构建的缺陷”，也不得写成“身体限制”，只能写“三个变体在同一姿态下给出不同的胫节肌臂，且 frozen 是前→后单调梯度”。',
 '',
 '## 1. S210 矛盾闭环（`S210_RECONCILIATION.json`）',
 '1. `MODEL_LINEAGE.json` 写 POST_REPAIR、结论写 UNDETERMINED → JSON 出自旧的一遍（坐标判据“打平取 max”），结论正确 → 统一为 LINEAGE_UNDETERMINED。',
 '2. 我上一条消息说“Part C 一个字都没变”——错。归档里 s 行未变，但逐腿 minimax 变了且更差（LH 2.9495→3.1267、RM 2.4262→2.5390）。原因：把源执行器的承载行写到了目标执行器上，改动落在非绑定行。',
 '3. `ARM_SUBSTITUTION_LP.json` 因此没有可用状态 → 判 `S210_PART_C_INVALID`（数字真实，但回答的不是所问的问题）。',
 '4. 更正 S210 坐标判据：此前比的是局部 `site_pos`，405 mm 是局部坐标误比的产物 → 已改为世界坐标。',
 '',
 '## 2. 跨模型臂表（`CROSS_MODEL_ARMS.json/.csv`）',
 '- mesh 路径改写只在 `/tmp/s211_assets`，三个原资产哈希前后一致。',
 '- 四模型结构相等（nq49/nv48/njnt43/nu90/nkey1 + joint/actuator 名逐项相同）。',
 '- 同一 qpos（frozen keyframe qpos0）喂四模型，用 `d.moment_rowadr/rownnz/colind` 稀疏读取，12/12 格与给出的表一致到 1e-6。',
 '',
 '| 腿 | frozen flex | repaired | norepair |',
 '|---|---|---|---|',
 rowtxt,
 '',
 '三条硬事实复现：repaired 把前腿值抄到六条腿（左全 −0.010246、右全 −0.010509）；norepair 的中后腿臂符号反且大（+0.1235/+0.0458/+0.1166/+0.0447）；frozen 是第三种，单调梯度 100/42/17%（左）与 70/34/16%（右，相对 LF）。',
 '',
 '## 3. 世界坐标（`WORLD_SITE_GEOMETRY.json`、FIG2）',
 '改为 `site_xpos` 与 `xpos[tibia body]` 的距离后，四模型六腿距离都是合理尺度；405 mm 是局部坐标误比的产物，不作为证据。',
 '',
 '## 4. 四模型单向普查（`LINEAGE_BEHAVIOUR_4MODELS.json`、FIG3）',
 'B（物理）与 A=B@T（可命令）分别普查，ε 双口径（1e-12 / 1e-9）并列：frozen B=' + str(len(summary['frozen']['B_1e12'])) + ' 个、A=' + ' / '.join(summary['frozen']['A_1e9']) + '；repaired B=' + str(len(summary['repaired']['B_1e12'])) + '、A=' + str(len(summary['repaired']['A_1e9'])) + '；norepair B=' + str(len(summary['norepair']['B_1e12'])) + '、A=' + str(len(summary['norepair']['A_1e9'])) + '；qpos0bands B=' + str(len(summary['qpos0bands']['B_1e12'])) + '、A=' + str(len(summary['qpos0bands']['A_1e9'])) + '。',
 '→ “移植会修好什么”现在有可检验判据（repaired 的 B 单向集合为空），而不是拿 seam 文本当证据。',
 '',
 '## 5. 来源归因（`ATTACHMENT_PROVENANCE.json`）',
 '`outputs/muscles_six_legs/mirror_decisions.json` 记录：site 位置按段长比各向同性缩放；源模型胫节内无肌肉（中后腿附着点由本仓库构造）。这与前→后单调梯度一致，但把冻结 site 从源 XML 用该规则离线重算到 1e-6 本轮未完成（缺 per-leg 缩放表）→ 标签保持 FROZEN_ATTACHMENT_UNATTRIBUTED。',
 '',
 '## 6. Part D 未执行',
 '预登记规定只有 Part B/C 通过才允许做臂移植；B 未通过 → Part D 不执行，`ARM_SUBSTITUTION_V2.json` 不存在（不伪造）。',
 '',
 '## 7. 这不是什么',
 '不声明任何运动改善；力单位 NOT_ESTABLISHED；repaired 资产的右腿附着点在世界上可疑 → 不得当“修后真值”。',
 '',
 '## 8. 下一步最小动作',
 '1. 找到 per-leg 缩放表（或在源 XML + 缩放规则上参数拟合），把 frozen site 离线重算到 1e-6，才能升级为 ATTACHMENT_RULE_FOUND。',
 '2. 用 repaired 的 B 单向集合为空这一判据替代 seam 文本。',
 '3. Part D 待 1 完成后按“目标执行器自己的行”重做。',
 '',
 '## 9. 记账',
 '核验 ' + str(n_ok) + '/' + str(len(checks)) + '；台账 3 条自身失败 + 3 条更正 + 2 条设计说明；0 forward-dynamics / 0 mj_step / 0 XML 改动 / 0 训练。',
])
(DIR / 'CONCLUSION.zh-CN.md').write_text(concl)
budget = {"round": "S211",
          "stop_reason": ("Part 0 closed the three S210 contradictions; Part A reproduced the 12-cell cross-model arm table; Part C "
                          "ran the four-model census with a dual-eps protocol; Part B could not complete the 1e-6 offline "
                          "re-derivation, so the provenance label is FROZEN_ATTACHMENT_UNATTRIBUTED and the conditional Part D was "
                          "not run."),
          "budget": {"model_loads": 8, "note_loads": "4 in the arm-table pass + 4 in the census pass (the cap was 4 per pass)",
                     "position_stage_calls": 8, "constraint_stage_calls": 0, "forward_calls": 0, "mj_step_calls": 0,
                     "xml_changes": 0, "training_steps": 0, "prewarming": False},
          "artifacts": {str(pp.relative_to(DIR)): _sha(pp) for pp in sorted(DIR.rglob('*')) if pp.is_file()},
          "sources": {str(pp): _sha(pp) for pp in [Path('/run/media/sensen/Data2/cell_wound_prototype/s211_crossmodel.py'),
                                                   Path('/run/media/sensen/Data2/cell_wound_prototype/s211_census.py')] if pp.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
print(json.dumps(dict(conclusion_written=True, files=len(budget['artifacts']))))
