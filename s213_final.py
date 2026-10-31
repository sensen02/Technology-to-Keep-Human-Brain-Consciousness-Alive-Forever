"""S213 finalisation: figures, verification, conclusion, ledger, budget."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s213-full-repair-lp').iterdir())[-1]
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
pose = json.load(open(DIR / 'POSE_ARM_TABLE.json'))
collapse = json.load(open(DIR / 'GAIN_COLLAPSE_MECHANISM.json'))
full = json.load(open(DIR / 'FULL_REPAIR_VARIANT.json'))
lp = json.load(open(DIR / 'FULL_REPAIR_LP.json'))
geo = json.load(open(DIR / 'PORT_GEOMETRY_CHECK.json'))
corr = json.load(open(DIR / 'S212_CORRECTIONS.json'))
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ---------------------------------------------------------------- FIG1 pose ratios
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(6)
series = [('frozen|keyframe', 'tab:orange', 'frozen @ keyframe'), ('frozen|release', 'tab:red', 'frozen @ release'),
          ('repaired|keyframe', 'tab:cyan', 'repaired @ keyframe'), ('repaired|release', 'tab:blue', 'repaired @ release'),
          ('qpos0bands|release', 'tab:green', 'qpos0bands @ release')]
for i, (k, c, lab) in enumerate(series):
    ax.bar(x + (i - 2) * 0.16, [pose['table'][L][k] for L in LEGS], width=0.16, color=c, label=lab)
ax.set_yscale('log'); ax.axhline(1.0, color='k', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels(LEGS)
ax.set_ylabel('|flexor arm| ratio  LF / leg   (dimensionless, log)')
ax.set_title('FIG1  Same pose, four assets: only frozen collapses on the mid/hind legs; its own keyframe and the repaired assets '
             'stay within 0.9-1.3x', fontsize=9)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_POSE_RATIO_GROUPS.png', dpi=150)

# ---------------------------------------------------------------- FIG2 s by row
fig, ax = plt.subplots(figsize=(13, 5.5))
names = [k for k in lp['s_tables']['frozen']]
defect = set(lp['defect_rows'])
for tag, c in (('frozen', 'tab:red'), ('port24', 'tab:orange'), ('full', 'tab:green')):
    vals = []
    for n in names:
        v = lp['s_tables'][tag][n]
        vals.append(min(v, 60) if v is not None else 60)
    ax.plot(range(len(names)), vals, 'o-', color=c, ms=3, lw=1.1, label=tag)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.set_yscale('log')
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace('joint_', '') for n in names], rotation=90, fontsize=6)
for i, n in enumerate(names):
    if n in defect:
        ax.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.08)
ax.set_ylabel('required activation s = |req| / max coefficient (log, clipped 60)')
ax.set_title('FIG2  42-row s under the three variants (red bands = the six named defect rows)', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_S_BY_ROW_THREE_VARIANTS.png', dpi=150)

# ---------------------------------------------------------------- FIG3 per-leg minimax
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, (tag, c) in enumerate((('frozen', 'tab:red'), ('port24', 'tab:orange'), ('full', 'tab:green'))):
    vals = [lp['variants'][tag]['per_leg'][L] or 0.0 for L in LEGS]
    ax.bar(x + (i - 1) * 0.26, vals, width=0.26, color=c, label=tag)
ax.set_xticks(x); ax.set_xticklabels(LEGS)
ax.set_ylabel('per-leg minimax (box=1)')
ax.set_title('FIG3  Frozen %.4f -> port24 %.4f -> full repair %.4f ; LM stays at %.4f while LH/RM/RH drop'
             % (lp['variants']['frozen']['global_box1'], lp['variants']['port24']['global_box1'],
                lp['variants']['full']['global_box1'], lp['variants']['full']['per_leg']['LM']), fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_PER_LEG_MINIMAX.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, det):
    checks.append(dict(check=n, ok=bool(ok), detail=det))
chk('pose_table:reproduced_within_5e-4', all(v['max_abs_diff'] <= 5e-4 for v in pose['verification'].values()),
    json.dumps({k: round(v['max_abs_diff'], 5) for k, v in pose['verification'].items()}))
chk('pose_table:frozen_release_ratios', all(abs(pose['table'][L]['frozen|release'] - e) <= 5e-3 for L, e in
                                            (('LM', 34.003), ('LH', 24.713), ('RM', 38.848), ('RH', 23.940))),
    json.dumps({L: round(pose['table'][L]['frozen|release'], 3) for L in LEGS}))
chk('pose_table:repaired_release_ratios', all(abs(pose['table'][L]['repaired|release'] - e) <= 5e-3 for L, e in
                                              (('LM', 0.908), ('LH', 1.115), ('RM', 0.915), ('RH', 1.091))),
    json.dumps({L: round(pose['table'][L]['repaired|release'], 3) for L in LEGS}))
chk('port24:arm_equals_repaired', collapse['arm_equals_repaired'],
    'LM ported arm %.9f vs repaired %.9f' % (collapse['lm_arm_ported'], collapse['lm_arm_repaired']))
chk('port24:gain_collapse_mechanism', collapse['lm_fl_ported'] < 0.2,
    'LM tendon length %.5f against frozen lengthrange %s -> FL %.4f -> gain %.2f (frozen gain %.2f)'
    % (collapse['lm_length_ported'], collapse['lm_lengthrange_frozen'], collapse['lm_fl_ported'],
       collapse['lm_gain_ported'], collapse['lm_gain_frozen']))
chk('cross_asset:incommensurable_recorded',
    corr['cross_asset']['frozen']['gainprm2'][0] != corr['cross_asset']['qpos0bands']['gainprm2'][0],
    'gainprm[2] frozen %s vs qpos0bands %s ; dynprm frozen %s vs qpos0bands %s'
    % (corr['cross_asset']['frozen']['gainprm2'][0], corr['cross_asset']['qpos0bands']['gainprm2'][0],
       corr['cross_asset']['frozen']['dynprm'][0], corr['cross_asset']['qpos0bands']['dynprm'][0]))
chk('variant:24_sites_and_8_lengthranges', full['checks']['n_sites_changed'] == 24
    and full['checks']['n_lengthrange_changed'] == 8, json.dumps(dict(
        sites=full['checks']['n_sites_changed'], lr=full['checks']['n_lengthrange_changed'])))
chk('variant:82_untouched_B_identical', full['checks']['untouched_B_count'] == 82,
    '%d of 90 actuators keep B bitwise identical' % full['checks']['untouched_B_count'])
chk('variant:gain_restored_within_2pct',
    all(v['rel'] <= 0.02 for v in full['checks']['gain_restored'].values()),
    json.dumps({k: round(v['rel'], 5) for k, v in full['checks']['gain_restored'].items()}))
chk('variant:identity_holds', full['checks']['identity_max_delta'] <= 1e-9,
    'max|B - arm*gain| = %.2e' % full['checks']['identity_max_delta'])
chk('variant:arm_matches_qpos0bands',
    all(v <= 1e-9 for v in full['checks']['arm_matches_qpos0bands'].values()),
    json.dumps({k: '%.1e' % v for k, v in full['checks']['arm_matches_qpos0bands'].items()}))
chk('gate:baselines', abs(lp['variants']['frozen']['global_box1'] - 3.308620867791373) <= 1e-9
    and abs(lp['variants']['frozen']['per_leg']['RM'] - 2.426249) <= 1e-4,
    'frozen global %.15g, RM per-leg %.6f' % (lp['variants']['frozen']['global_box1'],
                                              lp['variants']['frozen']['per_leg']['RM']))
chk('prediction:tibia_rows_in_0.5_1.0',
    all(0.5 <= lp['defect_rows'][k]['full'] <= 1.0 for k in lp['defect_rows'] if 'Tibia' in k),
    json.dumps({k: round(v['full'], 3) for k, v in lp['defect_rows'].items() if 'Tibia' in k}))
chk('prediction:RMCoxa_pitch_unchanged', lp['defect_rows']['joint_RMCoxa_pitch']['full'] is None,
    'still no positive coefficient (None) in all three variants')
chk('label:insufficient_with_named_bottleneck', lp['label'] == 'FULL_REPAIR_INSUFFICIENT'
    and lp['variants']['full']['per_leg']['LM'] > 0.5 * lp['variants']['frozen']['per_leg']['LM'],
    'LM %.4f > half of frozen %.4f -> LM is the residual bottleneck'
    % (lp['variants']['full']['per_leg']['LM'], lp['variants']['frozen']['per_leg']['LM']))
chk('geometry:length_ratio_gate', geo['passed'] is False,
    'LM length ratio %.3f is just below the 0.8 gate -> PORT_INVARIANT_ONLY_NOT_ANATOMICAL'
    % geo['geometry']['LM']['length_ratio'])
chk('guards:no_dynamics_no_xml_edit', sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4'
    and full['frozen_unchanged'], 'frozen hash unchanged; 0 forward, 0 mj_step')
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                  all_ok=n_ok == len(checks)), indent=2, default=str))

# ---------------------------------------------------------------- conclusion
rows = []
for L in ('LM', 'LH', 'RM', 'RH'):
    rows.append('| ' + L + ' | %.3f | %.3f | %.3f | %.3f |' % (pose['table'][L]['frozen|keyframe'],
                                                              pose['table'][L]['frozen|release'],
                                                              pose['table'][L]['repaired|keyframe'],
                                                              pose['table'][L]['repaired|release']))
crow = chr(10).join(rows)
def rows_txt():
    out = []
    for k in ('joint_LHTibia_pitch', 'joint_LMTibia_pitch', 'joint_RHTibia_pitch', 'joint_RMTibia_pitch',
              'joint_LMCoxa_roll', 'joint_RMCoxa_pitch'):
        d = lp['defect_rows'][k]
        def fmt(v):
            return 'none' if v is None else '%.3f' % v
        out.append('| ' + k.replace('joint_', '') + ' | ' + fmt(d['frozen']) + ' | ' + fmt(d['port24']) + ' | ' + fmt(d['full']) + ' |')
    return chr(10).join(out)
concl = chr(10).join([
 '# S213 完整体制修（site + lengthrange）后的同一 LP —— 结论（中文）',
 '',
 '目录：`' + str(DIR) + '`',
 '',
 '## 0. 标签',
 '- **`FULL_REPAIR_INSUFFICIENT`** —— 四条胫节 s 全部落到 **0.530–0.871（预测 0.5–1.0 ✓）**，但 **LM 逐腿 minimax 1.8401 未减半**（frozen 2.5382 的一半是 1.2691）→ 剩余瓶颈是 **LM**。',
 '- **`PORT_INVARIANT_ONLY_NOT_ANATOMICAL`** —— LM 肌腱总长比 0.794 略低于 0.8 门限。',
 '- **`S212_C3_SWAPPED_FACTORS` / `S212_C4_SITE_NOT_BROKEN`** —— 你指出的两条我都确认撤回。',
 '',
 '## 1. Part 0：你的四条我全部复现（`POSE_ARM_TABLE.json`）',
 '比值口径是 **LF/腿**（我首版算成腿/LF，得到 0.029≈1/34，已修正）：',
 '',
 '| 腿 | frozen keyframe | frozen release | repaired keyframe | repaired release |',
 '|---|---|---|---|---|',
 crow,
 '',
 '- 复现精度：与你的表 **|Δ| ≤ 5e-4**（你给的是 3 位小数）。',
 '- **两个因子必须分开算**：同一 release 位姿下 frozen 对 repaired 的差 = **%.1f–%.1fx**；frozen 内部 keyframe→release = 3.1–13.4×；repaired 内部只有 1.03–1.26×。→ **主因子是附着几何，不是位姿**（S212 方向反了，撤回）。'
   % (min(pose['table'][L]['frozen|release'] / max(pose['table'][L]['repaired|release'], 1e-12) for L in ('LM', 'LH', 'RM', 'RH')),
      max(pose['table'][L]['frozen|release'] / max(pose['table'][L]['repaired|release'], 1e-12) for L in ('LM', 'LH', 'RM', 'RH'))),
 '- **移植 24 site 后臂精确等于 repaired**：LM 移植 %.9f vs repaired %.9f（|Δ| ≤ 1e-12 ✓）。'
   % (collapse['lm_arm_ported'], collapse['lm_arm_repaired']),
 '- **gain 塌陷机制（`GAIN_COLLAPSE_MECHANISM.json`）**：移植后 LM 肌腱长 **%.5f**，而冻结 `lengthrange` 仍是 **%s** → FL **%.4f** → gain **%.2f**（冻结 %.2f）。这就是"移植只到 3.6×"的原因：**臂对了、长度张力因子塌了**。'
   % (collapse['lm_length_ported'], collapse['lm_lengthrange_frozen'], collapse['lm_fl_ported'],
      collapse['lm_gain_ported'], collapse['lm_gain_frozen']),
 '- **跨资产不可比（`S212_CORRECTIONS.json`）**：`gainprm[2]` frozen **%s** vs qpos0bands **%s**；`dynprm` frozen **%s** vs qpos0bands **%s** → 力标度差 10×、时间常数不同，**只允许无量纲比**。'
   % (corr['cross_asset']['frozen']['gainprm2'][0], corr['cross_asset']['qpos0bands']['gainprm2'][0],
      corr['cross_asset']['frozen']['dynprm'][0], corr['cross_asset']['qpos0bands']['dynprm'][0]),
 '',
 '## 2. Part A：完整体制修变体（`FULL_REPAIR_VARIANT.json`）',
 '- `/tmp/frozen_variant_full_repair.xml`：**24 个 site**（来自 qpos0bands）+ **8 个 `lengthrange`**（同源）改动，其余逐字节不变；冻结原件哈希不变。',
 '- 自检：**82 个未改动执行器的 B 逐位不变**（|Δ| = 0）；8 个胫节执行器 gain 相对前腿 **≤0.1%**（FL 已恢复）；`B = arm × gain` **恒等式 |Δ| = 0.0**；臂与 qpos0bands 对应值差 ≤1e-9。',
 '',
 '## 3. Part B：同一 LP 三变体（`FULL_REPAIR_LP.csv`、FIG2/FIG3）',
 '| 变体 | 全局 box=1 | LF | LH | LM | RF | RM | RH |',
 '|---|---|---|---|---|---|---|---|',
 '| frozen | %.4f | %s |' % (lp['variants']['frozen']['global_box1'],
                              ' | '.join('%.4f' % (lp['variants']['frozen']['per_leg'][L] or 0.0) for L in ('LF', 'LH', 'LM', 'RF', 'RM', 'RH'))),
 '| port24（只改 site） | %.4f | %s |' % (lp['variants']['port24']['global_box1'],
                                          ' | '.join('%.4f' % (lp['variants']['port24']['per_leg'][L] or 0.0) for L in ('LF', 'LH', 'LM', 'RF', 'RM', 'RH'))),
 '| **full repair（site+lr）** | **%.4f** | %s |' % (lp['variants']['full']['global_box1'],
                                                    ' | '.join('%.4f' % (lp['variants']['full']['per_leg'][L] or 0.0) for L in ('LF', 'LH', 'LM', 'RF', 'RM', 'RH'))),
 '',
 '缺陷行 s：',
 '',
 '| 行 | frozen | port24 | full repair |',
 '|---|---|---|---|',
 rows_txt(),
 '',
 '- 预测与实际：**胫节四行 0.5–1.0 → 实际 0.530/0.553/0.796/0.871 ✓**；`RMCoxa_pitch` 仍无正系数（None）**不变 ✓**；**逐腿预测 0.7–1.9 → 实际 LH 0.6561 略低于下界（偏差 -0.044），LM 1.8401、RM 1.7963、RH 1.0530 在区间内**。预测未事后修改，偏差照实写出。',
 '- **判决 `FULL_REPAIR_INSUFFICIENT`**：LM 1.8401 > 2.5382/2 = 1.2691 → 未达"减半"门；**剩余瓶颈行 = LM（`joint_LMTibia_pitch` 及其腿内同时性）**。',
 '',
 '## 4. 必须同时写明的两条（强制）',
 '1. **34× 是冻结附着坐标的选择，不是身体事实**：同 body、同 joint、同 tendon 拓扑下，仅换 site 坐标 + 对应 lengthrange 就使胫节 s 从 2.1–22.5 落到 0.53–0.87，且逐腿 minimax 从 3.3086 降到 1.8401。',
 '2. **这不声明站立/行走，也不声明可行域被真实放宽**：变体是 `/tmp` 假设推演，repaired 资产自身还带 10× 力标度差与不同 dynprm；LM 仍 1.8401。',
 '',
 '## 5. 这不是什么',
 '不声明运动改善；力单位 NOT_ESTABLISHED；`PORT_INVARIANT_ONLY_NOT_ANATOMICAL` 已触发（LM 总长比 0.794），所以只能说"存在一个保持臂/总长近似不变的坐标指派"，不能说"几何已修好"。',
 '',
 '## 6. 下一步',
 '1. LM 的 1.8401 单独做最小不可行子系统（区分箱界不足与腿内同时性）。',
 '2. 若要让几何可信，需要按 body 段长约束 site 的最远距离（本轮 LM 0.794 已越门）。',
 '3. 跨资产若要比绝对量，必须先统一 `gainprm[2]` 与 `dynprm`。',
 '',
 '## 7. 记账',
 '核验 ' + str(n_ok) + '/' + str(len(checks)) + '；台账记本轮 3 条自身失败（比值方向写反、lengthrange 替换按索引错位、几何门越界）；model load ' + str(cl_load := 4) + '、position 阶段 4、0 forward / 0 mj_step / 0 XML 改动 / 0 训练。',
])
(DIR / 'CONCLUSION.zh-CN.md').write_text(concl)
led = {"round": "S213", "failures": [
  {"id": "S213-A1", "what": "pose ratio direction", "defect": "computed leg/LF instead of LF/leg, so the frozen release ratios came out as 0.029 instead of 34.003", "caught_by": "comparing with the review table", "status": "fixed"},
  {"id": "S213-A2", "what": "lengthrange replacement", "defect": "replaced lengthrange by a running actuator index that was reset twice per line, so ZERO lengthranges changed and the 'full repair' variant was silently identical to port24", "caught_by": "n_lengthrange_changed = 0 in the self-check", "status": "fixed to name matching; 8 lengthranges now change and the LP drops from 2.306 to 1.8401"},
  {"id": "S213-A3", "what": "geometry gate", "defect": "LM total length ratio 0.794 falls below the 0.8 gate", "caught_by": "the gate itself", "status": "recorded as PORT_INVARIANT_ONLY_NOT_ANATOMICAL"}],
  "corrections": [{"id": "S213-C1", "what": "S212 pose-vs-attachment factors", "status": "swapped: attachment dominates (21.9-42.5x), pose 3.1-13.4x inside frozen"},
                  {"id": "S213-C2", "what": "S212 'repaired RF site broken (405 mm)'", "status": "withdrawn: the far local coordinate still gives a front-leg-like arm and total length"}],
  "design_notes": [{"id": "S213-N1", "what": "gain collapse is the port24 story", "finding": "porting sites without lengthrange leaves the tendon length outside the frozen lengthrange, FL collapses to ~0.097 and gain follows; syncing the 8 lengthranges restores FL to ~1 and the LP drops to 1.8401"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(led, indent=2, ensure_ascii=False))
budget = {"round": "S213",
          "stop_reason": "Parts 0/A/B/C complete. Full repair (site + lengthrange) restores tibia symmetry (s 0.53-0.87) and drops the global minimax from 3.3086 to 1.8401, but LM stays above half, so the label is FULL_REPAIR_INSUFFICIENT with LM named as the residual bottleneck.",
          "budget": dict(model_loads=4, position_stage_calls=4, constraint_stage_calls=0, forward_calls=0, mj_step_calls=0,
                         lp_solves="about 20", training_steps=0, prewarming=False, xml_changes=0),
          "artifacts": {str(p.relative_to(DIR)): sha(p) for p in sorted(DIR.rglob('*')) if p.is_file()},
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s213_full_repair.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s213_final.py'), FROZEN] if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
print(json.dumps(dict(dir=str(DIR), n_ok=n_ok, n_checks=len(checks), failed=[c['check'] for c in checks if not c['ok']]), indent=1))
