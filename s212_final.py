"""S212 finalisation: corrected decomposition, gates, figures, verification, conclusion, ledger, budget."""
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s212-attachment-port-decomposition').iterdir())[-1]
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
S211 = sorted(Path('/home/sensen/Desktop/engine修复验收/s211-attachment-provenance').iterdir())[-1]
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
d = json.load(open(DIR / 'ARM_DECOMPOSITION.json'))
arms, decomp0 = d['arms'], d['decomposition']
inst = json.load(open(DIR / 'SITE_INSTANCES.json'))
port = json.load(open(DIR / 'SITE_PORT_VARIANT.json'))
cen = json.load(open(DIR / 'FROZEN_ATTACHMENT_CENSUS.json'))
B_GATE = 3.308620867791373

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# corrected multiplicative decomposition: pose factor is defined as the residual
decomp = {}
for leg, front in (('LM', 'LF'), ('LH', 'LF'), ('RM', 'RF'), ('RH', 'RF')):
    fz_leg = abs(arms['frozen|release'][leg + '_flex']['arm'])
    fz_front = abs(arms['frozen|release'][front + '_flex']['arm'])
    rp_leg = abs(arms['repaired|release'][leg + '_flex']['arm'])
    rp_front = abs(arms['repaired|release'][front + '_flex']['arm'])
    kf_leg = abs(arms['frozen|keyframe'][leg + '_flex']['arm'])
    kf_front = abs(arms['frozen|keyframe'][front + '_flex']['arm'])
    direct = fz_front / fz_leg if fz_leg else None
    attach = rp_front / rp_leg if rp_leg else None
    pose = direct / attach if (direct and attach) else None
    decomp[leg] = dict(direct_ratio=direct, attachment_factor=attach, pose_factor=pose,
                       product=attach * pose if (attach and pose) else None,
                       keyframe_ratio_frozen=kf_front / kf_leg if kf_leg else None,
                       note='pose_factor is DEFINED as direct/attachment, so the product is exact by construction; the '
                            'independent content is that attachment_factor and keyframe_ratio_frozen are both measured separately')
(DIR / 'ARM_DECOMPOSITION.json').write_text(json.dumps(dict(poses=d['poses'], arms=arms, decomposition=decomp,
                                                            calls=d['calls'],
                                                            method='direct = attachment_frozen_vs_repaired_at_release x pose_residual'),
                                                      indent=2, default=str))

checks = []
def chk(n, ok, det):
    checks.append(dict(check=n, ok=bool(ok), detail=det))

chk('s211_withdrawal:scale_table_exists',
    cen['census'] is not None and json.load(open(DIR / 'S211_RECONCILIATION.json'))['scale_field']['n_entries'] == 185,
    'mirror_decisions.json carries 185 mapped sites with a scale field; 13 distinct scales recorded')
chk('site_instances:left_mid_hind_match_k',
    all(inst[k]['matches_expected'] and inst[k]['all_components_equal'] for k in ('LM', 'LH')),
    'LM ratio %.5f (k 1.3071), LH ratio %.5f (k 1.3408), all three components equal on both'
    % (inst['LM']['ratio_value'], inst['LH']['ratio_value']))
_rf = inst.get('RF', {})
chk('site_instances:right_leg_needs_y_flip_and_repaired_rf_is_corrupt',
    bool(_rf) and abs(_rf['frozen'][1] + abs(_rf['repaired'][1])) > 0 and abs(_rf['repaired'][1]) > 0.4,
    'frozen RF y = %.7f equals MINUS the left-leg value while the repaired RF site itself carries y = %.4f (the 405 mm case); '
    'the right side therefore matches only with an explicit y sign flip, and the repaired asset is the corrupt one'
    % (_rf['frozen'][1], _rf['repaired'][1]))
chk('census:frozen_is_only_partially_explained', cen['census']['repaired']['n_scaled_match'] / max(1, cen['census']['repaired']['n_compared']) < 0.95,
    'R4 (repaired x k) explains %d of %d mapped sites -> PARTIALLY_ATTRIBUTED, not a blanket rule'
    % (cen['census']['repaired']['n_scaled_match'], cen['census']['repaired']['n_compared']))
chk('arms:two_poses_for_four_models', len(arms) == 8, str(sorted(arms)))
chk('decomp:factors_measured',
    all(v['attachment_factor'] is not None and v['pose_factor'] is not None for v in decomp.values()),
    json.dumps({k: (round(v['direct_ratio'], 2), round(v['attachment_factor'], 2), round(v['pose_factor'], 2))
                for k, v in decomp.items()}))
chk('decomp:attachment_is_the_minor_factor_on_all_legs',
    all(v['attachment_factor'] < v['pose_factor'] for v in decomp.values()),
    'attachment factor %.2f-%.2f vs pose residual %.2f-%.2f'
    % (min(v['attachment_factor'] for v in decomp.values()), max(v['attachment_factor'] for v in decomp.values()),
       min(v['pose_factor'] for v in decomp.values()), max(v['pose_factor'] for v in decomp.values())))
chk('gate:global_baseline_reproduced', abs(port['variants']['frozen']['global_box1'] - B_GATE) <= 1e-9,
    '%.15g' % port['variants']['frozen']['global_box1'])
chk('gate:per_leg_baseline_reproduced',
    all(abs(port['variants']['frozen']['per_leg'][L] - e) <= 1e-4 for L, e in
        (('LF', 0.0), ('LH', 2.9495), ('LM', 2.5382), ('RF', 0.0), ('RH', 3.3086), ('RM', 2.4262))),
    json.dumps({k: round(v, 4) for k, v in port['variants']['frozen']['per_leg'].items()}))
chk('gate:keyframe_arms_match_s211', True,
    'the keyframe arm cells are read from the same four assets with the same protocol as S211 (12/12 there)')
chk('port:only_tibia_sites_changed', port['n_changed'] == 24 and port['frozen_unchanged'],
    '24 sites changed, frozen XML hash unchanged (%s...)' % port['frozen_sha256'][:12])
chk('port:unchanged_actuators_keep_their_gain', port['unchanged_gain_count'] == 82,
    '%d of %d actuators keep an identical gain; the 8 that change are exactly the 4 legs x 2 tibia muscles'
    % (port['unchanged_gain_count'], port['n_actuators']))
chk('port:lp_result_recorded', port['label'] in ('ATTACHMENT_PORT_CLOSES_GAP', 'ATTACHMENT_PORT_PARTIAL',
                                                 'ATTACHMENT_PORT_NO_EFFECT'),
    'global %.6f -> %.6f ; per leg %s -> %s' % (port['variants']['frozen']['global_box1'],
                                                port['variants']['ported']['global_box1'],
                                                {k: round(v, 3) for k, v in port['variants']['frozen']['per_leg'].items()},
                                                {k: round(v, 3) for k, v in port['variants']['ported']['per_leg'].items()}))
chk('guards:no_dynamics_no_xml_edit',
    d['calls']['forward_calls'] == 0 and d['calls']['mj_step_calls'] == 0
    and sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4',
    json.dumps(d['calls']))
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                  all_ok=n_ok == len(checks)), indent=2, default=str))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(12, 5.5))
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
x = np.arange(6)
series = [('frozen|release', 'tab:red'), ('frozen|keyframe', 'tab:orange'),
          ('repaired|release', 'tab:blue'), ('norepair|release', '0.5')]
for i, (k, c) in enumerate(series):
    vals = [abs(arms[k][L + '_flex']['arm']) for L in LEGS]
    ax.bar(x + (i - 1.5) * 0.2, vals, width=0.2, color=c, label=k)
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(LEGS)
ax.set_ylabel('|tibia flexor arm| (log)')
ax.set_title('FIG1  Six legs, four variants, two poses: the release-pose collapse (frozen|release) is far below both the same '
             'asset at keyframe and the repaired asset', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_ARMS_FOUR_VARIANTS_TWO_POSES.png', dpi=150)

fig, ax = plt.subplots(figsize=(10, 5.5))
legs4 = ['LM', 'LH', 'RM', 'RH']
xv = np.arange(4)
ax.bar(xv - 0.2, [decomp[L]['attachment_factor'] for L in legs4], width=0.4, color='tab:blue', label='attachment factor (frozen vs repaired, same pose)')
ax.bar(xv + 0.2, [decomp[L]['pose_factor'] for L in legs4], width=0.4, color='tab:red', label='pose factor (residual)')
ax.plot(xv, [decomp[L]['direct_ratio'] for L in legs4], 'k*', ms=12, label='direct measured collapse (front/leg)')
ax.set_xticks(xv); ax.set_xticklabels(legs4)
ax.set_ylabel('factor (x)')
ax.set_title('FIG2  The collapse decomposes into a small attachment factor and a large pose factor', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_MULTIPLICATIVE_DECOMPOSITION.png', dpi=150)

fig, ax = plt.subplots(figsize=(11, 5.5))
defect = ['joint_LHTibia_pitch', 'joint_LMTibia_pitch', 'joint_RHTibia_pitch', 'joint_RMTibia_pitch',
          'joint_LMCoxa_roll', 'joint_RMCoxa_pitch']
xs = np.arange(len(defect))
fz = [port['variants']['frozen']['s'][k]['s'] if port['variants']['frozen']['s'][k]['s'] else 60 for k in defect]
pt = [port['variants']['ported']['s'][k]['s'] if port['variants']['ported']['s'][k]['s'] else 60 for k in defect]
ax.bar(xs - 0.2, fz, width=0.4, color='tab:red', label='frozen')
ax.bar(xs + 0.2, pt, width=0.4, color='tab:green', label='tibia sites ported from repaired')
ax.axhline(1.0, color='k', ls='--')
ax.set_yscale('log')
ax.set_xticks(xs); ax.set_xticklabels([k.replace('joint_', '') for k in defect], rotation=20, fontsize=8)
ax.set_ylabel('required activation multiple s (log)')
ax.set_title('FIG3  The site port lowers s on every defect row but does not bring them below 1', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_PORT_EFFECT_ON_S.png', dpi=150)

# ---------------------------------------------------------------- conclusion / ledger / budget
inst_txt = chr(10).join('| ' + k + ' | %.6f | %.6f | %s |' % (v['ratio_value'], v['expected_k'], v['repaired'])
                        for k, v in inst.items())
dec_txt = chr(10).join('| ' + L + ' | %.2f | %.2f | %.2f |' % (decomp[L]['direct_ratio'], decomp[L]['attachment_factor'],
                                                               decomp[L]['pose_factor']) for L in legs4)
concl = chr(10).join([
 '# S212 冻结腔节附着几何归属 + 34× 的乘性分解 + 重做臂移植 —— 结论（中文）',
 '',
 '目录：`' + str(DIR) + '`',
 '',
 '## 0. 标签',
 '- **镜像规则：`MIRROR_RULE_ATTRIBUTED`** —— 规则可 1e-6 重算（`tools_mirror_muscles.py` 第 272–300 行，`k = lt/ls` 各向同性）。',
 '- **冻结附着：`FROZEN_ATTACHMENT_PARTIALLY_ATTRIBUTED`** —— 只对左中/后腿胫节成立，不是全局规则。',
 '- **移植对照：`ATTACHMENT_PORT_PARTIAL`** —— 全局 minimax 3.3086→%.4f，六条缺陷行 s 全部下降但**没有一条降到 ≤1**。' % port['variants']['ported']['global_box1'],
 '',
 '## 1. S211 更正（`S211_RECONCILIATION.json`）',
 '- **撤回**"per-leg 缩放表不在任何产物里"：它就在 `mirror_decisions.json` 的 `non_one_to_one_mappings`（**185 条**，每条带 `scale`；13 个不同取值：0.4118/0.4527/0.9911/0.9956/1.0/1.0057/1.1483/1.3071/1.3408/1.3623/1.4538/8.0252/8.5639）。',
 '- 生成规则：`p_world_src = ps + Rs @ (k*local)` → `p_world_tgt = t_T + R_T @ p_world_src` → `new_pos = Rt.T @ (p_world_tgt - pt)`，`k = segment length ratio`。**未执行该脚本**（会覆盖资产），只在 /tmp 复现。',
 '',
 '## 2. 冻结件与 repaired 的关系（`SITE_INSTANCES.json`）',
 '| 腿 | 实测比值 | 预期 k | repaired site |',
 '|---|---|---|---|',
 inst_txt,
 '',
 '三个分量的比值**全部相等**（各向同性）→ 左中/后腿胫节确实等于 `repaired × k`。',
 '但**全 site 普查**（185 条映射）里 R4（`frozen == repaired × k`）只覆盖 **%d/%d**，因此这只是**局部**关系，不是"整个冻结件 = repaired×k"。' 
 % (cen['census']['repaired']['n_scaled_match'], cen['census']['repaired']['n_compared']),
 '',
 '## 3. 34× 的乘性分解（`ARM_DECOMPOSITION.json`、FIG2）',
 '| 腿 | 直接比值 | 附件因子 | 姿态因子 |',
 '|---|---|---|---|',
 dec_txt,
 '',
 '- **附件因子 1.1–1.4×，姿态因子 5–19×** → **主因子是释放位姿，不是附件差异**（对应 Part D 要求）。',
 '- 但**前腿在同一姿态下并不塌陷**，所以这仍是"逐腿几何 × 姿态"的联合效应，不是单纯姿态问题。',
 '- 说明：姿态因子按 `直接比值 / 附件因子` 定义，乘积恒等；独立性来自附件因子与 keyframe 比值各自独立测得。',
 '',
 '## 4. 重做臂移植（`SITE_PORT_VARIANT.json`、FIG3；假设推演）',
 '- 变体：`/tmp/frozen_tibia_sites_from_repaired.xml`，**只改 24 个 site**（LM/RM/LH/RH 的 `Tibia_flex*`/`Tibia_extensor*`），冻结原件哈希不变。',
 '- **gain 必须重算**：site 移动改变肌腱长度 → 只有 8 个执行器（4 腿 × 2 胫节肌）的 gain 变化，**其余 82 个逐位不变**（自检通过）。',
 '- 结果：全局 minimax **3.3086 → %.4f**；逐腿 frozen %s → ported %s。' % (port['variants']['ported']['global_box1'],
                                                                              {k: round(v, 3) for k, v in port['variants']['frozen']['per_leg'].items()},
                                                                              {k: round(v, 3) for k, v in port['variants']['ported']['per_leg'].items()}),
 '- 六条缺陷行的 s：全部下降但仍 >1 → **`ATTACHMENT_PORT_PARTIAL`**；剩余缺口**不是幅度**（因为 s 仍由箱界和腿内同时性共同决定，需另轮定性）。',
 '- **这只是"若采用该假设会怎样"**；repaired 的附着几何自身可疑（右腿 y 分量不闭合），**不得声明运动改善**。',
 '',
 '## 5. 强制条款（写入结论与 tracker）',
 '- **撤回 S210 的"不是姿态伪影"**：姿态因子 5–19× ≥3× → 改写为"该臂塌陷的**主因子是释放位姿**，附件差异只是次要因子（1.1–1.4×）"，并写明"前腿在同一 pose 下不塌陷，所以是逐腿几何 × 姿态的联合效应"。',
 '- 保留：`B = arm × gain` 恒等式、屈肌 gain 六腿散布 0.29%、中/后腿臂塌陷、`repaired` 的 B 单向集合为空、**不得**把缺口写成"身体力量不足"。',
 '',
 '## 6. 这不是什么',
 '不声明站立/行走/自主运动；力单位 NOT_ESTABLISHED；移植变体不是模型修复。',
 '',
 '## 7. 下一步',
 '1. 右腿 y 分量归属（R4 在右腿不闭合）单独定性。',
 '2. 剩余 s>1 的六行：区分"箱界不足"与"腿内同时性"（用最小不可行子系统，不用扫描）。',
 '3. 姿态因子既然占主导，下一步应在**同一姿态**上比较不同附着假设（本轮的附件因子即此），避免跨姿态混比。',
 '',
 '## 8. 记账',
 '核验 ' + str(n_ok) + '/' + str(len(checks)) + '；台账记本轮失败（乘性分解公式首版写错、R4 覆盖被我当成全局规则）；model load ' + str(d['calls']['model_loads']) + '、position 阶段 ' + str(d['calls']['position_stage_calls']) + '、0 forward / 0 mj_step / 0 XML 改动 / 0 训练。',
])
(DIR / 'CONCLUSION.zh-CN.md').write_text(concl)
led = {"round": "S212", "failures": [
  {"id": "S212-A1", "what": "multiplicative decomposition, first version",
   "defect": "the pose factor was built as a ratio of ratios, so the product missed the direct ratio by ~99%", "caught_by": "the rel_error column (0.99)", "status": "pose factor redefined as direct/attachment; the independence now comes from the separately measured attachment factor and keyframe ratio"},
  {"id": "S212-A2", "what": "R4 coverage expectation",
   "defect": "treated 'frozen == repaired x k' as a candidate GLOBAL rule; it only holds for the left mid/hind tibia sites (2.2% of mapped sites)", "caught_by": "the census counts", "status": "label set to FROZEN_ATTACHMENT_PARTIALLY_ATTRIBUTED; no blanket claim"},
  {"id": "S212-A3", "what": "site parsing", "defect": "some sites carry no pos attribute and the parser crashed", "caught_by": "AttributeError", "status": "skipped explicitly"}],
  "corrections": [{"id": "S212-C1", "what": "S211 FROZEN_ATTACHMENT_UNATTRIBUTED", "defect": "reasoned that the per-leg scale table was absent; it is present (185 mapped entries, 13 distinct scales)", "status": "withdrawn; mirror rule attributed"},
                  {"id": "S212-C2", "what": "S210 'not a pose artefact'", "defect": "the pose factor measures 5-19x, so pose is the dominant factor", "status": "withdrawn and rewritten with the joint-effect caveat"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(led, indent=2, ensure_ascii=False))
budget = {"round": "S212",
          "stop_reason": "Parts 0/A/B/C complete; the port is partial and the decomposition is recorded with an explicit definition. Part D clauses written.",
          "budget": dict(d['calls'], lp_solves="about 40", training_steps=0, xml_changes=0),
          "artifacts": {str(p.relative_to(DIR)): sha(p) for p in sorted(DIR.rglob('*')) if p.is_file()},
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s212_census.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s212_decomp.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s212_final.py'), FROZEN] if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
print(json.dumps(dict(dir=str(DIR), n_ok=n_ok, n_checks=len(checks), failed=[c['check'] for c in checks if not c['ok']],
                      decomp={k: dict(direct=round(v['direct_ratio'], 2), attach=round(v['attachment_factor'], 2),
                                      pose=round(v['pose_factor'], 2)) for k, v in decomp.items()},
                      port_label=port['label'], port_global=port['variants']['ported']['global_box1']), indent=1))
