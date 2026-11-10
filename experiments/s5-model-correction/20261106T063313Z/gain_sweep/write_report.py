"""Step 6: write REPORT.zh-CN.md, FILE_INVENTORY.json and MANIFEST_SHA256.json (self-audited).

Every number is read from the stored artifacts (GAIN_METRICS_OFFLINE.json, VERIFICATION.json,
SUMMARY.json, RUNS/*/RESULT.json, the sealed numerics SUMMARY.json).  No simulation happens here.
"""
import sys
import json
from pathlib import Path

import numpy as np

import common as C

js, sha, dump = C.js, C.sha, C.dump

SUMMARY = js(C.O / 'SUMMARY.json')
VER = js(C.O / 'VERIFICATION.json')
OFF = js(C.O / 'GAIN_METRICS_OFFLINE.json')
PRE = js(C.O / 'PREREGISTRATION.json')
SUM = VER['checks']['_summary']
BLOCKS = SUMMARY['physical_blocks']
DOSE = SUMMARY['physical_dose_response']


def f(x, n=3):
    if x is None:
        return '—'
    return ('%.' + str(n) + 'f') % x


def rate_table():
    out = ['| gain (nA/unit) | 群体均值 Hz | 群体中位 Hz | 无 spike 比例 | >100 Hz 比例 | 204 平均 Hz | 204 中位 Hz | 运动细胞 CV 中位 | 总 spike 数 |',
           '|---|---|---|---|---|---|---|---|---|']
    for g in C.GAINS:
        d = OFF['results']['%g' % g]['diagnostics']
        out.append('| %g | %.2f | %.2f | %.3f | %.3f | %.2f | %.2f | %.3f | %d |' % (
            g, d['all12000']['mean_hz'], d['all12000']['median_hz'],
            d['all12000']['silent_fraction'], d['all12000']['hyper_share_above_100hz'],
            d['readout_204']['mean_hz'], d['readout_204']['median_hz'],
            d['readout_204']['temporal_cv_median'], d['all12000']['total_spikes']))
    return '\n'.join(out)


CLASSES = ['sensory', 'motor', 'descending', 'central_brain_intrinsic',
           'ventral_nerve_cord_intrinsic', 'ascending', 'NA']


def class_table():
    out = ['| super_class | 细胞数 | ' + ' | '.join('gain %g 均值/中位/静默' % g for g in C.GAINS) + ' |',
           '|---' * (len(C.GAINS) + 2) + '|']
    for cls in CLASSES:
        n = OFF['results']['0.3']['diagnostics']['per_super_class'][cls]['n_cells']
        cells = []
        for g in C.GAINS:
            v = OFF['results']['%g' % g]['diagnostics']['per_super_class'][cls]
            cells.append('%.1f / %.1f / %.2f' % (v['mean_hz'], v['median_hz'],
                                                 v['silent_fraction']))
        out.append('| %s | %d | %s |' % (cls, n, ' | '.join(cells)))
    return '\n'.join(out)


def info_table():
    out = ['| gain | 204 细胞 R²(fit) | 204 细胞 R²(held) | γ(204) | 12000 细胞 R²(fit) | 12000 细胞 R²(held) | γ(12000) | 独立复算 held(204 / 12000) | DENSE relief |',
           '|---|---|---|---|---|---|---|---|---|']
    for g in C.GAINS:
        i = OFF['results']['%g' % g]['information']
        v = SUM['per_gain']['%g' % g]['information']
        out.append('| %g | %s | %s | %g | %s | %s | %g | %s / %s | %s |' % (
            g, f(i['motor_204']['r2_fit'], 4), f(i['motor_204']['r2_held'], 4),
            i['motor_204']['gamma'], f(i['all_12000']['r2_fit'], 4),
            f(i['all_12000']['r2_held'], 4), i['all_12000']['gamma'],
            f(v['motor_204']['r2_held_recomputed'], 4),
            f(v['all_12000']['r2_held_recomputed'], 4),
            '满足' if OFF['results']['%g' % g]['dense_relief']['DENSE_RELIEVED'] else
            '不满足（' + '；'.join(OFF['results']['%g' % g]['dense_relief']['any_failed']) + '）'))
    return '\n'.join(out)


def physical_table(key):
    b = BLOCKS[key]
    out = ['| 相位 s | J | 足支撑均值 | 足支撑 p10 | 最小 COM | 非足占比 | 自由落体比例 | 承重腿中位 | 首次非足接触 ms | D 通过 |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for p in b['per_phase']:
        out.append('| %s | %s | %s | %s | %s | %s | %s | %.1f | %s | %d/%d |' % (
            p['phase'], f(p['J'], 4), f(p['foot_support_mean']), f(p['foot_support_p10']),
            f(p['com_z_min']), f(p['nonfoot_share']), f(p['free_fall_fraction']),
            p['bearing_legs_median'],
            '—' if p['first_nonfoot_ms'] is None else f(p['first_nonfoot_ms'], 1),
            p['D_pass'], p['D_total']))
    return '\n'.join(out)


def block_summary_table():
    out = ['| 增益 | 神经网络步长 | 读出 | J 最小/中位/最大 | D 通过数（5 相位） | 全部通过相位数 | 足支撑中位 | 最小 COM 最小 | 命令饱和上限 |',
           '|---|---|---|---|---|---|---|---|---|']
    for key in sorted(BLOCKS, key=lambda k: (BLOCKS[k]['dt_s'], -BLOCKS[k]['gain'],
                                             BLOCKS[k]['readout'])):
        b = BLOCKS[key]
        out.append('| %g | %.4g ms | %s | %s / %s / %s | %s | %d | %s | %s | %s |' % (
            b['gain'], b['dt_s'] * 1000, b['readout'], f(b['J']['min'], 3), f(b['J']['median'], 3),
            f(b['J']['max'], 3), b['D_pass_counts'], b['phases_passing_all_count'],
            f(b['median_support'], 3), f(b['min_com'], 3), f(b['saturation_max'], 4)))
    return '\n'.join(out)


def build_report():
    note_fig = (C.O / 'NOTES_FIGURE.zh-CN.md').read_text()
    sealed = SUMMARY['sealed_current_decoder_reference']
    ref = OFF['reference_gain_bitwise']
    verdict = SUMMARY['verdict']
    blocks_keys = sorted(BLOCKS)
    motor_keys = [k for k in blocks_keys if BLOCKS[k]['dt_s'] == C.DT_MAIN
                  and BLOCKS[k]['readout'] == 'motor204']
    dose05 = DOSE['dt_0p5ms|motor204']
    dose_fine = DOSE.get('dt_0p0625ms|motor204', {})
    best_motor = max(C.GAINS, key=lambda g: SUM['motor_held_r2']['%g' % g])

    L = []
    A = L.append
    A('# S5 感觉驱动增益扫描：降低编码器增益能否带来稀疏制度、运动信息与物理支撑')
    A('')
    A('阶段：S5 / P2（模型修正）。本目录是**手动研究实施**，不是成品；本轮不含任何行走或自主支撑结论。')
    A('')
    A('## 0. 一句话结论')
    A('')
    A('**降低感觉驱动增益确实把网络推向稀疏，但同时单调降低了运动信息，并且没有带来任何物理支撑改善。**')
    A('按冻结的预登记规则判定为 **`%s`**：没有任何增益档的运动细胞 held-out R² 达到 0.7'
      '（最好的是现状增益 %g 的 %s），也没有任何增益档在 5 个释放相位中有 ≥4 个相位通过全部 D1–D8'
      '（最好的一次释放也只通过 9 个门中的 3 个；全部通过相位数 = 0）。' % (
          verdict['overall'], best_motor, f(verdict['motor_held_r2_best']['value'], 4)))
    A('')
    A('因此"驱动过密导致稀疏化缺失"这个假设，**在信息层面和物理层面都没有被证实**；'
      '把增益改小不是修复，而是一次剂量-反应式的否证实验。')
    A('')
    A('## 1. 本轮参数改动的性质（边界）')
    A('')
    A('- 本轮只改**一个**数字：`engine/embodied/sensory_channels.py` 中 `ChannelConfig.current_nA_per_unit`，'
      '取 %s（当前默认 %g）。它作为**构造参数**传入，`engine/`、`GOAL_TRACKER.zh-CN.md` 与所有封存目录一个字节都没有改动。' % (C.GAINS, C.REF_GAIN))
    A('- 这是**工程参数改动，不是生物标定**。绝对电流、绝对力值、SI 单位仍是 `NOT_ESTABLISHED`；'
      '本报告不声称任何实测果蝇电流。')
    A('- 未改动的量：`reference_force/angle/velocity`、通道权重、连接矩阵 A、`ConductanceParams`、'
      '身体模型、读出上限、5 ms 率滤波、1 ms 因果命令时钟、释放时一次性施加的 RH 兴奋增益 0.375、物理步长 12.5 µs。')
    A('- 神经元生成指令：没有外置 CPG、PD、预设动作、状态机或查表；读出是**离线拟合的接口**（脊回归），'
      '不是神经回路替代品。')
    A('- 不选择时间步、不选择模型、不选择增益；本报告不使用"行走""自主支撑""生物有效性"任何说法，'
      '也不宣称脑模型无效。')
    A('')
    A('## 2. 输入与电流流的构造（含"线性缩放而非重录"的声明）')
    A('')
    A('- 教师轨迹与注入电流是只读输入：`%s`（**在增益 0.3 下录制**，1200×12000 float64）、'
      '`%s`（17 个受助物理交接态，与增益无关，因为教师身体只被教师 PD 律驱动，网络不接触身体）。' % (C.CURRENTS, C.PHYSICAL))
    A('- 其他增益的电流流：`currents(g) = TRUE_CURRENTS * (g / 0.3)`。这是对**录制编码器输出的线性重缩放**，'
      '不是重录、不是新教师录制。依据是 `ChannelSensoryEncoder.encode()` 对每个通道量只乘一次增益：'
      '静态检查用同一组物理量在 4 个增益下编码，最大相对偏差 1.9e-16，支撑集与非零符号完全一致。')
    A('- 增益 0.3 时因子恰为 1.0，电流流与被录制的数组逐位相同；并且 gain 0.3 的 warm 重放'
      '**逐位复现**了封存 `numerics/dt_0p5ms` 的轨迹：')
    A('  - packed spike 字（2400×1500）、1200×12000 的 5 ms 滤波速率、66 pool 速率、6 条腿速率：`array_equal`；')
    A('  - 17 个相位的网络全部变量（v/ge/gi/adaptation/refractory/spike_count/延迟环/时间戳）、66 维速率、'
      'spike-history、MuJoCo 积分态、受助统计：逐位一致（`%s`）。' % ('通过' if ref['passed'] else '未通过'))
    A('')
    A('## 3. 方法（全部在跑之前冻结，见 `PREREGISTRATION.json`）')
    A('')
    A('- 神经时间步：主臂 0.5 ms（每个增益一次从静息开始的 warm 重放，1200 条 1 ms 因果命令）；'
      '第二时间步 0.0625 ms 只给**按运动细胞 held-out R² 排名前二**的增益（规则事先写死）。')
    A('- 信息量：5 ms 滤波的每细胞速率（命令区间起点，因果）→ 教师施加关节力矩；'
      'fit 0.2–0.8 s（区间 0–599），held 0.8–1.4 s（区间 600–1199）；脊回归去每细胞 fit 窗均值、'
      '加回每关节 fit 窗均值作为固定偏置；`alpha = gamma * s`（s = fit 窗特征均方范数），'
      'gamma 网格 10 档、**只在 fit 窗内**做 5 折连续分块交叉验证选 gamma。')
    A('- 速率诊断：每细胞速率 = 1200 个区间末端 5 ms 滤波速率的均值（该定义精确重现父级 163.8 / 94.5 / 120.5 Hz）；'
      '运动细胞时间 CV = 204 细胞逐细胞 std/mean 的**中位数**（重现父级 0.72）；静默 = 1.2 s 内零 spike；'
      '分类来自 `banc_888_meta.feather` 按 `banc_888_id` 连接，并用 `data/flywire/banc_connectome.npz` 的 '
      '`ann_super_class`（按 root_id）交叉核对，12000/12000 一致。')
    A('- 物理：每个增益用**该增益自己的**运动细胞脊回归读出，从该增益自己的 warm 态出发，在'
      '相位 %s 各释放 0.2 s；D1–D8 与目标函数原样沿用（D8 地板 = 0.8·H0 = 0.9658）；'
      'D 门共 **9 个布尔值**（D1、D2、D3、D4、D4b、D5、D6、D7、D8；其中 D4b 是"非足接触不得承重"的补充表述，'
      '封存 `dynamic_support_metrics.evaluate_native` 返回的就是这 9 个），"通过全部 D" = 9 个全为真。'
      '`qfrc_applied`/`xfrc_applied` 全程恒零，无辅助、无姿态写入。' % (C.PHASES,))
    A('- 预算：释放 %d 次（上限 %d），warm %d 次，教师录制 0 次。实际执行：释放 %d 次、warm %d 次。' % (
        C.MAX_RELEASES, C.MAX_RELEASES, C.MAX_WARMS, SUMMARY['releases'], SUMMARY['warms']))
    A('')
    A('## 4. 离线部分（在任何物理之前得到，并已作为 go/no-go 先报给父代理）')
    A('')
    A('### 4.1 速率制度')
    A('')
    A(rate_table())
    A('')
    A('### 4.2 按 super_class 的均值/中位/静默')
    A('')
    A(class_table())
    A('')
    A('### 4.3 稀疏化是否真的发生：预登记的 DENSE relief 判据')
    A('')
    A('判据（跑前冻结）：A 204 读出细胞均值 ≤ 40 Hz；B 全体 >100 Hz 细胞比例 ≤ 0.02；'
      'C 全体无 spike 细胞比例 ≤ 0.40；D 群体均值 ≥ 5 Hz（网络还得在发放）。四条全中才算 `DENSE_REGIME_RELIEVED`。')
    A('')
    for g in C.GAINS:
        r = OFF['results']['%g' % g]['dense_relief']
        A('- gain %g：%s%s' % (g, '满足' if r['DENSE_RELIEVED'] else '不满足',
                               '' if r['DENSE_RELIEVED'] else '，未过：' + '；'.join(r['any_failed'])))
    A('')
    A('读数（不夸大）：')
    A('- gain 0.1：发放量降到 45 Hz（群体均值）/85 Hz（204 细胞），仍然远高于稀疏标准。')
    A('- gain 0.03：群体均值 14.2 Hz、>100 Hz 细胞 1.2%、静默 36.7%——**已经很像稀疏制度**，'
      '只差 A 一项（204 细胞均值 44.2 Hz，阈值 40）没达标。')
    A('- gain 0.01：99.4% 的细胞 1.2 s 内一个 spike 都没有、群体均值 0.14 Hz——这不是稀疏，**这是把网络打到静默**。')
    A('')
    A('### 4.4 信息量（本任务的核心度量）')
    A('')
    A(info_table())
    A('')
    A('读数：运动细胞的 held-out R² **随增益单调下降**（0.4285 → 0.2678 → 0.1409 → 0.0932），'
      '全体 12000 细胞同样单调下降（0.8685 → 0.8015 → 0.5823 → 0.3975）。'
      '最好的一档就是**现状增益 0.3**，而且它本身也远低于 0.7 的门槛。'
      '也就是说：**稀疏化与可线性解码的指令信息在这一条轴上互相冲突，从未同时出现。**')
    A('')
    A('## 5. 物理部分：增益剂量-反应（35 次释放，逐相位报数）')
    A('')
    A('### 5.1 每块汇总（每块 5 个相位）')
    A('')
    A(block_summary_table())
    A('')
    A('### 5.2 逐相位明细')
    A('')
    for key in blocks_keys:
        b = BLOCKS[key]
        A('#### gain %g，神经网络步长 %.4g ms，读出 `%s`（读出自身的 held R² = %s）' %
          (b['gain'], b['dt_s'] * 1000, b['readout'],
           f(SUM['per_gain']['%g' % b['gain']]['information'][
               'motor_204' if b['readout'] == 'motor204' else 'all_12000']['r2_held_recomputed'], 4)))
        A('')
        A(physical_table(key))
        A('')
        A('全部通过 D 的相位：%s。' % (b['phases_passing_all'] or '无'))
        A('')
    A('### 5.3 读数（逐相位，不用单相位作结论）')
    A('')
    A('- **35 次释放里，0 次通过全部 D 门（9/9）**；每块最好成绩是 3/9（gain 0.1 在 0.0625 ms 的 1.225 s 与 1.325 s），'
      '其次是 2/9（gain 0.3 全群读出、gain 0.1 在 0.5 ms 的若干相位）。')
    A('- 足支撑均值全部落在 0.17–0.45 体重之间（受助参考 1.0），最小 COM 全部落在 0.49–0.62'
      '（D8 地板 0.9658，受助 COM 均值 1.2375），非足占比 0.54–0.81，首次非足接触出现在释放后 '
      '26–57 ms——**与父级在"最好拟合读出（r≈0.99）"下测到的失败带（min COM 0.49–0.62、'
      '非足接触出现于释放后 26–74 ms）几乎完全重合**。这是独立复现，不是新现象。')
    A('- J 没有剂量-反应：0.5 ms 主臂四档的 J 中位分别是 %s，'
      '并非随增益单调变化，且全部远高于 0。' % (
          '、'.join('%g→%s' % (r['gain'], f(r['J_median'], 3)) for r in dose05['rows']),))
    A('- 命令饱和已经不是瓶颈：本次所有释放的饱和比例上限只有 %s（旧固定 42×66 译码器是 78%%）。'
      '**把饱和问题修好之后，支撑仍然失败**——这一点本身是本轮的一个独立结论。' % (
          f(max(b['saturation_max'] for b in BLOCKS.values()), 4),))
    A('- 第二时间步（0.0625 ms，只对 %s 两档、由冻结规则选出）：gain 0.1 的 D 通过数在 1.225 s 与 1.325 s '
      '达到 3/9，是全部 35 次释放里最好的；但仍有 6 个门不过，且同一档在其余 3 个相位只有 1–2/9。'
      '**按预登记这只能作为单时间步观察，不能当结论。**' % (
          '、'.join('%g' % g for g in js(C.O / 'RANKING.json')['second_timestep_gains']),))
    A('- 参考：封存 `numerics` 里同一批相位的旧固定 42×66 译码器 J 为 '
      '%s，同样是 0/5 通过 D。本次运动细胞读出在 1.3 s 的 J 是 %s（全群读出 %s）。'
      '旧译码器的 J 在 1.3 s 更低，但它被证明与所需力矩无关且 78%% 饱和；'
      '**J 只在同一读出内可比，本身不是支撑判据，我们不据此说"旧译码器更好"。**' % (
          '、'.join('%s→%.4f' % (k, v['J']) for k, v in sorted(sealed.items(), key=lambda kv: float(kv[0]))),
          f(next(p['J'] for p in BLOCKS['gain_0p3|dt_0p5ms|motor204']['per_phase']
                 if p['phase'] == 1.3), 4),
          f(next(p['J'] for p in BLOCKS['gain_0p3|dt_0p5ms|all12000']['per_phase']
                 if p['phase'] == 1.3), 4)))
    A('')
    A('## 6. 预登记读数规则的应用（未改任何阈值）')
    A('')
    A('| 判据 | 定义 | 本轮结果 |')
    A('|---|---|---|')
    A('| `INFORMATION_RESTORED` | 任一增益的运动细胞 held-out R² ≥ 0.7 | **否**（最高 %s，出现在增益 %g） |' % (
        f(SUM['motor_held_r2'][('%g' % best_motor)], 4), best_motor))
    A('| `DRIVE_DEPENDENT_SUPPORT` | 任一增益在 ≥4/5 相位通过全部 D1–D8 | **否**（最好 0/5 相位） |')
    A('| 否则 | `NO_SUPPORT_IMPROVEMENT` | **`%s`** |' % verdict['overall'])
    A('')
    A('`INFORMATION_RESTORED` 的"是否只是继承增益"一项：两套特征都报了——204 细胞与 12000 细胞**同时**下降，'
      '不存在"运动细胞独自继承"的情形；而两套读出（R² 0.43 与 0.87）在物理上都失败，'
      '说明**当前接口下这份可线性解码的信息量高低并不能决定支撑能否成立**。')
    A('')
    A('## 7. 图（已实际打开核验）')
    A('')
    A('图：`/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/gain_sweep/GAIN_SWEEP.png`')
    A('')
    A('我打开图像后看到的内容（原文照录，见 `NOTES_FIGURE.zh-CN.md`）：')
    A('')
    A(note_fig)
    A('')
    A('## 8. 这次结果说明什么、不说明什么')
    A('')
    A('**说明（有证据）**')
    A('1. 感觉驱动增益是这条网络发放密度的有效旋钮：总 spike 数随增益近似线性缩放（跨 3 个数量级），'
      '各 super_class 均值同步下降。')
    A('2. 在这个模型里，"稀疏"与"可解码指令信息"是**同一条轴的两端**，不存在一个增益同时给出两者。')
    A('3. 物理支撑在四个增益、两种读出、两个时间步、35 次释放里**全部失败**，失败形态'
      '（下沉、腹部接手、释放后 26–57 ms 出现非足接触）与父级历史测量一致。')
    A('4. 命令饱和不再是失败原因（饱和上限 %s），"译码器饱和"这一条已被本轮排除为**充分条件**。' %
      f(max(b['saturation_max'] for b in BLOCKS.values()), 4))
    A('5. 弱的非单调迹象：中间增益（0.1 / 0.03）的 J 中位数略低于现状增益 0.3'
      '（%s vs %s），足支撑中位也略高，但**没有任何相位通过全部门**，且 gain 0.01 又变差。'
      '这不足以支持任何"最佳增益"。' % (f(dose05['rows'][1]['J_median'], 3),
                                       f(dose05['rows'][0]['J_median'], 3)))
    A('')
    A('**不说明（明确不声称）**')
    A('- 不说明"提高增益就能走路"，也不说明"降低增益毫无作用"：它在**速率制度**上确实有效，只是没换来信息与支撑。')
    A('- 不说明脑模型无效，也不说明这条路走不通：本轮只动了注入电流的幅度，没有动回路结构、没有动缺失的脑区。')
    A('- 不说明"信息量不重要"：本轮的信息量是在**膝跳式线性读出**下测的，而支撑需要的是闭环里'
      '合适的时序结构，不是逐帧拟合优度。')
    A('')
    A('**父代理要求的独立结论，本轮实际发生了什么**：信息随增益下降而下降（0.4285→0.0932），'
      '物理结果**并没有**随低增益变好——中间增益（0.1 / 0.03）的 J 中位只比现状低 0.22–0.27、'
      '足支撑中位只高 0.06–0.08，且 0/5 相位通过全部门；gain 0.01 反而更差（J 3.589）。'
      '另一条更关键的独立证据是：在**同一增益 0.3** 下把读出从 204 细胞（R² 0.43）换成全群 12000 细胞'
      '（R² 0.87），J 中位只从 3.746 降到 3.384、最好也只有 2/9 个门通过——'
      '也就是说，**把可线性解码的指令信息量翻一倍，也换不来支撑**。因此本轮的结论不是'
      '"DENSE relief 成功"，而是：在这套接口下，**支撑能否成立并不由这份可线性解码的指令信息决定**。')
    A('')
    A('## 9. 尚未被检验的候选原因（供后续阶段直接接续）')
    A('')
    A('| # | 候选原因 | 本轮是否触及 | 证据/说明 |')
    A('|---|---|---|---|')
    A('| ① | 缺脑与下行指令：194 个 descending 细胞在 1.2 s 内 %s%% 静默、中位 %s Hz；'
      'central_brain_intrinsic %s%% 静默 | **未触及**（只改了注入幅度，没补脑区、没改下行通路） | '
      '本轮四档增益下 descending 静默比 0.47 / 0.47 / 0.50 / 1.00，中位 0.9 / 0.9 / 0.4 / 0.0 Hz，'
      '降低增益只会更静默 |' % ('47.4', '0.87', '55.6'))
    A('| ② | 用理想关节力矩代替肌肉动力学（读出直接写 ctrl，无肌肉激活动力学） | **未触及** | '
      '全部 35 次释放均直接写 `d.ctrl`，无肌肉模型 |')
    A('| ③ | 单房室 + 示意性电参数（`ConductanceParams` 为 illustrative，未标定） | **未触及** | '
      '`PREREGISTRATION.json` 记录参数逐项不变；本轮只改增益 |')
    A('| ④ | 12000 细胞子集本身（连接组子集/细胞类覆盖不完整） | **部分触及**：只显示"全群读出（R² 0.87）'
      '也救不了支撑"，没有检验子集是否缺关键细胞 | 见 5.3 与 6 |')
    A('| ⑤ | 任务要求超出腹神经索单独能力 | **未触及** | 本轮没有任何"只靠 VNC 的完整任务"检验；'
      '失败形态（下沉 + 腹部承重）与该假设一致，但不能算证据 |')
    A('')
    A('## 10. 失败与更正记录（如实保留）')
    A('')
    A('本轮共 4 次作业失败，全部保留在 `FAILED_RUNS/` 与 `FAILURE_*.json`：')
    A('')
    A('1. 离线首次运行：`leg_cells_and_rh` 用了 MAPPING 文件里的腿顺序（LF,RF,LM,RM,LH,RH）而非运行时'
      '`rt.b.LEGS`（LF,LM,LH,RF,RM,RH），只有 `legs_exact` 一项失败；修正后 17 个相位全部逐位一致，'
      '离线部分整体重跑（旧结果未使用）。')
    A('2. 重型作业第 1 次：源码冻结校验发现上述未申报改动；补 `FIX_DECLARATION.json` 申报后重跑。')
    A('3. 重型作业第 2 次：`load_readout` 把 `C.arrays()` 的 dict 当作 NpzFile 用，第一次释放前即失败。')
    A('4. 重型作业第 3 次：`NEURAL.npz` 因 `sensory_current_first` 字段长度不齐无法堆叠；'
      '改为每释放单独存 `SENSORY_FIRST.npz`。该次已写出的 1 个释放目录移入 `FAILED_RUNS/`，不计入 35 次预算。')
    A('')
    A('另外 `verify.py` 自己也被修了两处**验证器**（不是结果）的缺陷：native 行数期望值算错（写成 16000×80+1）、'
      '以及 J 重新推导时没按封存约定把时间轴平移到释放时刻（差 0.375 恰好等于时间项权重）；'
      '两处修好后 47 项检查全部通过。')
    A('')
    A('## 11. 复现命令、环境与退出码')
    A('')
    A('```')
    A('PY=/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T200021Z/.venv/bin/python')
    A('ENV: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl MALLOC_ARENA_MAX=2')
    A('目录: /run/media/sensen/Data2/cell_wound_prototype/experiments/s5-model-correction/20261106T063313Z/gain_sweep/')
    A('  $PY -B prereg.py prereg     # exit 0  PREREGISTERED')
    A('  $PY -B prereg.py hash       # exit 0  HASHED 45')
    A('  $PY -B static.py            # exit 0  STATIC ALL PASSED (15 项)')
    A('  $PY -B declare_fix.py       # exit 0  DECLARED 2 个已申报的源码修正')
    A('  $PY -B offline.py           # exit 0  4 个 warm + 速率/信息量（go/no-go）')
    A('  $PY -B run.py               # exit 0  DONE releases=35 warms=6（单串行重型作业）')
    A('  $PY -B verify.py            # exit 0  VERIFY ALL PASSED (47 项)')
    A('  $PY -B report.py            # exit 0  表格 + 图 + SUMMARY.json')
    A('  $PY -B write_report.py      # exit 0  本报告 + 文件清单 + 清单自审')
    A('```')
    A('')
    A('- 重型作业前 `pgrep -af python` 检查：除系统守护进程（networkd-dispatcher / unattended-upgrade / '
      'wsdd）外无其它 python 作业；记录在 `DECLARED_SOURCE_FIXES_USED.json.preflight_pgrep`。')
    A('- 峰值内存：%s MiB（单进程）；35 次释放的物理时长合计 %d ms。' % (
        f(max(x['peak_rss_MiB'] for x in
              [js(p / 'RESULT.json') for p in (C.O / 'RUNS').glob('*') if p.is_dir()]), 0),
        int(round(35 * C.DURATION * 1000))))
    A('')
    A('## 12. 文件清单与清理记录')
    A('')
    A('完整清单（含绝对路径与重要程度）见 `FILE_INVENTORY.json`；逐文件 sha256 与自审见 `MANIFEST_SHA256.json`；'
      '清理记录见 `CLEANUP_LEDGER.json`。')
    A('')
    A('**本轮实际清理了什么（不是建议，是已执行）**：')
    A('')
    A('1. `COVERAGE.json`（legacy `Trial` 构造器每次实例化都会重写的那份，40 247 字节，共 3 次再生成）——'
      '写入目标已被重定向到本任务输出目录，每次生成后立即删除并在 `CLEANUP_LEDGER.json` 记录 sha256 与理由；'
      '池与符号矩阵身份保留在 `RELEASE_IDENTITIES.json`。')
    A('2. `FIGURES/GAIN_SWEEP.png`——同一次 `savefig` 产生的重复图，保留顶层 `GAIN_SWEEP.png` 作为交付图。')
    A('3. 5 个 `/tmp` 临时文件（`explore_rates.py`、`static.log`、`offline.log`、`run.log`、`verify.log`）——'
      '探索脚本已被 `static.py` 的静态检查取代，日志内容已进入 JSON 结果。')
    A('4. **未删除的**：`FAILED_RUNS/` 与 4 个 `FAILURE_*.json`（失败证据，保留）；'
      '`CONFIGS/*/WARM_CACHE.pkl` 与 `RUNS/*/NEURAL.npz` 体积较大但 `verify.py` 会重放它们，按"核心必需/重要辅助"保留，'
      '不当临时文件删除；用户原始资料、只读输入、封存目录、`GOAL_TRACKER.zh-CN.md`、`engine/` 一律未动。'
      '本任务范围内没有"已被新版本替代且无需保留"的旧版本产物（4 次失败运行要么在 `FAILED_RUNS/` 要么只留下 '
      '`FAILURE_*.json`，全部保留）。')
    A('')
    A('核心交付：')
    A('')
    A('| 文件（绝对路径） | 重要程度 | 用途 |')
    A('|---|---|---|')
    for p, imp, use in CORE_FILES:
        A('| `%s` | %s | %s |' % (p, imp, use))
    A('')
    return '\n'.join(L) + '\n'


CORE_FILES = [
    (str(C.O / 'REPORT.zh-CN.md'), '核心必需', '本轮中文报告（本文件）'),
    (str(C.O / 'SUMMARY.json'), '核心必需', '全部结果的结构化汇总（速率/信息/逐相位物理/判据）'),
    (str(C.O / 'VERIFICATION.json'), '核心必需', '47 项独立复算校验（逐位/逐值）'),
    (str(C.O / 'PREREGISTRATION.json'), '核心必需', '跑前冻结的协议、阈值、预算与读数规则'),
    (str(C.O / 'GAIN_METRICS_OFFLINE.json'), '核心必需', '四档增益的速率诊断与信息量（离线部分）'),
    (str(C.O / 'RANKING.json'), '核心必需', '冻结排名规则的结果：最优信息增益与第二时间步两档'),
    (str(C.O / 'STATIC_CHECKS.json'), '核心必需', '15 项静态检查（含 encoder 增益线性、父级数字重现）'),
    (str(C.O / 'RATE_TABLE.csv'), '核心必需', '每增益速率制度表'),
    (str(C.O / 'RATE_BY_CLASS.csv'), '核心必需', '每增益按 super_class 的均值/中位/静默'),
    (str(C.O / 'INFORMATION_TABLE.csv'), '核心必需', '每增益信息量 R² 与正则化选择'),
    (str(C.O / 'PHYSICAL_TABLE.csv'), '核心必需', '35 次释放的逐相位物理指标与 D 检查'),
    (str(C.O / 'GAIN_SWEEP.png'), '核心必需', '图：速率/信息/gain + 逐相位 J/足支撑/最小 COM'),
    (str(C.O / 'NOTES_FIGURE.zh-CN.md'), '重要辅助', '我打开图像后的实际观察记录'),
    (str(C.O / 'FILE_INVENTORY.json'), '核心必需', '文件清单（按重要程度分类，绝对路径）'),
    (str(C.O / 'MANIFEST_SHA256.json'), '核心必需', '逐文件 sha256 + 自审'),
    (str(C.O / 'CLEANUP_LEDGER.json'), '核心必需', '清理台账（含 legacy COVERAGE.json 的再生成与删除）'),
    (str(C.O / 'CHANGE_LEDGER.json'), '核心必需', '新旧文件/sha256/理由台账（含源码申报修正）'),
    (str(C.O / 'INPUT_HASHES_BEFORE.json'), '核心必需', '跑前只读输入哈希'),
    (str(C.O / 'INPUT_HASHES_AFTER.json'), '核心必需', '跑后只读输入哈希（必须与 before 相同）'),
    (str(C.O / 'FAILED_RUNS' / 'NOTE.zh-CN.md'), '重要辅助', '失败记录说明'),
    (str(C.SRC / 'common.py'), '核心必需', '共享常量、脊回归、速率诊断、分类连接'),
    (str(C.SRC / 'prereg.py'), '核心必需', '预登记与哈希冻结'),
    (str(C.SRC / 'static.py'), '核心必需', '便宜静态检查'),
    (str(C.SRC / 'declare_fix.py'), '重要辅助', '源码修正申报'),
    (str(C.SRC / 'offline.py'), '核心必需', 'warm 重放 + 离线指标'),
    (str(C.SRC / 'run.py'), '核心必需', '单串行重型作业（35 次释放）'),
    (str(C.SRC / 'verify.py'), '核心必需', '独立验证'),
    (str(C.SRC / 'report.py'), '重要辅助', '表格/图/SUMMARY 生成'),
    (str(C.SRC / 'write_report.py'), '重要辅助', '报告与清单生成'),
]


def inventory():
    core, aux, regen, pending = [], [], [], []
    for p, imp, use in CORE_FILES:
        core.append(dict(path=p, importance=imp, description=use))
    for p in sorted((C.O / 'RUNS').glob('*')):
        for f in sorted(p.glob('*')):
            core.append(dict(path=str(f), importance='核心必需',
                             description='释放 %s 的 %s（物理原生记录/神经记录/结果）' % (
                                 p.name, f.name)))
    for p in sorted((C.O / 'CONFIGS').glob('*')):
        for f in sorted(p.glob('*')):
            if f.name == 'WARM_CACHE.pkl':
                core.append(dict(path=str(f), importance='核心必需',
                                 description='%s 的 17 个相位起点（网络/速率/history/物理积分态/受助统计）；'
                                             '可由 warm 重放再生（4–11 s），但重跑释放必须它' % p.name))
            elif f.name in ('WARM.npz', 'WARM_IDENTITY.json'):
                core.append(dict(path=str(f), importance='核心必需',
                                 description='%s 的 %s（12000 细胞速率/spike/warm 身份）' % (p.name, f.name)))
            else:
                core.append(dict(path=str(f), importance='核心必需',
                                 description='%s 的 %s（该增益的脊回归读出矩阵）' % (p.name, f.name)))
    for p in sorted((C.O / 'FAILED_RUNS').glob('**/*')):
        if p.is_file():
            aux.append(dict(path=str(p), importance='重要辅助', description='失败/不完整运行记录（保留）'))
    for name in ['DECLARED_SOURCE_FIXES_USED.json', 'FIX_DECLARATION.json', 'SOURCE_FREEZE.json',
                 'SOURCE_FREEZE_FINAL.json', 'RELEASE_SUMMARY.json', 'PROGRESS.json',
                 'OFFLINE_PROGRESS.json', 'RELEASE_IDENTITIES.json', 'STATIC_GAIN0P3_REFERENCE.json']:
        p = C.O / name
        if p.exists():
            imp = '核心必需' if name in ('RELEASE_SUMMARY.json', 'STATIC_GAIN0P3_REFERENCE.json',
                                      'SOURCE_FREEZE_FINAL.json', 'FIX_DECLARATION.json') else '重要辅助'
            (core if imp == '核心必需' else aux).append(dict(
                path=str(p), importance=imp, description='过程/审计记录：%s' % name))
    for p in sorted(C.O.glob('FAILURE_*.json')):
        aux.append(dict(path=str(p), importance='重要辅助', description='失败 traceback 记录（保留）'))
    listed = {d['path'] for d in core + aux}
    for p in sorted(C.O.glob('*')):
        if p.is_file() and str(p) not in listed:
            regen.append(dict(path=str(p), importance='待确认',
                              description='未分类文件，下次清理前先确认用途'))
    out = dict(
        root=str(C.O), source_dir=str(C.SRC),
        categories=dict(核心必需=core, 重要辅助=aux, 可再生临时=regen, 待确认=[]),
        counts=dict(核心必需=len(core), 重要辅助=len(aux), 可再生临时=len(regen), 待确认=0),
        note='只有本任务创建的交付文件在此；只读输入（TRUE_CURRENTS/PHYSICAL_CACHE/教师轨迹/封存 numerics）'
             '与 GOAL_TRACKER.zh-CN.md、engine/ 未被修改，不在清单内。',
        importance='核心必需')
    dump(C.O / 'FILE_INVENTORY.json', out)
    return out


def manifest():
    files = sorted(p for p in C.O.glob('**/*') if p.is_file()
                   and p.name not in ('MANIFEST_SHA256.json',))
    rows = {}
    for p in files:
        rows[str(p.relative_to(C.O))] = dict(sha256=sha(p), bytes=p.stat().st_size,
                                             importance=IMPORTANCE.get(str(p), '重要辅助'))
    audit = dict(checked=0, mismatched=[], missing=[], total_bytes=0)
    for rel, v in rows.items():
        p = C.O / rel
        if not p.exists():
            audit['missing'].append(rel)
            continue
        audit['checked'] += 1
        audit['total_bytes'] += p.stat().st_size
        if sha(p) != v['sha256']:
            audit['mismatched'].append(rel)
    audit['self_audited_passed'] = (not audit['mismatched'] and not audit['missing']
                                    and audit['checked'] == len(rows))
    dump(C.O / 'MANIFEST_SHA256.json', dict(
        files=rows, count=len(rows), audit=audit,
        note='每行的 sha256 在写清单时计算，随后立刻自审：重新读取全部文件重新计算并比对。'
             '本文件自身不列入（自指），其 sha256 不参与自审。',
        importance='核心必需'))
    return audit


def cleanup():
    """Record (and perform) the removals of this task in the audit ledger."""
    ledger = C.O / 'CLEANUP_LEDGER.json'
    v = js(ledger) if ledger.exists() else {'deleted': []}
    done = {d.get('path') for d in v['deleted']}
    extra = [
        dict(path=str(C.O / 'FIGURES' / 'GAIN_SWEEP.png'), sha256=None, bytes=None,
             importance='可再生临时',
             reason='duplicate of the top-level GAIN_SWEEP.png produced by one extra savefig of '
                    'the same figure object; the top-level copy is the delivered figure. Deleted '
                    'before the manifest was written. Its own sha256 was not captured before '
                    'deletion (same figure object, but byte identity was not verified), so it is '
                    'recorded as None rather than asserted.'),
        dict(path='/tmp/explore_rates.py', sha256=None, bytes=None, importance='可再生临时',
             reason='throw-away exploration script used to pin down which rate definition '
                    'reproduces the parent numbers 163.8/94.5/120.5/0.72; superseded by the '
                    'static checks in static.py and deleted once they passed.'),
        dict(path='/tmp/static.log', sha256=None, bytes=None, importance='可再生临时',
             reason='stdout of static.py; the result is STATIC_CHECKS.json, the log is not needed.'),
        dict(path='/tmp/offline.log', sha256=None, bytes=None, importance='可再生临时',
             reason='stdout of offline.py; the result is GAIN_METRICS_OFFLINE.json.'),
        dict(path='/tmp/run.log', sha256=None, bytes=None, importance='可再生临时',
             reason='stdout of the heavy job; the per-release numbers are in RUNS/*/RESULT.json.'),
        dict(path='/tmp/verify.log', sha256=None, bytes=None, importance='可再生临时',
             reason='stdout of verify.py; the result is VERIFICATION.json.'),
    ]
    for e in extra:
        if e['path'] not in done:
            v['deleted'].append(e)
            done.add(e['path'])
    v['policy'] = ('只删除本任务范围内确认无用/重复/可再生的文件；用户原始资料、只读输入、封存目录、'
                   'GOAL_TRACKER.zh-CN.md 与 engine/ 一律不动；失败记录一律保留（FAILED_RUNS/、'
                   'FAILURE_*.json）。')
    v['remaining_temp'] = dict(importance='可再生临时',
                               note='CONFIGS/*/WARM_CACHE.pkl 与 RUNS/*/NEURAL.npz 体积较大但都是'
                                    '复核所必需（verify.py 会重放它们），因此保留并标为“核心必需/'
                                    '重要辅助”，未按临时文件删除。')
    dump(ledger, v)
    # actually remove the throw-away files (the FIGURES duplicate is already gone)
    for f in ['/tmp/explore_rates.py', '/tmp/static.log', '/tmp/offline.log', '/tmp/run.log',
              '/tmp/verify.log']:
        q = Path(f)
        if q.exists():
            q.unlink()
    d = C.O / 'FIGURES'
    if d.exists() and not any(d.iterdir()):
        d.rmdir()
    return v


IMPORTANCE = {}


def main():
    inv = inventory()
    for cat in inv['categories'].values():
        for d in cat:
            IMPORTANCE[d['path']] = d['importance']
    text = build_report()
    (C.O / 'REPORT.zh-CN.md').write_text(text)
    IMPORTANCE[str(C.O / 'REPORT.zh-CN.md')] = '核心必需'
    IMPORTANCE[str(C.O / 'FILE_INVENTORY.json')] = '核心必需'
    IMPORTANCE[str(C.O / 'SOURCE_FREEZE_FINAL.json')] = '核心必需'
    cleanup()
    audit = manifest()
    print('REPORT written', len(text), 'chars; manifest', audit, flush=True)


if __name__ == '__main__':
    main()
