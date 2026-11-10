"""Generate REPORT.zh-CN.md from the delivered artifacts.  Cheap, no simulation.

Re-run after FIGURE_OBSERVATION.md exists so the actual visual inspection is included.
"""
import numpy as np

import common as C

DTCN = {'dt_0p5ms': '0.5 ms', 'dt_0p0625ms': '0.0625 ms'}
GATES = ['D1_no_nonfoot_ground_force', 'D2_no_free_fall', 'D3_foot_support_not_below_assist',
         'D4_foot_support_never_collapses', 'D4b_nonfoot_is_not_carrying_the_body',
         'D5_com_height_not_below_assist', 'D6_force_balance_consistent',
         'D7_tripod_redundancy', 'D8_still_above_absolute_floor']
SHORT = {g: g.split('_')[0] for g in GATES}


def f(x, n=4):
    return '—' if x is None else ('%.*f' % (n, x))


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def main():
    s = C.js(C.O / 'SUMMARY.json')
    st = C.js(C.O / 'READOUT_STATIC.json')
    ann = C.js(C.O / 'CELL_ANNOTATION.json')
    cl = C.js(C.O / 'CLOSEDLOOP_TRACKING.json')
    ver = C.js(C.O / 'VERIFICATION.json')
    bit = C.js(C.O / 'R0_BITWISE.json')
    amend = C.js(C.O / 'AMENDMENT_APPLIED.json')
    pre = C.js(C.O / 'PREAMENDMENT/READOUT_STATIC.json')
    agg = s['per_condition']
    vd = s['verdict']

    L = []
    A = L.append
    A('# S5 全群体读出（12000 个细胞）能否让同一批封存神经预热态支撑身体')
    A('')
    A('目录：`%s`　源码：`%s`' % (C.O, C.SRC))
    A('')
    A('**性质声明**：本轮是**读出接口的离线工程拟合**，不是生物标定；神经元仍然生成指令，'
      '没有任何非神经生成器产生运动。**不主张**行走、自主支撑、生物有效性或绝对力值/国际单位；'
      '**不主张**脑模型无效；**不选择**时间步长，也**不选择**模型。')
    A('')

    # ---------------------------------------------------------------- 1
    A('## 0. 一页结论（先看这里）')
    A('')
    A('1. **预注册判定 = `%s`**。3 个全群体/对照读出 × 2 个神经步长 × 5 个释放相位，'
      '共 30 次新释放（外加 10 次 R0 逐比特对照）：**没有任何一次释放全部通过 9 项 D 检查**；'
      '最小 COM 落在 %s–%s，而 D5 下界是 1.1375、D8 地板是 0.9658；足支撑 %s–%s，'
      '辅助参考是 1.000。也就是说：**换成用整个 12000 细胞群体的读出，同一批封存神经预热态'
      '仍然没能把身体撑起来**。'
      % (s['overall_verdict'], f(min(p['com_z_min'] for k, v in agg.items() if k.startswith(('P1', 'P2', 'P3'))
                                    for p in v['per_phase']), 3),
         f(max(p['com_z_min'] for k, v in agg.items() if k.startswith(('P1', 'P2', 'P3'))
                                    for p in v['per_phase']), 3),
         f(min(p['foot_support_mean'] for k, v in agg.items() if k.startswith(('P1', 'P2', 'P3'))
                                    for p in v['per_phase']), 3),
         f(max(p['foot_support_mean'] for k, v in agg.items() if k.startswith(('P1', 'P2', 'P3'))
                                    for p in v['per_phase']), 3)))
    A('2. **接口层大幅改善，但物理不跟着走**：与教师力矩的相关从 R0 的 −0.083 升到 '
      'P1 %s / P2 %s，饱和比例从 0.784 降到 %s / %s，命令 RMS 与教师 RMS 之比从 2.393 收到 '
      '%s / %s，留出 R² 达到 %s / %s。**接口变好了，承重没变好**——这是本轮最干净的一条对照。'
      % (f(st['per_timestep']['dt_0p5ms']['P1']['overall_correlation_with_teacher'], 3),
         f(st['per_timestep']['dt_0p5ms']['P2']['overall_correlation_with_teacher'], 3),
         f(st['per_timestep']['dt_0p5ms']['P1']['saturation_fraction_overall'], 5),
         f(st['per_timestep']['dt_0p5ms']['P2']['saturation_fraction_overall'], 5),
         f(st['per_timestep']['dt_0p5ms']['P1']['command_rms_over_teacher_rms'], 3),
         f(st['per_timestep']['dt_0p5ms']['P2']['command_rms_over_teacher_rms'], 3),
         f(st['fits']['P1']['r2_held']), f(st['fits']['P2']['r2_held'])))
    A('3. **唯一的「改进」只来自 D2，且只在单一步长成立**：P2 在 0.5 ms 步长的 4/5 相位把'
      'D 检查通过数从 1 提到 2，但增加的那一项始终只是 D2（不再整体自由落体）；'
      '0.0625 ms 步长只有 2/5 相位提升。按预注册规则**只能记作「单步长观察」，不构成达标**。')
    A('4. **这条读出读的主要不是运动细胞**：全群体读出的权重质量只有 3.2%（P1）/ 3.4%（P2）'
      '落在 204 个运动细胞上；`super_class` 上 sensory 0.237/0.207、VNC 中间神经元 '
      '0.587/0.611、motor 仅 0.047/0.049；`cell_function` 上 proprioception+tactile 占 '
      '0.314/0.284，leg_motor 占 0.042/0.044。主导细胞是腿上的钟形感器（campaniform）'
      '与 VNC 中间神经元。**所以这 0.87 的 R² 应读作「用编码体状态的活动重建了 PD 力矩」，'
      '不是「连接组生成了运动指令」。**')
    A('5. **闭环期内力矩一开始就接近、随后漂移**：P1/P2/P3 的命令与教师力矩在同一时刻的相关，'
      '前 20 ms 是中位 0.89–0.97，前 50 ms 0.75–0.88，100 ms 后稳定在 0.65–0.80；'
      '而 R0 的命令全程在 −0.12 到 0.04。瓶颈不在「读不出正确力矩」，'
      '在「这条闭环一旦离开教师轨迹就回不来」。')
    A('')
    A('## 1. 预注册判定')
    A('')
    A('判定规则在跑物理之前冻结（`PREREGISTRATION.json`，含 A1–A4 修正记录），未改动任何门限：')
    A('')
    A('- SUPPORT_ACHIEVED：某个读出在 **5 个相位中至少 4 个**同时满足全部 9 个 D 检查，'
      '且在 **0.5 ms 与 0.0625 ms 两个步长都**如此。')
    A('- PARTIAL：没有读出达到 SUPPORT_ACHIEVED，但某个读出的 D 检查通过数（每相位，满分 9）'
      '相对封存 R0 基线在 **≥4/5 相位**提升，且两个步长都如此。')
    A('- NO_IMPROVEMENT：其余情况，照实报。')
    A('')
    A('**总体判定：`%s`**' % s['overall_verdict'])
    A('')
    rows = []
    for r in ('P1', 'P2', 'P3'):
        rows.append([r,
                     '%d/5' % vd[r]['SUPPORT_detail']['dt_0p5ms']['count'],
                     '%d/5' % vd[r]['SUPPORT_detail']['dt_0p0625ms']['count'],
                     '%d/5' % vd[r]['D_count_detail']['dt_0p5ms']['count'],
                     '%d/5' % vd[r]['D_count_detail']['dt_0p0625ms']['count'],
                     vd[r]['statement']])
    A(table(['读出', '0.5 ms 全过 D 的相位', '0.0625 ms 全过 D 的相位',
             '0.5 ms D 计数提升相位', '0.0625 ms D 计数提升相位', '单个读出判定'], rows))
    A('')
    A('**两个步长分别判定（不得混写）**')
    A('')
    for tag in DTCN:
        aid = agg['R0__%s' % tag]['per_phase']
        rows = []
        for r in ('P1', 'P2', 'P3'):
            ph = agg['%s__%s' % (r, tag)]['per_phase']
            rows.append([r, '%d/5' % sum(1 for p in ph if p['D_passed']),
                         '%d/5' % vd[r]['D_count_detail'][tag]['count'],
                         '%d → %d' % (max(p['D_pass_count'] for p in aid),
                                      max(p['D_pass_count'] for p in ph)),
                         '是' if vd[r]['SUPPORT_detail'][tag]['met'] else '否',
                         '是' if vd[r]['D_count_detail'][tag]['met'] else '否'])
        A('- **%s 步长（%s）**' % (DTCN[tag], tag))
        A('')
        A(table(['读出', '全过 9 项 D 的相位', 'D 计数提升的相位', '最好相位 D 计数 R0 → 该读出',
                 '达到 SUPPORT 门槛', '达到 PARTIAL 门槛'], rows))
        A('')
        A('  该步长结论：%s'
          % ('**没有任何读出达到 SUPPORT，也没有任何读出在 ≥4/5 相位提升 D 计数**'
             if not any(vd[r]['D_count_detail'][tag]['met'] for r in ('P1', 'P2', 'P3'))
             else '**没有任何读出达到 SUPPORT**；D 计数在 ≥4/5 相位提升的读出：%s'
                  % ', '.join(r for r in ('P1', 'P2', 'P3')
                              if vd[r]['D_count_detail'][tag]['met'])))
        A('')
    a05 = [r for r in ('P1', 'P2', 'P3') if vd[r]['D_count_detail']['dt_0p5ms']['met']]
    a06 = [r for r in ('P1', 'P2', 'P3') if vd[r]['D_count_detail']['dt_0p0625ms']['met']]
    A('**两步长一致性**：0.5 ms 达到 D 计数门槛的读出 %s；0.0625 ms 达到的读出 %s。'
      % (a05 or '无', a06 or '无'))
    A('')
    if a05 and not a06:
        A('> **这里只能记作「单步长观察」**：%s 的 D 检查通过数在 0.5 ms 步长的 4/5 相位提升'
          '（%s），但在 0.0625 ms 步长只有 2/5 相位提升，**未通过 PARTIAL 门槛**。'
          '按预注册规则，本轮**不写成任何形式的达标**。'
          % (', '.join(a05), vd[a05[0]]['D_count_detail']['dt_0p5ms']['phases_improved']))
        A('')
    A('> 因为没有任何读出在任何相位全过 9 项 D 检查，本轮**不构成支撑达标**；'
      '也不存在「某相位达标但相位不足」这类更强结论。')
    A('')

    # which D checks actually carry the changes
    changed = {}
    for tag in DTCN:
        for r in ('P1', 'P2', 'P3'):
            base = agg['R0__%s' % tag]['per_phase']
            new = agg['%s__%s' % (r, tag)]['per_phase']
            for b, n in zip(base, new):
                for g in GATES:
                    if n['D_checks'][g] and not b['D_checks'][g]:
                        changed.setdefault(g, set()).add((r, tag))
                    if b['D_checks'][g] and not n['D_checks'][g]:
                        changed.setdefault('LOST_' + g, set()).add((r, tag))
    A('**D 计数变化到底来自哪几项检查**（相对同相位 R0）：')
    A('')
    rows = []
    for g in GATES:
        got = sorted(changed.get(g, set()))
        lost = sorted(changed.get('LOST_' + g, set()))
        rows.append([SHORT[g], len(got), ', '.join('%s@%s' % (a, DTCN[b]) for a, b in got) or '—',
                     len(lost), ', '.join('%s@%s' % (a, DTCN[b]) for a, b in lost) or '—'])
    A(table(['D 检查', '新增通过的 (读出@步长)', '明细', '反而失去的', '明细'], rows))
    A('')
    only_d2 = set(changed) - {g for g in GATES}
    A('也就是说：本轮 D 计数的所有**提升**都只来自 D2（无自由落体），'
      '而承重相关的 D1/D3/D4/D4b/D5/D8 在所有 40 次释放里**一次都没通过过**。'
      'D7（三足冗余）只在基线 R0 的 0.5 ms/1.300 s 这一个相位通过，'
      '而 P1/P2/P3 在 0.5 ms 把它丢掉了——即新读出不只没拿走承重门，连 R0 那唯一一个'
      '冗余门也丢了。D2 变好只说明身体不再整体自由落体，不说明被支撑住：'
      '同相位的足支撑 0.18–0.53、最小 COM 0.49–0.62 见第 5 节，仍远低于 D3 下界 0.900、'
      'D5 下界 1.1375 与 D8 地板 0.9658。'
      if (set(changed) & set(GATES)) == {'D2_no_free_fall'}
      else '本轮发生变化的 D 检查见上表。')
    A('')

    # ---------------------------------------------------------------- 2
    A('## 2. 实验设计（跑前冻结）')
    A('')
    A('- 读出全部**只用 0.2–0.8 s 拟合**，0.8–1.4 s 留作留出检验，拟合后冻结，再上物理。')
    A('- 特征：12000 个细胞各自的 5 ms 滤波放电率，取每个 1 ms 指令区间**起点**的值'
      '（`WARM.npz` 的 `cell_filtered_rates` 前移一格，0.2 s 处为零），'
      '与封存 66 池约定 `command_source_rates` 完全镜像。')
    A('- 目标：教师施加的关节力矩，按 1 ms 指令边界取样（0.5 ms 索引 400+2k）。')
    A('- alpha 尺度自由参数化：`alpha = gamma · s`，`s =` 拟合窗特征平方范数均值，'
      '`gamma` 用拟合窗内 5 段连续分块交叉验证选取；留出窗不参与任何选择。')
    A('- 同一批封存预热态（`WARM_CACHE.pkl`）被 4 个读出共用，**没有新的预热重放、'
      '没有新的教师录制、没有新的连接组内容**。')
    A('- 网格：4 个读出（R0 对照 / P1 / P2 / P3）× 2 个神经步长 × 5 个相位 = **40 次释放**，'
      '每次 0.2 s。R0 必须在两个步长逐比特复现封存 `numerics` 结果，否则任务在任何 '
      'P1/P2/P3 数字出现之前停止。')
    A('- 其它一切不变：身体、矩阵、连接组、5 ms 池/细胞滤波、1 ms 因果指令时钟、'
      'RH 0.375 仅释放时施加一次、42 个理想关节力矩执行器、12.5 µs 物理步长、'
      '裁剪到 ±limits、零外力/零姿态写。')
    A('')

    # ---------------------------------------------------------------- 3
    A('## 3. 接口层结果')
    A('')
    A('### 3.1 拟合与留出 R²')
    A('')
    rows = []
    for k, lab in (('P1', 'P1 全部 12000，无常数项'), ('P2', 'P2 全部 12000，逐细胞去均值+固定偏置'),
                   ('P3', 'P3 仅 204 运动细胞（同 P2 协议）'),
                   ('P3b_raw_no_constant_diagnostic', 'P3b 204 细胞、原始率、无常数项（诊断）'),
                   ('P2z_no_offset_diagnostic', 'P2z 去均值但不给偏置（诊断，退化）'),
                   ('P4_trend_removed_diagnostic', 'P4 去 100 ms 趋势（诊断，非因果）')):
        d = st['fits'][k]
        rows.append([lab, d['n_features'], f(d['gamma'], 4), '%.4g' % d['alpha'],
                     f(d['r2_fit']), '**%s**' % f(d['r2_held'])])
    A(table(['拟合', '特征数', '所选 gamma', 'alpha', '拟合窗 R²', '留出窗 R²'], rows))
    A('')
    A('- 打乱对照（时间置换目标）留出 R² = %s。' % f(st['shuffle_control']['mean'], 4))
    A('- 教师力矩 RMS：全窗 %s，拟合窗 %s，留出窗 %s；其直流分量 RMS %s，'
      '波动分量 RMS %s。' % (f(st['teacher_rms']['full'], 3), f(st['teacher_rms']['fit'], 3),
                            f(st['teacher_rms']['held'], 3), f(st['teacher_rms']['static_full'], 3),
                            f(st['teacher_rms']['fluctuation_full'], 3)))
    A('- 12000 细胞中有非零放电率的：拟合窗 %d、留出窗 %d（全窗 %d）。'
      % (st['data_check']['cells_with_nonzero_rate_fit_window'],
         st['data_check']['cells_with_nonzero_rate_held_window'],
         st['data_check']['cells_with_nonzero_rate_full_window']))
    A('')
    A('**父代理已公布数字的重现**：R0 的接口相关 %s（父 −0.0831）、饱和率 %s（父 0.784）、'
      '符号一致 %s（父 0.4653）、命令/教师 RMS 比 %s（父 2.393）；'
      '12000 特征留出 R² %s（父 0.881）；去 100 ms 趋势后 %s（父 0.841）；'
      '204 细胞 %s（父 0.27–0.41）；打乱对照 %s（父 −0.405）。'
      % (f(st['per_timestep']['dt_0p5ms']['R0']['overall_correlation_with_teacher']),
         f(st['per_timestep']['dt_0p5ms']['R0']['saturation_fraction_overall'], 5),
         f(st['per_timestep']['dt_0p5ms']['R0']['sign_agreement']),
         f(st['per_timestep']['dt_0p5ms']['R0']['command_rms_over_teacher_rms'], 3),
         f(st['fits']['P2']['r2_held']), f(st['fits']['P4_trend_removed_diagnostic']['r2_held']),
         f(st['fits']['P3b_raw_no_constant_diagnostic']['r2_held']),
         f(st['shuffle_control']['mean'], 3)))
    A('')
    A('### 3.2 每个读出的接口统计（命令 = 裁剪后实际施加值，全 1200 个 1 ms 区间）')
    A('')
    for tag in DTCN:
        A('**%s 预热缓存**（同一冻结矩阵，未按步长重拟合）' % DTCN[tag])
        A('')
        rows = []
        for name in ('R0', 'P1', 'P2', 'P3', 'P2z', 'P3b'):
            d = st['per_timestep'][tag][name]
            rows.append([name, f(d['overall_correlation_with_teacher']),
                         f(d['per_joint_correlation_median']),
                         f(d['sign_agreement']), f(d['command_rms_over_teacher_rms'], 3),
                         f(d['saturation_fraction_overall'], 5),
                         f(d['r2_fit_window']), f(d['r2_held_window'])])
        A(table(['读出', '与教师力矩相关', '逐关节中位相关', '符号一致率',
                 '命令/教师 RMS', '饱和比例', '拟合窗 R²', '留出窗 R²'], rows))
        A('')
    A('要点：R0 的命令几乎与教师力矩无关（%s）且 %s 的元素顶在限幅上；'
      'P1/P2 的命令相关 %s/%s，饱和比例 %s/%s，命令 RMS 与教师 RMS 之比 %s/%s。'
      % (f(st['per_timestep']['dt_0p5ms']['R0']['overall_correlation_with_teacher']),
         f(st['per_timestep']['dt_0p5ms']['R0']['saturation_fraction_overall'], 4),
         f(st['per_timestep']['dt_0p5ms']['P1']['overall_correlation_with_teacher']),
         f(st['per_timestep']['dt_0p5ms']['P2']['overall_correlation_with_teacher']),
         f(st['per_timestep']['dt_0p5ms']['P1']['saturation_fraction_overall'], 5),
         f(st['per_timestep']['dt_0p5ms']['P2']['saturation_fraction_overall'], 5),
         f(st['per_timestep']['dt_0p5ms']['P1']['command_rms_over_teacher_rms'], 3),
         f(st['per_timestep']['dt_0p5ms']['P2']['command_rms_over_teacher_rms'], 3)))
    A('')
    A('### 3.3 P1 与 P2 的对照（慢漂移是否主导）')
    A('')
    A('P1 无常数项：力矩的直流分量也必须由**放电率水平本身**产生；'
      'P2 把每个细胞在拟合窗的均值显式减掉，并把逐关节拟合窗平均力矩作为**唯一固定偏置**加回，'
      '所以 P2 只从**波动**里取信息。')
    A('')
    A('- P1 留出 R² = %s，P2 留出 R² = %s：**去均值没有让它变差**。'
      '也就是说「慢漂移/绝对率水平」并不是这条读出之所以能重建教师力矩的原因，'
      '快速波动本身就带了几乎同样多的信息。'
      % (f(st['fits']['P1']['r2_held']), f(st['fits']['P2']['r2_held'])))
    A('- 反证：P2z（去均值但不给偏置）留出 R² = %s。减掉逐细胞均值后如果不同时提供常值偏置，'
      '这段读出在数学上**无法**表达平均力矩（预测均值≈0，目标均值≈2.43），'
      '所以它必然是退化的——这正是必须提供偏置的原因，也说明 P2 的偏置是接口的一部分，'
      '而不是从神经活动里读出来的东西。' % f(st['fits']['P2z_no_offset_diagnostic']['r2_held']))
    A('')
    A('### 3.4 命令的可复现性（用已保存数组独立重算）')
    A('')
    A('- 12000 细胞的 5 ms 滤波：从封存 `WARM.npz` 的打包脉冲逐比特重放，'
      '与封存 `cell_filtered_rates` **逐比特相等**（两 个步长都通过，见 '
      '`CELV_FILTER_CHECK_*.json` / `SUMMARY.json:cell_filter_check`）。'.replace('CELV', 'CELL'))
    for c in ver['checks']:
        if c['check'] == 'applied_command_reproduced_from_saved_spikes':
            A('- 40 次释放的施加命令：从各自 `NEURAL.npz` 保存的 12000 位脉冲字重新解包、按同一步长'
              '重放 5 ms 滤波、套用保存的 `W`/`b`、裁剪，再与记录的 `applied` 比较：'
              '**%d 次释放全部可复现**，最大绝对差 %.3g（0 表示逐比特相同，不是仅仅在容差内），'
              '越界元素 %d 个。'
              % (c['detail']['releases'], c['detail']['worst']['max_abs'],
                 c['detail']['elements_above_limit']))
    A('- 也就是说：**每个读出的命令都可以只凭已保存的数组独立重算出来**，'
      '不依赖运行时的任何隐藏状态。')
    A('')

    # ---------------------------------------------------------------- 4
    A('## 4. 预注册修正记录（跑物理之前，A1–A4）')
    A('')
    A('离线拟合阶段发现两处设计缺陷，均在**任何物理释放之前**修好并记账'
      '（`PREREGISTRATION.json:amendments`、`CHANGE_LEDGER.json:tool_fixes`、'
      '`AMENDMENT_APPLIED.json`），修正前的产物原样保留在 `PREAMENDMENT/`。')
    A('')
    rows = []
    for a in C.js(C.O / 'PREREGISTRATION.json')['amendments']:
        rows.append([a['id'], a['target'], a['new']])
    A(table(['编号', '对象', '修正后定义'], rows))
    A('')
    A('**A1（尺度错误）**：父代理原分析给出的 `alpha=100`，在本轮特征尺度下对应 '
      '`gamma ≈ 3e-7`，即**近乎无正则**。本轮冻结的原始 alpha 网格上界 1e5 同样远低于'
      '核矩阵自身尺度 `s = %s`，导致修正前所有交叉验证曲线单调指向网格边界、'
      '`alpha` 其实没有被真正选择。改为尺度自由参数化后重拟合，'
      '父代理公布的 0.881 由 P2 以 %s 复现（差异来自正则量级而非结论）。'
      '**这一点应在下游引用中更正：0.881 是「近乎无正则的 12000 特征岭回归」，不是调过参的结果。**'
      % ('%.4g' % st['fits']['P1']['scale'], f(st['fits']['P2']['r2_held'])))
    A('')
    A('**A2（P2 退化）**：原式把偏置绑在同一组收缩权重上（`b2 = -W2 @ m_fit`）且目标未中心化，'
      '数学上无法表达平均力矩；修正前拟合窗交叉验证在全部 alpha 上为 %s，'
      '留出 R² = %s（这个数字在看到后才改设计，按实记录，未隐瞒）。'
      '改为标准「带截距岭回归」形式后留出 R² = %s。退化的原式保留为 P2z，只做接口、不释放。'
      % (f(pre['fits']['P2']['cv_best'], 3), f(pre['fits']['P2']['r2_held']),
         f(st['fits']['P2']['r2_held'])))
    A('')
    A('修正只依据**拟合窗内**的交叉验证曲线与数学推导；留出窗从未参与任何参数选择；'
      '判定规则、相位表、步长表、释放网格一个都没动。'
      '（修正前后：`PREREGISTRATION.json` %s → %s。）'
      % (amend['prereg_old_sha256'][:12], amend['prereg_new_sha256'][:12]))
    A('')

    # ---------------------------------------------------------------- 5
    A('## 5. 物理层结果')
    A('')
    A('### 5.1 R0 对照的逐比特复现')
    A('')
    A('- 10 次 R0 释放（2 步长 × 5 相位）全部与封存 `numerics/RUNS` **逐比特一致**：'
      'NATIVE 行、BODY/NEURAL 数组、RESULT 的 J 与 D 检查全部相等（%d/10 通过）。'
      '因此本节所有 P1/P2/P3 与 R0 的成对比较都是**同一批封存预热态上的配对比较**，'
      '不是另起一套基线。' % sum(1 for v in bit.values() if v['passed']))
    A('')
    for tag in DTCN:
        A('### 5.2 %s 步长（%s）' % (DTCN[tag], tag))
        A('')
        base = agg['R0__%s' % tag]
        A('封存基线 R0 逐相位：')
        A('')
        rows = [[f(p['phase'], 3), f(p['J']), f(p['foot_support_mean']),
                 f(p['com_z_min']), f(p['first_positive_nonfoot_s'], 4),
                 p['D_pass_count']] for p in base['per_phase']]
        A(table(['相位', 'J', '足支撑均值', '最小 COM', '首次非足接触 (s)', 'D 通过数/9'], rows))
        A('')
        for r in ('P1', 'P2', 'P3'):
            a = agg['%s__%s' % (r, tag)]
            A('**%s @ %s**（J 散布 %s；足支撑散布 %s；最小 COM 散布 %s；'
              '全过 D 的相位 %d/5）'
              % (r, DTCN[tag], f(a['J_spread']['spread']), f(a['foot_support_mean_spread']['spread']),
                 f(a['com_z_min_spread']['spread']), a['phases_passing_all_D']))
            A('')
            rows = []
            for p in a['per_phase']:
                rows.append([f(p['phase'], 3), f(p['J']), f(p['foot_support_mean']),
                             f(p['com_z_min']), f(p['first_positive_nonfoot_s'], 4),
                             p['D_pass_count'],
                             ''.join(SHORT[g] if p['D_checks'][g] else '·' for g in GATES)])
            A(table(['相位', 'J', '足支撑均值', '最小 COM', '首次非足接触 (s)', 'D 通过数/9',
                     'D1 D2 D3 D4 D4b D5 D6 D7 D8（· = 未过）'], rows))
            A('')
        A('逐相位 D 检查通过数对照（R0 → P1 / P2 / P3）：')
        A('')
        rows = []
        for i, p in enumerate(base['per_phase']):
            rows.append([f(p['phase'], 3), p['D_pass_count']]
                        + [agg['%s__%s' % (r, tag)]['per_phase'][i]['D_pass_count']
                           for r in ('P1', 'P2', 'P3')])
        A(table(['相位', 'R0', 'P1', 'P2', 'P3'], rows))
        A('')
        for r in ('P1', 'P2', 'P3'):
            d = vd[r]['D_count_detail'][tag]
            A('- %s 在 %s 的 D 计数提升相位：%s（%d/5，门槛 4/5 → %s）'
              % (r, DTCN[tag], d['phases_improved'] or '无', d['count'],
                 '达到' if d['met'] else '未达到'))
        A('')

    # ---------------------------------------------------------------- 6
    A('## 6. 闭环释放期命令与教师力矩的关系随时间如何变化')
    A('')
    A('定义：对每个 1 ms 指令区间 k，取闭环**实际施加**的命令，与教师在其自己轨迹上'
      '**同一时刻**施加的力矩（`BODY.npz` ctrl，0.5 ms 索引 `round(ph*2000)+2k`）比较，'
      '把 42 个关节 × 窗口内全部区间摊平后算 Pearson 相关。教师力矩是教师状态的 PD 函数，'
      '所以这条相关性的下降反映的是**释放轨迹偏离教师轨迹**，不是读出本身的缺陷。'
      '注意它不是「同一状态下的力矩」比较：一旦轨迹分开，比的就是两个不同状态。')
    A('')
    A('截断说明：教师录制到 1.4 s 为止，晚相位（1.325 / 1.375 s）的 100 ms 与 200 ms 窗口'
      '会被截到实际重叠长度（1.375 s 相位只有 26 个 1 ms 区间有教师样本），'
      '窗口内实际用到的区间数记在 `CLOSEDLOOP_TRACKING.json:per_release.available_ms`，'
      '没有外推。')
    A('')
    for tag in DTCN:
        A('**%s 步长**（括号内为 5 个相位各自的值）' % DTCN[tag])
        A('')
        rows = []
        for r in ('R0', 'P1', 'P2', 'P3'):
            d = cl['per_condition']['%s__%s' % (r, tag)]
            rows.append([r] + ['%s (%s)' % (f(d['%dms' % w]['median'], 3),
                                            ', '.join(f(x, 2) for x in
                                                      (d['%dms' % w]['per_phase'] or [])))
                               for w in (20, 50, 100, 200)])
        A(table(['读出', '前 20 ms', '前 50 ms', '前 100 ms', '前 200 ms'], rows))
        A('')
    A('读法与结论：若相关性在最初 20 ms 已经很高、随后单调下降，是「一开始对、随后因状态漂移'
      '而失效」；若从一开始就低，是「一开始就不对」。**本轮是前者**：P1/P2/P3 在最初 20 ms 的'
      '中位相关是 0.89–0.97（五个相位最低也有 0.76），50 ms 时 0.75–0.88，100 ms 后稳定在'
      '约 0.65–0.80，不再继续掉；同一套释放里 R0 的命令相关始终在 −0.12 到 0.04 之间。'
      '配合 `lag5ms_control`（把命令整体后移 5 ms 再比，相关性没有系统性升高）看，'
      '这不是对齐错位问题：**读出在释放一开始就把教师力矩重建得相当准，随后闭环把状态带离'
      '教师轨迹，力矩也就跟着偏离**。换句话说，瓶颈不在「这条读出读不出正确力矩」，'
      '而在「这条闭环一旦离开教师轨迹就回不来」。')
    A('')
    A('')

    # ---------------------------------------------------------------- 7
    A('## 7. 这条读出到底在读什么（关键解释性证据）')
    A('')
    A('把 12000 个细胞按 **banc_888_id** 联接到 `%s`：匹配 %d/12000，未匹配 %d；'
      '若改用 root_id 联接只能匹配 %d/12000（少 %d 个），所以本报告坚持用 banc_888_id。'
      % (ann['source_meta'], ann['join']['matched'], ann['join']['unmatched'],
         ann['join']['matched_if_key_were_root_id'],
         12000 - ann['join']['matched_if_key_were_root_id']))
    A('')
    A('### 7.1 权重质量集中在少数细胞')
    A('')
    rows = []
    for r in ('P1', 'P2'):
        d = ann['per_readout'][r]
        rows.append([r, f(d['top50_shared_energy_share'], 3), f(d['top200_shared_energy_share'], 3),
                     d['cells_carrying_50pct'], d['cells_carrying_90pct']])
    A(table(['读出', 'top-50 占权重质量', 'top-200 占权重质量', '占一半质量所需细胞数',
             '占九成质量所需细胞数'], rows))
    A('')
    A('### 7.2 逐字段的权重质量类别占比（全部 12000 细胞）')
    A('')
    for r in ('P1', 'P2'):
        A('**%s**' % r)
        A('')
        for fld in ('super_class', 'cell_class', 'nerve', 'body_part_sensory',
                    'body_part_effector', 'cell_function'):
            rows = [[d['value'], f(d['energy_share'], 3), d['count'], f(d['count_share'], 3)]
                    for d in ann['per_readout'][r]['profile'][fld]]
            A(table(['%s' % fld, '权重质量占比', '细胞数', '细胞数占比'], rows))
            A('')
    A('### 7.3 主导细胞本身的注释')
    A('')
    for r in ('P1', 'P2'):
        d = ann['per_readout'][r]
        A('- **%s**：top-50 中 super_class = motor 的细胞 %d 个；'
          'super_class 以 sensory 开头的细胞 %d 个；按权重质量分类 %s'
          '（motor %s、sensory %s、intrinsic %s、上行/下行 %s）。'
          % (r, d['top50_cells_that_are_motor_class'],
             d['top50_cells_that_are_sensory_super_classes'],
             d['super_class_energy']['classified_as'],
             f(d['super_class_energy']['motor'], 3), f(d['super_class_energy']['sensory'], 3),
             f(d['super_class_energy']['intrinsic'], 3),
             f(d['super_class_energy']['ascending_descending'], 3)))
    A('')
    rows = [[t['rank'], t['banc_888_id'],
             (t['annotation'] or {}).get('cell_type', 'NA'),
             (t['annotation'] or {}).get('super_class', 'NA'),
             (t['annotation'] or {}).get('cell_class', 'NA'),
             (t['annotation'] or {}).get('nerve', 'NA'),
             (t['annotation'] or {}).get('body_part_sensory', 'NA'),
             (t['annotation'] or {}).get('body_part_effector', 'NA'),
             '是' if t['is_204_motor_cell'] else '否']
            for t in ann['per_readout']['P2']['top50']]
    A('**P2 的 top-50 细胞（按行范数）**')
    A('')
    A(table(['#', 'banc_888_id', 'cell_type', 'super_class', 'cell_class', 'nerve',
             'body_part_sensory', 'body_part_effector', '属于 204 运动细胞'], rows))
    A('')
    A('### 7.4 结论：这条读出的信息来自哪一类细胞')
    A('')
    for r in ('P1', 'P2'):
        d = ann['per_readout'][r]
        fn = {e['value']: e['energy_share'] for e in d['profile']['cell_function']}
        A('- **%s**：`cell_function` 上 proprioception %s + tactile %s = %s 的权重质量落在'
          '本体感受/触觉类，`leg_motor` 只有 %s；`super_class` 上 sensory %s、'
          'ventral_nerve_cord_intrinsic %s，motor 只有 %s；204 个运动细胞合计只承接 %s。'
          % (r, f(fn.get('proprioception', 0.0), 3), f(fn.get('tactile', 0.0), 3),
             f(fn.get('proprioception', 0.0) + fn.get('tactile', 0.0), 3),
             f(fn.get('leg_motor', 0.0), 3),
             f(d['super_class_energy']['sensory'], 3),
             f({e['value']: e['energy_share']
                for e in d['profile']['super_class']}.get('ventral_nerve_cord_intrinsic', 0.0), 3),
             f(d['super_class_energy']['motor'], 3),
             f(ann['motor_cell_weight_share']['%s_energy_share_on_the_204_motor_cells' % r], 3)))
    A('')
    A('**说明白**：主导这条全群体读出的不是运动神经元，而是**编码身体状态的'
      '感觉/本体感受类细胞（腿上的钟形感器 campaniform、弦音器 chordotonal、'
      '以及大量 VNC 中间神经元）**。P1 的 top-1 到 top-8 全是 `SNpp53`'
      '（campaniform_sensillum_neuron，后/中足），P2 的 top-1/2/3/5/7/8/10 是 `IN19A060` 等'
      'VNC 中间神经元。因此这条读出能在留出窗拿到 %s 的 R²，最自然的解释是：'
      '**它用编码体状态的活动把教师的 PD 力矩重建了出来**，而不是连接组自己生成了运动指令。'
      '这正是第 8 节要划的因果边界。'
      % f(st['fits']['P2']['r2_held'], 3))
    A('')
    A('**204 个运动细胞占整个读出权重质量的比例**：P1 %s、P2 %s。'
      '换言之当前 204 细胞读出只看得到的那部分细胞，只承接了全群体读出权重质量的一小部分，'
      '其余权重落在当前读出从未使用的细胞上。'
      % (f(ann['motor_cell_weight_share']['P1_energy_share_on_the_204_motor_cells'], 3),
         f(ann['motor_cell_weight_share']['P2_energy_share_on_the_204_motor_cells'], 3)))
    A('')

    # ---------------------------------------------------------------- 8
    A('## 8. 因果边界的判定标准（明白写出来）')
    A('')
    A('教师力矩是**身体状态的 PD 函数**。因此：')
    A('')
    A('1. 若某个全群体读出主导权重落在**编码体状态的感觉/本体感受类细胞**上，'
      '那么它能在离线拟合里复现教师力矩，只是「用编码体状态的活动重建了 PD 力矩」；'
      '这类拟合**不构成**「连接组生成了运动指令」的证据。')
    A('2. 若 P1/P2 **撑起了**身体（本轮的预注册判定允许的强结论级别），'
      '结论上限仍然是：在闭环中，同一批封存预热态经由一个**离线监督拟合的线性读出**'
      '把体状态信息转成了足以承重的关节力矩。这**不计为**连接组自主生成行走指令。')
    A('3. 若 P1/P2 **没撑起**身体，则说明即使拿到了留出 R² ≈ %s 的可线性重建信息，'
      '在这条闭环里依然不够。' % f(st['fits']['P2']['r2_held'], 2))
    A('4. 无论上述哪种情况，本轮都**不**主张行走、自主支撑、生物有效性、绝对力值单位，'
      '也不主张脑模型无效；**不选择**步长与模型。')
    A('')
    A('**本轮落在第 3 种情形**：P1/P2 都没有撑起身体（0 个相位全过 D，最小 COM 0.49–0.62，'
      '足支撑 0.18–0.53，远低于 D5 下界 1.1375 与 D8 地板 0.9658）。'
      '因此本轮结论是：**即使拿到了留出 R² ≈ %s 的可线性重建信息，在这条神经—肌肉—物理闭环里'
      '依然不够**。' % f(st['fits']['P2']['r2_held'], 3))
    A('')
    A('并且第 7.4 节的实测结果让这条边界更硬：主导权重落在感觉/本体感受类与 VNC 中间神经元上，'
      '运动细胞的权重质量只有 0.047（P1）/ 0.049（P2），204 个运动细胞合计只有 0.032/0.034；'
      '所以这 %s 的 R² 本身就应该读成「**用编码体状态的活动重建了 PD 力矩**」，'
      '而不是「连接组生成了运动指令」。' % f(st['fits']['P2']['r2_held'], 3))
    A('')
    A('（本报告不使用「完成度百分比」这类说法；上面出现的比例都是权重质量占比这类可直接复算的'
      '测量量。绝对力值与 SI 单位仍然 NOT_ESTABLISHED。）')
    A('')

    # ---------------------------------------------------------------- 9
    A('## 9. 独立验证')
    A('')
    A('`verify.py` 用本地 numpy 重新实现岭回归重拟合与交叉验证选 alpha、5 ms 细胞滤波、'
      '支撑序列、目标函数、D 门与预注册判定，不调用 `run.py`/`static.py`：'
      '**%d/%d 项通过，verified = %s**。' % (ver['passed'], ver['total'], ver['verified']))
    A('')
    if not ver['verified']:
        A('未通过项：')
        A('')
        for c in ver['checks']:
            if not c['passed']:
                A('- `%s`：%s' % (c['check'], c['detail']))
        A('')
    A('其中独立的留出 R² 重算：P1 %s、P2 %s、P3 %s（与 `READOUT_STATIC.json` 一致）。'
      % (f(ver['independent_refit']['P1']['held']), f(ver['independent_refit']['P2']['held']),
         f(ver['independent_refit']['P3']['held'])))
    A('')

    # ---------------------------------------------------------------- 10
    A('## 10. 图（`FIGURE_pop_readout.png`）')
    A('')
    fo = C.O / 'FIGURE_OBSERVATION.md'
    if fo.exists():
        A(fo.read_text().strip())
    else:
        A('（见 `FIGURE_OBSERVATION.md`）')
    A('')

    # ---------------------------------------------------------------- 11
    A('## 11. 未建立 / 未执行')
    A('')
    A('- 绝对力值与物理单位：**NOT_ESTABLISHED**。')
    A('- 行走、自主支撑、生物有效性：**NOT_CLAIMED / NOT_ESTABLISHED**。')
    A('- 未新增预热重放、未重录教师、未改连接组、未改 `engine/`、未改 `GOAL_TRACKER.zh-CN.md`、'
      '未写任何封存目录。')
    A('- 未给任何结果追加相位/种子/重复（按指令）。')
    A('- 只有一次重型计算任务（串行 40 次释放），跑前用 pgrep 确认无其它重型 python 任务。')
    A('')
    A('## 11b. 失败与修正（照实列出）')
    A('')
    A('- `run.py` 第一次启动即在**预检**阶段崩溃（退出码 1，`FAILURE_1793954876274728711.json`）：'
      '细胞滤波自检把按**神经步**排的数组（2400/19200 行）和封存的按 **1 ms 区间**排的轨迹'
      '（1200 行）直接相减，触发 `ValueError`。此时 `RUNS/` 尚未创建，**没有任何释放被生产出来**。')
    A('- `run.py` 第二次启动被**冻结断言**拦下（退出码 1，'
      '`FAILURE_1793954955887084506.json`）：冻结断言要求源码逐比特不变，与刚才的修正冲突。'
      '改为「偏差必须由 `CHANGE_LEDGER.tool_fixes` 桥接、仍拒绝任何未声明改动」的形式，'
      '并把使用的桥接记录写进 `DECLARED_SOURCE_FIXES_USED.json`。')
    A('- `closedloop_corr.py` 第一次运行 `IndexError`：教师录制止于 1.4 s，晚相位的窗口下标越界；'
      '改为按「有教师样本的 1 ms 区间数」截断后重跑。')
    A('- `static.py` 第一次运行 `ZeroDivisionError`（窗口边界写错），修好后才有修正前的拟合结果。')
    A('- 全部四次修改都在**没有任何物理结果之时**完成，逐条记录在 `CHANGE_LEDGER.json:tool_fixes`'
      '（旧 sha256 / 新 sha256 / 理由）；`verify.py` 复核后 73/73 通过，未声明改动为空。')
    A('')
    A('## 12. 交付文件与重要程度')
    A('')
    A('见 `FILE_INVENTORY.json`（含绝对路径与每个文件的重要程度：核心必需 / 重要辅助 / '
      '可再生临时 / 待确认）与 `MANIFEST_SHA256.json`（自审计）。')
    A('')
    (C.O / 'REPORT.zh-CN.md').write_text('\n'.join(L) + '\n')
    print('REPORT', C.O / 'REPORT.zh-CN.md', len('\n'.join(L)), flush=True)


if __name__ == '__main__':
    main()
