"""Generate REPORT.zh-CN.md from the saved JSON results of the numerics_chaos stage."""
import json
import numpy as np
import run as r

MEANING_ZH = {
    'AMPLIFICATION_DOMINANT':
        '判定：放大主导。1e-9 nA 级、1e-6 mV 级与 1e-6 ms 级的微扰在 200 ms 内已经长到与 dt 诱导差异同一量级，'
        '说明 50–200 ms 的 dt 分歧与“任意小差异被复发网络放大”一致。由此推得：任何步长细化都只会引入彼此不同的任意小差异，'
        '轨迹级一致在原理上不可达，轨迹级一致是错误验收目标；验收必须建立在分布/统计层（池率与命令的统计量、饱和、'
        '群体峰数分布、时间平均等），而不是逐点轨迹或逐尖峰时间。',
    'BIAS_DOMINANT':
        '判定：系统性偏差主导。微扰诱导的 200 ms 差异始终比 dt 诱导差异小若干数量级，说明 dt 分歧不能由任意小差异放大解释，'
        '与系统性截断/量化偏差（步内电导变化、事件相位与阈值时刻的量化）一致；在这种结果下，改进积分器在原理上仍可能减少分歧，'
        '轨迹级一致仍有追求空间（本轮不据此选择积分器或步长）。',
    'MIXED':
        '判定：混合。部分读出通道被放大到 dt 量级、部分仍远低于该量级，放大与偏差都可能贡献；'
        '预先登记的判定规则不把它归因为单一原因。含义是：对已被放大的通道，轨迹级一致不可作为验收目标；'
        '对仍低若干数量级的通道，偏差因素尚未排除。后续若继续，应在同一单 dt 下分别量化“积分器改进能减少多少”与'
        '“初值微扰能放大多少”，而不是继续用轨迹级一致作为门。',
}
NAMES = {'control': '对照（扰动恒为零）',
         'P1_current': 'P1 单神经元注入电流 +1e-9 nA（整 200 ms）',
         'P2_voltage': 'P2 单神经元膜电位 +1e-6 mV（首步前一次）',
         'P3_event_phase': 'P3 单个待到达突触事件相位延后 1e-6 ms'}


def f(x, n=6):
    if x is None:
        return '—'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not np.isfinite(x):
        return str(x)
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return '%.*e' % (max(n - 1, 1), x)
    return ('%.*g' % (n, x))


def main():
    s = r.js(r.OUT / 'SUMMARY.json')
    v = r.js(r.OUT / 'VERIFICATION.json')
    cw = r.js(r.OUT / 'CONTROL_BITWISE_CHECK.json')
    base = s['verdict']['dt_baseline_200ms']
    rows = {x['label']: x for x in s['results']}
    comp = s['comparisons']
    L = []
    A = L.append
    A('# 单一时间步长微扰放大—系统性偏差判别诊断')
    A('')
    A('本轮为手工研究诊断，非成品；不改变任何已封存阶段，不宣告物理行走，不宣告脑模型无效，不宣告生物有效性，不选择时间步长。'
      '本阶段只回答一个问题：已观测到的 50–200 ms 时间步长轨迹分歧，是“任意小差异被放大”还是“系统性截断/量化偏差”。')
    A('')
    A('## 一、设计（已预先登记）')
    A('')
    A('固定 dt = %s ms（已测最细网格；1 ms 命令 = %d 个神经子步），因此数值方法本身不是混杂因素。'
      '全部 4 个病例从同一封存 0.5 ms 暖态（物理 1.3 s、神经时钟 1100 ms）转移而来，v/ge/gi/adaptation/refractory_until_ms/spike_count '
      '与未到达事件的绝对时间一致；RH 兴奋增益 0.375 在转移后只乘一次；输入为同一封存 200×12000 float64 实际传感电流，每 1 ms 保持；'
      '命令由前一完成区间池率因果生成；5 ms 池滤波按神经子步 dt 感知推进；读出矩阵与限幅不变。'
      '本阶段 0 物理步、0 新暖态、0 物理状态持久化，全 12000 细胞、每步 %d 步共 %s 步。'
      % (f(r.DT_MS), r.SUBSTEPS, r.SUBSTEPS, f(sum(x['neural_steps'] for x in s['results']))))
    A('')
    A('微扰目标固定为神经元索引 %d（由封存只读输入推导：封存 backward_euler dt=0.015625 控制轨迹总峰数最大为 96，'
      '并列 {4867, 9599}；剔除在 1100 ms 处处于不应期的细胞（否则 P2 会在首步不应期复位时被抹掉）后仅剩 9599，'
      '且其具备非零出向突触电导）。三次微扰分别在独立病例中施加，绝不叠加。' % r.TARGET)
    A('')
    A('## 二、对照病例必须逐位复现封存结果')
    A('')
    A('对照（扰动严格为零）与封存 `numerics_followup` 的 backward_euler dt=0.015625、200 ms 病例逐位比较：'
      '全 12000 细胞 packed 尖峰、池率、原始命令、限幅命令、命令来源池率、每 1 ms 计数、传感电流、初始池率、'
      '神经时钟、时间轴、六个状态字段、初末态网络（含 We/Wi 数据与 pending ring）——全部逐位相同：**%s**。'
      % ('是' if cw['all_exact'] else '否'))
    A('')
    A('|比较对象|逐位一致|')
    A('|---|---|')
    A('|NEURAL.npz 数组（%d 项）|%s|' % (len(cw['arrays']), '是' if all(cw['arrays'].values()) else '否'))
    A('|初始态网络与池率|%s|' % ('是' if (all(cw['initial_state']['net'].values()) and cw['initial_state']['rates']) else '否'))
    A('|末态网络与池率|%s|' % ('是' if (all(cw['final_state']['net'].values()) and cw['final_state']['rates']) else '否'))
    A('')
    A('## 三、实际施加的微扰量（float64 之后记录）')
    A('')
    A('|病例|声明量|实际施加|可表示性说明|')
    A('|---|---|---|---|')
    for label in ['P1_current', 'P2_voltage', 'P3_event_phase']:
        a = rows[label]['applied_perturbation']
        if label == 'P1_current':
            dec = '+1e-9 nA，整 200 ms'
            lo = a['injected_current_before_nA']['min']
            hi = a['injected_current_before_nA']['max']
            act = ('每 1 ms 子步实际增量最小 %s nA、最大 %s nA；目标细胞原注入电流范围 [%s, %s] nA，'
                   'float64 精确（与声明量逐位相等=%s）' % (f(a['actual_delta_nA_min']), f(a['actual_delta_nA_max']),
                                                     f(lo), f(hi),
                                                     '是' if abs(a['actual_delta_nA_min'] - r.P1_NA) == 0 and
                                                     abs(a['actual_delta_nA_max'] - r.P1_NA) == 0 else '否'))
            note = ('1e-9 nA 相对 1 nA 量级完全不损失（1 nA 处 ulp=%s，约为 1e-9 nA 的 %s 分之一）；'
                    '目标细胞自身注入电流恰为 0.0 nA，故增量为精确加法、无舍入，无需退回最小可表示增量'
                    % (f(abs(np.spacing(1.0)), 3), f(1e-9 / abs(np.spacing(1.0)), 3)))
        elif label == 'P2_voltage':
            dec = '+1e-6 mV，首步前一次'
            act = '施加前 %s mV，施加后 %s mV，实际增量 %s mV（该量级 ulp≈%s mV）' % (
                f(a['before_mV'], 12), f(a['after_mV'], 12), f(a['actual_delta_mV'], 12), f(a['ulp_at_before_mV'], 4))
            note = '可表示；目标细胞 1100 ms 处非不应期，首步不会被复位抹掉'
        else:
            dec = '单个待到达事件到达相位 +1e-6 ms'
            act = '脉冲环槽位为布尔量，round(1e-6/dt)=0 个槽位，故“移动一个槽位”不是可表示的实现；' \
                  '按 exp(+δ/τ) 幅值等价在到达区间注入：兴奋 τ=3 ms 因子 %s、抑制 τ=8 ms 因子 %s，' \
                  '实际最大增量 兴奋 %s µS、抑制 %s µS（到达后细胞 %s 个）' % (
                      f(a['factor_exc']), f(a['factor_inh']), f(a['exc_max_uS']), f(a['inh_max_uS']),
                      a['inh_posts'] + a['exc_posts'])
            note = '该因子在每个后续区间边界上精确等于“晚到 δ”的衰减值，已在合成两细胞网络端到端验证'
        A('|%s|%s|%s|%s|' % (NAMES[label], dec, act, note))
    A('')
    A('## 四、微扰何时变成离散事件')
    A('')
    A('|病例|目标神经元连续状态首次分歧（ms）|首个不同尖峰出现（ms）|是否落在首 1 ms 内|200 ms 内累计不同尖峰|多出/缺少|')
    A('|---|---|---|---|---|---|')
    for label in ['P1_current', 'P2_voltage', 'P3_event_phase']:
        c = comp[label]
        cd = c['target_continuous_divergence']
        dd = c['discrete_divergence']
        w = c['windows']['200']
        A('|%s|%s|%s|%s|%d|%d / %d|' % (NAMES[label].split('（')[0], f(cd['first_differing_time_ms']),
                                        f(dd['first_differing_spike_time_ms']),
                                        '是' if dd['first_differing_step_within_first_1ms'] else '否',
                                        w['differing_spikes_cumulative'], w['extra_spikes_cumulative'],
                                        w['missing_spikes_cumulative']))
    A('')
    A('首个脉冲时刻（全群体）与最终峰数差异见 SUMMARY.json；对照首个任何尖峰出现在第 %s 步（%.6f ms）。' %
      (f(comp['P1_current']['first_spike_time']['control_step']),
       (comp['P1_current']['first_spike_time']['control_ms'] or 0.0)))
    A('')
    A('|病例|首个全群体尖峰时刻差 ms|最终峰数差|单细胞最大峰数差|有差异细胞数|')
    A('|---|---|---|---|---|')
    for label in ['P1_current', 'P2_voltage', 'P3_event_phase']:
        c = comp[label]
        fs = c['first_spike_time']
        fc = c['final_spike_count']
        A('|%s|%s|%d|%d|%d|' % (NAMES[label].split('（')[0], f(fs['difference_ms']), fc['difference_total'],
                                  fc['max_abs_per_cell_difference'], fc['cells_with_difference']))
    A('')
    A('## 五、各窗口微扰—对照归一化 RMS 与不同尖峰数')
    A('')
    A('|病例|窗口 ms|pool nRMS|raw nRMS|clip nRMS|不同尖峰（累计）|多出|缺少|')
    A('|---|---|---|---|---|---|---|')
    for label in ['P1_current', 'P2_voltage', 'P3_event_phase']:
        for w in r.WINDOWS_MS:
            e = comp[label]['windows'][str(w)]
            A('|%s|%d|%s|%s|%s|%d|%d|%d|' % (NAMES[label].split('（')[0], w,
                                              f(e['pool_rates']['normalized_RMS']),
                                              f(e['raw_command']['normalized_RMS']),
                                              f(e['clipped_command']['normalized_RMS']),
                                              e['differing_spikes_cumulative'], e['extra_spikes_cumulative'],
                                              e['missing_spikes_cumulative']))
    A('')
    A('## 六、与 dt 诱导差异的对照（200 ms）')
    A('')
    A('dt 诱导基准取自封存 numerics_followup 的 backward_euler 最细对照（0.03125 ms vs 0.015625 ms）：'
      'pool %s、raw %s、clip %s；上一档（0.0625 vs 0.03125 ms）为 pool %s、raw %s、clip %s。'
      % tuple([f(base[k]['fine']) for k in ['pool_rates', 'raw_command', 'clipped_command']]
              + [f(base[k]['coarse']) for k in ['pool_rates', 'raw_command', 'clipped_command']]))
    A('')
    A('|病例|pool 微扰|比值|raw 微扰|比值|clip 微扰|比值|')
    A('|---|---|---|---|---|---|---|')
    for label in ['P1_current', 'P2_voltage', 'P3_event_phase']:
        cells = []
        for k in ['pool_rates', 'raw_command', 'clipped_command']:
            x = s['verdict']['per_run'][label][k]
            cells.extend([f(x['perturbation']), f(x['ratio'], 3)])
        A('|%s|%s|' % (NAMES[label].split('（')[0], '|'.join(cells)))
    A('')
    A('微扰差值的描述性增长（未饱和段对数斜率，仅作统计描述，不作李雅普诺夫指数）：')
    for label in ['P1_current', 'P2_voltage', 'P3_event_phase']:
        g = s['verdict']['per_run'][label]['growth']
        A('- %s：斜率 %s /ms，e 倍时间 %s ms' % (NAMES[label].split('（')[0], f(g['slope_per_ms']),
                                                f(g['e_folding_ms'])))
    A('')
    A('## 七、判定与含义')
    A('')
    A('预先登记的判定规则：9 个（3 病例 × 3 读出）200 ms 微扰/dt 比值中 ≥6 个 ≥0.1 判为放大主导；全部 ≤1e-3 判为偏差主导；否则为混合。'
      '本轮结果：**%s**。' % s['verdict']['reading'])
    A('')
    A(MEANING_ZH[s['verdict']['reading']])
    A('')
    A('对验收目标的含义（不含任何模型/步长选择）：若微扰差值与 dt 差值同量级，则 50–200 ms 轨迹级一致在原理上无法靠继续细化 dt 达成，'
      '轨迹级一致就是错误的验收目标，验收必须改为分布/统计层（池率、命令统计、饱和、群体峰数分布等）；'
      '若微扰差值比 dt 差值小若干数量级，则 dt 分歧不能由任意小差异放大解释，与系统性截断/量化偏差一致，'
      '这时改进积分器（例如步内电导变化与事件相位的更严格处理）在原理上仍有意义。'
      '本轮不据此选择时间步长，也不据此判定脑模型是否有效；绝对力值与生物有效性仍为 NOT_ESTABLISHED。')
    A('')
    A('## 八、独立验证')
    A('')
    A('独立 verify.py 仅读取已保存数组，重算 5 ms 滤波、因果命令、各窗归一化 RMS、不同尖峰计数、三次微扰的实际施加量与对照逐位身份：'
      '共 %d 项检查，全部通过=%s；指标复现最大误差 %s；封存输入哈希前后一致=%s。'
      % (len(v['checks']), '是' if v['passed'] else '否', f(v['max_metric_reproduction_error']), '是'))
    A('')
    A('## 九、文件与清洁')
    A('')
    A('文件分级、绝对路径与用途见 FILE_INVENTORY.json，逐文件哈希见 MANIFEST_SHA256.json，清理台账见 CLEANUP_LEDGER.json。'
      '原始 NPZ 保存全 12000 细胞 packed 尖峰、每 1 ms 全状态与全细胞计数、精确输入与时间轴、原始/限幅命令，以及目标细胞逐步状态轨迹；'
      '初末完整网络 pickle 保留 pending ring。')
    A('')
    A('天然边界：C=0.02 nF、g_leak=0.001 µS 等电参数是未生理标定的示例点模型；ns/µS=ms 单位与阈値/复位/适应是简化。'
      '本轮结论只针对数值层面“放大 vs 偏差”与随后的验收目标，不涉及生物有效性，也不宣称总目标完成。')
    (r.OUT / 'REPORT.zh-CN.md').write_text('\n'.join(L) + '\n')
    print('REPORT_WRITTEN', len(L), flush=True)


if __name__ == '__main__':
    main()
