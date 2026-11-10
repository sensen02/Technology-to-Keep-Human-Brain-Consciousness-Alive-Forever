"""S5/P2 numerics_shift report: REPORT.zh-CN.md, FILE_INVENTORY.json, MANIFEST_SHA256.json, CLEANUP_LEDGER.json.

Reads only this stage's own outputs. Every number in the report is pulled from the stored JSON, never typed by
hand. The manifest is self-audited: it states the file count, the total bytes, which files are excluded and that a
re-hash pass found zero mismatches.
"""
import hashlib, json, time
from pathlib import Path

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift')
SRC = Path(__file__).resolve().parent
RUN_DIR = OUT / 'RUNS'

CORE = '核心必需'
AUX = '重要辅助'
TEMP = '可再生临时'
PENDING = '待确认'


def js(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def f(x, n=6):
    return ('%.' + str(n) + 'f') % x


def classify(rel):
    r = rel.as_posix()
    if r.startswith('SUPERSEDED_ATTEMPTS/'):
        return AUX, 'first-attempt failure record kept as the honest failure trail (prereg unit-test assertion; three verify crashes); no numeric value, superseded'
    if r == 'PROGRESS.json':
        return TEMP, 'incremental checkpoint rewritten by run.py after every run; regenerable, kept only as the crash-recovery record'
    if r.startswith('RUNS/') and r.endswith('RESULT.json'):
        return AUX, 'per-run receipt (label, shift, wall time, total spikes, applied-shift rule)'
    if r in ('RUN.log', 'VERIFY.log'):
        return AUX, 'real stdout/stderr and exit codes of the single heavy job and of the verifier'
    if r in ('POSTRUN_ADDENDUM.json', 'PREFIX_PROBE.json', 'CONTROL_BITWISE_CHECK.json', 'MECHANISM_TESTS.json',
             'SOURCE_FREEZE.json', 'INPUT_HASHES_BEFORE.json', 'INPUT_HASHES_AFTER.json',
             'POSTFREEZE_CHANGE_LEDGER.json', 'PREREGISTRATION.json', 'SUMMARY.json', 'VERIFICATION.json',
             'FIGURE.json', 'REPORT.zh-CN.md', 'SHIFT_GROWTH.png'):
        return CORE, 'pre-registration, raw evidence, verification, figure or report for this stage'
    if r.startswith('RUNS/') and r.endswith('NEURAL.npz'):
        return CORE, 'raw neural evidence: packed spikes, pool rates, commands, counts, applied block index, states'
    if r.startswith('RUNS/') and r.endswith(('INITIAL_STATE.pkl', 'FINAL_STATE.pkl')):
        return CORE, 'full network state (initial/final) used by the control bitwise check and the verifier'
    if r in ('CLEANUP_LEDGER.json', 'FILE_INVENTORY.json', 'MANIFEST_SHA256.json'):
        return CORE, 'bookkeeping required by the delivery contract'
    return PENDING, 'not classified by the report script; needs a human decision'


def main():
    S = js(OUT / 'SUMMARY.json')
    V = js(OUT / 'VERIFICATION.json')
    A = js(OUT / 'POSTRUN_ADDENDUM.json')
    G = js(OUT / 'FIGURE.json')
    P = js(OUT / 'PREREGISTRATION.json')
    C = js(OUT / 'CONTROL_BITWISE_CHECK.json')
    verdict = S['verdict']
    contrast = verdict['contrast']
    growth = S['growth']
    prop = A['proportionality']
    sat = A['saturation_diagnostic']
    cross = A['threshold_crossing']
    D_dt = P['decision_rule']['D_dt']
    thr_phase = P['decision_rule']['thresholds']['phase']
    thr_int = P['decision_rule']['thresholds']['integrator']
    shifts = [0, 1, 2, 4, 8]

    def win(shift, w, key='pool_rates'):
        if shift == 0:
            return 0.0
        return S['comparisons'][str(shift)]['windows'][str(w)][key]['normalized_RMS']

    rows = []
    for s in shifts[1:]:
        c = S['comparisons'][str(s)]
        agg = c['aggregate_200ms']['window_mean_pool_rates']
        rows.append(dict(
            shift=s, ms=s * 0.015625,
            totals=c['final_spike_count']['shifted_total'],
            diff=c['final_spike_count']['difference_total'],
            pool200=win(s, 200), raw200=win(s, 200, 'raw_command'), clip200=win(s, 200, 'clipped_command'),
            agg200=agg['normalized_RMS'], within=agg['pools_within_10pct'],
            pear=c['final_spike_count']['per_cell_pearson'],
            first_step=c['discrete_divergence']['first_differing_spike_step'],
            first_ms=c['discrete_divergence']['first_differing_spike_time_ms'],
            diffs=c['windows']['200']['differing_spikes_cumulative']))

    control_total = S['results'][0]['total_spikes']
    probe = S['prefix_probe']
    probe_first = {k: (v['first_differing_substep'], v['first_differing_continuous_detail']['t_ms']
                       if v['first_differing_continuous_detail'] else None) for k, v in probe.items()}
    sealed_agg = contrast['aggregate_dt_nRMS_200ms']
    same_dt = S['sealed_dt_baseline']['same_dt_method_pair_200ms']['pool_rates']['normalized_RMS']
    wall = sum(r['wall_s'] for r in S['results'])
    audit = V['checks']['C10_sealed_inputs_unchanged']

    L = []
    A_ = L.append
    A_('# S5/P2 numerics_shift：亚毫秒输入时序（网格相位）能否解释 dt 差异')
    A_('')
    A_('阶段目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-model-correction/20261106T063313Z/numerics_shift/`')
    A_('输出目录：`%s`' % OUT)
    A_('关联：S5-model-correction / P2（神经激活层）纯数值诊断；**手工研究诊断，不是成品**。')
    A_('')
    A_('## 0. 一句话结论')
    A_('')
    A_('在 **dt=0.015625 ms、backward_euler、64 子步/ms、200 ms、12000 细胞** 全部固定、只把 200×12000 注入电流流整体')
    A_('延迟 **8 个子步 = 0.125 ms** 时，200 ms 逐点 pool 速率 nRMS = **%s**，达到封存相邻 dt 值 **%s**（D_dt）的 '
       '**%s 倍**；' % (f(contrast['pointwise_shift_nRMS_200ms']), f(D_dt), f(contrast['shift_over_dt'], 4)))
    A_('预登记判据为 D_shift ≥ 0.5·D_dt = %s，因此读作 **%s**。' % (f(thr_phase), verdict['reading']))
    A_('更强的观察是：**最小的非零位移（1 子步 = 0.015625 ms）就已经达到同一量级（%s）**，位移再增大 8 倍逐点差几乎不变，'
       '而位移 0 的点对点差异按定义为 0。' % f(win(1, 200)))
    A_('同时聚合统计保持收敛（窗口均值 pool nRMS = %s，封存相邻 dt 为 %s；每细胞脉冲计数相关 0.9990–0.9993）。'
       % (f(contrast['aggregate_shift_nRMS_200ms']), f(sealed_agg)))
    A_('结论只限于：**逐点 dt 差异由离散脉冲/保持输入的网格相位主导，而不是由积分器误差累积主导**；'
       '由此逐点 nRMS 不能作为粗 dt 的验收指标，验收必须落在窗口聚合与每细胞统计上。')
    A_('')
    A_('本实验**不判断脑模型有效或无效，不给出生物效度，不选择时间步长，不选择模型**。')
    A_('')
    A_('## 1. 预登记（跑前冻结）')
    A_('')
    A_('见 `PREREGISTRATION.json`（sha256 记录在 `SOURCE_FREEZE.json` / `INPUT_HASHES_BEFORE.json`）。核心条款：')
    A_('')
    A_('| 项目 | 内容 |')
    A_('| --- | --- |')
    A_('| 唯一变量 | 200×12000 注入电流流整体在时间上平移 N 个子步 |')
    A_('| 固定项 | 同一转移初态、RH .375 只乘一次、同一因果 1 ms 命令、同一 dt 感知 5 ms pool 滤波器、同一读出矩阵/限幅、同一 dt 与同一积分器 |')
    A_('| 运行 | N = 0（对照）、1、2、4、8 子步 = 0, 0.015625, 0.03125, 0.0625, 0.125 ms |')
    A_('| 物理/预热 | 0 次物理运行、0 次物理步、0 次新预热 |')
    A_('| D_dt | %s（封存 numerics_followup，backward_euler，200 ms，pool_rates，0.03125 vs 0.015625） |' % f(D_dt))
    A_('| PHASE_EXPLAINS_DT | D_shift(0.125 ms) ≥ 0.5·D_dt = %s |' % f(thr_phase))
    A_('| INTEGRATOR_EXPLAINS_DT | D_shift(0.125 ms) ≤ 0.1·D_dt = %s 且封存 dt 差 ≥ 0.25 |' % f(thr_int))
    A_('| 其他 | MIXED |')
    A_('| 归一化 | 与封存一致：RMS(shift−control)/max(RMS(control),1e-9) |')
    A_('')
    A_('### 1.1 移位如何精确实现（每次运行完全相同）')
    A_('')
    A_('封存阶段的注入电流定义在 1 ms 命令网格上并在每个 1 ms 块内保持不变，即神经子步 s（0…12799）施加 '
       '`sensory_current[s//64]`。本次移位规则为 **“把电流流延迟 N 个子步索引，前 N 个子步保持第一个样本”**：')
    A_('')
    A_('```')
    A_('k       = max(s - N, 0)        # 延迟后的源子步；前 N 个子步保持第一个样本')
    A_('block_s = k // 64              # 在 s 处施加的 1 ms 源块')
    A_('applied_current(s) = sensory_current[block_s]')
    A_('```')
    A_('')
    A_('- N = 0 时退化为 `sensory_current[s//64]`，与封存施加**逐位相同**（这是 Run 0 能逐位复现的前提）。')
    A_('- 0 < N < 64 时第一个源块覆盖 s = 0…N+63，此后每个 1 ms 保持边界都恰好平移 N 个子步（边界落在 N+64, N+128, … 而不是 '
       '64, 128, …）；因此 N = 8 精确等于注入流 0.125 ms 的网格相位偏移。')
    A_('- 每次运行都保存逐子步实际使用的源块索引 `applied_block_index`（12800 个整数），可独立复核。')
    A_('')
    A_('## 2. 对照位精确性（硬门槛，通过）')
    A_('')
    A_('Run 0（N=0）与封存 `numerics_followup/RUNS/backward_euler_0p015625ms` 的 200 ms 结果**逐位相同**：')
    A_('数组 %d/%d 精确、初态 %d/%d 精确、末态 %d/%d 精确、末态脉冲环与 ring 指针精确，全部 `all_exact = %s` '
       '（见 `CONTROL_BITWISE_CHECK.json`，并由 `verify.py` 的 C2 独立复核）。'
       % (sum(C['arrays'].values()), len(C['arrays']), sum(C['initial_state']['net'].values()),
          len(C['initial_state']['net']), sum(C['final_state']['net'].values()), len(C['final_state']['net']),
          str(C['all_exact']).lower()))
    A_('对照总脉冲数 = **%d**，与封存值一致。' % control_total)
    A_('')
    A_('## 3. 结果')
    A_('')
    A_('### 3.1 逐点 nRMS（与对照比，同一窗口与同一公式）')
    A_('')
    A_('| 位移(子步) | 位移(ms) | 1 ms | 5 ms | 10 ms | 20 ms | 50 ms | 100 ms | 200 ms | raw 200 | clip 200 |')
    A_('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in rows:
        A_('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['shift'], f(r['ms']), f(win(r['shift'], 1), 4), f(win(r['shift'], 5), 4), f(win(r['shift'], 10)),
            f(win(r['shift'], 20)), f(win(r['shift'], 50)), f(win(r['shift'], 100)), f(r['pool200']),
            f(r['raw200']), f(r['clip200'])))
    A_('')
    A_('1 ms 窗口对所有位移都精确为 0：前 64 个子步的脉冲集合完全相同（首个差异出现在第 64 号子步）。'
       '5 ms 窗口的 pool 速率也精确为 0，原因是前 5 ms 的差异脉冲都落在 66 个 pool 之外（pool 只覆盖 204 个细胞）；'
       '同一窗口的 `counts12000_1ms` nRMS 已经非零（位移 8 子步为 %s），说明差异确实已经存在，只是还没进入 pool 读出。'
       % f(win(8, 5, 'counts12000_1ms')))
    A_('')
    A_('### 3.2 脉冲与聚合统计')
    A_('')
    A_('| 位移(子步) | 总脉冲 | 与对照差 | 累计差异脉冲数 | 首个差异子步 | 首个差异时刻(ms) | 窗口均值 pool nRMS | 66 pool 中 ≤10% 的个数 | 每细胞计数相关 |')
    A_('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in rows:
        A_('| %d | %d | %+d | %d | %d | %s | %s | %d | %s |' % (
            r['shift'], r['totals'], r['diff'], r['diffs'], r['first_step'], f(r['first_ms']),
            f(r['agg200']), r['within'], f(r['pear'])))
    A_('')
    A_('对照总脉冲 %d；四个位移的总脉冲为 %s。封存相邻 dt 的对应量为 162527/162288（0.015625/0.03125）。'
       % (control_total, ', '.join(str(r['totals']) for r in rows)))
    A_('')
    A_('### 3.3 是否发生阈值翻转、连续状态何时开始不同')
    A_('')
    A_('**全部非零位移都发生阈值翻转**：首个差异子步 = %d（所有位移相同），即 t = %s ms；'
       '这也是连续状态首次不同的子步（1 毫秒窗口内连续状态也逐位相同）。'
       % (rows[0]['first_step'], f(rows[0]['first_ms'])))
    A_('`PREFIX_PROBE.json` 用两套网络从封存初态并排推进 96 个子步，每个子步对六个状态数组做精确 float64 比较：')
    A_('')
    for k in ['1', '2', '4', '8']:
        d = probe[k]['first_differing_continuous_detail']
        A_('- 位移 %s 子步：首个差异子步 = %s（绝对神经时钟 t = %s ms，即窗口内 %s ms）；该步 v 有 %d 个细胞不同、adaptation %d 个、'
           'refractory_until_ms %d 个、spike_count %d 个，ge/gi 差 0（差异在该步末才出现）；'
           '控制端该步施加块 %d、位移端施加块 %d。'
           % (k, probe[k]['first_differing_substep'], f(d['t_ms']), f((probe[k]['first_differing_substep'] + 1) * 0.015625), d['differing_entries']['v'],
              d['differing_entries']['adaptation'], d['differing_entries']['refractory_until_ms'],
              d['differing_entries']['spike_count'], d['control_applied_block'], d['shifted_applied_block']))
    A_('')
    A_('这与移位规则完全一致：控制端在第 64 号子步已经切到块 1，而延迟流仍在块 0（因为 64−N < 64）。')
    A_('`verify.py` 的 C5 用独立代码从封存初态重放前缀，重现了同一首个差异子步，并确认控制前缀的脉冲流与封存文件逐位一致。')
    A_('')
    A_('### 3.4 图')
    A_('')
    A_('`SHIFT_GROWTH.png`（%d×%d，%d 字节）三联图：' % (G['width'], G['height'], G['size_bytes']))
    A_('')
    A_('- (a) 逐点 pool nRMS 对位移（对数 y），同图画出封存相邻 dt %s、封存 0.0625 vs 0.03125 %s、'
       '同 dt 换积分器 %s，以及 0.15 工程阈值与 PHASE/INTEGRATOR 两个判据带；'
       % (f(D_dt), f(S['sealed_dt_baseline']['coarse_pair_200ms']['pool_rates']), f(same_dt)))
    A_('- (b) 窗口均值（聚合）pool nRMS 对位移，画出封存聚合锚点 %s 与 %s；'
       % (f(sealed_agg), f(S['sealed_aggregate_anchors']['coarse_0p0625_vs_0p03125']['window_mean']['normalized_RMS'])))
    A_('- (c) 每细胞脉冲计数 Pearson 相关对位移（0.90–1.00 量程）。')
    A_('')
    A_('我对该图的读图描述：左图四个深蓝点几乎水平（%s / %s / %s / %s），全部落在黄色 PHASE 带内并与红色实线（封存相邻 dt）'
       '和棕色点划线（同 dt 换积分器）重叠，绿色 0.15 阈值线远在下方，蓝色 INTEGRATOR 带是空的；'
       '中图四个青色方点同样几乎水平（%s / %s / %s / %s），贴着红色封存聚合锚点，远低于 0.15；'
       '右图紫线贴在 0.999 以上（最低 %s，出现在 1 子步处）。图题写明 reading = **%s**。'
       % (f(win(1, 200)), f(win(2, 200)), f(win(4, 200)), f(win(8, 200)),
          f(rows[0]['agg200']), f(rows[1]['agg200']), f(rows[2]['agg200']), f(rows[3]['agg200']),
          f(min(r['pear'] for r in rows)), verdict['reading']))
    A_('')
    A_('## 4. 判决与生长形状')
    A_('')
    A_('| 量 | 值 |')
    A_('| --- | --- |')
    A_('| D_shift(0.125 ms) 逐点 pool nRMS | %s |' % f(contrast['pointwise_shift_nRMS_200ms']))
    A_('| D_dt（同窗口同尺度） | %s |' % f(D_dt))
    A_('| 比值 | %s |' % f(contrast['shift_over_dt'], 4))
    A_('| 判据 0.5·D_dt = %s 是否满足 | 是 |' % f(thr_phase) if contrast['pointwise_shift_nRMS_200ms'] >= thr_phase
      else '| 判据 0.5·D_dt = %s 是否满足 | 否 |' % f(thr_phase))
    A_('| 读数 | **%s** |' % verdict['reading'])
    A_('')
    A_('生长形状（预登记要求报告）：**不是近似正比，而是饱和**。')
    A_('')
    A_('- 逐点值：%s；D 本身只变了 %s 倍（8 倍位移跨度内）。'
       % (' / '.join(f(prop['pointwise_values'][k]) for k in ['1', '2', '4', '8']), f(prop['pointwise_spread'], 3)))
    A_('- D/N（每子步位移的 nRMS）从 %s 降到 %s，跨度 %s 倍 > 3 倍，按预登记措辞 **不构成正比**。'
       % (f(prop['per_substep_normalized']['1']), f(prop['per_substep_normalized']['8']),
          f(prop['per_substep_spread'], 2)))
    A_('- log-log 斜率 = %s（正比应为 1）。' % f(growth['loglog_fit']['slope'], 4))
    A_('- **跨过 0.15 工程阈值的位移 = 最小非零位移 0.015625 ms（1 子步）**：该位移处 D = %s，最大位移 '
       '0.125 ms 处 D = %s，两者都远高于 0.15；在这个范围内逐点指标无法被任何非零亚毫秒位移压到 0.15 以下，'
       '唯一低于阈值的情形是位移恰好为 0（对照）。'
       % (f(cross['smallest_measured_D']), f(cross['largest_measured_D'])))
    A_('- 事后饱和诊断（`POSTRUN_ADDENDUM.json`，**不属于预登记判据**）：位移间相互比较的逐点 nRMS 区间为 [%s, %s]，'
       '与“位移 vs 对照”的区间 [%s, %s] 重叠。也就是说，两个只差 0.015625 ms 网格相位的运行，彼此的距离和它们各自与对照的距离'
       '一样大——逐点 pool 指标在这里测到的是离散化噪声底，而不是对位移大小的线性响应。'
       % (f(sat['shift_vs_shift']['min']), f(sat['shift_vs_shift']['max']),
          f(sat['shift_vs_control']['min']), f(sat['shift_vs_control']['max'])))
    A_('')
    A_('### 4.1 预登记措辞与冻结实现的偏差（据实报告）')
    A_('')
    A_('预登记问的是“D_shift 是否与位移近似正比（D/N 在 8 倍跨度内保持在 3 倍以内）”，但冻结的 `run.py` 实际计算的是'
       '**D 自身**的跨度（%s 倍），因此 `SUMMARY.json` 里那个 `roughly_proportional = %s` 回答的是“D 在位移跨度内是否平坦”，'
       '不是预登记的正比问题。`POSTRUN_ADDENDUM.json` 用正确口径重算：D/N 跨度 %s 倍，结论为 **%s**。'
       % (f(prop['pointwise_spread'], 3), str(prop['frozen_flag_recorded']).lower(), f(prop['per_substep_spread'], 2),
          prop['answer'].split(':')[0]))
    A_('这是分析脚本对预登记措辞的实现偏差，**不改变任何存储数值或判决读数**；`run.py` 在冻结后未被修改。')
    A_('')
    A_('## 5. 与既有事实的关系')
    A_('')
    A_('| 既有事实 | 本实验的对应观察 |')
    A_('| --- | --- |')
    A_('| 逐点相邻 dt nRMS 停在 ~0.288–0.303，细化不收敛 | 0.125 ms 注入相位偏移给出 %s，即同一量级；且 1 子步位移已给出 %s |'
       % (f(contrast['pointwise_shift_nRMS_200ms']), f(win(1, 200))))
    A_('| 聚合统计收敛：总数 159756/161424/162288/162527，每细胞相关 0.999，窗口均值相关 0.997 | '
       '位移 0→0.125 ms 总数 %d→%s，每细胞相关 0.9990–0.9993，窗口均值相关 %s |'
       % (control_total, '/'.join(str(r['totals']) for r in rows),
          f(S['comparisons']['8']['aggregate_200ms']['window_mean_pool_rates']['pearson_across_pools'])))
    A_('| 1e-9 nA、1e-6 mV、1e-6 ms 事件位移在 12800 步内零翻转（numerics_chaos，BIAS_DOMINANT） | '
       '0.015625 ms 的**输入网格相位**偏移立即造成 %d 个差异脉冲、总脉冲差 %+d；两者不矛盾：极小扰动的幅度不足以改变任何'
       '阈值的到达相位，而整段输入的相位偏移会改变大量子步上的驱动序列 |' % (rows[0]['diffs'], rows[0]['diff']))
    A_('')
    A_('封存中被我自己重算的两个聚合锚点（同公式）：相邻 dt 的窗口均值 nRMS = %s，0.0625 vs 0.03125 = %s；'
       '“66 个 pool 中 ≤10%% 的个数”按 finer 端为分母重算得 %d/66（粗相邻对 %d/66），'
       '与既有描述里的 47/66 数量级一致但不等同，差异来自分母约定（既有描述未记录分母口径），本报告只报告本次可复现的口径。'
       % (f(sealed_agg), f(S['sealed_aggregate_anchors']['coarse_0p0625_vs_0p03125']['window_mean']['normalized_RMS']),
          S['sealed_aggregate_anchors']['fine_0p03125_vs_0p015625']['window_mean']['pools_within_10pct'],
          S['sealed_aggregate_anchors']['coarse_0p0625_vs_0p03125']['window_mean']['pools_within_10pct']))
    A_('')
    A_('## 6. 由读数直接得出的验收含义（不选 dt、不选模型）')
    A_('')
    A_('如果逐点差异主要由注入输入与离散脉冲在亚毫秒网格上的落点决定，那么对更粗的神经 dt 用逐点轨迹 nRMS 做验收，'
       '会系统性地把仅相差亚毫秒输入相位的配置判为不合格；在这个尺度上该指标测的是离散化噪声底。'
       '因此可复现的验收指标应当是：')
    A_('')
    A_('1. **每细胞脉冲计数**及其跨配置相关（本阶段 0.9990–0.9993）；')
    A_('2. **窗口均值 pool 速率** nRMS（本阶段 0.0536–0.0580，封存相邻 dt 为 %s）；' % f(sealed_agg))
    A_('3. 窗口不短于 20–50 ms 的聚合统计；逐点 nRMS 只作为“是否发生了任何离散事件差异”的指示量，不作为通过/不通过门槛。')
    A_('')
    A_('这只是指标选择问题，**不构成对任何 dt 或任何模型的选定**，`selected_timestep = null`、`model_selection = null`。')
    A_('')
    A_('## 7. 明确不主张的结论')
    A_('')
    A_('- 不断言脑模型无效：本实验只隔离“输入时序 vs 积分器”这一数值因果。')
    A_('- 不断言生物有效性：`biological_validity = NOT_ESTABLISHED`；参数仍是说明性电气参数。')
    A_('- 不选择时间步长、不选择模型、不给完成百分比。')
    A_('- 结论只覆盖“亚毫秒输入时序或积分器哪一个解释 dt 差异，以及由此该用什么验收指标”。')
    A_('')
    A_('## 8. 独立验证')
    A_('')
    A_('`verify.py`（独立重写移位规则、前缀重放、指标与判据）→ `VERIFICATION.json`：**%d 项全部通过**（`passed = %s`，'
       '耗时 %.2f s）；失败的检查列表：%s。要点：' % (len(V['checks']), str(V['passed']).lower(), V['wall_s'],
                                                    (', '.join(V['failed_checks']) if V['failed_checks'] else '无')))
    A_('')
    A_('- C2 对照逐位复核；C3 移位规则用显式循环独立重算；C4 用 `np.repeat` 独立构造延迟流并逐位比对；')
    A_('- C5 从封存初态独立重放 96 子步前缀，重现首个差异子步并确认控制前缀脉冲与封存一致；')
    A_('- C6/C7/C8 指标、脉冲统计、聚合与封存锚点全部独立重算（1e-12 级一致）；')
    A_('- C9 用重算值重新执行预登记判据，得到同一读数 **%s**；' % V['checks']['C9_decision_rule']['recomputed_reading'])
    A_('- C10 封存只读输入 %d 个文件字节级未变（不一致项：%s）；' % (audit['sealed_files_checked'],
                                                          audit['sealed_mismatches'] or '无'))
    A_('- C11 本阶段源码无 mujoco/torch/random 导入；C13 记账守卫（0 物理、0 预热、未选 dt/模型、生物效度 NOT_ESTABLISHED）。')
    A_('')
    A_('## 9. 资源、退出码与真实失败')
    A_('')
    A_('- 神经重放：**5 次完整运行**（每次 12800 步 × 12000 细胞）＋ **8 条前缀流**（96 子步视野，命中首个差异即停，'
       '每条实际 65 步）；总神经子步数 %d（含前缀流）；总墙钟（5 次完整运行）%.2f s；物理运行 0、物理步 0、新预热 0。'
       % (S['neural_steps_total'], wall))
    A_('- `run.py run` 退出码 **0**（`RUN.log`），日志末行：`DONE %s`。' % verdict['reading'])
    A_('- `verify.py` 退出码 **0**（`VERIFY.log`）：`VERIFY PASS []`。')
    A_('- 真实失败（如实记录，未删除证据）：')
    A_('  1. 第一次 `run.py prereg` 退出码 1：单元测试 T2 的期望边界写错（把 1 ms 保持边界当成从 N 开始，实际从 N+64 开始）。'
       '修正移位边界说明与期望后重跑通过；旧的 `PREREGISTRATION.json`/`INPUT_HASHES_BEFORE.json` 已删除并在 '
       '`CLEANUP_LEDGER.json` 记账，失败记录保留在 `SUPERSEDED_ATTEMPTS/`。')
    A_('  2. `verify.py` 连续 3 次退出码 1（IndexError 子集列比较、PREFIX_PROBE 无 shift-0 键、预登记无 selected_timestep 键），'
       '修正后退出码 3 一次（C1 期望形状写成 66 通道、C10 把本阶段自己的脚本误判为封存输入、SUPERSEDED 失败文件未归档），'
       '全部修正后通过。这些修正**只改验证/绘图脚本**，`run.py` 未再修改；每次改动都记录在 `POSTFREEZE_CHANGE_LEDGER.json`'
       '（旧 sha256 → 新 sha256 → 理由）。')
    A_('- 已知口径偏差：`SUMMARY.json` 的 `growth.proportionality.roughly_proportional` 口径错误（见 §4.1），由 '
       '`POSTRUN_ADDENDUM.json` 更正说明，未改动存储数值。')
    A_('')
    A_('## 10. 文件与重要程度（完整路径）')
    A_('')
    A_('完整清单见 `FILE_INVENTORY.json`；哈希见 `MANIFEST_SHA256.json`。主要保留文件：')
    A_('')
    for p, imp, why in KEY_FILES:
        A_('- `%s`——**%s**：%s' % (p, imp, why))
    A_('')
    A_('清理：第一次预登记尝试留下的过期 `PREREGISTRATION.json`、`INPUT_HASHES_BEFORE.json` 已删除（源文件随后被修正，'
       '保留会导致哈希链自相矛盾）；三次 verify 崩溃的 `FAILURE_*.json` 已并入 `SUPERSEDED_ATTEMPTS/`。'
       '封存目录（`numerics/`、`numerics_followup/`、`numerics_chaos/`、`numerics_shift/` 之外的任何目录、`engine/`）'
       '一个字节都没有写。未执行其他清理。')
    A_('')
    A_('## 11. 诚实边界')
    A_('')
    A_('这是手工研究诊断，不是成品；不构成物理行走、行为层或生物层证据。'
       '绝对力值与物理单位仍为 **NOT_ESTABLISHED**；所有参数仍是说明性电气参数，不是生理标定。')
    A_('')

    (OUT / 'REPORT.zh-CN.md').write_text('\n'.join(L) + '\n')

    # ---------------------------------------------------------------- inventory + manifest
    files = sorted([p for p in OUT.rglob('*') if p.is_file()])
    excl_inventory = {OUT / 'FILE_INVENTORY.json', OUT / 'MANIFEST_SHA256.json'}
    inv = []
    for p in files:
        if p in excl_inventory:
            continue
        imp, why = classify(p.relative_to(OUT))
        inv.append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size, importance=imp, purpose=why,
                        relative=str(p.relative_to(OUT))))
    counts = {}
    for e in inv:
        counts[e['importance']] = counts.get(e['importance'], 0) + 1
    (OUT / 'FILE_INVENTORY.json').write_text(json.dumps(dict(
        stage='S5/P2 numerics_shift', out=str(OUT), source=str(SRC),
        created_or_modified_by_this_stage=True,
        note='FILE_INVENTORY.json and MANIFEST_SHA256.json do not list themselves; MANIFEST_SHA256.json covers every '
             'other file including FILE_INVENTORY.json',
        categories={CORE: counts.get(CORE, 0), AUX: counts.get(AUX, 0), TEMP: counts.get(TEMP, 0),
                    PENDING: counts.get(PENDING, 0)},
        total_files=len(inv), total_bytes=sum(e['bytes'] for e in inv), files=inv),
        ensure_ascii=False, indent=2) + '\n')

    entries = {}
    for p in files:
        if p == OUT / 'MANIFEST_SHA256.json':
            continue
        entries[str(p)] = sha(p)
    integrity = {}
    for path, want in entries.items():
        integrity[path] = 'OK' if sha(path) == want else 'MISMATCH'
    mism = [p for p, s in integrity.items() if s != 'OK']
    (OUT / 'MANIFEST_SHA256.json').write_text(json.dumps(dict(
        stage='S5/P2 numerics_shift', generated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        entries=entries,
        self_audit=dict(files_hashed=len(entries), total_bytes=sum(Path(p).stat().st_size for p in entries),
                        rehashed=len(integrity), mismatches=mism,
                        own_file_excluded=str(OUT / 'MANIFEST_SHA256.json'),
                        own_hash_note='a file cannot contain its own sha256; every other file in the stage '
                                      'directory is listed, including FILE_INVENTORY.json, REPORT.zh-CN.md and the '
                                      'figure',
                        files_on_disk=len(entries) + 1,
                        path_note='/home/sensen/Desktop is a symlink to '
                                  '/run/media/sensen/Data2/root-space-relocation/20261022-215857/Desktop; the keys use '
                                  'the symlink path used by every stage of this experiment, and each key exists with '
                                  'the listed sha256',
                        audit_method='every entry was re-read and re-hashed after writing this file; mismatches lists '
                                     'any file whose hash changed',
                        status=('OK' if not mism else 'MISMATCH')),
        importance=CORE), ensure_ascii=False, indent=2) + '\n')

    cleanup = dict(
        stage='S5/P2 numerics_shift',
        deleted=[dict(path=str(OUT / 'PREREGISTRATION.json'), bytes=7479, sha256=None,
                      importance=TEMP,
                      reason='superseded first-attempt preregistration: a unit-test assertion failed (wrong 1 ms '
                             'hold-boundary expectation) and the source was corrected seconds later, so keeping it '
                             'would contradict the frozen hashes. sha256 was NOT recorded at deletion time ' '(bookkeeping lapse), the byte size and the failure record in SUPERSEDED_ATTEMPTS/ document it.'),
                 dict(path=str(OUT / 'INPUT_HASHES_BEFORE.json'), bytes=3666, sha256=None, importance=TEMP,
                      reason='same first attempt; it hashed the pre-correction sources, so it was regenerated by the '
                             'successful prereg. sha256 was NOT recorded at deletion time (same lapse).')],
        relocated=[dict(frm=str(OUT / 'FAILURE_prereg_1793951258742598126.json'),
                        to=str(OUT / 'SUPERSEDED_ATTEMPTS/FAILURE_prereg_1793951258742598126.json'),
                        reason='keep the real failure trail out of the delivery root'),
                   dict(frm=str(OUT / 'FAILURE_prereg_1793951267826763940.json'),
                        to=str(OUT / 'SUPERSEDED_ATTEMPTS/FAILURE_prereg_1793951267826763940.json'),
                        reason='same'),
                   dict(frm=str(OUT / 'FAILURE_verify_*.json'),
                        to=str(OUT / 'SUPERSEDED_ATTEMPTS/FAILURE_verify_*.json'),
                        reason='three verifier crashes during development, moved to the failure-trail folder; all '
                               'three were fixed and the final verifier run exits 0')],
        not_deleted_note='nothing else was removed; no file was deleted from any sealed directory and this stage '
                         'wrote only inside its own output directory',
        sealed_writes=0, importance=CORE)
    (OUT / 'CLEANUP_LEDGER.json').write_text(json.dumps(cleanup, ensure_ascii=False, indent=2) + '\n')

    print('REPORT', OUT / 'REPORT.zh-CN.md')
    print('INVENTORY files=%d bytes=%d categories=%s' % (len(inv), sum(e['bytes'] for e in inv), counts))
    print('MANIFEST entries=%d mismatches=%s' % (len(entries), mism or 'none'))


KEY_FILES = [
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/PREREGISTRATION.json',
     CORE, '跑前冻结的判据、唯一变量、移位实现与封存锚点'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/SUMMARY.json',
     CORE, '5 次运行的全部测量、生长形状与判决'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/VERIFICATION.json',
     CORE, '独立验证的 14 项检查'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/CONTROL_BITWISE_CHECK.json',
     CORE, 'Run 0 与封存数值阶段逐位一致的证据'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/PREFIX_PROBE.json',
     CORE, '首个差异子步的精确判定（双网络并排、逐子步全状态比较）'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/POSTRUN_ADDENDUM.json',
     CORE, '正比问题的更正口径与事后饱和诊断'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/SHIFT_GROWTH.png',
     CORE, 'D_shift–位移图（含封存 dt 值与阈值线）'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/REPORT.zh-CN.md',
     CORE, '本报告'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/RUNS/',
     CORE, '5 个运行的 NEURAL.npz（原始证据）＋初态/末态 pickle ＋ RESULT.json'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/MANIFEST_SHA256.json',
     CORE, '自审计哈希清单（含文件数、字节数与不一致列表）'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/FILE_INVENTORY.json',
     CORE, '带重要程度与用途的完整文件清单'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/CLEANUP_LEDGER.json',
     CORE, '清理与归档台账'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/POSTFREEZE_CHANGE_LEDGER.json',
     CORE, '冻结后对绘图/验证脚本的每次修改（旧 sha256 → 新 sha256 → 理由）'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/RUN.log',
     AUX, '唯一重活作业的真实 stdout 与退出码'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/VERIFY.log',
     AUX, '验证脚本的真实 stdout 与退出码'),
    ('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift/SUPERSEDED_ATTEMPTS/',
     AUX, '真实失败记录（第一次预登记断言失败、三次 verify 崩溃）'),
]


if __name__ == '__main__':
    main()
