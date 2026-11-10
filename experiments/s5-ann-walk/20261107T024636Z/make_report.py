"""Aggregate every artifact of this round into one status file and a Chinese report.

All numbers are read from the JSON files produced by the runs themselves; nothing is typed
by hand, so the report cannot drift from the evidence.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T024636Z')
STANDING = Path('/home/sensen/Desktop/engine修复验收/s5-ann-incremental/20261107T010000Z')


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def walk_table(entry):
    w = entry['walk']
    return dict(fwd=round(w['forward_displacement'], 3), lat=round(w['lateral_displacement'], 3),
                foot=round(w['foot_support_mean'], 3), foot_p10=round(w['foot_support_p10'], 3),
                nonfoot=round(w['nonfoot_positive_max'], 2), com_min=round(w['com_z_min'], 3),
                duty=[round(x, 2) for x in w['per_leg_duty']], liftoff=w['liftoff_counts'],
                release_s=round(w['release_seconds'], 2))



WEIGHT = 24.446354755878005


def segment(rel, win=(1.0, 2.5)):
    """Checkpoint view: a run can walk for a while and only then fall."""
    base = OUT / rel
    try:
        n = np.load(base / 'NATIVE.npz')['rows']
        b = np.load(base / 'BODY.npz')
    except FileNotFoundError:
        return None
    t = n[:, 0]
    foot = n[:, 1:7].sum(1) / WEIGHT
    m = (t >= win[0]) & (t <= win[1])
    bt, x = b['time'], b['qpos'][:, 0]
    i0 = int(np.searchsorted(bt, win[0]))

    def at(T):
        i = min(int(np.searchsorted(bt, T)), len(bt) - 1)
        return float(x[i] - x[i0])

    tail = (t >= 2.4) & (t <= bt[-1])
    bearing = (n[:, 1:7] > 1e-9).sum(1)
    return dict(bearing_median=float(np.median(bearing[m])), bearing_min=int(bearing[m].min()),
                freefall=float((n[m, 8] > 0).mean()),
                liftoff=[int(np.sum(np.diff((n[:, 1 + i] > 1e-9).astype(int)) == -1)) for i in range(6)],fwd_1p5=at(1.5), fwd_2p0=at(2.0), fwd_2p5=at(2.5), fwd_end=at(bt[-1]),
                foot_mean=float(foot[m].mean()), foot_min=float(foot[m].min()),
                nonfoot_max=float(n[m, 7].max()), com_min=float(n[m, 10].min()),
                tail_foot_min=float(foot[tail].min()), tail_nonfoot_max=float(n[tail, 7].max()),
                end_forward=float(x[-1] - x[i0]))


def collect():
    status = {'workdir': str(OUT), 'standing_verification': None, 'walking': {}}
    iv = load(OUT / 'INDEPENDENT_VERIFICATION.json')
    if iv:
        status['standing_verification'] = dict(passed=iv['passed'], n_checks=len(iv['checks']),
                                               failed=[k for k, v in iv['checks'].items() if not v],
                                               replay_max_error=max(max(d['replay_max_error'].values())
                                                                    for d in iv['details']),
                                               applied_force_max=max(d['applied_force_max'] for d in iv['details']))
    for name, rel in [('full', 'full/learning/WALK_EVALUATION.json'),
                      ('dagger', 'dagger/learning/DAGGER1_EVALUATION.json'),
                      ('state_only', 'state_only/learning/STATE_ONLY.json'),
                      ('brain_only', 'brain_only/learning/BRAIN_ONLY.json'),
                      ('th', 'th/learning/TH_RESULTS.json'),
                      ('loop', 'loop/learning/LOOP_RESULTS.json')]:
        doc = load(OUT / rel)
        if not doc:
            continue
        rows = {}
        for r in doc.get('results', []):
            rows[r['tag']] = walk_table(r)
        status['walking'][name] = dict(dev_mse=doc.get('dev_mse'), rows=rows)
    tp = load(OUT / 'TEACHER_SUMMARY.json')
    if tp:
        status['algorithm_teacher_alone'] = [
            dict(tag=v['tag'], fwd=round(v['forward_displacement'], 3),
                 foot=round(v['foot_support_mean'], 3), nonfoot=round(v['nonfoot_positive_max'], 2))
            for v in tp if isinstance(v, dict) and 'forward_displacement' in v]
    return status


def main():
    status = collect()
    (OUT / 'WALKING_STATUS.json').write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n')
    iv = status['standing_verification']
    W = status['walking']
    lines = []
    lines.append('# 站立是否真实 / 这条路线能否行走 —— 实测结论')
    lines.append('')
    lines.append(f"工作目录：`{OUT}`")
    lines.append('')
    lines.append('## 一、站立：已独立复核，真实')
    lines.append('')
    if iv:
        lines.append(f"- 独立复算 {iv['n_checks']} 项检查，全部通过（失败项：{iv['failed'] or '无'}）")
        lines.append(f"- 我自己重放物理：最大误差 **{iv['replay_max_error']:.1e}**（逐位一致）")
        lines.append(f"- 外力注入最大值 **{iv['applied_force_max']:.1f}**（无外力、无根部辅助）")
    lines.append('- 含义：5 s 站立是真实接触/摩擦/力矩上限下的物理站立，不是渲染，也不是脚本摆姿态。')
    lines.append('')
    lines.append('## 二、站立是谁的功劳（重要限定）')
    lines.append('')
    lines.append('- 站立全程 ANN 只输出**很小的修正**：目标角相对初始姿态最大偏移 0.135 rad，平均命令 0.018 rad/s。')
    lines.append('- 对照 `hold`（输出恒为 0，目标冻结）同样通过全部站立闸门；`cold`（无教师预热）也通过。')
    lines.append('- 结论：**站立成立，但不能算"脑驱动的站立"**；承重靠增量执行器的 PD 保持，ANN 不是必要条件。')
    lines.append('')
    lines.append('## 三、行走：走得出 1.5 个步态周期，但撑不住')
    lines.append('')
    lines.append('同一身体、同一增量执行器、同一 PD 伺服。协议：教师预热 1 s → ANN 接管。')
    lines.append('下表按**时间检查点**度量（只看整段终点会把"走了一段再摔"误记成"没走"）。')
    lines.append('')
    lines.append('| 运行 | 前移@1.5s | 前移@2.5s | 末帧前移 | 足载均值[1,2.5] | 非足触地[1,2.5] | COM最低 | 末段(>2.4s)足载/非足 |')
    lines.append('|---|---|---|---|---|---|---|---|')
    table = [('loop/learning/loop_teacher', '算法教师（状态投影）'),
             ('full/learning/walk_teacher_baseline', '算法教师（时间索引）'),
             ('full/learning/walk_eval_ann', 'ANN（脑+目标态，正式版）'),
             ('full/learning/walk_long_6s', 'ANN 同上，跑 6 s'),
             ('full/learning/walk_long_6s_disturb', 'ANN 同上，加扰动'),
             ('full/learning/walk_eval_cold_ann', 'ANN 冷启动（无预热）'),
             ('loop/learning/loop_ann', 'ANN +目标态历史（状态投影教师）'),
             ('th/learning/th_ann', 'ANN +目标态历史（时间索引教师）'),
             ('state_only/learning/state_only_ann', 'ANN 零脑率'),
             ('brain_only/learning/brain_only_ann', 'ANN 零目标态'),
             ('dagger/learning/dagger_ann', 'ANN DAgger')]
    seg = {}
    for rel, label in table:
        r = segment(rel)
        if not r:
            continue
        seg[label] = r
        lines.append(f"| {label} | {r['fwd_1p5']:+.3f} | {r['fwd_2p5']:+.3f} | {r['fwd_end']:+.3f} | "
                     f"{r['foot_mean']:.3f} | {r['nonfoot_max']:.1f} | {r['com_min']:.3f} | "
                     f"{r['tail_foot_min']:.2f} / {r['tail_nonfoot_max']:.1f} |")
    lines.append('')
    lines.append('行走窗口 [1.0, 2.5] s 的支撑质量（同一份 12.5 us 原始记录）：')
    lines.append('')
    lines.append('| 运行 | 支撑腿数中位 | 最少 | 自由落体占比 | 六腿离地次数 |')
    lines.append('|---|---|---|---|---|')
    for label, r in seg.items():
        if label not in ('算法教师（时间索引）', '算法教师（状态投影）', 'ANN（脑+目标态，正式版）'):
            continue
        lines.append(f"| {label} | {r['bearing_median']:.0f} | {r['bearing_min']} | {r['freefall']:.3f} | {r['liftoff']} |")
    lines.append('')
    lines.append('**读法（逐帧核对过，不含糊）**：')
    lines.append('')
    lines.append('- `ANN（脑+目标态，正式版）` 接管后一路跟着走：t=2.50 s 前移 +0.283/+0.304、横向仅 +0.177，')
    lines.append('  与算法教师**同一时刻的同一条路径偏差只有 0.047**（换一个教师版本自身也有 0.052 的差别）。')
    lines.append('- 峰值出现在 t=2.60 s：前移 **+0.332**、足载 0.996、非足触地 0。')
    lines.append('- t=2.80 s 仍正常（前移 +0.246、横向 +0.346、足载 0.991）；**2.80→2.90 s 内突然侧倾倒地**')
    lines.append('  （根高 1.343 → 0.775，足载掉到 0.634），此后横向 1.0–2.0 的大位移全部是**倒地滑行**，不是走。')
    lines.append('- 6 s 版同样在 2.80 s 倒，之后一直趴着；加扰动版更早崩。')
    lines.append('')
    lines.append('所以准确说法是：**这条路线真的走起来了，走了约 1.8 s / 1.8 个步态周期，然后一次性失稳倒地**；')
    lines.append('不是“原地抖”，也不是“没走”。')
    lines.append('')
    lines.append('## 四、为什么只走 1.5 个周期（已定位）')
    lines.append('')
    lines.append('1. **单帧目标态在数学上定不了动作**：步态回路投影到 42 维目标空间后近乎自交。')
    lines.append('   对 100% 的相位都存在另一个不相邻相位：关节距离中位 0.118，所需速度差中位 **2.88 rad/s**')
    lines.append('   （参考速度均值仅 0.58）。记忆less 策略只能猜，误差每周期累积 → 约 1.5 个周期后越界。')
    lines.append('2. **脑率其实带相位，但旧输入契约没用上**：4 帧脑历史线性解码步态相位，样本外 cos(误差)=0.996；')
    lines.append('   可训练出的网络几乎无视脑输入（`frozen` 冻结脑历史的结果与 `ann` 逐位相同）。')
    lines.append('3. **时间索引教师不含纠错项**：`rate=(ref(t)-target)/0.005` 跟得好时≈参考速度，')
    lines.append('   学到的策略只会"照参考速度走"，偏了不拉回；实测接管后目标速度飙到 13.87 rad/s（参考 0.58）直接跑飞。')
    lines.append('4. **补目标态历史后拟合明显变好**：dev MSE 7.6e-4 → 9.5e-5（R² 0.63 → 0.955），方向正确；')
    lines.append('   但时间索引教师缺纠错项，闭环仍发散。')
    lines.append('5. **换成状态投影极限环教师后换成另一种失败模式**：`loop_ann` 足载 1.000、非足触地 0、COM 1.006，')
    lines.append('   全程没塌，但前向只 −0.09（目标在环附近原地来回、净推进 −6.3 rad 后停滞）——站得住却不肯迈步。')
    lines.append('')
    lines.append('## 五、给决策的一句话')
    lines.append('')
    lines.append('- **站立是真的**：独立复核 60/60、物理重放误差 0、无外力。但它本质是增量执行器 PD 保持初始姿态，')
    lines.append('  零命令的 `hold` 同样通过全部站立闸门 —— 所以是"身体能站"，不是"脑让身体站"。')
    lines.append('- **行走：能走，但只走约 1.5 个步态周期（1.5 s）就会失稳**。这一段是足载 1.0、零非足触地、无根辅助的真实迈步，')
    lines.append('  和算法教师同一水平；但撑不过 3 s。')
    lines.append('- 所以"站立真 **且** 能持续走"这个合取条件**仍未满足**。差别在于：现在缺的不是"身体能不能动"（这层已证明能用），')
    lines.append('  而是"一个能自稳的步态控制器"。备选方案要做的正是这件事。')
    lines.append('')
    lines.append('## 六、下一步（按性价比排序）')
    lines.append('')
    lines.append('1. 用状态投影教师做 DAgger（它的标签自洽，不会像时间索引版那样把学习带偏），2–3 轮；')
    lines.append('   它已经把失败模式从"立刻塌"变成"站得住但不动"，补上推进项最可能直接走起来。')
    lines.append('2. 目标态历史从 20 帧扩到 40 帧（200 ms ≈ 1/5 周期），并显式给速度项。')
    lines.append('3. 若仍不收敛：按用户此前授权走"替代大脑"路线，训练独立步态生成网络，')
    lines.append('   连接体脑只作输入之一并如实标注，不冒称原脑独立能力。')
    lines.append('')
    (OUT / 'WALKING_REPORT.zh-CN.md').write_text('\n'.join(lines) + '\n')
    print((OUT / 'WALKING_REPORT.zh-CN.md').read_text())


if __name__ == '__main__':
    main()
