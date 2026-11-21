"""Render the completed experiment receipt from its verified metrics."""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent / 'run'
KEYS = ('baseline', 'prior_practice', 'practice100')


def read(path):
    return json.loads(path.read_text())


def table_row(group, key, label, units):
    return f"| {label} | {units} | " + " | ".join(f"{group[k][key]:.4f}" for k in KEYS) + " |"


def main():
    summary = read(RUN/'summary.json')
    audit = read(RUN/'audit.json')
    rows = []
    for split, title in [('flat','平地（7 回合）'),('old_terrain','上轮留出的起伏面（4 张）'),('new_terrain','本轮留出的起伏面（20 张）')]:
        group = summary[split]
        rows.extend([
            table_row(group,'error_50ms_mm',f'{title} · 50 ms 标记误差','mm'),
            table_row(group,'error_200ms_mm',f'{title} · 200 ms 标记误差','mm'),
            table_row(group,'estimated_endpoint_mm',f'{title} · 自估足压 2 秒终点误差','mm'),
            table_row(group,'estimated_tail_rms_mm_s',f'{title} · 自估足压末段速度 RMS','mm/s'),
        ])
    results = read(RUN/'evaluation.json')
    subset = [r for r in results if r['split']=='new_terrain']
    walking = []
    standing = []
    for start, matching in [(11701,subset[::2]), (11903,subset[1::2])]:
        vals = [sum(r[k]['estimated_net_mm'] for r in matching)/len(matching) for k in KEYS]
        teacher = sum(r['teacher_net_mm'] for r in matching)/len(matching)
        walking.append(f"| {start} | {teacher:.3f} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        real_speeds = [read(RUN.parent/'episodes'/r['trial']/'receipt.json')['tail_rms_mm_s'] for r in matching]
        estimated_speeds = [sum(r[k]['estimated_tail_rms_mm_s'] for r in matching)/len(matching) for k in KEYS]
        standing.append(f"| {start} | {sum(real_speeds)/len(real_speeds):.3f} | "
                        + " | ".join(f"{v:.3f}" for v in estimated_speeds) + " |")
    speed_gate = 0.30
    below = {split: sum(r['practice100']['estimated_tail_rms_mm_s'] <= speed_gate
                        for r in results if r['split'] == split) for split in summary}
    training = read(RUN/'training.json')
    contents = f"""# P1 替代身体百倍规模地形加练

训练集共 **2,800 个 2 秒回合**，包含 **{audit['new_train_episodes']:,} 个独立地形的新增物理回合**和上一轮的 28 个训练回合。新增训练回合共执行 **{audit['native_train_steps']:,} 个** MuJoCo 原生物理步，实际完成 **{audit['updates']:,} 次**优化器更新。总回合数和更新次数均是上轮的 100 倍。

## 数据来源与训练

沿用原 35–65 µm 轻微起伏范围、规范身体和局部运动控制器。每张地形重新求解完整身体的碰撞、运动和承重，同步生成 5 ms 光学及足压样本。新增地形分摊给两个开发起步相位、12 条既有下行记录。替代身体继续读与相同下行记录配对的 226 针解码结果。指令和起步相位的数量受到现有记录限制。本轮沿用原指令时序。训练只更新动作预测网络，两路前足反馈参数与原权重逐值核对。

从 226 针首轮权重开始，分段扩大可采样数据。每次优化展开 14 步，混合测量足压和自身估计足压，保持平地样本在批次中占 8/64。使用固定种子和余弦学习率衰减，最终 8,000 次更新对完整语料采样。比较反映这套扩充数据与延长训练的总体结果。

## 留出回合

50 ms 和 200 ms 误差为给定外测足压时预测的 21 个身体标记的平均三维误差。自估足压以最初两帧为起点递推两秒，末段 RMS 按最后 200 ms 胸部平面速度计算。所有模型使用相同下行记录、同一光学参考和同一足压估计器。20 张新测试地形共享两个独立起步相位，每相位十张。

| 指标 | 单位 | 226 针首轮 | 上轮地形加练 | 本轮百倍加练 |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

| 新地形起步相位 | 教师净位移 mm | 226 针首轮 mm | 上轮地形加练 mm | 本轮百倍加练 mm |
|---|---:|---:|---:|---:|
{chr(10).join(walking)}

| 新地形起步相位 | 教师末速 RMS mm/s | 226 针首轮 mm/s | 上轮地形加练 mm/s | 本轮百倍加练 mm/s |
|---|---:|---:|---:|---:|
{chr(10).join(standing)}

以先前的 0.30 mm/s RMS 速度线核对，新权重在原平地 {below['flat']}/7、上轮起伏面 {below['old_terrain']}/4、本轮新起伏面 {below['new_terrain']}/20 达线。这是速度单项。教师的末速由物理步直接计算。11903 起步相位在这组有限时长的记录内依然有持续运动。逐回合结果在 evaluation.json。两秒递推使用归档下行。

## 复验

当前目录的 reproduce.sh 从已有收据和训练检查点续跑。最终权重和对应的 skeleton.json 在 run/，审计结果在 run/audit.json。权重 SHA256 为 {training['weights_sha256']}。
"""
    (RUN/'REPORT.zh-CN.md').write_text(contents)


if __name__ == '__main__':
    main()
