"""Build the paired report from verified entry-point results."""
import json
from pathlib import Path
from train import HERE, VELOCITIES, HELD


def ms(value):
    return '全窗未出现' if value is None else f'{value:.1f}'


def first_failure(detail):
    if detail is None:
        return '通过全窗'
    checks = [
        (detail['total'] < .8, '总足载低于 0.8 倍体重'),
        (detail['total'] > 1.2, '总足载高于 1.2 倍体重'),
        (detail['loaded'] < 3, '承重足少于三只'),
        (detail['nonfoot'] > 0, '非足承重'),
        (detail['com_ratio'] < .9, '质心高度低于初始的 90%'),
        (detail['upright'] < .95, '躯干直立分量低于 0.95'),
        (detail['depth_mm'] < -.012, '足地穿透超过 0.012 mm'),
    ]
    return '、'.join(label for failed, label in checks if failed)


def main():
    data=json.loads((HERE/'ACCEPTANCE.json').read_text())
    cases=data['cases']
    frozen=data['frozen']
    train=data['training']
    extra=frozen['parallel_training']
    rows=[]
    for i,v in enumerate((*VELOCITIES,*HELD)):
        before,after=cases[f'parent_{i}'],cases[f'new_{i}']
        rows.append(f"| {'开发' if i<3 else '新留出'} | {v:+.3f} | {ms(before['first_stance_violation_ms'])} → {ms(after['first_stance_violation_ms'])} | {ms(before['first_nonfoot_ms'])} → {ms(after['first_nonfoot_ms'])} | {before['post_tail_feet']:.3f} → {after['post_tail_feet']:.3f} | {first_failure(after['first_failure_detail'])} |")
    dev=sum(cases[f'new_{i}']['post_entire_stand'] for i in range(3))
    hold=sum(cases[f'new_{i}']['post_entire_stand'] for i in range(3,6))
    digest=frozen['sha256']
    bad=cases['new_0']['first_failure_detail']
    worst=[min(cases[f'{label}_{i}']['first_stance_violation_ms'] or 250 for i in range(3,6)) for label in ('parent','new')]
    summary=f"""# 独立 BANC 站立十倍时长续训

从用户的分批 CMA-ES 站立权重出发，用相同的 1 ms 地面、90 肌肉身体、站姿和 250 + 250 ms 准备流程进行闭环续训。父版六段优化日志计时合计 {data['budget']['previous_total_s']:.2f} 秒。本次主训练实际累计 {train['training_seconds']:.2f} 秒，为 {data['time_multiplier']:.3f} 倍。训练后的完整 250 ms 站立通过数为开发 {dev}/3、新留出 {hold}/3。结果为软件研究候选。

| 用途 | 准备初速 mm/s | 首次越门 ms，父版 → 续训 | 首次非足承重 ms，父版 → 续训 | 末 100 ms 平均足载，倍体重 | 续训版首次越门触发项 |
| --- | ---: | ---: | ---: | ---: | --- |
"""+'\n'.join(rows)+f"""

新留出最差首次越门从 {worst[0]:.1f} 到 {worst[1]:.1f} ms。开发使用名义和 ±0.03 mm/s 三种初速，新增三组只在选定权重后运行。同一身体准备状态用于父子配对。在模型规定范围内连续承重才记全窗通过，非足承重首发只是一项独立指标。
"""
    if bad:
        summary+=f"\n续训名义状态的首个越门拍，总足载 {bad['total']:.4f} 倍体重，承重足 {bad['loaded']} 只，质心高度为初始的 {bad['com_ratio']:.4f}，直立分量 {bad['upright']:.8f}，足面最低距离 {bad['depth_mm']:.5f} mm。\n"
    summary+=f"""
## 权重、计时和复验

主搜索共 {train['updates']} 次实际 CMA 更新、{train['candidates']:,} 个候选、{train['episodes']:,} 个开发闭环回合。开发轨迹可在出现非足承重或质心低于 88% 后停止。两个较低优先级进程同期探索负初速较强的候选，完成 {extra['updates']} 次更新、{extra['candidates']:,} 个候选和 {extra['episodes']:,} 个闭环回合。并行分支用时与主训练重叠，十倍计时只计主训练。每次分块固定其他字段，记录候选向量、评分、优化器状态与选中模型。用户原版的计时来自日志创建到末次输出，是近似墙钟口径。新训练计时采用各已完成优化块的单调时钟，完整验收另计。

用开发条件锁定的分支为 {frozen['winner']}，交付权重 SELECTED.npz，SHA-256 为 `{digest}`。父版的 321,164 条已有边原始符号与逐边 ±8% 边界保留。相对父检查点，本次变化数为 {json.dumps(data['changed_from_parent'],ensure_ascii=False)}。身体、地面、传导时延、发放阈值、默认权重及教师控制保持原配置。运动切断对照首次越门为 {ms(cases['cut']['first_stance_violation_ms'])} ms。所有正式回合在撤手后均关闭教师。闭环回合逐拍下发上一神经区间的运动读出，切断对照逐拍为零。新进程名义重放的 {data['replay_arrays']} 个数组逐值一致。

项目根目录运行命令如下。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python -B experiments/banc-muscle/run_stance.py /tmp/banc-10x-check --parameters {HERE/'SELECTED.npz'}
```

原始结果、首次越门各门数据及计时见 ACCEPTANCE.json。新留出幅度都在支撑准备期开始前施加，本轮未测试撤手后突然施扰。自然果蝇的逐肌束支配仍未建立。
"""
    (HERE/'REPORT.zh-CN.md').write_text(summary)


if __name__=='__main__':
    main()
