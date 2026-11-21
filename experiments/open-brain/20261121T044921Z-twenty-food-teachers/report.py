"""Summarize measured food collection and the paired replay of the new model."""
from collections import Counter
from acquire_cohort import HERE,digest,read


def main():
    manifest=read(HERE/'cohort.json')
    accepted=manifest['accepted']
    training=read(HERE/'twenty'/'training.json')
    identity=read(HERE/'twenty'/'cohort-identity.json')
    evaluation=read(HERE/'twenty'/'closed-loop.json')
    assert len(accepted)==18 and len(set(identity['seeds']))==20
    assert identity['cohort_sha256']==digest(HERE/'cohort.json')
    assert training['weights_sha256']==digest(HERE/'twenty'/'weights.pt')
    assert training['train_seeds']==identity['seeds']
    band=Counter(r['band'] for r in accepted)
    side=Counter(r['side'] for r in accepted)
    first=manifest['original_teacher_seeds']
    capture=sum(r['accepted_frames'] if 'accepted_frames' in r else r['observed_frames'] for r in accepted)
    table=[]
    for i,row in enumerate(accepted,1):
        x,y=row['goal_mm']
        table.append(f'| {i} | {row["seed"]} | {x:.2f} | {y:.2f} | {row["initial_yaw_deg"]:.1f} | '
                     f'{row["intake"]:.5f} | {row["first_bite_s"]:.3f} | {row["observed_frames"]} |')
    results=[]
    for sample in evaluation:
        turn='加练前' if sample['trial'].endswith('-previous') else '20 条示范加练'
        results.append(f'| {sample["seed"]} | {turn} | {sample["intake"]:.5f} | '
                       f'{sample["contact_time_s"]:.3f} | {sample["tail_rms_mm_s"]:.3f} | '
                       f'{"通过" if sample["full_meal_pass"] else "未通过"} |')
    text=f"""# P1 平地觅食 20 条合格教师示范

新增 18 条摄入达标的独立物理回合，与最初 {first[0]}、{first[1]} 两条成功示范组成 20 条训练回合。共尝试 {len(manifest['attempted'])} 个新场景，失败记录单独归档。近、中、远距离分别收录 {band['near']}、{band['middle']}、{band['far']} 条，食源在左、右两侧分别 {side[1]}、{side[-1]} 条。新增样本提供 {capture:,} 帧外测姿态、足压和同步的 226 针下行读数。

使用原工程食量单位。每条成功教师必须摄入至少 0.18，库存同步下降，正摄入窗发生实际口器接触，全程无跌倒且达到原餐后站稳判据。教师采用同一冻结 Eon 和头侧嗅觉策略。场景按食源位置、初始朝向和气味强度变化，神经随机种子沿用 29。采集串行并设置 4200 MiB 进程组上限。

| 序号 | 场景种子 | 食源 x mm | 食源 y mm | 起步偏航 ° | 摄入 | 首口 s | 观测帧 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

从上一轮长递推权重开始，在这 20 条示范上继续做 {training['updates']:,} 次动作优化器更新，保持 500 ms 递推训练和通用步态数据回放。前足反馈与脑权重冻结。闭环维持原 226 针电流写入与电压分选。88224 和 88225 在本轮数据之外，88207 是训练场景。旧权重使用同场景既有的 6.4 秒闭环档案。

| 场景 | 权重 | 摄入 | 口器累计接触 s | 末段速度 RMS mm/s | 完整通过 |
|---|---|---:|---:|---:|---|
{chr(10).join(results)}

队列、各回合输入、实际命令和文件校验值在 cohort.json，训练和模型权重校验值在 twenty/training.json。完整通过同时核对摄入、姿态和最后 140 ms 四足站稳。
"""
    (HERE/'REPORT.zh-CN.md').write_text(text)


if __name__=='__main__':
    main()
