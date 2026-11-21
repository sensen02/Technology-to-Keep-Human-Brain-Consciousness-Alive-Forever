"""Render the food-practice report after the paired autonomous evaluations."""

from practice import OUT,HELD,demo,read_json,digest


def main():
    variants=('expanded','long-rollout')
    tables=[read_json(OUT/v/'closed-loop.json') for v in variants]
    expanded=tables[0]
    long=tables[1]
    train=[read_json(OUT/v/'training.json') for v in variants]
    for v,r in zip(variants,train):
        assert digest(OUT/v/'weights.pt')==r['weights_sha256']
    fmt=lambda n:f'{n:.5f}'
    rows=[]
    for ref in (r for r in expanded if r['trial'].endswith('-previous')):
        seed=ref['seed']
        group=(ref,next(r for r in expanded if r['seed']==seed and r['trial'].endswith('-expanded')),
                   next(r for r in long if r['seed']==seed and r['trial'].endswith('-long-rollout')))
        for name,r in zip(('加练前','扩数加练','长递推加练'),group):
            rows.append(f'| {seed} | {name} | {fmt(r["teacher_intake"])} | {fmt(r["intake"])} | '
                        f'{r["closest_oral_center_mm"]:.2f} | {r["contact_time_s"]:.3f} | '
                        f'{r["first_tip_s"] if r["first_tip_s"] is not None else "无"} | '
                        f'{r["tail_rms_mm_s"]:.2f} | {"通过" if r["full_meal_pass"] else "未通过"} |')
    rows='\n'.join(rows)
    final=[r for r in long if r['trial'].endswith('-long-rollout')]
    n_ate=sum(r['meal_amount_pass'] for r in final)
    full=sum(r['full_meal_pass'] for r in final)
    flats=read_json(OUT/'long-rollout'/'flat-regression.json')
    flat=flats[0]
    seeds=set(train[1]['train_seeds'])|set(HELD)
    sensors=[]
    for s in sorted(seeds):
        item=read_json(demo(s)/'RESULT.json')
        sensors.append(f'| {s} | {"留出" if s in HELD else "训练"} | {item["duration_s"]:.2f} | '
                       f'{fmt(item["intake"])} | {item["side"]} |')
    sensors='\n'.join(sensors)
    text=f"""# P1 替代身体平地觅食加练

长递推版在 {len(final)} 个电极闭环对照中摄入达标 {n_ate} 个，完整进食并站稳通过 {full} 个。

本轮延续冻结的头侧嗅觉策略和 Eon，训练替代身体动作网络及足压估计器。使用 226 针仿真的原电流写入与混合电压读出。学生只从起步的两帧外测标记初始化，随后依靠预测姿态和自估足压运行，脑实时发令。食物位置用于环境气味生成、虚拟口端接触和训练采样分层。

教师补跑先遇到主机 OOM。恢复后采用单进程顺序发起物理回合，在独立进程组设置 4200 MiB 限额，纯嗅觉教师关闭了诊断用头侧 RGB 渲染。八机位外部体标和口器图像仍用于姿态与足压同步。物理收据与有完整地板读数的分支才进入训练。本轮新增 8 个完整物理回合，其中 6 个训练、2 个留出。原始失败回合作为监督或对照按预先划分保留。

| 场景 | 用途 | 教师实际时长 s | 教师摄入 | 食源侧 |
|---|---|---:|---:|---:|
{sensors}

## 训练

扩数加练从上一轮混合足压权重出发，用 6 个场景做 {train[0]['updates']:,} 次动作更新；长递推从扩数权重出发，用 8 个场景再做 {train[1]['updates']:,} 次动作更新。每段另做 1,400 次足压估计器更新。两段都交替使用实测与自算足压，并穿插 108 个既有平地或微起伏回合以约束走路退步。每回合训练监督来自外测标记及测力地板，实际入脑的感觉仍由原电极链路决定。训练选择增加近食源、转向、早期起步窗口。扩数版展开最多 200 ms，长递推版展开最多 500 ms 并强化起步胸部朝向和逐足承重误差。另有 600、950、1,300 次更新的开发阶段独立归档，未计入两段主结果。

## 在线闭环验收

每个学生回合运行 6.4 s。摄入门槛采用原 0.18 工程单位，逐帧核对正摄入一定有合格口器接触，并与食物库存扣减一致。完整验收同时要求全程胸部直立指标未跌破 0.55，且最后 140 ms 至少四足承重、胸部直立超过 0.94、最大平移速度不超过 0.8 mm/s。直立与承重由替代身体姿态和足压读数计算，属于仿真判据。88207 用于训练，88224 与 88225 的完整回合留出。

| 场景 | 权重 | 教师摄入 | 学生摄入 | 口端最近距离 mm | 接触累计 s | 首次倾倒 s | 末 200 ms 速度 RMS mm/s | 完整通过 |
|---|---|---:|---:|---:|---:|---:|---:|---|
{rows}

旧规范平地的 11701 走路回归中，百倍父权重、觅食加练前权重、长递推权重的净位移分别为 {flat['parent']['estimated_net_mm']:.2f}、{flat['previous']['estimated_net_mm']:.2f}、{flat['current']['estimated_net_mm']:.2f} mm。长递推的末段速度为 {flat['current']['estimated_tail_rms_mm_s']:.3f} mm/s，父权重为 {flat['parent']['estimated_tail_rms_mm_s']:.3f} mm/s。另一个相位与留出定指令预测见 long-rollout 下的回归文件。

原电极场和注册范围内的反向刺激未重标定。每个运行的脑端到达和编码数量由保存的脉冲数组独立核对。未登记神经活动、实物光学和地板仍属于平台本身的范围限制。实跑命令、权重哈希、逐场结果和轨迹图分别见收集目录、扩数版和长递推版。
"""
    (OUT/'REPORT.zh-CN.md').write_text(text)


if __name__=='__main__':main()
