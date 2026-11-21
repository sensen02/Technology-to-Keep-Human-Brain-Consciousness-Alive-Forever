# 平地颈部干预执行与复验

- P1、S1/S3/S5，新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T031132Z-flat-neck-assay/`。父运行 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T020425Z-front-touch-timing-repair/`。先锁定 PROTOCOL.zh-CN.md和schedule.py。四独立身体、同一站姿、同一PFL3刺激、正常本体反馈、无外力或教师动作。记录项目限定并保留失败的方向及恢复行为门。
- `bash drive.sh derive_assay.py` 生成归档派生件，仅放宽传导阻断的数组为按个体端口寻址。支持旧全批次掩码。固定传播时延、模型、受体、局部膜模型和身体权重身份同父；实际依赖源哈希见各INPUTS及RESULT。Python3.12.14、Torch2.5.1 CPU、OMP/OPENBLAS/MKL各1线程，MuJoCo EGL。
- `bash drive.sh run_assay.py` 分别以10901/11311/11701写入FORMAL，各四个回合；11701另写REPLAY，新进程。四次运行各20000批步，exit0，核原始警告及有限性门。计数以FINAL_ASSAY.json为准，共16身体320000原生步、脑80000批320000 lane tick。两历史身体MJB及输出同字节或逐值相同。 `audit_queue.py` 1500批3000 lane tick单独核在途取消及对侧隔离，exit0；实时颈含探针总81500批323000 lane tick。PPO/GAE不适用，优化更新0。
- `assess_assay.py`、`render_assay.py` exit0。新进程逐数组54精确。12正式行为回合推进、安全、撤刺激后站稳均通过。预定行为方向与恢复效果各自报告。三链路存在但定向主终点未过。
- 已实际打开脉冲图与四时刻身体帧。MATCHED_ASSAY.mp4 1408×224、200帧、25fps、8秒，实际2秒仿真以0.25倍播放。ffprobe和FFmpeg全解码exit0。录像取已保存的11701实际身体状态，去除渲染反光。复核源及轨迹哈希、桌面同字节。后续拆分初始步态相位和外部刺激种子的影响。
