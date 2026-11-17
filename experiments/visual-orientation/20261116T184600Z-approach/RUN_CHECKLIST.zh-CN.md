# 视觉趋近、调速与转向（P1/P2、S2/S5）

首课冻结标准平地身体、姿态、有限 PD、12.5 μs 原生步及 5 ms 身体决策。操作顺序是视觉转向、撤销转向并站稳交接、快走、依据视觉距离减速、停稳。视图 20 ms 发布，实际交付沿用 2 ms 工程参数。双 DNa02 的获准来源与旧视觉编解码、双 DNg97 的来源及工程前进读出组成新独立 DN 通路。当前控制按原地转与调速走交替使用身体原语。虚拟红标在这课使用固定直径，距离读出仅覆盖该尺寸及留出条件。

- approach_runtime.py 生成来源绑定，连接和原始发布图固定。approach_vision.py 采集 1,600 组实际渲染眼图，用 1,380 组训练，220 组留出测方位、距离和出现。已保存权重及数据指纹。
- 主课 approach.py LEFT --angle=20 --shift=.12 与 RIGHT --angle=-22 --shift=-.30，加同左试验切断 DN 与空场。LEFT_REPEAT 按同一输入从原始起点重积分，身体、原生力、眼图、消息逐档摘要比较。按台架已有原生滑移 0.50、六足周期及站稳门额外核查，保留早期功能通过与滑移未过记录。
- 课任务门至少靠近 .45 原生长度，实际转角方向正确且达到初始偏角 45%，末窗方位偏差小于 15°，慢段实测均速比快段至少低 .004 原生长度/s，最后通过原站稳。颜色存在、方位和距离由双眼输入产生，身体通过颈后实际收到的前进和偏航指令执行。原生跨测器记录每个求解步的接触与力。
- 障碍物候选为实际实体绿盒。首次裸全局几何与过近夹具退出 1，日志保留；其中后续诊断明确原始近端与口器在第一帧接触。随后增加不接触初态的凸起夹具位置，并按字段扩展全身外力数组，保持冻结姿态和物理步。另测只刹停、无障碍响应和闭环视觉触发的绕行。
- 归档环境 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python、MuJoCo，BLAS/OMP/MKL 单线程、EGL。
- `drive.sh` 下的光学训练、LEFT／RIGHT／CUT／BLANK、LEFT_REPEAT、SETTLE_60／SETTLE_83、PATH_QUALITY 审计均正常完成。子任务级成功与原生行走门分列。延后减速的首次左右进程退出 143，目录保留启动身份。只复跑左侧，`./drive.sh late_brake.py LATER_LEFT_RETRY --near=2.32` 与 `./drive.sh audit_paths.py LATE_QUALITY.json LATER_LEFT_RETRY` 均退出 0，功能门通过，原生行走门失败。右侧使用原参数的通过试验，延后减速右侧暂无复跑。
- `./drive.sh obstacle.py FINAL_SAFE` 退出 0，`./drive.sh obstacle.py FINAL_BLIND --blind` 与 `./drive.sh detour.py FINAL_DETOUR` 因正式碰撞门退出 1，均完成原生积分及归档。无响应组接触口器，绕行组初次侧转接触右前足垫。相邻姿态几何回查保留子步原生力时间，见 `OBSTACLE_CONTACT_DIAGNOSIS.json`。
- `./drive.sh render_approach.py` 回放原始归档姿态及双眼，无新增物理步。left／right／safe／blind／detour 视频用 `ffprobe -count_frames` 完整解码，依次为 292、343、42、289、703 帧。已实际查看各场景关键帧。
