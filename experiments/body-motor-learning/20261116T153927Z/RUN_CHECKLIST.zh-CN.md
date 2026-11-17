# 小范围地形适应（S2/S5）

本轮在人工搭建大脑项目的身体台架实施单个低凸台、浅缺口和缓坡。运行目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z`。各回合沿用冻结 71 mesh 完整身体、原三球足、42 路有限 PD、5 ms 决策与 12.5 μs 原生步。地形用真实静态碰撞几何。保留自由根，无额外外力、附着和回合内重置。

- 低凸台记录腿段反向触力、抬足、足垫是否在台面或远侧承重、实际迈过的具名足、尾足和身体的越界，以及撤令后的末窗站稳。与同几何关闭修正和同路径平地对照比较，使用实际全程或足够阶段的原生接触。
- 浅缺口识别真实落脚未获支撑，配对相同位置填平、相同缺口关闭收腿。区分在近侧止步、足端到远侧、身体通过、其他脚承重及主动替换承重。局部安全制动与重新给推进意图独立保存边界。覆盖从实际初态到停稳的连续轨迹。
- 缓坡记录实际倾角、上坡进度、足底与坡面真实承重、躯干高度变化、末端停稳。对照计算支撑、足相对表面的切向滑动、坡面相对与原全局高度。旧平地门同时报告，坡面物理口径单列。
- 两反射的组合按同一运行入口和可验收动作来源判定。能用同一组合入口在不同场景分别触发才报告组合。实际触力和足载近邻判别优先，教师中的几何、相位或计时若存在，登记其控制范围。以真实物理复验为准，训练、闭环和证据身份可分别复查。
- 采用早期诊断与给定尺度的地点后重新起步、稳态行走、尾部停稳，记录时间窗和是否所有相关足／躯干完全过去。原 0.50 滑移等工程门不追改。完整 D1–D8 和上层/颈闭环按原范围单独验收。保留命令、退出码、失败、同次视频和帧数。

## 运行记录

物理解释器为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，MuJoCo 3.9，`MUJOCO_GL=egl`，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。各回合的输入哈希、有限执行器身份、初态和地形见各自 `RUN_IDENTITY.json`。进程退出码和物理结果门分别记录。

- 低凸台。`block/drive.sh` 跑两条短探针、实际低台教师、无低台几何先验的对照、平地和两次独立重放，`block/drive_tactile.sh` 跑同一触力试验的开启/关闭/重复。扩展审计与同次渲染退出 0。首个平地和首个扩展包装器收集 143，已写完整原生记录，独立重放退出 0 并逐字节相同。 [低凸台命令与退出记录](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/block/RUN_CHECKLIST.zh-CN.md)
- 浅缺口。`gap/drive_safe.sh clear CLEAR_FULL --native`、`gap/drive_safe.sh clear CLEAR_REPEAT --native`、远沿持足、关闭反射、只关右前足和填平配对均退出 0。`gap/drive_safe.sh clear CLEAR_TRANSIT --long-ms 45600 --native` 退出 0，原生长程前进门失败。初版脚本写完积分后，因报告缺少 `walk_ms` 退出 1，修复后独立重跑，旧产物保留。修正地板后 `gap/drive_hybrid.sh course HYBRID_COURSE --native`、`gap/drive_hybrid.sh course HYBRID_COURSE_REPEAT --native`、同布局填平、关闭、局部审核及 `gap/audit_transit.py` 退出 0，通过独立复验。根网格复核第一次导入模块同名冲突退出 1，第二次因重复解压主动停止，改为预载 `qpos` 后用 `block/certify.py` 的原网格评估函数退出 0，未增加任何求解步。全程录像验证 773 帧并核对桌面字节。 [缺口命令与退出记录](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/gap/REPORT.zh-CN.md)
- 组合入口。`learn/compose.py` 和同目录两个地形接入脚本均从冻结条件重新积分。两条旧反射关闭对照、远沿复踏整段和实心凸台局部触力均退出 0，实际资格见各 JSON。 [控制入口与实际切换](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/REPORT.zh-CN.md)
- 缓坡。`learn/ramp_course.py` 实物三段盒配对退出 0，过坡联合门失败。`learn/ramp_hfield.py` 整体替换以同一 225 ms 原生接触退出 0，身体未起步。恢复原平面后，`learn/ramp_hfield.py --over-original-plane` 短坡、40 秒走动长坡和同长配平原生退出 0，`learn/audit_ramp.py` 和最终回放退出 0。两条同长全程各 3,336,000 步，坡臂原生走停通过，末端质心抬升指标未通过。 [缓坡实际命令和测量](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T153927Z/learn/RUN_CHECKLIST.zh-CN.md)
- 桌面初次复制调用了未安装的 `python`，退出 127，随后 `python3` 复制、逐字节比较和 `ffprobe -count_frames` 均退出 0。低凸台全程 875 帧，凸台触力耐受 280 帧，浅缺口远沿复踏 636 帧，组合入口绊脚 669 帧。实际查看了同次关键帧，桌面归档 `/home/sensen/Desktop/engine修复验收/小范围地形适应-20261116T153927Z`。
- 缓坡 41.7 秒实际回合回放全解码 1,200 帧，实际查看起点、越坡和停步画面，桌面视频及关键帧与原文件逐字节一致。
