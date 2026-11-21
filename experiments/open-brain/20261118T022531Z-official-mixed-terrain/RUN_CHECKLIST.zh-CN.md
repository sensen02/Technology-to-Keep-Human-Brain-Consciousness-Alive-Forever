# 官方混合地形实验记录

- 本轮P1、S1/S3/S5，目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T022531Z-official-mixed-terrain/`。父实验为其同级的20261118T020425Z-front-touch-timing-repair和20261118T002600Z-ascending-behavior-comparison。PROTOCOL.zh-CN.md先锁定相位、场地、指标、控制和匹配方案。STOCK_PROTOCOL.zh-CN.md记录额外官方原装能力对照。
- 归档 Python3.12.14、Torch2.5.1 CPU、FlyGym2.1.0、MuJoCo EGL、OMP/OPENBLAS/MKL各1线程。初始位置移动一次，后续自由根无写入；身体73 qpos、72 qvel、42有限位置执行及6足垫，相同编译模型哈希；无外力、辅助或PPO。阶梯地面由官方类直接编译，ARENA.json登记291几何体和16005显式接触对，固定资产和实际模块哈希见INPUTS/RESULT。首个全足平区几何断言不合适，旧日志保留；改为原站姿前中足在平区、后足在同高桥面，随后smoke 300步exit0并核法向足力统计与旧采样一致。
- 首次三实时进程的未限定同名模块路径导致加载出错，未进入积分、未改旧结果，import_failure留存；改用显式归档时序模块后执行。采用 `bash drive.sh derive.py`，`bash drive.sh run_live.py 10901 FORMAL/seed10901`、`bash drive.sh run_live.py 11311 FORMAL/seed11311`、`bash drive.sh run_live.py 10901 REPLAY/seed10901`。四体各20000步，三个批次退出0，活跃墙钟约380秒/批（不含末盘写检查）。受控相位开始、结束与历史相同，刺激开启.22至1.5秒，随后自由站稳。
- `bash drive.sh teacher.py` 每相位分别运行forward_probe和native_command repair_command。两个固定命令回合和四个颈后命令重放全部exit0，后者记录数组与实时颈后命令逐值相同，并用与实时身体相同编译模型。独立 `bash drive.sh stock_official.py` exit0，另保存真实原装教程20000步与500步预热。程序退出不表示穿越合格。
- `assess_arena.py`、`height_context.py`、`render_arena.py` exit0。核对17DN及两AN实际延迟、修复相位输入、原背景刺激、数据与源码哈希；独立闭环重放61数组精确。18同体回合360000步、60000颈和脑批步各240000 lane tick；另加smoke 300、原装教程20500步。录像1600×448、200帧、25fps、8秒，为两秒模拟的0.25倍。ffprobe与FFmpeg全解码exit0，已实际打开时间戳帧图与曲线，检查起步、地面接触和停步的完整身体。未声称逐帧人工观看。
- 修复1/2实际越过并停稳，仍有关节越限，两组整场验收0/2。受控参考过缝1/2，原脑与旧翻译均0/2。官方同体同原脑命令过缝2/2但末速度及安全门未过。原装教程只是能力对照。没有完整D1–D8、新功能训练、视觉、飞行或P2更改。研究样本与图表置桌面，下一最小动作为用穿越片段做同命令身体纠错，首查前跗节碰块时姿态与缝边承重切换。
