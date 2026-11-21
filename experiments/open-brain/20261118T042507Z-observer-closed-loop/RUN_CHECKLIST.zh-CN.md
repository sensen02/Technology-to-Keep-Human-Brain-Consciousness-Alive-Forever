# 翻译层闭环检查表

- P1/S1/S3/S5，目录 /home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T042507Z-observer-closed-loop/ 。父台架 /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T031132Z-flat-neck-assay/ 和本工作树的20261118T035424Z-adaptive-neck。先声明正式和开环压力协议。冻结输入及权重，同站姿自行反馈、无外力、无根写入。
- Python3.12.14、Torch2.5.1CPU、OMP/OPENBLAS/MKL单线程、MuJoCo归档环境。derive.py从已通过的父台架派生。drive.sh run_live.py 的11701正式、11903正式、11903独立REPLAY各20000拍，墙钟至完成约280、291、294秒，三次exit0。输入及生成件SHA见INPUTS/RESULT。
- drive.sh check_wrapper.py、shadow_stress.py、assess_live.py、plot_live.py均exit0。冻结批预测及流式算法四组20000拍逐值匹配。实时17DN和2AN与纯颈后编码审计。共12身体240000原生步，脑/颈各60000批步240000 lane tick，优化0。新进程57数组精确，11701正常关闭逐值匹配旧基线。未跑飞行、P2或真实实验。
- 正常和漂移闭环均0/2达到完整门，保留11903旧接口首次停稳失败和开启后的其他差异。实际打开 CLOSED_LOOP.png，图取完整轨迹与读数；无视频。下一最小步处理健康条件下的入脑相位差。
