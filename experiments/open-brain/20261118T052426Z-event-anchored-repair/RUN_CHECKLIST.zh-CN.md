# 相位保护修复验收检查表

- P1及S1/S3/S5，目录 /home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T052426Z-event-anchored-repair/ 。父试验为同树的20261118T042507Z-observer-closed-loop、20261118T050548Z-phase-localization。预登记两相位四组以及11701独立复验，原物理和神经条件固定，判据未放宽。
- Python3.12.14、Torch2.5.1CPU、OMP/OPENBLAS/MKL各1线程，MuJoCo归档EGL环境。python3 derive.py 生成运行入口。drive.sh run_live.py 11701 FORMAL/seed11701、11903 FORMAL/seed11903、11701 REPLAY/seed11701 三命令各20000批步，实测墙钟约283、293、283秒后完成步进，退出0。体12回合240000步，脑及颈各60000批240000lane tick，优化0。完整输入、身体MJB和源码SHA见INPUTS/RESULT。
- drive.sh check_anchor.py、check_trust_transition.py、assess.py、plot.py均exit0。前两项读取实算BANC到达重放编码，未新增物理切断或延迟积分。两旧基线逐值相同，按2AN/17DN逐拍验证传输及原相位编码，独立新进程60数组逐值相同。11903原末停稳失败保留。
- EVENT_ANCHORED_REPAIR.png由实际合成与闭环结果生成并已打开。未生成视频，未跑飞行、P2、PPO/GAE或实体实验。先前失败的开环及闭环基准不覆盖。下一步若事件时戳本身被污染，需要额外证据与单独验收。
