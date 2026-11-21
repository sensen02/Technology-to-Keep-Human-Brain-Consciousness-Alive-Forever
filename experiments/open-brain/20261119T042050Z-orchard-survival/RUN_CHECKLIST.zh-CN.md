# P1 / S5 果园生存实验记录

- 输入为种子821果园、站姿快照、Eon种子29、同一 FlyBody 及已登记神经元来源的 BANC 子图。模式为 FRESH、EMPTY_GROUND、GUIDED。原果园与先前行为身份资料分别位于相邻的 20261119T012530Z 和 20261119T021500Z 实验目录。
- 对照条件及成功判据登记于 PROTOCOL.zh-CN.md。能源参数为工程量，口器糖和触觉保持原接口。无模型或权重更新，无额外外力，无在回合中强制修改实际位姿。指导组的目标坐标只有工程可达性对照访问，另外两组只使用身体传感与像素候选。
- Python 3.12.14、MuJoCo 3.1.6、EGL。本次重型仿真串行，TF、OMP、BLAS 各1线程，EGL软件渲染2线程，CUDA关闭。
- 正式三组命令 ./REPLAY.sh 新绝对目录，完成退出0。正式验算原始门和逐步能量明细见 BASELINE/ASSESSMENT.json 及各子目录中的原始 MEAL、TRACE、EYES 和 NEURAL。对应固定4秒旧自主运行的24项逐值核验见 FOUR_SECOND_PREFIX_REPLAY.json。
- 诊断批处理包装脚本退出143。曲线引导组已写全原始数据，独立验算退出0并通过，地面制动随后单独运行5秒退出0。命令为上一场景目录 run_env.sh 调用本目录 survival.py，后加 GUIDED 输出目录 --duration 4 --curved-guide 或 FRESH 输出目录 --duration 5 --approach-brake。两组一起的独立验算见 DIAGNOSTICS/ASSESSMENT.json。
- 外枝果轨迹参考改动可在同一原始 DNp11 发放、颈后到达与读出下独立比较。两条轨迹均没有触发落果。地面停步诊断代谢记录可在相同配置长回合中复现。两餐数值阳性对照以固定两段口器接触驱动，无物理或神经仿真，见 LEDGER_PROBE.json。
- 地面制动长回合调用同一 run_env.sh 与 survival.py，参数 FRESH 新输出目录 --duration 12 --approach-brake，退出0。前五秒13项逐值复验、独立代谢核对、postmeal.py、figure.py、frames.py、源码哈希及 git diff --check 退出0。图和实际位姿帧人工打开。
- 六个物理回合合计705200原生步、352600神经拍。耗能台架提前归零的回合按20ms块结束。既有体、脑和颈权重更新0，P2不在此次范围。无空中停果、第二餐或长期生存结论。
