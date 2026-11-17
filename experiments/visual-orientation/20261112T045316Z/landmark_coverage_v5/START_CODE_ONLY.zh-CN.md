# 实际RGB姿态覆盖候选5：父执行说明

本子仅CODE_ONLY和标准库检查，没有数值/模型/数组/forward/Core/渲染/训练。本子已向父交付全部绝对入口；父已报告库生成bash85exit0、独立2573项审核和实际开图完成。真实训练尚待父运行结果，不能用代码检查替代实际成功。

## 唯一固定方案

- 当前范围 `/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/`。
- 库616实际双眼帧：4个训练几何×20保存实际姿态×7场景=560；2新V8/V9几何×4新姿态索引×7=56。20训练索引5799+400i(i0..19)，新验证5899/8499/11099/15199。
- 旧光学/完整71opaque身体/目标和两个闭合罩完全沿用冻结farv3算法。父真实RGB诊断报告unique_train_hashes340、unique_backgrounds20。重复空场像素和同轨迹邻近姿态不计独立新运动轨迹。实际720训练序列是80静态背景组合×9motif，不是720次独立物理回合。
- 新图像训练720、验证48、控制32，只存实际616帧ID与离线标签/权重/时钟，未把validation模型输出用于前期诊断。
- 不使用digitalmirror，初始化固定79dbcdcd38c1c3de94588bff9923b3758f82b1abbc63f017e499ece464340fee，不挑历史最优。相机时序、14参数架构、原损失/所有功能门与motor789f不变。
- freshAdam1e-4/fixed1000/full1920BPTT/batch4，save0/500/1000仅final1000；360pair按原uniform2of360无放回取样，全部720序列边际1/180。记录实际每pair取样次数和未取列表，不为漏取而追加步数。

## 代码检查

`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/check_codeonly.py` 已实际stdlibexit0，33项全部通过、无heavyimports。CODEONLY_CHECK.json保存源SHA与计数。检查已经运行，exclusive产物不能盲覆盖。

## 父统一CPU1训练命令

确认无其他heavy运行，真实库与独立审核、实际图像检查完成后：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/policy/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/policy/PIPELINE.log 2>&1
```

顺序：实际全9.6秒batch4 production/scan/all14梯度/两pair均值梯度/临时Adam/RSS≤1800MiB预检，以及config+pose唯一查询/720覆盖/权重/输入时序/库及诊断身份检查；冻结源与数据；1000正式更新；0/500/1000三个独立进程保存生产RGB重放；final48新验证+32控制一次；独立RGB重放与功能门复算。预检和checkpoint所用pairIDs[0,80]分别为方向pulse和反转，避免纯零目标数值测试。

JOBS.json保存阶段PID/命令/日志/退出码，CURRENT_STATE分开功能与完整性。每checkpoint的pair_coverage及TRAIN/RESULT.json报告pair_selection_counts/unseen_pair_ids/minmax，verify_saved从实际PCG64完整重建核验。未取pair不补训，valFAIL不挑500或加1000。

严格新loader：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_coverage_v5/policy/common.py`，load_checked(path,digest,require_final=True)，格式VISION_COVERAGE_POLICY_V5，仅final1000可验证/准入；最终文件policy/TRAIN/checkpoint_001000.pt，SHA待实际保存。

## 证据限制

数据覆盖与更新预算一起增加，不能称只隔离验证了过拟合唯一根因。验证新索引+几何仍来自同一条neutral身体轨迹，不是独立新动物/新运动轨迹。真实图像多样性本身不是策略能力。旧近域和远域所有失败保留；本scope不改阈值/网络/运行时控制，没有坐标输入、颜色质心steering、减bias、墙检测归零或记忆reset。

离线全部通过只允许父另立真实运动头部身体验证，不等于视觉导航、嗅觉融合、减速停驻、摄入、生物复眼或意识证据。任何实际执行错误先查已有stage日志，保留数据与失败，不盲重启已经完成的训练/物理。所有本scope执行源自READY停止修改。
