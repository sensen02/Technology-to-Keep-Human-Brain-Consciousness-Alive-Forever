# 固定权重：观察域×首AN初值四格诊断

本子仅CODE_ONLY，13项标准库静态检查实际通过exit0，0Torch／NumPy／arrays／模型forward／训练／Core／物理。第一次检查的字符串次数错误已在纯静态阶段修为AST直接核四格循环，并保留CODEONLY_FIRST_ATTEMPT.json；模型诊断算法未变。

## 父唯一CPU1实际命令

`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/feedback_domain_diagnostic/diagnose.py --execute-parent > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T225331Z/feedback_domain_diagnostic/RUN.log 2>&1`

本子未执行该数值命令。运行输出INPUTS_BEFORE、OUTPUTS.npz、RESULT.json均exclusive，不覆盖已完成诊断或body/calibration旧证据。

## 四格与实际可比性

固定最终eed1...权重、脑初始h0、首DNhold0和实际5ms单决策延迟，四格：原归档V_empty X＋首AN0；同归档X＋真实warmAN；新P_empty X＋首AN0；新X＋真实warmAN。warmAN只第一次脑输入改变，以原neural_decisions row1159编码／row1160held及capture5795ms/release5797ms独立核，不清后续AN流、不改变身体记录。

archivedcold必须逐位复原原EVALUATION/V_empty turn、DN、AN、brainhidden；actualwarm必须逐位复原真实body1920ANN turn、DN、AN、brainhidden和两heldports。若任何匹配失败保存实际输出和失败flags，不能解释矩阵结果或重跑物理。

报告窗口0—.1、.1—.6、0—.6、全9.6、固定4.12—4.62恢复描述窗，并输出初值／观察域各比较差异。该恢复窗只是固定描述，不再引用视觉3.62发布或造新pass门。参数tensor和所有输入／源SHA前后核对不变。

原四训练111维均值／std统计及floor1e−6数量、actualsettle／teacher／ANN z值极值／p95／超trainminmax比例／top15原始索引、ANabs>.95饱和比例为描述统计，不能当新通过门或以此删失败数据。

固定保存的观察不随反事实动作变化，所以四格重放不能证明身体真的会走成这些反事实轨迹，也不能直接证明双稳态。结果用于区分首AN初值敏感、离线外生观察到在线分布变化及二者交互，之后才有针对性地预登记新训练；不得在原runtime加归零规则或减偏置掩盖问题。
