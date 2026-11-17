# 长历史气味行为修复候选：运行检查表

对应S4/S5；父独占调度全部实际计算，本子仅CODEONLY准备。现阶段尚未生成数组、导入Torch/NumPy/物理后端或前向/训练。CODEONLY检查不代表真实梯度或行为通过。

## A 跑前已固定

- PREREG先于代码，24train/12validation×1920steps/9.6s；6motif×4训练/2验证时序强度组合，不使用旧AB作为盲测。
- parent sensorbfd6严格load，same2→32GRU→turn，全部8参数可训/alpha不变；motor789f rawhash核验但不导入运动模型，不改运动权重/历史FAIL。
- 单candidate freshAdam1e-4/fixed1000/batch4，保存initial/500/1000，只有final1000一次验证；不选500、不加步/自动更改lr。
- 目标仅离线训练，online纯2受体输入+32连续状态，禁止runtime ifodor/deadzone/reset/gating。latent辅助损失只训练，网络输出仍tanh(latent)。
- 新验证先分项noodor/equal/on/reversal/off RMS逐条判断，不用总MSE掩盖off失败。未验证原生物有效性、身体刺激、导航或摄入。
- 必须父资源放行后单CPU1顺序执行预检/训练/新进程保存审核/验证/验证重放；没有PPO，不写无关PPO测试。

## B 实际运行待执行

父精确pipeline入口run_candidate.py（纯stdlib、不常驻模型），JOBS逐stage PID/log/exit，CURRENT_STATE动态排除身份。流程preflight(自动生成并冻结数据)→train1000→verify_initial/500/1000各新进程→finalvalidation→independentaudit_validation，无任何物理调用。

尚未运行以上阶段，所有结果字段以实际输出为准。若任何阶段代码/数值失败，保留日志和已完成artifact，停止依赖阶段、不盲目重复训练；父协调最小新scope修复。

## C 必要真实审核

- 实际完整1920batch4 production/scan值+hidden/全部梯度/unitAdam/每periodloss人工公式/fresh8Adam计数/parentexact/输入边界。
- initial/500/1000 live-save model/Adam/RNG/curve/schedule exact，loader构造后恢复RNG；新进程实际24train生产重放turn/latent/hidden exact、scan误差≤1e-12、实际batch调度重构、再生数据exact。
- final12validation once，所有on/off分项gate按预登记；新进程12序列生产重放及功能flag独立重算。
- 原source/data/权重hash前后核验，新FROZEN不包括CURRENT_STATE/JOBS/报告/visual；原AB/旧sensor checkpoint与VERIFY不修改。

## D 后续仅条件允许

若offline全12目标PASS且父协调，另登记最多6B完整身体回合/同motor新sensor匹配noodor，不能复用旧sensor不同权重的noodor回合。若offlineFAIL只据记录诊断有限候选，不自动进入物理或无限训练。视频由父排期，这个候选pipeline本身不制作视频。

当前无heavy、无backgroundjob、无无用临时文件。父当前render/plot为唯一heavy，本子不抢占。
