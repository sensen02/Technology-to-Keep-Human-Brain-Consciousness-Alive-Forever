# 历史审计完整性与当前来源新鲜度

手工研究原型，不是成品。

## 实际故障

编码器安全修复和主入口范围诊断之后，31项已保存审计测试中5项失败，原因是测试把历史source_bundle绑定到了当前运行指针，或要求工作区当前代码哈希等于历史采集哈希。数值断言未因此自动证明失效，也不能把来源漂移忽略。

## 正确区分

1. 历史证据：定位它明确记录的不可变运行包，验证manifest、NPZ等实际产物哈希，并复查数值关系。不能改指针指向新包使旧审计“通过”。
2. 当前来源新鲜度：逐文件报告相同、改变、缺失。变化必须显式输出，若请求当前源码复现，严格模式必须失败并要求重跑。
3. 旧代码只有哈希没有完整源码快照：旧哈希无法恢复历史源码；标记历史源码快照缺失，不能声称已验证那些历史字节。已有逐位回归是额外数值证据，不替代历史源码归档。
4. XML、原始数据、产物被改动是不同问题，不能与合法代码漂移混为一谈。

## 实施

历史来源定位与独立新鲜度报告修复覆盖6项审计（含新200ms实验），不覆盖旧结果。父任务在最新指针推进后复跑43项审计/放电滤波测试全部通过，子任务含完整包测试共54项通过。六项旧审计均如实报告当前源码漂移；没有宣称历史源码字节已认证或新旧执行等价。

新鲜度报告：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/audit_source_freshness.json`

验证工具：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/audit_provenance.py`

新增测试：`/run/media/sensen/Data2/cell_wound_prototype/test_audit_provenance.py`

未来主闭环新增实际源码字节归档：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/source_snapshot.py`。仅保存明确声明的代码/姿态输入文件，不扫描目录、不执行归档内容；哈希漂移、重复ZIP条目、路径穿越、符号链接及超限显式拒绝。23项安全与篡改测试通过。

主闭环在运行前捕获源码，发布前再次核对当前哈希并验证归档，把source_snapshot.zip与结果一起原子发布。不是完整环境/连接数据归档，也不是签名认证。旧实验未保存的字节不能凭空补成“当时归档”。

新完整包：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/22aa1705-173a-49dc-bcfd-861e408de2a7`

源码快照：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/22aa1705-173a-49dc-bcfd-861e408de2a7/source_snapshot.zip`

7组真实闭环全部22个数组与63ab前序逐位一致；验证工具核验了快照实际字节。此变更只提高证据可追溯性，没有提升站立表现。

