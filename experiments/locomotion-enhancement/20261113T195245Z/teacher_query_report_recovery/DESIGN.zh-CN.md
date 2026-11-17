# 阶段B报告恢复与独立保存数组审核（ONLYCODE）

原206主程序实际退出码1，原因是report_text缺少json全局；数值部分已完成并保存STAT、9份查询数组和9份case JSON，总14710查询、9Core，原记录的积分／NN／优化器调用均0、源与RNG不变，三教师自身完整轨迹复现差0。**本恢复不会把原exit1改成exit0，也不会重跑任何查询。**

新recover.py只从原query.py的AST中抽取report_text一个函数，编译到明确的“REPORT_TEXT_ONLY_EXPLICIT_JSON_GLOBAL”命名空间。json显式绑定，builtins显式提供；递归遍历该函数及嵌套code对象的LOAD_GLOBAL／LOAD_NAME，逐项解析，缺项即拒绝。已用标准库对实际STAT做纯文本干跑，完整生成2967字符，依赖只有json和str，均已解析；没有生成实际恢复输出、没有加载数组或执行原main。

正式父运行时，只用NumPy读取已保存9份查询npz和其原轨迹／样本／教师JSONL，独立计算：行数、时间／preaction切帧、10个可直接重算的mask条件、finite动作差／qualified RMSE、有效标签数及每mask失败原因。第11个actual_restore在输出数组没有逐行七hash，只能标明引用原强断言与STAT.first_restore／HANDOVER记录，不冒重新恢复了每行Core；5.8s七机械hash则可用source trace[5799]、target[5760:5800]和明确0外力独立重建比对。

三资格教师建议action与原teacher_samples.Y42、私有q_des／desiredtarget／IK／接触数与原JSONL候选行流式对照，重新算max差，不调用教师、FK、mj_forward或任何模型。学生行source COM／1msNF／actualtarget／ctrl／normalizedtarget字段与原保存索引核对。无效行、NaN缺标签和错误列表都保留；query_error行按原明确missing-label语义全mask false，不把缺失动作填0。

附加报告完整有效区间、从row0开始的最长有效前缀、最长连续有效段、每mask失败数及最常拒绝条件。失败可重叠，不相加当总无效数。另从真实targeterror／targetoffset字段重算范围与RMS，并独立描述teacher desiredtarget−actualtarget和query−actualaction的相关性；教师法则本就依赖target，相关不是因果证明。

所有原STAT source3487项、STAT/FAILURE/query源码/QUERY_EXECUTION.log/预检/9npz+9JSON前后SHA一致。原0字节REPORT不含科学内容，只记录其size=0或已不存在，不纳必须存在的hash依赖；新报告审核完成后父可实际清理该无用占位。原FAILURE、日志、有内容数值数组和源都保留。

输出只写新目录REPORT.zh-CN.md、AUDIT.json、VERIFY.json。新VERIFY明确“数值阶段成功但原报告导出失败，本次为报告恢复和独立数组审核”；原NN／物理0引用原traps/STAT，不声称保存数组审核独立重演了现场。原目录VERIFY不会被补写，不改旧science或原失败。

父资源独占后，仅运行一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/teacher_query_report_recovery/recover.py --execute-parent-saved-report-recovery
```

全流程120秒／1800MiB，0Core、0teacher query、0NN、0积分、0优化器；不启动Dagger／teacher-mixing训练或第三模型。标签可行性结果先交父判断，短有效前缀不能冒充已恢复身体运动。
