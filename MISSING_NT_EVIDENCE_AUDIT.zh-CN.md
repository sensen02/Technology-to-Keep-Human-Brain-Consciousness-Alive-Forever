# 缺失verified根的本地证据分层

**手工研究原型，不是成品。只读来源核查，不补符号或连接。**

## 本轮结果

34个实际发放但verified为空的根全部属于 **prediction_only**：本地CSV及v888都没有非空verified标签，但有预测值。两个预测字段归一化比较0冲突；它们不构成独立生理验证，也不能升级为verified。

源根类别：27 sensory、6 ventral_nerve_cord_intrinsic、1 ascending。类型计数：SNpp52 10、SNpp45 11、CoHP8 5、SNpp44 1；以及IN14A092、AN05B104、IN21A022、IN14A086、IN14A096、IN26X001、IN14A038各1。类型名用于追溯身份，不用于推断递质。

34根中26根有原始直接运动池输出，共114条记录、28个唯一目标。本轮以banc_888_id唯一连接目标注释，全部目标super_class为motor；每条记录附带目标cell_type、subclass、body_part_effector及功能字段。sign继续null、modelled继续false。

## 证据边界

本轮只审计本地缓存、CSV、v888字段和此前原始连接记录；**没有完成外部文献、跨数据集同源类型或实验递质交叉表检索**。因此prediction_only仅描述已审计字段，不宣称整个科学文献不存在证据。manc_cell_type等名称不是已经验证的跨数据集逐根对应关系。预测字段一致可能只是共享上游来源，不是两次独立验证。

不能以多数源预测ACh、感觉器官名称或运动输出需求补入连接。当前严格网络覆盖不足与预测模型的潜在作用必须分开报告。

## 验证

新增测试验证：即使类型名含cholinergic，缺少递质字段仍不能取得递质资格；预测不升级为verified；34源唯一；114边/28目标一致；目标均motor；所有边仍未赋符号。与输出覆盖及递质来源测试联合 **7 tests / OK**。

已实际查看 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/missing_nt_evidence_audit.png`，prediction_only=34及原型标签清晰。

## 文件

源码：`/run/media/sensen/Data2/cell_wound_prototype/tools_missing_nt_evidence_audit.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_missing_nt_evidence_audit.py`

逐源及逐目标数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/missing_nt_evidence_audit.json`

图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/missing_nt_evidence_audit.png`

桌面目录：`/home/sensen/Desktop/cell_wound_prototype/missing_nt_evidence_audit/`

下一步应进行外部来源检索或明确限定的预测模型证据审查；不能把这一轮本地字段分层称为完成文献验证。
