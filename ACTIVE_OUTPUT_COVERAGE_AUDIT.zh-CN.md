# 34个verified缺失发放根的真实输出覆盖损失

**手工研究原型，不是成品。无符号原始连接计数，不是恢复实验。**

## 范围

沿用上一轮精确确认的34个实际发放但缓存ann_nt_verified为空的根，直接读取原始BANC连接缓存，逐根统计输出。当前12,000根tier和202个运动池目标来自经过manifest/identity哈希校验的身份及上一轮逐运动根诊断。源根唯一连接，全部再次检查verified为空。文件SHA256执行前后不变。

## 结果

| 互斥输出类别 | 原始边记录 | 原始突触计数 |
|---|---:|---:|
| 目标在tier外 | 133 | 670 |
| tier内非运动目标，因源verified缺失排除 | 1412 | 15897 |
| tier内运动池目标，因源verified缺失排除 | 114 | 1194 |
| 合计 | **1659** | **17761** |

其中tier内共1526条记录、17091个突触，占这34根全部原始输出突触的 **96.23%**。34根中26根有直接运动池输出，114条记录涉及 **28个唯一运动池目标**。

以上为原始缓存行和突触计数，**未应用每边上限或电导尺度**，不能等同于模型电导、兴奋强度或预期运动输出。运动池记录已包含在tier内类别中，不可重复相加。

## 解释

对这34个源根，当前输出损失主要不是目标落在tier外，而是tier内连接因源verified缺失被排除。tier外类别和tier内类别互斥；不能说tier外目标已通过递质资格检查，源标签仍然缺失。

这确立一个可量化的模型覆盖缺口，**不证明它是运动池零脉冲的唯一原因，也不证明补入预测符号会恢复运动或站立**。保留严格模式，不按预测标签、细胞名称或站立需求补边。JSON的114条直接运动记录均显式保留sign=null、modelled=false。

不能把本次34根的比例推广到全部BANC或全部tier。发放根来自特定左前腿20ms诊断输入、100ms观测协议。

## 验证

逐根及总计核对：全部=外部+内部，内部=运动+非运动；核对运动边清单行数、突触总和、唯一目标数及未赋符号状态。与递质来源测试联合 **5 tests / OK**。

已用read_image实际查看 `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_output_coverage_audit.png`，两面板计数133/1412/114及670/15897/1194与JSON一致，标签和原型说明清晰。

## 文件

报告：`/run/media/sensen/Data2/cell_wound_prototype/ACTIVE_OUTPUT_COVERAGE_AUDIT.zh-CN.md`

数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_output_coverage_audit.json`

图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/flybody_loop/active_output_coverage_audit.png`

源码：`/run/media/sensen/Data2/cell_wound_prototype/tools_active_output_coverage_audit.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_active_output_coverage_audit.py`

桌面目录：`/home/sensen/Desktop/cell_wound_prototype/active_output_coverage_audit/`

下一步优先核查这些源根的类型交叉表及可用实证递质/受体证据，并明确严格已知网络与未知连接范围，不以参数搜索掩盖证据缺失。
