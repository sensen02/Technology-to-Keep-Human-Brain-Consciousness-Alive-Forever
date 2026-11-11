# RIGHT_ONLY解剖LH承重行候选

- [x] 先PREREG；初始仅code-only，父完成左教师物理后明确释放heavy并授权完整固定流程。
- [x] 冻结rc与XML实际映射校验LH index1/head7:14、455有效参数与非LH行精确保留。
- [x] actual pre-action teacher载荷索引999+5k/post5799与actual_time<1e-8；实测误差0；mask仅loss。
- [x] 原final6407模型teacher-state LH动作误差及weighted梯度有限且非零诊断，norm.01811997/absmax.00310781，PARENT_LH_DIAGNOSTIC.json。
- [x] production/cache/gradient/unitAdam/冻结行/neutral完整预检≤1e-12；最大means差1.333e-15/梯度1.421e-17/unitLH参数1.111e-16，其他行生产输出严格父模型。
- [x] 固定4000 freshAdam.001，save2000/4000；独立新进程strict保存审核各16项全通过，最终weightedLH .0001006479322略高于初始.0001002894503，不称拟合改善。
- [x] 仅最终4000完整neutral/RIGHT顺序真实物理与独立审计，原门/30°不变；neutral173项完整性PASS/原基线严格，RIGHT149项完整性PASS，actual30.631151°/LHslip.50982879仍FAIL，其余门/全六腿/nativeNF0通过。
- [x] 所有失败保留、收齐job、报告父，tracker父负责；本阶段无运行任务，报告 /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/REPORT.zh-CN.md。

left_hind仅解剖腿名，部署仍RIGHT_ONLY yaw/.06、bound3、relu(-turn)，与左转教师任务分开。当前不宣称LH误差为根因或候选可通过。
