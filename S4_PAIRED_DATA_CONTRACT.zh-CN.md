# S4 claw/hook 配对数据契约 v1

这是手动研究工具，不是生物标定或成品。工具 `/run/media/sensen/Data2/cell_wound_prototype/tools_s4_paired_data_contract.py` 只校验内联JSON的结构资格，不下载数据、不执行来源代码、不拟合参数。运动脉冲—肌肉/力学数据不在v1范围。

## 顶层字段

- `schema_version`：整数1；`scope`：`sensory_claw_hook`。
- `data_kind`：`synthetic`或`biological`；类型是提交者声明，不是工具验证的事实。
- 非空文字：`dataset_id`、`species`、`license`、`preprocessing`、`split_policy`。
- `units`必须精确为time=s、angle=rad、velocity=rad/s、events=spike。其他单位先显式转换并记录预处理，工具不猜转换。
- `source`包含非空`uri`、`revision`、`identity_evidence`、`coordinate_convention`及64位小写十六进制`sha256`。摘要/身份仅声明，本工具不取外部资产核实。
- `trials`：2到1000条，合计最多100000个数组元素。JSON文件最多8MiB；重复JSON键和NaN/Infinity拒收。

## 每条trial

非空`trial_id`、`subject_id`、全数据集唯一语义的`raw_recording_id`、`joint`、`channel`、`leg`；`trial_id`不得重复。channel仅claw_position/hook_direction，leg为lf/lm/lh/rf/rm/rh。

`cell_root_id`必须是正十进制字符串，拒绝float和JSON整数，避免64位身份误差。合法格式不证明细胞真实存在或版本映射正确。合成样本root="1"是占位符，不是BANC身份声明。

`split`为train/validation/holdout；至少含train与holdout。相同raw_recording_id不得跨划分，改subject_id也不能规避；v1不强制subject-disjoint，不能据此声称跨个体泛化。提交者仍须保证不同ID不是同一原记录的别名，工具无法仅凭文字核实。

`sync`：非空common_clock_id、method、error_evidence、missing_data_policy；有限neural_offset_s、正neural_scale、非负max_error_s。映射：body_time = neural_time * neural_scale + neural_offset_s。该变换和误差是来源声明，不设虚构的生物允许误差；漂移/非线性同步须在上游处理并说明。

`time_s`至少两个有限数，严格递增；`angle_rad`与`velocity_rad_s`为同长度有限数数组，不接受bool；`spike_time_neural_s`是单细胞严格递增事件数组，允许无事件，不允许重复事件。同步映射后的每个事件必须位于身体样本首末时刻的闭区间。不插值、不验证速度是否为角度导数、不判断刺激是否足够辨识参数。

## 输出与限制

CLI：`/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/tools_s4_paired_data_contract.py <完整manifest路径> --output <新的完整report路径>`。

结构合格退出0，不合格退出1并报告字段路径；输出使用排他创建，拒绝覆盖旧报告。`biological_validation=false`、`calibration_executed=false`始终保留，即使提交者标记biological。来源/同步/身份须另行科学审查；train/holdout只是结构划分，不是已完成留出预测。S1收敛、生物有效性、站立/行走均不由本工具验收。

测试 `/run/media/sensen/Data2/cell_wound_prototype/test_s4_paired_data_contract.py` 内synthetic_fixture用于工程测试，任意数字不来自动物。桌面样本与拒收例见 `/home/sensen/Desktop/engine修复验收/S4-contract-v1`。下一步扩展运动配对契约及取得真实公开数据资格审查，不使用合成样本估计生产参数。
