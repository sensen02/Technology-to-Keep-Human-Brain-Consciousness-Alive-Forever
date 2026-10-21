# S4 运动事件—肌肉观测配对契约 v1

手动研究工具，不是生物标定或成品。实现 `/run/media/sensen/Data2/cell_wound_prototype/tools_s4_motor_data_contract.py` 只进行结构校验，不拟合、运行模型或推算力。

## 与感觉契约共享的字段

顶层schema_version=1，scope=motor_muscle，data_kind显式synthetic/biological。dataset_id/species/license/preprocessing/split_policy及source中的uri/revision/sha256/identity_evidence/coordinate_convention同感觉契约。units严格time=s、angle=rad、velocity=rad/s、events=spike。

trial_id/subject_id/raw_recording_id/joint/leg、十进制字符串cell_root_id、split、身体time_s/angle_rad/velocity_rad_s及spike_time_neural_s、sync沿用 `/run/media/sensen/Data2/cell_wound_prototype/S4_PAIRED_DATA_CONTRACT.zh-CN.md`。至少train和holdout，同原始记录禁止跨划分。每条记录只表示一个已识别运动细胞，不把同时间不同细胞事件合并；多细胞池需分别记录。实现内部复用感觉结构验证器时用一个固定channel占位，该内部适配不是运动细胞属于claw的声明。

## 新增trial字段

- muscle_bundle_id：非空肌束身份。
- recruitment_evidence：神经—肌束关联证据说明；结构工具不科学核实。
- pool_coverage：complete/partial/unknown，只指提交者声明的记录覆盖，不保证解剖完整。
- observation：肌束观测对象，必须有kind、unit、time_s、values。

支持的kind/unit：emg/V、calcium/deltaF/F、activation_estimate/1、force/N、torque/N*m。不同单位须先有来源地转换；模型单位不能直接冒充N。EMG或钙信号不是活化，不能自动由其推断力矩。activation_estimate值必须在[0,1]；calcium可有负deltaF/F，EMG和有符号力/力矩不作无依据限幅。

每个observation必须有非空measurement_model、model_revision、calibration_evidence、clock_mapping_evidence，以及非负有限max_sync_error_s/sensor_delay_s。观测time_s已经映射到公共身体时钟；sensor_delay_s单列为观测延迟，不自动平移时间，也不等同肌肉动力学时间常数。同步声明不是仪器同步已验证。

观测time_s至少两个严格递增有限样本，values等长有限数，时间都在身体采样窗口内。拒绝bool、NaN、Infinity和不合法数组。读取文件预算8MiB、trial最多1000，神经/身体/肌肉数组合计最多100000元素。输入拒绝重复JSON键；输出排他创建保留旧报告。

## 使用与边界

`/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/tools_s4_motor_data_contract.py <完整输入路径> --output <新的完整报告路径>`。

结构合格退出0，不合格退出1。biological_validation与calibration_executed始终false，即使data_kind=biological。来源摘要/精确身份/招募/测量模型均是声明，不能用结构通过替代独立证据。无跨个体泛化保证、无生物允许误差发明、无参数估计或行为验收。

测试 `/run/media/sensen/Data2/cell_wound_prototype/test_s4_motor_data_contract.py` 使用明确synthetic夹具，root1及来源零摘要是占位符，不是BANC或动物数据。桌面样本 `/home/sensen/Desktop/engine修复验收/S4-motor-contract-v1/synthetic-valid.json`。生产接口/参数/身份根/历史证据未修改。
