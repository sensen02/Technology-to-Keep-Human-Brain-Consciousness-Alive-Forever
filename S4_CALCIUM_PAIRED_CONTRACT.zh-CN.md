# S4 有界感觉钙信号—关节角配对契约 v1

## 结论边界

这是手动实施的研究原型、结构准入工具，不是成品或真实动物标定成果。服务 S4 的数据准备，不证明 S1 数值收敛、S3 身份映射、稳定站立/行走或生物有效性。

新增工具 `/run/media/sensen/Data2/cell_wound_prototype/tools_s4_calcium_paired_contract.py`；测试 `/run/media/sensen/Data2/cell_wound_prototype/test_s4_calcium_paired_contract.py`。读取既有钙元数据资格工具和感觉事件契约后，另立钙数值契约；只从既有事件工具复用 `load` 和资源上限，不调用其单细胞/spike 校验器。

所有结果始终 `calibration_ready=false`、`biological_validation=false`、`calibration_executed=false`。不下载/执行外部资产，不拟合参数，不修改配置。合法文件尚未到位，不能以合成夹具冒充原始数据或绕过此前 HTTP403。

## 输入格式

顶层 `schema="s4_calcium_paired_v1"`，`data_kind` 为 `synthetic_fixture` 或 `biological`。要求 `dataset_id`、`species`、`license`、`reporter`、`fluorescence_metric`、`split_policy`；声明文本最长4096字符。`source` 要求 `uri`、`revision`、`sha256`（64位小写十六进制或 `unknown`）、布尔 `asset_bytes_verified`。摘要和核验标志仅声明，不独立验证外部资产。

`units` 要求 `time="s"`、`angle="rad"`、非空 `fluorescence` 单位声明。荧光指标和单位是独立声明，不能假定原始荧光、ΔF/F、z-score 互换；不提供单位转换或生物阈值。报告保留指标、reporter、单位和逐ROI身份/来源声明。科学含义仍须实际来源核查。

每个 `trials` 记录要求：

- 精确非空字符串 `subject_id`、`trial_id`、`roi_id`、`raw_recording_id`，不能 `unknown`、数值化或带首尾空格。这些是分组身份，不要求 BANC root。
- `split` 为 `train`、`validation`、`holdout`，至少有train及holdout。
- `identity_level` 为 `roi_population`、`single_cell` 或 `unknown`；`roi_identity` 为非空声明或 `unknown`。ROI群体原样保留，单细胞也仅声明，不由reporter/亚型推断。
- `response_origin` 明确 `measured`、`predicted` 或 `unknown`。非measured有缺口；即使measured也不代表科学核验完成。
- `leg`、`joint`、`angle_convention` 为声明或 `unknown`，不得缺失后自动猜测。
- 数组 `time_s`、`angle_rad`、`fluorescence_time_s`、`fluorescence` 为有限数值列表，每个至少2点且四个长度相等。两条时间轴严格递增；角度/荧光不要求单调，允许负值，不凭空施加生物范围。

拒绝未定义字段，包括 `cell_root_id`、`banc_mapping`、spike字段，避免假根和伪事件进入此契约。没有从荧光反演spike，也没有荧光到肌肉活化推断。

## 同步与处理

`sync` 必须有 `common_clock_id`、`method`、`error_evidence`、`missing_data_policy` 声明，以及 `fluorescence_offset_s`、`fluorescence_scale`、`max_error_s` 数值或字面 `unknown`。已知scale必须正、max_error必须非负；不设虚构生物容差。

已知offset/scale时计算 `paired_time = fluorescence_time_s * scale + offset`：变换后有限、严格递增，且与 `time_s` **逐项精确相等**。这是预对齐配对数据格式要求，不是生物精度结论。独立采样率的数据须上游显式重采样并声明，工具不做插值、不用可调误差放宽准入。未知同步允许结构准入但逐项记录缺口，不能变成标定就绪。

`processing` 要求 `baseline_method`、`normalization`、`filtering`、`resampling`、`motion_correction`、`observation_model`；均可显式 `unknown`，不能空白/缺失。缺口逐路径报告；通过只意味着声明格式有效。

## 身份与留出限制

完整(subject,trial,ROI,raw recording)组合不能重复。相同raw recording不能跨split或更换subject；相同(subject,trial)不能更换raw recording；同(subject,raw recording,ROI)的身份层级/描述须一致。一个trial允许多个不同ROI，不强迫它们变成独立动物或BANC细胞。

v1检查**声明的原始录制记录不交叉**，不强制subject不交叉。隐藏资产别名无法检测；同动物跨录制也不能证明跨个体泛化。train/holdout数量和不交叉仅工程条件，未执行训练或留出预测。

## 资源、数值与安全

复用 `/run/media/sensen/Data2/cell_wound_prototype/tools_s4_paired_data_contract.py` 的有界JSON loader：最多8MiB、拒绝所有层级重复JSON键和NaN/Infinity常量、保留输入SHA256。解析后的 `1e999` 虽由Python变为inf，数值校验拒收。CLI也捕获过深嵌套的RecursionError为拒收报告。

最多1000个ROI记录，数组最多100000点、**所有四数组合计最多100000数值项**。拒bool、字符串数值、非有限数值及绝对值超过2^53的整数。浮点时间严格递增能拒绝精度折叠；对已在上游舍入的浮点值无法恢复原始精度，此格式不会声称检测所有浮点损失。已知时钟映射若溢出或折叠拒收。

CLI只读取本地manifest，使用排他创建输出，不覆盖已有报告：

```sh
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/tools_s4_calcium_paired_contract.py /absolute/path/manifest.json --output /absolute/path/new-report.json
```

通过退出0并输出 `STRUCTURAL_PASS_WITH_GAPS`；拒收退出1并写错误。路径/输出不可用仍可能由系统报错，不做绕过。纯函数 `synthetic_fixture()` 只在内存返回明确 `NOT_ANIMAL` / `synthetic_fixture` 的工程夹具，ROI群体、predicted，不含任何虚构根或动物资产摘要。

## 本轮验证

工作目录 `/run/media/sensen/Data2/cell_wound_prototype`；解释器 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python`。

 focused命令：

```sh
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -m unittest -v test_s4_calcium_paired_contract
```

17个测试方法通过，退出0。随后运行 `-m unittest -v test_s4_calcium_paired_contract test_s4_calcium_qualification test_s4_paired_data_contract test_s4_motor_data_contract`，4模块34个方法通过；`-m py_compile tools_s4_calcium_paired_contract.py test_s4_calcium_paired_contract.py` 和 `git diff --check` 无报错。共2次unittest调用（17、34方法），1次py_compile、1次diff检查；两次测试调用各内嵌3次CLI子进程。测试包含29个字段畸形子案例、8个非有限/bool/精度案例、9个时序/配对计数案例、8个同步数值案例；另覆盖身份/录制泄漏、多ROI、unknown缺口、预算、重复JSON键、非有限常量、指数溢出、CLI正常/拒收/禁止覆盖。CLI由单个测试方法执行3次（预期返回码0、非0禁止覆盖、1重复键拒收），不是3次动物实验。无全量回归或真实资产核验，无图像生成或动物曲线查看。
