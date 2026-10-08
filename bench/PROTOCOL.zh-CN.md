# 7 µm / 20 µm 仅记录电极与前端：台架测量协议

这是**手动实施的软件工具与待执行台架协议，不是已验证的硬件成品**。当前没有真实台架测量。所有阈值是待批准的工程要求，不是文献测量值或真实测试结果。此工具不连接动物、人体、在体组织或身体解码器，不加载 fullBANC；不增加刺激功能。台架的已知测试信号注入仅为前端校准，不等于生物刺激授权。

## 文件与命令（完整绝对路径）

- 配置模板：`/run/media/sensen/Data2/cell_wound_prototype/bench/config.template.json`
- 空测量模板：`/run/media/sensen/Data2/cell_wound_prototype/bench/measurements.template.csv`
- 导入/评估器：`/run/media/sensen/Data2/cell_wound_prototype/bench/evaluate.py`
- 软件合成夹具生成器：`/run/media/sensen/Data2/cell_wound_prototype/bench/synthetic_fixture.py`
- 软件自测：`/run/media/sensen/Data2/cell_wound_prototype/run_bench_selftest.py`
- 默认可供后续 UI 读取的状态：`/run/media/sensen/Data2/cell_wound_prototype/outputs/bench/status.json`（本次未修改 UI）

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
python3 /run/media/sensen/Data2/cell_wound_prototype/run_bench_selftest.py
python3 -m bench.evaluate --config /run/media/sensen/Data2/cell_wound_prototype/bench/config.template.json --output /run/media/sensen/Data2/cell_wound_prototype/outputs/bench/status.json
python3 -m bench.synthetic_fixture
python3 -m bench.evaluate --config /run/media/sensen/Data2/cell_wound_prototype/outputs/bench/synthetic/config.json --measurements /run/media/sensen/Data2/cell_wound_prototype/outputs/bench/synthetic/measurements.csv --output /run/media/sensen/Data2/cell_wound_prototype/outputs/bench/synthetic/status.json
```

真实测试先将模板分别复制成 `/run/media/sensen/Data2/cell_wound_prototype/bench/config.actual.json` 与 `/run/media/sensen/Data2/cell_wound_prototype/bench/measurements.actual.csv`，人工填写后执行：

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
python3 -m bench.evaluate --config /run/media/sensen/Data2/cell_wound_prototype/bench/config.actual.json --measurements /run/media/sensen/Data2/cell_wound_prototype/bench/measurements.actual.csv --output /run/media/sensen/Data2/cell_wound_prototype/outputs/bench/status.json
```

工具只允许输出在 `/run/media/sensen/Data2/cell_wound_prototype/outputs/bench/` 内。合成数据永久独立存放在 `/run/media/sensen/Data2/cell_wound_prototype/outputs/bench/synthetic/`，其数值仅测试代码分支，不能复制成工程指标或真实记录。默认状态文件不能用合成状态冒充实测。

## 1. 测量前冻结配置和证据

固定直径 7 µm、中心间距 20 µm，不改变它们。必须明确 `diameter_interpretation` 为 `shaft`（杆体）或 `exposed_contact`（暴露接触）。杆体直径**不能推定暴露有效接触面积**。单独记录 `exposed_active_area_um2`、面积测量/图纸证据 ID、暴露形状、绝缘范围、润湿条件；必要时记录侧壁面积。即使解释为圆形接触，也只能在确认几何后另算 πd²/4，并保留证据，工具不会自动填入。

填入真实电极、前端标识及参考电极/参考输入的连接描述和证据。模板不指定材料、电源电压、ADC 位数、采样率或仪器型号，不能据此假定任何硬件规格。模板的 CH1、1000 Hz 只是待替换的计划占位；依据工程要求填写所有通道与频率网格。每一个指标都按该网格检查，包括噪声 ASD 的采样点。串扰须至少两个通道且测试本计划子集内所有有向通道对；128 通道上限不表示硬件规格。若阵列有 256 通道，使用真实全局通道 ID 分为经批准的子集/通道对计划，每个计划单独版本及输出；不能把一个子集通过解释为完整阵列通过，跨子集串扰仍需独立覆盖和人工审查。

`config_version` 每次变更递增。`measurement_plan_evidence_id` 链接批准的计划；`threshold_rationale_evidence_id` 链接需求/风险/校准依据。每个 `planned_engineering_thresholds` 指标使用规定单位；只填已批准的上下界，未定保持 null，不能为了得到 PASS 设置无依据的宽松范围。上下界为闭区间，不包含测量不确定度自动余量；应在批准阈值与方法时约定不确定度、重复性、保护带和温度/介质条件。没有完整元数据或阈值，不能 BENCH_PASS。

## 2. 测量顺序（仅非生物台架）

1. 检查绝缘、接线、屏蔽、参考、接地和安全限流；在适合电极测试的已知非生物负载/介质中操作。保留设置照片、连接图、介质/温度、仪器校准和原始数据证据。具体限值由负责工程师依据真实硬件决定，不采用本软件合成值。
2. 阻抗：各通道/频率测量复阻抗幅值和相位；记录测试激励和参考方式到方法证据。CSV 分别导入 `impedance_magnitude`（ohm）和 `impedance_phase`（deg）。不把 |Z| 当成电阻，也不从该值推断接触面积。
3. 输入噪声：指定接地/终端阻抗/等效电极负载，记录前端增益与带宽，按输出噪声除以已校准增益得到输入参考 ASD。记录实际测量带宽、采样率、时间长度、窗函数、谱估计方式和单边/双边约定；CSV 的 ASD 统一为**单边幅度谱密度 V/√Hz**，不是 RMS 电压、功率谱密度或未经换算的输出噪声。每频点 ASD 是测量量，不自动推定全带宽 RMS。参考噪声另测，不能把其缺失当作零。
4. 已知注入：使用校准的输入端正弦峰值电压（不是峰峰值/RMS）；测量幅度增益 V/V 和相对输入的相位 [-180,180] deg。记录源阻抗、前端/ADC/滤波器状态、采样率和线性范围。
5. 串扰：每次仅一个源通道注入，测量另一个受扰通道。定义 `crosstalk` = 20 log10(受扰输出幅值 / 同设置下驱动通道输出幅值)，单位 dB，通常负值；不是功率比。`channel` 是受扰通道，`source_channel` 是驱动通道，禁止相同。负载与参考连接固定并有证据。
6. 混叠：保持真实采样/抗混叠设置，注入高于 Nyquist 的已知单频信号，在折叠频率测残留。定义 `alias_rejection` = 20 log10(同幅度带内参考输出 / 带外注入折叠输出)，正值表示抑制。CSV `frequency_hz` 为折叠频率，`injected_frequency_hz` 为实际注入频率；工具验证折叠关系。保留带内参考原始幅度与抑制估计方法，不能用数字滤波模拟值代替测量。
7. 饱和：只在真实硬件安全范围内逐渐增大输入；方法证据预先约定削顶、失真、非线性或过载判定及步长。`saturation_input` = 首次达到约定饱和判据的已知输入 V_peak，和本行 `injected_amplitude_v_peak` 相等；保留前一未饱和点、原始波形及恢复行为。未观察到饱和是删失/下界，不能把最大试验值冒充实测起点；本版不接受删失量，保持该项缺测并补测/修订计划。
8. 参考：单独评估 `reference_noise_asd`、连接可靠性、参考改变引起的漂移和共模行为；CSV 仅自动评估 ASD，其他观察作为方法/证据附件人工审查。工具不声称已自动评估全部硬件安全或共模性能。

## 3. CSV 严格格式

CSV 表头及列顺序必须与空模板完全一致，不接受重复/多余/缺少列或首尾空白。每行一个“指标、通道、源通道、频率”唯一键；重复测量应先按批准统计方法汇总并在方法证据保留原始重复数据，不能随意删掉坏点。

必填文本：`schema_version`=`bench-1.0`；`config_version` 与配置相同；`evidence_id` 原始记录可追溯 ID（可以多项共享同一原始记录）；`provenance`=`measured` 或 `synthetic`，一个文件不可混用；`metric`；`channel`；`unit`；`reference_id`；`setup_id`；`instrument_id`；`calibration_evidence_id`；`method`（方法或方法文档 ID）；`timestamp_utc`（ISO8601 带 UTC 时区）。ID 是索引，不表示软件已核实证据真实性；工程人员仍须保存并审查证据原件。

必填数值：`frequency_hz`（配置网格中正频率）、`value`（有限数）、`duration_s`（正测量时间）。ASD 必填 `bandwidth_low_hz`、`bandwidth_high_hz`、`sample_rate_hz`，满足 0 ≤ low < high ≤ fs/2 且频点位于带内。增益/相位/串扰/混叠/饱和必填 `injected_amplitude_v_peak`、`injected_frequency_hz`、`sample_rate_hz`；除混叠外注入频率必须等于测量频率并低于 Nyquist。非适用列留空，串扰以外 `source_channel` 必须空。拒绝 NaN、Infinity、缺数、负幅度/噪声、越界相位、重复键、未知通道/频率和错单位。不自动换算 kΩ、µV、峰峰值或 RMS，以防静默单位错误。

支持的指标/单位：

| metric | unit |
|---|---|
| impedance_magnitude | ohm |
| impedance_phase | deg |
| input_noise_asd | V/sqrt(Hz) |
| gain | V/V |
| phase | deg |
| crosstalk | dB |
| alias_rejection | dB |
| saturation_input | V_peak |
| reference_noise_asd | V/sqrt(Hz) |

## 4. 状态与真实性

- `NOT_RUN`：至少一项缺测；缺测绝不默认为通过或零值。已测且超阈值时总状态优先 FAIL，其他缺测仍各自 NOT_RUN。
- `INVALID_INPUT`：格式、版本、单位、元数据或数值验证失败；CLI 返回 2，状态 JSON 保留错误。
- `UNCONFIGURED`：数据齐全但要求/面积/参考证据或阈值未批准/填写。
- `FAIL`：真实测量超出计划阈值；CLI 返回 1。
- `BENCH_PASS`：仅代表完整台架数据对**所填计划工程阈值**通过。所有报告 `readiness_claim=false`，`live_or_body_validation=NOT_EVALUATED`；不是可在体使用、安全性、体解码或生物学验证结论。
- `SYNTHETIC_ONLY`：软件合成输入，永不改为真实 BENCH_PASS；内部 `engineering_evaluation` 只用于测试评估分支。

CLI 返回 0 仅表示合法处理成功，不等于通过（包括 NOT_RUN、UNCONFIGURED、SYNTHETIC_ONLY）；自动化必须检查 JSON 状态而不是仅检查退出码。报告保留配置版本、配置/CSV SHA-256、绝对输入路径、每项证据 ID、实际值和计划阈值，方便后续 UI 分开展示“计划”和“已测”。本版不联网、不核验仪器签名/记录真实性、不读取原始波形，可信度取决于可审查的实测证据链。

## 5. 内存及范围

Python 标准库，无 NumPy 依赖；单输入最多 16 MiB、CSV 最多 20,000 行、最多 128 通道/100 频率。Linux CLI 与自测将地址空间限制为 480 MiB（低于 512 MiB）；不运行 fullBANC、既有仿真、活体采集或身体解码。大数据须按批准方案分批汇总，不能绕过上限。此版属于台架协议、数据验证和评估工具；真实硬件台架执行仍未完成。
