# S4 calcium指标：文件级来源资格与补证清单

手动研究原型，不是成品。本轮只读两份现有Parquet的schema/footer，核对本地README/receipt/历史报告，以及固定公开仓库版本文本；无模型/拟合/作者代码执行/数据改动/新下载二进制。聚焦metric处理链，不重做既有留出分析或荧光→神经电流标定。

## 一、现有证据分层

| 层级/量 | 可证实事实 | 仍不能证实/决策 |
|---|---|---|
| 本地文件身份 | receipt锁定claw SHA7fc8a108…、hook SHAea61eee2…及字节/member CRC；26990/25102行由本轮实际footer读取，不是receipt字段 | ZIP全包SHA/CRC历史未验收；不能提升为原始成像完整性 |
| 本轮实际文件footer/schema | 两文件仅有ARROW:schema键，calcium字段为double，schema/字段未见metric/reporter/F0/归一化参数元数据 | double不是生物单位，不足以区分ΔF/F、小数/百分比、ratio或zscore；元数据缺席不证明作者无记录 |
| 本地归档README第91行 | 文字称calcium为normalized z-scored GCaMP/tdTomato ratio；第67/69行列Mamiya2018来源 | README确实对包含这两文件的restrained-data段声明该指标，不是完全没有metric文档；但缺逐文件导出映射、与2018ΔF/F层级不同，适用性未解决，不能当已审计确认 |
| 2018原实验/源码历史核查 | GCaMP6f、ΔF/F；公开ROI代码在整段所有连续BaselineN帧均值中取最小值，产生每ROI/录制一个标量F0，非逐帧滑动F0；右前腿 | 此轮未重新fetch原论文/2018代码；来源历史报告，不是2025整理列处理证明，实际参数/重导出未知 |
| 2025固定消费者源码 | imaging_fit_activation_function.m读入*_Mamiya2018.parquet后另算calcium_norm=calcium/max(calcium(frames)) | 证明消费后再归一化，不证明输入calcium先前是何metric；默认示例club，不称本轮claw/hook已执行 |
| 绘图配置 | calcium轴[-5,50]，calcium_norm[-.1,1.1] | 显示范围不是单位或尺度处理证据，不能由数值量级反推出百分比/ratio |
| 当前project接口 | 既有0.3nA/归一化单位和100Hz满控制等是假设 | 即使metric补证完整仍缺观测模型、群体ROI到细胞、电生理/输入电流校准，不能直接转换nA/Hz/肌肉act |

因此两份表继续使用**原尺度calcium，metric/processing/reporter文件级unknown**；不因论文层级GCaMP6f/ΔF/F已知就更新契约为已标定。不改原数据、适配输出或生产增益。

## 二、本轮新增可复核公开来源边界

固定树身份 `e1233f4a987c532c9f1ab42273af21a0a6a50393` 来自[GitHub tree API](https://api.github.com/repos/chrisjdallmann/feco-inhibition/git/trees/main?recursive=1)，HTTP200、truncated=false。树身份是本次查询快照，不是本地20250903 Dryad ZIP构建来源证明。

- [固定消费者源码](https://raw.githubusercontent.com/chrisjdallmann/feco-inhibition/e1233f4a987c532c9f1ab42273af21a0a6a50393/code/imaging_fit_activation_function.m) HTTP200：revision29July2025；另建calcium_norm；sampling_rate=ceil(1/data.time(2))是消费者表达式不是硬件同步证据；x使用L1C_flex-90仅用于拟合，不应移植为真实侧别转换。
- [固定imaging_config.toml](https://raw.githubusercontent.com/chrisjdallmann/feco-inhibition/e1233f4a987c532c9f1ab42273af21a0a6a50393/code/imaging_config.toml) HTTP200：仅显示轴设置，不能校准metric。
- [固定data README](https://raw.githubusercontent.com/chrisjdallmann/feco-inhibition/e1233f4a987c532c9f1ab42273af21a0a6a50393/data/README.md) HTTP200：是文件列表，含Mamiya2018用于调模型，**没有本地180行README的Variables/metric段**。两份README不是同一文本，不能静默替换本地文档或把远端文件列表当metric确认；这更正历史方法报告第122行称其为本地文档“公开对应”的等价表述，不覆盖旧报告。

树列表此限定版本code下未列明显2018→2025 Parquet导出脚本；这只是路径范围观察，未读所有notebook，不证明仓库或作者不存在导出实现。不得把目录缺席当永久阻塞或全网搜索结论。

## 三、最小文件级补证请求（仅草稿，未发送）

向整理来源维护者核对两文件精确SHA及下面字段，避免泛问“是否ΔF/F”。本轮不联系作者、不申请凭据、不大包下载。

1. 两份`*_Mamiya2018.parquet`的calcium原始变量名和单位：ΔF/F fraction/percent、GCaMP/reference ratio、zscore或其他？逐文件是否一致？
2. 指标计算及导出脚本的固定commit/文件路径；从2018信号到表的每步变换、顺序、参数、ROI聚合层级，以及输入输出记录crosswalk。
3. F0窗口/参数、是否最大值归一化/百分比乘100/zscore，统计在哪个ROI/试次/动物范围计算，是否涉及所有动物；缺失/滤波/重采样处理。
4. reporter/reference通道在两份输入的实际含义与来源；若tdTomato未适用，泛README第91行是否只针对2025数据？
5. 后续侧别和时间同步单独核对，不能用回答metric替代：右前腿→L1字段crosswalk，time零点/共同clock/插值误差。

**补证接受条件**：精确关联本地输入SHA或有可信版本映射；可重现的算法/参数/变换链及对应record/ROI范围；缺项继续unknown。不接受“论文用了ΔF/F所以表必然是ΔF/F”、单张图轴或正常数值分布作为证据。导出使用全动物归一化时应重新评估跨动物留出独立性，但当前处理未知不能断言已泄漏。

## 四、下一有界动作与停止条件

不再单独重复整理unknown列表。下一步若继续，只做有明确增量的**导出来源检索**：限定此前未读的固定仓库code/README及其中明确指向原始成像→Parquet的文本链接，查是否存在输入→表生成路径和逐文件metric说明；不重复已读consumer/footer/config，不把simulation_parquet_to_csv名字当成成像导出证据；不执行模型或转换、不拉原始影像/大ZIP、不遍历远程共享。若只找到读取/再归一化脚本，明确“此有限检索未找到文件级补证”，交付未发送的精准请求，停止metric支线，转S3身份来源缺口。不拿进一步拟合改善MAE填补来源缺失。

## 五、完整本地来源

- `/home/sensen/Desktop/engine修复验收/S4-calcium-input/receipt.json`
- `/home/sensen/Desktop/engine修复验收/S4-calcium-input/README.md`
- `/home/sensen/Desktop/engine修复验收/S4-calcium-input/claw_magnet_Mamiya2018.parquet`
- `/home/sensen/Desktop/engine修复验收/S4-calcium-input/hook_flexion_01_magnet_Mamiya2018.parquet`
- `/home/sensen/Desktop/engine修复验收/S4-2018原方法与整理差异.md`
- `/home/sensen/Desktop/engine修复验收/S4-接口标定准备报告.md`
- `/home/sensen/Desktop/engine修复验收/S4-真实适配验收报告.md`
- `/home/sensen/Desktop/engine修复验收/S4-observation-holdout/观测留出误差报告.md`

本轮增加schema/footer直接观察与固定仓库README差异，未解决生物metric；历史116适配包/52092唯一行不是本轮新实验。S4 calibration_ready/biological_validation不变false，身体收敛/站立行走仍未验证。无新图、无长时或过夜任务，不复用旧图冒充新成果。
