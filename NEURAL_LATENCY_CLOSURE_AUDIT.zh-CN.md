# 神经延迟证据审计：累计总量中点不是反馈延迟

## 交付性质与结论

这是**本次手动独立实施并测试的研究原型审计，不是成品，也不是神经—肌肉闭环成功报告**。只新增审计脚本、独立本报告及专用目录产物；未修改 canonical、历史代码或旧结果。结果 JSON 的 `PASS` 仅指合成自测与审计执行成功。

1. 旧约88–110 ms一类“50%延迟”的推断不能继续使用：审阅的第23/24步实际计算的是**截至试验结束的累计差异脉冲总量的分位时间**，不是响应幅值达到稳态变化50%的时间。本次精确复现第24步200 ms正/负92.5/88.0 ms、第23步96.5/90.5 ms。未将110 ms作为本次复现值。
2. 在同一声明网络中，motor池两个人工partition相对零输入的**首个个体差异脉冲均4.5 ms**；下行readout为10.5/10.0 ms。因果5/10/20 ms窗口率的首次非零差异与之相同。此处零背景基线全静默，因此计数差异与个体差异恰好一致；一般情况下个体脉冲差异可在总数中相互抵消。
3. motor池在本次所有长度、所有窗口都**未通过预声明的有限尾部平台检验**，因此响应10%/50%均记为 `null`。这不是“无限延迟”，也不证明不存在稳态；只是本协议不允许定义该量。
4. 4.5 ms也**不是反馈延迟、肌肉激活延迟或稳定性证据**。本脚本没有神经肌接头、肌肉映射、力矩、身体积分、姿态回授；不能由此宣布25 ms预算内闭环可行，更不能沿用旧累计量宣布其在原理上不可能。

## 源码证据与度量修正

- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_24_motor_path.py` 第130–155行记录累计motor脉冲，再以 `change[-1]` 的10%/50%作阈值；第110–123行用每个通道 `idx[0::2]` 与 `idx[1::2]` 分别加正电流。
- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_23_step_latency.py` 第111–138行对下行readout采用同样累计阈值。注释称“filtered descending rate”并不改变实际测量量。
- `/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py` 的 `step` 在区间起点施加电流、区间终点报告脉冲，本审计按此标时；精度为0.5 ms栅格。

若差异率在真实延迟D后立刻恒定为r，则 `C(t)=r·max(t-D,0)`，旧50%时间为 `(T+D)/2`。零延迟D=0时仍返回T/2；更改试验结束时间T即可改变所谓“延迟”。这已经构成反例。

新率定义为 `1000 × [C(t)-C(max(0,t-W))] / (W × 池大小)`，W固定为5、10、20 ms；使用零刺激前历史与固定分母，不用中心窗、不引用未来数据。首次差异采用精确确定性比较，不是假设有噪声下的统计显著性。

平台门在执行前写入脚本：至少120 ms观测，末100 ms分成4个25 ms块，非零平均差异率，块均值范围/平均绝对值≤10%，末段逐样本标准差/平均绝对值≤20%。通过时才定义相对于该末段平均值的首次有符号10%/50%越阈时刻。这是**操作性、窗口依赖的有限尾部平台**，不是严格渐近稳态证明；越阈时间仍包含窗口滤波影响，也不是实时因果识别稳态的算法。平台阈值本身是审计假设，未针对网络结果放宽。

## 合成自测先行

先独立运行 `--self-test-only`，成功后再运行网络；完整网络命令内也先执行所有自测。

- 合成恒定率质量输入（确定性解析测试，不冒充生物脉冲），D=0和20 ms；T=100/200/400/800 ms。
- 旧50%对D=0得到50/100/200/400 ms，对D=20得到60/110/210/410 ms，严格等于 `(T+D)/2`。
- 首次差异分别0.5和20.5 ms（右端点采样）。平台通过的长试验中，10 ms因果窗响应10%/50%分别D+1/D+5 ms，与试验长度无关；这是已知窗对理想阶跃的响应，不是内在传递延迟。
- 追加任意未来高值数据不改变已存在的因果率前缀；零响应不定义阈值；斜坡拒绝平台；负响应阈值正常；个体差异可被总数抵消。

## 原网络复用与人工partition披露

使用原默认 `TierConfig(background_mean_nA=0, background_sd_nA=0)`：12000神经元、两次tier生长、非视觉路由，原默认连接权重再乘0.2；`ConductanceParams()`原参数、backward Euler、dt=0.5 ms，无预热，与第24步相同。全部配置和参数保存到结果JSON，未调参。

motor身份沿用 `ann_flow == efferent OR ann_super_class/ann_class包含motor`，397个；readout为195个。claw/hook/hairplate各腿有符号通道沿原顺序交替取索引，加0.3 nA。正partition覆盖250个细胞，负partition242个，两者直接刺激motor数均为0。

**正/负只是实验条件标签。交替索引是人工partition，不是数据集提供的细胞方向标签，不是生物屈伸、拮抗肌群或生物方向表征。** 通道注释与对每个细胞赋予方向是不同层次的证据。导电参数属于未生理标定的点神经元简化；连接符号与有效权重也不能由本试验证明真实。

运行零输入、正partition、负partition各800 ms，截取100/200/400/800 ms前缀进行长度对照；另独立重启三个100 ms条件，逐脉冲验证与800 ms前缀完全相同。即6次确定性网络运行、总5400个神经步（3×1600+3×200）。没有把这些前缀称为独立统计重复，也没有计算虚假的置信区间。

## 测量结果

下表单位均ms，“旧50%”只能称累计总量中点：

| 池/人工partition | 首次个体差异（各长度相同） | T=100旧50% | T=200旧50% | T=400旧50% | T=800旧50% |
|---|---:|---:|---:|---:|---:|
| motor 正 | 4.5 | 37.5 | 92.5 | 189.5 | 393.0 |
| motor 负 | 4.5 | 38.5 | 88.0 | 185.0 | 379.0 |
| readout 正 | 10.5 | 49.5 | 96.5 | 185.5 | 381.0 |
| readout 负 | 10.0 | 46.5 | 90.5 | 174.5 | 379.0 |

motor正累计差异总数859/1611/3061/5992，负984/1804/3364/6416。刺激不变，首次差异不变，旧50%却近似随试验长度线性增长，直接展示旧推断的终点依赖。

所有motor 10%/50%响应时间均不定义。唯一通过平台门的组合是readout正partition、800 ms、20 ms窗，操作性响应10%/50%=11.5/17.0 ms；其余readout组合亦为null。孤立门通过不代表跨窗口/长度稳健，更不能从下行池偷换为motor或肌肉结论。

图包含合成旧中点、motor旧中点与10 ms因果motor率三面板；已调用图像视图实际检查，长度依赖与明显瞬态/后续波动均可见，未将其称为稳定无噪平台。

## 精确复现命令与环境

工作目录：`/run/media/sensen/Data2/cell_wound_prototype`。

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_latency_audit.py --self-test-only
/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_latency_audit.py
```

两条最终命令均退出0。解释器 `/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python`，Python3.12.14，NumPy2.5.3，SciPy1.18.1；最终完整审计记录耗时4.451233452011365 s。无需MuJoCo，未使用其他环境的site-packages拼接。

实施过程中一次早期网络任务由于NPZ标签在循环内重复读取造成无谓开销，主动终止；改成一次加载标签后重跑。单位契约随后改为本审计实际单位并重跑全部最终测试；未将终止试跑作为成功结果。最终JSON哈希与磁盘脚本一致。

### SHA256

| 对象（绝对路径） | SHA256 |
|---|---|
| `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_latency_audit.py` | `b93f2ebc067c2fcae0dcca05a4051b8f206e58c40c41f0e364646902dc9c4708` |
| `/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz` | `170dbe3b68c68941abc4bcc150b7fb26f927782f97ad95b7c90dbcbb37d86c36` |
| `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` | `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6` |
| `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_24_motor_path.py` | `ddc5240a55cddd0abe8520f4fb151e58d209c1ba6e320acb4da535091ba3c9e9` |
| `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_24_motor_path.json` | `f5ac809c0244b43a6409c603b6b0eac53eb43992758f18675fc8da8517c77833` |
| `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/latency_spike_traces.npz` | `aa138ee64a86f5d8c313932d46d8c93eb3505f1912f30385a7aa88d155496515` |
| `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/latency_audit.png` | `7661f6adc2962e4201dfe535c88744bba88d3f75a25e2d660107db98c6c8fdbe` |

其他依赖哈希、完整命令、解释器、配置、端点依赖数值及平台诊断在完整JSON的provenance/measured中。canonical XML、结果契约模块、第23/24步源码及第24步旧JSON在运行前后逐文件哈希不变。

## 新增文件与结果契约

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_latency_audit.py`
- 独立说明：`/run/media/sensen/Data2/cell_wound_prototype/NEURAL_LATENCY_CLOSURE_AUDIT.zh-CN.md`
- 合成自测结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/latency_selftest.json`
- 完整审计结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/latency_audit.json`
- 每个池/条件的逐细胞逐步布尔脉冲、root/local索引和时间：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/latency_spike_traces.npz`
- 验证图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/latency_audit.png`

两份JSON都经 `engine/canonical_physics.write_result` 写出，保留question/status/measured/provenance/what_this_does_not_support契约。通过允许的extra覆盖units为本实验ms/s/Hz/spikes/nA/uS/mV/nF，并覆盖illustrative_vs_measured，明确解析合成测试、仿真测量与假设参数，不沿用身体质量单位或“全部为测量”的误导模板。NPZ/PNG是JSON引用并哈希的辅助产物，不冒充独立结论。

## 后续需要什么才能谈闭环

需要在另一个独立实验中固定真实输入时刻，测感觉编码、网络、motor到肌肉分配、激活动力学、力矩、身体运动及回授的完整时序；检查方向是否纠错、幅值是否足够、噪声与窗口/参数敏感性、闭环与开环对照及机械预算是否适用。本审计撤销的是**旧延迟推断的证据资格**，不是替代上述闭环验证。
