# 多感官脑侧校准：代码准备完成，尚未运行数值

本子实际只执行标准库CODE_ONLY检查41项，全true／exit0。未导入NumPy/Torch、加载数组payload／模型、运行forward、训练、物理、渲染或网络。所有旧科学源保持不变。实际训练和验收必须由父串行资源执行，不能把本说明当校准成功。

父准确入口：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_calibration/run.py --execute-parent-fusion-calibration > /run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_calibration/PIPELINE.log 2>&1
```

流程7阶段：preflight／fixed400train／verify0／verify200／verify400／finalevaluate／independentmetric-audit。全部显式flag；失败停止并保留，不盲重跑／覆盖已生成数据、权重或日志，不缩有效批、不加步。

模型API：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_calibration/model.py` 的 `NeuralBridge.brain_step(delivered_an, modalities, h)`，modalities形状(B,2)且顺序为[visual_raw,odor_raw]，脑实际输入[AN1,visual1,odor1]；返回(DN2,h16)。normalize／feedback／decoder／allowed_raw_indices保持原接口。
- 同目录common.py的 `load_checkpoint(path,digest,final=True)`严格加载新格式final400，返回(model,payload)。`initialize()`返回(newmodel,parentpayload,old_parent_2input_model)，最后一项专用于初始视觉保真预检。
- `load_sensory()`一次返回(vision,odor,visualpayload,odorpayload)，完整隔离common／model／sys.path；视觉99698／嗅觉66f500冻结，旧deployment_qualified=false仍保留。
- `real_queue_rollout(model,x,u,binding_path,digest)`，x(T,B1,111)、u(T,B1,2)。实际1ms队列9600tick、2ms释放／5ms消费，仅传AN／DN；感官神经标量在脑侧，不直接进motor。

初始2123的GRU输入权重48×2扩48×3，前2列逐位copy／第三0，所有其他张量精确copy；独立数值预检比较新旧1920输出／状态及任意odor不影响初始视觉，容差1e-12。自然零受体经66f的神经输出非严格0；纯视觉6条使用该自然无气味流，数值零留作明确辅助／保真对照，不以运行规则清零。

27条来源：6最新视觉实际RGB+同步身体；6B夹持；6C空间真实受体；6时间对齐单模态神经流人工配对(身体取视觉，并非同时多感官物理)；3实际旧身体的两模态神经输入显式零辅助。最新视觉3080取1160:3080；旧B/C2880取960:2880，不混后缀。逐帧实际RGB、8视觉回合的缓存／GRU32／raw／latent，12B/C受体滤波pending／原66f GRU32／rawturn先精确重放后才准入数据。

训练teacher只有loss：单模态原NN输出，配对强度加权公式明确工程监督，所有目标一决策延迟／首0。运行模型里没有融合公式、条件标签、位置方位、sign/deadzone、目标速度指令或门控。训练只10张量，感官网络／反馈编码器／重建头／norm冻结。

有效27、微批9×3，三完整1920图分别回传且每loss/3，整次更新仅clip／Adam一次，不时间detach；训练freshAdam3e-4／400更新／保存0,200,400，仅final400评价。实际预检full27独立Python图先backward并释放，再独立JIT微批累积；损失、全部梯度、Adam参数与状态容差1e-10，RSS<1800MiB。超限必须报告而非偷偷改有效批。wallcap1800秒，不扩预算。

336记忆是motor256＋visual32＋odor32＋brain16，cache16／filter2＋pending2／AN-DN接收和消息另记。此校准不载入身体、不执行新物理，也不证明独立前向／停车／食物到达／天然感官偏好／真实颈部膜电生理。旧D normal8/cut5差3<4的确认失败完整保留，D不进入训练。

准备证据：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_fusion_calibration/CODEONLY_CHECK.json`

设计审阅：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/olfactory_design/设计审阅.zh-CN.md`
