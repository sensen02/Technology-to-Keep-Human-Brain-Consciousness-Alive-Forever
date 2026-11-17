# 固定神经颜色对立编码候选7：离线72项必需配置全部通过，身体仍待验证

父代理报告真实流程已完成：初次bash91在正式数据生成/训练/冻结前因generate_data没有导出顶层main中止；父保留失败日志和原SHA，仅补main入口并通过新的恢复启动器执行bash92。bash92全部阶段exit0已收集，没有训练重跑、追加预算、择保存点或改门。当前结果以实际checkpoint与独立审核为准，不能把最外执行错误当模型行为失败或隐去。

## 实际结构与初始化

策略输入仍是保存的实际双眼RGB，不含坐标、距离、目标标签、visibility标志或时钟。网络内部先用固定1×1神经颜色对立层weights=[1,−.5,−.5]、bias0作用于RGB/255，再经ReLU成为单通道，进入新Conv1to8；后续conv2/eyehead/adapter/慢GRU32/turnhead与原架构一致。固定层是registered buffers，不参加Adam，仍有14个可训练张量。

新Conv1的weight/bias按seed20261112 fresh初始化，其余12个可训练张量逐位继承be62264966789c59a82e27dc696d710d0751f4c9ac8c1b575d26b0a1920a7651。没有用旧RGB卷积通道求和假装等价。固定前端、通道变化、局部初始化与后续训练构成联合候选，不能说已隔离唯一根因。

这是工程红选择受体层，不是已验证果蝇复眼、生物光谱模型或颜色识别能力。没有颜色质心、模板匹配、按颜色位置生成方向的算法，也没有运行时ifwall归零、减bias、deadzone、神经记忆reset。固定颜色选择层是否适用于其他颜色目标、光照和自然场景均未建立。

## 数据、预算与预检

- 实际RGB库728组双眼：旧672像素精确前缀＋V12/V13两新几何×4新姿态×7场景=56新验证图；新库独立审核和父实际开图已完成。
- 原720条训练frameIDs/targets/pointweights/episode_scale/采集发布时钟全部逐位复用neutralv6；240中性episode scale4、其余480 scale1，逐点weights仍sum1。没有增加训练图、数字镜像或改变采样频率。
- 同相机50Hz/20ms发布延迟，行为GRU5ms使用已发布held帧，完整9.6秒1920GRU步和481输入帧，训练不detach循环历史。360uniformpairs/batch4，freshAdam1e-4/fixed1000，仅final1000接受新验证。
- 真实preflight161项通过，包括生产/ScriptScan新opponent路径、全14参数梯度、临时Adam、显式scaledloss、pairedmean梯度、固定buffer非参数、独立NumPy像素算子等价、freshconv1与其余12继承身份及旧720数据精确核。预检峰值RSS912.195MiB，约2.987s；原tol1e-10不变。
- 正式1000更新用时884.597s，峰值RSS758.445MiB。保存0/500/1000并分别新进程从实际RGB同生产形状重放，模型/Adam/RNG/pair记录完整核验。实际pair162/249/257未取到，按预登记如实保存，未补训。

最终权重：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/TRAIN/checkpoint_001000.pt`

SHA256：`996980806ac107137d6d5f4743de207fb3350e84a8a145dc5f52b8807398bb87`。

## 新实际图像验证结果

48条普通验证加24条必需遮挡/纯罩控制全部原功能门通过，另8条固定画面仅描述，不要求turn0，也不把其列为新增闭环成功。独立验证审核integrity_all_pass=true；功能all_targets_pass=true。

| 配置类 | 数量 | 关键结果 | 原门 |
|---|---:|---|---|
| 空场 | 8 | turn RMS约.001237 | ≤.02通过 |
| 中心目标 | 8 | turn RMS最大.011905 | ≤.05通过 |
| 左目标脉冲 | 8 | 普通off恢复RMS最大.003916，方向/响应通过 | ≤.05及方向门通过 |
| 右目标脉冲 | 8 | 普通off恢复RMS最大.004148，方向/响应通过 | ≤.05及方向门通过 |
| 左→右 | 8 | 两个末段符号及响应门通过 | 全通过 |
| 右→左 | 8 | 两个末段符号及响应门通过 | 全通过 |
| 左可见→闭罩遮挡 | 8 | 已发布遮挡后恢复窗口通过 | ≤.05通过 |
| 右可见→闭罩遮挡 | 8 | 已发布遮挡后恢复窗口通过 | ≤.05通过 |
| 仅双罩无目标 | 8 | turn RMS约.001237 | ≤.05通过 |

训练侧原始像素诊断证明在当前渲染场景中灰地面和灰罩的对立响应为0、红目标非零，但输入RGB从未人工替换。新的有效输入域对齐是明确工程机制；不能由这些场景推断所有自然背景或其他颜色也能零响应。

20个训练保存姿态与4个验证保存姿态均来自同一条neutral身体轨迹，新几何/新索引的留出不是独立新轨迹或生物重复。720长序列是静态头部图像钳制组合，无法单凭本次离线通过证明实际运动头部闭环、自由视觉导航或任意时长稳健性。

## 下一阶段与保留限制

父已安排另一个子代理准备同一最终权重的实际身体8条件，必须先以真实SHA、当前新架构identity、独立离线通过及原motor789f绑定准入，再执行完整1+4.8+9.6秒、真实头部RGB、遮挡/固定帧对照。当前本报告没有身体结果，不声称视频或实际视觉闭环通过。

原近域中心/右→左/遮挡失效，远域与镜像及数据覆盖/中性训练失败全部保留；旧气味延迟切断差值门FAIL、原运动滑移FAIL亦不改。没有因本次离线通过追写任何旧结果。视觉气味融合、神经减速停驻、摄入、生物复眼或原连接组能力未建立。

## 权威证据

- 实际预检：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/PREFLIGHT.json`
- 训练实录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/TRAIN/RESULT.json`
- 最终功能结果：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/VALIDATION/RESULT.json`
- 独立审核：`/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/landmark_opponent_v7/policy/VALIDATION/VERIFY.json`

本次报告仅读取既有JSON文本与写报告，未导入数值库、加载模型/数组、运行forward/训练/物理，所有已冻结核心与运行输入均保持不变。报告不纳sourcefreeze。
