# 新大角度RIGHT三神经bias有限物理校准：四项均slip FAIL，selected=null

固定1代4候选已全部真实完成、分别独立审核，并按事前规则收口：没有任何候选同时通过全部门，因此`selected=null`，没有执行中性确认，没有第二代、扩扫描、额外训练或RIGHT重跑。所有job已collect，无运行任务。

这是新yaw/.06大角度RIGHT_ONLY任务，从最新LH455模型`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/left_hind_stance_refit/TRAIN/checkpoint_004000.pt`（SHA256 `ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff`）只改变三个神经常量bias：`head.lh_bias[1,4,6]`，经冻结XML/rc实际映射验证对应global head rows8/11/13、joint_LHCoxa_pitch/LHTrochanter_pitch/LHTibia_pitch。其余每个tensor精确保留。候选格式`RIGHT_THREE_BIAS_ACTUAL_V1`，optimizer=None、training_updates=0；父模型4000仅作为血缘记录，不冒充新训练。不是旧小幅LEFT CEM重开，也不是每步外置控制器。

PCG64 seed20261111，两独立标准正态3维方向各取±反向，mean0、sigma.005、component bounds±.02，固定顺序+d0/-d0/+d1/-d1。全部四个candidate SHA/proposal/source身份在任何actual之前冻结。36项零Core/零积分预检全通过，含RNG顺序/状态/范围、实际3关节映射、严格delta-only、zero-delta完整函数、neutral全2880 means/192hidden、parent RIGHT实际同输入192hidden/非目标输出严格一致、选择tie/null规则。

冻结证据：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/PROPOSAL.json`，SHA256 `58019b6ec00d0da201aa0b25158cfd084f631dedce8381fba90e6fcb36e80e13`。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/FROZEN.json`。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/PREPARATION.json`。

每项严格顺序CPU1、完整1s稳定+4.8s原straightteacher prime+9.6s ANN，RIGHT命令1s smoothstep至-1，不reset。源/身体/执行器/全部compiled数组不变：71opaque mesh/49qpos48qvel/42有限PD/native12.5us/noextra/noadhesion/运行rootwrites0。每项15400个1ms完整端点、2880个5ms实际神经决策/192hidden/action、geometry/handover，1,232,000原生积分步（ANN768,000）。四项合计4,928,000原生步，未生成巨量native数组，NF峰值每原生步轻量统计。

| candidate | yaw度 | max slip | world-X signed速度 | native ANN NF | 六腿LF/LH/LM/RF/RH/RM | 独立审核 | 行为 |
|---|---:|---:|---:|---:|---|---:|---|
| c00 | -30.6311489091 | .5094933642 | .08120266959 | 0 | 1/2/2/2/2/1 | 147全通过 | slip FAIL |
| c01 | -30.5601400374 | .5310892115 | .08175589992 | 0 | 1/2/2/2/2/1 | 147全通过 | slip FAIL |
| c02 | -30.8031838500 | .5113190277 | .08224349767 | 0 | 1/2/2/2/2/1 | 147全通过 | slip FAIL |
| c03 | -30.4436875925 | .5226272747 | .08122641124 | 0 | 1/2/2/2/2/1 | 147全通过 | slip FAIL |

四项的原support/nonfoot/COM/forward门、all6、strictnative NF0、实际signed RIGHT yaw≥30°均通过，只有原slip≤.5门失败。未把最大滑移最低的c00误称通过或selected。c00相对父模型.5098287866只降低.0003354224，没有跨阈值。

候选完整权重与SHA：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/c00/candidate.pt`：`04202be484ee6cf528b65ea3e8860f247bcbeba970975c9ee5018a8646902a8d`，delta[.004547481116,.002184460935,-.008565056804]。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/c01/candidate.pt`：`42e09d8da00ad2ed0f311ef700d3748d6c1c8c9e438329c201ff96a4ef4d9eb2`，上述反向。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/c02/candidate.pt`：`3b03df8f538f6c65334162c73d003fcd42eea49cde5063cc74ee1a124ad7f6ca`，delta[.004068864971,.003931456485,.002463336657]。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/c03/candidate.pt`：`59d572a83e09057f39979587fdd91be4ce28f996e85b49f102dc75b042e03bca`，上述反向。

各自同目录的EVAL_right完整结果和VERIFY_right独立报告记录所有实际数据；每项新进程重放全部2880生产means/192hidden/actions逐位匹配，完整1ms指标/周期/来源/感觉历史/动作持有/连续交接、原native计数/NF/extra、所有source/input/artifact SHA、有限PD及compiled完整身体不变全部核验。独立审核不重新积分，不做优化。

最终证据：

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/FINAL_SUMMARY.json`：actual_RIGHT_rounds=4，selected=null，selected_all_required_pass=false。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/SELECTED.json`：完整四项审核后按事前minimum(slip,index)过滤全部门的选择证明，selected=null。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/neural_right/right_three_bias_calibration/FINAL_AUDIT.json`：独立纯Python9719项全通过，绑定四项模型/轨迹/源码/身份/artifact SHA、精确同窗全部criteria、完整固定预算、selection proof SHA、selectednull及没有额外neutral。此最终审核零Core/零模型重放/零积分/零优化，实际神经重放由每项已绑定的147项独立审核完成。

运行：bash25零Core准备exit0并collect；bash26顺序4实际+4新进程独立审核pipeline exit0并collect；最终纯Python独立audit前台exit0。pipeline为无torch常驻的纯Python编排，始终只有一个单线程实际物理或重放进程；没有子代理。全部captured源/权重保持，不改旧baseline、小转网络、旧失败视频或用户裁定。没有待清理无用临时文件。

该family按固定4候选预算停止，不扩同family。当前已证明多项新RIGHT网络/候选可真实转约30°，但尚无本系列完整slip通过模型，原直行证据严格保留。限制仍是单条教师训练/标称初态/9.6s、替代ANN/原声明派生完整身体/有限PD；不是完整原生D1—D8或原生slip重新认证、原连接组/SI/生物有效性/任意扰动或食物趋近成功。

父提出的后续DAgger方向仅做过源码只读可行性识别，未新建标注实现、文件、模型或运行额外Core：uniformpref oracle action主要依赖实际qpos/qvel、接触几何/Jacobian、qfrc_bias/passive和actualtarget，以及从1s实际qpos初始化后连续递推的private q_des。现有TRACE可逐决策恢复实际状态并mj_forward零积分，理论可先验证960warm动作/几何/target链后产生CORRECTIVE_EXAMPLES。原TRACE没有直接保存private q_des/contact list/solverwarmstart，不能声称私有链与原始未保存数组直接bitexact；actualctrl为末native步起点值，不能由1ms末态重新反算相等。不得reset_from逐帧或把拟议teacherlabel积分进合成target；post标签未实际执行，不称qualifiedteacherPASS。是否实施由父另行预登记决定，本family不自动追加。
