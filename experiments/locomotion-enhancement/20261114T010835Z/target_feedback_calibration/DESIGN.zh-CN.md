# 一个有限条件性target-feedback训练候选（ONLYCODE）

本010835目录是新训练身份，保持fcbd的412encoder96／GRU96／head192／bound6／state352／8训练张量与部署运算完全不变，初始化所有父权重逐位复制。新FORMAT为INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1，实际父SHA为fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b。没有给部署step添加新算法控制器、速度乘零、时钟、教师root或额外神经输入。

## 条件性反馈的范围

12条训练序列与8条无标签变长DEV、监督mask／target全部沿用。外生physical112、oldafter256、task2与真实双精度qpos历史保留；源旧均值可精确分解为缓存old_nontarget_neural=directoldmean+frozen_gain*actualX126，然后条件oldmean_pred=nontarget−gain*clip((predtarget−q0)/halfspan,-10,10).float32.double。

新训练scan用该oldmean组成原412features，运行同encoder／GRU／head得到action；predtarget按真实2rad/s动作缩放、actual maxrate／bounds逐1ms五次更新，下一神经tick读取该预测寄存器。**预测target路径不detach**，旧权重冻结不等输入梯度截断。源身体q、qvel、载荷等没有随预测动作重新产生，所以始终标CONDITIONAL_NOT_PHYSICAL_COUNTERFACTUAL，不能据训练target误差宣称身体恢复。

forced_pre_targets模式每tick强制真实pretarget，只用于生产值一致性验证；普通训练完整1920自反馈，invalid监督行仍推进历史。float32转换保留真实感觉值／autograd近似，不用额外STE，不平滑clamp、不删饱和样本。42:84目标误差也在独立fullproduction反馈校验中按双精度actual_q重建；它目前不进可训练encoder，但不能因此忽略126:168通过旧prior和oldmean42的通路。

## 数据与联合loss

每条action监督仍是首20tick×10、其他×1与valid相乘后按各序列实际有效质量归一，再平均42／12。新增targetloss权重1：仅9条真实执行来源（原8+old_walk_control）每5ms末预测target对真实posttarget的半关节范围归一均方，除1920×42×9。三条assisted真实target积了混合动作，**绝不监督未混合teacher动作的累计target**，它们只参与原合格action mask监督，target_selector为0。

v1残差prior仍.1／1920×42×6，全部8参数对父初始化元素均方anchor*.001，微批按B/12分摊。B12／micro4×3／完整1920BPTT，freshAdam3e−4固定600，保存0/300/600、只final600资格；新预算wall5400s／RSS1800MiB，无自动加步、延期或多候选。

## 必须通过的预检

1. 部署参数和20条实际观察（12训练+8真实变长DEV）迁移与父一致；部署mean/state逐位核对，不改变actor。
2. Torch按真实actualaction逐1ms五次积分，对已实际通过的13来源target重放≤1e−12；真实q0／halfspan／bounds及float32 target编码精确。数据normalizer用既有identity_contract与compiled hash，不Core或float32反推。
3. forced真实pretarget的conditional scan对原生产step全12序列≤1e−12；自反馈scan与每tick完整old.step＋内部X两个target片段替换的独立fullproduction一致≤1e−12，old256逐位一致。Python/JIT全部六输出同门。
4. 数值输入扰动保留原契约：42:84和195:237不改变输出，126:168改变旧均值但old256不变；不是用autograd零证明wholemean独立。预测初target梯度canary必须有限非零。
5. 完整1920 B12与3批B4的loss／全部8梯度／Adam／参数差≤1e−10；两个坐标32tick中心差分epsilon1e−4，只作为量化cast近似诊断，绝对误差≤1e−3或相对≤.05，未过先FAIL不放宽；不能代替完整BPTT。
6. 实测完整micro更新×600×1.3，加三次真实setupcache／conditional与production12／DEV8／IO余量，预测≤5400s才READY_FIXED600_5400S；fullgraph及时释放，RSS上限不变。预检失败保留证据，父另行裁定，不执行后偷延时。

## 身份与存储

source严格绑定父实际212存储45×3与EVAL58资格、13源累积诊断actual VERIFY.source_hashes中执行363678版本及POSTRUN_WORDING_AUDIT；不把后期c7 CODECHECK当现场源码。所有旧source、mask、失败、数组、权重与科学清单只读。

common.load_checkpoint(path,digest,final=True)仍两返回，严格新FORMAT／600及非final0/300/600、canonical8 Adam计数／状态、RNG、旧789f冻结摘要。每保存live model／Adam／RNG精确roundtrip；新进程重放20条actualX和12条完整条件target反馈，缓存对完整生产≤1e−12，保存conditional NPZ并检查全部有限及权重不变，0Core。

final actualX拟合和conditional拟合分开；9条target normalizedRMSE、绝对峰和末误差单列，assisted3没有target监督分数。DEV8不造teacherY、只固定观察输出／状态描述。独立NumPy审核保存指标，无NN／身体／教师查询；低targetloss不是bodyPASS。

## 父执行接口

父读源码后标准库绑定：

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration/sources.py --prepare-source-metadata
```

独占CPU托管预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration/preflight.py --execute-parent-preflight
```

仅actual all_passed和timing READY_FIXED600_5400S后运行一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration/run.py --execute-parent-calibration
```

三主stage／三fresh保存审核，日志、命令、exit完整。actual最终SHA完成前未知；之后必须四项无教师身体同门验证，再脑ANN→真实DN2 queue→decoder2与full432感觉回归，不省略。本子没有执行任何数值库、数组、模型、Core、训练或物理；完整预检与效果待父实证。
