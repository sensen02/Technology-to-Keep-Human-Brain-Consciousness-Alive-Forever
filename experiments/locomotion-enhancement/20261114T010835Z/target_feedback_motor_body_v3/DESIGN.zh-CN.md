# Target-feedback身体组件：同一草稿迁移CAL v4（ONLYCODE，S2/S5）

ROOT仍为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_v3`；目录名保留v3，实际CAL唯一指向 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4`。

状态 **FIRSTREADY_CAL_V4_MATCHED_FINAL600_UNKNOWN**。父明确授权只在此前未READY、未执行草稿中迁移，不新增重复目录。KIND沿TARGET_FEEDBACK_ORDER_REPAIR_MOTOR_COMPONENT_BODY_V1，MODELFORMAT INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1，scope **BodyANNcomponent_NOTnewbrainneckdeployment**。新final600 SHA未知，prepare不接受猜测或祖先final资格，不绑定。父84续训运行状态不推断新存储或行为PASS。

旧最早冻结bodycomponent/v2cal/v3cal及所有科学数据不改。v2装饰helper globals入口失败、v3完整预检末RSS1800门失败均保留。v4只修资源scope：fullB12参考预检RSS3072MiB，训练micro4与存储审核仍1800MiB、wall5400s、数学/数据/精度/300新增更新不变。3072绝不能写成训练允许。父82actualv4预检92checks全true，报告peak1863.84MiB归参考预检；本子只读其JSON元数据，不运行数值。

新adapter按固定SHA读取最早冻结body adapter/prepare AST，将CAL字符串v2→v4，__file__使ROOT落此目录。实际bodyentry仍直接编译原161918 common/preflight/runner/audit/run AST，没有多层物理循环克隆。source3同一common globals；warm2/settle0、原所有数值常量不变。逆CAL身份与metadata读取适配后prepare AST精确恢复，旧adapter逆CAL也精确；所有函数新ROOT与source3核实。

**actualv4 metadata与口头简称不同，必须按实际文件适配**：actualv4 INPUT_BINDINGS只有resource_amendment，没有namespace_recovery或resume_lineage。小范围_v4_lineage先核resource_amendment五字段exact(参考3072/训练1800/wall5400/旧v3无序列化PREFLIGHT)，再核真正v4 PREFLIGHT自己的92checks、resource_scope和两阶段RSS门。其后从actualv4 data_hashes已SHA固定的v3 INPUT_BINDINGS读取原namespace_recovery六字段，从同一映射固定的v2 INPUT_BINDINGS读取原resume_lineage；三者恢复checkpoint/path/SHA相同且读前后再次hash。此处只读祖先声明，不要求v3 PREFLIGHT存在或allpass，也不向v4捏造字段。

上述只是metadata兼容。原strictprepare仍独立要求actualv4自己的PREFLIGHT同source/data/READY_RESUME300_TO600、新TRAIN IDENTITY原AdamRNGanchor、RESULT total600/new300/already300、JOBS三阶段exit0、STAGE及最终600 actualSHA。新保存300/450/600各combined20actualX+12conditional六输出≤1e−12及CONDITIONAL数组SHA，EVAL conditional12/actualX12/DEV8与独立EVALVERIFY全部真实通过。祖先0PASS与old300FAIL/b562原checkpoint separately保留，不能升旧失败为PASS或要求不存在新0。完整8Adam计数600/矩、原anchor与RNG/PTidentity由strict common.load_checkpoint(final=True)进一步数值校验；metadata不冒PTpayload已检查。所有source/data/proof beforeafterhash后exclusive写新body INPUT_BINDINGS，无覆盖。

body模型public352/obs238+cmd2/412encoder/GRU96/head192/8tensor/frozen789f生产部署与旧一致，targetfeedback训练scan不进入body控制。四case walk[1,0]、stop[0,0]、L[0,1]、R[0,-1]，1秒settle200无NNstep＋4.8秒960prime_old[1,0]仅old256/new96h0＋9.6秒pure完整step_details/tanhmean；5.8实际物理state/target/buffer/previous/352不reset、教师完全撤除，无beta或混合。

完整71opaque/free-root42有限PD/native12.5us/noadhesion/noexternalforces/noactualrootwrite。原100ms连续COM<.8或NF>.5W早停，partial失败保留无retry。原门：walk原五reference含forward.05/slip.5加六腿周期/NF0/full；stopdrift≤.1BL、过渡.5秒后XYRMS≤.01BL/s、absYaw≤5deg/support.9/COM1.2/NF0/full/cyclesNA；pivot同向yaw≥45deg、drift≤.5BL、六腿load-unload-reload+安全全窗。legacyforward/slip每例单列，不修改任何历史门。

公共preflight严格新final600先NN后physical、1Core0step/352finite42/8Adam600/old789f/71opaquePD/bounds；之后固定4phys＋4freshNN352sourceaudit＋summary9stage，max4×1232000=4928000native。全保存productionreplay/独立指标/cycles/native/identity/torque沿旧。若未来保存schema错仅newrecovery读已存结果，不重物理。

本子标准库AST/JSON/sourcehash/受保护import与functionglobals检查，0Torch/NumPy/NPZpayload/PTload/model/NN/Core/积分/训练/渲染。父actualv4final600资格完成前不执行下列后3命令。ACTUAL必须替真实新finalSHA：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_v3/static_check.py

/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_v3/prepare.py --bind-completed-final600 --sha256 ACTUAL

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_v3/preflight.py --execute-parent-preflight

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_v3/run.py --execute-parent-fourcases
```

末条父singlemanagedCPU1收全部9stage。FIRSTREADY源冻结后不再编辑。父统一主tracker；完整人工brainNN→realDN2queue→NNdecoder2/full432感觉集成未部署不可省，身体组件不是成品或生物/电极/意识证明。
