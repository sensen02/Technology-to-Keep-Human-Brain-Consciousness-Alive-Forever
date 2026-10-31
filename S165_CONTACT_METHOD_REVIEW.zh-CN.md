# S165 / P1 原生软接触静态逆关系独立方法审查

## 范围和实际执行

本审查为独立、只读、离线方法审查；不评价尚未收到的 S165 新提取数值。新增 MuJoCo 导入、model load、forward、inverse、step、physics、neural 调用均为 **0**。仅读取本地官方头文件和 S154 已保存 NPZ，以 NumPy 检查数组类型与形状，并计算来源 SHA256；不使用网络、不修改任何历史文件。本文件是新增方法报告，不是站立或预热成功证据。

初次查找命令出现退出码2（候选目录不存在），随后系统 `python` 不存在退出127；改用项目明确的 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B`，只执行 NumPy 读取，退出0。项目已有多项修改，未触碰。另对 `/tmp` 的源码查找因系统私有目录权限报错，未绕过。未找到本地官方 native C / MJX constraint 实现，不宣称已读 C 实现。

## 本地官方声明与来源

1. `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mujoco.h`：487–488行 `mj_referenceConstraint` 计算 efc_vel、efc_aref；490–494行 `mj_constraintUpdate` 计算 efc_state、efc_force、qfrc_constraint，可选 cone Hessian；明确 cost=s(jar)，jar=Jac*qacc-aref。SHA256：`a7ea9ac88b128afd3f2055451972d38a8374df2a3600c3ffc3371dd5a24e2c81`。
2. `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mjtype.h`：511–520行类型0 equality、1 DOF friction、2 tendon friction、3 joint limit、4 tendon limit、5 frictionless contact、6 pyramidal contact、7 elliptic contact；523–529行区分 quadratic、friction linear 两支、elliptic cone cost 状态。SHA256：`e13006a26de8d2636e22de23acf398b1b25f61bdc9543eeaac8247c8f9be067a`。
3. `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/mujoco/include/mujoco/mjdata.h`：330–344行声明 efc_type/J/frictionloss/KBIP/D/R，D为constraint mass、R为inverse constraint mass；407行 aref为reference pseudo-acceleration。SHA256：`5842f00188e714821ca2fdcddd457b43a25a7b31b352648545fe3ff6078f16ca`。

安装包库名为 libmujoco.so.3.9.0；本次未调用版本 API。上述声明支持 API 语义与类型。下述 cost/projection 是数学推导，不冒充本次逐行 C 实现核验。

## 已保存 S154 实际约束

来源 `/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/CONSTRAINT_ROWS.npz`，SHA256：`6b253b4f4e2c8714f3c8e62bcb053569b4090c6d334bfca25c52331858e2f143`。

实读 sample000–sample084，共85快照，各72个 efc rows，所有 efc_type=6；合计6120行均为 native pyramidal，无 equality、frictionloss、joint/tendon limit、elliptic 行。各快照 J_dense 为72×48，18 contacts。STOP_FULL_STATE 已保存 qpos49、qvel48等，不需加载模型确认该形状。本结论只覆盖这些保存快照，不保证 S165 零速重算后约束集合不变。

此历史 NPZ 未保存 efc_R、efc_D、efc_aref，不能仅靠已有 f/J 数组唯一恢复零速原生软力；须使用生产者新增合法提取，不得补造。

## 推导：处方零加速度下的原生软力

令 a=qacc，j=J a−aref，D=diag(1/R)，要求有效 R_i>0。对于 native pyramidal 每个非负 edge，标量代价为

    s_i(j_i)=1/2 D_i min(j_i,0)^2
    f_i=−∂s_i/∂j_i=max(−D_i j_i,0)

因此处方 a=0 时：

    f0_i=max(D_i aref_i,0)=max(aref_i/R_i,0)
    qfrc_constraint0=J.T @ f0

这不是将一般 forward 求解错误地解耦：forward 中未知 a 通过 J a 和质量方程耦合所有约束；本题已处方 a=0，原生 inverse constitutive relation 在给定 j 下逐标量可求。不能用 efc_AR 或其对角替代原生 regularizer R，也不能另加 J M^-1 J.T 后再计算这一步逆关系。质量矩阵仍属于正向加速度问题；在静态平衡 Ma=0中消失。

pyramidal J 已内含 contact frame、摩擦系数和正负 edge basis，edge变量不是直接的normal/tangent分量。原生 μ 不应再次以任意圆锥替换、重复缩放或额外猜测约束；物理接触力应按同一 native edge mapping 解码。

### 类型例外与适用矩阵

| efc_type | 零加速度力 | 不能省略的条件 |
|---|---|---|
|0 equality|aref/R，可正可负|不能截成非负|
|1、2 frictionloss|clip(aref/R,−loss,+loss)|loss须来自 efc_frictionloss，非contact μ|
|3、4 joint/tendon limit|max(aref/R,0)|仅已构建的有效limit行|
|5 frictionless contact|max(aref/R,0)|有效标量法向行|
|6 pyramidal contact|max(aref/R,0)|逐native edge，保留native J/μ|
|7 elliptic contact|耦合的R加权friction-cone投影|不可逐分量max或简单clipping|

统一写法为 f0=argmin_{f∈K}[1/2 f.T R f−aref.T f]。K对 equality 为实数，对frictionloss为有界区间，对scalar unilateral/pyramid为非负半轴，对elliptic为物理椭圆摩擦锥。R为正时该力唯一。零速度 frictionloss 的 aref 通常为0，从有限R软律得到0，而非允许任意区间内的静摩擦力；不应把刚性stiction假设悄悄放进exact-soft测试。

## 静态可行性与rigid relaxation必须分开

必须在相同q、qvel=0下取得 J/R/D/aref/passive/bias/actuator force coefficients。aref含速度参考项，不可直接使用动态aref冒充零速；passive和肌肉velocity dependence也须在零速重新取值。

若实际广义actuator力在候选box上严格仿射，记为 B act+b_act，则原生软接触静态平衡为

    B act + b_act + qfrc_passive + qfrc_external
      − qfrc_bias + J.T f0 = 0
    0 ≤ act ≤ 0.5，或单列 0 ≤ act ≤ 1

其中 f0固定，不是优化变量。qfrc_external须完整包含qfrc_applied与Cartesian xfrc_applied的广义贡献。矩阵应覆盖全部48 DOF，不能只检查根部/竖直力，不将root DOF2冒充COM。若存在actuator force clipping、joint actuator force limit或其他非仿射操作，须明确处理，不能仅凭肌肉增益形式声称整个box全局线性。肌肉力bias与qfrc_bias是不同项，不可漏加/重复。

另一个以 f≥0 自由变化的 LP 保留原生pyramidal几何/摩擦可行域，但删除固定pose的软接触constitutive relation，应明确命名 **native-pyramid rigid-force relaxation**。其可行不能证明原生软静态可行；其不可行可作为较强的条件性排除（前提所有力项/边界/rows均建模正确）。零速度exact-soft不可行但rigid可行，可能只是该穿透量固定了不合适的软预载，不意味着其他邻近姿态也无法支撑。

推荐输出LP求解状态、所有48平衡残差、box违例、变量/行尺度、容差及必要的不可行证书/残差下界；不把数值solver一句infeasible当严格数学证书。自由根部行若actuation对其完全无作用，可先检查其固定外力残差是否已经排除平衡。

## 无额外engine的验证建议

在生产者已提取的动态样本上计算

    f_pred=max(D*(aref−J@qacc),0)

与实际 efc_force 比较；另核 J.T@efc_force 与 qfrc_constraint。finite-iteration forward可能有solver残差，应报告差值与容差，不能要求所有模型都machine-exact。再对零速样本计算固定 f0，做LP/残差分析；无需追加constraintUpdate或inverse调用。保存type、D、R、aref、frictionloss、J、contact μ和row mapping，可独立重算。

最后，act静态力平衡只是瞬时机体平衡；完整常状态还需act_dot=0及兼容允许ctrl（例如实际activation filter可能要求ctrl=act，须核实际模型），并不证明神经命令可达、稳定、真实单位标定、预热或自主站立。质量/惯量/力SI含义仍 **NOT_ESTABLISHED**；不能因native数值闭合就宣称生物/单位资格成立。
