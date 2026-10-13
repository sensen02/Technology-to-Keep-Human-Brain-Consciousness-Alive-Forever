# 运动输出普查：16个肌肉在结构上无法被神经驱动

**手工研究原型，不是成品。** 只读已发布run的manifest与保存的放电/命令数据，不重算池、不调参、不新建控制器。

## 一、结构容量

| 类别 | 数量 |
|---|---:|
| 驱动器总数 | 90 |
| **完全没有神经池（不可被指令）** | **16** |
| 有池但全程零放电 | 7 |
| 实际被驱动 | 67 |

- 池细胞合计251个，200ms内共**1,283个脉冲**（平均每个池细胞约5个）。
- 40个命令区间中有2个区间全部90个命令为0（对应最初的感觉→运动潜伏期）。

## 二、不可被指令的16个肌肉——全部是髋部肌束

| 肌肉 | 腿 |
|---|---|
| pleural_promotor | LF, LM, LH, RF, RM, RH（六条腿全部） |
| tergopleural_promotor a/b | LM, LH, RM, RH（四条腿各2个） |
| sternal_adductor | LM |
| pleural_remotor_and_abductor | RM |

即**中腿与后腿的髋部前推/后拉肌束没有任何神经通路**，LE/RF虽同样缺pleural_promotor，但其tergopleural_promotor可被驱动。这些肌肉的命令因此在任何感觉或网络动力学下都固定不变。

## 三、有池但零放电的7个

RM trochanter_flexor（7细胞×2）、accessory_trochanter_flexor（RF/LM/RM各3细胞）、pleural_remotor_and_abductor（LH/RH各1细胞）。属于"存在但本episode未被驱动"。

## 四、拮抗肌驱动不对称

| 腿/关节族 | 无驱动的方向 | 各方向脉冲数 |
|---|---|---|
| LH/coxa | promotor, remotor | adductor 12, rotator 17 |
| LM/coxa | adductor, promotor | remotor 1, rotator 49 |
| RH/coxa | promotor, remotor | adductor 12, rotator 18 |
| RM/coxa | promotor, remotor | adductor 2, rotator 54 |
| **RM/trochanter** | **flexor** | **extensor 58, flexor 0** |

## 五、与前几轮审计的收敛

1. **右中腿**：支撑审计发现它跗节离地0.073mm、初始垂直力为0；本次普查发现它的**转节屈肌驱动为0**。相关而非已证因果，但值得作为可检验假设。
2. **髋部控制**：力矩审计发现崩塌时最缺的正是coxa_yaw/roll；本次普查发现髋部promotor/remotor肌束在多数腿上**结构上不存在神经通路**。两者指向同一处。
3. 因此"命令模式与时序错误"有了具体结构成分：**不是调错了，而是有一部分输出映射根本缺失**——这也是为什么调增益不会解决问题。

## 六、下一步（需先补证据，不靠调参）

1. 对tergopleural/pleural promotor这16个**unresolved**池：核对注释来源（`source_annotation`记录了"target absent"的原因），判断是注释缺失还是确实无对应细胞；若是注释缺失，须找可核验来源补池，**不得靠猜测**。
2. 对零放电池：区分"池存在但网络未驱动"与"池细胞无可用连接"，需要单独诊断。
3. 在补全输出映射之前，任何站立结论都会被"部分肌肉不可控"限制，报告必须带此限定。

## 七、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_motor_output_census.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_motor_output_census.py`（5项通过）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/motor_output_census/motor_output_census.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/motor_output_census/motor_output_census.png`

边界：单一episode与单一tier；池成员为既有注释、未重新推导；脉冲数为已发布5ms计数；共享池（8对）仍是"未经验证的募集假设"，本次不据此推断募集规律；本普查不是能力成果，站立判据未变。
