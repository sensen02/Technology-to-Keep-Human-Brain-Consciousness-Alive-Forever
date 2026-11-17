# 空间气味源 C 阶段最小接口设计（不可执行）

状态DESIGN_ONLY。父当前唯一heavy为候选3 physical_B 六回合，尚未据完整B功能门准入C。本目录当前只本文本设计，不生成场景执行预登记、不可绑定策略资格、不加载模型/数组/Core、不运行前向/物理/训练。

## 准入与六布局提案

只有新behavior SHA66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034与冻结motor789f的完整B六回合全部功能通过并各自完整性审核true，才可由父确认下一C执行scope和preparebinding。旧sensor或旧B失败不能被覆盖或改PASS。

| ID | forward offset d | bearing angle |
|---|---:|---:|
| C_left_near | .6 | +.18rad |
| C_right_near | .6 | -.18rad |
| C_left_far | .65 | +.28rad |
| C_right_far | .65 | -.28rad |
| C_front | .6 | 0 |
| C_noodor | 不产生浓度 | 无 |

d为前向offset，lateral=d*tan(angle)，不是径向距离。source sigma.5、amplitude1.25、patch_radius.06，ANNelapsed.6出现/.2smoothrise，完整9.6秒ANN，不增加停车控制。

.65取代早期.8建议是在C任何回合前依据旧neutral几何所做的开发设计选择，不能冒称验证结果后筛选成功场景。父已保存SPATIAL_BASELINE_SCREEN：neutralheadproxy净位移约.6847375，但总弧长约.9303478；因此不能用净位移断言.8绝不可能，也不能把总弧长当一定可达。近侧.6/±.18及远侧.65/±.28 patch不被无刺激中性路径自然进入，只是可辨识的必要几何信息，不是闭环可达充分证明。front自然可能入patch，不能拿它单独证明感觉依赖。

执行前需将父几何筛查原文与同motor源身份绑定：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/SPATIAL_BASELINE_SCREEN.zh-CN.md`。

## 最小运行派生

在新scope中最小复用已冻结physical_B核心，不改旧执行器/身体/权重/时序：

1. 唯一替换浓度来源，由B钳制pair改为环境spatial adapter。
2. 5.8s交接时，用实际头部采样中点与水平head+X/left basis计算worldcenter一次；之后固定worldcenter，不随身体移动。原head local_xyz为唯一采样点，图与感受一致。
3. 每5ms动作前从当前实际head位姿算两点worldXY，采样固定Gaussian浓度，沿用原C/(1+C)/tau.05/5ms causal pending，动作t使用之前区间响应，不用futureC。
4. 行为只2受体+32memory，运动原238+耦合命令+256memory；source坐标、时钟、bearing、距离与inside仅环境诊断，不增部署输入或脚本动作门。
5. 同1ssettle+4.8teacher+9.6ANN、交接不reset、原native步非足/extra/计数汇总、完整trace+288chain保留。原工程滑移FAIL独列，不替换阈值。

建议独立source adapter提供：`anchor(actual_head_position,actual_head_rotation)`、`concentration(actual_sensor_xyz,ANN_elapsed)`、`diagnostics(actual_sensor_xyz,ANN_elapsed)`。只有环境读取时间和源参数，actor只接输出两个receptor数。

## 保存字段与复核

沿用B所有文件/schema，仅增加环境字段：

- neural_decisions每5ms：`sensor_xyz_now` (2,3)、`source_center_xy` (2)、`source_anchor_valid`、实际C（原`concentration`2已存在）。
- geometry每1ms：原sensor_xyz/midpoint外加source_center_xy、source_anchor_valid、center_distance、edge_distance=max(0,center_distance-.06)、inside_patch。
- 源未锚定的prime字段用0配validfalse，不写NaN、不偷看未来anchor；所有指标只处理validtrue的ANN窗口。
- 撤辅后anchor恒定，并绑定HEAD localxyz和原始handoverhead姿态；新进程audit独立重算anchor、每5msGaussianC、受体pending、每1ms距离、arrival、原cycle/全部物理指标及实际288神经链。

到达：源onset .6后实际inside连续200个1ms采样才确认.2s，任何确认都不停仿真、不置零动作、不重置状态。报告d_start（handover时proxy到patch）、d_min、d_end、firstentry、confirmedarrival、maxdwell、entry/exit次数、是否入后再离开/overpass，避免把穿过说成停车或摄入。ANN仍完整跑满9.6s。

## 无气味复用与预算

优先证明C_noodor与新B_noodor感受输入/初始filter/行为与运动状态/完整prefix/执行环境相同，源参数只存在环境诊断而不影响actor。若可严格证明，独立C_noodor_REUSE.json只引用B权威真实trial与hash，不复制虚构新轨迹、不算重复物理，C六layout提案对应uniquephysics5。若任何身份或确定性证明不够，允许新跑第6个C_noodor但不超预算6。

当前不为了复用加载数组或构造证明。真实复用决定与具体JSON模式由父在B结果收齐后新预登记冻结。

## 尚不确定

- B真实身体的off命令恢复和方向切换尚待六回合终判，离线通过不足以绕B门。
- 相同Gaussian场在移动头部上产生的强度/差分可能与静态训练域不同。
- 半幅运动滑移偏高仍会影响闭环方向效率，A功能通过不等于高质量运动。
- 新C是开发回合、先不训练；成功也不等于感觉依赖确认，留出D与匹配对照后续另冻。

本设计不冒称代码、场景资格、空间趋近或摄入已完成。
