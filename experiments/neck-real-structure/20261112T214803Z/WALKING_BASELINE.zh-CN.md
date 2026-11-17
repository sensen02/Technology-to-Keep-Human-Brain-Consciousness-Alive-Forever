# 新三部分架构中的既有行走基线登记

用户要求开始人工脑—真实结构颈部—其他部分的新模型，并把已做过的行走在这里登记。本登记不重训、不重跑物理，不将旧架构通过转移为新架构通过。

## 已做过并有证据的行走

- 完整派生身体71个不透明mesh，49qpos/48qvel/42有限增量PD执行器，真实仿真重力、接触和摩擦，无附着、额外托举或根轨迹写入。
- 控制器为替代人工网络SlowNeuralFeedbackGRU128，238维工程感觉输入、42输出；不是原连接组，也不包含已验证的真实颈部结构。与后来的256维双向转弯组合运动网络区分。
- 1秒稳定＋4.8秒教师预热后连续9.6秒ANN自主阶段，1,232,000原生物理步、2880神经记录，1920条自主更新；交接不reset。
- 原五项工程门：支撑1.000000049933006≥.90；原生非足0；COM最低1.2561640816296098≥1.2；有符号前进.0713190529277483≥.05；原生滑移.49411203866010717≤.5，全部PASS。这里数值为原模型单位，不代表独立SI／生物有效性。
- 六腿完整周期LF1，LH/LM/RF/RH/RM各2；两个固定初始脉冲确认通过，不能外推任意扰动、任意持续时间稳健性。
- 原始D1—D8按预登记方法验收，保留旧D7 UNKNOWN与保守充分证明、D2/D6方法历史，不能简化成无条件生物认证。

## 权重与具体结果

最终权重：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/BC_GRU_ROLE_3000/checkpoint_002500.pt`

SHA256 `246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809`。

验收：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/FINAL_ACCEPTANCE.json`

原生审核：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500_AUDIT.json`

原同次视频：
`/home/sensen/Desktop/engine修复验收/行走改进实施-20261110/自主行走验收通过-2500步-同次原生轨迹.mp4`

原视频实际物理15.4秒，770帧、30fps播放约25.67秒(约1.667倍慢放)，不要与后来视觉50fps视频混同。原验收已经全解码并实际目视核对。

## 本次实际核验

运行 `/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261112T214803Z/verify_walking_baseline.py`，exit0，12检查全部通过：权重、原生审核、视频和视频清单的实际文件SHA与归档值一致，两个确认回合RESULT SHA一致，原门/时段/链记录核对成立。输出WALKING_BASELINE.json。本次没有重新构造身体、推理、仿真或解码视频，不把归档审核说成新物理实验。

## 在新架构中怎么使用

- 可作为旧工程基线、仿真教师／目标运动参考。
- 不得把42维旧动作直接从新人工脑送到执行器，绕过真实结构颈部。
- 身体侧若暂复用旧运动网络，须显式登记其自主步态贡献；任务相关脑控制必须来自颈部下行端口，并以切断／置乱对照核验。
- 颈部结构、上行反馈与接口落实后，新人工脑通过它训练；需要重新做完整身体及原行走门和接口干预，旧PASS不能继承。

当前：旧基线已验证登记；新三部分模型 **NOT_ADMITTED_NECK_SOURCE_PENDING**，没有新训练更新。
