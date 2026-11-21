# 果蝇真实颈部闭环实验平台

本仓库以虚拟果蝇为验证对象，研究原身体退出后，双向颈部接口和学习式替代身体能否保持感觉运动功能。当前目标和下一项同脑实例切换试验见 [研究路线图](GOAL_ROADMAP.zh-CN.md)。

当前工作只限于果蝇仿真和软件实验。它不是意识证明、生物脑替代、真实动物植入方案或临床系统。

## 两个项目，严格分开

| 项目 | 脑侧 | 目标 | 当前入口 |
|---|---|---|---|
| 开源果蝇脑模型适配 | 已有开源脑模型；训练翻译层或有限微调 | 让原模型经真实颈部接口接收感觉、下达身体指令 | [`experiments/open-brain/`](experiments/open-brain/) |
| 人工大脑学习 | 独立训练的人工神经网络 | 学习意图、感觉整合与行为决策，并经真实颈部接口控制身体 | [`experiments/neck-real-structure/`](experiments/neck-real-structure/) |

两个项目可以复用身体物理模型、颈部来源数据和验收工具，但不得混用模型身份、权重、训练日志或验收结论。

P1 已将 OmniSim 单针的 BANC v888 损伤结果接入冻结的 Eon、嗅觉 SAC 与 FlyBody。研究入口收录单针诊断、受控镜像墙、果园迁移和左绕受损闭环的配对证据。[当前入口](experiments/open-brain/NECK_ODOR_CURRENT.json) · [联合实验报告](experiments/open-brain/20261119T202650Z-neck-odor-integration/REPORT.zh-CN.md)

P1 的替代身体可从仿真电极读出下行活动，并通过模拟刺激返回前足感觉。保留下行响应的收步候选在 4 个平地场景均达到摄入门槛，3 个完成摄入后站稳。这是短时仿真功能保持证据，切换期间和长期连续运行仍待验证。[接触后收步实验](experiments/open-brain/20261121T060600Z-postcontact-settling/REPORT.zh-CN.md) · [验收汇总](experiments/open-brain/20261121T060600Z-postcontact-settling/acceptance.json)

另有独立的 [BANC—肌肉身体研究](experiments/banc-muscle/README.zh-CN.md)，研究局部感觉、腹神经索到运动输出与六足承重。各实验的本地数据和模型权重随归档保存，部分运行入口依赖这些资产及固定版本的仿真环境。

## 强制架构

```text
仿真身体与其他感觉
        ⇅
真实来源的颈部神经结构
        ⇅
脑模型
```

- 脑—身体控制不得绕过颈部直接写入关节动作。
- 颈部神经元、纤维和连接必须具有可追溯来源；工程假设必须显式标注。
- 编码器和解码器属于两端接口，可以训练，但不能冒充真实颈部结构。
- 当前颈部可塑性关闭；固定传导延迟、突触电流和膜动力学仍含待标定参数。
- 解剖结构真实不等于生理动力学、植入接口或生物有效性已经验证。

权威约束见 [`NECK_REAL_STRUCTURE_REQUIREMENTS.zh-CN.md`](NECK_REAL_STRUCTURE_REQUIREMENTS.zh-CN.md)，当前状态见 [`NECK_CURRENT_STATE.json`](NECK_CURRENT_STATE.json)。

## 当前实证边界

### 已完成或已限定通过

- 派生完整身体上的站立、恢复、行走和转向基线已经形成；当前认可的限定行走结果见 [`WALKING_CURRENT_STATE.json`](WALKING_CURRENT_STATE.json)。
- 行走结果使用用户授权的替代 ANN、完整派生身体和有限 PD；它不等于原连接组直接产生步态，也不证明任意时长或任意扰动鲁棒性。
- 固定颈部动力学已包含逐通道传导延迟、突触电流和膜状态，并完成一组地面动作回归。
- 开源脑项目的身体侧已针对固定颈部重新训练，现有地面命令基线通过限定验收。
- 内感觉接口已实现于 [`engine/embodied/interoception.py`](engine/embodied/interoception.py)，独立检查入口为 [`verify_fly_interoception.py`](verify_fly_interoception.py)。

### 正在进行

- 开源脑项目已有低凸块物理教师以及据其训练的地形身体候选。候选在开发集和留出集均通过 5/6，尚未通过完整资格检查。
- 当前地形身体入口见 [`experiments/open-brain/TERRAIN_BODY_CURRENT.json`](experiments/open-brain/TERRAIN_BODY_CURRENT.json)，本轮协议见 [`experiments/open-brain/20261117T184709Z-body-terrain-repair/PROTOCOL.zh-CN.md`](experiments/open-brain/20261117T184709Z-body-terrain-repair/PROTOCOL.zh-CN.md)。
- 下一步是采集合格教师数据、加入偏离纠正、重训身体网络并复验原 69 项地面命令；之后才扩展到斜坡、粗糙地面、树干和墙面。

### 尚未完成

- 完整颈部神经结构及全部上下行通道的生理标定。
- 人工脑与真实颈部结构、完整身体之间的全链路验收。
- 飞行气动、起飞、空中转向和着陆；这些任务暂缓。
- 颈部在线可塑性、真实电极双向映射和生物实验验证。
- 意识、身份连续性或离体意识维持的任何实证结论。

## 身体侧与脑侧的职责

身体侧负责快速局部闭环：本体感觉、接触、负载分配、中央模式发生、足端附着、站立恢复和对地形偏差的快速纠正。身体网络不保存用于任务规划的长期记忆。

脑侧负责意图和跨感觉决策：目标选择、速度和方向意图、停止/后退、视觉或嗅觉导航，以及需要跨时间整合的信息。

颈部传递的是神经活动，不是抽象的“左转”字符串。工程接口负责在脑模型、真实来源颈部通路和身体读出之间学习可识别的编码。

## 关键入口

- [`BODY_MUSCLE_REVIEW.zh-CN.md`](BODY_MUSCLE_REVIEW.zh-CN.md)：身体、腿部几何、肌肉—肌腱和运动映射资产审计。
- [`RECOVERY_DESIGN.zh-CN.md`](RECOVERY_DESIGN.zh-CN.md)：人工大脑项目的恢复与三部分架构设计。
- [`WALKING_RESEARCH_PROTOCOL.zh-CN.md`](WALKING_RESEARCH_PROTOCOL.zh-CN.md)：行走训练与评估规范。
- [`GOAL_ROADMAP.zh-CN.md`](GOAL_ROADMAP.zh-CN.md)：当前研究路线。
- [`GOAL_TRACKER.zh-CN.md`](GOAL_TRACKER.zh-CN.md)：运行记录、证据和下一最小动作。
- [`engine/embodied/`](engine/embodied/)：共享身体、感觉和闭环实现。
- [`experiments/banc-walking-recovery/`](experiments/banc-walking-recovery/)：已验收行走的代码和证据。
- [`experiments/open-brain/`](experiments/open-brain/)：开源脑模型适配项目。
- [`experiments/body-motor-learning/`](experiments/body-motor-learning/)：身体运动学习实验。
- [`experiments/locomotion-enhancement/`](experiments/locomotion-enhancement/)：复杂地形与运动增强实验。

## 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-body.txt
```

不同实验可能固定了不同的 Python、MuJoCo、FlyGym 或模型版本。运行前应先阅读对应目录中的 `PROTOCOL`、`PREREG`、`RUN_CHECKLIST` 和 `REPORT`，不要用一个环境静默重放全部历史实验。

内感觉基础检查：

```bash
.venv/bin/python verify_fly_interoception.py
```

## 证据纪律

- 训练、验证和最终重放使用不同身份，保存权重后必须独立加载复验。
- 失败结果保留在 Git 历史或本地实验目录，不改写成成功。
- 大型权重、原生逐步轨迹、视频和下载数据不进入 Git；仓库只保留实现、协议、报告与小型汇总证据。
- 任何“真实”表述都必须区分连接组结构、工程动力学假设、生理测量和行为验收。

## 仓库清理说明

旧的 wound-calcium、逐轮 `S*`、`AUTONOMOUS_LOOP_*`、`ARENA_FIX_*` 以及一次性根目录分析脚本已从当前版本移除。它们没有被改写；需要历史复核时可从清理前提交 `362369566be365b9290416b122f855cef26bb069` 读取。

当前根目录只保留项目入口和强制资产。新的实验实现应放在对应项目目录中，不再把轮次脚本或报告堆到仓库根目录。
