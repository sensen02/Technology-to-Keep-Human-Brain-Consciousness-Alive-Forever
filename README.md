# Technology to Keep a Human Brain Conscious — 果蝇级原型工作台

> 目标（原作者一句话，原样保留在文末）：用电极记录神经元活动，摄像机记录运动，计算触觉，在生物正常生活中训练能取代身体的 AI 模型，再分离大脑，人工调配血液供给，虚拟世界模拟，用训练好的 AI 去电极刺激等，维持意识和正常生活。

本仓库是这套设想在**果蝇尺度**上的**无头（headless）工程原型**：不是成品，而是把每一层都做成**可测量、可证伪**的实验台，并把测量结果与失败一起记下来。

---

## 现在能跑通什么（都有实测数字）

| 层 | 内容 | 状态 |
|---|---|---|
| 身体 | FlyGym `NeuroMechFly`（MuJoCo 3.9） | 质量 **1.02431 mg**，与真实成蝇一致（差 3%） |
| 场景 | 草地（三层高度场：20 µm 枯叶层 + 显式草叶脊 + 莲座；CC0 贴图带出处 sidecar） | `grass_scene.py` / `meadow_scene.py` |
| 六相机 arena | r=22 mm 环、fovy 50°、12 个静态 fiducial（三层） | 静态重建 RMS **0.0249 mm**（对半切分标定，held-out） |
| 肢体 | 20 个自发光机载 marker + 四闸门身份安全跟踪 | 六条腿 tibia 关节角中位 **0.74–1.45°**（对仿真器真值） |
| 触觉 | 接触力 + 穿透 + 几何面积 `A = π(2Rh − h²)`，Hertz / Greenwood-Williamson 双区 | 面积–穿透 α = **0.954, r² = 0.999**（未封顶段）；**力/压强不可引用**（见下第 2 条） |
| 神经/组织 | 七层神经-组织栈，隔离 / 多速率场景 | 见 `ALL_LAYERS_STATUS.zh-CN.md` |
| 工作台 | 本地服务（8766），场景清单 `/api/scene/manifest.json` | 可视化在果蝇**体内**，不另开面板 |

**平台对现实的逐项核对见 [`PLATFORM_VS_REALITY.zh-CN.md`](PLATFORM_VS_REALITY.zh-CN.md)** —— 那里同时写着哪一项对得上、哪一项对不上。

---

## 平台对现实：核对结果摘要

**对得上**
- 质量 1.02431 mg（3%）、体长 2.87 × 3.00 × 1.11 mm、腿节长度（来源即 NeuroMechFly 对真实雌蝇的测量）
- 摔倒前的步态：每帧 **4.91/6** 条腿着地、**0%** 无腿着地的帧、单腿占空比 0.69–0.95

**对不上（必须写在最前面）**
1. **果蝇在 t ≈ 0.65 s 翻倒仰面躺死，再没恢复。** 所以 **3 秒级记录里 80% 是躺着不动的果蝇**；行为统计只能在 **t < 0.45 s** 的窗口里做。
2. **站立姿态下六足法向力之和是体重的 13–16 倍**——**已解释**：读自由关节的完整力平衡可见，约束力（地面顶上来 +221）与执行器力（伺服把身体往下拉 −240）**等大反向**，二者都是体重的 10–25 倍。其中只有约 **4.5%** 是真正的"承载体重"。所以触觉记录的"地面反作用力"主要是**控制器把果蝇按进地面的内应力**，**绝对力/压强数值不可引用**（但原因已明确）。几何路线（面积 ← 穿透）不经过力，不受影响。
3. **控制器是开环的，所以果蝇永远不会自己站起来。** `CPGController.step(self)` 不接受任何观测（无朝向、无接触、无反馈），只是"相位 → 预编程角度"查表。仰面躺倒后它继续同一节律蹬腿，没有翻身反射。**这与第 2 项同源**：开环位置伺服持续把腿往地里压，正是它把自己弹起翻转的原因。重力本身已三重验证正确（自由落体 a = −9810；已知质量静止 Σ|Fn|/(m·g) = 1.0000；约束力与我逐接触求和逐帧吻合）。

4. **没有粘附模型**（真实果蝇靠爪垫附着）；接触面积上限 `max_footprint_fraction = 0.5` 是**声明值**，而干净窗口里 **37.7%** 的接触已撞上限，对这部分面积只是下界。

**曾经报错、已更正**（保留，不删）
- "行走 GRF = 11,400 × 体重" → 实际 **均值 2.89 ×**（单位换算错误：模型质量单位是**克**）。
- "80% 悬空 ⇒ 在弹跳" → 实际是**摔倒躺地**，不是悬空。
- "tibia 骨长 0.961 mm" → 那是 tibia + 跗节链；解剖学 tibia 只有 **0.518 mm**。
- marker 半径按**米**传入（模型长度单位是 **mm**）→ 亚微米球，等于没渲染。这是"marker 看不见"的真正原因。
- 加 marker **body** 会把 `c_rostrum` / `c_haustellum` / 两只眼**删掉**；改成在已有 segment body 上加 geom。

---

## 快速开始

```bash
cd <this repo>
python3 -m venv venv        && venv/bin/pip install numpy scipy matplotlib pillow  # 解析侧
python3 -m venv venv_body   && venv_body/bin/pip install -r requirements.txt       # 仿真侧
```

```bash
# 平台对现实的核对
OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_reality_check.py

# 录一集六相机（草地 + marker）
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python run_arena_record.py \
    --seeds 3 --seconds 0.15 --fps 300 --width 800 --height 640 --rigs bare \
    --gl egl --marker-radius-mm 0.12

# 肢体关节角（四闸门身份安全跟踪）
OPENBLAS_NUM_THREADS=1 venv/bin/python limb_tracking.py --episode outputs/arena/arena_bare_seed3

# 触觉
OPENBLAS_NUM_THREADS=1 venv_body/bin/python run_tactile_record.py --rigs bare --seconds 3
OPENBLAS_NUM_THREADS=1 venv/bin/python tools_test_tactile.py     # 33 项自检
```

**两个环境是分开的**：解析侧（numpy/scipy/matplotlib）不需要 MuJoCo；仿真侧需要 FlyGym。凡是 import `flygym` 的工具都必须在 `venv_body` 里跑。

**`MUJOCO_GL=egl` 必须在 python 启动前设好。** `BodyBackend(gl_backend=...)` 会设这个环境变量，但在某些 import 顺序下 MuJoCo 已经选定了后端，症状是离根因很远的 `GLFWError: X11: The DISPLAY environment variable is missing`。

---

## 模型单位（踩过坑，所以写在这里）

长度 = **mm**，质量 = **克**（果蝇 1.02431e-3，即 1.02431 mg），重力 −9810，因此
**`m·g = 10.0485` 就是模型自己的"一个体重"**，任何静息姿态下六足法向力之和都必须等于它。

- 把半径按**米**传进这个模型 → 小 1000 倍 → 亚微米球（"marker 看不见"的根因）。
- 把模型力单位当 mN → 得到"11,400 × 体重"这种假数字。

---

## 测量纪律（本项目最重要的一部分）

1. **每个参数都带出处标签**：`DERIVED` / `DECLARED` / `TAKEN` / `MEASURED`。声明值与测量值不混用。
2. **预注册的判据不回退**；被推翻的结论**保留**并标注替代它的新结论（见上面的"曾经报错"）。
3. **说不出来的就写"未解决"**，不猜。上面那 13 倍力偏差就是 unresolved 状态。
4. **长任务成果必须看图验证**，不是看日志。
5. **样本交付到桌面**，仓库只留代码与小 JSON：`/home/sensen/Desktop/cell_wound_prototype/`。

---

## 不要声称

- 解剖学配准（CNS 位置是 landmark 配准；脑/VNC 划分只是显示划分）
- 与真实果蝇的生物学同步
- 测量到真实的果蝇神经活动
- 意识 / 身份 / 存活
- 硬件就绪（bench 状态 `NOT_RUN`，`measurement_count = 0`，`readiness_claim = false`）
- 触觉与真实果蝇的相似度（见"平台对现实"第 2、3 条）

## 物种 / 数据纪律

BANC 与 FAFB 是**不同标本**，不做 ID 或坐标拼接。BANC 为 CC BY 4.0（doi:10.7910/DVN/7WTH1N），使用需署名。密钥文件（`.flywire_api_token`、`.flywire_cookies.txt`）**永不入库、永不外发**，已在 `.gitignore` 中。

## 未做

Spike sorting / 源分离；飞行、嗅觉、肌肉、睡眠；灌流闭环（P9）；P7 三维多层（超出棱柱堆叠）；P8 免疫/再生；P4 未批准；**束缚张力**未建模（只有 `tether_supported` 标志）；实验室/仪器侧（滤波、ADC、噪声预算、硬件接口）**有意不做**。

## 下一步（按证据排序）

1. **修摔倒**：0.65 s 后仰面躺死，使所有长记录不可用。
2. **解释 13 倍力偏差**——很可能与第 1 项同源（`solref` 过硬 → 约束力虚高 → 弹起 → 翻转）。两个问题一起查。
3. 修完后重录 tactile，并把每个记录**截断到干净窗口**，再谈触觉相似度。
4. 然后才是粘附模型与接触面积上限的物理化。

---

> 原方案（保留）：大概方案就是用电极记录神经元活动，摄像机记录运动，计算触觉，在生物正常生活中训练能取代身体的 AI 模型，再分离大脑，人工调配血液供给，虚拟世界模拟，用训练好的 AI 去电极刺激等，维持意识和正常生活。
