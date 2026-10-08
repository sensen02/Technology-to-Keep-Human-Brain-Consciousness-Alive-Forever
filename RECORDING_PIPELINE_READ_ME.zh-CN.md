# 数据记录全链路 — 先看这一页

**这是手工搭建、可复现、已测试的工程原型，不是成品，也不是经过认证的测量系统。**

## 1. 这一轮回答了什么

| 你的要求 | 结论 | 证据在哪 |
|---|---|---|
| 1 果蝇能正常行动 | ✅ 16 段全部正常行走：平均着地腿数 4.5–4.7、胸高 ~0.9 mm、速度 17–19 mm/s | `RECORDING_PIPELINE.zh-CN.md` 第 2 节 |
| 2 电极会记录它的信号 | ✅ 记录轨迹在步态频率处的能量是**纯噪声通道的 10.2 倍** | 同上第 3 节 |
| 2′ 电极自重影响行动 | ✅ **同种子配对**，5 % 与 20 % 负载都显著改变了速度、路程、胸高、着地腿数；但**幅度不是单调的**（5 % 反而更大），本轮**不下**"越重越慢"的结论 | 同上第 1.3–1.5 节 |
| 3 能从外部相机/图片重建行为轨迹 | ✅ 固定相机 **240/240 帧全检出**，去偏置后误差 **0.32 mm（约半个像素）**，x 相关 **> 0.9996**；近距相机 0.77 mm | 同上第 4 节 |
| 4/5 根据运动轨迹逆推触觉 | ❌ **做到了 chance 水平但没有超过它**（0.742 vs 0.757，16/16 段）。原因已定位、已量化 | 同上第 5.3 节 |

**第 4/5 项是负面结果，这是本轮最重要的诚实交代。** 原因不是检测不行，
而是单目相机在 60 fps、团块只有 10×20 px 时**测不出逐腿相位**；
我另外用 240 fps 单独验证过，仍然不锁定，所以这不是"再调调参数"的问题。

## 2. 五个文件怎么用

```bash
cd /run/media/sensen/Data2/cell_wound_prototype

# 1) 录制：仿真 + 电极 + 两台相机（需要 body 环境，有 MuJoCo/FlyGym）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  venv_body/bin/python run_recording_pipeline.py --seeds 0,1,2,3 --seconds 4 \
  --fps 60 --width 240 --height 320 --loads 0.20,0.05

# 2) 只从照片重建（项目环境，这个脚本不 import MuJoCo、不读真值）
for cam in worldcam detailcam; do
  venv/bin/python run_recording_vision.py \
    --episodes 'outputs/electrode_payload/episode_*_seed*' --camera $cam
done

# 3) 与真值比对（唯一同时读两边的文件）
venv/bin/python run_recording_verify.py --seeds 0,1,2,3

# 4) 图 + 报告
venv/bin/python run_recording_figures.py --seeds 0,1,2,3
venv/bin/python tools_recording_report.py

# 5) 自检（负载注入的算术 + MuJoCo 侧断言）
venv_body/bin/python tools_test_electrode_payload.py
```

## 3. 先看哪几张图

`figures/` 目录里：

1. `fig_payload_seed0123.png` — **电极自重的影响**：右图每条线是一颗种子，
   5 % 与 20 % 负载都把速度拉低，配对关系一眼可见。
2. `fig_trajectory_seed0123.png` — **只从照片恢复的轨迹**（红）叠在仿真真值（灰）上，
   右下角是位置误差曲线。
3. `fig_camera_strip_detailcam_seed0123.png` — 近距相机的真实帧，
   六条腿看得清；这是"腿画得出来"的证据，也说明为什么"测得出足端高度"是另一回事。
4. `fig_electrode_seed0123.png` — 电极记录轨迹与它的频谱。
5. `fig_touch_seed0123.png` — 触觉推断的**失败**：左下那张相位扫描曲线
   说明任何全局相位偏移都救不回来。

## 4. 三个必须一起读的限定

1. **电极轨迹是模型量，不是实测记录。** 声源用的是仿真自己的执行器驱动通道，
   驱动到电压的耦合常数是**声明值**。它回答的是"按本工程的 7 µm 记录链噪声预算，
   这样的电极能记到什么"。
2. **7 µm 电极本身的重量对运动没有可测影响**（＝体重的 0.0095 %）。
   有影响的是夹持器/导线——那部分在本工程里**没有几何定义**，
   所以按体重百分比扫参数，真实果蝇能拖动多少负载本工程**没有**实验依据。
3. **位置误差里有一个常数偏置**（1.45 mm），它来自"轮廓质心是整个身体的中心，
   而真值是胸部原点"这个定义差别，不是跟踪误差。去掉它才是算法的真实表现。

## 5. 完整清单

| 文件 | 内容 |
|---|---|
| `RECORDING_PIPELINE.zh-CN.md` | 结果（所有数字自动填入，无手抄） |
| `RECORDING_PIPELINE_METHOD.zh-CN.md` | 方法、层级标注、踩过的坑 |
| `comparison.json` | 16 段的全部指标 |
| `recording_index.json` | 录制清单（负载、帧数、质量指纹） |
| `body_mass.json` | 未加载体重（负载的分母） |
| `figures/` | 6 张图 |
| `frames_example_*/` | 两台相机各 6 张原始帧 |
| `*.py` | 全部脚本 |
