# 全部并进一个场景：17 图层完成

时间：2026-10-04
性质：**手工开发并测试的研究原型**；无实物、无实验；未改动实验侧（滤波、ADC、噪声、硬件接口一律未动）。

上一次清单里的六项 + 三维，**全部做完并接进同一个场景**。场景现在是 **17 个图层、6.98 MB**（上限 31 MB），场景回归 **140/140**，浏览器零错误。

---

## 1. 逐项结果

| 项 | 做法 | 数据来源 | 载荷 |
|---|---|---|---|
| **三维多层组织** | 棱柱堆叠：平面用已验证的六角网格，厚度方向一维柱 | `tissue3d.py` 现算 | 35 KB |
| **二维顶点力学（含切口）** | 真实尺度贴腹部，31 帧逐帧更新，**死亡细胞（NaN）按标记跳过不画** | `mechanics_cut_n24.npz`（1248 顶点） | 453 KB |
| **断颈/眼损五臂对比** | 眼/脑/颈近/颈远/体电位 + 刺激 + 记录伪迹，逐帧表 | `integrated_*.npz` × 5 | 95 KB |
| **场景环境** | 地面贴图（**CC0 署名随附**）+ 天顶/天底渐变色 | `assets/natural_ground_photo.png` | 23 KB |
| **胞内空间** | **场内读数**（D_Ca、守恒偏差、收敛阶、大 D 极限） | `metrics_cellspace.json` | 含于 3.3 KB |
| **血管氧** | **场内读数**（按间距的均值/最低/缺氧占比） | `metrics_organism.json` | 含于 3.3 KB |
| **电极链诊断** | **场内读数**（前端 6/2/1、刺激 11/1、记录 77 项） | `electrode_*.json` | 含于 3.3 KB |

**为什么后三项是读数而不是画出来的场**——这是查过数据之后的结论，不是省事：

- **胞内空间**是**单细胞**径向/网格模型（`metrics_cellspace.json` 的 scope 自己这么写），组织里没有对应的空间场；
- **血管氧**的产物只有**按血管间距的汇总统计**（mean/min/max 氧分压、缺氧占比），**没有二维或三维氧场数组**（我逐个文件找过，`outputs/**` 下没有任何 oxygen 场数据）。用这些统计量画一个场，就是画一个从未被计算过的东西；
- **电极链**本身就是标量与判定，不是场。

所以我做了 `layer_readouts` 把它们**钉在同一个场景里**，并在图层说明里写明"无空间场"。产物里有一条测试专门断言**没有任何伪造的场键**（`readouts: no fabricated field key anywhere`）。

---

## 2. 本轮修掉的七个 bug（都写进注释）

1. **场景服务的文件名正则不允许点号**（`^[A-Za-z0-9_]+\.(bin|json)$`）——`ground.jpg` 与 `data.js` 全被 404 挡掉。这解释了为什么之前地面贴图一直加载不出。已允许 `.jpg/.png` 并加 MIME。
2. **力学补丁导出全是 NaN**：源数据用 NaN 标记**已移除的细胞**（868/38688 分量），而 `v.mean(axis=0)` 对含 NaN 的数组返回 NaN ⇒ 质心是 NaN ⇒ **每个顶点都被污染**，导出的 31 帧没有任何有限值。改用 `np.nanmean` 并**保留 NaN 作为死亡标记**。
3. **客户端把死亡细胞画在错位置**：NaN 顶点保持上一帧坐标。改为移到视野外（`FAR`），并在图层里统计每帧死亡数。
4. **五个场景的子序列长度不同**：`stimulus_source_nA`/`artifact_recorded_mV` 是 4800 点而电位是 4801 点，用同一个索引会**越界一位**。改为每个序列按**自身长度**重建索引。
5. **场景载荷缺 `frames` 字段**，前端读到 `undefined`。
6. **场景数值表太宽**被面板裁掉，且 HUD 无滚动、无 monospace。改为三位有效数字定宽 + `white-space: pre` + 可滚动。
7. **测试对图像分片的校验**用了数值 dtype 表，且把 NaN **按分量**计数（出现 9.33 个"顶点"）。改为按魔数校验图像、按**整顶点**计死亡数。

---

## 3. 场景现在的全貌（17 层）

身体骨架与网格、脑+腹神经索（39,988 体细胞）、突触（24,000）、逐神经元活动、六足接触力与 CPG、下行指令、固定 7/20 电极阵列（450 通道）、电极↔神经元成员数、上皮伤口钙、慢生理账本、旁分泌配体场、血细胞与碎片清除、二维顶点力学（含切口）、断颈/眼损五臂对比、场景环境（CC0 地面）、无空间场的层读数、三维多层组织。

**一个坐标系、一个时钟、一个页面**：能定位的都在身体坐标系里随体运动；没有空间坐标的（O₂ 账本、五臂电位表、胞内空间/血管氧/电极诊断）是**同一场景内的读数面板**，不是另一个页面。

---

## 4. 全量回归

| 套件 | 结果 |
|---|---|
| `node tools_test_scene.cjs` | **140/140** |
| `python3 -m unittest test_workbench_backend -q` | 24 tests OK |
| `python3 test_body_export_regression.py` | PASS |
| `node test_viewer_modules.cjs` | PASS |
| `venv/bin/python paracrine.py` | PASS |
| `venv/bin/python immune.py` | PASS |
| `venv/bin/python source_separation.py` | PASS |
| `venv/bin/python ecm.py` | PASS |
| `venv/bin/python tissue3d.py` | PASS |

服务 http://127.0.0.1:8766/ 健康，17 图层。

---

## 5. 声明（不因图层变多而改变）

- 逐神经元活动是**真实连通组上的工程传播**，不是实测放电；
- 脑/VNC 是**标志点配准**、内部层是**示意嵌入**，不是解剖注册；
- 五臂场景是**同一协议下的仿真**，不是实验条件；记录伪迹与刺激是仿真量；
- 三维多层的单元格是**棱柱**，厚度变形**放大 2000× 才可见**（增益已声明）；
- 电极 7 µm / 20 µm 固定未优化，电学参数仍为假设，硬件测量 **0 次**；
- 源分离**必需但未实现**（本轮未重新尝试）；
- 所有新增参数均为 illustrative，来源在各自 `PARAM_SOURCES`；
- 不声称解剖注册、生理同步、意识或存活。

---

## 6. 产物

- 新增：`scene_layers_sim.py` 的 `export_mechanics_patch` / `export_scenarios` / `export_environment` / `export_readouts`
- 修改：`serve_workbench.py`（图像分片与 MIME）、`scene_build_cli.py`、`viewer/scene/scene.js`、`viewer/scene/hud.css`、`tools_test_scene.cjs`
- 场景数据：`viewer/scene/mech_pos.bin`、`scenarios.json`、`environment.json`、`ground.jpg`、`readouts.json`、`tissue3d*.bin`
- 本文件：`/run/media/sensen/Data2/cell_wound_prototype/ALL_LAYERS_STATUS.zh-CN.md`
- 截图：`/run/media/sensen/Data2/cell_wound_prototype/viewer/scene_screenshot_final.png`
