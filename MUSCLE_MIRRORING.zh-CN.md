# 把肌肉从 1 条腿镜像到 6 条腿：做了什么、拿到了什么、还缺什么

## 一、"找别的项目"的结论：**没有**

当前唯一、也是最先进的肌肉模型就是 **Özdil, Ning, Phelps, Wang-Chen, Elisha, Blanke, Ijspeert, Ramdya,
"Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster"
（arXiv 2509.06426, ICLR 2026）**。它自己写着：

> **"We modeled 15 muscle-tendon units (MTUs) per foreleg"**

**FlyGym 里那 15 条肌肉的模型就是这一篇的模型，作者声明只做了前腿。** 排除理由也写了：tibia 里的肌肉因
X 光扫描未覆盖 tibia 而排除，trochanter 肌肉因功能不明而省略。真实解剖是**每腿约 19 条肌肉、约 69 个运动神经元**。

**而它给出了镜像的权威依据**：

> **"Notably, the muscle structure across legs is nearly identical, with a few exceptions including
> tergal depressor of the trochanter (TDT) muscles in the middle legs that facilitate jump escape."**

## 二、神经侧本来就齐了（实测）

BANC 里**六条腿的运动神经元全都有，按腿神经分开、按肌肉命名**：

| 腿 | 腿神经 | 肌肉命名运动神经元 |
|---|---|---|
| 前左 / 前右 | prothoracic | 29 / 29 |
| 中左 / 中右 | mesothoracic | 45 / 46 |
| 后左 / 后右 | metathoracic | 48 / 46 |
| | | **合计 243** |

而且各腿数目高度一致（`LFTibia_flex`→每腿 5 个、`LFTibia_extensor`→每腿 2 个、`LFF_trochanter_flexor`→中/后每腿 7 个），
**这本身就是镜像依据的独立佐证**。产物 `outputs/mn_muscle_map_per_leg.json` 里每条肌肉×每条腿都带神经元 ID。

## 三、镜像实施（`tools_mirror_muscles.py`）

生成 `outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml`：

- **90 条肌腱 + 90 条肌肉执行器，六条腿各 15 条**
- 自一致性：**231 个 site、0 重名、222 个引用全部解析、90 条肌肉执行器→肌腱引用全部解析**
- **编译通过**（经 FlyGym 的网格加载器）：`nq=14 nv=14 nu=90 ntendon=90 muscle_actuators=90`
- **与原始模型逐项对照**：

| 量 | 原始（1 腿） | 镜像后（6 腿） | 差 |
|---|---|---|---|
| nu（执行器） | 15 | **90** | **+75** |
| ntendon | 15 | **90** | **+75** |
| nq / nv / njnt | 14 / 14 / 14 | 14 / 14 / 14 | **0** |
| nbody / ngeom | 73 / 72 | 73 / 72 | **0** |
| 总质量 | 0.00249427 | 0.00249427 | **0** |
| 原 15 条执行器 | — | **全部保留** | — |

**只有执行器和肌腱增加，其它一个没动。**

### 镜像不是一一对应，两处被我实测出来并记录

1. **只有前腿有 trochanter 体段**：前腿 9 段（Coxa, Trochanter, Femur, Tibia, Tarsus1-5），中/后腿 8 段（**无 Trochanter**）。
   所以 trochanter 上的 site 映射到该腿的 **femur**，**24 个 site 受影响**，逐条记录在 `mirror_decisions.json`。
2. **各腿骨长不同**（实测 rest pose）：femur 前 0.575 / 中 0.784 / 后 0.836 mm；tibia 0.511 / 0.667 / 0.685。
   纯刚体变换会把肌肉留在前腿的**绝对**偏移上，所以每个 site 再按所在体段的**轴向长度比缩放**。
   **这是近似**：横向走的肌纤维本应只做轴向缩放，我用的各向同性缩放。
3. **7 条肌肉是胸肌，site 挂在共享的 `Thorax` 上**。用"体段到体段"的变换对它们**等于恒等变换**（根本不动，会被插到错误的腿窝），
   所以改用**以该腿 coxa 为锚的世界变换**（45 个 site 走这条路径）。

## 四、还缺什么：**中/后腿根本没有关节**

这是镜像之后暴露出来的**真正阻塞**。模型只有 **14 个关节，全部在前腿**：

```
joint_LFCoxa_yaw/pitch/roll, joint_LFTrochanter_yaw/pitch/roll, joint_LFTibia_pitch   (7)
joint_RFCoxa_roll/yaw/pitch, joint_RFTrochanter_yaw/pitch/roll, joint_RFTibia_pitch   (7)
```

**中腿（LM/RM）和后腿（LH/RH）没有任何关节** —— 它们的 body 是焊在胸上的静态几何。

意味着：

- **真收获**：**右前腿原本有 7 个关节却一条肌肉都没有，现在它有 15 条 → 真正可肌驱的腿从 1 条变成 2 条。**
- **没到位的**：中/后腿那 60 条肌肉虽然编译通过、结构正确，但**拉的是不能动的 body**，产生不了运动。

## 五、下一步（明确、可做）

**给中/后腿补关节**，按其自身解剖对照前腿的关节组（`Coxa yaw/pitch/roll` + `Femur pitch` + `Tibia pitch`，
**无 Trochanter 段所以 3+1+1**），4 条腿 × 5 = **20 个关节**。补完之后：

1. 中/后腿的 60 条镜像肌肉才有作用对象；
2. 每腿的肌肉↔运动神经元映射（`mn_muscle_map_per_leg.json`）才能接到具体关节上；
3. 才谈得上"整只动物的行为由神经元经肌肉产生"。

**在那之前，"6 条腿都能被神经控制"是不成立的，我不打算说它成立。**

## 六、诚实边界

- 镜像的 site 位置**对 5 条腿是推导值，不是实测解剖**；已写进生成文件的 XML 注释与 `mirror_decisions.json`。
- MN→肌肉映射是**按命名匹配推导**（`DERIVED`），权威表在 Azevedo et al.（bioRxiv 2022.12.15.520299），本次**未取到全文**，未经其核对。
- 源模型本身缺 tibia 肌肉、缺 trochanter 肌肉、**缺中腿 TDT**，镜像原样继承。
