# P1 新版果园嗅觉 SAC 迁移测试

使用 orchard.blend 的 ground-003，在果皮两侧分别取真实表面点。两处出生点均相距对应果皮点 17.6 mm，入口世界方位分别为 195° 和 270°。每块保留 48 × 40 mm 原尺度土面、可接触的局部果皮及相交的草叶或细枝。开阔入口按中心线近乎水平，带草叶／细枝入口到果皮前中心线下降约 1.12 mm。官方 Blender 场景与既有权重哈希保持一致。

土面从原三角面裁剪成碰撞体，果皮、草叶和细枝用源三角面派生薄碰撞体。材料接触参数采用现有 FlyBody 土面基准。两次预检的土面射线与源采样最大差分别为约 0.00012 mm 和 0.00012 mm。果皮取食点在土面上约 1.19 mm 和 1.05 mm，碰撞射线对齐误差小于 0.00003 mm。两块地面都从同一已稳定的关节快照开始，根的姿态与高度按局部坡度配准。同一块的气味对照共用准确初始姿态和神经种子。

部署使用上一轮导出的同一 SAC 权重。左右触角从实际所在坐标读取工程静风扩散，距离查询只扫到身体周围的土面、草枝或果皮。气味源贴在可食斑块旁的最近自由网格单元，偏移 0.45 和 0.57 mm。RGB 保留为诊断，控制每 80 ms 根据双触角、短时变化和近距避障线索更新。冻结 Eon 和 BANC 实际发放到达，身体继续使用同一 FlyBody。口器需要进入指定果皮斑块，与果皮距离不超过 0.55 mm 并有足端支撑才可从该处扣减库存。

第一轮站姿预检复用了动作数组。FlyBody 按自己的接口原地叠加翼部控制量，造成额外翼运动。正式闭环沿用逐步传入独立动作副本的入口。按该接口修正预检后，两个裁剪点均在末 80 ms 保持立姿、至少六足接触和平移速度低于 0.015 mm/s。

## 实际闭环

统计将起步、摔倒、口器近接、库存变化和末端站稳分开。图中蓝线为实际胸部轨迹，绿色点为起点，红色点为实际果皮点，虚线表示取食斑块的水平范围。

开气味后两个入口都有下行信号实际过颈，两个关气味对照在原地站稳。开阔入口的最高前进距离为 2.20 mm，0.5442 s 出现跌倒。带草叶和细枝入口跑满 6.4 s，最高前进距离 1.50 mm，最大平面偏移约 3.07 mm，期间未触发跌倒，末段没有通过停稳门。全部回合摄入为零。两个开气味主回合均没有接触草叶、细枝或果皮，也没有离开几何裁剪区。最近口器分别离取食斑块中心 15.74 和 16.25 mm。

开阔入口用同一源地形的高度格交叉复验。该表示的射线高程与原采样最大相差约 0.008 mm，起始站姿及嗅觉接口相同。果蝇前进 8.42 mm，1.6902 s 摔倒，摄入仍为零。与三角面结果相比，步行距离和失稳时间对土面碰撞表示敏感。两种表示的现有证据都显示进食前发生失稳。图中紫虚线为高度格复验，果皮色块包含对高处果皮的俯视投影。

两个主配对在气味首次采样前各有 32 行物理轨迹、1600 拍神经记录逐值一致。主开气味组有 97 和 1348 个具名下行到达，高度格开气味组有 374 个。关气味组在含同一真实食物的几何中均为 0 个过颈下行，胸部最大平面位移约 0.013 和 0.056 mm。301200 原生物理步和 150600 神经采样中，全部 17 路下行延迟及库存守恒通过。所有行为回合保持原模型权重和零额外外力。

对这一个果实的两处入口，当前门槛属于开始行走后的支持与推进。后续可在同一果园裁剪场地重放同一过颈驱动，优先定位起步换脚和坡地接触，再复测绕草与近果皮取食。全园、多目标、风中羽流及飞行留待独立验证。

## 复验入口

[原始配对统计](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer/ORCHARD_AUDIT.json) · [地形与轨迹](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer/ORCHARD_TRANSFER.png) · [站姿和几何预检](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer/PREFLIGHT.json) · [开阔入口的源坐标](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer/CLEAR/SOURCE.json) · [带草叶入口的源坐标](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer/DEBRIS/SOURCE.json) · [复验清单](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T194948Z-orchard-odor-transfer/RUN_CHECKLIST.zh-CN.md)
