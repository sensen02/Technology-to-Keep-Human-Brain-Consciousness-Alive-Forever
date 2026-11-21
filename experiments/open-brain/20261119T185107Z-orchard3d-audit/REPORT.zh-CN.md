# P1 新版果园与嗅觉审阅

2026-11-19。范围为开源 Eon / BANC / FlyBody 的觅食闭环，以及 `/home/sensen/orchard-3d/simulation` 作为新场景的接入条件。

## 食物可以从视野外开始

上一轮半小时 SAC 的在线观测只有 RGB 果实候选、沟沿视觉、前足反馈、步伐和饱食相关量。开局无视觉候选或近期记忆时，当前导航入口给出零推进和转向。四个带坡回合的静止结果属于该视觉配置。抬高果实的视觉轮廓后双方起步，仍在第一道沟跌倒。增加嗅觉可以检验搜索起步，跨沟和承重仍要独立验证。 [观测输入](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T084744Z-closed-loop-forage/common.py:111) · [当前无目标行为](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T084744Z-closed-loop-forage/controller.py:35) · [上一轮对照](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T154718Z-compound-sac/REPORT.zh-CN.md)

早期 P1 有独立的双触角采样和头侧嗅觉翻译。空场仅嗅觉的单次试验在 0.78 秒首口，累计摄入 0.7805。短墙仅嗅觉的单次试验绕过墙端，口器最近 0.706 mm，未进入 0.60 mm 的摄入半径，摄入为零。两条探索性双感官短墙回合有摄入并存在撞墙。已成功的配置走头侧工程翻译，经 PFL3 驱动冻结 Eon，随后经过具名 BANC 下行。直接同时驱动 DM1／VA2 和 PFL3 的旧探针有推进抑制，仍需单独标定。头部触角读数在头侧接入，身体感觉和运动下行继续按各自的颈部路径接线。 [旧嗅觉消融与闭环](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T231627Z-open-fruit-loop/REPORT.zh-CN.md)

## 新版果园实测

Blender 场景按米建模，目录档案有 75 枚落果和 62 枚树冠果。落果 AABB 最大轴约 94.5–170.5 mm，树冠果中心离世界零高 2.52–5.13 m。卷叶、碎石、草叶、树枝都有可查询几何。五条 60 Hz 回放沿预设路径。归档常速的落果路径长 0.752 m，历时约 3.48 秒，属于预设飞行速度。碰撞验收使用直径 2 mm 的运动学球，归档有避让间隙、叶片摆动和五条无碰撞路线。目前复眼观测为两眼各 721 个小眼，按绿色和蓝色通道采样，P1 的 RGB 红色候选和蓝色沟沿标记需要重新标定。果园的 `wind` 驱动叶子摆动，当前观测没有触角浓度，果实标注提供位置和包围盒。落脚承重、果皮附着、可食斑块、口器摄入及库存需要对接 FlyBody。 [场景接口](/home/sensen/orchard-3d/simulation/README.md) · [场景既有验收](/home/sensen/orchard-3d/simulation/evidence/validate.json)

复验使用同一颗 `ground-001`、朝向果心的水平头部，每组 24 个方位，日光，叶摆幅参数 0.65。高度从各起点土面量起，144 个静态位姿中有两处需要排除。表中命中指至少一个采样射线击中该果。编号只用于离线计数。

| 距果心平面距离 | 眼高 2 mm | 眼高 40 mm |
| --- | --- | --- |
| 100 mm | 23/23 | 24/24 |
| 250 mm | 24/24 | 23/23 |
| 500 mm | 20/24 | 22/24 |

500 mm 低眼高的四次目标漏视，中央对照射线分别先撞到其他苹果两次、枯叶一次和碎石一次。其他苹果仍有零星或明显采样。100 mm 低眼高的 180° 点与枝条相交，250 mm 高眼位的 15° 点经果皮奇偶射线检查位于邻果内部。原场景的最近表面筛选会漏过内部点，两个位置都在本表中排除。命中指标仅统计射线交点，视觉分类成功率仍待测。此次采样限于单颗地果、有利朝向和静态头位。全场随机姿态、FlyBody 步态、空中停落和摄入留待闭环验收。 [静态原始结果](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T185107Z-orchard3d-audit/GROUND_VISIBILITY.json) · [归档统计](/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T185107Z-orchard3d-audit/SCENE_ARCHIVE_AUDIT.json)

## 接入顺序

先选能稳定站立、可绕行、能平稳取食的一小块落果地形。给果皮上的同一个有限库存斑块配置局部气味源。环境在实际头姿下采两触角浓度，有风时再给头部相对气流观测。可以先用静风扩散建立对照，再加平流、间歇羽流和反转风向。在有风的行为数据里，走路的果蝇闻到苹果醋会迎风，丢失气味会转入局部搜索。模型可用这个作为训练行为参照。位置和果实身份只给环境和事后评价。 [果蝇步行嗅觉导航实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC6103744/)

首个闭环从同一站姿、相同食物和相同碰撞几何，配对完整感官、仅嗅觉、仅视觉、双关四组。在视线遮住可食目标时，分别记录开步、绕障、口器实际摄入量和站稳。记录是否有跌倒和是否能重获气味。用 Eon 生成运动指令并经 BANC 下行。小块地面成功后，逐步加远路和多果背景，最后才加树冠飞行停果与长期能量账本。半米路程以 10 mm/s 的直行量级约需 50 秒，回合时长应按路程和绕行余量设计。

## 复验命令

使用 Blender 4.5.16 LTS CPU 双线程，下面的命令运行两个独立进程，JSON 逐字节一致。探针、比较、归档统计和场景四项 SHA 均退出 0。实际打开毫米近景图。无新增行为积分或训练。

```bash
run_dir='/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T185107Z-orchard3d-audit'
orchard_dir=/home/sensen/orchard-3d/simulation
orchard_blender=/run/media/sensen/_dde_data/orchard-3d-work/tools/blender-4.5.16-linux-x64/blender
env OPENBLAS_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$orchard_blender" --disable-autoexec --background "$orchard_dir/orchard.blend" \
  --threads 2 --python-exit-code 1 --python "$run_dir/probe_visibility.py" -- \
  "$orchard_dir" "$run_dir/GROUND_VISIBILITY_REPLAY.json"
cmp "$run_dir/GROUND_VISIBILITY.json" "$run_dir/GROUND_VISIBILITY_REPLAY.json"
sha256sum -c "$run_dir/SCENE.SHA256"
```
