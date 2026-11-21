# P1 完整颈束 226 针电流写入试验复跑

使用已落盘的 full5、当前冻结颈部归档、OmniSim 源树与 HIP 构建。主世界重新推进植入过程到落座，将一个物理触点映射为实际尖端或齐平位点，屏蔽开路触点。四个条件各自有独立电极模块和事件队列，读取同一份受损材料和逐轴突损伤状态。该工程模型允许叠加多个独立导体电流。静默、单侧和双侧测试均只从电极产生外场。主世界记录的终点为颈部脑侧边界，未接入 Eon 全脑或身体。

下列命令重新生成本目录的派生结果。独立验证可把 Python 文件复制到新的目录后在那里执行。

```bash
cd /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261121T005613Z-patch226-write
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
"$py" -Bu choose.py
"$py" -Bu live.py
"$py" -Bu live.py --replay
cmp registered.json registered-replay.json
"$py" -Bu background.py
"$py" -Bu readback.py
python3 -B summarize.py
```

归档的生物结构、full5 初态与核心实现使用 SOURCE_SHA256 校验。OmniSim 在 omnisim/modules/electrode_observation.py 提供 pulse_pattern，同时电流先叠加后判阈，保留原双极接口。回归测试为 tests/test_electrode_observation.py。
