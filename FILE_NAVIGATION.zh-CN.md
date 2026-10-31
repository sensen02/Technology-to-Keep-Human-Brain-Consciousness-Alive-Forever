# 当前项目文件导航

本轮为S2主线做保守分类整理，不删研究证据、不改模型/接口、不运行仿真。

## 当前入口（保留原位）

- 目标路线：`/run/media/sensen/Data2/cell_wound_prototype/GOAL_ROADMAP.zh-CN.md`
- 当前状态与历史：`/run/media/sensen/Data2/cell_wound_prototype/GOAL_TRACKER.zh-CN.md`
- 神经—身体闭环：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py`
- 引擎：`/run/media/sensen/Data2/cell_wound_prototype/engine/`
- 当前验收：`/run/media/sensen/Data2/engine修复验收/`（小写s2-系列仍原位）
- 当前约束定位：`/run/media/sensen/Data2/engine修复验收/s2-initial-constraint-localization/`
- 运行环境：`/run/media/sensen/Data2/cell_wound_prototype/venv_body/`

## 历史归档

- 历史外置证据：`/run/media/sensen/Data2/项目文件归档/history-evidence/`
- 散落历史辅助工具：`/run/media/sensen/Data2/项目文件归档/loose-tools/`
- 桌面历史交付：`/home/sensen/Desktop/engine修复验收/_历史归档/`
- 原始候选清单：`/run/media/sensen/Data2/项目文件归档/MANIFEST.json`
- 实际迁移清单：`/run/media/sensen/Data2/项目文件归档/EXECUTED_MANIFEST.json`（只读封存目录若移动受限则保留原位，不改权限）
- 迁移过程记录：`/run/media/sensen/Data2/项目文件归档/JOURNAL.jsonl`
- 完整性验证：`/run/media/sensen/Data2/项目文件归档/VERIFICATION.json`

旧路径保留兼容符号链接，因此报告引用、数据读取和输入SHA路径继续有效；不能保证归档脚本的运行兼容性（自身目录、相对import或Path(__file__).resolve()可能受影响），重跑历史脚本前应按清单还原。链接不是新增副本。本轮不靠搬家节省空间，也不将历史失败数据称为垃圾。

## 暂不搬动的对象

项目根目录脚本/协议文件数量较多，但其中存在import、Path(__file__)、相对资源及交付引用，直接移动并放链接不能保证Python的resolve语义不变，所以本轮保留。模型/数据/文献/outputs、锁文件、环境、vendor/upstream与旧zip/快照也保留，不凭年龄或大小判断无用。其他点云和Windows项目不在本次范围。

## 安全回退

按迁移清单逐项确认旧路径是指向对应新路径的符号链接后，仅移除该兼容链接，再将归档实体rename回原路径。任何原路径已被真实文件占用或链接不符时停止，不覆盖。归档内文件未改写，原证据哈希不变。
