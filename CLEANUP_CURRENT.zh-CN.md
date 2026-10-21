# 当前清理记录与项目概览

> 本文件记录本次**保守、只读式审查**。没有删除数据、实验输出、历史文档、依赖环境、第三方目录或用户已有文件。

## 1. 审查与实际清理

子任务独立副本不包含主工作区忽略的缓存，因此其初步检查为空；主任务在最终测试后复查实际工作区，删除了下列可再生成的 Python 缓存（没有进入依赖环境、vendor/upstream、data/outputs 清理）：

- `__pycache__`：1,809,749 bytes
- `engine/__pycache__`：274,545 bytes
- `engine/embodied/__pycache__`：248,472 bytes
- `workbench/__pycache__`：39,873 bytes
- `bench/__pycache__`：22,061 bytes
- `diagnostics/__pycache__`：54,416 bytes

机器清单：`outputs/experiment_platform/cleanup_current.json`。
没有删除用户已有修改、数据、实验结果、历史文档、第三方资产、虚拟环境或运行时锁。缓存会在后续运行时重新生成。

## 3. `.gitignore` 检查

现有 `.gitignore` 已覆盖本项目需要忽略的可再生内容：

- `venv/`、`venv_body/`、`.venv/`
- `vendor/`、`data/`、`outputs/`、`upstream/`
- `__pycache__/`、`*.pyc`、`*.pyo`
- `MUJOCO_LOG.TXT`、凭据文件、常见大媒体与数组输出

本次未改写该文件，以保留已有规则、注释和用户意图。

## 4. 项目模块概览

### 4.1 根目录解析/实验模块

- **神经与组织原型**：`cellspace.py`、`cellstate.py`、`transport.py`、`growth.py`、`cpm.py`、`mechanics.py`、`merks.py`、`model.py`、`model_late.py`、`vasculogenesis.py`、`wound_healing.py`、`immune.py`、`hormone.py`、`ligand.py`。
- **果蝇身体、场景与传感**：`arena.py`、`grass_scene.py`、`meadow_scene.py`、`tactile.py`、`limb_tracking.py`、`scene_build.py`、`scene_layers_sim.py`、`viz.py`。
- **连接组/感觉运动/电极**：`build_banc_cns.py`、`build_flywire_connectome.py`、`electrode.py`、`electrode_payload.py`、`source_separation.py`、`neural_*.py` 及相关 `tools_*` 审计脚本。
- **交付与验证**：`deliver_*.py`、`verify_*.py`、`check_*.py`、`test_*.py`、`run_*.py`。其中 `test_*.py` 是回归/契约测试，`run_*.py` 多为实验入口或验证入口，不能按名称推断为临时文件。

### 4.2 `engine/`

核心分层仿真引擎与守卫：单位（`units.py`）、规则/出处（`rules.py`）、参数和生物档案（`params.py`、`profile.py`）、层管线（`layers.py`、`project_layers.py`）、检查与记录（`checks.py`、`record.py`），以及电极、神经、感觉运动和闭环身体实现。`engine/embodied/` 是身体后端、场景、连接性、神经运动桥和闭环契约的实现目录。

### 4.3 `workbench/`

本地工作台服务与任务/数据记录：`runs.py`、`bench_service.py`、`datasets.py`、`artifacts.py`、`recording_child.py`。

### 4.4 `viewer/`

可视化工作台及其模块、场景和前端依赖；其中 `viewer/vendor/` 属于第三方内容，保留且不递归清理。

### 4.5 `bench/`、`diagnostics/`、`sources/`

- `bench/`：评测协议、合成 fixture、配置与测量模板。
- `diagnostics/`：针对身体、接触、神经、力学、姿态和闭环的诊断脚本。
- `sources/`：文献与来源材料，不是缓存。

## 5. 主要入口

- **推荐阅读与入口说明**：`README.md`；引擎说明：`engine/README.zh-CN.md`。
- **依赖**：解析侧 `requirements.txt`；身体仿真侧 `requirements-body.txt`。
- **实验/验证入口**：根目录 `run_*.py`、`test_*.py`、`tools_*.py`，应依据对应报告和脚本内容选择，不能批量删除。
- **Shell 入口**：`build_scene.sh`、`setup_body_env.sh`、`deliver_*.sh`、`launch_*.sh` 等；它们可能被历史复现实验或交付流程引用。
- **服务入口**：`serve_workbench.py` 与 `workbench/`；运行时状态（包括锁）不可当作垃圾清理。

