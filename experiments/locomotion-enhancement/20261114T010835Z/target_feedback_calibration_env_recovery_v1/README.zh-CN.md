# 缺Torch执行环境恢复：科学代码／预算不变

原 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration/` 的首次pipeline使用venv_body，训练入口在import torch即失败。原失败JOBS、日志、FAILURE（若有）、源码、INPUT_BINDINGS及实际合格PREFLIGHT全保留，本目录不写原目录、不覆盖重试。原退出码1不改为0。

本恢复通过单层AST读取原13 Python文件，在新文件路径／globals下执行完全相同语句；只是ROOT输出迁移和执行环境provenance。linecache为重定位模块注册不可变虚拟原文，使inspect／TorchScript可以看到原定义，而不是两行wrapper。原模型、scan、loss、目标数据、mask、优化器、600次更新、5400秒／1800MiB门、保存0/300/600和独立审核均不变。

prepare仅标准库验证原input SHA、实际PREFLIGHT通过及READY_FIXED600_5400S、原失败log确为missingTorch。它复制原科学PREREG的JSON值，继承原输入和数据hash，另绑定原全部核心、失败JOBS／log／FAILURE及新wrapper／adapter／prepare／RECOVERY_PREREG。新PREFLIGHT只明确重绑来源等价wrapper并引用原合格数值预检SHA，不冒称又跑了一次预检。无需重跑已合格预检。

正确环境为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`。子仅标准库核了该文件与torch包__init__.py真实存在，没有导入torch／运行模型。prepare要求实际sys.executable路径并用find_spec('torch')验证父运行环境能找到Torch；正式数值调用仍由父独占执行。

先运行metadata准备：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration_env_recovery_v1/prepare.py --prepare-environment-recovery
```

成功后直接执行新namespace pipeline，不重跑数值preflight：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration_env_recovery_v1/run.py --execute-parent-calibration
```

所有结果写本恢复目录：新JOBS／TRAIN／EVALUATION／STAGE_VERIFICATION。没有原scope retry、没有新增模型／数据／预算。fit和targetloss依旧不是身体通过；未来body4与realDN2decoder2/full432要求不变。
