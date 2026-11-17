# v3入口全局作用域恢复：科学数值不变，仍仅续训300到总600

v2实际预检在common.initialize进入_preserve_rng时即失败：盲目克隆所有FunctionType，把contextlib.contextmanager生成的wrapper重绑到科学globals，缺少其定义域_GeneratorContextManager。原v2源／日志／输入／失败保持，不在原scope重跑或改代码；旧300权重、Adam、RNG未受这次入口失败影响。

v3只克隆三项科学函数到新共享g：_validate_payload、load_checkpoint、save_checkpoint。它们确需新ROOT／SAVE_STEPS／initialize／check_config。其他原函数保持同一对象及其原__globals__，包括装饰器wrapper、wrappedgenerator、hash／RNG／optimizer等纯helper。newinitialize/new_check_config显式覆盖g；旧300严格加载仍在旧adapter/path上下文中，finally恢复4alias/path。

staticcheck.py已在子实际用标准库执行通过：_preserve_rng is originalfunction，globals为contextlib、递归LOAD_GLOBAL/NAME全部解析；真实wrapper()只构造manager、不enter因此没有Torch导入。另一个纯stdlibcontextmanager fixture正常进入退出。3science新g全部全局解析，11purehelpers identity保留；不能把此标准库入口修复当数值模型资格。

其他13个v2科学Python文件逐字复制到新目录，不层层包装；B1分量缓存／原运算次序／B1学习核／损失／dtype／mask／412模型／352状态／8参数／1e−12与1e−10门及5400秒1800MiB预算不变。新prepare绑定原v2全部源和失败日志，并生成新scope单一INPUT_BINDINGS；不继承v2数值PASS，因为没有。完整新preflight仍需实跑。

正确环境下先metadata：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v3/prepare.py --prepare-resume300-order-repair
```

再独占预检：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v3/preflight.py --execute-parent-order-preflight
```

仅完整all_passed/timingREADY_RESUME300_TO600后恢复：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v3/run.py --execute-parent-order-resume
```

从严格b562...原300续接原Adam计数／anchor／RNG，only301..600共300新更新，不reseed／fresh训练状态／reanchor。新300初始资格、450、600各fresh审核；原0PASS与旧300FAIL保留，不修改历史。预检全12 B1/B12/micro4六输出对生产、13真实target/编码、FD/完整1920梯度Adam与恢复、资源时长都必须重新验证，无隐藏放宽或自动延期。之后仍需无教师body4和真实DN2decoder/fullintegrated任务；数值修复不等行为通过。
