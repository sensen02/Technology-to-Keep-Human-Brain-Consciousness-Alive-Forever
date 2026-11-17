# 最终600独立batch审核：ONLYCODE，等待父独占实际执行

本工具是 v4 最终 body 准入的附加数值门，不修改训练、verify、learning 或 controller。最终 checkpoint 的实际 SHA 尚未知，必须由父在 v4 全部阶段完成后提供；脚本不推断或填充 SHA，只接受 v4 的 checkpoint_000600.pt。300/450 不支持作本工具的资格检查点。

实际运行前必须已有 v4 三主阶段全部 COMPLETED/exit0、STAGE_VERIFICATION、TRAIN RESULT、新300/450/600三保存 VERIFY 全 PASS、最终 EVALUATION 与独立 EVALUATION_VERIFY 全 PASS，且 checkpoint/评估数组与各证明 SHA 一致。脚本在这些 JSON/字节身份检查通过后才导入数值后端。正在运行的 JOBS/TRAINING 不进入本目录的编写时静态 ledger；最终运行时资格文件另行登记前后 SHA，TRAINING.jsonl 始终不读取。

唯一数值入口（父须先把下列变量设为实际最终 SHA；不得照填占位或猜测）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/final_batch_parity_audit/audit.py --checkpoint /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/TRAIN/checkpoint_000600.pt --sha256 "${ACTUAL_FINAL600_SHA256:?父必须先赋实际最终SHA256}" --execute-parent-final-batch-audit
```

父仅在当前续训主任务完全结束、资源空闲后，通过 managed job 执行并 collect。工具自身设置单 CPU affinity、Torch intra/inter-op=1、OMP/BLAS/MKL=1，300秒 SIGALRM 与1800MiB RSS watchdog；父可再用外部受管超时兜底，不得扩大本预算。整个模型加载、prepare 和审核处于同一预算内。

所有模型操作均在显式执行入口后进行：严格调用冻结 v4 common.load_checkpoint(..., final=True)，显式绑定并 finally 恢复 v4 adapter/sources/common/model/learning/scan/data/utils 八个 public alias 与 sys.path，不克隆任何上下文管理器。v4 的独立 fullproduction_feedback 通过真实 m.step_details 路径生成每条1920 tick参照。no_grad 下依次检查每条 B1、全 B12、三组 micro4 的 mean/residual/h96/pretarget/posttarget/action；每项最大绝对误差均须≤1e-12，全部有限，old256须与来源cache逐位exact。保留原整批 tanh 路径用于检验，不能在审核中替换或修正被测实现。

不创建优化器、不 backward、不训练、不构建 Core、不进行物理积分。核对 strictload 与全程 RNG、前后权重digest、冻结旧权重、checkpoint SHA、v4科学源ledger、本工具源ledger、最终资格文件ledger及模块恢复；任何失败均不 qualified。

输出独占目录：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/final_batch_parity_audit/RUN_<实际完整SHA256>/`

目录已存在则拒绝覆盖；首次失败也保留。产物为 VERIFY.json、ROWS.json、SOURCE_LEDGER.json；VERIFY 记录异常与退出身份，ROWS 记录12条×3种batch×六输出最大误差。返回0仅表示附加数值门PASS；不代表body行为、真实颈部或全任务通过。

纯标准库静态入口（不导入Torch/NumPy、不读取实际checkpoint payload、不调用v4模型）：

```bash
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/final_batch_parity_audit/staticcheck.py
```

静态检查覆盖 AST/递归 LOAD_GLOBAL/contextmanager定义域、严格 final=True/no_grad/无优化器梯度调用，以及纯内存 JSON/hash 正例和六个拒绝例；这些 fixture 不接触运行中的 v4 元数据。静态PASS只能标 ONLYCODE_READY，不能记数值PASS。
