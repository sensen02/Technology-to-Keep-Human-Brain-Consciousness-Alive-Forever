# Fastpath 352 神经运动身体组件：仅代码准备（S2/S5）

根目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_body_component`。

范围固定为 **BodyANNcomponent_NOTnewbrainneckdeployment**。当前只是身体 ANN 组件诊断工具；不得当成人工脑绕过真实颈部的成品部署。父随后必须另建人工脑→真实来源 DN2 延迟队列→神经 2 输出解码器，再做完整感觉回归。新候选最终 SHA 与行为均未知，不预测 PASS。

新校准来源为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_calibration`。已实际读取 common/model/sources/run/adapter/verify_saved/evaluate/audit_eval/preflight 源码；没有修改任何校准文件。新 FORMAT 为 INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1；严格 common.load_checkpoint(path,digest,final=True) 先于物理后端。状态仍为 [128,64,64,96] 共352；42×192头读 [新h96,当前encoded96]；部署无新门控。

旧来源为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component`。其7个核心文件、PREREG、CODECHECK、INPUT_BINDINGS、SUMMARY及四项实际 RESULT/VERIFY 元数据均已核对。旧454581a4模型四项行为全失败，42项保存审核各为true：walk slip1.1392486094；stop最大漂移.147272613体长、RMS.05427844体长/秒、yaw−5.55393485°；左yaw21.516389°、原生非足峰73.72000496；右yaw−6.84323376°。原已通过789f模型保留不覆写。

最小复用：adapter.py 将旧原 common/preflight/runner/audit/run 入口 AST 单层 compile 到新 namespace。仅替换 ROOT、新 CAL、旧合格 TEACHER 的绝对路径、组件 FORMAT、motor FORMAT 和冻结文件清单 FILES。没有复制后再改物理循环或指标。OLD_SOURCE_SNAPSHOT.json 在新冻结前固定旧7core SHA；每次 AST 编译先核对它。所有函数直接在新 globals 中创建，classmethod/staticmethod 同样递归检查。static_check.py 反向恢复这些身份替换后要求完整 AST 相等，并要求每个入口数值常量 multiset 不变。

原四命令 walk[1,0]、stop[0,0]、left_pivot[0,1]、right_pivot[0,-1]，案例名称和顺序不变。每例1秒settle不NNstep，4.8秒教师辅助中960次prime_old仅推进旧256，新96始终0；5.8秒连续交接无真实状态/目标/缓冲/神经reset；之后9.6秒完整352 ANN自主，无教师。71个完整不透明mesh、42有限PD、无adhesion、无实际根写入或额外外力。unsafe仍100ms连续COM<.8或非足>.5W早停，失败原样保存且不重试。每例最多1232000原生步，四例4928000原生步。

验收完全沿旧PREREG：walk原5门加全六腿旧walkingcycles、NF0、完整窗；stop最大漂移≤.1体长、过渡.5秒后身体XY速度RMS≤.01体长/秒、绝对偏航≤5°、support≥.9、COM≥1.2、NF0、完整窗，周期NA；pivot同向偏航≥45°、最大漂移≤.5体长、全六腿load-unload-clearance-reload及同一安全完整窗。原forward/slip每例仍legacy单列，不能拿审计完整性通过替代行为门通过。

prepare.py仅标准库读取实际JSON及文件SHA，不读取NPZ payload或加载PT。必须有新校准实际3mainstages全部exit0、TRAIN更新600与实际PT SHA、STAGE_VERIFICATION、EVALUATION_VERIFY和实际EVALUATION数组文件SHA；还必须有0/300/600每保存新进程VERIFY与新源码明确输出的DEV_VERIFY_000000.json、DEV_VERIFY_000300.json、DEV_VERIFY_000600.json，以及DEV_EVALUATION.json。全部源、数据、证明文件hash在exclusive写新INPUT_BINDINGS前再次核对。未知SHA不硬编码，既有绑定不覆盖。公共preflight和9stage入口继承旧先NN后physical顺序，旧common/model别名加载使用sentinel/finally恢复。

本子任务仅执行下列静态命令，不执行后面的父数值命令：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_body_component/static_check.py
```

父审核新cal实际完成final600并取得ACTUAL值后，按顺序执行（ACTUAL必须替换为实际64位小写SHA；不按此字面占位符执行）：

```bash
/run/media/sensen/Data2/cell_wound_prototype/.venv/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_body_component/prepare.py --bind-completed-final600 --sha256 ACTUAL

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/.venv/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_body_component/preflight.py --execute-parent-preflight

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/.venv/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T195245Z/fastpath_motor_body_component/run.py --execute-parent-fourcases
```

末条由父使用managed后台job串行执行并追踪，固定4phys＋4freshNNsourceAudit＋summary共9stage。preflight为1Core零积分；本子没有执行它或绑定。CODECHECK记录准确源SHA、实际静态检查和零数值调用。冻结后不再修改core，父统一更新主tracker。
