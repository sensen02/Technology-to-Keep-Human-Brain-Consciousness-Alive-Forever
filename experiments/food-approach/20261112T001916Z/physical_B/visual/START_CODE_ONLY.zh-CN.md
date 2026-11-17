# 新行为权重B6视频与同步图：CODE_ONLY父执行入口

当前仅准备完成。新行为权重严格固定为66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034，motor仍789f。实际B6/独立审核/有效受体起算的v2汇总尚由父唯一heavy负责，本子不运行任何重模块或读取trial输出。

## 最小适配

旧AB visual源码不改、不复制整套框架。SOURCE_CONFIG.json记录旧contracts/render/plot三文件实际SHA、newROOT/newbehaviorSHA/loader、新BASE和HEAD绝对路径。新contracts将冻结旧纯函数完整绑定到新全局字典，仅一处AST适配让协议检查读取原AB PREREG，另外独立检查本scope INPUT_BINDINGS/FROZEN/PREFLIGHT/newcommon来源；不是把新候选PREREG伪装旧schema。

render同旧直接函数，只一处AST把head_audit_path赋值换成显式真正EXProot的HEAD_AUDIT_PATH，防止ROOT.parent层级错误。BASE明确原 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z`，ROOT明确 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B`。plot完整复用冻结直接函数。

运行前要求CODEONLY_CHECK_FINAL.json的全部新source hashes仍相同；新physical FROZEN和INPUT_BINDINGS/newloader实际SHA、preflight来源、motor/behaviorcheckpoint actual bytes、每个trial完整288独立VERIFY/run/condition/resultSHA与source/4arrayhash、全6新Bv2SUMMARY行与24artifact sourcehash均严格核验。保留Original5/Functional分列，不要求False改成True才展示。新版matchednoodor必须同newbehavior，不读旧bfd6身体作为比较基线。

不重新import候选common或执行model forward，避免sensor guard与物理导入次序冲突；重放正确性由独立actualaudit完成并在visual严密绑定。原full288/filter/headgeometry/metadata契约保持，旧bfd6会被拒绝。

## 检查记录

实际仅标准库check_codeonly.py，18项全部PASS/exit0：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/visual/CODEONLY_CHECK_FINAL.json`

验证了newROOT/HERE、原BASE、正确HEAD绝对路径、strictnewSHA拒旧bfd6、288审核、FALSE可诚实展示、两个新入口的契约namespace一致；Torch/NumPy/MuJoCo/model/runtime/physics均未导入，model/array/forward/Core/physics/render/plot/trialread均0。当前不调用verify_scope读取进行中trial，不占父heavy。

## 父命令（全部6新B实际、独立audit及B_FUNCTIONAL_SUMMARY齐全后才运行）

在父明确释放唯一heavy后逐条串行，不能同B6物理或其它模型任务并行：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/visual/render.py --condition B_left_pulse --execute-after-parent-release
```

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/visual/render.py --condition B_right_pulse --execute-after-parent-release
```

其它新B相同仅换condition；只允许注册的六个B ID，不能用新入口冒称A重测。

六条件同步图：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/visual/plot_sync.py --execute-after-parent-release
```

视频输出独占新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/visual/VIDEO_B_left_pulse/` 或对应B_ID，含ACTUAL_STIMULUS.mp4/DECODED_MONTAGE.png/VIDEO.json/FFPROBE.json/FRAME_MAPPING.json。图输出 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T001916Z/physical_B/visual/SYNC_SIX_CONDITIONS/SIX_CONDITIONS_SYNC.png` 和同目录PLOT.json。已有输出拒绝覆盖。

保持完整1+4.8+9.6秒/770帧50fps实时、71opaque全身/固定相机/真实保存qpos-qvel-forward仅视觉0integration、headlocalxyz实际重建、原compiled物理数组beforeafter不变、visualframebuffer恢复；英文phase/source/clampNOTNAVIGATION/ANN0=5.8、causalz与heldC、raw/appliedturn、bodyyaw相对handover及新matchednoodor差、ON/OFF起末包络、Original5和Functional分列；B6无bodyreceptor-cut或空间趋近声明。完整decode/ffprobe/hash和父实际打开montage/plot通过后，再发布桌面替换“当前”演示；首批失败的权威证据与旧source不删除。

本准备未修改旧AB visual、已冻结physical_B、候选3核心、任何验证门或主tracker；无后台job、无heavy执行。
