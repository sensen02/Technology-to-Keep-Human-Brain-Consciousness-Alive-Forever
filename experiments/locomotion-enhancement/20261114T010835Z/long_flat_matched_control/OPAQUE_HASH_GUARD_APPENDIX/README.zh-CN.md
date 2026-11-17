# 冻结后追加：父guard与末窗诊断入口

本附录是 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/` FIRST READY后的独立追加；不修改任何原冻结源码/XML/判据/READY。原FROZEN SHA256为41773f5310b50fa2fcb6319a5d77d104a53ab9248cd73ee1af786a780656e55f。

最后只读review发现原父guard两个确定工程问题：全后缀payload禁止误拦r.verify对历史NPZ/PT的opaque SHA；直接编译旧XML没有执行原Core默认滑摩擦MU1.5及mesh/material alpha1处理，而实际flat Core已执行，导致不对等比较。父尚未运行guard，未产生失败报告或数值结果。新追加入口只允许实际sha/sha256函数以rb或read_bytes读取opaque哈希；np.load等payload解码与NN/积分禁止保持。旧直接编译reference在named-array比较前施加同一冻结physics.rc源码MU常量（AST确认1.5）及原mesh/material alpha1；不会从flat模型复制参数，也不构造旧Core或重跑physics。此为对齐已存在的构造器处理，不是新摩擦参数变化。

附录orchestrate先核自身freeze与实际guard附录SHA，再委托原冻结one-trial/fresh336audit逻辑。追加诊断只读同次保存数组：四9.6窗与末窗root/COM净位移和XY路程、yaw净/路程、target最大关节跨度、policy RMS、原.1W逐腿loaded比例、heading投影COM路程；加上原诊断的动作、targetstep、motorhidden变化及strictcycle完成事件。原全窗五门/周期不改；不会据whole PASS声称末窗持续运动，也不新增人为停止阈值。非生物有效性结论。

先标准库复核（本子已运行；不执行科学入口）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/preflight.py
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/OPAQUE_HASH_GUARD_APPENDIX/sourcecheck.py
```

仅父在当前95任务资源释放并决定实际执行后，使用下列**附录guard命令取代根README的guard命令**。仍1flatCore/0steps/0NN/0payload，无旧Core；constructor mj_forward允许，不是积分。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/OPAQUE_HASH_GUARD_APPENDIX/parent_zero_step_guard.py --parent-resource-released
```

guard实际全部通过之后，仅一个父managed background任务使用**附录orchestrate命令取代根README的orchestrate命令**，仍one longcase+fresh336 audit，最大3536000native，无retry、无render：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/OPAQUE_HASH_GUARD_APPENDIX/orchestrate.py --parent-resource-released
```

输出仍在根scope的新case/guard/SUMMARY路径。所有科学执行未知，编译parity与行为必须实际验证；附录静态通过不能当actualguard PASS。不会改当前95、主tracker、旧rough/plane结果或新motor模型身份。
