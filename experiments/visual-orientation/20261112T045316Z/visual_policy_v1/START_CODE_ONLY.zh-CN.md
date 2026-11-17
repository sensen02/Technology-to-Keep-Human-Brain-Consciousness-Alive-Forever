# 视觉行为候选1：父代理唯一资源执行说明

代码准备已完成，实际仅标准库check_codeonly.py执行exit0，19项检查全PASS。未导入Torch/NumPy/MuJoCo/模型，未加载数组、forward、训练或Core。实际图像库已由相机链生成/独立审核/父目视，训练链运行前仍会严格核其manifest/340check proof/RGB/physicalidentity/hash。

完整父托管命令（不可与渲染、其他训练或物理重任务并行）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v1/run_candidate.py > /run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z/visual_policy_v1/PIPELINE.log 2>&1
```

run_candidate为纯标准库逐新进程串行执行：preflight（缺DATASET时生成真实图像frameID序列；实际fullbatch4梯度/Adam/RSS/时序验证；全部通过后FROZEN）→train固定500→独立verify_saved0/250/500→只final500 evaluate→audit_validation。

每个stage命令/PID/日志/开始结束/exit在该根JOBS.json，动态CURRENT_STATE明确阶段。子进程失败时pipeline停止，父需先读对应EXECUTION.log，不盲重启，不覆盖已有DATASET/FROZEN/TRAIN/VALIDATION或改已冻结源。预检中临时unitAdam与正式训练更新分开记录，0物理不是0模型forward。

架构：共享每眼Conv3→8→8，空间512→8feature；左右拼16→Linear32→slowGRU32→turn1tanh。实际uint8RGB /127.5−1，CNN每capture一次，GRU5ms/慢alpha原式；20ms采样+20ms发布延迟，held输入缓存非神经记忆。预卷真实off图像，480个20msANN区间/1920个5ms决策；481capture中末帧在9.6s前尚未发布，不提前读。首版STATIC_HEAD实际图像钳制，不声称移动身体相机视觉闭环已完成；motor未加载/训练、嗅觉不接。

数据24训练/12验证/8控制按固定pose与config映射；库112个actualframepairs并不算112独立episode。14个视觉参数张量（包括CNN）全训练，完整BPTT/encoder梯度不detach，batch4取2左右配对。内存预检保守峰值RSS≤1800MiB，正式训练也检查；若超门，先把资源问题如实报告，不缩短时窗或悄悄冻结CNN。生产逐capture卷积与扫描批量卷积浮点误差门1e-10；保存审计使用相同生产batch4/形状要求bitexact，不为凑exact换容差。

预登记功能：emptyRMS≤.02、center≤.05、左右符号/反转≥.9、第一张发布可见帧后2s内正确方向，发布撤除帧后.5—1s RMS≤.05，遮挡realwall/occluder-only≤.05。固定旧帧与眼交换仅诊断，无“必须零输出/必须反转”的伪门；完整左右场景镜像与仅交换眼顺序不同。报告读全离线分项，不用总loss代替门。

保存0/250/500仅final500验证一次；失败保留不追加500、不择250。allofflinePASS加独立审核仅允许父另立最多6身体条件预登记，pipeline不启动身体、不称视觉导航/融合/停车/摄入。所有核心运行源码自CODEONLY_CHECK后停止修改；本说明不纳科学运行source identity。
