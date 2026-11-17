# 分段恢复候选2：左侧通过，右侧恢复仍失败

父唯一托管bash-59已收集exit0。真实preflight76checks全PASS，最大production/scan/gradient/Adam误差8.326672684688674e-17，旧24训练输入/target/C/time逐位复用，仅8pulse四段lossweights改变，其他16weights保持exact。固定freshAdam1e-4、500更新完成，0/250/500保存均新进程审核，只有最终500接受一次新V2/V3验证。没有物理积分或motor模型加载，不选择250或追加500。

最终权重：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/TRAIN/checkpoint_000500.pt`，SHA256 `fc3a0c20f854afe162afdf3c87cbddea2b8879a7e29d3cde97f487640e101641`。

权威功能结果：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/VALIDATION/RESULT.json`，all_targets_pass=false。

独立审核：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/VALIDATION/VERIFY.json`，integrity_all_pass=true。

## 分项结果

| 验证条件 | 恢复窗口RMS | ≤.05门 |
|---|---:|---|
| 左脉冲V2 | .0404061945 | PASS |
| 左脉冲V3 | .0457602302 | PASS |
| 右脉冲V2 | .0601211798 | FAIL |
| 右脉冲V3 | .0640791809 | FAIL |

全部其余sign/latency/reversal/noodor/equal门通过。无气味RMS .0190913≤.02接近边界；equal RMS .0160933/.0188175≤.05。12配置含重复的确定性无气味序列，不能称12次独立重复。

## 父只读恢复轨迹诊断

训练和审核任务全部结束后，父单线程NumPy只读 `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261112T000308Z/VALIDATION/OUTPUTS.npz`，exit0，无Core/forward/新训练。

- 无气味turn均值+.0190142、标准差.0017133，最后.5s均值+.0190175。
- 左侧恢复窗口均值−.0087056/−.0117387，标准差.0394572/.0442290。
- 右侧恢复窗口均值+.0418670/+.0437267，标准差.0431475/.0468414，正向峰约+.10146/+.10753。

两侧都有恢复瞬态，右侧正向反冲与整体正偏差叠加；不再是首批旧模型长期近饱和。但这不是偏差根因的隔离证明。原random batch4并不保证左右配对，是否使优化产生方向偏置仍为候选解释。

当前不进入身体试验，未生成physical_B INPUT_BINDINGS/FROZEN/trials。该子目录仅为保留未执行的可复用CODE_ONLY适配器，22项标准库检查不是身体资格；prepare必须allofflinePASS，不能将该fc3a失败权重冒称准入。

下一只读设计候选拟保持输入/目标/四段weights/architecture不变，评估镜像配对的训练batch能否改善偏差。尚未冻结或运行下一训练，不做运行时减偏置、deadzone、reset或刺激门控。当前失败权重/数据/源码/审核均保留。
