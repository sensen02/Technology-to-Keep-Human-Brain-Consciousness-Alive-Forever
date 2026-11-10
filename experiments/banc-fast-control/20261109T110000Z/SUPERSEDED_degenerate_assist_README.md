# 作废说明：`SUPERSEDED_degenerate_assist_FAST_*.json`

这两个文件是用 `eval_fast.py` 的**第一版**跑出来的，辅助段有 bug，数字不能当"1 s 辅助 + 3 s 自主"协议的结果：

1. `seg = ts['targets_rad'][offsets0:offsets1][:args.assist]` 只取到第一段（161 个 200 Hz 样本），而循环条件 `t < seg.shape[0]` 在 **1 ms tick** 下只跑 161 tick = **0.161 s**，不是 1 s；
2. 同一行把 **200 Hz 样本当作 1 ms 样本**逐 tick 播放，等于**5 倍速**回放。

修正后（每 5 tick 播放一个 200 Hz 样本 + 拼接多段凑满 1 s）的数字在 `evalF_*.log`，报告以那些为准。
