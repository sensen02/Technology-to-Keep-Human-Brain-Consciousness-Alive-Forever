# 条件反馈数值诊断（只读，S2/S5）

实际完成两条1920 tick；wall 119.678s，RSS 678.13MiB；训练/优化器/Core/物理均0。

300 checkpoint SHA256：b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14。0步沿用已有PASS，不重扫。

## stop

原JIT相对fullproduction最大误差：{"mean": 1.1104296822328497e-08, "residual": 2.841329393987735e-10, "h96": 1.3481882277233126e-11, "pretarget": 1.110067593401709e-10, "posttarget": 1.110067593401709e-10, "action": 1.1103593803088452e-08}
强制相同fullproduction pretarget最大误差：{"mean": 1.7763568394002505e-15, "residual": 1.3322676295501878e-15, "h96": 1.4432899320127035e-15, "pretarget": 0.0, "posttarget": 2.220446049250313e-15, "action": 1.7763568394002505e-15}
Python/JIT最大误差：{"mean": 0.0, "residual": 0.0, "h96": 0.0, "pretarget": 0.0, "posttarget": 0.0, "action": 0.0}
首float32 offset分歧：{"index": [1895, 0, 9], "a": -0.015269247815012932, "b": -0.015269248746335506, "abs_error": 9.313225746154785e-10, "tick_zero_based": 1895, "time_after_handover_s": 9.475, "channel_zero_based": 9, "target_a": 2.48008731986231, "target_b": 2.4800873198623012, "normalized_a": -0.015269248280669058, "normalized_b": -0.015269248280676344, "midpoint_normalized": -0.015269248280674219, "target_at_midpoint": 2.480087319862304, "normalized_minus_midpoint_a": 5.160802341031001e-15, "normalized_minus_midpoint_b": -2.1250362580715887e-15, "adjacent_float32": true}
分歧前误差：{"mean": 1.7763568394002505e-15, "residual": 1.3322676295501878e-15, "h96": 1.4432899320127035e-15, "pretarget": 8.881784197001252e-15, "posttarget": 8.881784197001252e-15, "action": 1.7763568394002505e-15}
old256 exact=True；原运算顺序直接三项缓存误差：{"mean": 0.0, "residual": 0.0, "h96": 0.0, "pretarget": 0.0, "posttarget": 0.0, "action": 0.0}；门通过=True。
B1实际oldmean直接还原逐位一致=True；抵消重构误差=0.0。
批量与sequential B1三项残差=[3.1086244689504383e-15, 0.0, 0.0]；批量替换后的误差={"mean": 3.1086244689504383e-15, "residual": 1.9984014443252818e-15, "h96": 1.887379141862766e-15, "pretarget": 7.549516567451064e-15, "posttarget": 7.549516567451064e-15, "action": 3.1086244689504383e-15}。

## old_walk_control

原JIT相对fullproduction最大误差：{"mean": 2.248139174820807e-12, "residual": 2.806088694740083e-14, "h96": 1.3322676295501878e-15, "pretarget": 2.220446049250313e-14, "posttarget": 2.220446049250313e-14, "action": 2.247382835385281e-12}
强制相同fullproduction pretarget最大误差：{"mean": 1.4988010832439613e-15, "residual": 1.4988010832439613e-15, "h96": 1.5543122344752192e-15, "pretarget": 0.0, "posttarget": 2.220446049250313e-15, "action": 1.4849232954361469e-15}
Python/JIT最大误差：{"mean": 0.0, "residual": 0.0, "h96": 0.0, "pretarget": 0.0, "posttarget": 0.0, "action": 0.0}
首float32 offset分歧：{"index": [564, 0, 11], "a": 2.60403021457023e-06, "b": 2.6040304419439053e-06, "abs_error": 2.2737367544323206e-13, "tick_zero_based": 564, "time_after_handover_s": 2.82, "channel_zero_based": 11, "target_a": -2.061733032759806, "target_b": -2.0617330327598036, "normalized_a": 2.6040303271166906e-06, "normalized_b": 2.6040303293097237e-06, "midpoint_normalized": 2.6040303282570676e-06, "target_at_midpoint": -2.0617330327598045, "normalized_minus_midpoint_a": -1.140376974030936e-15, "normalized_minus_midpoint_b": 1.0526561244309125e-15, "adjacent_float32": true}
分歧前误差：{"mean": 8.326672684688674e-16, "residual": 8.326672684688674e-16, "h96": 6.661338147750939e-16, "pretarget": 2.220446049250313e-15, "posttarget": 2.220446049250313e-15, "action": 8.326672684688674e-16}
old256 exact=True；原运算顺序直接三项缓存误差：{"mean": 0.0, "residual": 0.0, "h96": 0.0, "pretarget": 0.0, "posttarget": 0.0, "action": 0.0}；门通过=True。
B1实际oldmean直接还原逐位一致=True；抵消重构误差=0.0。
批量与sequential B1三项残差=[3.552713678800501e-15, 0.0, 0.0]；批量替换后的误差={"mean": 4.218847493575595e-15, "residual": 1.6653345369377348e-15, "h96": 1.5543122344752192e-15, "pretarget": 1.1324274851176597e-14, "posttarget": 1.1324274851176597e-14, "action": 4.2049697057677804e-15}。

最早逐tick链条已定位：stop 首次 mean 双精度分歧为 tick5/channel1，1.3877787807814457e-17；首次 pretarget 双精度分歧为 tick47/channel17，1.734723475976807e-17。old_walk_control 对应 tick8/channel1，6.938893903907228e-18，以及 tick186/channel3，8.673617379884035e-18。两条首次 mean 分歧与共同 fullproduction pretarget 下的首次 oldmean 抵消重构分歧完全同tick/channel/误差；共同 pretarget 下 oldmean 抵消误差全程最大分别8.881784197001252e-16与4.440892098500626e-16。

实测支持的因果链：原加回actual prior再减predicted prior的运算顺序在预测target上产生双精度尾差，递归target积累后跨float32量化midpoint，导致原严格门失败。actual target上抵消重构仍可逐位exact，因此只检actualX不足以发现此问题。Python/JIT全exact，不能归因于JIT。保留base neural head、weighted RIGHT、weighted LEFT三项且按原顺序减prior再加两支，在这两条上消除了全部差异；未新增controller。

权重前后digest均为4e3b4d920af42d8612ec87b686407bc492668e5e3a1cca0fa8c4dac3461c98ca；source SHA前后一致=True；RNG全程一致=True；指定checkpoint SHA前后一致=True；strict load保留原完整性检查并成功，未发生alias恢复或绕过。

环境为指定Python、torch2.5.1+cpu、numpy2.5.3，OMP/OPENBLAS/MKL/NUMEXPR/VECLIB均1，torch intraop1；interop默认登记为24，进程CPU affinity严格只有CPU0，实际资源限制仍单CPU。所有数值结果来自唯一一次managed bash-64，已collect退出码0；最终NPZ与JSON须保持本次身份，后续修复另建scope。

结论仅覆盖这两条固定物理观察的条件数值重放；直接三项是新文件测试，不修改部署，不代表全12条/全batch预检或物理行为通过。下一最小动作是另立修复scope，以源精确三项缓存替换原抵消重构，并验证完整12条与实际训练batch；原1e-12门不变。本子任务未改GOAL_TRACKER（旧scope不得改），由父任务将结果关联S2/S5台账。

完整记录：/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/conditional_parity_diagnostic/RESULT.json；首分歧窗口与逐tick误差：/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/conditional_parity_diagnostic/FIRST_DIVERGENCE.npz。
