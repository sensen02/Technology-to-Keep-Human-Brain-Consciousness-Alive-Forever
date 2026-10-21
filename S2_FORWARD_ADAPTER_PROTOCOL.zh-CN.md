# S2/P2 无积分真实wrapper适配检查协议

手动研究原型，不是成品。本轮授权界：仅两个fresh model/data各一次mj_forward，绝不mj_step/积分/姿态搜索/参数搜索或修改旧runner。任一接口缺失/不一致明确失败，不靠额外forward修饰结果。

1. canonical case：用原canonical_physics.load_model，初forward之前赋历史 `/run/media/sensen/Data2/cell_wound_prototype/outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz` 的projected_seated_qpos，qvel/act/ctrl=0，保留canonical solver/接触/积分选项。先保存qpos/qvel/act/ctrl/time，调用一次mj_forward后逐字节比较未积分状态，warnings=0。
2. artificial case：独立人工XML，平地和两个空间分离的free sphere，初始有微小穿透；NEWTON/pyramidal/noslip0，保留island enabled。该模型只验证多岛字段，不是果蝇行为。只一个预定构造，若实际不足2岛则报告未覆盖，不搜索/调整重试。同样一次forward、状态不变、无警告。
3. 从真实mujoco module读取3.9.0、mjNSOLVER/mjNISLAND，solver结构array实际shape和字段。复制nisland/nefc、岛行区间及双向map/iefc_state/global state，并依真实island enable flag走路径。把wrapper统计记录转换为flat(capacity*slots,7)，列顺序按tools_s2_final_state_contract.STAT_COLUMNS，从零开始仅填有记录槽；未使用槽明确零占位非观测。确认wrapper公开每个active slot的实际下标布局，不猜reshape。如果可见shape与期望不同，按明示布局适配并报告原shape。
4. 保存原始numeric无pickleNPZ及meta JSON；纯validate_capture生成final状态和分岛stats；保存qacc/J/aref/D/R/force/type，允许独立同端点分段对照但REPORT_ONLY、不要求r为零、不将公式误差变通过阈值。只比较pyramidal/limit非负行，非适用type另计；不构造M/AR或stationarity。
5. 桌面独立新目录，源码/新工具hash、原82source及2NPZ哈希执行前后不变；归档人工XML、真实字段shape/运行环境/forward次数。新脚本事前可读schema辅助，但不额外调用forward。内存上界：nefc4096/nisland128/nv256/单case数值64MiB，不截断行；stats超过mjNSOLVER标truncated。
6. 独立read-only核查只读新档案，独立逆映射/分岛slot、状态不变和hash；聚焦回归。图展示原始global/final标签与分段力差、标题区分canonical/artificial，父实际查看。无全量回归/动态轨迹/生物/站立行走验收。本轮仅真实接口适配；不能给历史848行补造未观测岛内state。
