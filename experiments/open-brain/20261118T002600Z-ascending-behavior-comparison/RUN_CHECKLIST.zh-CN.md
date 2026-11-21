# 执行记录（P1，S1/S3/S4/S5）

- 目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T002600Z-ascending-behavior-comparison；用户日期2026-11-17，UTC目录身份。目标为测试行为，不训练。
- 预登记：PROTOCOL.zh-CN.md；四脑侧模式、运动背景选择、两相位、原始身体/相对/感觉影响门提前固定。
- 预检context.py exit0，PREFLIGHT/RESULT.json；Neural_runtime.py是同字节副本，避免与原身体源码context.py模块名冲突，未改细胞方程。
- 首正式：bash /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T002600Z-ascending-behavior-comparison/drive.sh live.py 10901 /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261118T002600Z-ascending-behavior-comparison/FORMAL/seed10901，exit0，live10901.log。
- 后续正式/重放实际命令/退出码：LIVE_COMMANDS.json，全0。各INPUTS包含完整源码/库/资产/参数/权重SHA及相同编译身体哈希。
- 原脑/颈/身体/8参数权重全部冻结，优化更新0。候选感觉只读自己真实颈后到达；正式PN上游事件逐拍共享；不规定两侧身体命令。
- 独立完整重放4身体×11数组＋9神经数组精确。审计DN逐通道传导＋1tick调度、AN精确传导、纯颈后编码和源到运动读出无交集；TRANSMISSION_AUDIT.json全过。
- 程序运行/assess.py/exploratory.py/audit_transmission.py退出0。身体8/8推进/安全/撤刺激后站稳；原脑对照及受控参考均0/2相对门，主感觉影响门未过，不追改容差。
- 探索性翻译版上行消融另存EXPLORATORY_SENSORY_EFFECT.json，正式指标查看后补充，不替代预登记验收。
- 图/视频：MEDIA.json；两PNG实际打开；视频ffmpeg完整解码exit0、ffprobe400帧/25fps/16s/1600×480；仅渲染地板去反射。MEDIA_REVIEW.json留证。
- 数量：12身体/240000原生步；原脑63200批步/252800 lane-time步；固定颈60000批步/240000 lane-time步；参考和原脑受体局部LIF各60000步。
- 失败/未执行：行为一致性未过、参考感觉影响未重复过门；候选14AN回馈尚未接回；无避障/主动触觉停步/长期/视觉嗅觉/飞行验收；PPO/GAE不适用。
- 下一最小动作：以失败基线开展同状态教师的活动脑/下行/行为监督校准；原身体阶段接受不撤销，P2不混。
