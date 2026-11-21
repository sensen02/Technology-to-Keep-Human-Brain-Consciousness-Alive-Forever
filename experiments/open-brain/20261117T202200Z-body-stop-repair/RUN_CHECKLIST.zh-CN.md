# 本轮执行检查表（P1 / S2、S3、S5）

跑前协议：/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T202200Z-body-stop-repair/PROTOCOL.zh-CN.md。选模的留出前显式修订：SELECTION_REVISION.zh-CN.md。后续未采用的训练-only读出试探：NEURAL_ONLY_PREREG.json，不复用旧留出冒充新数据。

- 问题/身份/目标/预算/种子/原门/控制来源：见PROTOCOL；冻结70body、66有限铰链、42＋6、原有限PD、工程附着；ANN＋局部CPG与来源化读出。未改物理、无根写入/额外力。
- 输入身份：实际导入本地模块和教师PREPARE资产、初态、权重在各INPUTS.json；源码未在活动运行中修改。新文件新入口独立保存，全部终局SHA检查。
- 实际命令：本目录`bash drive.sh diagnose_saved.py`、inspect_phase.py、shadow_batch.py、policy_repair.py、probe_repair.py、`batch_repair.py dev DEV_ROUTING --checkpoint ROUTING_FIXED.pt`、fit_repair.py、probe_fit.py、`batch_repair.py dev DEV_FIT --checkpoint FIT_REPAIR/epoch030.pt`、select.py、`batch_repair.py held HELD --checkpoint ROUTING_FIXED.pt`、`batch_repair.py replay REPLAY --checkpoint ROUTING_FIXED.pt`、probe_neural.py、verify_repair.py、media_repair.py；均退出0。原错误旧drive绝对路径exit2，0积分。
- 实际环境：{"executable": "/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python", "python": "3.12.14", "packages": {"numpy": "2.5.3", "torch": "2.5.1+cpu", "mujoco": "3.9.0", "flygym": "2.1.0"}, "threads": 1, "render": "EGL"}。
- 正确性：非PPO/GAE不适用；真实监督复合损失中心差分、Adam/RNG续接、全2,920更新可加载权重SHA、每更新参数digest逐项核验。选用候选实际0SGD，补训2,920updates为未采用实验；不混淆二者。
- 物理：50回合850,000原生步；新脑/颈tick0。开发10/12、新留出4/9，不全合格。六新进程复跑各17数组精确；27开发/留出/重放45,900决策输出精确；compiled BODY字节相同。
- 失败/未执行：高块181新回归、263旧失败、新2711三停车失败/3253两地形失败保留；补训开发8/12/旧读出探针1/4拒绝。原69回归、全D1–D8、逐足完整滑移/附着账本及新脑感觉闭环未执行。
- 媒体：原生qpos同次渲染，只标红左中腿/去反射；340帧/.125x、1440×320、25fps/13.6s；ffmpeg/ffprobe0，实际打开STOP_REPAIR_STATES与STOP_REPAIR_METRICS。
- 当前结论：原指定停车高抬腿在对照中修复，但整体验收NOT_QUALIFIED、deployable=false；原生产及旧地形学生指针不覆盖，P2不动；无后台任务/真实动物或硬件副作用。下一动作见报告及BODY_QUALIFICATION。
