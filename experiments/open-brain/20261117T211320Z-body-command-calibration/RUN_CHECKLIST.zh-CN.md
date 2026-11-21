# 本轮运行检查表（P1，S2/S3/S5）

跑前协议PROTOCOL.zh-CN.md、同类DNa01补齐A01_EXTENSION.zh-CN.md；用户裁定OWNER_ACCEPTANCE.json。当前正式teacher-relative 7/8、学生动作绝对门8/8、原接受走停绝对2/2、相对1/2（平地胸高低.149044mm）；速度组yaw差10.107°>10°的原门失败保留；原绝对结果分栏，旧历史不改。

- 身份：同规范48路/70body/66有限铰链，原PD/附着/初态/力限；ANN＋来源化工程读出/局部CPG；没有新脑/颈部可塑性或物理调参。
- 来源：各INPUTS.json实际导入依赖/教师资产/初态/模型/固定颈部参数和读出/输入权重SHA；同命令、同编译模型，终局核对。
- 命令环境：本目录drive.sh，{"executable": "/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python", "python": "3.12.14", "packages": {"torch": "2.5.1+cpu", "mujoco": "3.9.0", "flygym": "2.1.0", "numpy": "2.5.3"}, "threads": 1, "render": "EGL"}。
- 实际命令：commands.py dev首版exit1（2完成源保留，额外20,000tick未存）；commands_v2.py dev/held、commands_a01.py dev/held退出0；run_body.py烟测exit1（20,000积分TRACE保存，NumPy bool JSON）；batch_pair_v2.py dev DEV、batch_a01.py dev DEV_A01、fit_commands_v3.py、evaluate_candidate.py CALIBRATED --checkpoint FIT/epoch040.pt、freeze_candidate.py CALIBRATED FIT/epoch040.pt均0；batch_pair_v2.py held HELD、batch_a01.py held HELD_A01均同checkpoint且0；parking_regression.py、replay.py、audit_command_sources_v2.py、verify.py、media.py均0。
- PPO/GAE不适用：真实监督复合损失中心差分、Adam/RNG续接、每个4,120update可加载权重/参数SHA、固定40轮选择、独立重放核验完成；模板/足垫/相位/正则化缓冲不变。
- 数据：16当前教师32,000帧＋20,400保留地形，共52,400；校准16/16，正式6211新相位8组（首次使用）；早期模型右转/速度偏离保留。没有不报告失败的换种子，未用正式标签微调。
- 实际积分：身体74回合1,468,000步，颈部500,000tick含失败未存20k；新Eon全脑0。额外只读FK/核验/渲染不计积分。
- 核验：五代表新进程18数组逐值一致，学生teacher_calls0，逐决策动作重算、身体与命令精确绑定，所有警告0。未执行：旧69完整回归/全D1–D8/逐足完整附着滑移账本/飞行/新脑上行闭环。
- 媒体：同次原生状态四动作并排960×1280、400帧、25fps、16s/.125x；ffmpeg全帧解码与ffprobe0，实际打开COMMAND_STATES/COMMAND_METRICS；只去反射不改动作。
- 收尾：手动研究原型；新限定教师相对指令入口，旧通过BODY_CURRENT及源资产不覆盖。桌面报告/视频及源证据关联，tracker/路线图更新，无后台常驻任务。下一动作由用户次序在此身体基线扩组合或地形课程。
