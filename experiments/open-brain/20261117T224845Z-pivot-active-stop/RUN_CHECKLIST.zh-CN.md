# P1 / S2、S3、S5 本轮检查表

- 跑前PROTOCOL、补充PIVOT_RATE_CALIBRATION/SMALL_PIVOT_PREREG、正式前PIVOT_SELECTION完整保留。用户教师相对标准，不追改绝对旧结果。
- 固定权重5e513...ff57e、70body/66限位/48执行、固定真实来源颈部/工程动力学/静态读出/局部CPG；变更仅脑侧DNa02脉冲周期/持续时长。训练0，PPO/GAE/Adam变更不适用。
- 实际入口本目录drive.sh：commands.py dev；batch_v2.py dev DEV；commands_rate.py120/3、180/4与pivot_rate.py3/4；commands_small.py120/5、pivot_rate.py5；freeze_weight.py；stopping_commands.py、stopping_formal.py；commands_small.py120/2、pivot_formal.py；replay.py、audit_commands.py、verify.py、media.py均exit0。ffmpeg全解码和ffprobe0。未运行原run_body.py/批次旧副本；打印字段兼容在执行前另存v2。无未收集后台任务。
- 环境{"python": "3.12.14", "executable": "/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python", "packages": {"torch": "2.5.1+cpu", "mujoco": "3.9.0", "flygym": "2.1.0", "numpy": "2.5.3"}, "threads": 1, "render": "EGL"}，各INPUTS实际导入及身体资产/初态/权重/颈部参数/读出/命令SHA，结束复核。47回合940000原生步、420000颈部tick、新Eon0/新训练0。
- 原始基线2相位10组相对通过，三制动开发绝对6/6；长转/两低率长转失败保留；短串开发教师左高度失败保留、学生完整通过；正式相对5/5、教师绝对5/5、学生绝对3/5（两原地瞬时高度FAIL），不混淆。正式转角终点修订在运行前声明，旧active值保留。
- 五新进程正式学生18数组精确；26学生逐决策输出精确、身体/命令同一、无警告；保存原生qpos/qvel/接触载荷/实际动作/足端轨迹。完整D1–D8/69整套/全附着滑移账本/飞行/脑感觉闭环未执行。
- 媒体同次原生姿态800×1200、400帧/25fps/16s/.125x，仅去反射；实际打开3图含长转共同限制，源视频全解码0。报告桌面交付；手动研究原型，不作成品/生理真实性保证。
- 下一动作可按用户次序使用有限角度转向/主动制动基线，或研究持续大角度转向的全身承重与摆腿协同；飞行暂缓，不自动扩任务。


追加释放时序：RELEASE_SEQUENCE_PREREG、commands_sequence.py/sequence_run.py/sequence_replay.py/sequence_verify_v2.py/sequence_media.py均exit0，新相位9337三组追加全门通过/18数组精确；新增180k身体步60k颈tick，合计1120k/480k、训练0。旧基线未覆写；新增视频1600×720/400帧/.125x已全解码、实际开图，详见SEQUENCE_ACCEPTANCE及RUN_TOTALS。
