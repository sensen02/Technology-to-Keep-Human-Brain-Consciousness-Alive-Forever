# 地面下行三通道运行检查表

- P1/S1–S5，独立根 /run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T071007Z-ground-dn-response。父双向结构为 051249Z，冻结身体来自 062447Z。旧运行输入与失败完整保留。初始协议、载荷测量补充及 REFINE 复验契约见各文档，未更改正式验收门。使用归档 Python 3.12.14、Torch 2.5.1 CPU、MuJoCo 3.9、FlyGym 2.1、单线程 EGL。0 PPO/0 权重更新。正式以前冻结 113 项输入 SHA，运行后复核。修订 DNg13 有独立 SHA、种子、时序和读出身份。
- bash batch.sh DEV_LEGACY dev、DEV_LOAD dev、DEV_NATIVE dev、DEV_FINAL dev 均退出 0。前三个开发组分别保留接触标记抖动、聚合传感器载荷与实际体重不闭合、停步前接触计数偏差的判定。更正为统一实际跗节完整运动周期和求解器负载。最终开发 audit 四对通过。先前对错误目录的 rg 命令退出 2，随后定位 simulation.py 正确读取。
- audit_ground.py freeze、bash batch.sh FORMAL held、REPLAY held、audit_ground.py audit 退出 0。正式原读出 7/8 对，通过包括两条制动。DNg13 左 11.81° 失败留存。REFINE/refine.py calibrate、REFINE/batch.sh FORMAL、REFINE/batch.sh REPLAY、REFINE/refine.py verify 退出 0。新 DNg13 四对通过。两种正式与复跑共 960,000 原生步，所有 24 条回合完全重放。所有有效动作皆来自冻结身体输出。工程位移和足端运动幅度、真实承重与逐根到达独立审核；未验完整 D1–D8 或低滑移。
- 第一次渲染 Tk 后端环境退出 1，保留 MEDIA_TK_ERROR。MPLBACKEND=Agg 运行 render_ground.py、FFmpeg 全解码及 ffprobe 退出 0。100 帧、30 fps、3.333 秒，模拟 2 秒；6 栏完整／切断视频。实际打开响应曲线和起步、转向、停步关键帧。结果、两张图和视频桌面 cmp 退出 0。
- 飞行按用户要求暂缓。下一最小动作是按该实测身体接口把目前仍为示范的发放换成冻结 Eon 实际发放，保留同源根和单切配对。DNg60 同时撤令余动 0.0254 mm 单列。
