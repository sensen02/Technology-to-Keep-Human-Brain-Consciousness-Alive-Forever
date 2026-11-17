# 视觉平地走停首课和下行扩展

本轮属于人工搭建大脑项目，首课采用冻结身体、一个低速档和零推进。DNg97 推进调制、DNg13 走中转向独立校准。已有 DNa02 视觉转向闭环作参考，速度和弧线的身体成绩沿用既有验收。

## 双 DNg100 通路与固定指令

BANC v888 官方左全分辨率骨架和右 L2 骨架在已登记两道截面上连续。两侧连续分支与各自前颈来源标记相差 136.84 和 274.07 nm。官方 v2 边表内 exact root ID 对应 2,148 条定向 VNC 边及 2,067 个靶点。运行通过这些轴突交付活动，在 VNC 工程投射后读出单档推进。2 ms 是工程模拟延迟。信号级已经实测正常交付、切断清空、恢复、收到零和超期归零。 [来源绑定](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/DNg100_BINDING.json) · [官方投射](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/DNg100_GRAPH.json) · [信号预检](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/FORWARD_LINK_PREFLIGHT.json)

同一站姿与同一 9.6 秒推进请求各跑一个完整和切断回合。两条各 11.3 秒、904,000 个原生积分，前 505 ms 的姿态、速度、质心和足载逐字节相等。完整回合实际前进约 .688412 原生长度，1 ms 接触口径最大滑移 .480813，六足完成周期，撤令后站稳。切断组位移约 .000050，保持站立。 [身体配对](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/FORWARD_BODY_PAIR.json)

## 人工脑首课

采用冻结双眼视觉编码与新训练的两特征线性走停头。眼图中出现正前方地标后发送低速推进，地标消失后撤销。首课仅发送单档推进，偏航、多个速度档及身体上行分别验证。两条真实源链教师回合各执行 912,000／923,200 原生步，原生滑移 .481387／.481388，六足换步及起止站稳通过，撤令后 200 ms 就获得合格站稳窗。训练使用以上教师实际发布的 1,145 帧（185 无目标、960 有目标），240 次 Adam 更新。训练集正确率 100%，冻结头权重散列 `5dcc193c76306d47128e04cdbc1deef7b4fa830a88cad1be40cc42126559afca`。 [走停训练记录](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/WALK_HEAD.json)

新场景留出的正常、切断、恢复与空场四种物理回合全通过。四条各 934,400 原生步。正常组在 0.805 s 接到有效推进，走 9.6 s，前速 .0717132 原生长度/s，滑移 .481389，六足完整换步，撤令 200 ms 后达停稳窗。切断时脑有推进意图，身体没有收到推进，平地位移仅 −.0000499。开启、切断、恢复在首次执行动作变化前的身体状态逐字节一致。恢复组通道重开后由新的到达活动恢复走动，原生滑移 .493042，撤令 455 ms 后站稳。空场保持站姿。 [四臂因果验收](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/WALK_ACCEPTANCE.json) · [同次视觉与身体回放](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/WALK_MEDIA/视觉起步与停稳.mp4)

## 调速和转向扩展

DNg97（oDN1）有诱发向前行走的实验证据。DNg100（BDN2）具有更直接的活动与前速关联，且激活强度可调步频。工程台架冻结已验收身体时，通过左右 DNg97 官方源活动测得 .0718675 → .0840153 → .0708912 原生长度/s。每条 904,000 原生步，原生最大滑移 .4770646，完整走停门通过，独立复跑的原始姿态、动作、接触与交付档逐字节相同。切断以相同发送请求保持接近零位移。该标定的读出及档位是工程实现。 [DNg97 台架](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172138Z-dng97/SPEED_CONNECTED/DNG97_ACCEPTANCE.json) · [行走通路实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446846/) · [步频研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/)

DNa02 在原视觉实验实际完成左右原地转向。DNg13 的 v888 两轴突来源与官方 VNC 连接已核验。DNa02 的内侧缩步及 DNg13 的外侧增步用于走中转向校准。四条各 16.1 秒、1,288,000 个原生步的实际工程探针都走稳和停稳，固定摆足落点偏移未达到 1° 的实测偏航门。DNg13 单开相对零活动组变化 +.00954°，DNa02 探针变化 −.05429°，双开变化 −.03179°。各组保留真实轨迹和未过门的结果。原双 DNa02 的原地视觉转向属于单独已通过基线。 [DNg13 来源](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/DNG13_SOURCE.json) · [DNa02 与 DNg13 探针](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T172142Z-dng13/STEERING_AUDIT.json) · [走中转向实验](https://doi.org/10.1016/j.cell.2024.08.033)

四个视觉留出回合的 2,332 张双眼图、9,344 个动作时点由冻结权重独立复算。视觉特征、脑意图、颈部交付日志和 42 维执行器动作均精确重现。 [离线复核](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T170000Z/WALK_OFFLINE_REPLAY.json)
