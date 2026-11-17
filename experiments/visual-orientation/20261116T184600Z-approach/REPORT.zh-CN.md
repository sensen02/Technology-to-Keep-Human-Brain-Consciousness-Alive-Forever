# 视觉趋近、调速、转向和单障碍探针

人工搭建大脑项目。使用同一冻结 71 网格身体、42 路有限执行器、同一站姿、12.5 μs 原生物理和 5 ms 身体决策。从已发布的双眼帧估计红标方位与距离，脑按视觉输出转向、走近及减速，实际通过双 DNa02 和双 DNg97 源纤维。末端按距离撤令，身体负责站稳。1,600 组实际渲染光学样本中 1,380 供训练，220 留出。方位平均误差 .484°，距离平均误差 .0494 原生长度，出现判别 220/220。当前距离依赖固定标记尺寸。 [光学评估](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/VISION_RESULT.json) · [真实来源组合绑定](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/APPROACH_BINDING.json)

原始左侧目标 20°，身体实际转 +9.054°，初末距离 2.680 → 2.133，实测快慢前速 .08599 → .07212，末态站稳。原始右侧目标 −22°，身体实际转 −11.391°，距离 2.680 → 2.060，实测快慢前速 .08365 → .07153，末态站稳。两组均有六足周期。走动段原生滑移左 .51013，右 .48347，原 .50 行走质量门只在右侧通过。左组真实行为和原门失败分别保留，独立复跑轨迹、接触、双眼及颈部记录逐字节相同。空场和切断均不推进。切断后脑仍发转向意图，初次动作前与同场景左侧轨迹逐字节相同。 [原始场景与同态审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/APPROACH_ACCEPTANCE.json)

人工脑当前用训练过的视觉读出和显式趋近策略，验证的是先转向、交接快速走、减速和停稳。这轮保留独立、有来源的下行范围。走中连续弧线和由奖励学习整个任务待后续单独检验。

左侧增加两次转走交接的站稳时间，原生滑移分别为 .51078、.51087。把减速时的视觉距离阈值由 2.41 调到 2.32 后复跑，左侧仍转向、走近 .546 原生长度、从 .08259 降到 .07277 并停稳，滑移改善到 .50418，仍超过 .50。双侧原始成对结果继续作为同参数基线。 [前两次左侧复测](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/PATH_QUALITY.json) · [延后减速复测](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/LATE_QUALITY.json)

单障碍采用真实碰撞绿盒，初态无接触，开启和关闭障碍响应的身体初态与初始双眼相同。开启时识别绿盒后保持站立，零触盒。关闭时前进约 .166 原生长度，于 3.213325 s 第一次触盒，峰值约 .03225 倍体重。此处验证的是检测障碍后抑制起步。 [开启](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/FINAL_SAFE/RESULT.json) · [关闭](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/FINAL_BLIND/RESULT.json)

进一步试验让脑从视觉触发侧转，交接侧向绕行并重新追踪红标。28.041 s 实测最大侧移 .512 原生长度，最终较起点靠近目标 .608，停稳。2.58125 s 在初次侧转时已触盒，原生峰值约 .24839 倍体重。首次触盒附近的归档姿态显示右前足跗节足垫扫进盒边，无碰撞绕行门判失败。下一次先检验转向足端的扫掠间隙，找出允许启动侧转的距离，再验收零触盒绕过、重新接近和停稳。 [绕行结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/FINAL_DETOUR/RESULT.json) · [接触诊断](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/OBSTACLE_CONTACT_DIAGNOSIS.json)

颈部采用有解剖和连接组来源的上下行接口中的指定下行纤维，队列时延为工程参数。全颈束、生理传输及本课的身体感觉融合尚待验证。 [运行检查表](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T184600Z-approach/RUN_CHECKLIST.zh-CN.md)
