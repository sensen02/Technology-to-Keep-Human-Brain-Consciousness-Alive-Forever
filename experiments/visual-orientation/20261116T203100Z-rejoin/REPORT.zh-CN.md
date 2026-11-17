# 远障碍重追与转向恢复

人工搭建大脑项目。原场景为同一实体绿盒和固定红标，沿用原站姿、71 个身体网格、18 个足垫、有限 PD 和 12.5 μs 原生物理。脑请求通过有来源的 DNa02 和 DNg97 工程接口。双眼图像每 20 ms 发布，身体每 5 ms 使用颈后指令。视觉修复使用经原眼图验证的红色标记像素，补足神经读出的出现误判。转向重追超过连续 3.2 秒时撤令约 .9 秒，再按眼图重新发出动作。这是一套限定视觉任务的工程策略。

## 停滞诊断

- B 的归档位姿、速度和控制量复建短窗。两组同画面和原生状态至首个 25 ms 颈后动作差异。旧组原地不动。修复组在 1.325 s 收到前进，累计 4.675 s 颈后推进，实测沿原朝向 .347，目标距离 4.743 → 4.398。5.9370875 s 右中足第四跗节擦盒，零接触门失败。该短窗证明身体恢复推进并定位了近障碍余量不足。 [短窗配对](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/PAIR_B_SHORT.json) · [接触证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/SHORT_CONTACT.json)
- A 首轮复建仍使用站姿执行器目标，两组均实际右转。补齐归档的执行器目标和足端高度变化后，初次匹配保持到 505 ms。持续右转组 1.5–4.5 s 实测偏航幅度约 7.47×10⁻⁸°，撤令 .9 s 再发右转组实际转过 21.87°。两组各 360,000 原生步、全程零触盒和零非足地面接触，最小质心高度分别 1.275 和 1.259。 [身体右转恢复配对](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/PIVOT_TARGETS/RESULT.json)

## 从原站姿连续复验

沿用旧斜走时长只换重追时，确实恢复了推进，仍在 50.5751875 s 擦到盒子，首触附近仍为右中足第四跗节。额外增加 9.4 秒斜走，让身体以更大侧向间距回转。加宽组与旧 B 同态匹配至首次颈后差异 30.045 s，共 1,502 幅双眼图像与 2,403,600 个原生力步骤完全相同。 [原宽失败](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/full/RESULT.json) · [原宽擦碰证据](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/FULL_CONTACT.json) · [同态验证](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/PAIR_WIDER.json)

加宽组从原站姿连续完成 90 秒共 7,200,000 原生步，盒力逐样本为零，地面非足力为零，初转足垫最小净空 .321。全身 71 网格和 18 足垫的最靠后边缘在末窗仍越过箱体远缘至少 .177 原生长度。回转后重新跟随红标，先收到 .08318 的颈后推进请求，后降到 .071713。斜走和后续两段连续推进均通过原工程行走门，最大逐腿滑移分别 .437、.470、.485。目标实际距离由 6.000 缩至 1.956。90 秒预算止点仍在减速接近，预登记 1.18–1.72 的终止距离和站稳门未过，故整回合退出 1。 [加宽原生结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/wider/RESULT.json) · [步态与力审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/WIDER_QUALITY.json) · [加宽回放](/home/sensen/Desktop/engine修复验收/远障碍重追修复-20261116T203100Z/全身越障后恢复趋近.mp4)

另从加宽组归档终点恢复位置、速度和执行器目标，重新初始化运行内部状态，做 16.5 秒停车诊断，1,320,000 原生步始终零触盒、零地面非足接触。真实距离由 1.956 缩至 1.297，回合末脑估距仍约 1.636，未满足 1.46 的停止触发，末端转向使 200 ms 偏航变化约 1.00°，超过 .5° 站稳门。诊断退出 1。下一最小动作是在斜侧接近的姿态补真实双眼近距校准，再测减速直至稳定停车。 [停车诊断](/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261116T203100Z-rejoin/finish/RESULT.json)

## 触觉预测

本轮提交训练设计。首课由脑用指令副本和实际到达的 AN 活动预测未来左右前足触觉。使用固定侧缘自发轻触、随机外触和无触对照监督训练，按整回合做盲测和信号消融。现有侧碰工程通路的范围按标定约束，承重另验。 [首课采样、训练和验收](/run/media/sensen/Data2/cell_wound_prototype/RECOVERY_DESIGN.zh-CN.md:387)
